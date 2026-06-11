"""
K4 inference engine: wraps the original K4 modules with model-loading
and confidence-mapping logic.

Design goals:
  - Lazy loading: embedder / embeddings are loaded only once at startup.
  - Thread-safety: the embedder is protected by an asyncio Lock so that
    concurrent requests serialise embedding calls (important on single GPU).
  - Confidence: raw anomaly scores are mapped to [0, 1] via a smoothed
    sigmoid based on the known normal-score distribution.
  - Scoring: uses KNN mean distance as the primary anomaly score (much more
    robust than PRDC + ML detector for dense embeddings).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import Settings, resolve_device, resolve_embedder_path
from .model_loader import load_all
from .preprocess import normalize_log


class K4Engine:
    """
    Inference engine backed by a saved K4 model.

    Lifecycle:
      K4Engine.load(model_version)  -> load all artifacts into memory
      K4Engine.detect_one(log)      -> dict with is_fault / confidence / ...
      K4Engine.detect_batch(logs)  -> list[dict]
    """

    def __init__(self) -> None:
        self._embedder: Optional[SentenceTransformer] = None
        self._device: str = "cpu"
        self._normal_embeddings: Optional[np.ndarray] = None
        self._scaler: Any = None
        self._detector: Any = None
        self._threshold: float = 0.0
        self._normal_mean: float = 0.0
        self._normal_std: float = 1.0
        self._config: dict[str, Any] = {}
        self._loaded_version: Optional[str] = None

        # Protects embedder calls on a shared GPU
        self._embed_lock: asyncio.Lock = asyncio.Lock()

        # Torch tensors for KNN scoring (avoid re-creating each request)
        self._normal_emb_tensor: Optional[torch.Tensor] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self, model_version: str) -> None:
        """Load all model artifacts for the given version."""
        print(f"[K4Engine] Loading model version '{model_version}' …")

        artifacts = load_all(model_version)
        self._config = artifacts["config"]
        self._device = resolve_device()

        # SentenceTransformer – load on target device
        embedder_name = self._config.get("embedder_name", Settings().DEFAULT_EMBEDDER)
        self._embedder = SentenceTransformer(
            resolve_embedder_path(embedder_name), device=self._device
        )

        # Normal embeddings (kept in CPU RAM)
        self._normal_embeddings = artifacts["normal_embeddings"]

        # Pre-convert to torch tensor for fast KNN scoring
        self._normal_emb_tensor = torch.from_numpy(
            self._normal_embeddings
        ).float().to(self._device)

        # Detector + scaler (for backward compatibility / PRDC mode)
        self._scaler = artifacts["scaler"]
        self._detector = artifacts["detector"]

        # Threshold & score distribution
        thresh_info = artifacts["threshold"]
        self._threshold = thresh_info["threshold"]
        self._normal_mean = thresh_info["normal_scores_mean"]
        self._normal_std = max(thresh_info["normal_scores_std"], 1e-6)

        self._loaded_version = model_version
        print(
            f"[K4Engine] Loaded.  embedder={embedder_name}  "
            f"detector={self._config.get('detector_type')}  "
            f"device={self._device}  "
            f"n_ref={self._normal_embeddings.shape[0]}  "
            f"threshold={self._threshold:.4f}"
        )

    def warm_up(self, n_samples: int = 10) -> None:
        """Run dummy predictions to JIT-compile / warm caches."""
        if self._embedder is None:
            return
        device = self._device
        n = min(n_samples, len(self._normal_embeddings))
        dummy = ["warm up log entry"] * n
        self._embedder.encode(dummy, batch_size=n, show_progress_bar=False)
        print("[K4Engine] Warm-up complete.")

    @property
    def loaded_version(self) -> Optional[str]:
        return self._loaded_version

    @property
    def device(self) -> str:
        return self._device

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    # ── Core inference ──────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> np.ndarray:
        """Async wrapper around SentenceTransformer.encode."""
        async with self._embed_lock:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._embedder.encode(
                    texts,
                    batch_size=Settings().DEFAULT_EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
            )
            return result

    def _knn_score(self, query_emb: np.ndarray, k: int | None = None) -> np.ndarray:
        """
        Compute KNN mean distance anomaly score.

        For each query embedding, find the k nearest normal embeddings
        and return the mean distance.  Higher = more anomalous (far from
        the normal manifold).

        Args:
            query_emb: (n_query, d) numpy array
            k: number of nearest neighbors (default from config)

        Returns:
            (n_query,) array of mean k-NN distances (higher = more anomalous)
        """
        if len(query_emb) == 0:
            return np.array([], dtype=np.float32)

        k = k or self._config.get("k", 5)

        query_tensor = torch.from_numpy(query_emb).float().to(self._device)

        # Compute distances: (n_query, n_ref)
        d = torch.cdist(query_tensor, self._normal_emb_tensor)

        # Top-k SMALLEST distances -> k nearest neighbors
        topk_dists, _ = torch.topk(d, k, largest=False)
        knn_mean = topk_dists.mean(dim=1)

        return knn_mean.cpu().numpy()

    def _compute_prdc(self, query_emb: np.ndarray) -> np.ndarray:
        """Compute PRDC features for a batch of query embeddings."""
        from .prdc import compute_prdc_batch

        return compute_prdc_batch(
            self._normal_embeddings,
            query_emb,
            k=self._config.get("k", 5),
            batch_size=Settings().DEFAULT_BATCH_SIZE,
            device=self._device,
        )

    def _score(self, prdc: np.ndarray) -> np.ndarray:
        """Return anomaly scores for PRDC features (legacy path)."""
        prdc_scaled = self._scaler.transform(prdc)
        name = type(self._detector).__name__

        if name == "GaussianMixture":
            raw = self._detector.score_samples(prdc_scaled)
            return -raw
        elif name == "IsolationForest":
            raw = self._detector.decision_function(prdc_scaled)
            return -raw
        elif name == "OneClassSVM":
            raw = self._detector.decision_function(prdc_scaled)
            return -raw
        else:
            return self._detector.score(prdc_scaled)

    def _to_confidence(self, raw_score: float) -> float:
        """
        Map raw anomaly score -> [0, 1] confidence that the log is FAULTY.

        We use a smoothed sigmoid centred on the threshold:
            z = (raw - threshold) / std
            confidence = sigmoid(z)
        This gives:
            score << threshold  -> confidence ~ 0  (confident normal)
            score ~= threshold  -> confidence ~= 0.5  (uncertain)
            score >> threshold  -> confidence ~= 1  (confident fault)
        """
        z = (raw_score - self._threshold) / self._normal_std
        z = max(-500.0, min(500.0, z))
        conf = 1.0 / (1.0 + math.exp(-z))
        return round(float(conf), 3)

    # ── Public API ──────────────────────────────────────────────────────────────

    async def detect_one(
        self,
        log: str,
        *,
        return_prdc: bool = False,
        return_normalized: bool = False,
    ) -> dict[str, Any]:
        """
        Classify a single log entry.

        Returns:
            dict with is_fault, confidence, raw_score, threshold, model_version,
            and optionally prdc / normalized_log.
        """
        if self._embedder is None:
            raise RuntimeError("K4Engine is not loaded. Call .load() first.")

        # 1. Normalize
        norm_log = normalize_log(log)

        # 2. Embed
        emb = await self._embed([norm_log])

        # 3. Score: KNN distance (primary, robust for dense embeddings)
        raw_score = float(self._knn_score(emb)[0])

        # 4. Decision
        is_fault = raw_score >= self._threshold
        confidence = self._to_confidence(raw_score)

        result: dict[str, Any] = {
            "is_fault": bool(is_fault),
            "confidence": confidence,
            "raw_score": round(raw_score, 6),
            "threshold": round(self._threshold, 6),
            "model_version": self._loaded_version,
        }

        if return_prdc:
            prdc = self._compute_prdc(emb)
            result["prdc"] = {
                "precision": round(float(prdc[0, 0]), 4),
                "recall": round(float(prdc[0, 1]), 4),
                "density": round(float(prdc[0, 2]), 4),
                "coverage": round(float(prdc[0, 3]), 4),
            }

        if return_normalized:
            result["normalized_log"] = norm_log

        return result

    async def detect_batch(
        self,
        logs: list[str],
        *,
        return_prdc: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Classify a batch of log entries.

        Processing is vectorised – the full batch is embedded together,
        then scored together, so throughput scales well.
        """
        if self._embedder is None:
            raise RuntimeError("K4Engine is not loaded. Call .load() first.")

        if not logs:
            return []

        # 1. Normalize
        norm_logs = [normalize_log(log) for log in logs]

        # 2. Embed (single batch)
        embeddings = await self._embed(norm_logs)

        # 3. Score: KNN distance
        raw_scores = self._knn_score(embeddings)

        results = []
        for i, score in enumerate(raw_scores):
            is_fault = score >= self._threshold
            confidence = self._to_confidence(float(score))

            item: dict[str, Any] = {
                "index": i,
                "is_fault": bool(is_fault),
                "confidence": confidence,
                "raw_score": round(float(score), 6),
            }

            if return_prdc:
                prdc_batch = self._compute_prdc(embeddings[i : i + 1])
                item["prdc"] = {
                    "precision": round(float(prdc_batch[0, 0]), 4),
                    "recall": round(float(prdc_batch[0, 1]), 4),
                    "density": round(float(prdc_batch[0, 2]), 4),
                    "coverage": round(float(prdc_batch[0, 3]), 4),
                }

            results.append(item)

        return results
