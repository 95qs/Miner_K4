"""
K4 inference engine: wraps the original K4 modules with model-loading
and confidence-mapping logic.

Design goals:
  - Lazy loading: embedder / embeddings are loaded only once at startup.
  - Thread-safety: the embedder is protected by an asyncio Lock so that
    concurrent requests serialise embedding calls (important on single GPU).
  - Confidence: raw anomaly scores are mapped to [0, 1] via a smoothed
    sigmoid based on the known normal-score distribution.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, resolve_device, resolve_embedder_path
from .model_loader import load_all
from .preprocess import normalize_log


class K4Engine:
    """
    Inference engine backed by a saved K4 model.

    Lifecycle:
      K4Engine.load(model_version)  → load all artifacts into memory
      K4Engine.detect_one(log)      → dict with is_fault / confidence / ...
      K4Engine.detect_batch(logs)    → list[dict]
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

        # Detector + scaler
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
        # We only need to run through the embedder; the detector part
        # is negligible, so we skip the full pipeline.
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
            # Run in a thread pool so FastAPI's async event loop is not blocked
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
        """Return anomaly scores for PRDC features."""
        prdc_scaled = self._scaler.transform(prdc)
        return self._detector.score(prdc_scaled)

    def _to_confidence(self, raw_score: float) -> float:
        """
        Map raw anomaly score → [0, 1] confidence that the log is FAULTY.

        We use a smoothed sigmoid centred on the threshold:
            z = (raw - threshold) / std
            confidence = sigmoid(z)
        This gives:
            score << threshold  → confidence ≈ 0  (confident normal)
            score ≈ threshold  → confidence ≈ 0.5 (uncertain)
            score >> threshold → confidence ≈ 1  (confident fault)
        """
        z = (raw_score - self._threshold) / self._normal_std
        # Clamp to avoid math overflow in exp
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

        # 3. PRDC
        prdc = self._compute_prdc(emb)

        # 4. Score
        raw_score = float(np.atleast_1d(self._score(prdc))[0])

        # 5. Decision
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

        # 1. Normalize
        norm_logs = [normalize_log(log) for log in logs]

        # 2. Embed (single batch)
        embeddings = await self._embed(norm_logs)

        # 3. PRDC
        prdc = self._compute_prdc(embeddings)

        # 4. Score
        raw_scores = self._score(prdc)

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
                item["prdc"] = {
                    "precision": round(float(prdc[i, 0]), 4),
                    "recall": round(float(prdc[i, 1]), 4),
                    "density": round(float(prdc[i, 2]), 4),
                    "coverage": round(float(prdc[i, 3]), 4),
                }

            results.append(item)

        return results
