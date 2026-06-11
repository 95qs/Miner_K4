"""
Training script: build a K4 model from normal logs and persist it.

Usage:
    # From the K4-service/ directory:
    python -m service.train_service \
        --data-path ../K4/syslog_dev \
        --model-version syslog_gmm_k5_v1 \
        --embedder all-MiniLM-L6-v2 \
        --detector gmm \
        --k 5

    # Point at any directory containing train_normal.jsonl:
    python -m service.train_service \
        --data-path /path/to/my/logs \
        --model-version my_model \
        --train-file train_normal.jsonl \
        --val-file val_normal.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from .prdc import compute_prdc_batch
from .detectors import (
    DetectorFactory,
)
from .preprocess import normalize_log

from .config import Settings, resolve_embedder_path
from .model_loader import save_model


def resolve_device(requested: str | None) -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_jsonl_logs(file_path: str) -> list[str]:
    """Load log texts from a JSONL file (one dict with 'content' key per line)."""
    logs = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            logs.append(item["content"])
    return logs


def load_csv_logs(file_path: str) -> list[str]:
    """Load log texts from a CSV file (expects 'content' column)."""
    import pandas as pd
    df = pd.read_csv(file_path)
    return df["content"].tolist()


def run_training(
    data_dir: str | Path | None,
    model_version: str,
    embedder_name: str = "all-MiniLM-L6-v2",
    detector_type: str = "gmm",
    k: int = 5,
    n_components: int = 3,
    device: str | None = None,
    normalize: bool = True,
    seed: int = 42,
    train_file: str = "train_normal.jsonl",
    val_file: str = "val_normal.jsonl",
) -> dict:
    """
    Train a K4 model from normal log files.

    Expected layout under data_dir (or files given as absolute paths):
        train_normal.jsonl   – normal logs for training
        val_normal.jsonl     – normal logs for threshold calibration (optional)
        test_normal.jsonl    – held-out normal logs (optional)
        test_anomaly.csv     – labelled fault logs (optional)

    If val_file / test files are missing the script will skip those steps.
    """
    data_dir = Path(data_dir) if data_dir else None
    device = resolve_device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    settings = Settings()
    models_dir = settings.MODELS_DIR

    print("=" * 70)
    print("K4 Training Service")
    print(f"  Model version:  {model_version}")
    print(f"  Embedder:        {embedder_name}")
    print(f"  Detector:        {detector_type}")
    print(f"  k:               {k}")
    print(f"  Device:          {device}")
    print(f"  Normalize:       {normalize}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    def load(path: str) -> list[str]:
        p = Path(path)
        if not p.is_absolute():
            p = data_dir / path if data_dir else Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        if p.suffix == ".jsonl":
            return load_jsonl_logs(str(p))
        elif p.suffix == ".csv":
            return load_csv_logs(str(p))
        else:
            raise ValueError(f"Unsupported file type: {p}")

    print("\n[Data] Loading logs …")
    train_logs = load(train_file)
    print(f"  Train: {len(train_logs):,} logs")

    val_logs = []
    val_path = (data_dir / val_file) if data_dir else Path(val_file)
    if val_file and val_path.exists():
        try:
            val_logs = load(val_file)
            print(f"  Val:   {len(val_logs):,} logs")
        except Exception:
            pass

    if normalize:
        print("\n[Normalize] Applying variable substitution …")
        train_logs = [normalize_log(l) for l in train_logs]
        if val_logs:
            val_logs = [normalize_log(l) for l in val_logs]

    # ── 2. Embedding ───────────────────────────────────────────────────────────
    print(f"\n[Embedding] Loading {embedder_name} on {device} …")
    embedder = SentenceTransformer(
        resolve_embedder_path(embedder_name), device=device
    )

    t0 = time.time()
    print("  Encoding train logs …")
    train_embeddings = embedder.encode(
        train_logs,
        batch_size=settings.DEFAULT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embedding_time = time.time() - t0
    print(f"  Train embedding time: {embedding_time:.2f}s")
    print(f"  Shape: {train_embeddings.shape}")

    val_embeddings = None
    if val_logs:
        t0 = time.time()
        print("  Encoding val logs …")
        val_embeddings = embedder.encode(
            val_logs,
            batch_size=settings.DEFAULT_EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"  Val embedding time: {time.time() - t0:.2f}s")

    # ── 3. PRDC ────────────────────────────────────────────────────────────────
    print(f"\n[PRDC] Computing descriptors (k={k}) …")
    t0 = time.time()
    prdc_train = compute_prdc_batch(
        train_embeddings, train_embeddings,
        k=k, batch_size=settings.DEFAULT_BATCH_SIZE, device=device,
    )
    prdc_time = time.time() - t0
    print(f"  PRDC train time: {prdc_time:.2f}s")

    prdc_val = None
    if val_embeddings is not None:
        t0 = time.time()
        prdc_val = compute_prdc_batch(
            train_embeddings, val_embeddings,
            k=k, batch_size=settings.DEFAULT_BATCH_SIZE, device=device,
        )
        print(f"  PRDC val time: {time.time() - t0:.2f}s")

    # ── 4. Detector ────────────────────────────────────────────────────────────
    print(f"\n[Detector] Training {detector_type} …")
    t0 = time.time()

    detector_params = {}
    if detector_type == "gmm":
        detector_params["n_components"] = n_components

    detector = DetectorFactory.create(detector_type, **detector_params)
    detector.fit(prdc_train)
    print(f"  Detector training time: {time.time() - t0:.4f}s")

    # ── 5. Threshold calibration ──────────────────────────────────────────────
    # If we have labelled anomaly data we can calibrate a real threshold;
    # otherwise fall back to the detector's own internal threshold.
    if prdc_val is not None:
        # Self-evaluation on val set (val is also "normal")
        val_scores = detector.score(prdc_val)
        # Use 95th percentile of normal scores as threshold
        threshold = float(np.percentile(val_scores, 95))
        normal_mean = float(val_scores.mean())
        normal_std = float(val_scores.std())
        print(f"\n[Threshold] {threshold:.6f}  (95th percentile of val scores)")
        print(f"  Val score mean={normal_mean:.4f} std={normal_std:.4f}")
    else:
        # Conservative default: use a small fraction above detector's mean
        train_scores = detector.score(prdc_train)
        normal_mean = float(train_scores.mean())
        normal_std = float(train_scores.std())
        threshold = float(np.percentile(train_scores, 95))
        print(f"\n[Threshold] {threshold:.6f}  (95th percentile of train scores, fallback)")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    print(f"\n[Save] Persisting model to {models_dir}/{model_version} …")
    model_path = save_model(
        model_version=model_version,
        config={
            "embedder_name": embedder_name,
            "detector_type": detector_type,
            "k": k,
            "n_training_samples": len(train_logs),
            "embedding_dim": int(train_embeddings.shape[1]),
            "n_components": n_components,
            "detector_params": detector_params,
        },
        normal_embeddings=train_embeddings,
        scaler=detector.scaler,
        detector=detector.model,
        threshold=threshold,
        normal_scores_mean=normal_mean,
        normal_scores_std=normal_std,
        device=device,
    )

    total_time = embedding_time + prdc_time
    print(f"\n[Done] Training complete in {total_time:.2f}s")
    print(f"       Model saved at: {model_path}")

    return {
        "model_version": model_version,
        "embedder": embedder_name,
        "detector": detector_type,
        "k": k,
        "n_training_samples": len(train_logs),
        "embedding_dim": int(train_embeddings.shape[1]),
        "threshold": threshold,
        "normal_mean": normal_mean,
        "normal_std": normal_std,
        "embedding_time": embedding_time,
        "prdc_time": prdc_time,
        "total_time": total_time,
        "device": device,
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist a K4 model")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Directory containing train/val files (or absolute paths)")
    parser.add_argument("--model-version", type=str, required=True,
                        help="Directory name for the saved model, e.g. 'syslog_gmm_k5_v1'")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--detector", type=str, default="gmm",
                        choices=["gmm", "kde", "ocsvm", "deepsvd"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", type=str, default="train_normal.jsonl")
    parser.add_argument("--val-file", type=str, default="val_normal.jsonl")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write result summary to this JSON file")

    args = parser.parse_args()

    result = run_training(
        data_dir=args.data_path,
        model_version=args.model_version,
        embedder_name=args.embedder,
        detector_type=args.detector,
        k=args.k,
        n_components=args.n_components,
        device=args.device,
        normalize=not args.no_normalize,
        seed=args.seed,
        train_file=args.train_file,
        val_file=args.val_file,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
