"""
Model persistence: save K4 trained artifacts to disk and reload them.

Saves:
  config.json     – k, embedder_name, detector_type, n_training_samples
  normal_embeddings.npy – numpy array (n_ref, embedding_dim)
  scaler.pkl      – StandardScaler from sklearn
  detector.pkl    – GMM / KDE / OCSVM / DeepSVDD model
  threshold.json  – decision threshold + score statistics
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from .config import Settings


class ModelSaveError(Exception):
    pass


class ModelLoadError(Exception):
    pass


def get_model_path(model_version: str, filename: str) -> Path:
    """Return the absolute path to a file inside a model version directory."""
    settings = Settings()
    root = settings.MODELS_DIR / model_version
    if not root.is_dir():
        raise ModelLoadError(f"Model version '{model_version}' not found at {root}")
    path = root / filename
    if not path.exists():
        raise ModelLoadError(f"Model file '{filename}' not found in {root}")
    return path


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(
    model_version: str,
    config: dict[str, Any],
    normal_embeddings: np.ndarray,
    scaler: Any,
    detector: Any,
    threshold: float,
    normal_scores_mean: float,
    normal_scores_std: float,
    device: str,
) -> Path:
    """
    Persist all K4 artifacts to disk.

    Args:
        model_version:    Directory name under MODELS_DIR (e.g. "syslog_gmm_k5_v1")
        config:           Dict with embedder_name, detector_type, k, n_training_samples,
                          embedding_dim
        normal_embeddings: numpy array (n_ref, embedding_dim)
        scaler:           sklearn.preprocessing.StandardScaler
        detector:         Trained sklearn model
        threshold:        Decision threshold (optimal F1)
        normal_scores_mean: Mean of normal-score distribution
        normal_scores_std: Std  of normal-score distribution
        device:           Device string ("cuda" / "cpu")

    Returns:
        Path to the saved model directory.
    """
    settings = Settings()
    root = settings.MODELS_DIR / model_version
    root.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    cfg = {
        **config,
        "model_version": model_version,
        "device": device,
    }
    (root / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # 2. normal_embeddings.npy
    np.save(root / "normal_embeddings.npy", normal_embeddings)

    # 3. scaler.pkl (joblib is more stable for sklearn objects)
    joblib.dump(scaler, root / "scaler.pkl")

    # 4. detector.pkl
    joblib.dump(detector, root / "detector.pkl")

    # 5. threshold.json
    thresh = {
        "threshold": float(threshold),
        "normal_scores_mean": float(normal_scores_mean),
        "normal_scores_std": float(normal_scores_std),
    }
    (root / "threshold.json").write_text(json.dumps(thresh, indent=2))

    print(f"[ModelLoader] Saved model to {root}")
    return root


# ── Load ──────────────────────────────────────────────────────────────────────

def load_config(model_version: str) -> dict[str, Any]:
    path = get_model_path(model_version, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_normal_embeddings(model_version: str) -> np.ndarray:
    path = get_model_path(model_version, "normal_embeddings.npy")
    return np.load(path)


def load_scaler(model_version: str) -> Any:
    path = get_model_path(model_version, "scaler.pkl")
    return joblib.load(path)


def load_detector(model_version: str) -> Any:
    path = get_model_path(model_version, "detector.pkl")
    return joblib.load(path)


def load_threshold(model_version: str) -> dict[str, float]:
    path = get_model_path(model_version, "threshold.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all(model_version: str) -> dict[str, Any]:
    """Load all artifacts for a given model version. Returns a dict."""
    return {
        "config": load_config(model_version),
        "normal_embeddings": load_normal_embeddings(model_version),
        "scaler": load_scaler(model_version),
        "detector": load_detector(model_version),
        "threshold": load_threshold(model_version),
    }
