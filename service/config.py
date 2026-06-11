"""
Service configuration: device selection, paths, defaults.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Device ─────────────────────────────────────────────
    DEVICE: Literal["cuda", "cpu"] = "cuda"

    # ── Paths ──────────────────────────────────────────────
    # Base directory of this project (K4-service/)
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()
    MODELS_DIR: Path = BASE_DIR / "models"

    # Default model version when none specified
    DEFAULT_MODEL_VERSION: str = "default"

    # ── Model defaults (used by train_service) ──────────────
    DEFAULT_EMBEDDER: str = "all-MiniLM-L6-v2"
    DEFAULT_DETECTOR: Literal["gmm", "kde", "ocsvm", "deepsvd"] = "gmm"
    DEFAULT_K: int = 5
    DEFAULT_N_COMPONENTS: int = 3
    DEFAULT_BATCH_SIZE: int = 256
    DEFAULT_EMBEDDING_BATCH_SIZE: int = 512

    # ── Inference defaults ──────────────────────────────────
    # Warm up the model on startup to avoid cold-start latency
    WARM_UP: bool = True
    WARM_UP_SAMPLES: int = 10

    # ── Server ─────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    LOG_LEVEL: str = "info"

    # ── CORS ───────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_device() -> str:
    """Return the target device, respecting CUDA availability."""
    settings = Settings()
    if settings.DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    return settings.DEVICE


def resolve_embedder_path(embedder_name: str) -> str:
    """
    Resolve embedder model path.

    Priority:
      1. Local path already present under MODELS_DIR/<embedder_name>.
         We return the snapshot subdirectory directly (the directory that
         contains config.json / model.safetensors), so transformers can
         load the model without any network access.
      2. Fall back to the embedder name string so SentenceTransformer
         uses its default cache / download logic.

    Expected local layout:
        MODELS_DIR/
          <embedder_name>/          (e.g. all-MiniLM-L6-v2)
            blobs/
            refs/
            snapshots/
              <commit_hash>/        ← actual model files live here
                config.json
                model.safetensors
                ...
    """
    local_root = Settings().MODELS_DIR / embedder_name
    snapshots_dir = local_root / "snapshots"
    if snapshots_dir.is_dir():
        # Find the snapshot subdirectory (normally a symlink, but real dirs work too)
        candidates = list(snapshots_dir.iterdir())
        if candidates:
            # Use the first subdirectory (there should be exactly one)
            snapshot_path = candidates[0]
            # Verify it looks like a real snapshot (has config.json)
            if snapshot_path.joinpath("config.json").is_file():
                return str(snapshot_path)
    return embedder_name


def resolve_device() -> str:
    """Resolve device: explicit CUDA check, fallback to env var."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return os.getenv("DEVICE", "cpu")


settings = Settings()
