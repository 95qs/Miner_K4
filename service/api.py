"""
FastAPI application – routes and lifecycle management.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, resolve_device
from .engine import K4Engine
from .schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    BatchItemResponse,
    BatchSummary,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ModelInfoResponse,
)


# ── Global engine instance ────────────────────────────────────────────────────

_engine: K4Engine | None = None
_settings = Settings()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load the default model and warm up the engine.
    Shutdown: release resources.
    """
    global _engine

    # ── startup ────────────────────────────────────────────────────────────────
    device = resolve_device()
    print(f"[lifespan] Starting up, device={device}, CUDA available={_has_cuda()}")

    _engine = K4Engine()

    default_version = _settings.DEFAULT_MODEL_VERSION
    model_dir = _settings.MODELS_DIR / default_version

    if model_dir.exists():
        _engine.load(default_version)
        if _settings.WARM_UP:
            await asyncio.to_thread(_engine.warm_up, _settings.WARM_UP_SAMPLES)
    else:
        print(
            f"[lifespan] WARNING: default model '{default_version}' not found at "
            f"{model_dir}. The service will return 503 until a model is trained."
        )

    print("[lifespan] Startup complete.")
    yield

    # ── shutdown ────────────────────────────────────────────────────────────────
    print("[lifespan] Shutting down.")
    _engine = None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="K4 Log Anomaly Detection Service",
    description=(
        "REST API for detecting fault logs using the K4 algorithm "
        "(Sentence Embedding + PRDC + GMM/KDE/OCSVM detector)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_engine() -> K4Engine:
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialised. No engine loaded.",
        )
    return _engine


def _resolve_version(requested: str | None) -> str:
    return requested or _settings.DEFAULT_MODEL_VERSION


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Basic liveness and readiness probe."""
    engine = _get_engine()
    return HealthResponse(
        status="healthy",
        device=engine.device,
        loaded_models=[engine.loaded_version] if engine.loaded_version else [],
        cuda_available=_has_cuda(),
    )


@app.get("/models/{model_version}/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info(model_version: str) -> ModelInfoResponse:
    """Return metadata about a loaded model."""
    engine = _get_engine()
    if engine.loaded_version != model_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_version}' is not loaded.",
        )
    cfg = engine.config
    return ModelInfoResponse(
        model_version=model_version,
        embedder=cfg.get("embedder_name", "unknown"),
        detector=cfg.get("detector_type", "unknown"),
        k=cfg.get("k", 5),
        n_training_samples=cfg.get("n_training_samples", 0),
        embedding_dim=cfg.get("embedding_dim", 0),
    )


@app.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    tags=["detection"],
    summary="Detect a single log entry",
)
async def detect_single(request: DetectRequest) -> DetectResponse:
    """
    Classify one log entry as fault / normal and return a confidence score.

    The confidence is a value in [0, 1] where:
      - 1.0  → the model is highly confident this is a **fault**
      - 0.0  → the model is highly confident this is **normal**
      - 0.5  → uncertain
    """
    engine = _get_engine()
    result = await engine.detect_one(
        log=request.log,
        return_prdc=request.return_prdc,
        return_normalized=request.return_normalized,
    )
    return DetectResponse(**result)


@app.post(
    "/api/v1/detect/batch",
    response_model=BatchDetectResponse,
    tags=["detection"],
    summary="Detect a batch of log entries",
)
async def detect_batch(request: BatchDetectRequest) -> BatchDetectResponse:
    """
    Classify up to 1000 log entries in one request.

    Processing is vectorised for throughput. The response returns each entry's
    result plus a summary block.
    """
    engine = _get_engine()
    results = await engine.detect_batch(
        logs=request.logs,
        return_prdc=request.return_prdc,
    )

    fault_count = sum(1 for r in results if r["is_fault"])
    confidences = [r["confidence"] for r in results]

    items = [BatchItemResponse(**{k: v for k, v in r.items() if k != "raw_score"}) for r in results]

    return BatchDetectResponse(
        results=items,
        summary=BatchSummary(
            total=len(results),
            fault_count=fault_count,
            avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        ),
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "K4 Log Anomaly Detection Service", "version": "1.0.0"})
