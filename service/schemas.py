"""
Pydantic schemas for FastAPI request / response models.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request schemas ──────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """Single-log detection request."""

    log: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw log text to classify.",
        examples=["[799 WARNING][rafale.c:14876]CPU IERR Detected: CPU Error Status Register 0x12 Value = 0xF"],
    )
    model_version: Optional[str] = Field(
        None,
        description="Model version to use. Defaults to the configured default.",
    )
    return_prdc: bool = Field(
        False,
        description="Include PRDC feature vector in the response.",
    )
    return_normalized: bool = Field(
        False,
        description="Include the normalized log text in the response.",
    )

    @field_validator("log")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchDetectRequest(BaseModel):
    """Batch detection request."""

    logs: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of log texts (max 1000 per request).",
    )
    model_version: Optional[str] = Field(None)
    return_prdc: bool = Field(False)

    @field_validator("logs")
    @classmethod
    def strip_logs(cls, v: list[str]) -> list[str]:
        return [log.strip() for log in v]


# ── Response schemas ─────────────────────────────────────────────────────────

class DetectResponse(BaseModel):
    """Single-log detection response."""

    is_fault: bool = Field(..., description="Whether the log is classified as a fault.")
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence in [0, 1].")]
    raw_score: float = Field(..., description="Raw anomaly score from the detector.")
    threshold: float = Field(..., description="Decision threshold used.")
    model_version: str = Field(..., description="Model version used for this prediction.")
    prdc: Optional[dict[str, float]] = Field(
        None,
        description="PRDC feature vector (Precision, Recall, Density, Coverage).",
    )
    normalized_log: Optional[str] = Field(
        None,
        description="Normalized log text after variable substitution.",
    )


class BatchItemResponse(BaseModel):
    """Single entry inside a batch response."""

    index: int = Field(..., description="0-based index matching the input order.")
    is_fault: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    raw_score: float
    prdc: Optional[dict[str, float]] = None


class BatchDetectResponse(BaseModel):
    """Batch detection response."""

    results: list[BatchItemResponse]
    summary: BatchSummary


class BatchSummary(BaseModel):
    total: int
    fault_count: int
    avg_confidence: float


# ── Health / info schemas ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    device: str
    loaded_models: list[str]
    cuda_available: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    embedder: str
    detector: str
    k: int
    n_training_samples: int
    embedding_dim: int
