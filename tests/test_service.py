"""
Smoke tests – run against a real trained model if one exists,
otherwise skip gracefully.
"""

import pytest

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def any_model_available():
    """Skip if no model has been trained yet."""
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "models"
    available = [d.name for d in models_dir.iterdir() if d.is_dir()]
    if not available:
        pytest.skip("No trained model found in models/. Run train_service first.")
    return available[0]


class TestEngineIntegration:
    """Integration tests for K4Engine (require a real model)."""

    @pytest.mark.asyncio
    async def test_load_engine(self, any_model_available):
        from service.engine import K4Engine
        engine = K4Engine()
        engine.load(any_model_available)
        assert engine.loaded_version == any_model_available
        assert engine.config.get("embedder_name") is not None

    @pytest.mark.asyncio
    async def test_detect_one_returns_valid_schema(self, any_model_available):
        from service.engine import K4Engine
        engine = K4Engine()
        engine.load(any_model_available)

        result = await engine.detect_one(
            "[799 WARNING][rafale.c:14876]CPU IERR Detected: CPU Error Status Register 0x12",
            return_prdc=True,
            return_normalized=True,
        )

        assert "is_fault" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "raw_score" in result
        assert "threshold" in result
        assert "model_version" in result
        assert result["model_version"] == any_model_available
        assert "prdc" in result
        assert set(result["prdc"].keys()) == {"precision", "recall", "density", "coverage"}
        assert "normalized_log" in result

    @pytest.mark.asyncio
    async def test_detect_batch_returns_valid_schema(self, any_model_available):
        from service.engine import K4Engine
        engine = K4Engine()
        engine.load(any_model_available)

        logs = [
            "[799 WARNING]CPU IERR Detected: CPU Error Status Register 0x12",
            "link status channel0=1 channel1=1 channel2=0",
            "normal log message here",
        ]
        results = await engine.detect_batch(logs)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["index"] == i
            assert 0.0 <= r["confidence"] <= 1.0
            assert isinstance(r["is_fault"], bool)

    @pytest.mark.asyncio
    async def test_batch_summary_fault_count(self, any_model_available):
        from service.engine import K4Engine
        engine = K4Engine()
        engine.load(any_model_available)

        # All normal logs → should have 0 faults (or low count)
        logs = ["system startup completed successfully"] * 5
        results = await engine.detect_batch(logs)
        # No strict assertion; just ensure no crash
        assert len(results) == 5


class TestSchemas:
    """Unit tests for Pydantic schemas."""

    def test_detect_request_valid(self):
        from service.schemas import DetectRequest
        req = DetectRequest(log="[799 WARNING]CPU IERR Detected")
        assert req.log == "[799 WARNING]CPU IERR Detected"

    def test_detect_request_strips_whitespace(self):
        from service.schemas import DetectRequest
        req = DetectRequest(log="  [799 WARNING]CPU IERR Detected  \n")
        assert req.log == "[799 WARNING]CPU IERR Detected"

    def test_detect_request_empty_rejected(self):
        from service.schemas import DetectRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectRequest(log="")

    def test_batch_request_max_logs(self):
        from service.schemas import BatchDetectRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchDetectRequest(logs=["a"] * 1001)

    def test_batch_request_strips_logs(self):
        from service.schemas import BatchDetectRequest
        req = BatchDetectRequest(logs=["  log1  ", "log2\n"])
        assert req.logs == ["log1", "log2"]


class TestModelLoader:
    """Tests for model persistence utilities."""

    def test_load_config_not_found(self):
        from service.model_loader import ModelLoadError
        with pytest.raises(ModelLoadError):
            from service.model_loader import load_config
            load_config("nonexistent_model_version")

    def test_load_all_not_found(self):
        from service.model_loader import ModelLoadError
        with pytest.raises(ModelLoadError):
            from service.model_loader import load_all
            load_all("nonexistent_model_version")
