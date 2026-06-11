"""
Integration tests for K4 anomaly detection using real labeled data.

These tests run against the default trained model and verify detection
accuracy on actual normal and anomaly logs from syslog_dev/.

Tests use KNN distance scoring (the inference engine's primary scoring method).
"""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytest_asyncio"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir():
    base = Path(__file__).parent.parent / ".." / "K4" / "syslog_dev"
    if not base.exists():
        pytest.skip(f"Data directory not found: {base}")
    return base


@pytest.fixture(scope="module")
def anomaly_logs(data_dir):
    logs = []
    csv_path = data_dir / "test_anomaly.csv"
    with open(csv_path, encoding="utf-8") as f:
        f.readline()  # skip header: sn,content,severity,fault_type,label
        for line in f:
            parts = line.split(",", 2)
            if len(parts) >= 3:
                logs.append(parts[2].strip())
    return logs


@pytest.fixture(scope="module")
def normal_logs(data_dir):
    logs = []
    val_path = data_dir / "val_normal.jsonl"
    with open(val_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            logs.append(json.loads(line)["content"])
    return logs


@pytest.fixture
def engine():
    """Load the default model once per test function."""
    from service.engine import K4Engine

    eng = K4Engine()
    eng.load("default")
    return eng


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestEngineSmoke:
    """Basic engine loading and schema checks."""

    @pytest.mark.asyncio
    async def test_engine_loads(self, engine):
        assert engine.loaded_version == "default"
        assert engine.config.get("embedder_name") is not None
        assert engine._normal_embeddings is not None
        assert engine._normal_embeddings.shape[0] == 30000

    @pytest.mark.asyncio
    async def test_detect_one_schema(self, engine):
        result = await engine.detect_one(
            "test log message",
            return_prdc=True,
            return_normalized=True,
        )
        assert "is_fault" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "raw_score" in result
        assert "threshold" in result
        assert result["threshold"] > 0
        assert "model_version" in result
        assert "normalized_log" in result

    @pytest.mark.asyncio
    async def test_detect_batch_schema(self, engine):
        results = await engine.detect_batch(
            ["log a", "log b", "log c"],
            return_prdc=True,
        )
        assert len(results) == 3
        for r in results:
            assert "index" in r
            assert "is_fault" in r
            assert "confidence" in r
            assert "raw_score" in r
            assert "prdc" in r

    @pytest.mark.asyncio
    async def test_knn_score_is_used(self, engine):
        """
        Verify that the engine uses KNN distance scoring (not PRDC + ML detector).

        A real normal log from the training distribution should score near 0,
        while a deliberately out-of-manifold query should score much higher.
        """
        # Real normal log (similar to training data patterns)
        normal_result = await engine.detect_one(
            "[10704027.120000] NCSI(eth1): Get Link Status : Failed.Parameters Invalid"
        )

        # Deliberately weird query (out of manifold)
        ood_result = await engine.detect_one(
            "XYZ99ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
        )

        # OOD should score HIGHER than normal (KNN distance: farther = more anomalous)
        assert ood_result["raw_score"] > normal_result["raw_score"], (
            f"OOD score ({ood_result['raw_score']:.4f}) should be > "
            f"normal score ({normal_result['raw_score']:.4f})"
        )


# ---------------------------------------------------------------------------
# Accuracy tests on labeled data
# ---------------------------------------------------------------------------

class TestDetectionAccuracy:
    """Verify detection accuracy on real labeled logs."""

    @pytest.mark.asyncio
    async def test_anomaly_logs_are_detected(self, engine, anomaly_logs):
        """
        All labeled anomaly logs should be detected as faults.

        The test data covers: PCI faults, fan faults, CPU faults.
        These are out-of-manifold compared to the normal training distribution.
        """
        detected = 0
        threshold = engine._threshold
        scores = []

        for log in anomaly_logs:
            result = await engine.detect_one(log)
            scores.append(result["raw_score"])
            if result["is_fault"]:
                detected += 1

        detection_rate = detected / len(anomaly_logs)
        min_score = min(scores)
        max_score = max(scores)

        # At least 95% of anomalies should be detected
        assert detection_rate >= 0.95, (
            f"Anomaly detection rate {detection_rate*100:.1f}% < 95%. "
            f"Detected {detected}/{len(anomaly_logs)}. "
            f"Min score: {min_score:.4f}, max: {max_score:.4f}, threshold: {threshold:.4f}"
        )

        # All anomaly scores should be above threshold
        below_threshold = sum(1 for s in scores if s < threshold)
        assert below_threshold == 0, (
            f"{below_threshold} anomaly logs scored below threshold "
            f"(threshold={threshold:.4f})"
        )

    @pytest.mark.asyncio
    async def test_normal_logs_not_misclassified(self, engine, normal_logs):
        """
        Most normal logs should NOT be classified as faults.

        Allow up to 5% false positive rate (matching the threshold calibration).
        """
        fp = 0
        threshold = engine._threshold
        scores = []

        for log in normal_logs:
            result = await engine.detect_one(log)
            scores.append(result["raw_score"])
            if result["is_fault"]:
                fp += 1

        fp_rate = fp / len(normal_logs)

        # Tolerance: up to 5% false positives
        assert fp_rate <= 0.05, (
            f"False positive rate {fp_rate*100:.1f}% > 5%. "
            f"FP={fp}/{len(normal_logs)}. "
            f"Score stats: min={min(scores):.4f}, max={max(scores):.4f}, "
            f"mean={sum(scores)/len(scores):.4f}, threshold={threshold:.4f}"
        )

    @pytest.mark.asyncio
    async def test_score_separation(self, engine, anomaly_logs, normal_logs):
        """
        Verify that anomaly and normal logs produce clearly separated scores.

        There should be a wide gap between the highest normal score
        and the lowest anomaly score.
        """
        normal_scores = []
        anomaly_scores = []

        for log in normal_logs[:200]:
            result = await engine.detect_one(log)
            normal_scores.append(result["raw_score"])

        for log in anomaly_logs:
            result = await engine.detect_one(log)
            anomaly_scores.append(result["raw_score"])

        max_normal = max(normal_scores)
        min_anomaly = min(anomaly_scores)
        gap = min_anomaly - max_normal

        assert gap > 0, (
            f"No gap between normal and anomaly scores. "
            f"max_normal={max_normal:.4f}, min_anomaly={min_anomaly:.4f}. "
            f"Threshold may be miscalibrated."
        )

    @pytest.mark.asyncio
    async def test_specific_known_anomalies(self, engine):
        """
        Test a set of known anomaly patterns that should always be detected.
        """
        known_anomalies = [
            "Critical, Category: System Health, MessageID: RDU0002, Message: Fan redundancy is lost.",
            "[254 : 294 WARNING] [0x78] CPU_Conf_Status | Configuration Error | Processor | State Asserted",
            "PCI故障",
            "FAN3 Front Speed reading lower than threshold Lower Critical",
        ]

        for log in known_anomalies:
            result = await engine.detect_one(log)
            assert result["is_fault"], (
                f"Known anomaly not detected: '{log[:60]}'. "
                f"Score={result['raw_score']:.4f}, threshold={result['threshold']:.4f}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_log_returns_valid_response(self, engine):
        # Empty-ish logs should not crash
        result = await engine.detect_one("   ")
        assert "is_fault" in result
        assert "raw_score" in result

    @pytest.mark.asyncio
    async def test_very_long_log(self, engine):
        # Very long log should be handled
        long_log = "word " * 1000
        result = await engine.detect_one(long_log)
        assert "is_fault" in result
        assert "raw_score" in result

    @pytest.mark.asyncio
    async def test_batch_empty(self, engine):
        results = await engine.detect_batch([])
        assert results == []
