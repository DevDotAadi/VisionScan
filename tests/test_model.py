"""Unit tests for VisionScan Global — core engine."""

from __future__ import annotations

import numpy as np
import pytest

from utils.engine import (
    Prediction,
    RiskLevel,
    Certainty,
    _calibrate,
    _classify_risk,
    _classify_certainty,
    _recommend,
    preprocess,
    predict,
)


class TestPreprocess:
    def test_rgb_array(self):
        rgb = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        display, batch = preprocess(rgb)
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        assert 0.0 <= batch.max() <= 1.0

    def test_rgba_array(self):
        rgba = np.random.randint(0, 255, (80, 80, 4), dtype=np.uint8)
        _, batch = preprocess(rgba)
        assert batch.shape == (1, 224, 224, 3)

    def test_grayscale_array(self):
        gray = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
        _, batch = preprocess(gray)
        assert batch.shape == (1, 224, 224, 3)

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            preprocess("/nonexistent/path.jpg")


class TestCalibration:
    def test_identity(self):
        assert abs(_calibrate(0.7, 1.0) - 0.7) < 1e-5

    def test_softening(self):
        result = _calibrate(0.9, 3.0)
        assert result < 0.9
        assert result > 0.5

    def test_sharpening(self):
        assert _calibrate(0.7, 0.5) > 0.7

    def test_symmetry(self):
        hi = _calibrate(0.8, 2.0)
        lo = _calibrate(0.2, 2.0)
        assert abs(hi + lo - 1.0) < 1e-5

    def test_extremes(self):
        assert 0 < _calibrate(0.001, 2.0) < 1
        assert 0 < _calibrate(0.999, 2.0) < 1


class TestRiskClassification:
    def test_high_malignant(self):
        assert _classify_risk("Malignant", 0.95, 0.95) == RiskLevel.HIGH

    def test_low_benign(self):
        assert _classify_risk("Benign", 0.95, 0.05) == RiskLevel.LOW

    def test_inconclusive(self):
        assert _classify_risk("Benign", 0.55, 0.45) == RiskLevel.INCONCLUSIVE

    def test_moderate(self):
        assert _classify_risk("Malignant", 0.65, 0.65) == RiskLevel.INCONCLUSIVE
        assert _classify_risk("Malignant", 0.70, 0.70) == RiskLevel.MODERATE


class TestCertainty:
    def test_certain(self):
        assert _classify_certainty(0.90, 0.90) == Certainty.CERTAIN

    def test_uncertain(self):
        assert _classify_certainty(0.60, 0.70) == Certainty.UNCERTAIN

    def test_inconclusive(self):
        assert _classify_certainty(0.55, 0.50) == Certainty.INCONCLUSIVE


class TestRecommendation:
    def test_all_levels_produce_text(self):
        for risk in RiskLevel:
            for cert in Certainty:
                text = _recommend(risk, cert)
                assert isinstance(text, str)
                assert len(text) > 20

    def test_high_risk_urgency(self):
        assert "dermatology" in _recommend(RiskLevel.HIGH, Certainty.CERTAIN).lower()

    def test_inconclusive_hedging(self):
        text = _recommend(RiskLevel.INCONCLUSIVE, Certainty.INCONCLUSIVE)
        assert "uncertain" in text.lower() or "inconclusive" in text.lower()


class TestPredictionDataclass:
    def test_frozen(self):
        p = Prediction("Benign", 0.9, 0.1, RiskLevel.LOW, Certainty.CERTAIN, "ok")
        with pytest.raises(AttributeError):
            p.label = "Malignant"  # type: ignore[misc]

    def test_timestamp_populated(self):
        p = Prediction("Benign", 0.9, 0.1, RiskLevel.LOW, Certainty.CERTAIN, "ok")
        assert len(p.timestamp) > 10
