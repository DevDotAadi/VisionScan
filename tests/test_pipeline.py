"""Integration tests for VisionScan Global — pipeline, config, structure."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


class TestConfig:
    cfg_path = ROOT / "configs" / "default.yaml"

    def test_exists(self):
        assert self.cfg_path.exists()

    def test_parseable(self):
        cfg = yaml.safe_load(self.cfg_path.read_text())
        assert isinstance(cfg, dict)
        assert "model_name" in cfg

    def test_valid_model_name(self):
        cfg = yaml.safe_load(self.cfg_path.read_text())
        assert cfg["model_name"] in {"mobilenetv2", "efficientnetv2", "convnext"}

    def test_augmentation_keys(self):
        cfg = yaml.safe_load(self.cfg_path.read_text())
        aug = cfg.get("augmentation", {})
        assert "rotation_range" in aug
        assert "horizontal_flip" in aug


class TestFocalLoss:
    def test_positive_loss(self):
        import tensorflow as tf
        from train import FocalLoss
        loss = FocalLoss(0.25, 2.0)
        y_true = tf.constant([[0.0], [1.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.1], [0.9], [0.8], [0.2]])
        assert loss(y_true, y_pred).numpy() > 0

    def test_near_zero_for_perfect(self):
        import tensorflow as tf
        from train import FocalLoss
        loss = FocalLoss(0.25, 2.0)
        y_true = tf.constant([[0.0], [1.0]])
        y_pred = tf.constant([[0.001], [0.999]])
        assert loss(y_true, y_pred).numpy() < 0.01


class TestMetrics:
    def test_specificity(self):
        from utils.metrics import specificity
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        assert abs(specificity(y_true, y_pred) - 2 / 3) < 1e-5

    def test_specificity_perfect(self):
        from utils.metrics import specificity
        assert specificity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


@pytest.mark.parametrize("filepath", [
    "app.py", "train.py", "gradcam.py", "batch_test.py", "pdf_report.py",
    "utils/engine.py", "utils/metrics.py",
    "configs/default.yaml", "requirements.txt",
    "utils/gemini_client.py", "utils/prompt_templates.py", "utils/security_utils.py", "utils/db_utils.py",
    "README.md", "LICENSE", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
    "Dockerfile", ".github/workflows/ci.yml",
])
def test_required_file_exists(filepath: str):
    assert (ROOT / filepath).exists(), f"Missing: {filepath}"
