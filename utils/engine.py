"""Core inference engine for VisionScan Global.

Provides dataclass-driven prediction, temperature-scaled calibration,
risk stratification, and clinical recommendation generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

IMG_SIZE: int = 224
_MODEL_CACHE: dict[str, tf.keras.Model] = {}


# ── Enums & Dataclasses ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    INCONCLUSIVE = "Inconclusive"


class Certainty(str, Enum):
    CERTAIN = "Certain"
    UNCERTAIN = "Uncertain"
    INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float
    prob_malignant: float
    risk: RiskLevel
    certainty: Certainty
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


# ── Model Loading ────────────────────────────────────────────────────

def load_model(path: str | Path = "models/visionscan_mobilenet.h5") -> tf.keras.Model:
    """Load a Keras model with singleton caching."""
    key = str(path)
    if key not in _MODEL_CACHE:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        log.info("Loading model from %s", p)
        _MODEL_CACHE[key] = tf.keras.models.load_model(str(p), compile=False)
    return _MODEL_CACHE[key]


# ── Image Preprocessing ─────────────────────────────────────────────

def preprocess(image: str | Path | np.ndarray, size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Prepare an image for inference.

    Accepts a file path or an RGB/RGBA/grayscale numpy array.
    Returns ``(display_rgb, batch)`` where batch has shape ``(1, size, size, 3)``.
    """
    if isinstance(image, (str, Path)):
        raw = cv2.imread(str(image))
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[-1] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return rgb, normalised[np.newaxis]


# ── Temperature Scaling ──────────────────────────────────────────────

def _calibrate(prob: float, temperature: float = 1.0) -> float:
    """Apply temperature scaling to a sigmoid probability."""
    if temperature == 1.0:
        return prob
    clamped = np.clip(prob, 1e-7, 1.0 - 1e-7)
    logit = float(np.log(clamped / (1.0 - clamped)))
    return float(1.0 / (1.0 + np.exp(-logit / temperature)))


# ── Risk Stratification ─────────────────────────────────────────────

_INCONCLUSIVE_LO = 0.35
_INCONCLUSIVE_HI = 0.65


def _classify_risk(label: str, confidence: float, prob: float) -> RiskLevel:
    if _INCONCLUSIVE_LO <= prob <= _INCONCLUSIVE_HI:
        return RiskLevel.INCONCLUSIVE
    if label == "Malignant":
        return RiskLevel.HIGH if confidence >= 0.80 else RiskLevel.MODERATE
    return RiskLevel.LOW if confidence >= 0.80 else RiskLevel.MODERATE


def _classify_certainty(confidence: float, prob: float) -> Certainty:
    if _INCONCLUSIVE_LO <= prob <= _INCONCLUSIVE_HI:
        return Certainty.INCONCLUSIVE
    return Certainty.CERTAIN if confidence >= 0.75 else Certainty.UNCERTAIN


def _recommend(risk: RiskLevel, certainty: Certainty) -> str:
    if risk is RiskLevel.HIGH:
        return (
            "The model detected features associated with elevated risk. "
            "A prompt dermatology consultation is recommended."
        )
    if risk is RiskLevel.INCONCLUSIVE or certainty is Certainty.INCONCLUSIVE:
        return (
            "The model is uncertain. This result should be considered inconclusive, "
            "and professional evaluation is recommended if the lesion is concerning."
        )
    if certainty is Certainty.UNCERTAIN:
        return (
            "The model shows moderate confidence. Consider monitoring the lesion "
            "and consulting a dermatologist for a professional opinion."
        )
    return (
        "The lesion appears lower risk based on the model, but ongoing monitoring "
        "and professional evaluation remain important."
    )


# ── Prediction ───────────────────────────────────────────────────────

def predict(model: tf.keras.Model, batch: np.ndarray, *, temperature: float = 1.0) -> Prediction:
    """Run inference and return a fully-structured ``Prediction``."""
    raw = float(model.predict(batch, verbose=0)[0, 0])
    prob = _calibrate(raw, temperature)

    is_malignant = prob >= 0.5
    label = "Malignant" if is_malignant else "Benign"
    confidence = prob if is_malignant else 1.0 - prob

    risk = _classify_risk(label, confidence, prob)
    certainty = _classify_certainty(confidence, prob)
    recommendation = _recommend(risk, certainty)

    return Prediction(
        label=label,
        confidence=round(confidence, 4),
        prob_malignant=round(prob, 4),
        risk=risk,
        certainty=certainty,
        recommendation=recommendation,
    )
