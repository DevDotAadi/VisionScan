"""Grad-CAM explainability for VisionScan Global.

Generates class-activation heatmaps that highlight the image regions
most influential to the model's prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


def find_last_conv_layer(model: tf.keras.Model) -> str | None:
    """Return the name of the deepest Conv2D layer, searching nested sub-models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    return inner.name
    return None


def _build_gradient_model(model: tf.keras.Model, layer_name: str):
    """Build a model that outputs both conv activations and final predictions."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                target = layer.get_layer(layer_name)
                return tf.keras.Model(
                    inputs=model.input,
                    outputs=[target.output, model.output],
                )
            except ValueError:
                pass
    return tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )


def compute_heatmap(batch: np.ndarray, model: tf.keras.Model, layer_name: str) -> np.ndarray:
    """Compute a Grad-CAM heatmap for the given preprocessed batch.

    Returns a float32 array of shape ``(H, W)`` normalised to ``[0, 1]``.
    Falls back to a uniform map if gradient computation fails.
    """
    try:
        grad_model = _build_gradient_model(model, layer_name)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            target = preds[:, 0]

        grads = tape.gradient(target, conv_out)
        if grads is None:
            log.warning("Grad-CAM: gradients were None — returning uniform heatmap")
            return np.full((7, 7), 0.5, dtype=np.float32)

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
        cam = np.maximum(cam, 0)
        peak = cam.max()
        return cam / peak if peak > 0 else np.full_like(cam, 0.5)

    except Exception:
        log.exception("Grad-CAM computation failed")
        return np.full((7, 7), 0.5, dtype=np.float32)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto an RGB image.

    Args:
        image: RGB uint8 array of any size.
        heatmap: Float heatmap (any spatial size).
        alpha: Heatmap opacity.

    Returns:
        Blended RGB uint8 array matching ``image`` dimensions.
    """
    h, w = image.shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    coloured = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    img = image if image.dtype == np.uint8 else np.uint8(image * 255)
    return np.uint8(coloured * alpha + img * (1 - alpha))
