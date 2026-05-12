"""VisionScan Global — Modular Training Framework.

Config-driven, multi-architecture training with focal loss,
class weighting, and two-phase transfer learning.

Usage:
    python train.py
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Default Configuration ────────────────────────────────────────────

DEFAULTS: dict = {
    "data_dir": "data",
    "img_size": 224,
    "batch_size": 32,
    "model_name": "mobilenetv2",
    "dropout": 0.4,
    "dense_units": 128,
    "unfreeze_layers": 30,
    "phase1_epochs": 15,
    "phase1_lr": 1e-4,
    "phase2_epochs": 30,
    "phase2_lr": 1e-5,
    "loss": "focal",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "use_class_weights": True,
    "early_stopping_patience": 8,
    "reduce_lr_patience": 4,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    "use_mixed_precision": False,
    "model_path": "models/visionscan_mobilenet.h5",
    "results_dir": "results",
    "augmentation": {
        "rotation_range": 30,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
        "zoom_range": 0.25,
        "brightness_range": [0.8, 1.2],
        "horizontal_flip": True,
        "vertical_flip": True,
        "fill_mode": "reflect",
    },
}


# ── Focal Loss ───────────────────────────────────────────────────────

class FocalLoss(tf.keras.losses.Loss):
    """Binary focal loss — downweights easy examples to focus on hard ones."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, self.gamma) * bce)

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}


# ── Data Loading ─────────────────────────────────────────────────────

def _load_split(split_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label, cls in enumerate(("benign", "malignant")):
        folder = split_dir / cls
        if not folder.is_dir():
            log.warning("Missing folder: %s", folder)
            continue
        for p in folder.iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size))
                X.append(img.astype(np.float32) / 255.0)
                y.append(label)
            except Exception:
                log.debug("Skipping %s", p.name)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Model Builders ───────────────────────────────────────────────────

def _head(base_output, dropout: float, units: int) -> tf.Tensor:
    x = GlobalAveragePooling2D()(base_output)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(units, activation="relu")(x)
    x = Dropout(dropout / 2)(x)
    return Dense(1, activation="sigmoid")(x)


def _build(name: str, shape: tuple, dropout: float, units: int) -> tuple[Model, tf.keras.Model]:
    builders = {
        "mobilenetv2": lambda: tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet"),
        "efficientnetv2": lambda: tf.keras.applications.EfficientNetV2S(input_shape=shape, include_top=False, weights="imagenet"),
        "convnext": lambda: tf.keras.applications.ConvNeXtTiny(input_shape=shape, include_top=False, weights="imagenet"),
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Available: {list(builders)}")
    base = builders[name]()
    base.trainable = False
    output = _head(base.output, dropout, units)
    return Model(inputs=base.input, outputs=output), base


# ── Training ─────────────────────────────────────────────────────────

def train(cfg: dict) -> tuple[Model, dict]:
    """Execute the full training pipeline."""
    size = cfg["img_size"]
    bs = cfg["batch_size"]
    model_path = Path(cfg["model_path"])
    results = Path(cfg["results_dir"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    if cfg.get("use_mixed_precision"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled")

    data = Path(cfg["data_dir"])
    log.info("Loading training data…")
    X_train, y_train = _load_split(data / "train", size)
    log.info("  Train: %s  B=%d  M=%d", X_train.shape, (y_train == 0).sum(), (y_train == 1).sum())

    log.info("Loading validation data…")
    X_val, y_val = _load_split(data / "val", size)
    log.info("  Val:   %s  B=%d  M=%d", X_val.shape, (y_val == 0).sum(), (y_val == 1).sum())

    class_weights = None
    if cfg.get("use_class_weights"):
        n = len(y_train)
        n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
        class_weights = {0: n / (2 * n0) if n0 else 1.0, 1: n / (2 * n1) if n1 else 1.0}
        log.info("Class weights: %s", class_weights)

    train_gen = ImageDataGenerator(**cfg.get("augmentation", {})).flow(X_train, y_train, batch_size=bs, shuffle=True)
    val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=bs, shuffle=False)

    name = cfg.get("model_name", "mobilenetv2")
    log.info("Building %s…", name)
    model, base = _build(name, (size, size, 3), cfg["dropout"], cfg["dense_units"])

    loss_fn = (
        FocalLoss(cfg.get("focal_alpha", 0.25), cfg.get("focal_gamma", 2.0))
        if cfg.get("loss") == "focal"
        else "binary_crossentropy"
    )

    callbacks = [
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=cfg["reduce_lr_factor"], patience=cfg["reduce_lr_patience"], min_lr=cfg["min_lr"], verbose=1),
    ]

    model.compile(optimizer=Adam(learning_rate=cfg["phase1_lr"]), loss=loss_fn, metrics=["accuracy"])
    log.info("Phase 1: Training head (%d epochs)", cfg["phase1_epochs"])
    h1 = model.fit(train_gen, epochs=cfg["phase1_epochs"], validation_data=val_gen, class_weight=class_weights, callbacks=callbacks, verbose=1)

    unfreeze = cfg.get("unfreeze_layers", 30)
    log.info("Phase 2: Fine-tuning (unfreezing last %d layers, %d epochs)", unfreeze, cfg["phase2_epochs"])
    base.trainable = True
    for layer in base.layers[:-unfreeze]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=cfg["phase2_lr"]), loss=loss_fn, metrics=["accuracy"])
    h2 = model.fit(train_gen, epochs=cfg["phase2_epochs"], validation_data=val_gen, class_weight=class_weights, callbacks=callbacks, verbose=1)

    history = {k: h1.history[k] + h2.history[k] for k in h1.history}

    model.load_weights(str(model_path))
    probs = model.predict(val_gen, verbose=1).ravel()
    preds = (probs >= 0.5).astype(int)

    acc = float(np.mean(preds == y_val))
    prec = float(precision_score(y_val, preds, zero_division=0))
    rec = float(recall_score(y_val, preds, zero_division=0))
    f1 = float(f1_score(y_val, preds, zero_division=0))

    log.info("Val Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f", acc, prec, rec, f1)
    print(classification_report(y_val, preds, target_names=["Benign", "Malignant"], zero_division=0))

    metrics = {"model": name, "val_accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1_score": round(f1, 4)}
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_cm(y_val, preds, acc, results / "confusion_matrix.png")
    _plot_curves(history, name, results / "training_curves.png")

    log.info("Training complete. Model → %s", model_path)
    return model, metrics


def _plot_cm(y_true, y_pred, acc, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title(f"Confusion Matrix — {acc:.2%}", fontweight="bold")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_curves(history: dict, name: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["accuracy"], label="Train")
    ax1.plot(history["val_accuracy"], label="Val")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(history["loss"], label="Train")
    ax2.plot(history["val_loss"], label="Val")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(f"VisionScan — {name} Training Curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionScan Global — Training")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    config_path = Path(args.config)
    if config_path.exists():
        log.info("Loading config: %s", config_path)
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    else:
        log.info("Config not found at %s — using defaults", config_path)

    train(cfg)
