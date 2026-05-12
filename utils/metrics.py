"""Advanced evaluation metrics for VisionScan Global.

Computes accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC,
and generates confusion matrix, ROC curve, precision-recall curve,
and reliability diagram.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

log = logging.getLogger(__name__)

LABELS = ["Benign", "Malignant"]


def load_split(split_dir: str | Path, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Load images from ``benign/`` and ``malignant/`` sub-folders."""
    root = Path(split_dir)
    X, y = [], []
    for label_idx, cls in enumerate(("benign", "malignant")):
        folder = root / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            X.append(img.astype(np.float32) / 255.0)
            y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate(
    model_path: str | Path,
    data_dir: str | Path,
    results_dir: str | Path = "results",
) -> dict:
    """Run a full evaluation suite and save all outputs."""
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path), compile=False)
    X, y = load_split(Path(data_dir) / "val")
    log.info("Loaded %d validation images (B=%d, M=%d)", len(X), (y == 0).sum(), (y == 1).sum())

    probs = model.predict(X, batch_size=32, verbose=1).ravel()
    preds = (probs >= 0.5).astype(int)

    acc = float(np.mean(preds == y))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    spec = specificity(y, preds)
    f1 = float(f1_score(y, preds, zero_division=0))

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc_val = float(auc(fpr, tpr))
    pr_auc_val = float(average_precision_score(y, probs))

    log.info("Acc=%.4f  Prec=%.4f  Rec=%.4f  Spec=%.4f  F1=%.4f  ROC=%.4f  PR=%.4f",
             acc, prec, rec, spec, f1, roc_auc_val, pr_auc_val)
    print(classification_report(y, preds, target_names=LABELS, zero_division=0))

    metrics = {k: round(v, 4) for k, v in {
        "val_accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1_score": f1,
        "roc_auc": roc_auc_val, "pr_auc": pr_auc_val,
    }.items()}
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_cm(y, preds, acc, results / "confusion_matrix.png")
    _plot_roc(fpr, tpr, roc_auc_val, results / "roc_curve.png")
    _plot_pr(y, probs, pr_auc_val, results / "pr_curve.png")
    _plot_reliability(y, probs, results / "reliability_diagram.png")

    return metrics


def _plot_cm(y_true, y_pred, acc, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"Confusion Matrix — Accuracy: {acc:.2%}", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc(fpr, tpr, roc_auc_val, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pr(y_true, y_prob, pr_auc_val, path: Path) -> None:
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_arr, prec_arr, color="#e74c3c", lw=2, label=f"PR (AP = {pr_auc_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_reliability(y_true, y_prob, path: Path, n_bins: int = 10) -> None:
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    observed = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        observed.append(float(y_true[mask].mean()) if mask.sum() else np.nan)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
    plt.bar(centres, observed, width=1 / n_bins, alpha=0.5, color="#3498db", edgecolor="white")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/visionscan_mobilenet.h5")
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    args = p.parse_args()
    evaluate(args.model, args.data, args.results)
