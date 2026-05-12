"""Batch inference for VisionScan Global.

Processes a folder of images, records predictions to CSV,
and generates summary visualisations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from utils.engine import Prediction, preprocess, predict

log = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run(
    model: tf.keras.Model,
    folder: str | Path,
    output_dir: str | Path = "results/batch_testing",
    *,
    temperature: float = 1.0,
) -> pd.DataFrame | None:
    """Run predictions on every valid image in *folder*.

    Returns a DataFrame of results or ``None`` if no images were found.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for img_path in sorted(folder.rglob("*")):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            _, batch = preprocess(img_path)
            pred = predict(model, batch, temperature=temperature)
            rows.append({
                "Filename": img_path.name,
                "Prediction": pred.label,
                "Confidence": pred.confidence,
                "Malignant_Prob": pred.prob_malignant,
                "Risk": pred.risk.value,
                "Certainty": pred.certainty.value,
            })
        except Exception:
            log.warning("Skipping unreadable image: %s", img_path.name)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "batch_results.csv", index=False)
    _plot_distribution(df, output_dir)
    _plot_risk(df, output_dir)
    _plot_confidence(df, output_dir)
    log.info("Batch results saved to %s", output_dir)
    return df


def _plot_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["Prediction"].value_counts()
    colours = ["#27ae60" if c == "Benign" else "#e74c3c" for c in counts.index]
    plt.figure(figsize=(7, 5))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colours, startangle=90)
    plt.title("Prediction Distribution")
    plt.savefig(out / "prediction_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_risk(df: pd.DataFrame, out: Path) -> None:
    order = ["Low Risk", "Moderate Risk", "High Risk", "Inconclusive"]
    palette = {"Low Risk": "#27ae60", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c", "Inconclusive": "#8e44ad"}
    counts = df["Risk"].value_counts().reindex(order).fillna(0)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette=[palette.get(r, "#ccc") for r in counts.index])
    plt.title("Risk Level Distribution")
    plt.ylabel("Count")
    plt.savefig(out / "risk_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_confidence(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    for label, colour in [("Benign", "#27ae60"), ("Malignant", "#e74c3c")]:
        subset = df[df["Prediction"] == label]["Confidence"]
        if not subset.empty:
            plt.hist(subset, bins=20, alpha=0.6, color=colour, label=label, edgecolor="white")
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(out / "confidence_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
