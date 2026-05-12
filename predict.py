"""VisionScan Global — Standalone CLI Inference Script.

Predict risk assessment for a single skin lesion image.
Uses calibrated prediction and generates diagnostic visualisations.

Usage:
    python predict.py --image path/to/lesion.jpg
    python predict.py --image path/to/lesion.jpg --save results/prediction.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.engine import (
    Prediction,
    RiskLevel,
    load_model,
    predict,
    preprocess,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = ["Benign", "Malignant"]
COLORS = {"Benign": "#27ae60", "Malignant": "#e74c3c"}


def _visualise(
    img_rgb: np.ndarray,
    pred: Prediction,
    save_path: Path | None = None,
) -> None:
    """Generate and optionally save an annotated result visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: original image with title
    axes[0].imshow(img_rgb)
    title_colour = COLORS["Malignant"] if pred.label == "Malignant" else COLORS["Benign"]
    if pred.risk is RiskLevel.INCONCLUSIVE:
        title_colour = "#8e44ad"

    axes[0].set_title(
        f"Prediction: {pred.label}\nConfidence: {pred.confidence:.1%}\nRisk: {pred.risk.value}",
        fontsize=12, fontweight="bold", color=title_colour,
    )
    axes[0].axis("off")

    # Right: class probabilities horizontal bar chart
    probs = [1.0 - pred.prob_malignant, pred.prob_malignant]
    colors = [COLORS["Benign"], COLORS["Malignant"]]
    bars = axes[1].barh(CLASS_NAMES, probs, color=colors, edgecolor="white", height=0.4)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Probability", fontsize=11)
    axes[1].set_title("Calibrated Class Probabilities", fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    for bar, prob in zip(bars, probs):
        axes[1].text(
            prob + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", fontsize=10, fontweight="bold",
        )

    plt.suptitle("VisionScan Global — Skin Lesion Risk Assessment", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Visualization saved to %s", save_path)

    plt.close()


def run_prediction(
    image_path: str | Path,
    model_path: str | Path,
    save_path: str | Path | None = None,
    no_plot: bool = True,
) -> Prediction:
    """Load model, run prediction, display assessment, and save plot."""
    img_path = Path(image_path)
    model = load_model(Path(model_path))

    # Preprocess
    img_display, batch = preprocess(img_path)

    # Predict
    pred = predict(model, batch)

    # Console Output
    print("\n" + "=" * 50)
    print("         VisionScan Global Prediction Result")
    print("" + "=" * 50)
    print(f"  Image            : {img_path.name}")
    print(f"  Prediction       : {pred.label}")
    print(f"  Confidence       : {pred.confidence:.2%}")
    print(f"  Risk Category    : {pred.risk.value}")
    print(f"  Certainty        : {pred.certainty.value}")
    print(f"  Malignant Prob   : {pred.prob_malignant:.4f}")
    print("-" * 50)
    print(f"  Recommendation   : {pred.recommendation}")
    print("=" * 50 + "\n")

    # Save visualization if requested or plotting is enabled
    sp = Path(save_path) if save_path else None
    if not no_plot or sp:
        _visualise(img_display, pred, save_path=sp)

    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionScan Global: Standalone CLI Skin Lesion Prediction."
    )
    parser.add_argument("--image", "-i", required=True, help="Path to lesion image (JPG/PNG).")
    parser.add_argument("--model", "-m", default="models/visionscan_mobilenet.h5", help="Path to Keras model file.")
    parser.add_argument("--save", "-s", default=None, help="Save path for prediction visualization.")
    parser.add_argument("--no-plot", action="store_true", default=True, help="Headless execution (default).")

    args = parser.parse_args()
    try:
        run_prediction(args.image, args.model, args.save, args.no_plot)
    except Exception as e:
        log.error("Prediction failed: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
