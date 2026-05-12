"""VisionScan Global — Standalone Evaluation CLI Script.

Runs a full clinical validation suite on the validation split:
calculates accuracy, precision, recall, specificity, F1-score, ROC-AUC, PR-AUC,
and generates validation curve/matrix plots in `results/`.

Usage:
    python evaluate.py
    python evaluate.py --model models/visionscan_mobilenet.h5 --data data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.metrics import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionScan Global: Evaluate trained model performance on validation split."
    )
    parser.add_argument(
        "--model", "-m", default="models/visionscan_mobilenet.h5",
        help="Path to Keras model file."
    )
    parser.add_argument(
        "--data", "-d", default="data",
        help="Path to organised data folder containing train/val splits."
    )
    parser.add_argument(
        "--results", "-r", default="results",
        help="Path to results directory where plots/metrics will be saved."
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)
    results_dir = Path(args.results)

    if not model_path.is_file():
        log.error("Model file not found: %s. Please train a model first.", model_path)
        return

    if not (data_dir / "val").is_dir():
        log.error("Validation data folder not found: %s. Run prepare_data.py first.", data_dir / "val")
        return

    log.info("Starting evaluation on %s using model %s…", data_dir / "val", model_path)
    try:
        metrics = evaluate(model_path, data_dir, results_dir)
        log.info("✅ Evaluation completed. Metrics saved to %s", results_dir / "metrics.json")
    except Exception as e:
        log.error("Evaluation pipeline failed: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
