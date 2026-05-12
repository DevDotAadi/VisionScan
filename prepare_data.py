"""VisionScan Global — Dataset Preparation & Splitting.

Organises raw downloaded ISIC images from `data/raw/` (or general datasets)
into stratified train/validation structures ready for model training.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
DATA_DIR = Path("data")
VAL_SPLIT = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Locate and load the ground-truth metadata CSV."""
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV found in {raw_dir}")

    # Prioritise train or isic named CSVs
    selected_csv = csv_files[0]
    for csv_file in csv_files:
        if "train" in csv_file.name.lower() or "isic" in csv_file.name.lower():
            selected_csv = csv_file
            break

    log.info("Loading metadata from %s", selected_csv)
    return pd.read_csv(selected_csv)


def resolve_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ground-truth label columns to standard string classes."""
    if "target" in df.columns:
        df["label"] = df["target"].map({0: "benign", 1: "malignant"})
    elif "benign_malignant" in df.columns:
        df["label"] = df["benign_malignant"].str.lower()
    elif "diagnosis" in df.columns:
        df["label"] = df["diagnosis"].apply(
            lambda x: "malignant" if str(x).lower() in ("melanoma", "malignant") else "benign"
        )
    else:
        # Fallback if there are sub-directories and we can infer labels from directories
        df["label"] = "unknown"

    return df


def split_and_copy_dataset(raw_dir: Path, data_dir: Path, max_per_class: int | None = 3000) -> None:
    """Split the dataset into stratified train and validation subdirectories."""
    try:
        df = load_metadata(raw_dir)
        df = resolve_labels(df)
    except FileNotFoundError:
        # If no CSV is found, we fall back to reading files directly from class directories in data/raw/
        log.warning("No metadata CSV found. Scanning raw image subdirectories directly...")
        rows = []
        for cls in ("benign", "malignant"):
            folder = raw_dir / cls
            if folder.is_dir():
                for p in folder.iterdir():
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        rows.append({"image_name": p.stem, "label": cls, "ext": p.suffix})
        df = pd.DataFrame(rows)

    if df.empty or "label" not in df.columns:
        raise ValueError("No valid image data or labels found to split.")

    # Remove any rows with unknown labels
    df = df[df["label"].isin(["benign", "malignant"])].copy()

    log.info("Total images found: %d (Benign: %d, Malignant: %d)",
             len(df), (df["label"] == "benign").sum(), (df["label"] == "malignant").sum())

    # Create target split directories
    for split in ("train", "val"):
        for cls in ("benign", "malignant"):
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Perform splitting for each class
    for cls in ("benign", "malignant"):
        cls_df = df[df["label"] == cls].copy()

        # Limit images per class if specified
        if max_per_class and len(cls_df) > max_per_class:
            cls_df = cls_df.sample(max_per_class, random_state=RANDOM_SEED)

        # Shuffle
        cls_df = cls_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        n_val = int(len(cls_df) * VAL_SPLIT)
        val_set = cls_df.iloc[:n_val]
        train_set = cls_df.iloc[n_val:]

        copied_train = 0
        copied_val = 0

        # Match images and copy
        for split_name, subset in [("train", train_set), ("val", val_set)]:
            dest_dir = data_dir / split_name / cls

            for _, row in subset.iterrows():
                img_name = str(row["image_name"]) if "image_name" in row else str(row["isic_id"])
                copied = False

                # Handle files inside a sub-folder like raw/benign/ or raw/train_images/ or raw/
                possible_paths = [
                    raw_dir / f"{img_name}.jpg",
                    raw_dir / cls / f"{img_name}.jpg",
                    raw_dir / "train" / f"{img_name}.jpg",
                    raw_dir / "jpeg" / f"{img_name}.jpg",
                ]

                # If explicit extension is saved in row
                if "ext" in row:
                    possible_paths.insert(0, raw_dir / cls / f"{img_name}{row['ext']}")

                for p in possible_paths:
                    for ext in ("", ".jpg", ".jpeg", ".png", ".JPG"):
                        test_path = p.with_suffix(ext) if ext else p
                        if test_path.is_file():
                            shutil.copy2(test_path, dest_dir / (img_name + test_path.suffix))
                            if split_name == "train":
                                copied_train += 1
                            else:
                                copied_val += 1
                            copied = True
                            break
                    if copied:
                        break

        log.info("  %s: train: %d, val: %d", cls.capitalize(), copied_train, copied_val)


def verify_dataset(data_dir: Path) -> None:
    """Print a summary of the organized dataset folders."""
    log.info("\nDataset structure verification:")
    total = 0
    for split in ("train", "val"):
        for cls in ("benign", "malignant"):
            path = data_dir / split / cls
            if path.is_dir():
                count = len([f for f in path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
                log.info("  %s/%s: %d images", split, cls, count)
                total += count
    log.info("  Total split images: %d", total)


def main() -> None:
    """Main execution block."""
    log.info("Starting dataset splitting pipeline…")
    split_and_copy_dataset(RAW_DIR, DATA_DIR, max_per_class=3000)
    verify_dataset(DATA_DIR)
    log.info("✅ Dataset splits prepared successfully!")


if __name__ == "__main__":
    main()
