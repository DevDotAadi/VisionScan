"""VisionScan Global — Standalone ISIC Sample Downloader.

Downloads a small set of benign and malignant sample images from the public
ISIC Archive API v2, resizes them preserving aspect ratio, and saves metadata.
"""

from __future__ import annotations

import argparse
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ISIC_API_URL = "https://api.isic-archive.com/api/v2/images/"


def download_category(category: str, count: int, output_dir: Path) -> list[dict]:
    """Fetch images of a specific class from the ISIC API and save locally."""
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching metadata for %d %s images...", count, category)
    params = {"benign_malignant": category, "limit": count}

    try:
        r = requests.get(ISIC_API_URL, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        log.error("Failed to fetch ISIC metadata for %s: %s", category, e)
        return []

    metadata = []
    log.info("Downloading images…")
    for item in tqdm(results, desc=category.capitalize(), unit="img"):
        isic_id = item.get("isic_id")
        files = item.get("files", {})

        url = files.get("thumbnail_256", {}).get("url") or files.get("full", {}).get("url")
        if not url:
            continue

        try:
            img_resp = requests.get(url, timeout=10)
            img_resp.raise_for_status()

            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            # Pad & resize to 224x224
            img = ImageOps.pad(img, (224, 224), method=Image.Resampling.LANCZOS, color=(0, 0, 0))

            save_path = category_dir / f"{isic_id}.jpg"
            img.save(save_path, format="JPEG", quality=90)

            clinical = item.get("metadata", {}).get("clinical", {})
            metadata.append({
                "isic_id": isic_id,
                "benign_malignant": category,
                "diagnosis": clinical.get("diagnosis", "unknown"),
                "sex": clinical.get("sex", "unknown"),
                "age_approx": clinical.get("age_approx", "unknown"),
                "anatomical_site": clinical.get("anatomical_site_general", "unknown"),
                "file_path": str(save_path.relative_to(Path.cwd())),
            })
        except Exception as e:
            log.warning("Skipping %s due to error: %s", isic_id, e)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Download skin lesion samples from the ISIC Archive.")
    parser.add_argument("--benign", type=int, default=100, help="Number of benign images.")
    parser.add_argument("--malignant", type=int, default=100, help="Number of malignant images.")
    parser.add_argument("--output", default="test_images", help="Output directory path.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    if args.benign > 0:
        all_meta.extend(download_category("benign", args.benign, out_dir))
    if args.malignant > 0:
        all_meta.extend(download_category("malignant", args.malignant, out_dir))

    if all_meta:
        csv_path = out_dir / "isic_metadata.csv"
        pd.DataFrame(all_meta).to_csv(csv_path, index=False)
        log.info("✅ Saved metadata for %d images to %s", len(all_meta), csv_path)
    else:
        log.warning("❌ No images were successfully downloaded.")


if __name__ == "__main__":
    main()
