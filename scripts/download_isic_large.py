"""VisionScan Global — Large-Scale ISIC Dataset Downloader.

Downloads thousands of balanced dermoscopic images from the ISIC Archive public
API using multithreading with full clinical metadata preservation.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ISIC_API = "https://api.isic-archive.com/api/v2/images/"
VALID_CATEGORIES = ["benign", "malignant"]


def _fetch_page(category: str, limit: int, offset: int = 0) -> tuple[list, str | None]:
    """Fetch a single paginated batch of image records from the ISIC API."""
    params = {
        "benign_malignant": category,
        "limit": min(limit, 100),
        "offset": offset,
    }
    r = requests.get(ISIC_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []), data.get("next")


def fetch_metadata(category: str, count: int) -> list[dict]:
    """Retrieve up to *count* metadata records from the API for a category."""
    records = []
    offset = 0

    with tqdm(total=count, desc=f"Fetching {category} metadata", unit="img") as pbar:
        while len(records) < count:
            batch_limit = min(100, count - len(records))
            results, _ = _fetch_page(category, batch_limit, offset)
            if not results:
                break
            records.extend(results)
            offset += len(results)
            pbar.update(len(results))

    return records[:count]


def download_image(item: dict, output_dir: Path, target_size: tuple[int, int]) -> dict | None:
    """Download, preprocess, and save a single skin lesion image."""
    isic_id = item.get("isic_id")
    files = item.get("files", {})

    url = files.get("thumbnail_256", {}).get("url") or files.get("full", {}).get("url")
    if not url:
        return None

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))

        save_path = output_dir / f"{isic_id}.jpg"
        img.save(save_path, format="JPEG", quality=90)

        clinical = item.get("metadata", {}).get("clinical", {})
        return {
            "isic_id": isic_id,
            "benign_malignant": item.get("metadata", {}).get("clinical", {}).get("benign_malignant", "unknown"),
            "diagnosis": clinical.get("diagnosis", "unknown"),
            "sex": clinical.get("sex", "unknown"),
            "age_approx": clinical.get("age_approx", "unknown"),
            "anatomical_site": clinical.get("anatomical_site_general", "unknown"),
            "file_path": str(save_path.relative_to(Path.cwd())),
        }
    except Exception:
        return None


def download_category(
    category: str,
    count: int,
    output_base: Path,
    target_size: tuple[int, int],
    workers: int,
) -> list[dict]:
    """Retrieve metadata and download images for a category in parallel."""
    category_dir = output_base / category
    category_dir.mkdir(parents=True, exist_ok=True)

    items = fetch_metadata(category, count)
    log.info("  Found %d %s records. Starting downloads...", len(items), category)

    metadata_list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, item, category_dir, target_size): item
            for item in items
        }
        with tqdm(total=len(futures), desc=f"Downloading {category}", unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    res["benign_malignant"] = category
                    metadata_list.append(res)
                pbar.update(1)

    return metadata_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Download large balanced ISIC datasets.")
    parser.add_argument("--total", type=int, default=10000, help="Total number of images to download.")
    parser.add_argument("--output", default="data/raw", help="Raw images output folder.")
    parser.add_argument("--size", type=int, default=224, help="Target crop size.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent downloaders.")
    args = parser.parse_args()

    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    target_size = (args.size, args.size)
    per_class = args.total // 2

    log.info("\n" + "=" * 50)
    log.info("  VisionScan Global — ISIC Large-Scale Downloader")
    log.info("  Target: %d images (%d per class)", args.total, per_class)
    log.info("  Output: %s", out_base)
    log.info("  Size:   %d×%d", args.size, args.size)
    log.info("  Workers: %d", args.workers)
    log.info("=" * 50 + "\n")

    all_meta = []
    for category in VALID_CATEGORIES:
        meta = download_category(category, per_class, out_base, target_size, args.workers)
        all_meta.extend(meta)
        log.info("  ✅ %s: %d/%d downloaded successfully\n", category.capitalize(), len(meta), per_class)

    if all_meta:
        csv_path = out_base / "isic_metadata.csv"
        pd.DataFrame(all_meta).to_csv(csv_path, index=False)
        log.info("=" * 50)
        log.info("  Download Complete!")
        log.info("  Total images: %d", len(all_meta))
        log.info("  Metadata:     %s", csv_path)
        log.info("=" * 50)
    else:
        log.error("❌ No images downloaded. Check network connection.")


if __name__ == "__main__":
    main()
