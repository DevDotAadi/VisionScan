"""Security, sanitisation, and input validation utilities for VisionScan Global.

Protects against common security vectors:
- Path traversal
- Suspicious uploads
- Oversized or corrupt image files
- Metadata/Privacy leakage
- Cross-Site Scripting (XSS) / Unsafe HTML injection
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from PIL import Image

# Secure allowed list
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def secure_filename(filename: str) -> str:
    """Sanitise and normalize a filename to prevent path traversal or shell exploits."""
    base = Path(filename).name
    # Strip any characters except alphanumeric, underscore, dot, and hyphen
    sanitised = re.sub(r"[^\w\.\-]", "_", base)
    if not sanitised or sanitised in (".", ".."):
        sanitised = "uploaded_file.jpg"
    return sanitised


def sanitize_text(text: str) -> str:
    """Escape HTML content and strip tags to prevent XSS injection attacks."""
    if not text:
        return ""
    # Strip HTML tags first
    stripped = re.sub(r"<[^>]*>", "", str(text))
    # Escape characters like & < > " '
    return html.escape(stripped)


def validate_image_upload(
    uploaded_file,
    max_mb: float = 10.0,
) -> tuple[bool, str, Image.Image | None]:
    """Validate upload file bounds, MIME types, and verify image decodes correctly."""
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_mb:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds the maximum limit of {max_mb} MB.", None

    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Only JPG, JPEG, and PNG are allowed.", None

    if hasattr(uploaded_file, "type"):
        mime = uploaded_file.type.lower()
        if mime not in ALLOWED_MIME_TYPES:
            return False, f"Unsupported MIME type '{mime}'. Upload must be a valid JPEG or PNG image.", None

    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()

        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.load()

        return True, "Success", img

    except Exception as e:
        return False, f"Invalid or corrupted image file structure: {e}", None


def strip_metadata(img: Image.Image) -> Image.Image:
    """Return a copy of the image with all metadata (EXIF, GPS, etc.) removed to protect patient privacy."""
    # Create a new RGB or RGBA canvas and paste the original pixels to drop binary segment blocks (APP1, etc)
    clean_img = Image.new(img.mode, img.size)
    clean_img.paste(img)
    return clean_img
