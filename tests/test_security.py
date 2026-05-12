"""Unit tests for VisionScan Global — security and validation utilities."""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import pytest

from utils.security_utils import (
    secure_filename,
    sanitize_text,
    validate_image_upload,
    strip_metadata,
)


class TestSecurityUtilities:
    def test_secure_filename_strips_traversal(self):
        assert secure_filename("../../secrets.json") == "secrets.json"
        assert secure_filename("/abs/path/image.png") == "image.png"
        assert secure_filename("lesion; rm -rf *.jpg") == "lesion__rm_-rf__.jpg"
        assert secure_filename("..") == "uploaded_file.jpg"
        assert secure_filename("") == "uploaded_file.jpg"

    def test_sanitize_text_escapes_html(self):
        assert sanitize_text("<script>alert(1)</script>") == "alert(1)"
        assert sanitize_text("Hello <world> & brand") == "Hello  &amp; brand"
        assert sanitize_text("") == ""

    def test_strip_metadata_produces_clean_image(self):
        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="red")
        # Save dummy EXIF details to verify strip mechanism
        exif = img.getexif()
        exif[0x0112] = 1  # Orientation tag

        img_with_exif = io.BytesIO()
        img.save(img_with_exif, format="JPEG", exif=exif)
        img_with_exif.seek(0)

        loaded_with_exif = Image.open(img_with_exif)
        assert len(loaded_with_exif.getexif()) > 0

        # Run stripping
        clean_img = strip_metadata(loaded_with_exif)
        assert clean_img.getexif() is None or len(clean_img.getexif()) == 0

    def test_validate_image_upload_oversized(self):
        # Mock uploaded file with large byte array
        class MockUploadedFile:
            def __init__(self, name, size_bytes):
                self.name = name
                self.bytes = b"0" * size_bytes
                self.type = "image/jpeg"

            def getvalue(self):
                return self.bytes

            def seek(self, offset):
                pass

        # 11 MB upload (limit is 10 MB)
        mock_file = MockUploadedFile("test.jpg", 11 * 1024 * 1024)
        is_valid, msg, _ = validate_image_upload(mock_file, max_mb=10.0)
        assert not is_valid
        assert "exceeds the maximum limit" in msg

    def test_validate_image_upload_invalid_extension(self):
        class MockUploadedFile:
            def __init__(self, name):
                self.name = name
                self.bytes = b"0" * 100
                self.type = "text/plain"

            def getvalue(self):
                return self.bytes

        mock_file = MockUploadedFile("exploit.sh")
        is_valid, msg, _ = validate_image_upload(mock_file)
        assert not is_valid
        assert "Unsupported file extension" in msg
