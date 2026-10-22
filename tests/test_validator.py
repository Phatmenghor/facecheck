"""Unit tests for Step 1: Image validation and sanitization."""

import io
import pytest
from PIL import Image

from app.services.validator import (
    validate_and_load_image,
    detect_mime_from_magic_bytes,
    FileTooLargeError,
    UnsupportedFileTypeError,
    InvalidImageError,
)


def create_test_image_bytes(fmt: str = "JPEG", size: tuple = (100, 100)) -> bytes:
    """Helper to generate valid image bytes in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(200, 100, 50))
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_validate_valid_jpeg():
    raw = create_test_image_bytes("JPEG")
    img, mime, size = validate_and_load_image("photo.jpg", raw)
    assert mime == "image/jpeg"
    assert size == len(raw)
    assert img.size == (100, 100)
    assert img.mode == "RGB"


def test_validate_valid_png():
    raw = create_test_image_bytes("PNG")
    img, mime, size = validate_and_load_image("test.png", raw)
    assert mime == "image/png"
    assert size == len(raw)


def test_validate_valid_webp():
    raw = create_test_image_bytes("WEBP")
    img, mime, size = validate_and_load_image("sample.webp", raw)
    assert mime == "image/webp"
    assert size == len(raw)


def test_reject_unsupported_extension():
    raw = create_test_image_bytes("JPEG")
    with pytest.raises(UnsupportedFileTypeError) as exc:
        validate_and_load_image("image.gif", raw)
    assert "not supported" in str(exc.value)


def test_reject_magic_byte_mismatch():
    # File named .jpg but content is plain text
    fake_content = b"This is plain text pretending to be a JPG"
    with pytest.raises(UnsupportedFileTypeError) as exc:
        validate_and_load_image("fake.jpg", fake_content)
    assert "Unsupported file content" in str(exc.value)


def test_reject_corrupted_image():
    # Starts with JPEG magic bytes, but truncated/corrupt payload
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00badpayload"
    with pytest.raises(InvalidImageError) as exc:
        validate_and_load_image("corrupt.jpg", corrupt_bytes)
    assert "Corrupted or invalid" in str(exc.value)


def test_reject_empty_file():
    with pytest.raises(InvalidImageError):
        validate_and_load_image("empty.jpg", b"")


def test_reject_file_too_large(monkeypatch):
    from app.config import get_config
    cfg = get_config()
    # Temporarily set max size to 0.001 MB (1 KB) for testing
    monkeypatch.setattr(cfg.storage, "max_upload_size_mb", 0.001)

    raw = create_test_image_bytes("JPEG", size=(300, 300))
    with pytest.raises(FileTooLargeError) as exc:
        validate_and_load_image("large.jpg", raw)
    assert "exceeds maximum allowed" in str(exc.value)
