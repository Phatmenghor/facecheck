"""Unit tests for Input Hardening (Polyglots, Trailing Data, Decompression Bombs)."""

from __future__ import annotations

import io
import pytest
from PIL import Image

from app.services.validator import (
    validate_and_load_image,
    FileTooLargeError,
    InvalidImageError,
    UnsupportedFileTypeError,
)


def _create_clean_jpeg() -> bytes:
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_clean_png() -> bytes:
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_reject_polyglot_embedded_pdf():
    jpeg = _create_clean_jpeg()
    # Embed PDF header inside JPEG comments or trailing
    malicious = jpeg + b"\n%PDF-1.4\n1 0 obj\n<<>>\nendobj"
    with pytest.raises(InvalidImageError) as exc:
        validate_and_load_image("test.jpg", malicious)
    assert "polyglot" in str(exc.value).lower() or "trailing" in str(exc.value).lower()


def test_reject_polyglot_zip_apk():
    png = _create_clean_png()
    malicious = png + b"PK\x03\x04\x14\x00\x00\x00\x08\x00"
    with pytest.raises(InvalidImageError) as exc:
        validate_and_load_image("test.png", malicious)
    assert "polyglot" in str(exc.value).lower() or "trailing" in str(exc.value).lower()


def test_reject_jpeg_trailing_data():
    jpeg = _create_clean_jpeg()
    # Append 128 bytes of garbage after JPEG EOI (FF D9)
    tampered = jpeg + (b"A" * 128)
    with pytest.raises(InvalidImageError) as exc:
        validate_and_load_image("photo.jpg", tampered)
    assert "trailing data" in str(exc.value).lower()


def test_reject_png_trailing_data():
    png = _create_clean_png()
    tampered = png + (b"B" * 128)
    with pytest.raises(InvalidImageError) as exc:
        validate_and_load_image("graphic.png", tampered)
    assert "trailing data" in str(exc.value).lower()


def test_reject_extension_mime_mismatch():
    png = _create_clean_png()
    # Saved with .jpg extension but content is PNG
    with pytest.raises(UnsupportedFileTypeError) as exc:
        validate_and_load_image("imposter.jpg", png)
    assert "does not match" in str(exc.value).lower()


def test_reject_oversized_dimensions(monkeypatch):
    # Temporarily set max_dimension to 200 for testing
    from app.config import get_config
    cfg = get_config()
    orig = cfg.hardening.max_dimension
    cfg.hardening.max_dimension = 200

    try:
        img = Image.new("RGB", (300, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        with pytest.raises(FileTooLargeError) as exc:
            validate_and_load_image("large.jpg", raw)
        assert "dimension" in str(exc.value).lower() or "exceeds" in str(exc.value).lower()
    finally:
        cfg.hardening.max_dimension = orig
