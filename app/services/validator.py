"""Image file validation and sanitization service for FaceCheck API.

Hardened against:
- Decompression bombs (Pillow pixel limits and max dimensions)
- Polyglot files (embedded PDF, ZIP, ELF, PE, HTML/JS)
- Trailing data after EOF markers (JPEG EOI, PNG IEND, WebP RIFF)
- Content-type / extension spoofing
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_config


class FileTooLargeError(Exception):
    """Raised when file exceeds size or dimension limits (HTTP 413)."""
    pass


class UnsupportedFileTypeError(Exception):
    """Raised when file content or extension is not supported (HTTP 415)."""
    pass


class InvalidImageError(Exception):
    """Raised when file is corrupted, truncated, polyglot, or contains trailing data (HTTP 400)."""
    pass


# Known polyglot binary signatures that should never appear in a raw image upload
POLYGLOT_BINARY_SIGNATURES = [
    b"%PDF-",                                      # PDF document
    b"PK\x03\x04",                                # ZIP / Office document / APK / JAR
    b"\x7fELF",                                   # Linux ELF Executable
    b"This program cannot be run in DOS mode",    # Windows PE Executable DOS stub
    b"PE\x00\x00",                                # PE header signature
]

# Known polyglot text/script signatures (case-insensitive)
POLYGLOT_TEXT_SIGNATURES = [
    b"<!doctype html",    # HTML payload
    b"<html",             # HTML payload
    b"<script",           # Embedded script
    b"<?php",             # Embedded PHP script
]


def detect_mime_from_magic_bytes(data: bytes) -> str:
    """Identify image format by inspecting file header bytes directly.

    Returns:
        MIME type string ('image/jpeg', 'image/png', 'image/webp').

    Raises:
        UnsupportedFileTypeError: If header does not match allowed image formats.
    """
    if len(data) < 12:
        raise InvalidImageError("Uploaded file is empty or truncated")

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"

    raise UnsupportedFileTypeError(
        "Unsupported file content. Allowed formats: JPG, PNG, WEBP."
    )


def check_for_polyglot_payload(data: bytes) -> None:
    """Inspect byte stream for suspicious non-image file signatures (polyglots).

    Allows legitimate camera XMP metadata (XML) inside APP1 segments while strictly
    rejecting executable binaries, ZIP/archives, embedded scripts, and disguised vector files.
    """
    # 1. Check if an XML/SVG vector document is disguised as a raster image
    trimmed_head = data[:128].strip().lower()
    if trimmed_head.startswith(b"<?xml") or trimmed_head.startswith(b"<svg"):
        raise InvalidImageError("Disguised SVG/XML vector document detected (raster formats only).")

    # 2. Check first 2KB and last 2KB for executable or script headers
    search_regions = [data[:2048], data[-2048:] if len(data) > 2048 else b""]
    combined = b" ".join(search_regions)
    combined_lower = combined.lower()

    for bsig in POLYGLOT_BINARY_SIGNATURES:
        if bsig in combined:
            raise InvalidImageError(
                f"Suspicious polyglot signature detected in file content: {bsig[:16]!r}"
            )

    for tsig in POLYGLOT_TEXT_SIGNATURES:
        if tsig in combined_lower:
            raise InvalidImageError(
                f"Suspicious polyglot signature detected in file content: {tsig[:16]!r}"
            )


def check_trailing_data(data: bytes, mime_type: str) -> None:
    """Detect unexpected trailing data appended after image end-of-file markers."""
    if mime_type == "image/jpeg":
        # JPEG must end with FF D9 (EOI marker)
        eoi_idx = data.rfind(b"\xff\xd9")
        if eoi_idx == -1:
            raise InvalidImageError("Corrupted or invalid image: End of Image (EOI) marker FF D9 not found")
        trailing = len(data) - (eoi_idx + 2)
        if trailing > 32:
            raise InvalidImageError(
                f"Suspicious trailing data detected after JPEG EOI marker ({trailing} bytes)"
            )

    elif mime_type == "image/png":
        # PNG must end with IEND chunk: length (4 bytes = 0), IEND (4 bytes), CRC (4 bytes)
        iend_sig = b"IEND\xaeB`\x82"
        iend_idx = data.rfind(iend_sig)
        if iend_idx == -1:
            # Fallback check for chunk name IEND
            iend_idx = data.rfind(b"IEND")
            if iend_idx != -1:
                iend_idx += 8  # 4 bytes name + 4 bytes CRC
            else:
                raise InvalidImageError("Invalid PNG: IEND chunk not found")
        else:
            iend_idx += len(iend_sig)

        trailing = len(data) - iend_idx
        if trailing > 32:
            raise InvalidImageError(
                f"Suspicious trailing data detected after PNG IEND chunk ({trailing} bytes)"
            )

    elif mime_type == "image/webp":
        # RIFF header: bytes 4..8 store file length minus 8
        if len(data) >= 8:
            riff_len = struct.unpack("<I", data[4:8])[0] + 8
            trailing = len(data) - riff_len
            if trailing > 64:
                raise InvalidImageError(
                    f"Suspicious trailing data detected after WebP container ({trailing} bytes)"
                )


def validate_and_load_image(
    filename: str,
    raw_bytes: bytes,
) -> Tuple[Image.Image, str, int]:
    """Execute hardened Step 1 validation.

    1. Check byte size against max_upload_size_mb (raises FileTooLargeError).
    2. Check filename extension against allowed list (raises UnsupportedFileTypeError).
    3. Check magic bytes & verify extension matches detected MIME (raises UnsupportedFileTypeError).
    4. Reject polyglot files containing executable or document signatures (raises InvalidImageError).
    5. Check trailing data after image EOF (raises InvalidImageError).
    6. Verify structural integrity with Pillow decompression limit (raises FileTooLargeError or InvalidImageError).
    7. Check pixel dimensions against max_dimension and max_image_pixels (raises FileTooLargeError).
    8. Orient and convert to RGB.

    Returns:
        Tuple of (PIL.Image in RGB, detected_mime, file_size_bytes).
    """
    config = get_config()
    max_bytes = config.storage.max_upload_size_mb * 1024 * 1024
    file_size = len(raw_bytes)

    if file_size == 0:
        raise InvalidImageError("Uploaded file is empty")

    if file_size > max_bytes:
        raise FileTooLargeError(
            f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed {config.storage.max_upload_size_mb} MB"
        )

    # Configure Pillow decompression bomb protection
    Image.MAX_IMAGE_PIXELS = config.hardening.max_image_pixels

    # Check extension
    clean_name = Path(filename).name
    ext = clean_name.rsplit(".", 1)[-1].lower() if "." in clean_name else ""
    if ext not in config.storage.allowed_extensions:
        raise UnsupportedFileTypeError(
            f"File extension '.{ext}' is not supported. Allowed: {config.storage.allowed_extensions}"
        )

    # Check magic bytes
    mime_type = detect_mime_from_magic_bytes(raw_bytes)

    # Verify extension matches magic bytes (anti-spoofing)
    ext_to_mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    expected_mime = ext_to_mime.get(ext)
    if expected_mime and expected_mime != mime_type:
        raise UnsupportedFileTypeError(
            f"File extension '.{ext}' does not match detected content format '{mime_type}'"
        )

    # Reject polyglot files if configured
    if config.hardening.reject_polyglots:
        check_for_polyglot_payload(raw_bytes)

    # Reject trailing data if configured
    if config.hardening.reject_trailing_data:
        check_trailing_data(raw_bytes, mime_type)

    # Verify structural integrity
    try:
        buffer = io.BytesIO(raw_bytes)
        with Image.open(buffer) as img_test:
            # Check dimensions before decoding full buffer
            w, h = img_test.size
            total_pixels = w * h
            if w > config.hardening.max_dimension or h > config.hardening.max_dimension:
                raise FileTooLargeError(
                    f"Image dimension ({w}x{h}) exceeds maximum allowed ({config.hardening.max_dimension}x{config.hardening.max_dimension})"
                )
            if total_pixels > config.hardening.max_image_pixels:
                raise FileTooLargeError(
                    f"Total image pixels ({total_pixels:,}) exceeds maximum allowed ({config.hardening.max_image_pixels:,})"
                )
            img_test.verify()
    except Image.DecompressionBombError as bomb_err:
        raise FileTooLargeError(f"Decompression bomb detected: {bomb_err}")
    except (FileTooLargeError, UnsupportedFileTypeError):
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as exc:
        raise InvalidImageError(f"Corrupted or invalid image file: {exc}")

    # Reopen to normalize and orient to RGB
    try:
        buffer.seek(0)
        img = Image.open(buffer)
        img = ImageOps.exif_transpose(img)
        rgb_image = img.convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Failed to decode image data into RGB: {exc}")

    return rgb_image, mime_type, file_size
