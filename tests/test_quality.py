"""Unit tests for Image Quality Assessment."""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.services.quality import (
    compute_laplacian_variance,
    inspect_image_quality,
)


def _make_test_image(width: int = 200, height: int = 200, color: tuple = (128, 128, 128)) -> Image.Image:
    # Create image with texture so Laplacian variance is non-zero
    arr = np.random.randint(50, 200, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_quality_ok():
    img = _make_test_image(300, 300)
    res = inspect_image_quality(img, file_size_bytes=50_000, face_boxes=[[50, 50, 100, 100]])
    assert res.status == "OK"
    assert len(res.issues) == 0
    assert res.metrics["width"] == 300
    assert res.metrics["height"] == 300


def test_quality_low_resolution():
    small_img = Image.new("RGB", (64, 64), color="blue")
    res = inspect_image_quality(small_img, file_size_bytes=2_000)
    assert res.status == "LOW_QUALITY"
    assert any("resolution too low" in issue.lower() for issue in res.issues)


def test_quality_blur_detection():
    # Plain solid image has zero edge variance -> blur
    smooth_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    res = inspect_image_quality(smooth_img, file_size_bytes=10_000)
    assert res.status == "LOW_QUALITY"
    assert any("blurry" in issue.lower() for issue in res.issues)


def test_quality_underexposed():
    dark_img = Image.new("RGB", (200, 200), color=(10, 10, 10))
    res = inspect_image_quality(dark_img, file_size_bytes=5_000)
    assert res.status == "LOW_QUALITY"
    assert any("underexposed" in issue.lower() or "dark" in issue.lower() for issue in res.issues)


def test_quality_overexposed():
    bright_img = Image.new("RGB", (200, 200), color=(250, 250, 250))
    res = inspect_image_quality(bright_img, file_size_bytes=5_000)
    assert res.status == "LOW_QUALITY"
    assert any("overexposed" in issue.lower() or "bright" in issue.lower() for issue in res.issues)


def test_quality_small_face_size():
    img = _make_test_image(300, 300)
    # Face crop is 20x20 pixels (below min 48px)
    res = inspect_image_quality(img, file_size_bytes=50_000, face_boxes=[[10, 10, 20, 20]])
    assert res.status == "LOW_QUALITY"
    assert any("smaller than" in issue.lower() for issue in res.issues)
