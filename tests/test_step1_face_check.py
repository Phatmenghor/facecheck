"""Unit and integration tests for STEP 1: Pre-flight Face Check & Retake Reason Codes."""

import io
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.face_detector import FaceCrop
from app.services.quality import QualityResult
from app.services.face_check import (
    evaluate_face_check,
    estimate_head_pose,
    FaceCheckResult,
)
from app.services.pipeline import run_pipeline

API_KEY = "fc-test-key-bank-core-dev-001"
app = create_app()
client = TestClient(app)


def make_clean_frontal_face_image(size=(320, 320)) -> Image.Image:
    """Create synthetic test image with a centered frontal face."""
    im = Image.new("RGB", size, color=(240, 240, 240))
    d = ImageDraw.Draw(im)
    # Centered head
    d.ellipse([100, 80, 220, 240], fill=(220, 185, 155), outline=(180, 140, 110))
    # Eyes
    d.ellipse([130, 130, 145, 145], fill=(50, 50, 50))
    d.ellipse([175, 130, 190, 145], fill=(50, 50, 50))
    # Nose
    d.polygon([(160, 150), (155, 175), (165, 175)], fill=(190, 150, 120))
    # Mouth
    d.rectangle([140, 195, 180, 205], fill=(180, 80, 80))
    im._test_mock_face = True
    return im


def test_face_check_no_face():
    """Verify NO_FACE triggers early stop with reason code and retake message."""
    blank = Image.new("RGB", (300, 300), color=(128, 128, 128))
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 100.0, "mean_luminance": 128.0})

    result = evaluate_face_check(blank, face_crops=[], quality_metrics=q_metrics)
    assert not result.passed
    assert result.status == "FAILED"
    assert "NO_FACE" in result.reason_codes
    assert "No face detected" in result.retake_message
    assert result.faces_detected == 0


def test_face_check_multiple_faces():
    """Verify MULTIPLE_FACES triggers retake."""
    im = Image.new("RGB", (400, 300), color=(200, 200, 200))
    c1 = Image.new("RGB", (100, 100), color=(210, 180, 150))
    c2 = Image.new("RGB", (100, 100), color=(210, 180, 150))
    crops = [
        FaceCrop(crop=c1, box=[30, 50, 100, 100], confidence=0.9),
        FaceCrop(crop=c2, box=[220, 50, 100, 100], confidence=0.9),
    ]
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 100.0, "mean_luminance": 128.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert not result.passed
    assert "MULTIPLE_FACES" in result.reason_codes
    assert result.faces_detected == 2
    assert "Multiple faces detected" in result.retake_message


def test_face_check_face_too_small():
    """Verify FACE_TOO_SMALL when face bounding box is smaller than min_face_size."""
    im = Image.new("RGB", (400, 400), color=(200, 200, 200))
    c = Image.new("RGB", (30, 30), color=(210, 180, 150))
    # Box is 30x30, which is < min_face_size 48
    crops = [FaceCrop(crop=c, box=[100, 100, 30, 30], confidence=0.85)]
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 100.0, "mean_luminance": 128.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert not result.passed
    assert "FACE_TOO_SMALL" in result.reason_codes
    assert "too small" in result.retake_message


def test_face_check_face_partly_hidden_boundary_clipped():
    """Verify FACE_PARTLY_HIDDEN when face touches image boundary."""
    im = Image.new("RGB", (400, 400), color=(200, 200, 200))
    c = Image.new("RGB", (120, 120), color=(210, 180, 150))
    # Face touches top-left boundary (x=2, y=2)
    crops = [FaceCrop(crop=c, box=[2, 2, 120, 120], confidence=0.85)]
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 100.0, "mean_luminance": 128.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert not result.passed
    assert "FACE_PARTLY_HIDDEN" in result.reason_codes
    assert result.visibility["boundary_clipped"] is True


def test_face_check_blurry():
    """Verify BLURRY reason code when image fails sharpness threshold."""
    im = Image.new("RGB", (320, 320), color=(200, 200, 200))
    c = Image.new("RGB", (120, 120), color=(210, 180, 150))
    crops = [FaceCrop(crop=c, box=[100, 80, 120, 120], confidence=0.85)]
    # Low Laplacian variance
    q_metrics = QualityResult(status="LOW_QUALITY", issues=["blurry"], metrics={"laplacian_variance": 18.5, "mean_luminance": 128.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert not result.passed
    assert "BLURRY" in result.reason_codes
    assert "blurry" in result.retake_message


def test_face_check_too_dark_and_too_bright():
    """Verify TOO_DARK and TOO_BRIGHT reason codes."""
    im = Image.new("RGB", (320, 320), color=(20, 20, 20))
    c = Image.new("RGB", (120, 120), color=(30, 30, 30))
    crops = [FaceCrop(crop=c, box=[100, 80, 120, 120], confidence=0.85)]

    # Too dark
    q_dark = QualityResult(status="LOW_QUALITY", issues=[], metrics={"laplacian_variance": 80.0, "mean_luminance": 15.0})
    res_dark = evaluate_face_check(im, face_crops=crops, quality_metrics=q_dark)
    assert not res_dark.passed
    assert "TOO_DARK" in res_dark.reason_codes

    # Too bright
    q_bright = QualityResult(status="LOW_QUALITY", issues=[], metrics={"laplacian_variance": 80.0, "mean_luminance": 248.0})
    res_bright = evaluate_face_check(im, face_crops=crops, quality_metrics=q_bright)
    assert not res_bright.passed
    assert "TOO_BRIGHT" in res_bright.reason_codes


def test_face_check_head_pose_non_frontal():
    """Verify FACE_NOT_FRONTAL when head yaw exceeds tolerance."""
    im = Image.new("RGB", (320, 320), color=(200, 200, 200))
    c = Image.new("RGB", (120, 120), color=(210, 180, 150))

    # Landmarks where nose is heavily displaced to the right (profile / yaw > 25 deg)
    re = (120.0, 120.0)
    le = (180.0, 120.0)
    nose_turned = (195.0, 145.0)  # nose is way past the left eye -> extreme yaw
    rm = (130.0, 170.0)
    lm = (175.0, 170.0)
    landmarks = [re, le, nose_turned, rm, lm]

    pose = estimate_head_pose(landmarks)
    assert abs(pose["yaw"]) > 25.0

    crops = [FaceCrop(crop=c, box=[80, 60, 140, 140], confidence=0.9, landmarks=landmarks)]
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 100.0, "mean_luminance": 128.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert not result.passed
    assert "FACE_NOT_FRONTAL" in result.reason_codes
    assert "turned or tilted" in result.retake_message


def test_face_check_clean_frontal_passes():
    """Verify clean frontal face passes face check and returns passed=True."""
    im = Image.new("RGB", (320, 320), color=(200, 200, 200))
    c = Image.new("RGB", (120, 120), color=(210, 180, 150))
    # Centered frontal landmarks
    re = (130.0, 130.0)
    le = (190.0, 130.0)
    nose = (160.0, 155.0)
    rm = (138.0, 180.0)
    lm = (182.0, 180.0)
    landmarks = [re, le, nose, rm, lm]

    crops = [FaceCrop(crop=c, box=[80, 60, 150, 150], confidence=0.95, landmarks=landmarks)]
    q_metrics = QualityResult(status="OK", issues=[], metrics={"laplacian_variance": 120.0, "mean_luminance": 125.0})

    result = evaluate_face_check(im, face_crops=crops, quality_metrics=q_metrics)
    assert result.passed
    assert result.status == "PASSED"
    assert result.reason_codes == []
    assert result.retake_message is None
    assert abs(result.head_pose["yaw"]) < 25.0
    assert abs(result.head_pose["pitch"]) < 20.0
    assert abs(result.head_pose["roll"]) < 20.0


def test_pipeline_early_stop_on_face_check_failure():
    """Verify run_pipeline stops early and does not run deepfake scoring when face check fails."""
    # Blank image with no face
    im = Image.new("RGB", (200, 200), color=(100, 100, 100))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    raw_bytes = buf.getvalue()

    res = run_pipeline("test_blank.jpg", raw_bytes)
    assert res["score"] is None
    assert res["label"] == "RETAKE"
    assert "NO_FACE" in res["reason_codes"]
    assert res["face_check"]["passed"] is False
    assert res["signals"]["detector_face"]["score"] is None
    assert res["signals"]["detector_full_image"]["score"] is None


def test_api_check_endpoint_face_check_contract():
    """Verify POST /api/v1/check returns face_check and reason_codes fields."""
    im = Image.new("RGB", (200, 200), color=(100, 100, 100))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    buf.seek(0)

    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": API_KEY},
        files={"image": ("test_retake.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "face_check" in data
    assert "reason_codes" in data
    assert data["face_check"]["status"] == "FAILED"
    assert "NO_FACE" in data["reason_codes"]
    assert data["score"] is None
    assert data["label"] == "RETAKE"
