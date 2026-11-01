"""Unit tests for gender estimation in FaceCheck API."""

import io
from PIL import Image
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.gender_detector import estimate_gender

API_KEY = "fc-test-key-bank-core-dev-001"


def test_estimate_gender_fallback_on_tiny_image():
    tiny = Image.new("RGB", (10, 10), color=(100, 100, 100))
    gender, conf = estimate_gender(tiny)
    assert gender == "UNKNOWN"
    assert conf == 0.0


def test_estimate_gender_synthetic_male_portrait():
    # Synthetic male base image
    base = Image.new("RGB", (128, 128), color=(210, 175, 145))
    gender, conf = estimate_gender(base)
    assert gender in ("MALE", "FEMALE")
    assert 0.0 <= conf <= 1.0


def test_api_check_returns_gender_field():
    app = create_app()
    client = TestClient(app)

    # Simple valid image
    img = Image.new("RGB", (320, 320), color=(220, 190, 160))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    buf.seek(0)

    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": API_KEY},
        files={"file": ("test_gender.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "gender" in data
    assert "gender_confidence" in data
    # When no face is found on a blank square, gender is None
    if data["faces_found"] == 0:
        assert data["gender"] is None
