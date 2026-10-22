"""Integration tests for V1 Face Detection and Quality API endpoints."""

import io
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
VALID_API_KEY = "fc-prod-secret-key-bank-core-9872"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture_bytes(name: str) -> bytes:
    p = FIXTURES_DIR / name
    assert p.is_file(), f"Fixture {name} not found"
    return p.read_bytes()


def test_api_detect_faces_clean_portrait():
    raw = _load_fixture_bytes("valid_face_clean.jpg")
    res = client.post(
        "/api/v1/detect",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("face.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["faces_found"] == 1
    face = data["faces"][0]
    assert len(face["box"]) == 4
    assert face["confidence"] > 0.50
    assert len(face["landmarks"]) == 5
    assert data["image_width"] > 0
    assert data["image_height"] > 0


def test_api_detect_faces_multiple():
    raw = _load_fixture_bytes("multiple_faces.jpg")
    res = client.post(
        "/api/v1/detect",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("multi.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["faces_found"] >= 2


def test_api_detect_faces_none():
    raw = _load_fixture_bytes("no_face_landscape.jpg")
    res = client.post(
        "/api/v1/detect",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("scenery.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["faces_found"] == 0
    assert len(data["faces"]) == 0


def test_api_quality_clean_image():
    raw = _load_fixture_bytes("valid_face_clean.jpg")
    res = client.post(
        "/api/v1/quality",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("face.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("OK", "LOW_QUALITY")
    assert "metrics" in data
    assert "laplacian_variance" in data["metrics"]


def test_api_quality_blurry_image():
    raw = _load_fixture_bytes("blurry_image.jpg")
    res = client.post(
        "/api/v1/quality",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("blur.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "BLURRY" in data["issues"] or data["status"] == "LOW_QUALITY"


def test_api_detect_unauthorized_returns_401():
    raw = _load_fixture_bytes("valid_face_clean.jpg")
    res = client.post(
        "/api/v1/detect",
        files={"image": ("face.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 401
