"""Integration tests for POST /api/v1/check and GET /api/v1/health endpoints."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)
VALID_API_KEY = "fc-prod-secret-key-bank-core-9872"


def create_image_file(fmt: str = "JPEG", size=(150, 150), has_face: bool = False):
    buf = io.BytesIO()
    if has_face:
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "valid_face_clean.jpg"
        if fixture_path.is_file():
            img = Image.open(fixture_path)
            img.save(buf, format=fmt)
            buf.seek(0)
            return buf
        ref_path = Path(__file__).resolve().parent.parent / "eval" / "data" / "real" / "portrait_01.jpg"
        if ref_path.is_file():
            img = Image.open(ref_path)
            img.save(buf, format=fmt)
            buf.seek(0)
            return buf
    img = Image.new("RGB", size, color=(140, 100, 80))
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UP"
    assert "offline_mode" in data
    assert "timestamp" in data


def test_offline_swagger_docs():
    res = client.get("/docs")
    assert res.status_code == 200
    assert "swagger-ui-bundle.js" in res.text
    assert "swagger-ui.css" in res.text

    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    schema = openapi_res.json()
    assert "ApiKeyAuth" in schema["components"]["securitySchemes"]


def test_check_without_api_key_returns_401():
    img_buf = create_image_file("JPEG")
    res = client.post(
        "/api/v1/check",
        files={"file": ("test.jpg", img_buf, "image/jpeg")},
    )
    assert res.status_code == 401
    assert res.json()["detail"]["error"] == "UNAUTHORIZED"


def test_check_with_invalid_api_key_returns_401():
    img_buf = create_image_file("JPEG")
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": "bad_key"},
        files={"file": ("test.jpg", img_buf, "image/jpeg")},
    )
    assert res.status_code == 401


def test_check_unsupported_file_extension_returns_415():
    img_buf = create_image_file("JPEG")
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("test.gif", img_buf, "image/gif")},
    )
    assert res.status_code == 415
    assert res.json()["detail"]["error"] == "UNSUPPORTED_MEDIA_TYPE"


def test_check_corrupted_image_returns_400():
    corrupt_buf = io.BytesIO(b"\xff\xd8\xff\x00truncatedcorruptbytes")
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("corrupt.jpg", corrupt_buf, "image/jpeg")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "BAD_REQUEST"


def test_check_no_face_found():
    # 50x50 solid blank image with no face
    img_buf = create_image_file("JPEG", size=(50, 50), has_face=False)
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("no_face.jpg", img_buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["score"] is None
    assert data["label"] in ("NO_FACE_FOUND", "RETAKE")
    assert "NO_FACE" in data.get("reason_codes", ["NO_FACE"])
    assert data["faces_found"] == 0
    assert data["faces"] == []
    assert data["disclaimer"] == "Risk indicator only, not final proof."


def test_check_with_face_detected():
    img_buf = create_image_file("JPEG", size=(200, 200), has_face=True)
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("advertisement_face.jpg", img_buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()

    assert "request_id" in data
    assert data["score"] is not None
    assert 1 <= data["score"] <= 100
    assert data["label"] in ["LIKELY_REAL", "UNCERTAIN", "LIKELY_FAKE"]
    assert data["faces_found"] >= 1
    assert len(data["faces"]) >= 1

    # Check quality assessment object
    assert "quality" in data
    assert data["quality"]["status"] in ["OK", "LOW_QUALITY"]
    assert isinstance(data["quality"]["issues"], list)

    # Check signal breakdown
    assert "metadata" in data["signals"]
    assert "detector_face" in data["signals"]
    assert "detector_full_image" in data["signals"]
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) > 0

    # Verify audit log was written
    from app.config import get_config
    audit_file = Path(get_config().audit.log_file)
    assert audit_file.is_file()
    last_line = audit_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert data["request_id"] in last_line
    assert "bank_core_service" in last_line


def test_check_with_multipart_field_image_succeeds():
    img_buf = create_image_file("JPEG", size=(200, 200), has_face=True)
    res = client.post(
        "/api/v1/check",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("advertisement_face.jpg", img_buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "request_id" in data
    assert data["faces_found"] >= 1
    assert data["score"] is not None


def test_check_with_external_image_url_succeeds():
    # Test checking directly via external online URL with fallback to local bytes if offline
    from unittest.mock import patch, MagicMock
    from pathlib import Path

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "valid_face_clean.jpg"
    with open(fixture_path, "rb") as f:
        fake_bytes = f.read()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        test_url = "https://bank.internal.corp/kyc_test_avatar.jpg"
        res = client.post(
            "/api/v1/check-url",
            headers={"X-API-Key": VALID_API_KEY},
            json={"image_url": test_url},
        )
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "label" in data
        assert "face_check" in data
