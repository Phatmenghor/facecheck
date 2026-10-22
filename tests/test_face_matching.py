"""Unit and integration tests for Face Matching (SFace recognition and verification)."""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.face_matching.models import (
    get_face_recognizer,
    SFaceRecognizer,
    UnavailableFaceRecognizer,
    ModelUnavailableError,
)
from app.face_matching.matcher import compare_faces, FaceMatchResult
from app.face_matching.align import align_face_5point

client = TestClient(app)
VALID_API_KEY = "fc-prod-secret-key-bank-core-9872"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> Image.Image:
    path = FIXTURES_DIR / name
    assert path.is_file(), f"Fixture {name} not found"
    return Image.open(path).convert("RGB")


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_sface_recognizer_is_loaded():
    rec = get_face_recognizer()
    assert rec.is_loaded
    assert "SFace" in rec.model_name
    assert 0.30 <= rec.default_threshold <= 0.45


def test_sface_extract_embedding_dimension_and_norm():
    rec = get_face_recognizer()
    img = _load_fixture("valid_face_clean.jpg")
    bgr = np.array(img)[:, :, ::-1].copy()

    emb = rec.extract_embedding(bgr)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (128,)
    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_sface_cosine_similarity_computation():
    rec = get_face_recognizer()
    v1 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    # Identical vectors -> cosine similarity = 1.0
    sim_identical = rec.compute_similarity(v1, v1)
    assert pytest.approx(sim_identical, abs=1e-4) == 1.0

    # Opposite vectors -> cosine similarity = -1.0
    sim_opposite = rec.compute_similarity(v1, -v1)
    assert pytest.approx(sim_opposite, abs=1e-4) == -1.0

    # Orthogonal vectors -> cosine similarity = 0.0
    v2 = np.random.randn(128).astype(np.float32)
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    sim_ortho = rec.compute_similarity(v1, v2)
    assert pytest.approx(sim_ortho, abs=1e-4) == 0.0


def test_compare_faces_same_identity_passes():
    img_a = _load_fixture("valid_face_clean.jpg")
    # Slight crop/scale of the same image to simulate authentic repeat capture
    w, h = img_a.size
    img_b = img_a.crop((4, 4, w - 4, h - 4)).resize((w, h))

    res = compare_faces(img_a, img_b)
    assert isinstance(res, FaceMatchResult)
    assert res.matched is True
    assert res.decision == "PASS"
    assert res.status == "MATCH"
    assert res.similarity_score is not None
    assert res.similarity_score >= res.threshold
    assert res.calibrated_score is not None and res.calibrated_score >= 70


def test_compare_faces_rejects_missing_face():
    img_a = _load_fixture("valid_face_clean.jpg")
    img_blank = _load_fixture("no_face_landscape.jpg")

    res = compare_faces(img_a, img_blank)
    assert res.matched is False
    assert res.decision == "FAIL"
    assert res.status == "REJECTED"
    assert "NO_FACE_IN_IMAGE_B" in res.reason_codes


def test_compare_faces_rejects_multiple_faces():
    img_a = _load_fixture("valid_face_clean.jpg")
    img_multi = _load_fixture("multiple_faces.jpg")

    res = compare_faces(img_a, img_multi)
    assert res.matched is False
    assert res.decision == "REVIEW"
    assert res.status == "UNCERTAIN"
    assert "MULTIPLE_FACES_IN_IMAGE_B" in res.reason_codes


def test_compare_faces_unavailable_recognizer_returns_graceful_status():
    img_a = _load_fixture("valid_face_clean.jpg")
    img_b = _load_fixture("valid_face_clean.jpg")

    mock_rec = UnavailableFaceRecognizer()
    res = compare_faces(img_a, img_b, recognizer=mock_rec)
    assert res.decision == "UNAVAILABLE"
    assert res.status == "UNAVAILABLE"
    assert "MODEL_UNAVAILABLE" in res.reason_codes


def test_align_face_5point_standard_output():
    img = _load_fixture("valid_face_clean.jpg")
    bgr = np.array(img)[:, :, ::-1].copy()
    landmarks = [
        (110.0, 110.0),  # re
        (190.0, 110.0),  # le
        (150.0, 150.0),  # nose
        (120.0, 200.0),  # rm
        (180.0, 200.0),  # lm
    ]
    aligned = align_face_5point(bgr, landmarks, target_size=(112, 112))
    assert aligned.shape == (112, 112, 3)


def test_api_match_endpoint():
    img_a = _load_fixture("valid_face_clean.jpg")
    img_b = _load_fixture("valid_face_clean.jpg")

    bytes_a = _image_to_bytes(img_a)
    bytes_b = _image_to_bytes(img_b)

    res = client.post(
        "/api/v1/match",
        headers={"X-API-Key": VALID_API_KEY},
        files={
            "image_a": ("selfie1.jpg", bytes_a, "image/jpeg"),
            "image_b": ("selfie2.jpg", bytes_b, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] == "PASS"
    assert data["matched"] is True
    assert "similarity_score" in data
    assert "calibrated_score" in data
    assert "OpenCV-SFace" in data["model_version"]


def test_api_extract_embedding_endpoint():
    img = _load_fixture("valid_face_clean.jpg")
    bytes_img = _image_to_bytes(img)

    res = client.post(
        "/api/v1/extract-embedding",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("face.jpg", bytes_img, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["faces_detected"] >= 1
    assert len(data["embedding"]) == 128
    assert len(data["box"]) == 4
    assert len(data["landmarks"]) == 5
    assert "OpenCV-SFace" in data["model_version"]


def test_concurrent_face_matching_multithreaded():
    """Regression test: verify thread-safety of YuNet and SFace under concurrent threads."""
    import concurrent.futures
    img = _load_fixture("valid_face_clean.jpg")

    def worker():
        res = compare_faces(img, img)
        assert res.matched is True
        assert res.decision == "PASS"
        return res.similarity_score

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(16)]
        scores = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(scores) == 16
    for s in scores:
        assert s is not None and s >= 0.363

