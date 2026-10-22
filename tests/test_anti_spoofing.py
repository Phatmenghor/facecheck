"""Unit and integration tests for Presentation-Attack Detection (PAD) and Interactive Liveness."""

import io
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.anti_spoofing.pad_service import evaluate_pad, PADResult
from app.anti_spoofing.texture_detector import detect_print_attack
from app.anti_spoofing.replay_detector import detect_screen_replay
from app.anti_spoofing.challenge import get_challenge_manager

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


def test_detect_print_attack_on_simulated_print():
    img_print = _load_fixture("print_attack.jpg")
    res = detect_print_attack(img_print)
    assert res["score"] >= 0.40
    assert "details" in res


def test_detect_print_attack_on_clean_portrait():
    img_clean = _load_fixture("valid_face_clean.jpg")
    res = detect_print_attack(img_clean)
    assert res["is_attack"] is False
    assert res["score"] < 0.45


def test_detect_screen_replay_on_moire_fixture():
    img_screen = _load_fixture("screen_replay.jpg")
    res = detect_screen_replay(img_screen)
    assert res["score"] >= 0.40
    assert "details" in res


def test_detect_screen_replay_on_clean_portrait():
    img_clean = _load_fixture("valid_face_clean.jpg")
    res = detect_screen_replay(img_clean)
    assert res["is_attack"] is False
    assert res["score"] < 0.45


def test_evaluate_pad_service_structure():
    img_clean = _load_fixture("valid_face_clean.jpg")
    pad_res = evaluate_pad(img_clean)
    assert isinstance(pad_res, PADResult)
    assert pad_res.decision in ("PASS", "REVIEW", "FAIL")
    assert 0.0 <= pad_res.spoof_probability <= 1.0
    assert 1 <= pad_res.calibrated_risk <= 100
    assert "print_attack" in pad_res.signals
    assert "screen_replay" in pad_res.signals
    assert "digital_manipulation" in pad_res.signals
    assert "static" in pad_res.disclaimer.lower()


def test_api_pad_endpoint():
    img_clean = _load_fixture("valid_face_clean.jpg")
    raw = _image_to_bytes(img_clean)

    res = client.post(
        "/api/v1/pad",
        headers={"X-API-Key": VALID_API_KEY},
        files={"image": ("selfie.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "decision" in data
    assert "attack_type" in data
    assert "spoof_probability" in data
    assert "signals" in data
    assert "disclaimer" in data


def test_challenge_manager_lifecycle():
    cm = get_challenge_manager()
    session = cm.create_session(sequence_length=3)
    assert session.session_id.startswith("live_")
    assert len(session.challenge_sequence) == 3
    assert session.challenge_sequence[0] == "LOOK_STRAIGHT"

    # 1. Matching trajectory
    # Simulate head poses matching sequence
    seq = session.challenge_sequence
    pose_map = {
        "LOOK_STRAIGHT": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "TURN_LEFT": {"yaw": 24.0, "pitch": 0.0, "roll": 0.0},
        "TURN_RIGHT": {"yaw": -24.0, "pitch": 0.0, "roll": 0.0},
        "TILT_UP": {"yaw": 0.0, "pitch": -20.0, "roll": 0.0},
        "TILT_DOWN": {"yaw": 0.0, "pitch": 20.0, "roll": 0.0},
    }
    frame_poses = [pose_map[action] for action in seq]

    res = cm.verify_poses(session.session_id, frame_poses)
    assert res.passed is True
    assert res.status == "PASSED"

    # 2. Replay attack rejection: same session cannot be reused
    res_replay = cm.verify_poses(session.session_id, frame_poses)
    assert res_replay.passed is False
    assert res_replay.status == "FAILED"
    assert "SESSION_ALREADY_USED" in res_replay.reason_codes


def test_challenge_manager_mismatched_sequence_fails():
    cm = get_challenge_manager()
    session = cm.create_session(sequence_length=3)
    # Provide completely wrong poses (all looking down when looking straight or turn was required)
    mismatched_poses = [
        {"yaw": 0.0, "pitch": 30.0, "roll": 0.0},
        {"yaw": 0.0, "pitch": 30.0, "roll": 0.0},
    ]
    res = cm.verify_poses(session.session_id, mismatched_poses)
    assert res.passed is False
    assert res.status == "FAILED"


def test_api_challenge_endpoints():
    # 1. Create session
    create_res = client.post(
        "/api/v1/liveness/challenge/session",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert create_res.status_code == 200
    sess_data = create_res.json()
    sess_id = sess_data["session_id"]
    seq = sess_data["challenge_sequence"]

    pose_map = {
        "LOOK_STRAIGHT": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "TURN_LEFT": {"yaw": 25.0, "pitch": 0.0, "roll": 0.0},
        "TURN_RIGHT": {"yaw": -25.0, "pitch": 0.0, "roll": 0.0},
        "TILT_UP": {"yaw": 0.0, "pitch": -22.0, "roll": 0.0},
        "TILT_DOWN": {"yaw": 0.0, "pitch": 22.0, "roll": 0.0},
    }
    poses = [pose_map[s] for s in seq]

    # 2. Verify poses
    verify_res = client.post(
        "/api/v1/liveness/challenge/verify",
        headers={"X-API-Key": VALID_API_KEY},
        json={"session_id": sess_id, "frame_poses": poses},
    )
    assert verify_res.status_code == 200
    verify_data = verify_res.json()
    assert verify_data["passed"] is True
    assert verify_data["status"] == "PASSED"
