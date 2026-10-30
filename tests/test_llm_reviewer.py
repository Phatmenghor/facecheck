"""Tests for the LLM Biometric Reviewer Service and /api/v1/review/explain Endpoint."""

from __future__ import annotations

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm_reviewer import (
    LLMReviewerService,
    BiometricTelemetryInput,
    get_llm_reviewer,
)

client = TestClient(app)
VALID_API_KEY = "fc-prod-secret-key-bank-core-9872"


def test_llm_reviewer_zero_faces():
    """Verify that zero detected faces triggers an immediate REJECT with clear customer guidance."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(faces_found=0)
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "HIGH"
    assert res.risk_score >= 90
    assert "ZERO_FACES_DETECTED" in res.flagged_anomalies
    assert "center of the frame" in res.customer_guidance
    assert "NIST SP 800-63B" in res.bank_compliance_notes


def test_llm_reviewer_multi_faces():
    """Verify that multi-face scenes trigger MANUAL_REVIEW with single-subject guidance."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=4,
        face_boxes=[[10, 10, 50, 50], [70, 70, 50, 50], [130, 130, 50, 50], [190, 190, 50, 50]],
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "MANUAL_REVIEW"
    assert res.risk_level == "MEDIUM"
    assert "MULTI_FACE_SCENE" in res.flagged_anomalies
    assert "only person" in res.customer_guidance
    assert "ISO/IEC 19794-5" in res.bank_compliance_notes


def test_llm_reviewer_blur():
    """Verify that severe blur triggers REJECT with camera-cleaning and stabilization instructions."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        quality_status="LOW_QUALITY",
        quality_issues=["Image appears blurry (Laplacian variance: 12.4, threshold: 50.0)"],
        quality_metrics={"laplacian_variance": 12.4, "mean_luminance": 120.0},
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "HIGH"
    assert "DEFOCUS_MOTION_BLUR" in res.flagged_anomalies
    assert "blurry" in res.customer_guidance.lower()
    assert "ISO/IEC 29794-5" in res.bank_compliance_notes


def test_llm_reviewer_underexposure():
    """Verify that severe darkness triggers REJECT with lighting advice."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        quality_status="LOW_QUALITY",
        quality_issues=["Image is underexposed / very dark (mean luminance: 8.5/255, threshold: 25.0)"],
        quality_metrics={"laplacian_variance": 80.0, "mean_luminance": 8.5},
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "HIGH"
    assert "SEVERE_UNDEREXPOSURE" in res.flagged_anomalies
    assert "lighting" in res.customer_guidance.lower()
    assert "ICAO Doc 9303" in res.bank_compliance_notes


def test_llm_reviewer_print_attack():
    """Verify that print attack artifacts trigger CRITICAL risk and physical photo warning."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        pad_decision="FAIL",
        pad_attack_type="PRINT_ATTACK",
        pad_risk_score=85,
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "CRITICAL"
    assert res.risk_score >= 80
    assert "PRINT_PAPER_ARTIFACT" in res.flagged_anomalies
    assert "printed pictures" in res.customer_guidance
    assert "ISO/IEC 30107" in res.bank_compliance_notes


def test_llm_reviewer_screen_replay():
    """Verify that screen replay artifacts trigger CRITICAL risk and monitor warning."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        pad_decision="REVIEW",
        pad_attack_type="SCREEN_REPLAY",
        pad_risk_score=82,
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "CRITICAL"
    assert res.risk_score >= 80
    assert "SCREEN_MOIRE_REPLAY" in res.flagged_anomalies
    assert "screens" in res.customer_guidance
    assert "ISO/IEC 30107" in res.bank_compliance_notes


def test_llm_reviewer_impostor_mismatch():
    """Verify that cross-identity mismatch triggers CRITICAL risk rejection."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        match_decision="FAIL",
        match_similarity=0.142,
        match_threshold=0.363,
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "REJECT"
    assert res.risk_level == "CRITICAL"
    assert res.risk_score >= 90
    assert "IMPOSTOR_CROSS_IDENTITY_MISMATCH" in res.flagged_anomalies
    assert "does not match" in res.customer_guidance


def test_llm_reviewer_genuine_match():
    """Verify that genuine identity match triggers ACCEPT with LOW risk."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        match_decision="PASS",
        match_similarity=0.942,
        match_threshold=0.363,
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "ACCEPT"
    assert res.risk_level == "LOW"
    assert res.risk_score <= 15
    assert "Identity confirmed" in res.summary
    assert "proceed to account activation" in res.customer_guidance


def test_llm_reviewer_clean_bona_fide():
    """Verify that a clean single-subject portrait with optimal metrics triggers ACCEPT."""
    reviewer = get_llm_reviewer()
    telemetry = BiometricTelemetryInput(
        faces_found=1,
        face_boxes=[[50, 50, 100, 100]],
        quality_status="OK",
        quality_metrics={"laplacian_variance": 140.0, "mean_luminance": 115.0},
        pad_decision="PASS",
        pad_attack_type="GENUINE",
        pad_risk_score=10,
    )
    res = reviewer.review(telemetry)

    assert res.verdict == "ACCEPT"
    assert res.risk_level == "LOW"
    assert res.risk_score <= 20
    assert "Clean bona fide" in res.summary


def test_api_review_explain_json_endpoint():
    """Verify POST /api/v1/review/explain with JSON body telemetry."""
    payload = {
        "faces_found": 1,
        "face_boxes": [[20, 20, 100, 100]],
        "quality_status": "OK",
        "quality_metrics": {"laplacian_variance": 95.0, "mean_luminance": 110.0},
        "pad_decision": "PASS",
        "pad_attack_type": "GENUINE",
        "pad_risk_score": 12,
    }
    resp = client.post(
        "/api/v1/review/explain",
        json=payload,
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "telemetry" in data
    assert "explanation" in data
    assert data["explanation"]["verdict"] == "ACCEPT"
    assert data["explanation"]["risk_level"] == "LOW"


def test_api_review_explain_image_upload_endpoint():
    """Verify POST /api/v1/review/explain with direct file upload."""
    # Create simple synthetic test image
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("test_selfie.jpg", buf, "image/jpeg")}
    resp = client.post(
        "/api/v1/review/explain",
        files=files,
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "telemetry" in data
    assert "explanation" in data
    assert "verdict" in data["explanation"]
    assert "customer_guidance" in data["explanation"]
    assert "bank_compliance_notes" in data["explanation"]


def test_learning_store_record_and_few_shot(tmp_path):
    """Verify LearningStore persistence, few-shot retrieval, and dataset export."""
    from app.services.learning_store import LearningStore, HumanReviewFeedback

    db_path = tmp_path / "test_learning.db"
    store = LearningStore(db_path=db_path)

    fb = HumanReviewFeedback(
        sample_id="SAMPLE_TEST_01",
        scenario="Single Customer KYC Account Opening",
        status="APPROVED",
        notes="High-clarity biometric selfie. Customer passed all gates.",
        telemetry={"faces_found": 1, "quality_status": "OK"},
        llm_explanation={"verdict": "ACCEPT", "risk_score": 10},
    )
    res = store.record_feedback(fb)
    assert res["status"] == "SUCCESS"
    assert res["sample_id"] == "SAMPLE_TEST_01"

    few_shots = store.get_few_shot_examples("KYC", limit=1)
    assert len(few_shots) == 1
    assert few_shots[0]["sample_id"] == "SAMPLE_TEST_01"

    export_file = tmp_path / "finetune.jsonl"
    store.export_fine_tuning_dataset(export_file)
    assert export_file.is_file()
    assert export_file.stat().st_size > 0


def test_api_feedback_and_learning_stats_endpoints():
    """Verify POST /api/v1/review/feedback and GET /api/v1/review/learning-stats endpoints."""
    payload = {
        "sample_id": "API_TEST_SAMPLE_99",
        "filename": "customer_99_selfie.jpg",
        "scenario": "Account Opening Mobile Onboarding",
        "status": "APPROVED",
        "notes": "Verified by KYC Auditor #4. Customer facial features sharp.",
        "telemetry": {"faces_found": 1, "quality_status": "OK"},
        "llm_explanation": {"verdict": "ACCEPT", "risk_score": 8},
    }
    resp = client.post(
        "/api/v1/review/feedback",
        json=payload,
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["sample_id"] == "API_TEST_SAMPLE_99"
    assert data["learning_stats"]["total_records"] >= 1

    # Test GET /api/v1/review/learning-stats
    stats_resp = client.get(
        "/api/v1/review/learning-stats",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "total_records" in stats
    assert "approved_count" in stats

    # Test POST /api/v1/review/export-training-data
    export_resp = client.post(
        "/api/v1/review/export-training-data",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["status"] == "SUCCESS"
    assert export_data["samples_count"] >= 1


def test_env_database_configuration():
    """Verify that database and security connection parameters are loaded strictly from environment."""
    from app.config import get_config
    cfg = get_config()
    assert cfg.database.engine in ("postgresql", "sqlite")
    assert cfg.database.host is not None
    assert cfg.database.port == 5432
    assert cfg.database.dbname == "llm_face"


def test_continuous_daemon_api_endpoints():
    """Verify daemon monitoring and trigger REST endpoints."""
    status_resp = client.get(
        "/api/v1/learning/daemon/status",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "daemon_id" in status_data
    assert "status" in status_data
    assert "learning_stats" in status_data
    assert "online_streamed_count" in status_data

    # Test trigger cycle
    trigger_resp = client.post(
        "/api/v1/learning/daemon/trigger",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert trigger_resp.status_code == 200
    trigger_data = trigger_resp.json()
    assert trigger_data["status"] == "SUCCESS"
    assert "heartbeat" in trigger_data

    # Test online stream ingest endpoint
    stream_resp = client.post(
        "/api/v1/learning/stream-ingest?batch_size=1",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert stream_resp.status_code == 200
    assert stream_resp.json()["status"] == "SUCCESS"


def test_auto_review_and_online_sync_endpoints():
    """Verify autonomous AI auto-review, online sync, and purge endpoints."""
    from tests.fixtures_helper import get_fixture_bytes
    img_bytes = get_fixture_bytes("valid_face_clean.jpg")

    # 1. Test POST /api/v1/learning/auto-review
    auto_resp = client.post(
        "/api/v1/learning/auto-review",
        files={"image": ("test_auto_face.jpg", img_bytes, "image/jpeg")},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert auto_resp.status_code == 200
    auto_data = auto_resp.json()
    assert auto_data["sample_id"].startswith("auto_")
    assert "verdict" in auto_data["explanation"]
    assert auto_data["auto_feedback_recorded"] is True

    # 2. Test POST /api/v1/learning/online-sync
    sync_resp = client.post(
        "/api/v1/learning/online-sync",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert "status" in sync_data
    assert "dataset_file" in sync_data

    # 3. Test POST /api/v1/learning/purge
    purge_resp = client.post(
        "/api/v1/learning/purge",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert purge_resp.status_code == 200
    purge_data = purge_resp.json()
    assert purge_data["status"] == "SUCCESS"


def test_online_training_jobs_endpoint():
    """Verify GET /api/v1/learning/online-training/jobs lists sessions and training status."""
    resp = client.get(
        "/api/v1/learning/online-training/jobs",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert "provider" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_check_endpoint_ai_message_and_feedback():
    """Verify that /api/v1/check returns AI-crafted message response and biometric explanation."""
    from tests.fixtures_helper import get_fixture_bytes
    img_bytes = get_fixture_bytes("valid_face_clean.jpg")

    resp = client.post(
        "/api/v1/check",
        files={"image": ("kyc_customer_live.jpg", img_bytes, "image/jpeg")},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ai_message" in data
    assert "ai_feedback" in data
    assert data["ai_message"] is not None
    assert "Guidance:" in data["ai_message"]
    assert data["ai_feedback"]["verdict"] in ("ACCEPT", "REJECT", "MANUAL_REVIEW")


def test_ollama_provider_configuration(monkeypatch):
    """Verify that setting provider to 'ollama' auto-configures localhost:11434 and llama3.2."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from app.services.llm_reviewer import LLMReviewerService
    service = LLMReviewerService()
    assert service.provider == "ollama"
    assert service.base_url == "http://localhost:11434/v1"
    assert service.model == "llama3.2"
    assert service.api_key == "ollama"




