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
