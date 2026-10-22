"""Unit tests for Prometheus Metrics Collection and Endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.metrics import get_metrics


def test_metrics_collector_recording():
    metrics = get_metrics()

    metrics.record_request(status_code=200, api_key="test_service", duration_sec=0.12)
    metrics.record_score(score=25)
    metrics.record_score(score=55)
    metrics.record_score(score=88)
    metrics.record_low_quality()
    metrics.record_error("TEST_ERROR")

    text = metrics.export_metrics_text()
    assert "facecheck_requests_total" in text
    assert 'api_key="test_service"' in text
    assert 'range="1-30"' in text
    assert 'range="31-70"' in text
    assert 'range="71-100"' in text
    assert "facecheck_low_quality_total" in text
    assert "facecheck_processing_duration_seconds_bucket" in text


def test_metrics_endpoint_authorized():
    client = TestClient(app)
    # Valid API key from config.yaml
    headers = {"X-API-Key": "fc-prod-secret-key-bank-core-9872"}
    response = client.get("/api/v1/metrics", headers=headers)
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "facecheck_requests_total" in response.text


def test_metrics_endpoint_unauthorized():
    client = TestClient(app)
    # Invalid API key
    headers = {"X-API-Key": "invalid-key"}
    response = client.get("/api/v1/metrics", headers=headers)
    assert response.status_code == 401
