"""Unit tests for API key verification and rate limiting."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.dependencies import verify_api_key, check_rate_limit, _rate_limit_records


def test_verify_api_key_valid():
    # bank_core_service configured key
    valid_key = "fc-prod-secret-key-bank-core-9872"
    key_name = verify_api_key(x_api_key=valid_key)
    assert key_name == "bank_core_service"


def test_verify_api_key_query_param():
    valid_key = "fc-prod-secret-key-bank-core-9872"
    key_name = verify_api_key(api_key=valid_key)
    assert key_name == "bank_core_service"


def test_verify_api_key_missing():
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "UNAUTHORIZED"


def test_verify_api_key_invalid():
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="wrong_invalid_key_12345")
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "UNAUTHORIZED"


def test_rate_limiter_blocks_excessive_requests(monkeypatch):
    from app.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.security, "rate_limit_per_minute", 3)

    # Mock request from IP 192.168.1.100
    _rate_limit_records.clear()
    scope = {"type": "http", "client": ("192.168.1.100", 50000)}
    req = Request(scope)

    # First 3 requests succeed
    check_rate_limit(req)
    check_rate_limit(req)
    check_rate_limit(req)

    # 4th request must be rejected with 429
    with pytest.raises(HTTPException) as exc:
        check_rate_limit(req)
    assert exc.value.status_code == 429
    assert exc.value.detail["error"] == "TOO_MANY_REQUESTS"
