"""API key validation and rate limiting dependencies for FaceCheck API."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional
from fastapi import Header, HTTPException, Query, Request, status

from app.config import get_config

# Rate limiter storage: ip -> list of timestamps
_rate_limit_records: Dict[str, List[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def get_client_ip(request: Request) -> str:
    """Extract client IP address safely, respecting reverse proxy headers."""
    if hasattr(request, "scope") and "headers" in request.scope:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


def verify_api_key(
    request: Request = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> str:
    """Verify X-API-Key header or api_key query param in constant time.

    Args:
        request: FastAPI Request object.
        x_api_key: Value from X-API-Key header.
        api_key: Value from api_key query parameter (useful for browser media/images).

    Returns:
        Name of the authenticated API key.

    Raises:
        HTTPException(401): If neither header nor query param matches any configured key.
    """
    key_val = None
    if isinstance(x_api_key, str) and x_api_key.strip():
        key_val = x_api_key.strip()
    elif isinstance(api_key, str) and api_key.strip():
        key_val = api_key.strip()
    elif request is not None and hasattr(request, "query_params"):
        q_key = request.query_params.get("api_key") or request.query_params.get("key")
        if q_key:
            key_val = q_key.strip()

    if not key_val:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "Missing required 'X-API-Key' header or 'api_key' parameter."},
        )

    config = get_config()
    clean_key = key_val

    for entry in config.security.api_keys:
        # Constant-time comparison prevents timing attacks
        if hmac.compare_digest(clean_key.encode("utf-8"), entry.key.strip().encode("utf-8")):
            return entry.name

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "UNAUTHORIZED", "message": "Invalid API key."},
    )


def check_rate_limit(request: Request) -> None:
    """Sliding window thread-safe rate limiter per client IP address.

    Raises:
        HTTPException(429): If client IP exceeded allowed rate limit.
    """
    config = get_config()
    limit = config.security.rate_limit_per_minute
    if limit <= 0:
        return

    client_ip = get_client_ip(request)
    now = time.time()
    window_start = now - 60.0

    with _rate_limit_lock:
        # Periodic cleanup of stale IPs to prevent unbounded memory growth
        if len(_rate_limit_records) > 1000:
            stale_keys = [
                ip for ip, records in _rate_limit_records.items()
                if not records or records[-1] <= window_start
            ]
            for ip in stale_keys:
                _rate_limit_records.pop(ip, None)

        timestamps = [t for t in _rate_limit_records[client_ip] if t > window_start]
        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "TOO_MANY_REQUESTS", "message": f"Rate limit of {limit} requests per minute exceeded."},
            )

        timestamps.append(now)
        _rate_limit_records[client_ip] = timestamps
