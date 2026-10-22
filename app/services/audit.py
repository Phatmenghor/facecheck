"""Tamper-evident append-only audit log service using SHA-256 cryptographic hash chains."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import socket
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from app.config import get_config

logger = logging.getLogger("facecheck.audit")

_audit_lock = Lock()
GENESIS_PREV_HASH = "0" * 64


def canonical_hash(record_data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a JSON dictionary with sorted keys."""
    serialized = json.dumps(record_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_last_record_info(log_path: Path) -> Tuple[int, str]:
    """Read the last non-empty line of the audit log to retrieve the previous hash and index."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return 0, GENESIS_PREV_HASH

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return 0, GENESIS_PREV_HASH

        last_record = json.loads(lines[-1])
        last_index = int(last_record.get("index", len(lines)))
        last_hash = str(last_record.get("record_hash", GENESIS_PREV_HASH))
        return last_index, last_hash
    except Exception as exc:
        logger.warning("Could not read previous audit record: %s. Re-anchoring to genesis.", exc)
        return 0, GENESIS_PREV_HASH


def _send_syslog_message(host: str, port: int, protocol: str, message: str) -> None:
    """Forward audit log entry to external syslog daemon via UDP or TCP."""
    try:
        msg_bytes = (message + "\n").encode("utf-8")
        if protocol.lower() == "tcp":
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect((host, port))
                sock.sendall(msg_bytes)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(2.0)
                sock.sendto(msg_bytes, (host, port))
    except Exception as exc:
        logger.warning("Failed forwarding audit record to syslog (%s:%d): %s", host, port, exc)


def write_audit_record(
    request_id: str,
    api_key_name: str,
    score: Optional[int],
    label: str,
    client_ip: Optional[str] = None,
) -> Dict[str, Any]:
    """Record an immutable, tamper-evident audit entry chained via SHA-256.

    Contains strictly compliance metadata:
    - timestamp (ISO-8601 UTC)
    - request_id (UUID)
    - api_key_name
    - score (1-100 or null)
    - label (LIKELY_REAL, UNCERTAIN, LIKELY_FAKE, NO_FACE_FOUND)
    - client_ip
    - prev_hash (SHA-256 of the prior record)
    - record_hash (SHA-256 of this entry)

    Guarantees:
    - Thread-safe serialization
    - Append-only write
    - Forwarded to syslog if configured
    """
    config = get_config()
    if not config.audit.enabled:
        return {}

    log_path = Path(config.audit.log_file)

    with _audit_lock:
        last_index, prev_hash = _get_last_record_info(log_path)
        current_index = last_index + 1

        record: Dict[str, Any] = {
            "index": current_index,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request_id": request_id,
            "api_key_name": api_key_name,
            "score": score,
            "label": label,
            "client_ip": client_ip or "unknown",
            "prev_hash": prev_hash if config.audit.use_hash_chain else GENESIS_PREV_HASH,
        }

        # Calculate cryptographic hash over all fields
        rec_hash = canonical_hash(record)
        record["record_hash"] = rec_hash

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception as exc:
            logger.error("Failed to write append-only audit log: %s", exc)

    # Optional syslog forwarding (executed outside lock)
    if config.audit.syslog.enabled:
        _send_syslog_message(
            config.audit.syslog.host,
            config.audit.syslog.port,
            config.audit.syslog.protocol,
            json.dumps(record, ensure_ascii=False),
        )

    return record
