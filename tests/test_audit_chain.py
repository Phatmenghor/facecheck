"""Unit tests for SHA-256 Hash Chained Audit Log."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from app.services.audit import write_audit_record, canonical_hash
from scripts.verify_audit_log import verify_audit_log


def test_audit_hash_chain_verification(tmp_path, monkeypatch):
    log_file = tmp_path / "test_audit.log"

    from app.config import get_config
    cfg = get_config()
    orig_file = cfg.audit.log_file
    cfg.audit.log_file = str(log_file)

    try:
        # Write 5 consecutive records
        for i in range(1, 6):
            rec = write_audit_record(
                request_id=f"req-{i}",
                api_key_name="test_key",
                score=50 + i * 5,
                label="UNCERTAIN",
                client_ip="127.0.0.1",
            )
            assert rec["index"] == i
            assert "record_hash" in rec
            assert "prev_hash" in rec

        # Verify integrity
        assert verify_audit_log(log_file) is True

    finally:
        cfg.audit.log_file = orig_file


def test_audit_tampering_detection(tmp_path, monkeypatch):
    log_file = tmp_path / "tampered_audit.log"

    from app.config import get_config
    cfg = get_config()
    orig_file = cfg.audit.log_file
    cfg.audit.log_file = str(log_file)

    try:
        # Write 3 records
        for i in range(1, 4):
            write_audit_record(
                request_id=f"req-{i}",
                api_key_name="test_key",
                score=20 + i,
                label="LIKELY_REAL",
                client_ip="127.0.0.1",
            )

        assert verify_audit_log(log_file) is True

        # Tamper with record #2: alter score from 22 to 99
        lines = log_file.read_text(encoding="utf-8").splitlines()
        rec2 = json.loads(lines[1])
        rec2["score"] = 99
        lines[1] = json.dumps(rec2)
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Verifier MUST detect the tampering!
        assert verify_audit_log(log_file) is False

    finally:
        cfg.audit.log_file = orig_file
