#!/usr/bin/env python3
"""Audit log verification utility for FaceCheck API.

Verifies the cryptographic integrity of the SHA-256 hash chain in audit.log.
Detects modified records, deleted entries, inserted entries, and sequence gaps.

Usage:
    python scripts/verify_audit_log.py [--log-file logs/audit.log]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.audit import canonical_hash, GENESIS_PREV_HASH


def verify_audit_log(log_path: Path) -> bool:
    """Verify cryptographic integrity of audit log file.

    Returns:
        True if all records are intact and cryptographically linked; False otherwise.
    """
    if not log_path.is_file():
        print(f"[ERROR] Audit log file not found at: {log_path}")
        return False

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"[INFO] Audit log file is empty: {log_path} (0 records, chain is valid)")
        return True

    print(f"============================================================")
    print(f" FACECHECK AUDIT LOG INTEGRITY VERIFIER")
    print(f" Target file: {log_path}")
    print(f" Records to verify: {len(lines)}")
    print(f"============================================================")

    prev_hash_expected = GENESIS_PREV_HASH
    errors_found = 0

    for line_num, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as err:
            print(f"[FAIL] Line {line_num}: Malformed JSON syntax: {err}")
            errors_found += 1
            break

        stored_hash = record.get("record_hash")
        stored_prev_hash = record.get("prev_hash")
        record_idx = record.get("index")

        if not stored_hash:
            print(f"[FAIL] Record #{line_num}: Missing 'record_hash' attribute.")
            errors_found += 1
            break

        # Check sequence index
        if record_idx != line_num:
            print(
                f"[FAIL] Record sequence mismatch at line {line_num}: "
                f"expected index #{line_num}, got #{record_idx}"
            )
            errors_found += 1

        # Check cryptographic chain link
        if stored_prev_hash != prev_hash_expected:
            print(
                f"[FAIL] HASH CHAIN BROKEN at record #{line_num} (ID: {record.get('request_id')}):\n"
                f"       Expected prev_hash: {prev_hash_expected}\n"
                f"       Found prev_hash:    {stored_prev_hash}"
            )
            errors_found += 1
            break

        # Re-compute hash of record payload
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        computed_hash = canonical_hash(payload)

        if computed_hash != stored_hash:
            print(
                f"[FAIL] TAMPERING DETECTED in record #{line_num} (ID: {record.get('request_id')}):\n"
                f"       Stored hash:   {stored_hash}\n"
                f"       Computed hash: {computed_hash}"
            )
            errors_found += 1
            break

        prev_hash_expected = stored_hash

    print("------------------------------------------------------------")
    if errors_found == 0:
        print(f"[PASS] AUDIT LOG VERIFICATION SUCCESSFUL")
        print(f"       All {len(lines)} records cryptographically verified from genesis.")
        print(f"       Head Hash: {prev_hash_expected}")
        return True
    else:
        print(f"[FAIL] AUDIT LOG VERIFICATION FAILED ({errors_found} errors encountered).")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify FaceCheck SHA-256 audit log integrity")
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "audit.log"),
        help="Path to audit.log file (default: logs/audit.log)",
    )
    args = parser.parse_args()

    success = verify_audit_log(Path(args.log_file))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
