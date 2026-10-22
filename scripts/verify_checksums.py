"""Standalone checksum verification utility."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    checksum_file = root / "models" / "CHECKSUMS.txt"

    if not checksum_file.is_file():
        print(f"ERROR: Checksum file not found at {checksum_file}")
        sys.exit(1)

    print(f"Reading checksums from: {checksum_file}")
    all_passed = True

    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            expected_sha, rel_path = parts[0].strip(), parts[1].strip()
            target_path = root / rel_path

            if not target_path.is_file():
                print(f"[MISSING] {rel_path}")
                all_passed = False
                continue

            h = hashlib.sha256()
            with open(target_path, "rb") as tf:
                while chunk := tf.read(65536):
                    h.update(chunk)
            computed_sha = h.hexdigest().lower()

            if computed_sha == expected_sha.lower():
                print(f"[OK] {rel_path}")
            else:
                print(f"[FAILED] {rel_path} (Expected: {expected_sha}, Got: {computed_sha})")
                all_passed = False

    if all_passed:
        print("\nAll model integrity checks PASSED.")
        sys.exit(0)
    else:
        print("\nOne or more model integrity checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
