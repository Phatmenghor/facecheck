"""Secure asset downloader for FaceCheck API.

Hardened against:
- Incomplete downloads (streaming to .part file + SHA-256 verification)
- Disk exhaustion (pre-allocates / verifies free disk space)
- Connection drops (bounded retries with exponential backoff)
- Download timeouts (configurable per request)
- Tampered downloads (quarantines checksum mismatches)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from app.asset_manager.security import (
    calculate_sha256,
    check_free_disk_space,
    quarantine_file,
)

logger = logging.getLogger("facecheck.asset_manager.downloader")


class DownloadError(Exception):
    """Raised when an asset download fails or is rejected."""
    pass


def download_asset(
    url: str,
    dest_path: Path | str,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    timeout_seconds: int = 30,
    max_retries: int = 3,
    chunk_size: int = 65536,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download an asset from an approved official mirror with security checks.

    Args:
        url: Approved download URL.
        dest_path: Target file path on disk.
        expected_sha256: Optional expected SHA-256 checksum for verification.
        expected_size: Optional expected byte count for pre-flight disk check.
        timeout_seconds: Timeout per connection attempt.
        max_retries: Number of bounded retry attempts.
        chunk_size: Stream buffer size.
        progress_callback: Callback receiving (bytes_read, total_bytes).

    Returns:
        Verified Path to the downloaded file.
    """
    dest = Path(dest_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    # 1. Pre-flight disk space check
    required_space = (expected_size or 50 * 1024 * 1024) * 2  # 2x buffer
    has_space, free_bytes = check_free_disk_space(dest.parent, required_space)
    if not has_space:
        raise DownloadError(
            f"Insufficient disk space in {dest.parent}. Required: {required_space / (1024*1024):.1f} MB, "
            f"Available: {free_bytes / (1024*1024):.1f} MB"
        )

    headers = {
        "User-Agent": "FaceCheck-AssetManager/1.0 (Bank Verification Infrastructure)",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading asset from %s (attempt %d/%d)...", url, attempt, max_retries)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                if resp.status != 200:
                    raise DownloadError(f"Server returned HTTP {resp.status}")

                total_size = int(resp.headers.get("Content-Length", expected_size or 0))
                bytes_downloaded = 0

                with open(part_path, "wb") as f_out:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(bytes_downloaded, total_size)

            # Successfully written part file
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            if part_path.is_file():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(attempt * 2)
            else:
                raise DownloadError(f"Failed to download asset after {max_retries} attempts: {last_error}")

    # 2. Verify SHA-256 Checksum
    actual_sha = calculate_sha256(part_path)
    if expected_sha256 and expected_sha256.lower() != "n/a":
        if actual_sha.lower() != expected_sha256.lower():
            # Tampering or corruption detected: quarantine part file
            q_dir = dest.parent / "quarantine"
            q_path = quarantine_file(part_path, q_dir, f"Checksum mismatch: expected {expected_sha256}, got {actual_sha}")
            raise DownloadError(
                f"Asset integrity check failed! Expected {expected_sha256}, got {actual_sha}. "
                f"Quarantined to {q_path}"
            )

    # Move part file to final destination
    if dest.is_file():
        dest.unlink()
    shutil.move(str(part_path), str(dest))
    logger.info("Asset successfully downloaded and verified: %s (SHA256: %s)", dest, actual_sha)
    return dest
