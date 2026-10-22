"""Security utilities for FaceCheck Asset Management.

Includes:
- Path traversal protection (ZipSlip guard for zip/tar archives)
- Quarantine handler for corrupt/tampered files
- Disk space verification before downloads
- SHA-256 integrity verification
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple


class SecurityError(Exception):
    """Raised when an asset operation violates path traversal or security boundaries."""
    pass


def calculate_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 checksum of a file in streaming chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def check_free_disk_space(target_dir: Path | str, required_bytes: int) -> Tuple[bool, int]:
    """Check if the target directory volume has at least `required_bytes` free space.

    Returns:
        (has_sufficient_space: bool, free_bytes: int)
    """
    p = Path(target_dir).resolve()
    # Ensure directory exists for checking
    p.mkdir(parents=True, exist_ok=True)

    free_bytes = 0
    if platform.system() == "Windows":
        free_bytes_ct = ctypes.c_ulonglong(0)
        total_bytes_ct = ctypes.c_ulonglong(0)
        total_free_bytes_ct = ctypes.c_ulonglong(0)
        drive_path = ctypes.c_wchar_p(str(p))
        if ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            drive_path,
            ctypes.byref(free_bytes_ct),
            ctypes.byref(total_bytes_ct),
            ctypes.byref(total_free_bytes_ct),
        ):
            free_bytes = free_bytes_ct.value
    else:
        stat = os.statvfs(str(p))
        free_bytes = stat.f_bavail * stat.f_frsize

    has_space = free_bytes >= required_bytes
    return has_space, free_bytes


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Verify that target_path resolves strictly inside base_dir (anti-path-traversal)."""
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        return base_resolved in target_resolved.parents or base_resolved == target_resolved
    except Exception:
        return False


def safe_extract_zip(zip_path: Path | str, extract_to: Path | str) -> None:
    """Extract a ZIP archive safely with strict ZipSlip path traversal protection."""
    zip_p = Path(zip_path).resolve()
    dest = Path(extract_to).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_p, "r") as zf:
        for member in zf.infolist():
            # Check for path traversal in zip entry name
            member_path = dest / member.filename
            if not is_safe_path(dest, member_path):
                raise SecurityError(
                    f"Path traversal attempt detected in archive entry: {member.filename!r}"
                )
            zf.extract(member, dest)


def safe_extract_tar(tar_path: Path | str, extract_to: Path | str) -> None:
    """Extract a TAR/TGZ archive safely with strict path traversal protection."""
    tar_p = Path(tar_path).resolve()
    dest = Path(extract_to).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_p, "r:*") as tf:
        for member in tf.getmembers():
            member_path = dest / member.name
            if not is_safe_path(dest, member_path):
                raise SecurityError(
                    f"Path traversal attempt detected in tar entry: {member.name!r}"
                )
        tf.extractall(dest)


def quarantine_file(file_path: Path | str, quarantine_dir: Path | str, reason: str) -> Path:
    """Move a suspicious or tampered file to the quarantine directory with reason metadata."""
    src = Path(file_path).resolve()
    q_dir = Path(quarantine_dir).resolve()
    q_dir.mkdir(parents=True, exist_ok=True)

    dest = q_dir / f"{src.name}.quarantined_{os.urandom(4).hex()}"
    shutil.move(str(src), str(dest))

    # Write quarantine notice
    notice_file = dest.with_suffix(dest.suffix + ".reason.txt")
    notice_file.write_text(
        f"Quarantine Reason: {reason}\n"
        f"Original Path: {src}\n"
        f"Timestamp: {os.stat(dest).st_ctime}\n",
        encoding="utf-8",
    )
    return dest
