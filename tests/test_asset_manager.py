"""Unit tests for Asset Manager, Checksums, and ZipSlip Protection."""

import io
import os
import tarfile
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.asset_manager.security import (
    calculate_sha256,
    check_free_disk_space,
    is_safe_path,
    safe_extract_zip,
    safe_extract_tar,
    quarantine_file,
)
from app.asset_manager.manifest import build_manifest, MasterAssetManifest
from app.asset_manager.manager import AssetManager

client = TestClient(app)
VALID_API_KEY = "fc-prod-secret-key-bank-core-9872"


def test_calculate_sha256(tmp_path: Path):
    test_f = tmp_path / "hello.txt"
    test_f.write_bytes(b"facecheck banking integrity test")
    h = calculate_sha256(test_f)
    assert len(h) == 64
    import hashlib
    assert h == hashlib.sha256(b"facecheck banking integrity test").hexdigest()


def test_check_free_disk_space(tmp_path: Path):
    has_space, free_bytes = check_free_disk_space(tmp_path, required_bytes=1024)
    assert has_space is True
    assert free_bytes > 0


def test_zipslip_traversal_rejected(tmp_path: Path):
    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()

    # Create a malicious zip file containing a traversal path (e.g. ../../evil.txt)
    zip_p = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        # On Windows and POSIX, path traversal attempts to write outside extract directory
        zf.writestr("../../evil.txt", b"malicious payload")

    with pytest.raises(Exception) as exc:
        safe_extract_zip(zip_p, dest_dir)
    assert "traversal" in str(exc.value).lower()
    # Confirm file was not written outside
    assert not (tmp_path / "evil.txt").exists()


def test_quarantine_handler(tmp_path: Path):
    bad_file = tmp_path / "corrupt_model.onnx"
    bad_file.write_bytes(b"tampered fake bytes")

    q_dir = tmp_path / "quarantine"
    q_path = quarantine_file(bad_file, q_dir, "SHA256 mismatch detected")

    assert not bad_file.exists()
    assert q_path.is_file()
    assert q_path.parent == q_dir.resolve()
    # Reason file exists
    reason_file = q_path.with_suffix(q_path.suffix + ".reason.txt")
    assert reason_file.is_file()
    assert "SHA256 mismatch" in reason_file.read_text(encoding="utf-8")


def test_master_asset_manifest_generation():
    root = Path(__file__).resolve().parent.parent
    manifest = build_manifest(root)
    assert isinstance(manifest, MasterAssetManifest)
    assert len(manifest.assets) >= 3

    # Verify YuNet and SFace entries exist
    names = {a.name for a in manifest.assets}
    assert "face_detection_yunet" in names
    assert "face_recognition_sface" in names


def test_api_models_inventory_endpoint():
    res = client.get("/api/v1/models", headers={"X-API-Key": VALID_API_KEY})
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    model_names = [m["name"] for m in data["models"]]
    assert any("YuNet" in n for n in model_names)
    assert any("SFace" in n for n in model_names)
