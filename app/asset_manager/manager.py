"""High-level asset manager for FaceCheck models and datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.asset_manager.manifest import (
    APPROVED_ASSET_REGISTRY,
    AssetEntry,
    MasterAssetManifest,
    build_manifest,
)
from app.asset_manager.downloader import download_asset
from app.asset_manager.security import calculate_sha256

logger = logging.getLogger("facecheck.asset_manager")


class AssetManager:
    """Manages models, datasets, manifests, checksums, and quarantine."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = (project_root or Path(__file__).resolve().parent.parent.parent).resolve()
        self.assets_dir = self.project_root / "assets"
        self.models_dir = self.project_root / "models"
        self.manifests_dir = self.assets_dir / "manifests"
        self.checksums_dir = self.assets_dir / "checksums"
        self.quarantine_dir = self.assets_dir / "quarantine"
        self.reports_dir = self.assets_dir / "reports"

        # Ensure directory layout
        for d in [self.manifests_dir, self.checksums_dir, self.quarantine_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def generate_manifest(self) -> MasterAssetManifest:
        """Generate and save master manifest."""
        manifest = build_manifest(self.project_root)
        out_p = self.manifests_dir / "manifest.json"
        out_p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def verify_all_assets(self) -> Dict[str, Any]:
        """Verify presence and checksum of all registered assets."""
        manifest = self.generate_manifest()
        summary = {
            "total_registered": len(manifest.assets),
            "verified": sum(1 for a in manifest.assets if a.status == "VERIFIED"),
            "missing": sum(1 for a in manifest.assets if a.status == "MISSING"),
            "tampered": sum(1 for a in manifest.assets if a.status == "TAMPERED"),
            "assets": [a.model_dump() for a in manifest.assets],
        }
        return summary

    def download_model(self, model_name: str) -> Path:
        """Download an approved model from registered official mirror."""
        if model_name not in APPROVED_ASSET_REGISTRY:
            raise ValueError(f"Unknown asset: {model_name}. Allowed: {list(APPROVED_ASSET_REGISTRY.keys())}")

        meta = APPROVED_ASSET_REGISTRY[model_name]
        url = meta.get("source_url")
        if not url or url.startswith("local"):
            raise ValueError(f"Asset {model_name} does not have an external download URL.")

        target = self.project_root / meta["file_path"]
        expected_sha = meta.get("sha256")
        expected_sz = meta.get("size_bytes")

        def progress(bytes_read: int, total_bytes: int):
            if total_bytes > 0:
                pct = (bytes_read / total_bytes) * 100
                mb_read = bytes_read / (1024 * 1024)
                mb_total = total_bytes / (1024 * 1024)
                print(f"\r  Downloading {model_name}: {mb_read:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)

        res_path = download_asset(
            url=url,
            dest_path=target,
            expected_sha256=expected_sha,
            expected_size=expected_sz,
            progress_callback=progress,
        )
        print()  # newline
        return res_path

    def smoke_test_model(self, model_path: Path) -> Dict[str, Any]:
        """Run smoke test on a model file."""
        if not model_path.is_file():
            return {"status": "FAIL", "error": "Model file not found"}

        name = model_path.name
        if "yunet" in name.lower():
            try:
                import cv2
                det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320))
                return {"status": "PASS", "model_type": "YuNet Face Detector", "ready": det is not None}
            except Exception as exc:
                return {"status": "FAIL", "error": str(exc)}
        elif "sface" in name.lower():
            try:
                import cv2
                rec = cv2.FaceRecognizerSF.create(str(model_path), "")
                return {"status": "PASS", "model_type": "SFace Face Recognizer", "ready": rec is not None}
            except Exception as exc:
                return {"status": "FAIL", "error": str(exc)}
        elif "safetensors" in name.lower():
            try:
                from app.detectors.safetensors_detector import SafetensorsClassifierDetector
                det = SafetensorsClassifierDetector(model_path)
                return {"status": "PASS", "model_type": "Safetensors Deepfake Classifier", "ready": det.is_ready}
            except Exception as exc:
                return {"status": "FAIL", "error": str(exc)}

        return {"status": "UNKNOWN", "message": "No specific smoke test for model type"}


_default_asset_manager: Optional[AssetManager] = None


def get_asset_manager() -> AssetManager:
    global _default_asset_manager
    if _default_asset_manager is None:
        _default_asset_manager = AssetManager()
    return _default_asset_manager
