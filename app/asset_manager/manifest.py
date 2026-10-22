"""Asset manifest models and generation utilities."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.asset_manager.security import calculate_sha256


class AssetEntry(BaseModel):
    name: str = Field(..., description="Asset identifier")
    category: str = Field(..., description="'model' or 'dataset'")
    version: str = Field(default="1.0.0")
    file_path: str = Field(..., description="Relative path in repo")
    source_url: Optional[str] = Field(None, description="Official download URL")
    license: str = Field(..., description="License identifier (e.g. Apache-2.0, MIT, Research-Only)")
    commercial_use_permitted: bool = Field(..., description="Whether commercial use is permitted")
    access_restrictions: Optional[str] = Field(None, description="Terms or registration requirements")
    sha256: str = Field(..., description="Expected or verified SHA-256")
    size_bytes: int = Field(default=0)
    retrieval_date: Optional[str] = Field(None)
    status: str = Field(default="VERIFIED", description="'VERIFIED', 'MISSING', 'TAMPERED', 'QUARANTINED'")


class MasterAssetManifest(BaseModel):
    generated_at: str
    offline_guarantee: bool = True
    assets: List[AssetEntry] = Field(default_factory=list)


# Curated catalog of approved face and anti-spoofing models/datasets with terms & licenses
APPROVED_ASSET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 1. Models
    "face_detection_yunet": {
        "category": "model",
        "file_path": "models/face_detection_yunet_2023mar.onnx",
        "source_url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "license": "Apache-2.0",
        "commercial_use_permitted": True,
        "access_restrictions": "None. Official open-source OpenCV Zoo model.",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size_bytes": 232589,
    },
    "face_recognition_sface": {
        "category": "model",
        "file_path": "models/face_recognition_sface_2021dec.onnx",
        "source_url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "license": "Apache-2.0",
        "commercial_use_permitted": True,
        "access_restrictions": "None. Official open-source OpenCV Zoo model for face verification.",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size_bytes": 38696353,
    },
    "deepfake_forensic_classifier": {
        "category": "model",
        "file_path": "models/deepfake_detector/model.safetensors",
        "source_url": "local",
        "license": "Apache-2.0",
        "commercial_use_permitted": True,
        "access_restrictions": "Proprietary bank forensic feature probe.",
        "sha256": "0590cc911d01c496b2df84f736c26a737e10edf16d60a74a0ded82a7ef374aa7",
        "size_bytes": 155,
    },
    # 2. Industry Benchmark Datasets (Discovery & Terms Tracker)
    "celeba_spoof": {
        "category": "dataset",
        "file_path": "assets/datasets/celeba_spoof",
        "source_url": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html",
        "license": "Non-Commercial / Research-Only",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires signing agreement with CUHK MMLab. Commercial use strictly prohibited.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
    "oulu_npu": {
        "category": "dataset",
        "file_path": "assets/datasets/oulu_npu",
        "source_url": "https://sites.google.com/site/oulunpudatabase/",
        "license": "Non-Commercial / Academic Research",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires formal written request to University of Oulu. Automated download not permitted.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
    "faceforensics_plus_plus": {
        "category": "dataset",
        "file_path": "assets/datasets/faceforensics",
        "source_url": "https://github.com/ondyari/faceforensics",
        "license": "TUM Research License",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires completing Google Form request to Technical University of Munich. Automated scraping blocked.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
    "casia_face_anti_spoofing": {
        "category": "dataset",
        "file_path": "assets/datasets/casia_fas",
        "source_url": "http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp",
        "license": "Academic Research Agreement",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires signed release agreement to CBSR CASIA. Not for commercial systems.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
    "replay_attack": {
        "category": "dataset",
        "file_path": "assets/datasets/replay_attack",
        "source_url": "https://www.idiap.ch/en/scientific-research/data/replayattack",
        "license": "Idiap Research Data License",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires registration and user agreement with Idiap Research Institute.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
    "siw_mv2": {
        "category": "dataset",
        "file_path": "assets/datasets/siw_mv2",
        "source_url": "https://cvlab.cse.msu.edu/siw-mv2-dataset.html",
        "license": "MSU Research Agreement",
        "commercial_use_permitted": False,
        "access_restrictions": "Requires signed release agreement to Michigan State University CVLAB.",
        "sha256": "N/A - manual registration required",
        "size_bytes": 0,
    },
}


def build_manifest(project_root: Path) -> MasterAssetManifest:
    """Build manifest evaluating physical presence and checksums."""
    entries = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for name, meta in APPROVED_ASSET_REGISTRY.items():
        rel_p = meta["file_path"]
        abs_p = project_root / rel_p

        status = "MISSING"
        verified_hash = meta["sha256"]
        actual_size = meta["size_bytes"]

        if abs_p.is_file():
            actual_size = abs_p.stat().st_size
            try:
                calc_hash = calculate_sha256(abs_p)
                if meta["sha256"].lower() in ("n/a", "local") or calc_hash == meta["sha256"].lower():
                    status = "VERIFIED"
                    verified_hash = calc_hash
                else:
                    status = "TAMPERED"
                    verified_hash = calc_hash
            except Exception:
                status = "READ_ERROR"
        elif abs_p.is_dir():
            status = "VERIFIED" if any(abs_p.iterdir()) else "MISSING"

        entries.append(
            AssetEntry(
                name=name,
                category=meta["category"],
                version="1.0.0",
                file_path=rel_p,
                source_url=meta.get("source_url"),
                license=meta["license"],
                commercial_use_permitted=meta["commercial_use_permitted"],
                access_restrictions=meta.get("access_restrictions"),
                sha256=verified_hash,
                size_bytes=actual_size,
                retrieval_date=now_iso,
                status=status,
            )
        )

    return MasterAssetManifest(
        generated_at=now_iso,
        offline_guarantee=True,
        assets=entries,
    )
