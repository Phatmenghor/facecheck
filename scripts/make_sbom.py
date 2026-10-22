#!/usr/bin/env python3
"""CycloneDX Software Bill of Materials (SBOM) generator for FaceCheck API.

Generates a CycloneDX 1.5 JSON formatted SBOM detailing:
- Application components & version
- Installed Python packages, exact versions, and licenses
- Model artifacts and SHA-256 cryptographic hashes
- Output file: eval/reports/sbom.json

Usage:
    python scripts/make_sbom.py [--output eval/reports/sbom.json]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_sbom(output_path: Path) -> Dict[str, Any]:
    print("============================================================")
    print(" FACECHECK CYCLONEDX SBOM GENERATOR")
    print(f" Target output: {output_path}")
    print("============================================================")

    serial_number = f"urn:uuid:{uuid.uuid4()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    components: List[Dict[str, Any]] = []

    # 1. Root application component
    components.append({
        "type": "application",
        "name": "facecheck-api",
        "version": "1.0.0",
        "description": "Air-gapped AI Face Verification & Deepfake Risk Scoring REST API",
        "scope": "required",
        "licenses": [{"license": {"id": "Proprietary"}}],
    })

    # 2. Installed Python libraries
    print("[1/2] Scanning installed Python packages...")
    relevant_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pillow",
        "pyyaml",
        "cryptography",
        "numpy",
        "exifread",
        "python-multipart",
        "pytest",
        "httpx",
        "starlette",
    ]

    for pkg_name in relevant_packages:
        try:
            dist = importlib.metadata.distribution(pkg_name)
            meta = dist.metadata
            version = dist.version
            license_val = meta.get("License", "Unknown") or "Unknown"
            summary = meta.get("Summary", "")

            components.append({
                "type": "library",
                "name": pkg_name,
                "version": version,
                "description": summary,
                "scope": "required",
                "licenses": [{"license": {"name": license_val}}],
                "purl": f"pkg:pypi/{pkg_name}@{version}",
            })
        except importlib.metadata.PackageNotFoundError:
            continue

    # 3. Model weights artifacts
    print("[2/2] Fingerprinting model weights artifacts...")
    model_paths = [
        PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx",
        PROJECT_ROOT / "models" / "deepfake_detector" / "model.safetensors",
    ]

    for mp in model_paths:
        if mp.is_file():
            sha = get_file_sha256(mp)
            components.append({
                "type": "file",
                "name": mp.name,
                "version": "1.0.0",
                "description": f"Offline AI neural weights artifact: {mp.name}",
                "hashes": [{"alg": "SHA-256", "content": sha}],
                "licenses": [{"license": {"name": "Apache-2.0" if "yunet" in mp.name else "Internal-Proprietary"}}],
            })

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial_number,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [{"name": "FaceCheck Internal SBOM Tool", "version": "1.0.0"}],
            "component": {
                "type": "application",
                "name": "facecheck-api",
                "version": "1.0.0",
            },
        },
        "components": components,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")

    print("------------------------------------------------------------")
    print(f"[SUCCESS] SBOM generated with {len(components)} registered components.")
    print(f"Output saved to: {output_path}")
    print("============================================================")
    return sbom


def main():
    parser = argparse.ArgumentParser(description="Generate FaceCheck SBOM")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "reports" / "sbom.json"),
        help="Path for SBOM JSON output (default: eval/reports/sbom.json)",
    )
    args = parser.parse_args()
    generate_sbom(Path(args.output))


if __name__ == "__main__":
    main()
