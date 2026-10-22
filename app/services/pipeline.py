"""Direct image processing pipeline for FaceCheck API.

Executes validation, metadata inspection, face detection, quality analysis,
and AI deepfake scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from PIL import Image

from app.services.validator import validate_and_load_image
from app.services.metadata import inspect_metadata
from app.services.quality import inspect_image_quality
from app.services.face_detector import get_face_detector
from app.services.face_check import evaluate_face_check
from app.detectors.factory import get_detector
from app.services.fusion import compute_score_and_label
from app.services.gender_detector import estimate_gender

logger = logging.getLogger("facecheck.pipeline")


def run_pipeline(
    filename: str,
    raw_bytes: bytes,
) -> Dict[str, Any]:
    """Execute the full FaceCheck pipeline on raw bytes.

    Returns a serializable dictionary containing:
    - pil_image: (optional reference if needed in memory)
    - score, label, reasons
    - faces_found, faces
    - signals
    - quality
    """
    # 1. Validation & sanitization
    pil_image, mime_type, file_size = validate_and_load_image(filename, raw_bytes)

    # 2. Metadata inspection
    meta_res = inspect_metadata(raw_bytes, pil_image)
    meta_score = meta_res["score"]
    meta_detail = meta_res["detail"]
    tool_detected = meta_res.get("tool_detected")
    has_camera_exif = meta_res.get("has_camera_exif", False)

    # 3. Face detection with YuNet ONNX
    face_detector = get_face_detector()
    face_crops = face_detector.detect_and_crop(pil_image)
    faces_found = len(face_crops)
    face_boxes = [fc.box for fc in face_crops]

    # 4. Image Quality Check
    quality_res = inspect_image_quality(pil_image, file_size, face_boxes)

    # 5. Pre-flight Face Check (KYC Photo-only requirements)
    face_check_res = evaluate_face_check(pil_image, face_crops, quality_res)

    if not face_check_res.passed:
        # If exactly one face was detected, estimate gender so it is not omitted
        early_gender, early_conf = None, None
        if len(face_crops) == 1:
            early_gender, early_conf = estimate_gender(
                crop=face_crops[0].crop,
                full_image=pil_image,
                landmarks=face_crops[0].landmarks,
            )

        # STOP EARLY: Do not run AI detectors if pre-flight face check fails
        retake_msg = face_check_res.retake_message or "Photo retake required."
        return {
            "score": None,
            "label": "RETAKE",
            "reason_codes": face_check_res.reason_codes,
            "face_check": face_check_res.to_dict(),
            "gender": early_gender,
            "gender_confidence": early_conf,
            "faces_found": faces_found,
            "faces": [{"box": fc.box, "score": 0, "gender": early_gender or "UNKNOWN", "gender_confidence": early_conf or 0.0} for fc in face_crops],
            "signals": {
                "metadata": {"score": meta_score, "detail": meta_detail},
                "detector_face": {"score": None},
                "detector_full_image": {"score": None},
            },
            "quality": {
                "status": quality_res.status,
                "issues": quality_res.issues,
                "metrics": quality_res.metrics,
            },
            "reasons": [retake_msg],
            "disclaimer": "Risk indicator only, not final proof.",
        }

    # 6. AI deepfake detection (only runs when face check PASSED)
    detector = get_detector()

    # Full image prediction
    full_prob = detector.predict(pil_image)
    detector_full_score = max(1, min(100, int(round(full_prob * 100))))

    # Face crops prediction & gender estimation
    faces_list = []
    face_scores = []
    primary_gender = None
    primary_gender_conf = None

    for idx, fc in enumerate(face_crops):
        crop_prob = detector.predict(fc.crop)
        c_score = max(1, min(100, int(round(crop_prob * 100))))
        face_scores.append(c_score)

        g_label, g_conf = estimate_gender(
            crop=fc.crop,
            full_image=pil_image,
            landmarks=fc.landmarks,
        )
        faces_list.append({
            "box": fc.box,
            "score": c_score,
            "gender": g_label,
            "gender_confidence": g_conf,
        })
        if idx == 0:
            primary_gender = g_label
            primary_gender_conf = g_conf

    face_max_score = max(face_scores) if face_scores else None

    # 6. Score Fusion
    fused_score, label, reasons = compute_score_and_label(
        metadata_score=meta_score,
        face_max_score=face_max_score,
        full_image_score=detector_full_score,
        faces_found=faces_found,
        metadata_detail=meta_detail,
        tool_detected=tool_detected,
        has_camera_exif=has_camera_exif,
    )

    # If low quality, add cautionary warning to reasons
    if quality_res.status == "LOW_QUALITY" and quality_res.issues:
        reasons.append(
            f"Image quality warning: low quality input reduces score reliability ({', '.join(quality_res.issues)})"
        )

    return {
        "score": fused_score,
        "label": label,
        "reason_codes": [],
        "face_check": face_check_res.to_dict(),
        "gender": primary_gender,
        "gender_confidence": primary_gender_conf,
        "faces_found": faces_found,
        "faces": faces_list,
        "signals": {
            "metadata": {"score": meta_score, "detail": meta_detail},
            "detector_face": {"score": face_max_score},
            "detector_full_image": {"score": detector_full_score},
        },
        "quality": {
            "status": quality_res.status,
            "issues": quality_res.issues,
            "metrics": quality_res.metrics,
        },
        "reasons": reasons,
        "mime_type": mime_type,
        "file_size": file_size,
    }
