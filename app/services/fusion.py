"""Score fusion (1-100), label mapping, and explainable reasons."""

from __future__ import annotations

from typing import List, Optional, Tuple
from app.config import get_config


def compute_score_and_label(
    metadata_score: int,
    face_max_score: Optional[int],
    full_image_score: int,
    faces_found: int,
    metadata_detail: str,
    tool_detected: Optional[str] = None,
    has_camera_exif: bool = False,
) -> Tuple[Optional[int], str, List[str]]:
    """Step 5: Fuse signals into integer score 1-100, label, and plain-language reasons.

    Args:
        metadata_score: Metadata risk score 1-100.
        face_max_score: Highest risk score among face crops (1-100) or None if no faces.
        full_image_score: Full canvas detector risk score 1-100.
        faces_found: Count of detected faces.
        metadata_detail: Human-readable metadata findings.
        tool_detected: Optional tool signature name if found.
        has_camera_exif: Whether camera hardware EXIF was verified.

    Returns:
        Tuple of (score_or_none, label, reasons_list).
    """
    config = get_config()
    thresholds = config.fusion.thresholds

    reasons: List[str] = []

    # If no face is found
    if faces_found == 0 or face_max_score is None:
        reasons.append("No human face detected in the image")
        return None, "NO_FACE_FOUND", reasons

    weights = config.fusion.weights
    w_meta = weights.metadata
    w_face = weights.detector_face
    w_full = weights.detector_full_image

    total_weight = w_meta + w_face + w_full
    if total_weight <= 0:
        total_weight = 1.0

    weighted_val = (
        metadata_score * w_meta
        + face_max_score * w_face
        + full_image_score * w_full
    ) / total_weight

    score = int(round(weighted_val))
    # Clamp to strictly 1..100
    score = max(1, min(100, score))

    # Determine label
    if score <= thresholds.likely_real_max:
        label = "LIKELY_REAL"
    elif score <= thresholds.uncertain_max:
        label = "UNCERTAIN"
    else:
        label = "LIKELY_FAKE"

    # Formulate plain-language reasons
    if face_max_score >= 85:
        reasons.append("Face detector strongly suggests AI generation (synthetic / 3D cartoon illustration with absence of skin pore texture)")
    elif face_max_score >= 70:
        reasons.append("Face detector strongly suggests AI generation")
    elif face_max_score <= 30:
        reasons.append("Facial biometrics indicate natural human features")
    else:
        reasons.append("Facial biometrics show borderline or ambiguous patterns")

    if tool_detected:
        reasons.append(f"AI tool signature in metadata ({tool_detected})")
    elif has_camera_exif:
        reasons.append("Authentic camera hardware EXIF tags verified")
    else:
        reasons.append("No camera EXIF metadata found (weak signal)")

    if full_image_score >= 85:
        reasons.append("High non-photographic rendering ratio detected across full canvas")
    elif full_image_score >= 70:
        reasons.append("Synthetic generation patterns detected across full image")

    return score, label, reasons
