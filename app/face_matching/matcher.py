"""Face matching orchestration and decision calibration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.services.face_detector import get_face_detector, FaceCrop
from app.face_matching.models import (
    BaseFaceRecognizer,
    get_face_recognizer,
    ModelUnavailableError,
)

logger = logging.getLogger("facecheck.face_matching.matcher")


class FaceMatchResult(BaseModel):
    """Structured result of biometric face comparison."""
    decision: str = Field(..., description="'PASS', 'FAIL', 'REVIEW', or 'UNAVAILABLE'")
    matched: bool = Field(..., description="Whether similarity meets or exceeds calibrated threshold")
    similarity_score: Optional[float] = Field(None, description="Raw cosine similarity score (-1.0 to 1.0)")
    calibrated_score: Optional[int] = Field(None, ge=1, le=100, description="Calibrated similarity 1-100%")
    threshold: float = Field(..., description="Operating threshold used for decision")
    status: str = Field(..., description="'MATCH', 'NON_MATCH', 'UNCERTAIN', 'REJECTED', 'UNAVAILABLE'")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable decision reason codes")
    face_a_box: Optional[List[int]] = Field(None, description="Bounding box [x,y,w,h] of face in Image A")
    face_b_box: Optional[List[int]] = Field(None, description="Bounding box [x,y,w,h] of face in Image B")
    model_version: str = Field(..., description="Face recognition embedding model identifier")
    disclaimer: str = "Identity comparison only; liveness and physical document authenticity must be evaluated independently."


def format_raw_face_detection(fc: FaceCrop) -> np.ndarray:
    """Format FaceCrop bounding box and landmarks into cv2.FaceDetectorYN 15-float array."""
    arr = np.zeros(15, dtype=np.float32)
    arr[0] = fc.box[0]
    arr[1] = fc.box[1]
    arr[2] = fc.box[2]
    arr[3] = fc.box[3]
    if len(fc.landmarks) >= 5:
        # Landmarks: [re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y]
        for idx in range(5):
            arr[4 + idx * 2] = fc.landmarks[idx][0]
            arr[5 + idx * 2] = fc.landmarks[idx][1]
    arr[14] = fc.confidence
    return arr


def compare_faces(
    image_a: Image.Image,
    image_b: Image.Image,
    custom_threshold: Optional[float] = None,
    recognizer: Optional[BaseFaceRecognizer] = None,
) -> FaceMatchResult:
    """Compare two facial images and evaluate whether they belong to the same person.

    Guarantees:
    - Never uses image hashing as proof of identity.
    - Never uses demographic attributes in similarity computation.
    - Strictly reports UNAVAILABLE if face recognition model is missing.
    - Rejects ambiguous multi-face or zero-face inputs.
    """
    rec = recognizer or get_face_recognizer()
    threshold = custom_threshold if custom_threshold is not None else rec.default_threshold

    # 1. Face detection on both inputs
    detector = get_face_detector()
    crops_a = detector.detect_and_crop(image_a)
    crops_b = detector.detect_and_crop(image_b)

    # 2. Input validation and ambiguity rejection
    reason_codes: List[str] = []
    if len(crops_a) == 0:
        reason_codes.append("NO_FACE_IN_IMAGE_A")
    elif len(crops_a) > 1:
        reason_codes.append("MULTIPLE_FACES_IN_IMAGE_A")

    if len(crops_b) == 0:
        reason_codes.append("NO_FACE_IN_IMAGE_B")
    elif len(crops_b) > 1:
        reason_codes.append("MULTIPLE_FACES_IN_IMAGE_B")

    if reason_codes:
        status_label = "REJECTED" if any("NO_FACE" in r for r in reason_codes) else "UNCERTAIN"
        decision_label = "FAIL" if status_label == "REJECTED" else "REVIEW"
        return FaceMatchResult(
            decision=decision_label,
            matched=False,
            similarity_score=None,
            calibrated_score=None,
            threshold=threshold,
            status=status_label,
            reason_codes=reason_codes,
            face_a_box=crops_a[0].box if crops_a else None,
            face_b_box=crops_b[0].box if crops_b else None,
            model_version=rec.model_name,
        )

    # 3. Model Availability Verification
    if not rec.is_loaded:
        return FaceMatchResult(
            decision="UNAVAILABLE",
            matched=False,
            similarity_score=None,
            calibrated_score=None,
            threshold=threshold,
            status="UNAVAILABLE",
            reason_codes=["MODEL_UNAVAILABLE"],
            face_a_box=crops_a[0].box,
            face_b_box=crops_b[0].box,
            model_version=rec.model_name,
        )

    # 4. Extract aligned embeddings
    bgr_a = cv2.cvtColor(np.array(image_a), cv2.COLOR_RGB2BGR)
    bgr_b = cv2.cvtColor(np.array(image_b), cv2.COLOR_RGB2BGR)

    face_rec_a = format_raw_face_detection(crops_a[0])
    face_rec_b = format_raw_face_detection(crops_b[0])

    try:
        emb_a = rec.extract_embedding(bgr_a, face_rec_a)
        emb_b = rec.extract_embedding(bgr_b, face_rec_b)
    except ModelUnavailableError:
        return FaceMatchResult(
            decision="UNAVAILABLE",
            matched=False,
            similarity_score=None,
            calibrated_score=None,
            threshold=threshold,
            status="UNAVAILABLE",
            reason_codes=["MODEL_UNAVAILABLE"],
            face_a_box=crops_a[0].box,
            face_b_box=crops_b[0].box,
            model_version=rec.model_name,
        )

    # 5. Calculate normalized similarity score
    cos_sim = rec.compute_similarity(emb_a, emb_b)
    rounded_sim = round(float(cos_sim), 4)

    # Calibrate into 1-100% confidence scale based on SFace cosine distribution
    # Cosine ranges from ~-0.2 (different identities) to ~0.8+ (same person high res)
    # Cosine 0.363 is FAR=0.1% threshold (mapped to 70% confidence)
    if cos_sim >= threshold:
        calibrated_score = int(np.clip(70 + ((cos_sim - threshold) / (1.0 - threshold + 1e-5)) * 30, 70, 99))
    elif cos_sim >= (threshold - 0.12):
        calibrated_score = int(np.clip(45 + ((cos_sim - (threshold - 0.12)) / 0.12) * 24, 45, 69))
    else:
        calibrated_score = int(np.clip(max(1.0, (cos_sim + 0.2) / (threshold + 0.2) * 44), 1, 44))

    # 6. Calibrated Decision Logic
    review_margin = 0.12
    if cos_sim >= threshold:
        decision = "PASS"
        status = "MATCH"
        matched = True
    elif cos_sim >= (threshold - review_margin):
        decision = "REVIEW"
        status = "UNCERTAIN"
        matched = False
        reason_codes.append("BORDERLINE_SIMILARITY")
    else:
        decision = "FAIL"
        status = "NON_MATCH"
        matched = False
        reason_codes.append("LOW_SIMILARITY")

    return FaceMatchResult(
        decision=decision,
        matched=matched,
        similarity_score=rounded_sim,
        calibrated_score=calibrated_score,
        threshold=threshold,
        status=status,
        reason_codes=reason_codes,
        face_a_box=crops_a[0].box,
        face_b_box=crops_b[0].box,
        model_version=rec.model_name,
    )
