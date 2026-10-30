"""Unified Presentation-Attack Detection (PAD) service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from pydantic import BaseModel, Field

from app.anti_spoofing.texture_detector import detect_print_attack
from app.anti_spoofing.replay_detector import detect_screen_replay
from app.detectors.factory import get_detector

logger = logging.getLogger("facecheck.anti_spoofing.pad_service")


class AttackSignal(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Risk probability score 0.0 to 1.0")
    detected: bool
    details: str


class PADResult(BaseModel):
    decision: str = Field(..., description="'PASS', 'FAIL', 'REVIEW', or 'UNAVAILABLE'")
    attack_type: str = Field(
        ...,
        description="'GENUINE', 'PRINT_ATTACK', 'SCREEN_REPLAY', 'DIGITAL_MANIPULATION', or 'UNCERTAIN'",
    )
    spoof_probability: float = Field(..., ge=0.0, le=1.0, description="Maximum spoof probability score")
    calibrated_risk: int = Field(..., ge=1, le=100, description="Overall presentation-attack risk score 1-100")
    liveness_status: str = Field(
        default="INCONCLUSIVE_STATIC_IMAGE",
        description="'INCONCLUSIVE_STATIC_IMAGE', 'VERIFIED_INTERACTIVE_CHALLENGE', or 'SPOOF_DETECTED'",
    )
    reason_codes: List[str] = Field(default_factory=list, description="Reason codes for presentation attack")
    signals: Dict[str, AttackSignal] = Field(..., description="Breakdown across physical and digital attack vectors")
    disclaimer: str = (
        "Static-image presentation attack detection evaluates surface artifacts only and does not establish "
        "live human presence. For high-assurance KYC, an interactive challenge session or video sequence is required."
    )


def evaluate_pad(
    pil_image: Image.Image,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> PADResult:
    """Evaluate image against physical and digital presentation attack vectors.

    Covers:
    - Printed photographs (paper grain, halftone dot structure, diffuse reflectance, borders)
    - Electronic screen replays (2D moire grid patterns, screen bezel, glass specular glare)
    - Synthetic digital manipulation (AI generator noise, FFT spectral roll-off, chroma artifacts)
    """
    # 1. Physical Print Attack Check
    print_res = detect_print_attack(pil_image, face_box)
    print_score = float(print_res["score"])
    print_signal = AttackSignal(
        score=print_score,
        detected=bool(print_res["is_attack"]),
        details=str(print_res["details"]),
    )

    # 2. Electronic Screen Replay Attack Check
    replay_res = detect_screen_replay(pil_image, face_box)
    replay_score = float(replay_res["score"])
    replay_signal = AttackSignal(
        score=replay_score,
        detected=bool(replay_res["is_attack"]),
        details=str(replay_res["details"]),
    )

    # 3. Digital Manipulation / Deepfake Classifier
    detector = get_detector()
    try:
        if face_box and len(face_box) == 4:
            fx, fy, fw, fh = face_box
            w_img, h_img = pil_image.size
            mx, my = int(fw * 0.2), int(fh * 0.2)
            crop_box = (
                max(0, fx - mx),
                max(0, fy - my),
                min(w_img, fx + fw + mx),
                min(h_img, fy + fh + my),
            )
            if crop_box[2] > crop_box[0] + 10 and crop_box[3] > crop_box[1] + 10:
                eval_img = pil_image.crop(crop_box)
            else:
                eval_img = pil_image
        else:
            eval_img = pil_image
        digital_score = float(detector.predict(eval_img))
    except Exception as exc:
        logger.warning("Digital manipulation detector failed: %s", exc)
        digital_score = 0.50

    digital_signal = AttackSignal(
        score=round(digital_score, 3),
        detected=digital_score >= 0.70,
        details="Forensic feature probe analysis",
    )

    # 4. Synthesize Maximum Attack Risk
    max_spoof_prob = max(print_score, replay_score, digital_score)
    calibrated_risk = max(1, min(100, int(round(max_spoof_prob * 100))))

    reason_codes: List[str] = []
    if print_score >= 0.50:
        reason_codes.append("PRINT_ATTACK_DETECTED")
    if replay_score >= 0.50:
        reason_codes.append("SCREEN_REPLAY_DETECTED")
    if digital_score >= 0.70:
        reason_codes.append("DIGITAL_MANIPULATION_DETECTED")

    # 5. Attack Type Classification
    if print_score >= 0.50 and print_score >= replay_score and print_score >= digital_score:
        attack_type = "PRINT_ATTACK"
    elif replay_score >= 0.50 and replay_score >= print_score and replay_score >= digital_score:
        attack_type = "SCREEN_REPLAY"
    elif digital_score >= 0.70:
        attack_type = "DIGITAL_MANIPULATION"
    elif max_spoof_prob < 0.30:
        attack_type = "GENUINE"
    else:
        attack_type = "UNCERTAIN"

    # 6. Decision
    if max_spoof_prob >= 0.65:
        decision = "FAIL"
        liveness_status = "SPOOF_DETECTED"
    elif max_spoof_prob >= 0.35:
        decision = "REVIEW"
        liveness_status = "INCONCLUSIVE_STATIC_IMAGE"
        if not reason_codes:
            reason_codes.append("UNCERTAIN_PRESENTATION_ATTACK_METRICS")
    else:
        decision = "PASS"
        liveness_status = "INCONCLUSIVE_STATIC_IMAGE"

    return PADResult(
        decision=decision,
        attack_type=attack_type,
        spoof_probability=round(max_spoof_prob, 3),
        calibrated_risk=calibrated_risk,
        liveness_status=liveness_status,
        reason_codes=reason_codes,
        signals={
            "print_attack": print_signal,
            "screen_replay": replay_signal,
            "digital_manipulation": digital_signal,
        },
    )
