"""Pre-flight Face Check service for bank account opening (100% offline).

Enforces strict KYC selfie requirements before running AI detection:
- Exactly ONE face required (NO_FACE, MULTIPLE_FACES).
- Size and boundary checks (FACE_TOO_SMALL, FACE_PARTLY_HIDDEN).
- Image quality checks (BLURRY, TOO_DARK, TOO_BRIGHT).
- Head pose orientation check (FACE_NOT_FRONTAL): yaw, pitch, roll.

If any check fails, returns early with reason codes and a clear retake message.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.config import get_config
from app.services.face_detector import FaceCrop
from app.services.quality import QualityResult

logger = logging.getLogger("facecheck.face_check")


class FaceCheckResult:
    """Result of the pre-flight KYC face check."""

    def __init__(
        self,
        passed: bool,
        reason_codes: List[str],
        retake_message: Optional[str] = None,
        faces_detected: int = 0,
        head_pose: Optional[Dict[str, float]] = None,
        visibility: Optional[Dict[str, bool]] = None,
    ):
        self.passed = passed
        self.status = "PASSED" if passed else "FAILED"
        self.reason_codes = list(reason_codes)
        self.retake_message = retake_message
        self.faces_detected = faces_detected
        self.head_pose = head_pose or {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self.visibility = visibility or {"partly_hidden": False, "boundary_clipped": False}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reason_codes": self.reason_codes,
            "retake_message": self.retake_message,
            "faces_detected": self.faces_detected,
            "head_pose": self.head_pose,
            "visibility": self.visibility,
        }


def estimate_head_pose(landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
    """Estimate head pose (yaw, pitch, roll in degrees) from 5 facial landmarks.

    Landmarks order:
    0: right eye
    1: left eye
    2: nose tip
    3: right mouth corner
    4: left mouth corner
    """
    if len(landmarks) < 5:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    try:
        re = landmarks[0]
        le = landmarks[1]
        nose = landmarks[2]
        rm = landmarks[3]
        lm = landmarks[4]

        # 1. Roll: angle of eye line relative to horizontal
        eye_dx = float(le[0] - re[0])
        eye_dy = float(le[1] - re[1])
        roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

        # 2. Yaw: horizontal displacement of nose between eyes
        eye_mid_x = (re[0] + le[0]) / 2.0
        eye_dist = float(np.hypot(eye_dx, eye_dy))
        nose_offset_x = (nose[0] - eye_mid_x) / (eye_dist + 1e-5)
        yaw = float(np.clip(nose_offset_x * 90.0, -90.0, 90.0))

        # 3. Pitch: vertical position of nose between eyes and mouth
        eye_mid_y = (re[1] + le[1]) / 2.0
        mouth_mid_y = (rm[1] + lm[1]) / 2.0
        face_height = float(mouth_mid_y - eye_mid_y)
        nose_rel_y = (nose[1] - eye_mid_y) / (face_height + 1e-5)
        pitch = float(np.clip((nose_rel_y - 0.62) * 90.0, -90.0, 90.0))

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
        }
    except Exception as exc:
        logger.debug("Head pose estimation error: %s", exc)
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def evaluate_face_check(
    pil_image: Image.Image,
    face_crops: List[FaceCrop],
    quality_metrics: QualityResult,
) -> FaceCheckResult:
    """Perform pre-flight bank account opening face check.

    Reason codes:
    - NO_FACE: No face detected.
    - MULTIPLE_FACES: More than one face detected.
    - FACE_TOO_SMALL: Face bounding box is smaller than minimum required size.
    - BLURRY: Image sharpness / Laplacian variance is below threshold.
    - TOO_DARK: Mean luminance is below underexposure threshold.
    - TOO_BRIGHT: Mean luminance is above overexposure threshold.
    - FACE_NOT_FRONTAL: Head yaw, pitch, or roll exceeds frontal threshold.
    - FACE_PARTLY_HIDDEN: Face touches photo boundaries or key landmarks are clipped.
    """
    config = get_config()
    reasons: List[str] = []
    messages: List[str] = []

    img_w, img_h = pil_image.size
    face_count = len(face_crops)

    # 1. Require exactly one face
    if face_count == 0:
        return FaceCheckResult(
            passed=False,
            reason_codes=["NO_FACE"],
            retake_message="No face detected. Please position your face clearly in the camera frame.",
            faces_detected=0,
            visibility={"partly_hidden": True, "boundary_clipped": False},
        )

    if face_count > 1:
        return FaceCheckResult(
            passed=False,
            reason_codes=["MULTIPLE_FACES"],
            retake_message="Multiple faces detected. For bank identity verification, only the applicant must be in the photo. Please ensure bystanders or others are not in the frame.",
            faces_detected=face_count,
            visibility={"partly_hidden": False, "boundary_clipped": False},
        )

    # Exactly one face: analyze single face crop
    fc = face_crops[0]
    bx, by, bw, bh = fc.box

    # Threshold configurations
    face_check_cfg = getattr(config, "face_check", None)
    min_face_size = getattr(face_check_cfg, "min_face_size", 32) if face_check_cfg else getattr(config.quality, "min_face_size", 32)
    blur_threshold = getattr(config.quality, "blur_threshold", 50.0)
    dark_threshold = getattr(config.quality, "dark_luminance_threshold", 25.0)
    bright_threshold = getattr(config.quality, "bright_luminance_threshold", 235.0)

    # Head pose limits
    max_yaw = getattr(face_check_cfg, "max_yaw_degrees", 35.0) if face_check_cfg else 35.0
    max_pitch = getattr(face_check_cfg, "max_pitch_degrees", 25.0) if face_check_cfg else 25.0
    max_roll = getattr(face_check_cfg, "max_roll_degrees", 25.0) if face_check_cfg else 25.0
    boundary_margin = getattr(face_check_cfg, "boundary_margin_px", 5) if face_check_cfg else 5

    # 2. Size check
    if bw < min_face_size or bh < min_face_size:
        reasons.append("FACE_TOO_SMALL")
        messages.append("Face is too small in the frame. Please move closer to the camera.")

    # 3. Visibility and Boundary Clipping check (landmark-aware)
    if len(fc.landmarks) >= 5:
        # Check if any key facial landmark (eyes, nose, mouth) is clipped by frame edge
        lm_clipped = any(
            lx < 2 or lx > (img_w - 2) or ly < 2 or ly > (img_h - 2)
            for lx, ly in fc.landmarks
        )
        box_severely_clipped = (
            bx < -2
            or by < -2
            or (bx + bw) > (img_w + 2)
            or (by + bh) > (img_h + 2)
        )
        is_clipped = lm_clipped or box_severely_clipped
    else:
        is_clipped = (
            bx <= boundary_margin
            or by <= boundary_margin
            or (bx + bw) >= (img_w - boundary_margin)
            or (by + bh) >= (img_h - boundary_margin)
        )

    if is_clipped:
        reasons.append("FACE_PARTLY_HIDDEN")
        messages.append("Part of your face is cut off by the photo boundary. Please center your face.")

    # 4. Quality checks
    lap_var = float(quality_metrics.metrics.get("laplacian_variance", 100.0))
    mean_lum = float(quality_metrics.metrics.get("mean_luminance", 128.0))

    if lap_var < blur_threshold:
        reasons.append("BLURRY")
        messages.append("Photo is blurry. Please hold the phone steady and ensure the camera is in focus.")

    if mean_lum < dark_threshold:
        reasons.append("TOO_DARK")
        messages.append("Photo is too dark. Please take the photo in a well-lit environment.")
    elif mean_lum > bright_threshold:
        reasons.append("TOO_BRIGHT")
        messages.append("Photo is overexposed. Please avoid direct harsh glare or strong flash.")

    # 5. Background illumination & contrast check
    try:
        rgb_arr = np.array(pil_image.convert("L"), dtype=np.float32)
        # Sample 4 outer corners outside the face box
        c_size = max(4, min(16, int(min(img_w, img_h) * 0.08)))
        tl = np.mean(rgb_arr[:c_size, :c_size])
        tr = np.mean(rgb_arr[:c_size, -c_size:])
        bl = np.mean(rgb_arr[-c_size:, :c_size])
        br = np.mean(rgb_arr[-c_size:, -c_size:])
        bg_mean = (tl + tr + bl + br) / 4.0
        # If background is blown out (>240) while face is dark (<70), harsh backlight
        if bg_mean > 240.0 and mean_lum < 70.0:
            reasons.append("HARSH_BACKLIGHT")
            messages.append("Strong backlight detected behind you. Please move away from bright windows or lights.")
    except Exception:
        pass

    # 6. Head Pose check
    head_pose = estimate_head_pose(fc.landmarks)
    yaw = head_pose["yaw"]
    pitch = head_pose["pitch"]
    roll = head_pose["roll"]

    is_non_frontal = (abs(yaw) > max_yaw) or (abs(pitch) > max_pitch) or (abs(roll) > max_roll)
    if is_non_frontal:
        reasons.append("FACE_NOT_FRONTAL")
        messages.append("Face is turned or tilted away. Please look straight into the camera.")

    passed = len(reasons) == 0
    consolidated_message = " ".join(messages) if messages else None

    return FaceCheckResult(
        passed=passed,
        reason_codes=reasons,
        retake_message=consolidated_message,
        faces_detected=1,
        head_pose=head_pose,
        visibility={
            "partly_hidden": "FACE_PARTLY_HIDDEN" in reasons,
            "boundary_clipped": is_clipped,
        },
    )
