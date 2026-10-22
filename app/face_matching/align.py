"""Face alignment utilities using 5 facial landmarks."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("facecheck.face_matching.align")

# Canonical reference coordinates for 112x112 face alignment (ArcFace / SFace standard)
REFERENCE_LANDMARKS_112x112 = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)


def align_face_5point(
    bgr_img: np.ndarray,
    landmarks: List[Tuple[float, float]],
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align and crop face into canonical coordinates using similarity transform on 5 landmarks.

    Args:
        bgr_img: BGR numpy image.
        landmarks: 5 landmarks in order: right eye, left eye, nose, right mouth, left mouth.
        target_size: Output image dimensions (width, height).

    Returns:
        Aligned BGR face image.
    """
    if len(landmarks) < 5:
        return cv2.resize(bgr_img, target_size)

    src_pts = np.array(landmarks[:5], dtype=np.float32)

    # Scale reference landmarks if target size differs from 112x112
    ref_pts = REFERENCE_LANDMARKS_112x112.copy()
    if target_size != (112, 112):
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        ref_pts[:, 0] *= scale_x
        ref_pts[:, 1] *= scale_y

    try:
        # Estimate similarity transform matrix (rotation, translation, scale)
        tform, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
        if tform is None:
            # Fallback to least squares
            tform, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)

        if tform is not None:
            aligned = cv2.warpAffine(
                bgr_img,
                tform,
                target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned
    except Exception as exc:
        logger.debug("Affine alignment failed, falling back to direct crop: %s", exc)

    return cv2.resize(bgr_img, target_size)
