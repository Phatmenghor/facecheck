"""Image quality assessment service for FaceCheck API.

Evaluates uploaded images before biometrics to warn when low quality
may degrade detector confidence.

Checks:
- Minimum resolution (width / height)
- Face size in pixels (minimum bounding box dimension)
- Blur detection using Laplacian variance
- Over/under-exposure using mean grayscale luminance
- Heavy JPEG compression / compression artifacts
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.config import get_config

logger = logging.getLogger("facecheck.quality")


class QualityResult(BaseModel):
    status: str = Field(..., description="'OK' or 'LOW_QUALITY'")
    issues: List[str] = Field(default_factory=list, description="List of detected quality issues")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw computed quality metrics")


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute the variance of the Laplacian kernel across the grayscale image.

    Higher variance corresponds to sharper edges, lower variance to blur.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    # 3x3 discrete Laplacian operator:
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def compute_jpeg_blockiness(gray: np.ndarray) -> float:
    """Estimate 8x8 block boundary discontinuity typical in heavy JPEG compression."""
    h, w = gray.shape
    if h < 24 or w < 24:
        return 1.0

    # Vertical block boundaries (rows multiple of 8)
    row_bounds = gray[8::8, :]
    row_prev = gray[7:len(row_bounds) * 8:8, :]
    if row_bounds.shape == row_prev.shape and row_bounds.size > 0:
        diff_bound = np.mean(np.abs(row_bounds.astype(np.float32) - row_prev.astype(np.float32)))
    else:
        diff_bound = 0.0

    # Control rows in between block boundaries (rows offset by 4)
    row_ctrl1 = gray[4::8, :]
    row_ctrl2 = gray[3:len(row_ctrl1) * 8:8, :]
    if row_ctrl1.shape == row_ctrl2.shape and row_ctrl1.size > 0:
        diff_ctrl = np.mean(np.abs(row_ctrl1.astype(np.float32) - row_ctrl2.astype(np.float32)))
    else:
        diff_ctrl = 1.0

    ratio = float(diff_bound / (diff_ctrl + 1e-4))
    return ratio


def inspect_image_quality(
    pil_image: Image.Image,
    file_size_bytes: int,
    face_boxes: Optional[List[List[int]]] = None,
) -> QualityResult:
    """Evaluate image quality against configured thresholds.

    Args:
        pil_image: Loaded PIL Image (RGB).
        file_size_bytes: Uploaded byte length.
        face_boxes: Optional list of [x, y, w, h] bounding boxes.

    Returns:
        QualityResult with status ('OK' or 'LOW_QUALITY'), issue descriptions, and metrics.
    """
    config = get_config()
    q_cfg = config.quality

    issues: List[str] = []
    width, height = pil_image.size

    # Convert to grayscale array for luminance, blur, and blockiness
    gray = np.array(pil_image.convert("L"), dtype=np.float32)

    # 1. Resolution Check
    if width < q_cfg.min_width or height < q_cfg.min_height:
        issues.append(
            f"Image resolution too low ({width}x{height} px, minimum required: {q_cfg.min_width}x{q_cfg.min_height} px)"
        )

    # 2. Blur Check (Laplacian Variance)
    lap_var = compute_laplacian_variance(gray)
    if lap_var < q_cfg.blur_threshold:
        issues.append(
            f"Image appears blurry (Laplacian variance: {lap_var:.1f}, threshold: {q_cfg.blur_threshold:.1f})"
        )

    # 3. Luminance / Exposure Check
    mean_lum = float(np.mean(gray)) if gray.size > 0 else 0.0
    if mean_lum < q_cfg.dark_luminance_threshold:
        issues.append(
            f"Image is underexposed / very dark (mean luminance: {mean_lum:.1f}/255, threshold: {q_cfg.dark_luminance_threshold:.1f})"
        )
    elif mean_lum > q_cfg.bright_luminance_threshold:
        issues.append(
            f"Image is overexposed / very bright (mean luminance: {mean_lum:.1f}/255, threshold: {q_cfg.bright_luminance_threshold:.1f})"
        )

    # 4. Heavy JPEG Compression Check
    uncompressed_bytes = width * height * 3
    compression_ratio = float(file_size_bytes / uncompressed_bytes) if uncompressed_bytes > 0 else 1.0
    blockiness = compute_jpeg_blockiness(gray)

    if compression_ratio < q_cfg.heavy_compression_ratio and file_size_bytes > 1024:
        issues.append(
            f"Heavy JPEG compression detected (ratio: {compression_ratio:.4f}, artifact blockiness: {blockiness:.2f})"
        )
    elif blockiness > 2.2 and compression_ratio < 0.04:
        issues.append(
            f"Severe compression blockiness detected across 8x8 block boundaries (index: {blockiness:.2f})"
        )

    # 5. Face Size Check (if face crops detected)
    small_faces_count = 0
    if face_boxes:
        for box in face_boxes:
            # box is [x, y, w, h]
            w_box, h_box = box[2], box[3]
            if w_box < q_cfg.min_face_size or h_box < q_cfg.min_face_size:
                small_faces_count += 1

        if small_faces_count > 0:
            issues.append(
                f"{small_faces_count} detected face(s) are smaller than {q_cfg.min_face_size}x{q_cfg.min_face_size} px"
            )

    status = "LOW_QUALITY" if issues else "OK"

    metrics = {
        "width": width,
        "height": height,
        "laplacian_variance": round(lap_var, 2),
        "mean_luminance": round(mean_lum, 2),
        "compression_ratio": round(compression_ratio, 5),
        "blockiness_index": round(blockiness, 2),
        "face_count": len(face_boxes) if face_boxes else 0,
        "small_faces_count": small_faces_count,
    }

    return QualityResult(status=status, issues=issues, metrics=metrics)
