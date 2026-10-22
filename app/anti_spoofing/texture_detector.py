"""Physical print attack detection service for FaceCheck API.

Analyzes photographic inputs for:
- Paper texture and print substrate fiber grain
- Halftone / color laser printing dot matrices via 2D spatial frequency analysis
- Specular reflectance distribution (printed paper has flat diffuse reflection)
- Paper photo frame / cutout edge artifacts
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("facecheck.anti_spoofing.texture")


def detect_print_attack(
    pil_image: Image.Image,
    face_box: Tuple[int, int, int, int] | None = None,
) -> Dict[str, float | str | bool]:
    """Evaluate whether image contains characteristics of a printed photograph attack.

    Returns:
        Dictionary with 'score' (0.0 to 1.0), 'is_attack' (bool), and 'details'.
    """
    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    if h < 32 or w < 32:
        return {"score": 0.0, "is_attack": False, "details": "Image too small for texture analysis"}

    # 1. 2D Spectral Halftone Energy
    # Printed material exhibits concentrated harmonic spikes in mid-to-high spatial frequencies
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f_shift) + 1e-5
    log_mag = np.log(magnitude)

    # Concentric frequency ring ratios
    center_y, center_x = h // 2, w // 2
    r_max = min(center_y, center_x)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)

    mid_ring = (dist_from_center >= r_max * 0.35) & (dist_from_center <= r_max * 0.70)
    high_ring = dist_from_center > r_max * 0.70

    mid_energy = float(np.mean(log_mag[mid_ring])) if np.any(mid_ring) else 0.0
    high_energy = float(np.mean(log_mag[high_ring])) if np.any(high_ring) else 0.0
    spectral_ratio = float(mid_energy / (high_energy + 1e-4))

    # 2. Specular Reflectance Flatness (Standard Deviation of local high-pass laplacian)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_std = float(np.std(lap))

    # 3. Outer boundary paper border check
    # Many print attacks present a printed photo held in front of the lens
    margin = max(4, min(16, int(min(w, h) * 0.05)))
    top_border = np.mean(gray[:margin, :])
    bot_border = np.mean(gray[-margin:, :])
    left_border = np.mean(gray[:, :margin])
    right_border = np.mean(gray[:, -margin:])
    border_whiteness = float(np.mean([top_border, bot_border, left_border, right_border]))
    has_paper_border = border_whiteness > 238.0 and lap_std < 120.0

    # 4. Synthesize Print Attack Risk Score (0.0 - 1.0)
    score = 0.05
    reasons = []

    if spectral_ratio > 1.35 and lap_std < 45.0:
        score += 0.45
        reasons.append("Halftone frequency distribution indicative of laser/inkjet print")

    if has_paper_border:
        score += 0.40
        reasons.append("High-contrast white paper boundary detected around frame")

    if lap_std < 18.0 and not has_paper_border:
        score += 0.20
        reasons.append("Diffuse paper surface texture lacking camera sensor micro-grain")

    score = min(1.0, max(0.0, score))
    is_attack = score >= 0.50

    return {
        "score": round(score, 3),
        "is_attack": is_attack,
        "details": "; ".join(reasons) if reasons else "No print texture anomalies detected",
    }
