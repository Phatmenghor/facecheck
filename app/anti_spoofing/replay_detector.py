"""Screen replay attack detection service for FaceCheck API.

Analyzes images for:
- 2D Moiré interference patterns caused by digital screen subpixel grids
- High-contrast smartphone/tablet device bezel margins
- Harsh directional glass reflection glare typical of phone/monitor replays
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("facecheck.anti_spoofing.replay")


def detect_screen_replay(
    pil_image: Image.Image,
    face_box: Tuple[int, int, int, int] | None = None,
) -> Dict[str, float | str | bool]:
    """Detect screen replay presentation attacks (photographing phone, tablet, or monitor).

    Returns:
        Dictionary with 'score' (0.0 to 1.0), 'is_attack' (bool), and 'details'.
    """
    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    if h < 32 or w < 32:
        return {"score": 0.0, "is_attack": False, "details": "Image too small for replay analysis"}

    # 1. 2D Moiré Pattern Analysis via Peak Spectral Detection
    # Digital displays produce distinct periodic 2D sinusoidal interference peaks
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f_shift)
    h_c, w_c = h // 2, w // 2

    # Zero out DC central component and boundary cross axes
    # (Cross axes contain boundary step harmonics that occur in all rectangular images)
    mag[max(0, h_c - 6) : min(h, h_c + 7), max(0, w_c - 6) : min(w, w_c + 7)] = 0.0
    mag[max(0, h_c - 2) : min(h, h_c + 3), :] = 0.0
    mag[:, max(0, w_c - 2) : min(w, w_c + 3)] = 0.0

    # Isolate mid-to-high frequency band where screen subpixel grids resonate
    r_max = min(h_c, w_c)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - h_c) ** 2 + (x_grid - w_c) ** 2)
    hf_mask = (dist > 0.25 * r_max) & (dist < 0.90 * r_max)

    if np.any(hf_mask):
        mean_mag = float(np.mean(mag[hf_mask])) + 1e-5
        max_peak = float(np.max(mag[hf_mask]))
        peak_to_average = max_peak / mean_mag
    else:
        peak_to_average = 1.0

    # 2. Device Bezel Detection (Dark rectangular frame surrounding the display)
    margin = max(4, min(24, int(min(w, h) * 0.08)))
    top_border = np.mean(gray[:margin, :])
    bot_border = np.mean(gray[-margin:, :])
    left_border = np.mean(gray[:, :margin])
    right_border = np.mean(gray[:, -margin:])
    border_darkness = float(np.mean([top_border, bot_border, left_border, right_border]))
    center_brightness = float(np.mean(gray[margin:-margin, margin:-margin])) if h > margin*2 and w > margin*2 else 128.0

    has_bezel = (border_darkness < 35.0) and (center_brightness > 85.0)

    # 3. Specular Screen Reflection Glare
    # High-intensity blown-out specular highlights on glass screen (contiguous patch >= 80px)
    glare_mask = (gray > 250.0).astype(np.uint8)
    glare_pixels = int(np.sum(glare_mask))
    glare_ratio = float(glare_pixels / gray.size)

    has_screen_glare = False
    if 0.005 < glare_ratio < 0.08 and peak_to_average > 18.0:
        # Require a contiguous glare spot rather than tiny isolated eye points
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glare_mask)
        if any(stats[i, cv2.CC_STAT_AREA] >= 80 for i in range(1, num_labels)):
            has_screen_glare = True

    # 4. Synthesize Screen Replay Score
    score = 0.05
    reasons = []

    if peak_to_average > 35.0:
        score += 0.50
        reasons.append("High-frequency 2D moire interference detected from electronic screen pixel grid")
    elif peak_to_average > 22.0:
        score += 0.25
        reasons.append("Moderate periodic frequency spikes consistent with display subpixels")

    if has_bezel:
        score += 0.35
        reasons.append("Electronic display bezel frame detected enclosing the face")

    if has_screen_glare:
        score += 0.20
        reasons.append("Characteristic screen glass specular reflection pattern")

    score = min(1.0, max(0.0, score))
    is_attack = score >= 0.50

    return {
        "score": round(score, 3),
        "is_attack": is_attack,
        "details": "; ".join(reasons) if reasons else "No screen replay artifacts detected",
    }
