"""Lightweight facial gender estimation service running 100% offline.

Extracts facial anthropometric, chromatic, and textural biometrics:
- Landmark-guided hairless dermal anchor (nasal dorsum & sub-orbital malar zone)
- Chin and philtrum follicle luminance ratios (beard and mustache shadow)
- Stabilized logarithmic luminance comparison
- Lip-to-skin erythema chromatic contrast
- Lower face jawline aspect ratio (mouth width to interpupillary distance)
- Calibrated confidence score with smooth deadband
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("facecheck.gender")


def estimate_gender(
    crop: Image.Image,
    full_image: Optional[Image.Image] = None,
    landmarks: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[str, float]:
    """Estimate gender ('MALE' | 'FEMALE' | 'UNKNOWN') and calibrated confidence score.

    Args:
        crop: PIL Image containing cropped human face.
        full_image: Optional original PIL Image for precise landmark sampling.
        landmarks: Optional 5 facial landmarks [(re), (le), (nose), (rm), (lm)].

    Returns:
        Tuple of (gender_label, confidence_score).
    """
    try:
        if crop.size[0] < 16 or crop.size[1] < 16:
            return "UNKNOWN", 0.0

        # -------------------------------------------------------------
        # Mode 1: Full-Image Landmark-Guided Biometric Analysis
        # -------------------------------------------------------------
        if full_image is not None and landmarks is not None and len(landmarks) >= 5:
            rgb_full = np.array(full_image.convert("RGB"), dtype=np.float32)
            h_img, w_img, _ = rgb_full.shape

            re = np.array(landmarks[0], dtype=np.float32)
            le = np.array(landmarks[1], dtype=np.float32)
            nose = np.array(landmarks[2], dtype=np.float32)
            rm = np.array(landmarks[3], dtype=np.float32)
            l_m = np.array(landmarks[4], dtype=np.float32)

            ipd = max(1.0, float(np.linalg.norm(le - re)))
            mouth_center = (rm + l_m) / 2.0
            eye_mid = (re + le) / 2.0
            nose_mouth_d = max(1.0, float(np.linalg.norm(mouth_center - nose)))

            def sample_patch(pt: np.ndarray, r: int = 2) -> np.ndarray:
                x = int(np.clip(pt[0], r, w_img - r - 1))
                y = int(np.clip(pt[1], r, h_img - r - 1))
                return rgb_full[y - r : y + r + 1, x - r : x + r + 1]

            # 1. Hairless Dermal Anchor (Reference Skin Tone)
            # Nasal dorsum and sub-orbital malar cheeks are guaranteed hairless skin
            dorsum_pt = (eye_mid + nose) / 2.0
            r_malar = (re + nose) / 2.0
            l_malar = (le + nose) / 2.0

            dorsum = sample_patch(dorsum_pt, r=2)
            r_cheek = sample_patch(r_malar, r=2)
            l_cheek = sample_patch(l_malar, r=2)

            skin_ref = (
                np.mean(dorsum, axis=(0, 1))
                + np.mean(r_cheek, axis=(0, 1))
                + np.mean(l_cheek, axis=(0, 1))
            ) / 3.0
            ref_lum = float(0.299 * skin_ref[0] + 0.587 * skin_ref[1] + 0.114 * skin_ref[2])

            # 2. Lower face zones: Chin & Philtrum
            chin_y = float(min(h_img - 2, mouth_center[1] + nose_mouth_d * 0.38))
            chin = sample_patch(np.array([mouth_center[0], chin_y], dtype=np.float32), r=2)
            chin_mean = np.mean(chin, axis=(0, 1))
            chin_lum = float(0.299 * chin_mean[0] + 0.587 * chin_mean[1] + 0.114 * chin_mean[2])
            log_chin = float(np.log((chin_lum + 10.0) / (ref_lum + 10.0)))

            philtrum_pt = (nose + mouth_center) / 2.0
            philtrum = sample_patch(philtrum_pt, r=2)
            philtrum_mean = np.mean(philtrum, axis=(0, 1))
            philtrum_lum = float(0.299 * philtrum_mean[0] + 0.587 * philtrum_mean[1] + 0.114 * philtrum_mean[2])
            log_philtrum = float(np.log((philtrum_lum + 10.0) / (ref_lum + 10.0)))

            # 3. Lip Erythema Contrast vs Reference Skin
            lip = sample_patch(mouth_center, r=2)
            lip_mean = np.mean(lip, axis=(0, 1))
            lip_erythema = float(lip_mean[0] - (lip_mean[1] + lip_mean[2]) * 0.5)
            ref_erythema = float(skin_ref[0] - (skin_ref[1] + skin_ref[2]) * 0.5)
            norm_lip_contrast = float((lip_erythema - ref_erythema) / (ref_lum + 20.0))

            # 4. Chin Stubble Texture (Follicle micro-variation)
            chin_gray = 0.299 * chin[:, :, 0] + 0.587 * chin[:, :, 1] + 0.114 * chin[:, :, 2]
            stubble = float(np.std(chin_gray))

            # 5. Jawline / Mouth Width to IPD Aspect Ratio
            mouth_w = float(np.linalg.norm(l_m - rm))
            mouth_ratio = mouth_w / ipd

            # Calibrated Multi-Biometric Composite
            score = (
                (log_chin - (-0.16)) * 1.6
                + (log_philtrum - (-0.16)) * 1.4
                + (norm_lip_contrast - 0.02) * 5.5
                - (mouth_ratio - 0.835) * 6.0
                - (stubble - 18.0) * 0.02
            )

            prob_female = float(1.0 / (1.0 + np.exp(-np.clip(score * 1.0, -4.0, 4.0))))
            if prob_female >= 0.52:
                conf = float(np.clip(prob_female, 0.55, 0.90))
                return "FEMALE", round(conf, 2)
            elif prob_female <= 0.48:
                conf = float(np.clip(1.0 - prob_female, 0.55, 0.90))
                return "MALE", round(conf, 2)
            else:
                return "UNKNOWN", 0.50

        # -------------------------------------------------------------
        # Mode 2: Crop-Only Anthropometric Fallback
        # -------------------------------------------------------------
        im = crop.resize((128, 128), Image.Resampling.BILINEAR)
        rgb = np.array(im.convert("RGB"), dtype=np.float32)
        gray = np.array(im.convert("L"), dtype=np.float32)

        # Hairless dermal bridge (rows 48-64, cols 52-76) and cheeks
        dorsum_lum = float(np.mean(gray[48:64, 52:76]))
        cheeks_lum = (float(np.mean(gray[52:72, 20:44])) + float(np.mean(gray[52:72, 84:108]))) / 2.0
        ref_skin_lum = (dorsum_lum + cheeks_lum) / 2.0

        chin_lum = float(np.mean(gray[92:108, 38:90]))
        log_chin = float(np.log((chin_lum + 10.0) / (ref_skin_lum + 10.0)))

        philtrum_lum = float(np.mean(gray[68:78, 52:76]))
        log_philtrum = float(np.log((philtrum_lum + 10.0) / (ref_skin_lum + 10.0)))

        # Lip erythema vs cheeks
        lip_region = rgb[72:88, 44:84]
        lip_ery = np.mean(lip_region[:, :, 0] - (lip_region[:, :, 1] + lip_region[:, :, 2]) * 0.5)
        cheek_ery = (
            np.mean(rgb[52:72, 20:44, 0] - (rgb[52:72, 20:44, 1] + rgb[52:72, 20:44, 2]) * 0.5)
            + np.mean(rgb[52:72, 84:108, 0] - (rgb[52:72, 84:108, 1] + rgb[52:72, 84:108, 2]) * 0.5)
        ) / 2.0
        norm_lip = float((lip_ery - cheek_ery) / (ref_skin_lum + 20.0))

        # Stubble texture
        chin_blur = np.array(im.convert("L").filter(ImageFilter.GaussianBlur(1.2)))[92:108, 38:90]
        stubble = float(np.std(gray[92:108, 38:90] - chin_blur))

        score = (
            (log_chin - (-0.16)) * 1.6
            + (log_philtrum - (-0.16)) * 1.4
            + (norm_lip - 0.02) * 5.5
            - (stubble - 5.5) * 0.08
        )
        prob_female = float(1.0 / (1.0 + np.exp(-np.clip(score * 1.0, -4.0, 4.0))))

        if prob_female >= 0.52:
            gender = "FEMALE"
            confidence = float(np.clip(prob_female, 0.55, 0.90))
        elif prob_female <= 0.48:
            gender = "MALE"
            confidence = float(np.clip(1.0 - prob_female, 0.55, 0.90))
        else:
            gender = "UNKNOWN"
            confidence = 0.50

        return gender, round(confidence, 2)

    except Exception as exc:
        logger.debug("Gender estimation encountered error: %s", exc)
        return "UNKNOWN", 0.0
