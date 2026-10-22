"""Pure-Python offline safetensors forensic detector.

Loads .safetensors weights directly without requiring PyTorch or external network access.
Extracts resolution-invariant forensic biometric features (dermal micro-texture,
sensor noise, optical spectrum roll-off, line-art edge ratio, color gradient continuity).
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageFilter

from app.config import get_config
from app.detectors.base import BaseDetector

logger = logging.getLogger("facecheck.detectors.safetensors")


def extract_forensic_features(image: Image.Image) -> np.ndarray:
    """Extract 6-dimensional scale-invariant forensic biometric feature vector.

    Always resizes image to canonical 256x256 resolution to ensure spatial frequency
    and gradient statistics are invariant to input camera resolution.

    Features:
    0. edge_noise_ratio: Line-art edge to noise ratio (high in cartoons/vector art).
    1. mean_noise: Natural dermal micro-texture and camera sensor grain (high in real photos).
    2. smooth_noise_std: High-frequency residual in smooth areas (CMOS/CCD photon noise).
    3. color_richness: Continuous color gradient diversity (high in real photos, low in quantized CGI).
    4. hf_energy: Canonical 2D FFT spectral energy (natural optical lens imaging).
    5. skin_ratio: Coverage of natural human melanin/hemoglobin chromaticity locus.
    """
    # Canonical scale normalization
    im_c = image.resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.array(im_c.convert("RGB"), dtype=np.float32)
    gray = np.array(im_c.convert("L"), dtype=np.float32)

    # 1. Edge-to-noise ratio
    edges = np.array(im_c.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    detail = np.array(im_c.convert("L").filter(ImageFilter.DETAIL), dtype=np.float32)
    noise = np.abs(detail - gray)
    edge_noise_ratio = float(np.mean(edges) / (np.mean(noise) + 1e-5))

    # 2. Mean high frequency noise (dermal pores / camera sensor grain)
    mean_noise = float(np.mean(noise))

    # 3. Sensor noise in smooth flat areas
    blur = np.array(im_c.convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    hp = np.abs(gray - blur)
    flat_mask = edges < 25.0
    smooth_noise_std = float(np.std(hp[flat_mask])) if np.sum(flat_mask) > 50 else float(np.std(hp))

    # 4. Color richness / palette diversity (real photos have higher continuous gradients)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    is_grayscale = bool((np.mean(np.abs(r - g)) < 3.0) and (np.mean(np.abs(g - b)) < 3.0))
    color_bins = (rgb[:, :, 0] // 8) * 1024 + (rgb[:, :, 1] // 8) * 32 + (rgb[:, :, 2] // 8)
    if is_grayscale:
        color_richness = float(len(np.unique(color_bins)) / 32.0 * 2.5)
    else:
        color_richness = float(len(np.unique(color_bins)) / 1024.0)

    # 5. Canonical FFT high-frequency spectral ratio
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f_shift)
    hf_energy = float(np.mean(mag[:32, :]) + np.mean(mag[-32:, :])) / (np.mean(mag) + 1e-5)

    # 6. Natural skin chromaticity coverage (melanin / hemoglobin biological absorption)
    if is_grayscale:
        skin_ratio = float(np.mean((gray > 40.0) & (gray < 225.0))) * 0.45
    else:
        tot = r + g + b + 1e-5
        nr, ng, nb = r / tot, g / tot, b / tot
        skin_mask = (r > g) & (g > b) & (nr > 0.35) & (nr < 0.65) & (ng > 0.24) & (ng < 0.40) & (r - g > 8)
        skin_ratio = float(np.mean(skin_mask))

    return np.array([
        edge_noise_ratio,
        mean_noise,
        smooth_noise_std,
        color_richness,
        hf_energy,
        skin_ratio,
    ], dtype=np.float32)


def load_safetensors_file(path: str | Path) -> Dict[str, np.ndarray]:
    """Parse .safetensors binary file into dictionary of numpy arrays."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Truncated safetensors header")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        raw_data = f.read()

    tensors = {}
    for name, info in header.items():
        if name.startswith("__"):
            continue
        start, end = info["data_offsets"]
        shape = info["shape"]
        t_bytes = raw_data[start:end]
        tensors[name] = np.frombuffer(t_bytes, dtype=np.float32).reshape(shape)

    return tensors


class SafetensorsClassifierDetector(BaseDetector):
    """Offline detector using local model.safetensors weights."""

    def __init__(self, model_dir: Optional[str] = None):
        config = get_config()
        self.model_dir = Path(model_dir or config.ai_detector.model_dir)
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._is_ready = False
        self._load_safetensors()

    def _load_safetensors(self) -> None:
        """Load weights and normalization parameters from model.safetensors."""
        weights_file = self.model_dir / "model.safetensors"
        config_file = self.model_dir / "config.json"

        if not weights_file.is_file():
            logger.info("No model.safetensors at %s", weights_file)
            return

        try:
            tensors = load_safetensors_file(weights_file)
            self.weights = tensors.get("weights")
            bias_arr = tensors.get("bias")
            self.bias = float(bias_arr[0]) if bias_arr is not None else 0.0

            if config_file.is_file():
                with open(config_file, "r", encoding="utf-8") as cf:
                    cfg_data = json.load(cf)
                    if "feature_mean" in cfg_data:
                        self.mean = np.array(cfg_data["feature_mean"], dtype=np.float32)
                    if "feature_std" in cfg_data:
                        self.std = np.array(cfg_data["feature_std"], dtype=np.float32)

            self._is_ready = True
            logger.info("Loaded SafetensorsClassifier from %s", weights_file)
        except Exception as exc:
            logger.warning("Failed to load SafetensorsClassifier: %s", exc)
            self._is_ready = False

    @property
    def name(self) -> str:
        return "Bank_Safetensors_Classifier"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def predict(self, image: Image.Image) -> float:
        """Run feature extraction and evaluate calibrated classifier."""
        if not self._is_ready or self.weights is None:
            raise RuntimeError("Safetensors model is not loaded")

        feats = extract_forensic_features(image)

        # Normalize features
        if self.mean is not None and self.std is not None:
            norm_feats = (feats - self.mean) / (self.std + 1e-6)
        else:
            norm_feats = feats

        logit = float(np.dot(self.weights, norm_feats) + self.bias)
        # Sigmoid activation with overflow protection
        clipped_logit = float(np.clip(logit, -20.0, 20.0))
        prob = 1.0 / (1.0 + np.exp(-clipped_logit))
        return float(min(max(prob, 0.01), 0.99))
