"""ONNX Runtime detector for offline classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from app.config import get_config
from app.detectors.base import BaseDetector

logger = logging.getLogger("facecheck.detectors.onnx")


class OnnxDetector(BaseDetector):
    """Deepfake detector running locally via ONNX Runtime."""

    def __init__(self, onnx_path: Optional[str] = None):
        config = get_config()
        self.onnx_path = onnx_path or config.ai_detector.onnx_model_path
        self.session = None
        self._is_ready = False
        self._load_session()

    def _load_session(self) -> None:
        """Load ONNX file locally."""
        p = Path(self.onnx_path)
        if not p.is_file():
            logger.info("ONNX model file not found at '%s'.", self.onnx_path)
            return

        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(p), providers=providers)
            self._is_ready = True
            logger.info("Loaded ONNX detector from %s", p)
        except Exception as exc:
            logger.warning("Failed to load ONNX model: %s", exc)
            self.session = None
            self._is_ready = False

    @property
    def name(self) -> str:
        return "ONNX_Deepfake_Detector"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def predict(self, image: Image.Image) -> float:
        """Run preprocessing and inference on image."""
        if not self._is_ready or self.session is None:
            raise RuntimeError("ONNX detector is not loaded")

        resized = image.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        input_tensor = np.expand_dims(arr, axis=0)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        if len(logits) > 1:
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            fake_prob = float(probs[1])
        else:
            fake_prob = float(1.0 / (1.0 + np.exp(-logits[0])))

        return min(max(fake_prob, 0.0), 1.0)
