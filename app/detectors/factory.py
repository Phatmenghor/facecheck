"""Factory for AI detector engines."""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_config
from app.detectors.base import BaseDetector
from app.detectors.hf_detector import HuggingFaceDetector
from app.detectors.onnx_detector import OnnxDetector
from app.detectors.mock_detector import MockDetector

logger = logging.getLogger("facecheck.detectors.factory")

_detector_singleton: Optional[BaseDetector] = None


def get_detector(force_new: bool = False) -> BaseDetector:
    """Retrieve or initialize the active detector instance."""
    global _detector_singleton
    if _detector_singleton is not None and not force_new:
        return _detector_singleton

    config = get_config()
    active_type = config.ai_detector.active_detector.lower()

    detector: BaseDetector

    if active_type in ("huggingface", "safetensors"):
        from app.detectors.safetensors_detector import SafetensorsClassifierDetector
        detector = SafetensorsClassifierDetector()
        if not detector.is_ready:
            detector = HuggingFaceDetector()
        if not detector.is_ready and config.ai_detector.allow_fallback_to_mock:
            logger.info("Using Forensic Heuristic Detector.")
            detector = MockDetector()
    elif active_type == "onnx":
        detector = OnnxDetector()
        if not detector.is_ready and config.ai_detector.allow_fallback_to_mock:
            logger.info("ONNX model not on disk. Using MockDetector.")
            detector = MockDetector()
    elif active_type == "mock":
        detector = MockDetector()
    else:
        raise ValueError(f"Unknown detector type: {active_type}")

    _detector_singleton = detector
    return _detector_singleton
