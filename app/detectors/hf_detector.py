"""HuggingFace Transformers detector using local .safetensors files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from PIL import Image

from app.config import get_config
from app.detectors.base import BaseDetector

logger = logging.getLogger("facecheck.detectors.hf")


class HuggingFaceDetector(BaseDetector):
    """Detector using local Hugging Face model and processor."""

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        config = get_config()
        self.model_dir = model_dir or config.ai_detector.model_dir
        self.device_str = device or config.ai_detector.device
        self.processor = None
        self.model = None
        self.device = None
        self._is_ready = False
        self._load_offline_model()

    def _load_offline_model(self) -> None:
        """Load model strictly offline from local folder."""
        p = Path(self.model_dir)
        if not p.is_dir():
            logger.info("Model directory '%s' not present.", self.model_dir)
            return

        safetensors_path = p / "model.safetensors"
        if not safetensors_path.is_file():
            logger.warning("No model.safetensors found in '%s'. Refusing pickle weights.", self.model_dir)
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = torch.device(self.device_str if torch.cuda.is_available() and self.device_str == "cuda" else "cpu")
            self.processor = AutoImageProcessor.from_pretrained(
                str(p),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                str(p),
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_ready = True
            logger.info("Loaded Hugging Face safetensors model from %s on %s", p, self.device)
        except Exception as exc:
            logger.warning("Failed to load Hugging Face model: %s", exc)
            self.model = None
            self.processor = None
            self._is_ready = False

    @property
    def name(self) -> str:
        return "HuggingFace_Safetensors_Detector"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def predict(self, image: Image.Image) -> float:
        """Run classification on RGB image."""
        if not self._is_ready or self.model is None or self.processor is None:
            raise RuntimeError("Hugging Face detector is not loaded")

        import torch
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            fake_idx = 1
            if hasattr(self.model.config, "id2label"):
                for idx, lbl in self.model.config.id2label.items():
                    if any(term in str(lbl).lower() for term in ["fake", "ai", "synthetic"]):
                        fake_idx = int(idx)
                        break

            return float(probs[fake_idx].item())
