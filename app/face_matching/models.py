"""Face embedding models for FaceCheck verification."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from app.face_matching.align import align_face_5point

logger = logging.getLogger("facecheck.face_matching.models")


class ModelUnavailableError(Exception):
    """Raised when face recognition model weights are unavailable or failed to load."""
    pass


class BaseFaceRecognizer(ABC):
    """Abstract interface for face feature embedding models."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded and ready for inference."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name and version of active recognition model."""
        pass

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Calibrated decision threshold for acceptance."""
        pass

    @abstractmethod
    def extract_embedding(
        self,
        bgr_image: np.ndarray,
        face_box_and_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract L2-normalized 1D face embedding vector from image."""
        pass

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two normalized embeddings.

        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        # Ensure L2 normalization
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        e1 = emb1 / norm1
        e2 = emb2 / norm2
        cos_sim = float(np.dot(e1.flatten(), e2.flatten()))
        return max(-1.0, min(1.0, cos_sim))


class SFaceRecognizer(BaseFaceRecognizer):
    """OpenCV SFace deep neural network recognizer (Apache-2.0).

    Uses MobileFaceNet/ResNet backbone with CosFace loss trained on large-scale face datasets.
    Extracts 128-dimensional L2-normalized embedding.
    """

    def __init__(self, model_path: Optional[str | Path] = None):
        self._model_path = Path(model_path or "models/face_recognition_sface_2021dec.onnx")
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._model_path.is_file():
            try:
                self._recognizer = cv2.FaceRecognizerSF.create(str(self._model_path), "")
                logger.info("Loaded SFace face recognizer from %s", self._model_path)
            except Exception as exc:
                logger.warning("Failed to initialize cv2.FaceRecognizerSF: %s", exc)
                self._recognizer = None
        else:
            logger.info("SFace model file not present at %s", self._model_path)
            self._recognizer = None

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def model_name(self) -> str:
        return "OpenCV-SFace-2021dec"

    @property
    def default_threshold(self) -> float:
        # Standard SFace cosine threshold calibrated for FAR <= 0.1% on LFW/CFP-FP
        return 0.363

    def extract_embedding(
        self,
        bgr_image: np.ndarray,
        face_box_and_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.is_loaded:
            raise ModelUnavailableError(
                f"Face recognition model is unavailable at {self._model_path}"
            )

        with self._lock:
            if face_box_and_landmarks is not None:
                # OpenCV FaceRecognizerSF native alignment
                try:
                    aligned = self._recognizer.alignCrop(bgr_image, face_box_and_landmarks)
                except Exception:
                    aligned = cv2.resize(bgr_image, (112, 112))
            else:
                aligned = cv2.resize(bgr_image, (112, 112))

            feature = self._recognizer.feature(aligned)
        feature = feature.flatten()
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm
        return feature.astype(np.float32)


class UnavailableFaceRecognizer(BaseFaceRecognizer):
    """Placeholder when recognition model is deliberately omitted or missing."""

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "Unavailable-Model"

    @property
    def default_threshold(self) -> float:
        return 0.50

    def extract_embedding(
        self,
        bgr_image: np.ndarray,
        face_box_and_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise ModelUnavailableError("No face recognition model is configured or loaded.")


_face_recognizer_singleton: Optional[BaseFaceRecognizer] = None


def get_face_recognizer(force_reload: bool = False) -> BaseFaceRecognizer:
    """Retrieve or initialize singleton Face Recognizer."""
    global _face_recognizer_singleton
    if _face_recognizer_singleton is not None and not force_reload:
        return _face_recognizer_singleton

    sface = SFaceRecognizer()
    if sface.is_loaded:
        _face_recognizer_singleton = sface
    else:
        _face_recognizer_singleton = UnavailableFaceRecognizer()

    return _face_recognizer_singleton
