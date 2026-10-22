"""Abstract base detector interface for AI-generation classification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from PIL import Image


class BaseDetector(ABC):
    """Abstract interface for AI image / deepfake detectors."""

    @abstractmethod
    def predict(self, image: Image.Image) -> float:
        """Analyze a PIL RGB image and return an AI probability score.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Float probability in [0.0, 1.0], where 1.0 = AI/Fake, 0.0 = Real.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector identifier string."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if detector is loaded and ready for offline inference."""
        pass
