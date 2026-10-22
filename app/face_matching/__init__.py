"""Face matching and verification package."""

from app.face_matching.align import align_face_5point
from app.face_matching.models import (
    BaseFaceRecognizer,
    SFaceRecognizer,
    UnavailableFaceRecognizer,
    get_face_recognizer,
    ModelUnavailableError,
)
from app.face_matching.matcher import FaceMatchResult, compare_faces

__all__ = [
    "align_face_5point",
    "BaseFaceRecognizer",
    "SFaceRecognizer",
    "UnavailableFaceRecognizer",
    "get_face_recognizer",
    "ModelUnavailableError",
    "FaceMatchResult",
    "compare_faces",
]
