"""Facial matching, 1:1 biometric comparison, and model inventory router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.api.dependencies import verify_api_key, check_rate_limit
from app.api.models import (
    EmbeddingResponse,
    FaceMatchResult,
    ModelInventoryItem,
    ModelsResponse,
)
from app.services.validator import (
    validate_and_load_image,
    FileTooLargeError,
    UnsupportedFileTypeError,
    InvalidImageError,
)
from app.services.face_detector import get_face_detector
from app.face_matching.matcher import compare_faces, format_raw_face_detection
from app.face_matching.models import get_face_recognizer
from app.detectors.factory import get_detector

logger = logging.getLogger("facecheck.api.match")

router = APIRouter(tags=["Facial Matching & Model Inventory"])


@router.post("/extract-embedding", response_model=EmbeddingResponse)
async def extract_embedding_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Extract 128-dimensional L2-normalized facial embedding using OpenCV SFace."""
    upload = image or file
    if upload is None or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Field 'image' or 'file' is required."},
        )

    raw_bytes = await upload.read()
    try:
        pil_img, _, _ = validate_and_load_image(upload.filename, raw_bytes)
    except (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError) as err:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if isinstance(err, FileTooLargeError) else (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if isinstance(err, UnsupportedFileTypeError) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail={"error": type(err).__name__, "message": str(err)})

    rec = get_face_recognizer()
    if not rec.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "MODEL_UNAVAILABLE", "message": "Face recognition model is not loaded offline."},
        )

    detector = get_face_detector()
    crops = detector.detect_and_crop(pil_img)
    if not crops:
        return EmbeddingResponse(
            faces_detected=0,
            embedding=None,
            box=None,
            landmarks=None,
            model_version=rec.model_name,
        )

    # Use primary face
    primary = crops[0]
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    raw_det = format_raw_face_detection(primary)
    emb = rec.extract_embedding(bgr, raw_det)

    return EmbeddingResponse(
        faces_detected=len(crops),
        embedding=[round(float(v), 6) for v in emb.tolist()],
        box=primary.box,
        landmarks=[[float(pt[0]), float(pt[1])] for pt in primary.landmarks],
        model_version=rec.model_name,
    )


@router.post("/match", response_model=FaceMatchResult)
async def match_faces_endpoint(
    request: Request,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Compare two facial photographs and determine if they belong to the same person."""
    if not image_a.filename or not image_b.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Both image_a and image_b must include filenames."},
        )

    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    try:
        pil_a, _, _ = validate_and_load_image(image_a.filename, bytes_a)
    except (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError) as err:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if isinstance(err, FileTooLargeError) else (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if isinstance(err, UnsupportedFileTypeError) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail={"error": "IMAGE_A_INVALID", "message": str(err)})

    try:
        pil_b, _, _ = validate_and_load_image(image_b.filename, bytes_b)
    except (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError) as err:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if isinstance(err, FileTooLargeError) else (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if isinstance(err, UnsupportedFileTypeError) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail={"error": "IMAGE_B_INVALID", "message": str(err)})

    match_result = compare_faces(pil_a, pil_b, custom_threshold=threshold)
    return match_result


@router.get("/models", response_model=ModelsResponse)
def list_models_endpoint(
    api_key_name: str = Depends(verify_api_key),
):
    """Inspect offline local model weights and active loading states."""
    project_root = Path(__file__).resolve().parent.parent.parent

    items: List[ModelInventoryItem] = []
    # YuNet
    yunet_p = project_root / "models" / "face_detection_yunet_2023mar.onnx"
    det = get_face_detector()
    items.append(ModelInventoryItem(
        name="YuNet Face Detector",
        category="detection",
        file_path="models/face_detection_yunet_2023mar.onnx",
        loaded=det.is_loaded,
        size_bytes=yunet_p.stat().st_size if yunet_p.is_file() else 0,
    ))

    # SFace
    sface_p = project_root / "models" / "face_recognition_sface_2021dec.onnx"
    rec = get_face_recognizer()
    items.append(ModelInventoryItem(
        name="SFace Face Recognizer",
        category="recognition",
        file_path="models/face_recognition_sface_2021dec.onnx",
        loaded=rec.is_loaded,
        size_bytes=sface_p.stat().st_size if sface_p.is_file() else 0,
    ))

    # Safetensors Classifier
    st_p = project_root / "models" / "deepfake_detector" / "model.safetensors"
    ai_det = get_detector()
    items.append(ModelInventoryItem(
        name="Forensic Feature Probe",
        category="pad_classifier",
        file_path="models/deepfake_detector/model.safetensors",
        loaded=ai_det.is_ready,
        size_bytes=st_p.stat().st_size if st_p.is_file() else 0,
    ))

    return ModelsResponse(models=items)
