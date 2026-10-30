"""Presentation Attack Detection (PAD) and Interactive Active Liveness Challenge router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.dependencies import verify_api_key, check_rate_limit
from app.api.models import (
    PADResult,
    ChallengeSessionResponse,
    ChallengeVerifyRequest,
    ChallengeVerificationResult,
)
from app.services.validator import (
    validate_and_load_image,
    FileTooLargeError,
    UnsupportedFileTypeError,
    InvalidImageError,
)
from app.services.face_detector import get_face_detector
from app.anti_spoofing.pad_service import evaluate_pad
from app.anti_spoofing.challenge import get_challenge_manager

logger = logging.getLogger("facecheck.api.pad")

router = APIRouter(tags=["Presentation Attack Detection & Liveness"])


@router.post("/pad", response_model=PADResult)
async def pad_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Evaluate Presentation-Attack Detection across physical print, screen replay, and digital vectors."""
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

    detector = get_face_detector()
    crops = detector.detect_and_crop(pil_img)
    face_box = tuple(crops[0].box) if crops else None

    pad_res = evaluate_pad(pil_img, face_box)
    return pad_res


@router.post("/liveness/challenge/session", response_model=ChallengeSessionResponse)
def create_challenge_session_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Generate a single-use interactive liveness head-movement challenge sequence."""
    cm = get_challenge_manager()
    session = cm.create_session(sequence_length=3)
    return ChallengeSessionResponse(
        session_id=session.session_id,
        challenge_sequence=session.challenge_sequence,
        expires_at=session.expires_at,
    )


@router.post("/liveness/challenge/verify", response_model=ChallengeVerificationResult)
def verify_challenge_endpoint(
    payload: ChallengeVerifyRequest,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Verify submitted sequential head poses against assigned challenge movements."""
    cm = get_challenge_manager()
    result = cm.verify_poses(payload.session_id, payload.frame_poses)
    return result
