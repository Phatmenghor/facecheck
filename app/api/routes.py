"""Endpoints for FaceCheck API with Quality Analysis, Hardening, and Prometheus Metrics."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.config import get_config
from app.api.dependencies import verify_api_key, check_rate_limit, get_client_ip
from app.services.pipeline import run_pipeline
from app.services.validator import (
    validate_and_load_image,
    FileTooLargeError,
    UnsupportedFileTypeError,
    InvalidImageError,
)
from app.services.face_detector import get_face_detector, FaceCrop
from app.detectors.factory import get_detector
from app.services.audit import write_audit_record
from app.services.encryption import get_encryption
from app.services.metrics import get_metrics
from app.services.sandbox import get_sandbox, SandboxTimeoutError, SandboxWorkerError
from app.services.quality import inspect_image_quality
from app.face_matching.matcher import compare_faces, FaceMatchResult, format_raw_face_detection
from app.face_matching.models import get_face_recognizer
from app.anti_spoofing.pad_service import evaluate_pad, PADResult
from app.anti_spoofing.challenge import (
    get_challenge_manager,
    ChallengeSession,
    ChallengeVerificationResult,
)
from app.services.llm_reviewer import (
    get_llm_reviewer,
    BiometricTelemetryInput,
    BiometricReviewExplanation,
)

logger = logging.getLogger("facecheck.api")

router = APIRouter(prefix="/api/v1", tags=["FaceCheck"])


class FaceDetail(BaseModel):
    box: List[int] = Field(..., description="[x, y, w, h] coordinates")
    score: int = Field(..., description="Risk score 1-100 for this face crop")
    gender: str = Field(default="UNKNOWN", description="'MALE', 'FEMALE', or 'UNKNOWN'")
    gender_confidence: float = Field(default=0.0, description="Confidence score (0.0 to 1.0)")


class MetadataSignal(BaseModel):
    score: int
    detail: str


class DetectorScore(BaseModel):
    score: Optional[int]


class SignalsPayload(BaseModel):
    metadata: MetadataSignal
    detector_face: DetectorScore
    detector_full_image: DetectorScore


class QualityDetail(BaseModel):
    status: str = Field(..., description="'OK' or 'LOW_QUALITY'")
    issues: List[str] = Field(default_factory=list, description="Quality issue warnings")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic image quality metrics")


class FaceCheckDetail(BaseModel):
    status: str = Field(..., description="'PASSED' or 'FAILED'")
    passed: bool
    reason_codes: List[str] = Field(default_factory=list, description="KYC face check reason codes")
    retake_message: Optional[str] = Field(None, description="Actionable guidance for applicant retake")
    faces_detected: int = 0
    head_pose: Optional[Dict[str, float]] = Field(None, description="Head pose angles: yaw, pitch, roll in degrees")
    visibility: Dict[str, bool] = Field(default_factory=dict, description="Face visibility & boundary clipping flags")


class CheckResponse(BaseModel):
    request_id: str
    score: Optional[int] = Field(None, ge=1, le=100, description="1=Real, 100=Fake, null if RETAKE/NO_FACE")
    label: str = Field(..., description="'LIKELY_REAL', 'UNCERTAIN', 'LIKELY_FAKE', 'RETAKE', 'NO_FACE_FOUND'")
    reason_codes: List[str] = Field(default_factory=list, description="Reason codes e.g. NO_FACE, MULTIPLE_FACES, BLURRY, etc.")
    face_check: FaceCheckDetail = Field(..., description="Pre-flight face check results")
    gender: Optional[str] = Field(None, description="Primary detected gender: 'MALE', 'FEMALE', or 'UNKNOWN' (null if no face)")
    gender_confidence: Optional[float] = Field(None, description="Confidence score for primary gender estimation (0.0 to 1.0)")
    faces_found: int
    faces: List[FaceDetail]
    signals: SignalsPayload
    quality: QualityDetail
    reasons: List[str]
    disclaimer: str = "Risk indicator only, not final proof."


class HealthResponse(BaseModel):
    status: str
    offline_mode: bool
    face_detector_loaded: bool
    ai_detector_loaded: bool
    ai_detector_type: str
    device: str
    timestamp: str


@router.post("/check", response_model=CheckResponse)
async def check_image_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Analyze an uploaded image or external image URL and return an AI deepfake risk score.

    Supports:
    - Direct multipart file upload ('image' or 'file')
    - External online image URL download ('image_url')
    - Pre-flight face verification, quality assessment, and biometric gender estimation.
    """
    upload = image or file
    if upload is None and not image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Field 'image', 'file', or 'image_url' is required."},
        )

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    client_ip = get_client_ip(request)
    metrics = get_metrics()

    if upload is not None:
        if not upload.filename:
            metrics.record_error("MISSING_FILENAME")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "BAD_REQUEST", "message": "Uploaded file must include a valid filename."},
            )
        filename = upload.filename
        try:
            raw_bytes = await upload.read()
        except Exception as exc:
            metrics.record_error("PAYLOAD_READ_FAILED")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "PAYLOAD_READ_FAILED", "message": f"Could not read upload stream: {exc}"},
            )
    else:
        # Download from external online URL
        import urllib.request
        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0 FaceCheck/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                raw_bytes = resp.read()
        except Exception as exc:
            metrics.record_error("ONLINE_DOWNLOAD_FAILED")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "ONLINE_DOWNLOAD_FAILED", "message": f"Failed to download image from external URL: {exc}"},
            )
        from app.services.validator import detect_mime_from_magic_bytes
        ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
        try:
            m = detect_mime_from_magic_bytes(raw_bytes)
            ext = ext_map.get(m, ".jpg")
        except Exception:
            ext = ".jpg"
        filename = f"online_{uuid.uuid4().hex[:8]}{ext}"

    # Hardened, non-blocking pipeline execution inside worker thread with timeout
    config = get_config()
    timeout = config.hardening.sandbox_timeout_seconds

    try:
        pipeline_res = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline, filename, raw_bytes),
            timeout=timeout,
        )
    except FileTooLargeError as err:
        metrics.record_error("PAYLOAD_TOO_LARGE")
        metrics.record_request(413, api_key_name, time.perf_counter() - start_time)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "PAYLOAD_TOO_LARGE", "message": str(err)},
        )
    except UnsupportedFileTypeError as err:
        metrics.record_error("UNSUPPORTED_MEDIA_TYPE")
        metrics.record_request(415, api_key_name, time.perf_counter() - start_time)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "UNSUPPORTED_MEDIA_TYPE", "message": str(err)},
        )
    except InvalidImageError as err:
        metrics.record_error("INVALID_IMAGE")
        metrics.record_request(400, api_key_name, time.perf_counter() - start_time)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": str(err)},
        )
    except (asyncio.TimeoutError, SandboxTimeoutError) as err:
        elapsed = time.perf_counter() - start_time
        metrics.record_error("SANDBOX_TIMEOUT")
        metrics.record_request(504, api_key_name, elapsed)
        logger.error("Processing timed out for request %s: %s", request_id, err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": "GATEWAY_TIMEOUT", "message": "Image processing timed out."},
        )
    except SandboxWorkerError as err:
        elapsed = time.perf_counter() - start_time
        metrics.record_error("SANDBOX_WORKER_CRASH")
        metrics.record_request(500, api_key_name, elapsed)
        logger.critical("Worker crashed for request %s: %s", request_id, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": "Analysis worker process failed safely."},
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        metrics.record_error("INTERNAL_ERROR")
        metrics.record_request(500, api_key_name, elapsed)
        logger.error("Error processing check request %s: %s", request_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_SERVER_ERROR", "message": "An internal error occurred during image processing."},
        )

    # Record quality metrics
    if pipeline_res["quality"]["status"] == "LOW_QUALITY":
        metrics.record_low_quality()

    # Optional encryption with metadata stripping
    encryption_service = get_encryption()
    encryption_service.save_encrypted(raw_bytes, request_id)

    # Tamper-evident audit log
    fused_score = pipeline_res["score"]
    label = pipeline_res["label"]
    write_audit_record(
        request_id=request_id,
        api_key_name=api_key_name,
        score=fused_score,
        label=label,
        client_ip=client_ip,
    )

    # Record metrics
    elapsed = time.perf_counter() - start_time
    metrics.record_request(200, api_key_name, elapsed)
    metrics.record_score(fused_score)

    faces_list = [
        FaceDetail(
            box=f["box"],
            score=f["score"],
            gender=f.get("gender", "UNKNOWN"),
            gender_confidence=f.get("gender_confidence", 0.0),
        )
        for f in pipeline_res["faces"]
    ]

    return CheckResponse(
        request_id=request_id,
        score=fused_score,
        label=label,
        reason_codes=pipeline_res.get("reason_codes", []),
        face_check=FaceCheckDetail(**pipeline_res.get("face_check", {
            "status": "PASSED",
            "passed": True,
            "reason_codes": [],
            "retake_message": None,
            "faces_detected": pipeline_res.get("faces_found", 0),
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "visibility": {"partly_hidden": False, "boundary_clipped": False},
        })),
        gender=pipeline_res.get("gender"),
        gender_confidence=pipeline_res.get("gender_confidence"),
        faces_found=pipeline_res["faces_found"],
        faces=faces_list,
        signals=SignalsPayload(
            metadata=MetadataSignal(
                score=pipeline_res["signals"]["metadata"]["score"],
                detail=pipeline_res["signals"]["metadata"]["detail"],
            ),
            detector_face=DetectorScore(score=pipeline_res["signals"]["detector_face"]["score"]),
            detector_full_image=DetectorScore(score=pipeline_res["signals"]["detector_full_image"]["score"]),
        ),
        quality=QualityDetail(
            status=pipeline_res["quality"]["status"],
            issues=pipeline_res["quality"]["issues"],
            metrics=pipeline_res["quality"]["metrics"],
        ),
        reasons=pipeline_res["reasons"],
        disclaimer="Risk indicator only, not final proof.",
    )


class CheckUrlRequest(BaseModel):
    image_url: str = Field(..., description="External image HTTP/HTTPS URL")


@router.post("/check-url", response_model=CheckResponse)
async def check_url_endpoint(
    payload: CheckUrlRequest,
    request: Request,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Download an external image directly from a URL and run banking verification."""
    return await check_image_endpoint(
        request=request,
        image=None,
        file=None,
        image_url=payload.image_url,
        api_key_name=api_key_name,
        _=None,
    )


@router.get("/metrics")
def get_metrics_endpoint(
    api_key_name: str = Depends(verify_api_key),
):
    """Expose application metrics in Prometheus text exposition format."""
    metrics = get_metrics()
    text = metrics.export_metrics_text()
    return Response(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Verify service health, model load status, and offline guarantee."""
    config = get_config()
    face_det = get_face_detector()
    ai_det = get_detector()

    return HealthResponse(
        status="UP",
        offline_mode=config.server.offline_mode,
        face_detector_loaded=face_det.is_loaded,
        ai_detector_loaded=ai_det.is_ready,
        ai_detector_type=ai_det.name,
        device=config.ai_detector.device,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


# =============================================================================
# V1 FACE & BIOMETRIC VERIFICATION ENDPOINTS
# =============================================================================

class FaceBoxDetail(BaseModel):
    box: List[int] = Field(..., description="[x, y, w, h] coordinates")
    confidence: float = Field(..., description="Face detector confidence score")
    landmarks: List[List[float]] = Field(default_factory=list, description="5 facial landmarks [[x,y], ...]")


class DetectResponse(BaseModel):
    faces_found: int
    faces: List[FaceBoxDetail]
    image_width: int
    image_height: int


@router.post("/detect", response_model=DetectResponse)
async def detect_faces_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Detect faces and extract 5 facial landmarks using offline OpenCV YuNet."""
    upload = image or file
    if upload is None or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Field 'image' or 'file' with filename is required."},
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
    w, h = pil_img.size

    boxes = [
        FaceBoxDetail(
            box=c.box,
            confidence=c.confidence,
            landmarks=[[float(pt[0]), float(pt[1])] for pt in c.landmarks],
        )
        for c in crops
    ]
    return DetectResponse(faces_found=len(boxes), faces=boxes, image_width=w, image_height=h)


class QualityResponse(BaseModel):
    status: str = Field(..., description="'OK' or 'LOW_QUALITY'")
    issues: List[str]
    metrics: Dict[str, Any]


@router.post("/quality", response_model=QualityResponse)
async def quality_check_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Evaluate image quality against KYC biometric standards (blur, luminance, compression)."""
    upload = image or file
    if upload is None or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Field 'image' or 'file' is required."},
        )

    raw_bytes = await upload.read()
    try:
        pil_img, _, size_bytes = validate_and_load_image(upload.filename, raw_bytes)
    except (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError) as err:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if isinstance(err, FileTooLargeError) else (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if isinstance(err, UnsupportedFileTypeError) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail={"error": type(err).__name__, "message": str(err)})

    detector = get_face_detector()
    crops = detector.detect_and_crop(pil_img)
    face_boxes = [c.box for c in crops]
    q_res = inspect_image_quality(pil_img, size_bytes, face_boxes)

    return QualityResponse(status=q_res.status, issues=q_res.issues, metrics=q_res.metrics)


class EmbeddingResponse(BaseModel):
    faces_detected: int
    embedding: Optional[List[float]] = Field(None, description="128-dimensional L2-normalized embedding vector")
    box: Optional[List[int]] = Field(None, description="Bounding box [x, y, w, h]")
    landmarks: Optional[List[List[float]]] = None
    model_version: str


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
    import cv2
    import numpy as np
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


class ChallengeSessionResponse(BaseModel):
    session_id: str
    challenge_sequence: List[str]
    expires_at: float


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


class ChallengeVerifyRequest(BaseModel):
    session_id: str
    frame_poses: List[Dict[str, float]] = Field(..., description="List of observed head poses [{yaw, pitch, roll}]")


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


class ModelInventoryItem(BaseModel):
    name: str
    category: str
    file_path: str
    loaded: bool
    size_bytes: int


class ModelsResponse(BaseModel):
    models: List[ModelInventoryItem]


@router.get("/models", response_model=ModelsResponse)
def list_models_endpoint(
    api_key_name: str = Depends(verify_api_key),
):
    """Inspect offline local model weights and active loading states."""
    from pathlib import Path
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


class FullReviewResponse(BaseModel):
    telemetry: BiometricTelemetryInput
    explanation: BiometricReviewExplanation


@router.post("/review/explain", response_model=FullReviewResponse)
async def review_explain_endpoint(
    request: Request,
    payload: Optional[BiometricTelemetryInput] = None,
    image: Optional[UploadFile] = File(None),
    match_image: Optional[UploadFile] = File(None),
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Generate executive LLM biometric review explanations and customer retake guidance.

    Accepts either precomputed telemetry via JSON body or raw image uploads.
    Supports dual-mode: external LLM when configured, or air-gapped deterministic biometric reasoner.
    """
    reviewer = get_llm_reviewer()
    content_type = request.headers.get("content-type", "")

    # Mode A: JSON Body (telemetry precomputed)
    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = BiometricTelemetryInput(**body)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "INVALID_JSON", "message": f"Malformed JSON telemetry: {exc}"},
            )
        explanation = reviewer.review(payload)
        return FullReviewResponse(telemetry=payload, explanation=explanation)

    if payload is not None:
        explanation = reviewer.review(payload)
        return FullReviewResponse(telemetry=payload, explanation=explanation)

    # Mode B: Direct image upload analysis
    if image is None or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "BAD_REQUEST", "message": "Either JSON telemetry body or 'image' file upload is required."},
        )

    raw_bytes = await image.read()
    try:
        pil_img, _, size_bytes = validate_and_load_image(image.filename, raw_bytes)
    except (FileTooLargeError, UnsupportedFileTypeError, InvalidImageError) as err:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if isinstance(err, FileTooLargeError) else (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if isinstance(err, UnsupportedFileTypeError) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail={"error": type(err).__name__, "message": str(err)})

    detector = get_face_detector()
    crops = detector.detect_and_crop(pil_img)
    face_boxes = [c.box for c in crops]

    q_res = inspect_image_quality(pil_img, size_bytes, face_boxes)
    pad_res = evaluate_pad(pil_img, face_boxes[0] if face_boxes else None)

    match_res = None
    if match_image is not None and match_image.filename:
        match_bytes = await match_image.read()
        try:
            match_pil, _, _ = validate_and_load_image(match_image.filename, match_bytes)
            match_res = compare_faces(pil_img, match_pil)
        except Exception as exc:
            logger.warning(f"Secondary match image failed validation: {exc}")

    telemetry = BiometricTelemetryInput(
        filename=image.filename,
        faces_found=len(crops),
        face_boxes=face_boxes,
        quality_status=q_res.status,
        quality_issues=q_res.issues,
        quality_metrics=q_res.metrics,
        pad_decision=pad_res.decision,
        pad_attack_type=pad_res.attack_type,
        pad_spoof_prob=pad_res.spoof_probability,
        pad_risk_score=pad_res.calibrated_risk,
        match_decision=match_res.decision if match_res else None,
        match_similarity=match_res.similarity_score if match_res else None,
        match_threshold=match_res.threshold if match_res else None,
    )

    explanation = reviewer.review(telemetry)
    return FullReviewResponse(telemetry=telemetry, explanation=explanation)


