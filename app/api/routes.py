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
from app.services.face_detector import get_face_detector
from app.detectors.factory import get_detector
from app.services.audit import write_audit_record
from app.services.encryption import get_encryption
from app.services.metrics import get_metrics
from app.services.sandbox import get_sandbox, SandboxTimeoutError, SandboxWorkerError

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
