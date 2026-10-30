"""Continuous learning, LLM biometric review, image repository, and daemon control router."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.api.dependencies import verify_api_key, check_rate_limit
from app.api.models import (
    FullReviewResponse,
    FeedbackResponse,
    ExportTrainingResponse,
    DaemonStatusResponse,
    LearningStats,
    HumanReviewFeedback,
)
from app.services.validator import (
    validate_and_load_image,
    FileTooLargeError,
    UnsupportedFileTypeError,
    InvalidImageError,
)
from app.services.face_detector import get_face_detector
from app.services.quality import inspect_image_quality
from app.anti_spoofing.pad_service import evaluate_pad
from app.face_matching.matcher import compare_faces
from app.services.llm_reviewer import (
    get_llm_reviewer,
    BiometricTelemetryInput,
)
from app.services.learning_store import get_learning_store
from app.services.continuous_daemon import get_continuous_daemon

logger = logging.getLogger("facecheck.api.learning")

router = APIRouter(tags=["Continuous Learning & AI Review"])


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
            logger.warning("Secondary match image failed validation: %s", exc)

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


@router.post("/review/feedback", response_model=FeedbackResponse)
def submit_review_feedback_endpoint(
    payload: HumanReviewFeedback,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Submit human reviewer feedback / alert to persist in the continuous learning database."""
    store = get_learning_store()
    res = store.record_feedback(payload)
    stats = store.get_learning_stats()
    return FeedbackResponse(
        status=res.get("status", "SUCCESS"),
        sample_id=payload.sample_id,
        human_status=payload.status,
        learning_stats=stats,
    )


@router.get("/review/learning-stats", response_model=LearningStats)
def get_learning_stats_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Inspect continuous learning database statistics and fine-tuning readiness."""
    store = get_learning_store()
    return store.get_learning_stats()


@router.post("/review/export-training-data", response_model=ExportTrainingResponse)
def export_training_data_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Export accumulated human review records into an instruction-tuning JSONL dataset for LLM fine-tuning."""
    store = get_learning_store()
    exported_path = store.export_fine_tuning_dataset()
    stats = store.get_learning_stats()
    return ExportTrainingResponse(
        status="SUCCESS",
        exported_file=str(exported_path),
        samples_count=stats.total_records,
    )


@router.get("/learning/daemon/status", response_model=DaemonStatusResponse)
def get_daemon_status_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Inspect continuous learning daemon status, heartbeats, and database connectivity."""
    daemon = get_continuous_daemon()
    stats = daemon.store.get_learning_stats()
    return DaemonStatusResponse(
        daemon_id=daemon.daemon_id,
        status="ACTIVE" if daemon.running else "STANDBY",
        is_postgresql=daemon.store.is_postgresql,
        iterations_count=daemon.iterations_count,
        training_runs_count=daemon.training_runs_count,
        online_streamed_count=daemon.online_streamed_total,
        online_stream_enabled=daemon.online_stream_enabled,
        interval_seconds=daemon.interval_sec,
        learning_stats=stats,
    )


@router.post("/learning/stream-ingest")
def trigger_online_stream_endpoint(
    batch_size: int = 3,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Stream face images dynamically from the internet in memory, validate, auto-review with AI, and record in PostgreSQL."""
    daemon = get_continuous_daemon()
    ingested = daemon.ingest_from_online_stream(batch_size=batch_size)
    stats = daemon.store.get_learning_stats()
    return {
        "status": "SUCCESS",
        "ingested_count": ingested,
        "total_records": stats.total_records,
        "total_images_in_db": len(daemon.store.list_stored_images()),
    }


@router.post("/learning/auto-train")
def trigger_auto_train_endpoint(
    stream_count: int = 5,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Autonomous End-to-End Cycle: Ingest internet stream + AI auto-review + export fine-tune dataset + retrain model + online sync."""
    daemon = get_continuous_daemon()
    # 1. Stream from internet directly into PostgreSQL
    ingested = daemon.ingest_from_online_stream(batch_size=stream_count)
    # 2. Export fine-tuning dataset
    exp_file = daemon.store.export_fine_tuning_dataset()
    # 3. Retrain model
    trained, metrics = daemon.trigger_model_training()
    # 4. Trigger online sync
    from app.services.llm_reviewer import get_llm_reviewer
    sync_res = get_llm_reviewer().trigger_online_training_sync(exp_file)
    stats = daemon.store.get_learning_stats()
    return {
        "status": "SUCCESS",
        "streamed_and_reviewed": ingested,
        "model_retrained": trained,
        "training_metrics": metrics,
        "fine_tune_dataset": str(exp_file),
        "online_sync": sync_res,
        "total_learning_records": stats.total_records,
    }


@router.post("/learning/daemon/trigger")
def trigger_daemon_cycle_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Trigger an immediate continuous learning ingestion, fine-tuning export, and catalog cycle."""
    daemon = get_continuous_daemon()
    res = daemon.run_single_iteration()
    return res


@router.post("/learning/daemon/train")
def trigger_daemon_training_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Trigger an immediate automated model retraining session."""
    daemon = get_continuous_daemon()
    trained, metrics = daemon.trigger_model_training()
    return {
        "status": "SUCCESS" if trained else "FAILED",
        "trained": trained,
        "metrics": metrics,
    }


@router.post("/learning/auto-review")
async def auto_review_endpoint(
    image: UploadFile = File(...),
    scenario: str = "Digital Account Opening Auto-Review",
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Autonomous AI Reviewer: inspects biometric image and auto-records verdict into PostgreSQL."""
    raw_bytes = await image.read()
    reviewer = get_llm_reviewer()
    result = reviewer.auto_review_image(
        image_input=raw_bytes,
        sample_id=f"auto_{uuid.uuid4().hex[:8]}",
        scenario=scenario,
        auto_record_to_db=True,
    )
    return result


@router.post("/learning/online-sync")
def trigger_online_sync_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Export instruction fine-tuning dataset and dispatch to online LLM training API."""
    reviewer = get_llm_reviewer()
    res = reviewer.trigger_online_training_sync()
    return res


@router.post("/learning/purge")
def purge_invalid_samples_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Purge corrupted files and bad payloads from the active training candidate catalog."""
    daemon = get_continuous_daemon()
    res = daemon.purge_invalid_samples()
    return res


@router.get("/learning/online-training/jobs")
def get_online_training_jobs_endpoint(
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Retrieve history of online training sessions, fine-tuning jobs, and convergence logs."""
    reviewer = get_llm_reviewer()
    jobs = reviewer.get_online_training_jobs()
    return {
        "status": "SUCCESS",
        "provider": reviewer.provider,
        "model": reviewer.model,
        "total_jobs": len(jobs),
        "jobs": jobs,
    }


@router.get("/learning/images/{sample_id}")
def get_stored_image_endpoint(
    sample_id: str,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Stream raw biometric image directly from PostgreSQL database."""
    store = get_learning_store()
    img_record = store.get_image(sample_id)
    if not img_record or not img_record.get("image_bytes"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "IMAGE_NOT_FOUND", "message": f"Sample {sample_id} not found in database."},
        )
    return Response(
        content=img_record["image_bytes"],
        media_type=img_record.get("mime_type", "image/jpeg"),
        headers={
            "Cache-Control": "public, max-age=86400",
            "ETag": f'"{img_record.get("sha256", sample_id)}"',
            "Content-Disposition": f'inline; filename="{img_record.get("filename", sample_id + ".jpg")}"',
        },
    )


@router.get("/learning/images")
def list_stored_images_endpoint(
    category: Optional[str] = None,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """List metadata for all biometric images stored in PostgreSQL."""
    store = get_learning_store()
    images = store.list_stored_images(target_category=category)
    return {
        "status": "SUCCESS",
        "database_engine": "postgresql" if store.is_postgresql else "sqlite",
        "total_images": len(images),
        "images": images,
    }


@router.post("/learning/images/migrate")
def migrate_local_images_endpoint(
    delete_local: bool = True,
    api_key_name: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
):
    """Migrate all local folder images into PostgreSQL database and purge local files."""
    store = get_learning_store()
    res = store.migrate_local_images_to_db(delete_local_files=delete_local)
    return res
