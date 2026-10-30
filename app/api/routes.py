"""Master router aggregating all FaceCheck component sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

# Import sub-routers
from app.api.check_router import router as check_router
from app.api.match_router import router as match_router
from app.api.pad_router import router as pad_router
from app.api.learning_router import router as learning_router

# Re-export all models for backward compatibility
from app.api.models import (  # noqa: F401
    FaceDetail,
    MetadataSignal,
    DetectorScore,
    SignalsPayload,
    QualityDetail,
    FaceCheckDetail,
    CheckResponse,
    HealthResponse,
    CheckUrlRequest,
    FaceBoxDetail,
    DetectResponse,
    QualityResponse,
    EmbeddingResponse,
    ChallengeSessionResponse,
    ChallengeVerifyRequest,
    ModelInventoryItem,
    ModelsResponse,
    FullReviewResponse,
    FeedbackResponse,
    ExportTrainingResponse,
    DaemonStatusResponse,
    LearningStats,
    HumanReviewFeedback,
    FaceMatchResult,
    PADResult,
    ChallengeVerificationResult,
)

# Master API Router with /api/v1 prefix
router = APIRouter(prefix="/api/v1")

# Mount modular component routers
router.include_router(check_router)
router.include_router(match_router)
router.include_router(pad_router)
router.include_router(learning_router)
