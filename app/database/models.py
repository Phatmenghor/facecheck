"""Pydantic data models for FaceCheck database component."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HumanReviewFeedback(BaseModel):
    """Payload representing an auditor's evaluation of a biometric decision."""

    sample_id: str = Field(..., description="Sample ID or session identifier (e.g., IMG_01)")
    filename: Optional[str] = Field(None, description="Image filename")
    scenario: Optional[str] = Field(None, description="Account opening scenario or category")
    sha256: Optional[str] = Field(None, description="SHA-256 hash of reviewed image")
    telemetry: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Raw biometric telemetry")
    llm_explanation: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initial LLM prediction")
    status: str = Field(..., description="'APPROVED' or 'ALERTED'")
    issue_type: Optional[str] = Field(
        None,
        description="Issue category if alerted: 'FALSE_POSITIVE', 'FALSE_NEGATIVE', 'DETECTION_BOX_MISALIGNED', 'LLM_EXPLANATION_INACCURATE', 'OTHER'",
    )
    notes: Optional[str] = Field(None, description="Auditor commentary and correction reasoning")
    auditor_id: str = Field(default="kyc_officer_core", description="Identifier of the human reviewer")


class LearningStats(BaseModel):
    """Real-time statistics on continuous learning database."""

    total_records: int
    approved_count: int
    alerted_count: int
    pending_fine_tuning_samples: int
    issue_breakdown: Dict[str, int]
    database_file: str
    database_size_bytes: int
    database_engine: str = "postgresql"
    last_updated: Optional[str] = None


class StoredImageRecord(BaseModel):
    """Metadata record for a binary image stored in PostgreSQL kyc_image_store."""

    sample_id: str
    filename: str
    mime_type: str = "image/jpeg"
    image_size_bytes: int
    sha256: Optional[str] = None
    scenario: Optional[str] = None
    target_category: Optional[str] = None
    created_at: Optional[str] = None
