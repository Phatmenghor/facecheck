"""Continuous Learning & Active Memory Database for FaceCheck LLM Biometric Reviewer.

Stores human reviewer feedback, audit corrections, and biometric telemetry
in an ACID-compliant, persistent SQLite store (storage/learning_store.db).

Provides:
1. Active Memory / In-Context Learning: Dynamic Few-Shot retrieval of past auditor corrections.
2. Dataset Generation: Automated export of high-quality fine-tuning datasets (JSONL format).
3. Telemetry & Analytics: Real-time learning statistics on false-positive/negative rates.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("facecheck.learning_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "storage" / "learning_store.db"


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
    last_updated: Optional[str]


class LearningStore:
    """Thread-safe SQLite persistent store for LLM continuous learning & audit memory."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kyc_review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    filename TEXT,
                    scenario TEXT,
                    sha256 TEXT,
                    faces_found INTEGER DEFAULT 0,
                    quality_status TEXT,
                    laplacian_variance REAL,
                    mean_luminance REAL,
                    pad_decision TEXT,
                    pad_attack_type TEXT,
                    pad_risk_score INTEGER,
                    match_decision TEXT,
                    match_similarity REAL,
                    llm_verdict TEXT,
                    llm_risk_score INTEGER,
                    llm_summary TEXT,
                    llm_guidance TEXT,
                    human_status TEXT NOT NULL,
                    human_issue_type TEXT,
                    human_notes TEXT,
                    auditor_id TEXT DEFAULT 'kyc_officer_core',
                    telemetry_raw TEXT,
                    explanation_raw TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(sample_id)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_id ON kyc_review_events(sample_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_human_status ON kyc_review_events(human_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON kyc_review_events(scenario);")
            conn.commit()
            logger.info("Initialized continuous learning database at %s", self.db_path)

    def record_feedback(self, feedback: HumanReviewFeedback) -> Dict[str, Any]:
        """Record or update human auditor decision and notes."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t = feedback.telemetry or {}
        q_metrics = t.get("quality_metrics", {})
        exp = feedback.llm_explanation or {}

        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO kyc_review_events (
                    sample_id, filename, scenario, sha256,
                    faces_found, quality_status, laplacian_variance, mean_luminance,
                    pad_decision, pad_attack_type, pad_risk_score,
                    match_decision, match_similarity,
                    llm_verdict, llm_risk_score, llm_summary, llm_guidance,
                    human_status, human_issue_type, human_notes, auditor_id,
                    telemetry_raw, explanation_raw,
                    created_at, updated_at
                ) VALUES (
                    :sample_id, :filename, :scenario, :sha256,
                    :faces_found, :quality_status, :laplacian_variance, :mean_luminance,
                    :pad_decision, :pad_attack_type, :pad_risk_score,
                    :match_decision, :match_similarity,
                    :llm_verdict, :llm_risk_score, :llm_summary, :llm_guidance,
                    :human_status, :human_issue_type, :human_notes, :auditor_id,
                    :telemetry_raw, :explanation_raw,
                    :created_at, :updated_at
                )
                ON CONFLICT(sample_id) DO UPDATE SET
                    human_status = excluded.human_status,
                    human_issue_type = excluded.human_issue_type,
                    human_notes = excluded.human_notes,
                    auditor_id = excluded.auditor_id,
                    updated_at = excluded.updated_at;
            """, {
                "sample_id": feedback.sample_id,
                "filename": feedback.filename,
                "scenario": feedback.scenario,
                "sha256": feedback.sha256,
                "faces_found": t.get("faces_found", 0),
                "quality_status": t.get("quality_status", "OK"),
                "laplacian_variance": float(q_metrics.get("laplacian_variance", 0.0)),
                "mean_luminance": float(q_metrics.get("mean_luminance", 0.0)),
                "pad_decision": t.get("pad_decision"),
                "pad_attack_type": t.get("pad_attack_type"),
                "pad_risk_score": t.get("pad_risk_score", 0),
                "match_decision": t.get("match_decision"),
                "match_similarity": t.get("match_similarity"),
                "llm_verdict": exp.get("verdict"),
                "llm_risk_score": exp.get("risk_score", 0),
                "llm_summary": exp.get("summary"),
                "llm_guidance": exp.get("customer_guidance"),
                "human_status": feedback.status.upper(),
                "human_issue_type": feedback.issue_type,
                "human_notes": feedback.notes or "",
                "auditor_id": feedback.auditor_id,
                "telemetry_raw": json.dumps(t),
                "explanation_raw": json.dumps(exp),
                "created_at": now,
                "updated_at": now,
            })
            conn.commit()

        logger.info(f"Recorded review feedback for {feedback.sample_id}: {feedback.status}")
        return {"status": "SUCCESS", "sample_id": feedback.sample_id, "human_status": feedback.status}

    def get_few_shot_examples(self, scenario: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve high-confidence historical auditor decisions to inject into LLM prompt."""
        with self._lock, self._get_connection() as conn:
            if scenario:
                cursor = conn.execute("""
                    SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                           pad_attack_type, llm_verdict, human_status, human_notes, llm_guidance
                    FROM kyc_review_events
                    WHERE scenario LIKE ? AND human_status = 'APPROVED'
                    ORDER BY updated_at DESC LIMIT ?;
                """, (f"%{scenario}%", limit))
            else:
                cursor = conn.execute("""
                    SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                           pad_attack_type, llm_verdict, human_status, human_notes, llm_guidance
                    FROM kyc_review_events
                    WHERE human_status = 'APPROVED'
                    ORDER BY updated_at DESC LIMIT ?;
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def export_fine_tuning_dataset(self, output_path: Optional[Path] = None) -> Path:
        """Export all human-audited cases into OpenAI/Gemini JSONL supervised fine-tuning format."""
        target_file = Path(output_path or (PROJECT_ROOT / "storage" / "llm_biometric_finetune.jsonl"))
        target_file.parent.mkdir(parents=True, exist_ok=True)

        system_prompt = (
            "You are the Chief Biometric AI Officer for CP Bank digital account opening KYC verification. "
            "Analyze biometric computer vision telemetry and output structured JSON decisions with verdicts, "
            "actionable customer retake instructions, and NIST SP 800-63B / ISO 30107 regulatory compliance rationale."
        )

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                       pad_decision, pad_attack_type, pad_risk_score, match_decision, match_similarity,
                       llm_verdict, llm_risk_score, llm_summary, llm_guidance, human_status, human_notes
                FROM kyc_review_events
                ORDER BY id ASC;
            """)
            rows = cursor.fetchall()

        count = 0
        with open(target_file, "w", encoding="utf-8") as f:
            for r in rows:
                user_content = {
                    "sample_id": r["sample_id"],
                    "scenario": r["scenario"],
                    "faces_found": r["faces_found"],
                    "quality": {
                        "status": r["quality_status"],
                        "laplacian_variance": r["laplacian_variance"],
                        "mean_luminance": r["mean_luminance"],
                    },
                    "pad": {
                        "decision": r["pad_decision"],
                        "attack_type": r["pad_attack_type"],
                        "risk_score": r["pad_risk_score"],
                    },
                    "match": {
                        "decision": r["match_decision"],
                        "similarity": r["match_similarity"],
                    },
                }

                assistant_content = {
                    "verdict": r["llm_verdict"],
                    "risk_score": r["llm_risk_score"],
                    "summary": r["llm_summary"],
                    "customer_guidance": r["llm_guidance"],
                    "human_audited_assessment": r["human_notes"],
                    "compliance_alignment": "NIST SP 800-63B & ISO/IEC 30107-3",
                }

                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_content)},
                        {"role": "assistant", "content": json.dumps(assistant_content)},
                    ]
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info("Exported %d fine-tuning examples to %s", count, target_file)
        return target_file

    def get_learning_stats(self) -> LearningStats:
        """Calculate and return operational learning analytics."""
        with self._lock, self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM kyc_review_events;").fetchone()[0]
            approved = conn.execute("SELECT COUNT(*) FROM kyc_review_events WHERE human_status = 'APPROVED';").fetchone()[0]
            alerted = conn.execute("SELECT COUNT(*) FROM kyc_review_events WHERE human_status = 'ALERTED';").fetchone()[0]

            cursor = conn.execute("""
                SELECT human_issue_type, COUNT(*) as cnt
                FROM kyc_review_events
                WHERE human_issue_type IS NOT NULL AND human_issue_type != ''
                GROUP BY human_issue_type;
            """)
            breakdown = {row["human_issue_type"]: row["cnt"] for row in cursor.fetchall()}

            last_row = conn.execute("SELECT MAX(updated_at) FROM kyc_review_events;").fetchone()[0]

        size_bytes = self.db_path.stat().st_size if self.db_path.is_file() else 0

        return LearningStats(
            total_records=total,
            approved_count=approved,
            alerted_count=alerted,
            pending_fine_tuning_samples=total,
            issue_breakdown=breakdown,
            database_file=str(self.db_path.relative_to(PROJECT_ROOT)),
            database_size_bytes=size_bytes,
            last_updated=last_row,
        )


# Global singleton
_learning_store: Optional[LearningStore] = None


def get_learning_store() -> LearningStore:
    """Retrieve or create LearningStore singleton."""
    global _learning_store
    if _learning_store is None:
        _learning_store = LearningStore()
    return _learning_store
