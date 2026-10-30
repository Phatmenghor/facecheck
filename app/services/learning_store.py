"""Continuous Learning & Active Memory Database for FaceCheck LLM Biometric Reviewer.

Supports:
1. PostgreSQL 15.6 enterprise database for distributed banking continuous learning.
2. High-availability local SQLite fallback (storage/learning_store.db).
3. Automatic schema initialization, indexes, and historical migration.
4. Active In-Context RAG memory (few-shot retrieval).
5. Supervised fine-tuning JSONL instruction dataset generation.
6. Real-time telemetry, training session audit, and daemon heartbeat logging.
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

from app.config import get_config

logger = logging.getLogger("facecheck.learning_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "storage" / "learning_store.db"


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


class LearningStore:
    """Enterprise persistent store for LLM continuous learning with PostgreSQL + SQLite resilience."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.config = get_config()
        self.sqlite_path = Path(db_path or self.config.database.sqlite_path or DEFAULT_SQLITE_PATH)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.is_postgresql = False

        # Attempt PostgreSQL connection if configured
        if self.config.database.engine == "postgresql" and self.config.database.host:
            self._init_postgresql()

        # Always initialize SQLite as local cache/fallback
        self._init_sqlite()

        # Auto-migrate any existing SQLite records to PostgreSQL if connected
        if self.is_postgresql:
            self._migrate_sqlite_to_postgresql()

    def _get_pg_connection(self):
        """Create a fresh PostgreSQL connection using credentials from environment."""
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(
            host=self.config.database.host,
            port=self.config.database.port,
            dbname=self.config.database.dbname,
            user=self.config.database.username,
            password=self.config.database.password,
            connect_timeout=self.config.database.connect_timeout,
        )
        return conn

    def _init_postgresql(self) -> None:
        """Initialize PostgreSQL schema with tables, indexes, and constraints."""
        try:
            with self._get_pg_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Main KYC review events table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS kyc_review_events (
                            id SERIAL PRIMARY KEY,
                            sample_id VARCHAR(64) UNIQUE NOT NULL,
                            filename VARCHAR(255),
                            scenario VARCHAR(128),
                            sha256 VARCHAR(64),
                            faces_found INTEGER DEFAULT 0,
                            quality_status VARCHAR(32),
                            laplacian_variance DOUBLE PRECISION DEFAULT 0.0,
                            mean_luminance DOUBLE PRECISION DEFAULT 0.0,
                            pad_decision VARCHAR(32),
                            pad_attack_type VARCHAR(64),
                            pad_risk_score INTEGER DEFAULT 0,
                            match_decision VARCHAR(32),
                            match_similarity DOUBLE PRECISION,
                            llm_verdict VARCHAR(32),
                            llm_risk_score INTEGER DEFAULT 0,
                            llm_summary TEXT,
                            llm_guidance TEXT,
                            human_status VARCHAR(32) NOT NULL,
                            human_issue_type VARCHAR(64),
                            human_notes TEXT,
                            auditor_id VARCHAR(64) DEFAULT 'kyc_officer_core',
                            telemetry_raw JSONB,
                            explanation_raw JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_kyc_sample_id ON kyc_review_events(sample_id);
                        CREATE INDEX IF NOT EXISTS idx_kyc_human_status ON kyc_review_events(human_status);
                        CREATE INDEX IF NOT EXISTS idx_kyc_scenario ON kyc_review_events(scenario);

                        -- 2. Training dataset manifest table
                        CREATE TABLE IF NOT EXISTS kyc_dataset_manifest (
                            id SERIAL PRIMARY KEY,
                            sample_id VARCHAR(64) UNIQUE NOT NULL,
                            filename VARCHAR(255) NOT NULL,
                            scenario VARCHAR(128),
                            target_category VARCHAR(64),
                            sha256 VARCHAR(64),
                            storage_path TEXT,
                            labeled_status VARCHAR(32) DEFAULT 'UNLABELED',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        -- 3. Automated training sessions audit log
                        CREATE TABLE IF NOT EXISTS kyc_training_sessions (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(64) UNIQUE NOT NULL,
                            samples_count INTEGER NOT NULL,
                            convergence_score DOUBLE PRECISION,
                            loss DOUBLE PRECISION,
                            metrics JSONB,
                            model_path TEXT,
                            status VARCHAR(32) DEFAULT 'SUCCESS',
                            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            completed_at TIMESTAMPTZ
                        );

                        -- 4. Continuous learning daemon heartbeats
                        CREATE TABLE IF NOT EXISTS kyc_daemon_heartbeats (
                            id SERIAL PRIMARY KEY,
                            daemon_id VARCHAR(64) UNIQUE NOT NULL,
                            status VARCHAR(32) NOT NULL,
                            last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            samples_processed INTEGER DEFAULT 0,
                            training_triggered_count INTEGER DEFAULT 0,
                            metadata JSONB
                        );
                    """)
                conn.commit()
            self.is_postgresql = True
            logger.info(
                "Connected and initialized PostgreSQL continuous learning database at %s:%s/%s",
                self.config.database.host,
                self.config.database.port,
                self.config.database.dbname,
            )
        except Exception as exc:
            logger.warning(
                "PostgreSQL connection unavailable (%s). Falling back gracefully to local SQLite store.",
                exc,
            )
            self.is_postgresql = False

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        """Create SQLite tables and indexes."""
        with self._lock, self._get_sqlite_connection() as conn:
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

    def _migrate_sqlite_to_postgresql(self) -> None:
        """Migrate any records present in SQLite to PostgreSQL."""
        try:
            with self._get_sqlite_connection() as s_conn:
                s_cur = s_conn.execute("SELECT * FROM kyc_review_events")
                s_rows = [dict(r) for r in s_cur.fetchall()]

            if not s_rows:
                return

            migrated = 0
            with self._get_pg_connection() as p_conn:
                import psycopg2.extras
                with p_conn.cursor() as p_cur:
                    for r in s_rows:
                        telemetry_raw = r.get("telemetry_raw")
                        explanation_raw = r.get("explanation_raw")
                        t_json = json.loads(telemetry_raw) if isinstance(telemetry_raw, str) else (telemetry_raw or {})
                        e_json = json.loads(explanation_raw) if isinstance(explanation_raw, str) else (explanation_raw or {})

                        p_cur.execute("""
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
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s,
                                %s, %s
                            )
                            ON CONFLICT (sample_id) DO UPDATE SET
                                human_status = EXCLUDED.human_status,
                                human_issue_type = EXCLUDED.human_issue_type,
                                human_notes = EXCLUDED.human_notes,
                                updated_at = EXCLUDED.updated_at;
                        """, (
                            r["sample_id"], r.get("filename"), r.get("scenario"), r.get("sha256"),
                            r.get("faces_found") or 0, r.get("quality_status") or "OK",
                            float(r.get("laplacian_variance") or 0.0), float(r.get("mean_luminance") or 0.0),
                            r.get("pad_decision"), r.get("pad_attack_type"), r.get("pad_risk_score") or 0,
                            r.get("match_decision"), r.get("match_similarity"),
                            r.get("llm_verdict"), r.get("llm_risk_score") or 0, r.get("llm_summary"), r.get("llm_guidance"),
                            r["human_status"], r.get("human_issue_type"), r.get("human_notes"), r.get("auditor_id") or "kyc_officer_core",
                            json.dumps(t_json), json.dumps(e_json),
                            r["created_at"], r["updated_at"],
                        ))
                        migrated += 1
                p_conn.commit()
            if migrated > 0:
                logger.info("Migrated %d records from SQLite to PostgreSQL.", migrated)
        except Exception as exc:
            logger.warning("Migration from SQLite to PostgreSQL encountered non-critical error: %s", exc)

    def record_feedback(self, feedback: HumanReviewFeedback) -> Dict[str, Any]:
        """Record or update human auditor decision and notes in PostgreSQL and local SQLite."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t = feedback.telemetry or {}
        q_metrics = t.get("quality_metrics", {})
        exp = feedback.llm_explanation or {}

        # 1. Write to PostgreSQL if active
        if self.is_postgresql:
            try:
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor() as p_cur:
                        p_cur.execute("""
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
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s,
                                %s, %s
                            )
                            ON CONFLICT (sample_id) DO UPDATE SET
                                human_status = EXCLUDED.human_status,
                                human_issue_type = EXCLUDED.human_issue_type,
                                human_notes = EXCLUDED.human_notes,
                                auditor_id = EXCLUDED.auditor_id,
                                updated_at = EXCLUDED.updated_at;
                        """, (
                            feedback.sample_id, feedback.filename, feedback.scenario, feedback.sha256,
                            t.get("faces_found", 0), t.get("quality_status", "OK"),
                            float(q_metrics.get("laplacian_variance", 0.0)),
                            float(q_metrics.get("mean_luminance", 0.0)),
                            t.get("pad_decision"), t.get("pad_attack_type"), t.get("pad_risk_score", 0),
                            t.get("match_decision"), t.get("match_similarity"),
                            exp.get("verdict"), exp.get("risk_score", 0), exp.get("summary"), exp.get("customer_guidance"),
                            feedback.status.upper(), feedback.issue_type, feedback.notes or "", feedback.auditor_id,
                            json.dumps(t), json.dumps(exp),
                            now, now,
                        ))
                    p_conn.commit()
            except Exception as exc:
                logger.error("Failed writing feedback to PostgreSQL: %s (falling back to SQLite)", exc)

        # 2. Always persist to SQLite for local redundancy
        with self._lock, self._get_sqlite_connection() as s_conn:
            s_conn.execute("""
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
            s_conn.commit()

        logger.info(f"Recorded review feedback for {feedback.sample_id}: {feedback.status}")
        return {
            "status": "SUCCESS",
            "sample_id": feedback.sample_id,
            "human_status": feedback.status,
            "engine": "postgresql" if self.is_postgresql else "sqlite",
        }

    def get_few_shot_examples(self, scenario: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve historical auditor decisions to inject into LLM prompt."""
        # Try PostgreSQL first if available
        if self.is_postgresql:
            try:
                import psycopg2.extras
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        if scenario:
                            cur.execute("""
                                SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                                       pad_attack_type, llm_verdict, human_status, human_notes, llm_guidance
                                FROM kyc_review_events
                                WHERE scenario ILIKE %s AND human_status = 'APPROVED'
                                ORDER BY updated_at DESC LIMIT %s;
                            """, (f"%{scenario}%", limit))
                        else:
                            cur.execute("""
                                SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                                       pad_attack_type, llm_verdict, human_status, human_notes, llm_guidance
                                FROM kyc_review_events
                                WHERE human_status = 'APPROVED'
                                ORDER BY updated_at DESC LIMIT %s;
                            """, (limit,))
                        rows = cur.fetchall()
                        return [dict(row) for row in rows]
            except Exception as exc:
                logger.warning("PostgreSQL few-shot query failed: %s (using SQLite)", exc)

        # SQLite fallback
        with self._lock, self._get_sqlite_connection() as conn:
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

        rows = []
        if self.is_postgresql:
            try:
                import psycopg2.extras
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("""
                            SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                                   pad_decision, pad_attack_type, pad_risk_score, match_decision, match_similarity,
                                   llm_verdict, llm_risk_score, llm_summary, llm_guidance, human_status, human_notes
                            FROM kyc_review_events
                            ORDER BY id ASC;
                        """)
                        rows = [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.warning("PostgreSQL export query failed: %s (using SQLite)", exc)

        if not rows:
            with self._lock, self._get_sqlite_connection() as conn:
                cursor = conn.execute("""
                    SELECT sample_id, scenario, faces_found, quality_status, laplacian_variance, mean_luminance,
                           pad_decision, pad_attack_type, pad_risk_score, match_decision, match_similarity,
                           llm_verdict, llm_risk_score, llm_summary, llm_guidance, human_status, human_notes
                    FROM kyc_review_events
                    ORDER BY id ASC;
                """)
                rows = [dict(r) for r in cursor.fetchall()]

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

    def record_training_session(
        self,
        session_id: str,
        samples_count: int,
        convergence_score: float,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        status: str = "SUCCESS",
    ) -> None:
        """Record an automated classifier or LLM retraining session."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metrics_json = json.dumps(metrics or {})

        if self.is_postgresql:
            try:
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO kyc_training_sessions (
                                session_id, samples_count, convergence_score, loss,
                                metrics, model_path, status, started_at, completed_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET
                                convergence_score = EXCLUDED.convergence_score,
                                loss = EXCLUDED.loss,
                                metrics = EXCLUDED.metrics,
                                status = EXCLUDED.status,
                                completed_at = EXCLUDED.completed_at;
                        """, (
                            session_id, samples_count, convergence_score, loss,
                            metrics_json, model_path, status, now, now,
                        ))
                    p_conn.commit()
            except Exception as exc:
                logger.warning("Failed recording training session to PostgreSQL: %s", exc)

    def record_daemon_heartbeat(
        self,
        daemon_id: str,
        status: str,
        samples_processed: int = 0,
        training_triggered_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record heartbeat from the continuous background learning daemon."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        if self.is_postgresql:
            try:
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO kyc_daemon_heartbeats (
                                daemon_id, status, last_heartbeat, samples_processed,
                                training_triggered_count, metadata
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (daemon_id) DO UPDATE SET
                                status = EXCLUDED.status,
                                last_heartbeat = EXCLUDED.last_heartbeat,
                                samples_processed = EXCLUDED.samples_processed,
                                training_triggered_count = EXCLUDED.training_triggered_count,
                                metadata = EXCLUDED.metadata;
                        """, (daemon_id, status, now, samples_processed, training_triggered_count, meta_json))
                    p_conn.commit()
            except Exception as exc:
                logger.warning("Failed logging daemon heartbeat: %s", exc)

    def get_learning_stats(self) -> LearningStats:
        """Calculate and return operational learning analytics from PostgreSQL or SQLite."""
        if self.is_postgresql:
            try:
                import psycopg2.extras
                with self._get_pg_connection() as p_conn:
                    with p_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("SELECT COUNT(*) FROM kyc_review_events;")
                        total = cur.fetchone()[0]

                        cur.execute("SELECT COUNT(*) FROM kyc_review_events WHERE human_status = 'APPROVED';")
                        approved = cur.fetchone()[0]

                        cur.execute("SELECT COUNT(*) FROM kyc_review_events WHERE human_status = 'ALERTED';")
                        alerted = cur.fetchone()[0]

                        cur.execute("""
                            SELECT human_issue_type, COUNT(*) as cnt
                            FROM kyc_review_events
                            WHERE human_issue_type IS NOT NULL AND human_issue_type != ''
                            GROUP BY human_issue_type;
                        """)
                        breakdown = {row["human_issue_type"]: row["cnt"] for row in cur.fetchall()}

                        cur.execute("SELECT MAX(updated_at) FROM kyc_review_events;")
                        last_row_raw = cur.fetchone()[0]
                        last_row = last_row_raw.isoformat() if last_row_raw else None

                        cur.execute("SELECT pg_total_relation_size('kyc_review_events');")
                        size_bytes = cur.fetchone()[0]

                return LearningStats(
                    total_records=total,
                    approved_count=approved,
                    alerted_count=alerted,
                    pending_fine_tuning_samples=total,
                    issue_breakdown=breakdown,
                    database_file=f"PostgreSQL 15.6 @ {self.config.database.host}:{self.config.database.port}/{self.config.database.dbname}",
                    database_size_bytes=size_bytes,
                    database_engine="postgresql",
                    last_updated=last_row,
                )
            except Exception as exc:
                logger.warning("PostgreSQL stats failed: %s (falling back to SQLite)", exc)

        # SQLite fallback
        with self._lock, self._get_sqlite_connection() as conn:
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

        size_bytes = self.sqlite_path.stat().st_size if self.sqlite_path.is_file() else 0
        return LearningStats(
            total_records=total,
            approved_count=approved,
            alerted_count=alerted,
            pending_fine_tuning_samples=total,
            issue_breakdown=breakdown,
            database_file=str(self.sqlite_path.relative_to(PROJECT_ROOT)),
            database_size_bytes=size_bytes,
            database_engine="sqlite",
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
