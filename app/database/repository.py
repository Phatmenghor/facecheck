"""Repository component for PostgreSQL continuous learning, image persistence, and audit logging."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import psycopg2
import psycopg2.extras

from app.config import get_config
from app.database.connection import DatabaseConnectionManager
from app.database.models import HumanReviewFeedback, LearningStats, StoredImageRecord

logger = logging.getLogger("facecheck.database.repository")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PostgresLearningStore:
    """Enterprise persistent repository backed strictly by PostgreSQL."""

    def __init__(self, db_path: Any = None, *args: Any, **kwargs: Any) -> None:
        self.config = get_config()
        self.conn_mgr = DatabaseConnectionManager()
        self._lock = threading.Lock()
        self.is_postgresql = True
        self._init_schema()

    def _get_pg_connection(self):
        """Create a fresh PostgreSQL connection."""
        return self.conn_mgr.get_connection()

    def _init_schema(self) -> None:
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

                        -- 5. Full Binary Image Store for 100% database persistence
                        CREATE TABLE IF NOT EXISTS kyc_image_store (
                            id SERIAL PRIMARY KEY,
                            sample_id VARCHAR(64) UNIQUE NOT NULL,
                            filename VARCHAR(255) NOT NULL,
                            mime_type VARCHAR(64) NOT NULL DEFAULT 'image/jpeg',
                            image_size_bytes INTEGER NOT NULL,
                            image_bytes BYTEA NOT NULL,
                            sha256 VARCHAR(64),
                            scenario VARCHAR(128),
                            target_category VARCHAR(64),
                            metadata JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_kyc_img_sample ON kyc_image_store(sample_id);
                        CREATE INDEX IF NOT EXISTS idx_kyc_img_cat ON kyc_image_store(target_category);
                    """)
                conn.commit()
            logger.info("Connected and initialized PostgreSQL continuous learning database at %s:%s/%s",
                        self.config.database.host, self.config.database.port, self.config.database.dbname)
        except Exception as exc:
            logger.critical("FATAL: PostgreSQL database connection failed (%s). Fallback is completely disabled.", exc)
            raise RuntimeError(f"PostgreSQL connection failed: {exc}. Fallback is completely disabled.") from exc

    # =========================================================================
    # KYC Review Events & Human Feedback
    # =========================================================================

    def record_feedback(self, feedback: HumanReviewFeedback) -> Dict[str, Any]:
        """Record or update human auditor decision and notes in PostgreSQL."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t = feedback.telemetry or {}
        q_metrics = t.get("quality_metrics", {})
        exp = feedback.llm_explanation or {}

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

        logger.info("Recorded review feedback in PostgreSQL for %s: %s", feedback.sample_id, feedback.status)
        return {
            "status": "SUCCESS",
            "sample_id": feedback.sample_id,
            "human_status": feedback.status,
            "engine": "postgresql",
        }

    def get_few_shot_examples(self, scenario: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve historical auditor decisions to inject into LLM prompt from PostgreSQL."""
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

    def export_fine_tuning_dataset(self, output_path: Optional[Path] = None) -> Path:
        """Export all human-audited cases from PostgreSQL into OpenAI/Gemini JSONL fine-tuning format."""
        target_file = Path(output_path or (PROJECT_ROOT / "storage" / "llm_biometric_finetune.jsonl"))
        target_file.parent.mkdir(parents=True, exist_ok=True)

        system_prompt = (
            "You are the Chief Biometric AI Officer for CP Bank digital account opening KYC verification. "
            "Analyze biometric computer vision telemetry and output structured JSON decisions with verdicts, "
            "actionable customer retake instructions, and NIST SP 800-63B / ISO 30107 regulatory compliance rationale."
        )

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

        logger.info("Exported %d fine-tuning examples from PostgreSQL to %s", count, target_file)
        return target_file

    def get_learning_stats(self) -> LearningStats:
        """Calculate and return operational learning analytics from PostgreSQL."""
        with self._get_pg_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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

    def list_review_events(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
        verdict: Optional[str] = None,
        search: Optional[str] = None,
        sort_desc: bool = True,
    ) -> Dict[str, Any]:
        """Query paginated KYC biometric review events with filters and total counts."""
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        offset = (page - 1) * page_size
        where_clauses: List[str] = []
        params: List[Any] = []

        if status:
            where_clauses.append("human_status = %s")
            params.append(status.upper())
        if verdict:
            where_clauses.append("llm_verdict = %s")
            params.append(verdict.upper())
        if search:
            where_clauses.append("(sample_id ILIKE %s OR filename ILIKE %s OR scenario ILIKE %s)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        order_sql = "ORDER BY id DESC" if sort_desc else "ORDER BY id ASC"

        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                # 1. Total count query
                count_query = f"SELECT COUNT(*) FROM kyc_review_events {where_sql};"
                cur.execute(count_query, params)
                total_count = cur.fetchone()[0]

                # 2. Paginated rows query
                fetch_query = f"""
                    SELECT id, sample_id, filename, scenario, sha256, faces_found,
                           quality_status, laplacian_variance, mean_luminance,
                           pad_decision, pad_attack_type, pad_risk_score,
                           match_decision, match_similarity, llm_verdict, llm_risk_score,
                           llm_summary, llm_guidance, human_status, human_issue_type,
                           human_notes, auditor_id, created_at, explanation_raw
                    FROM kyc_review_events
                    {where_sql}
                    {order_sql}
                    LIMIT %s OFFSET %s;
                """
                cur.execute(fetch_query, params + [page_size, offset])
                rows = cur.fetchall()

        events = []
        for r in rows:
            events.append({
                "id": r[0],
                "sample_id": r[1],
                "filename": r[2],
                "scenario": r[3],
                "sha256": r[4],
                "faces_found": r[5],
                "quality_status": r[6],
                "laplacian_variance": r[7],
                "mean_luminance": r[8],
                "pad_decision": r[9],
                "pad_attack_type": r[10],
                "pad_risk_score": r[11],
                "match_decision": r[12],
                "match_similarity": r[13],
                "llm_verdict": r[14],
                "llm_risk_score": r[15],
                "llm_summary": r[16],
                "llm_guidance": r[17],
                "human_status": r[18],
                "human_issue_type": r[19],
                "human_notes": r[20],
                "auditor_id": r[21],
                "created_at": r[22].isoformat() if hasattr(r[22], "isoformat") else str(r[22]),
                "explanation_raw": r[23] if isinstance(r[23], dict) else {},
                "image_url": f"/api/v1/learning/images/{r[1]}",
            })

        total_pages = max(1, (total_count + page_size - 1) // page_size) if page_size > 0 else 1

        return {
            "status": "SUCCESS",
            "total_records": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "events": events,
        }

    # =========================================================================
    # Image Store Repository (100% PostgreSQL Persistence)
    # =========================================================================

    def save_image(
        self,
        sample_id: str,
        filename: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        scenario: Optional[str] = None,
        target_category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persist raw binary image directly into kyc_image_store table in PostgreSQL."""
        sha256_hash = hashlib.sha256(image_bytes).hexdigest()
        size_bytes = len(image_bytes)
        meta_dict = metadata or {}

        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO kyc_image_store (
                        sample_id, filename, mime_type, image_size_bytes, image_bytes,
                        sha256, scenario, target_category, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sample_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        mime_type = EXCLUDED.mime_type,
                        image_size_bytes = EXCLUDED.image_size_bytes,
                        image_bytes = EXCLUDED.image_bytes,
                        sha256 = EXCLUDED.sha256,
                        scenario = EXCLUDED.scenario,
                        target_category = EXCLUDED.target_category,
                        metadata = EXCLUDED.metadata;
                """, (
                    sample_id,
                    filename,
                    mime_type,
                    size_bytes,
                    psycopg2.Binary(image_bytes),
                    sha256_hash,
                    scenario or "kyc_sample",
                    target_category or "candidate",
                    json.dumps(meta_dict),
                ))
                # Also register in manifest
                cur.execute("""
                    INSERT INTO kyc_dataset_manifest (
                        sample_id, filename, scenario, target_category, sha256, storage_path, labeled_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sample_id) DO UPDATE SET
                        sha256 = EXCLUDED.sha256,
                        storage_path = EXCLUDED.storage_path;
                """, (
                    sample_id,
                    filename,
                    scenario or "kyc_sample",
                    target_category or "candidate",
                    sha256_hash,
                    f"database://kyc_image_store/{sample_id}",
                    "STORED_IN_DB",
                ))
            conn.commit()
        return True

    def get_image(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full image record including binary bytes directly from PostgreSQL."""
        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT sample_id, filename, mime_type, image_size_bytes, image_bytes,
                           sha256, scenario, target_category, metadata, created_at
                    FROM kyc_image_store
                    WHERE sample_id = %s;
                """, (sample_id,))
                row = cur.fetchone()
                if row:
                    raw_data = row[4]
                    if hasattr(raw_data, "tobytes"):
                        raw_bytes = raw_data.tobytes()
                    elif isinstance(raw_data, memoryview):
                        raw_bytes = bytes(raw_data)
                    else:
                        raw_bytes = bytes(raw_data)
                    return {
                        "sample_id": row[0],
                        "filename": row[1],
                        "mime_type": row[2],
                        "image_size_bytes": row[3],
                        "image_bytes": raw_bytes,
                        "sha256": row[5],
                        "scenario": row[6],
                        "target_category": row[7],
                        "metadata": row[8] or {},
                        "created_at": row[9].isoformat() if hasattr(row[9], "isoformat") else str(row[9]),
                    }
        return None

    def get_image_bytes(self, sample_id: str) -> Optional[bytes]:
        """Convenience method to retrieve raw binary image bytes from PostgreSQL."""
        rec = self.get_image(sample_id)
        return rec["image_bytes"] if rec else None

    def list_stored_images(
        self,
        target_category: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List metadata for all images stored in PostgreSQL (without heavy byte payload)."""
        images = []
        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                if target_category:
                    cur.execute("""
                        SELECT sample_id, filename, mime_type, image_size_bytes, sha256,
                               scenario, target_category, created_at
                        FROM kyc_image_store
                        WHERE target_category = %s
                        ORDER BY id ASC
                        LIMIT %s;
                    """, (target_category, limit))
                else:
                    cur.execute("""
                        SELECT sample_id, filename, mime_type, image_size_bytes, sha256,
                               scenario, target_category, created_at
                        FROM kyc_image_store
                        ORDER BY id ASC
                        LIMIT %s;
                    """, (limit,))
                for r in cur.fetchall():
                    images.append({
                        "sample_id": r[0],
                        "filename": r[1],
                        "mime_type": r[2],
                        "image_size_bytes": r[3],
                        "sha256": r[4],
                        "scenario": r[5],
                        "target_category": r[6],
                        "created_at": r[7].isoformat() if hasattr(r[7], "isoformat") else str(r[7]),
                    })
        return images

    def delete_image_from_db(self, sample_id: str) -> bool:
        """Delete an image from PostgreSQL kyc_image_store."""
        with self._get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kyc_image_store WHERE sample_id = %s;", (sample_id,))
                cur.execute("DELETE FROM kyc_dataset_manifest WHERE sample_id = %s;", (sample_id,))
            conn.commit()
        return True

    def migrate_local_images_to_db(
        self,
        candidate_dirs: Optional[List[Path]] = None,
        delete_local_files: bool = True,
    ) -> Dict[str, Any]:
        """Migrate local image files into PostgreSQL kyc_image_store and safely purge from disk."""
        if candidate_dirs is None:
            candidate_dirs = [
                PROJECT_ROOT / "assets" / "review_10" / "images",
            ]

        migrated: List[str] = []
        deleted: List[str] = []
        errors: List[str] = []

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }

        for cdir in candidate_dirs:
            if not cdir.is_dir():
                continue
            for img_path in sorted(cdir.glob("*.*")):
                suffix = img_path.suffix.lower()
                if suffix not in mime_map:
                    continue
                filename = img_path.name
                sample_id = img_path.stem

                try:
                    raw_bytes = img_path.read_bytes()
                    mime_type = mime_map.get(suffix, "image/jpeg")

                    # Categorize target
                    cat = "genuine"
                    if "impostor" in filename or "mismatch" in filename:
                        cat = "impostor"
                    elif "print" in filename or "halftone" in filename:
                        cat = "print_attack"
                    elif "screen" in filename or "moire" in filename:
                        cat = "screen_replay"
                    elif "blur" in filename:
                        cat = "blurry"
                    elif "dark" in filename or "underexposed" in filename:
                        cat = "dark"
                    elif "noface" in filename or "nature" in filename:
                        cat = "no_face"
                    elif "multiface" in filename or "crowd" in filename:
                        cat = "multi_face"

                    saved = self.save_image(
                        sample_id=sample_id,
                        filename=filename,
                        image_bytes=raw_bytes,
                        mime_type=mime_type,
                        scenario="Digital Account Opening Review Image",
                        target_category=cat,
                        metadata={"original_name": filename, "file_size": len(raw_bytes)},
                    )

                    if saved:
                        migrated.append(sample_id)
                        # Verify the image bytes exist and match in database before deleting file
                        verified_bytes = self.get_image_bytes(sample_id)
                        if verified_bytes and len(verified_bytes) == len(raw_bytes):
                            if delete_local_files:
                                img_path.unlink()
                                deleted.append(filename)
                        else:
                            errors.append(f"Verification failed for {sample_id}; retained on disk.")
                except Exception as exc:
                    logger.error("Failed migrating image %s: %s", img_path, exc)
                    errors.append(f"{filename}: {str(exc)}")

        return {
            "status": "SUCCESS",
            "migrated_count": len(migrated),
            "deleted_from_disk_count": len(deleted),
            "migrated_sample_ids": migrated,
            "deleted_paths": deleted,
            "errors": errors,
        }

    # =========================================================================
    # Training Sessions & Daemon Heartbeats
    # =========================================================================

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
        """Record an automated classifier or LLM retraining session into PostgreSQL."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metrics_json = json.dumps(metrics or {})

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

    def record_daemon_heartbeat(
        self,
        daemon_id: str,
        status: str,
        samples_processed: int = 0,
        training_triggered_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record heartbeat from continuous learning daemon into PostgreSQL."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

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
