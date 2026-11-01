-- Migration 004: Create Sanitized Observability Views & Training Firewall
-- Idempotent & Reversible

-- ------------------------------------------------------------------------------
-- 1. TRAINING FIREWALL: Strictly Consented & High-Confidence Data Only
-- ------------------------------------------------------------------------------

-- View 1: Eligible Training Samples (ML Worker View)
-- Invariants:
-- 1. AUTO_LOW_CONF samples are unconditionally excluded.
-- 2. Golden test set samples are unconditionally excluded.
-- 3. Withdrawn or expired customer consents are unconditionally excluded.
CREATE OR REPLACE VIEW ml_training.v_eligible_training_samples AS
SELECT 
    s.sample_id,
    s.source_type,
    s.license_type,
    s.label_origin,
    s.ground_truth_label,
    s.phash_hex,
    s.dhash_hex,
    s.quality_score,
    s.ingested_at
FROM ml_training.training_samples s
LEFT JOIN ml_training.consent_records c ON s.consent_id = c.consent_id
WHERE s.label_origin IN ('HUMAN', 'AUTO_HIGH_CONF')
  AND s.is_in_golden_test = false
  AND (
      s.source_type IN ('SYNTHETIC_IN_HOUSE', 'LICENSED_PUBLIC')
      OR (
          s.source_type = 'CONSENTED_CUSTOMER'
          AND c.is_withdrawn = false
          AND c.retention_expiry > CURRENT_TIMESTAMP
      )
  );

-- View 2: Consented KYC Training Candidates (Bridge from kyc_prod to ml_worker)
-- Exposes ONLY masked telemetry for consented samples; ZERO raw images or PII
CREATE OR REPLACE VIEW kyc_prod.v_consented_training_candidates AS
SELECT 
    d.check_id,
    d.created_at,
    d.applicant_id_masked,
    d.decision,
    d.match_score,
    d.pad_risk_score,
    d.reason_codes,
    d.quality_metrics
FROM kyc_prod.decisions d
JOIN ml_training.consent_records c ON c.applicant_id_hash = encode(digest(d.applicant_id_masked, 'sha256'), 'hex')
WHERE c.is_withdrawn = false
  AND c.retention_expiry > CURRENT_TIMESTAMP;

-- ------------------------------------------------------------------------------
-- 2. MONITORING OBSERVABILITY VIEWS (Strictly Read-Only, Masked, No PII)
-- ------------------------------------------------------------------------------

-- View 3: Decision Throughput & Verdict Distribution (Monitoring Web)
CREATE OR REPLACE VIEW kyc_prod.v_monitoring_decisions AS
SELECT 
    date_trunc('hour', created_at) AS time_window,
    COUNT(*) AS total_checks,
    COUNT(*) FILTER (WHERE decision = 'ACCEPT') AS accept_count,
    COUNT(*) FILTER (WHERE decision = 'REVIEW') AS review_count,
    COUNT(*) FILTER (WHERE decision = 'REJECT') AS reject_count,
    ROUND(AVG(match_score)::numeric, 4) AS avg_match_score,
    ROUND(AVG(pad_risk_score)::numeric, 2) AS avg_pad_risk_score
FROM kyc_prod.decisions
GROUP BY date_trunc('hour', created_at)
ORDER BY time_window DESC;

-- View 4: System & Pipeline Status
CREATE OR REPLACE VIEW kyc_prod.v_monitoring_pipeline_status AS
SELECT 
    k.is_training_enabled,
    k.is_promotion_enabled,
    k.reason AS kill_switch_reason,
    k.updated_at AS kill_switch_updated_at,
    (SELECT COUNT(*) FROM kyc_prod.audit_chain) AS audit_chain_length,
    (SELECT MAX(timestamp) FROM kyc_prod.audit_chain) AS last_audit_event_time
FROM kyc_prod.kill_switch_state k
ORDER BY k.state_id DESC
LIMIT 1;

-- View 5: Model Registry Status (Production vs Shadow vs Canary)
CREATE OR REPLACE VIEW ml_training.v_monitoring_model_metrics AS
SELECT 
    model_id,
    model_name,
    model_version,
    status,
    metrics->>'apcer' AS apcer,
    metrics->>'bpcer' AS bpcer,
    metrics->>'fmr' AS fmr,
    metrics->>'fnmr' AS fnmr,
    git_commit_sha,
    registered_at,
    approved_at
FROM ml_training.model_registry
ORDER BY registered_at DESC;

-- View 6: Training Runs & Audit Disagreement Monitoring
CREATE OR REPLACE VIEW ml_training.v_monitoring_training_runs AS
SELECT 
    run_id,
    stage,
    status,
    samples_ingested,
    auto_disagreement_rate,
    triggered_by,
    started_at,
    finished_at
FROM ml_training.training_run_audit
ORDER BY started_at DESC;

-- View 7: Security Alerts & Chain Health
CREATE OR REPLACE VIEW kyc_prod.v_monitoring_security_alerts AS
SELECT 
    sequence_num,
    timestamp,
    actor,
    event_type,
    payload->>'severity' AS severity,
    payload->>'description' AS description
FROM kyc_prod.audit_chain
WHERE event_type LIKE '%SECURITY%' OR event_type LIKE '%ALERT%'
ORDER BY timestamp DESC;
