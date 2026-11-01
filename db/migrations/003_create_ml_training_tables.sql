-- Migration 003: Create ML Training & Model Registry Tables
-- Idempotent & Reversible

-- Table 1: Consent & Provenance Registry
CREATE TABLE IF NOT EXISTS ml_training.consent_records (
    consent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    applicant_id_hash VARCHAR(64) NOT NULL,
    source_type VARCHAR(32) NOT NULL CHECK (source_type IN ('SYNTHETIC_IN_HOUSE', 'LICENSED_PUBLIC', 'CONSENTED_CUSTOMER')),
    license_type VARCHAR(64) NOT NULL,
    consent_scope JSONB NOT NULL DEFAULT '{"biometric_training": true}'::jsonb,
    consent_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_withdrawn BOOLEAN NOT NULL DEFAULT false,
    retention_expiry TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consent_applicant ON ml_training.consent_records (applicant_id_hash);
CREATE INDEX IF NOT EXISTS idx_consent_active ON ml_training.consent_records (is_withdrawn, retention_expiry);

-- Table 2: Training Samples Metadata
CREATE TABLE IF NOT EXISTS ml_training.training_samples (
    sample_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consent_id UUID REFERENCES ml_training.consent_records (consent_id) ON DELETE SET NULL,
    source_type VARCHAR(32) NOT NULL CHECK (source_type IN ('SYNTHETIC_IN_HOUSE', 'LICENSED_PUBLIC', 'CONSENTED_CUSTOMER')),
    license_type VARCHAR(64) NOT NULL,
    label_origin VARCHAR(32) NOT NULL CHECK (label_origin IN ('HUMAN', 'AUTO_HIGH_CONF', 'AUTO_LOW_CONF')),
    ground_truth_label VARCHAR(32) NOT NULL CHECK (ground_truth_label IN ('GENUINE', 'PRINT_ATTACK', 'REPLAY_ATTACK', 'SYNTHETIC_FACE')),
    phash_hex VARCHAR(64) NOT NULL,
    dhash_hex VARCHAR(64) NOT NULL,
    quality_score DOUBLE PRECISION,
    is_in_golden_test BOOLEAN NOT NULL DEFAULT false,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_samples_label_origin ON ml_training.training_samples (label_origin);
CREATE INDEX IF NOT EXISTS idx_samples_phash ON ml_training.training_samples (phash_hex);
CREATE INDEX IF NOT EXISTS idx_samples_golden ON ml_training.training_samples (is_in_golden_test);

-- Table 3: Model Registry & Two-Person Promotion Governance
CREATE TABLE IF NOT EXISTS ml_training.model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(64) NOT NULL,
    model_version VARCHAR(32) NOT NULL UNIQUE,
    sha256_hash VARCHAR(64) NOT NULL,
    signature TEXT NOT NULL,
    data_manifest_id UUID NOT NULL,
    git_commit_sha VARCHAR(40) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL CHECK (status IN ('CANDIDATE', 'SHADOW', 'APPROVED', 'CANARY', 'PRODUCTION', 'ROLLED_BACK')),
    approver_1 VARCHAR(64),
    approver_2 VARCHAR(64),
    approved_at TIMESTAMPTZ,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Two-Person Rule Constraint: Approver 1 and Approver 2 must be distinct individuals
    CONSTRAINT ck_distinct_approvers CHECK (approver_1 IS NULL OR approver_2 IS NULL OR approver_1 <> approver_2)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status ON ml_training.model_registry (status);
CREATE INDEX IF NOT EXISTS idx_model_registry_version ON ml_training.model_registry (model_version);

-- Table 4: Frozen Versioned Golden Test Set
CREATE TABLE IF NOT EXISTS ml_training.golden_test_set (
    test_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL REFERENCES ml_training.training_samples (sample_id) ON DELETE RESTRICT,
    attack_category VARCHAR(32) NOT NULL,
    capture_device VARCHAR(64) NOT NULL,
    demographic_group JSONB NOT NULL DEFAULT '{}'::jsonb,
    frozen_version VARCHAR(32) NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_golden_version ON ml_training.golden_test_set (frozen_version);

-- Table 5: Training Run Audit & Disagreement Monitoring
CREATE TABLE IF NOT EXISTS ml_training.training_run_audit (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stage VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'PAUSED_DISAGREEMENT', 'FAILED_GATING', 'ABORTED_KILL_SWITCH')),
    samples_ingested INT NOT NULL DEFAULT 0,
    auto_disagreement_rate DOUBLE PRECISION,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    triggered_by VARCHAR(64) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ
);
