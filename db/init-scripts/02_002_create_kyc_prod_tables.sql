-- Migration 002: Create Production KYC Tables
-- Idempotent & Reversible

-- Table 1: Decisions Ledger
CREATE TABLE IF NOT EXISTS kyc_prod.decisions (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    applicant_id_masked VARCHAR(64) NOT NULL,
    decision VARCHAR(16) NOT NULL CHECK (decision IN ('ACCEPT', 'REVIEW', 'REJECT')),
    match_score DOUBLE PRECISION,
    pad_risk_score DOUBLE PRECISION,
    reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    quality_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    llm_explanation JSONB,
    record_hash VARCHAR(64) NOT NULL,
    prev_hash VARCHAR(64) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON kyc_prod.decisions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON kyc_prod.decisions (decision);
CREATE INDEX IF NOT EXISTS idx_decisions_applicant ON kyc_prod.decisions (applicant_id_masked);

-- Table 2: Application-Level Encrypted Images (AES-256-GCM)
CREATE TABLE IF NOT EXISTS kyc_prod.encrypted_images (
    image_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_id UUID NOT NULL REFERENCES kyc_prod.decisions (check_id) ON DELETE CASCADE,
    image_type VARCHAR(32) NOT NULL CHECK (image_type IN ('SELFIE', 'ID_PHOTO', 'ID_FRONT', 'CROPPED_FACE')),
    ciphertext BYTEA NOT NULL,
    iv_nonce BYTEA NOT NULL, -- 96-bit (12-byte) initialization vector
    key_id VARCHAR(64) NOT NULL, -- KMS Key Identifier
    key_version VARCHAR(32) NOT NULL DEFAULT '1.0',
    algorithm VARCHAR(32) NOT NULL DEFAULT 'AES-256-GCM',
    ciphertext_hash VARCHAR(64) NOT NULL, -- SHA-256 of ciphertext for integrity
    purge_after TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_encrypted_images_check_id ON kyc_prod.encrypted_images (check_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_images_purge_after ON kyc_prod.encrypted_images (purge_after);

-- Table 3: Tamper-Evident Hash-Chained Audit Ledger
CREATE TABLE IF NOT EXISTS kyc_prod.audit_chain (
    sequence_num BIGSERIAL PRIMARY KEY,
    prev_hash VARCHAR(64) NOT NULL,
    record_hash VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actor VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_chain_timestamp ON kyc_prod.audit_chain (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_chain_event_type ON kyc_prod.audit_chain (event_type);

-- Table 4: Global Kill Switch State
CREATE TABLE IF NOT EXISTS kyc_prod.kill_switch_state (
    state_id SERIAL PRIMARY KEY,
    is_training_enabled BOOLEAN NOT NULL DEFAULT false,
    is_promotion_enabled BOOLEAN NOT NULL DEFAULT false,
    reason TEXT NOT NULL,
    changed_by VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed initial fail-safe disabled state if table is empty
INSERT INTO kyc_prod.kill_switch_state (is_training_enabled, is_promotion_enabled, reason, changed_by)
SELECT false, false, 'Initial fail-safe default state (offline/disabled)', 'SYSTEM_INIT'
WHERE NOT EXISTS (SELECT 1 FROM kyc_prod.kill_switch_state);

-- Table 5: Data Retention & Purge Policies
CREATE TABLE IF NOT EXISTS kyc_prod.retention_policies (
    policy_id SERIAL PRIMARY KEY,
    data_class VARCHAR(64) NOT NULL UNIQUE,
    retention_days INT NOT NULL CHECK (retention_days >= 0),
    auto_purge BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed standard bank retention baselines
INSERT INTO kyc_prod.retention_policies (data_class, retention_days, auto_purge)
VALUES
    ('EPHEMERAL_FAILED_SELFIE', 1, true),
    ('AUDIT_VERIFIED_KYC_IMAGE', 90, true),
    ('COMPLIANCE_DECISION_METADATA', 2555, false) -- 7 years retention for banking compliance
ON CONFLICT (data_class) DO NOTHING;
