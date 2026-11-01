-- Migration 005: Audit Chain Append-Only Trigger & External Anchors
-- Idempotent & Reversible

-- Table: External WORM Anchors
-- Stores cryptographic anchor checkpoints sent to external write-once storage (WORM bucket, syslog, or PKI timestamp)
CREATE TABLE IF NOT EXISTS kyc_prod.audit_anchors (
    anchor_id BIGSERIAL PRIMARY KEY,
    chain_sequence_num BIGINT NOT NULL,
    anchored_hash VARCHAR(64) NOT NULL,
    anchor_destination VARCHAR(64) NOT NULL, -- e.g. 'AWS_S3_OBJECT_LOCK', 'INTERNAL_WORM_VAULT'
    signature TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_anchors_seq ON kyc_prod.audit_anchors (chain_sequence_num);

-- ------------------------------------------------------------------------------
-- IMMUTABILITY TRIGGER: Reject any UPDATE or DELETE on kyc_prod.audit_chain
-- ------------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION kyc_prod.fn_prevent_audit_chain_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'SECURITY_VIOLATION: kyc_prod.audit_chain is strictly append-only. UPDATE and DELETE operations are forbidden.'
        USING ERRCODE = '55000'; -- Object not in prerequisite state / integrity violation
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prevent_audit_mutation ON kyc_prod.audit_chain;
CREATE TRIGGER trg_prevent_audit_mutation
    BEFORE UPDATE OR DELETE ON kyc_prod.audit_chain
    FOR EACH ROW
    EXECUTE FUNCTION kyc_prod.fn_prevent_audit_chain_mutation();

-- Also protect audit_anchors from mutation
DROP TRIGGER IF EXISTS trg_prevent_anchor_mutation ON kyc_prod.audit_anchors;
CREATE TRIGGER trg_prevent_anchor_mutation
    BEFORE UPDATE OR DELETE ON kyc_prod.audit_anchors
    FOR EACH ROW
    EXECUTE FUNCTION kyc_prod.fn_prevent_audit_chain_mutation();
