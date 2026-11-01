-- ==============================================================================
-- FaceCheck Biometric Verification Database Security Roles & Grants
-- Architecture: Dual-Schema Isolation (kyc_prod vs ml_training)
-- Compliance: Least-Privilege, Zero Superusers, Schema Firewall
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. BASELINE HARDENING: REVOKE PUBLIC PRIVILEGES
-- Prevent unauthorized or unauthenticated queries on public schemas
-- ------------------------------------------------------------------------------
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA kyc_prod FROM PUBLIC;
REVOKE ALL ON SCHEMA ml_training FROM PUBLIC;

-- ------------------------------------------------------------------------------
-- 2. CREATE ISOLATED SYSTEM ROLES
-- Note: Passwords must be injected via environment/secret manager, never hardcoded.
-- ------------------------------------------------------------------------------

-- Role 1: facecheck_migrator (DDL administrator for automated schema migrations only)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_migrator') THEN
        CREATE ROLE facecheck_migrator WITH LOGIN;
    END IF;
END $$;

-- Role 2: facecheck_app (Production online KYC API service)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_app') THEN
        CREATE ROLE facecheck_app WITH LOGIN;
    END IF;
END $$;

-- Role 3: facecheck_ml_worker (Background scheduled automated training worker)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_ml_worker') THEN
        CREATE ROLE facecheck_ml_worker WITH LOGIN;
    END IF;
END $$;

-- Role 4: facecheck_readonly (Observability & Monitoring Web Service)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_readonly') THEN
        CREATE ROLE facecheck_readonly WITH LOGIN;
    END IF;
END $$;

-- ------------------------------------------------------------------------------
-- 3. SCHEMA LEVEL PERMISSIONS
-- ------------------------------------------------------------------------------

-- Migrator has full DDL access on both schemas
GRANT USAGE, CREATE ON SCHEMA kyc_prod TO facecheck_migrator;
GRANT USAGE, CREATE ON SCHEMA ml_training TO facecheck_migrator;

-- App role operates strictly in kyc_prod
GRANT USAGE ON SCHEMA kyc_prod TO facecheck_app;
REVOKE ALL ON SCHEMA ml_training FROM facecheck_app;

-- ML Worker operates in ml_training and reads consented sanitized views from kyc_prod
GRANT USAGE ON SCHEMA ml_training TO facecheck_ml_worker;
GRANT USAGE ON SCHEMA kyc_prod TO facecheck_ml_worker;

-- Readonly role requires USAGE to query sanitized monitoring views
GRANT USAGE ON SCHEMA kyc_prod TO facecheck_readonly;
GRANT USAGE ON SCHEMA ml_training TO facecheck_readonly;

-- ------------------------------------------------------------------------------
-- 4. TABLE & VIEW LEVEL GRANTS (LEAST PRIVILEGE)
-- ------------------------------------------------------------------------------

-- --- ROLE: facecheck_app ---
-- Can insert/select decisions and encrypted images, append to audit chain
GRANT SELECT, INSERT, UPDATE ON kyc_prod.decisions TO facecheck_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON kyc_prod.encrypted_images TO facecheck_app;
GRANT SELECT, INSERT ON kyc_prod.audit_chain TO facecheck_app;
GRANT USAGE, SELECT ON SEQUENCE kyc_prod.audit_chain_sequence_num_seq TO facecheck_app;
GRANT SELECT, INSERT, UPDATE ON kyc_prod.kill_switch_state TO facecheck_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON kyc_prod.retention_policies TO facecheck_app;

-- Strict Invariant: App has NO ACCESS to ml_training tables
REVOKE ALL ON ALL TABLES IN SCHEMA ml_training FROM facecheck_app;

-- --- ROLE: facecheck_ml_worker ---
-- Has read/write in ml_training
GRANT SELECT, INSERT, UPDATE ON ml_training.consent_records TO facecheck_ml_worker;
GRANT SELECT, INSERT, UPDATE ON ml_training.training_samples TO facecheck_ml_worker;
GRANT SELECT, INSERT, UPDATE ON ml_training.model_registry TO facecheck_ml_worker;
GRANT SELECT ON ml_training.golden_test_set TO facecheck_ml_worker; -- Read-only on golden set
GRANT SELECT ON ml_training.v_eligible_training_samples TO facecheck_ml_worker;
GRANT SELECT, INSERT ON ml_training.training_run_audit TO facecheck_ml_worker;

-- ML Worker can SELECT strictly from consented sanitized view in kyc_prod
GRANT SELECT ON kyc_prod.v_consented_training_candidates TO facecheck_ml_worker;

-- Strict Invariant: ML Worker has NO ACCESS to raw KYC customer tables
REVOKE ALL ON kyc_prod.decisions FROM facecheck_ml_worker;
REVOKE ALL ON kyc_prod.encrypted_images FROM facecheck_ml_worker;
REVOKE ALL ON kyc_prod.audit_chain FROM facecheck_ml_worker;

-- --- ROLE: facecheck_readonly ---
-- Zero direct table access. SELECT strictly on sanitized monitoring views
REVOKE ALL ON ALL TABLES IN SCHEMA kyc_prod FROM facecheck_readonly;
REVOKE ALL ON ALL TABLES IN SCHEMA ml_training FROM facecheck_readonly;

-- Grant SELECT only on sanitized monitoring views (no raw images, no unmasked PII)
GRANT SELECT ON kyc_prod.v_monitoring_decisions TO facecheck_readonly;
GRANT SELECT ON kyc_prod.v_monitoring_pipeline_status TO facecheck_readonly;
GRANT SELECT ON kyc_prod.v_monitoring_security_alerts TO facecheck_readonly;
GRANT SELECT ON ml_training.v_monitoring_model_metrics TO facecheck_readonly;
GRANT SELECT ON ml_training.v_monitoring_training_runs TO facecheck_readonly;

-- Strict Invariant: Readonly possesses ZERO INSERT, UPDATE, or DELETE permissions
REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA kyc_prod FROM facecheck_readonly;
REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ml_training FROM facecheck_readonly;

-- ------------------------------------------------------------------------------
-- 5. DEFAULT PRIVILEGES (FUTURE PROOFING)
-- ------------------------------------------------------------------------------
ALTER DEFAULT PRIVILEGES IN SCHEMA kyc_prod REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml_training REVOKE ALL ON TABLES FROM PUBLIC;
