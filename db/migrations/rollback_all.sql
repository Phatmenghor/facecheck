-- Rollback Migration: Teardown all FaceCheck Schemas, Tables, Triggers & Roles
-- Reversible & Idempotent (Strictly for development / disaster recovery)

-- 1. Drop Triggers & Functions
DROP TRIGGER IF EXISTS trg_prevent_anchor_mutation ON kyc_prod.audit_anchors;
DROP TRIGGER IF EXISTS trg_prevent_audit_mutation ON kyc_prod.audit_chain;
DROP FUNCTION IF EXISTS kyc_prod.fn_prevent_audit_chain_mutation();

-- 2. Drop Sanitized Views
DROP VIEW IF EXISTS kyc_prod.v_monitoring_security_alerts;
DROP VIEW IF EXISTS ml_training.v_monitoring_training_runs;
DROP VIEW IF EXISTS ml_training.v_monitoring_model_metrics;
DROP VIEW IF EXISTS kyc_prod.v_monitoring_pipeline_status;
DROP VIEW IF EXISTS kyc_prod.v_monitoring_decisions;
DROP VIEW IF EXISTS kyc_prod.v_consented_training_candidates;
DROP VIEW IF EXISTS ml_training.v_eligible_training_samples;

-- 3. Drop Tables
DROP TABLE IF EXISTS ml_training.training_run_audit CASCADE;
DROP TABLE IF EXISTS ml_training.golden_test_set CASCADE;
DROP TABLE IF EXISTS ml_training.model_registry CASCADE;
DROP TABLE IF EXISTS ml_training.training_samples CASCADE;
DROP TABLE IF EXISTS ml_training.consent_records CASCADE;

DROP TABLE IF EXISTS kyc_prod.audit_anchors CASCADE;
DROP TABLE IF EXISTS kyc_prod.retention_policies CASCADE;
DROP TABLE IF EXISTS kyc_prod.kill_switch_state CASCADE;
DROP TABLE IF EXISTS kyc_prod.audit_chain CASCADE;
DROP TABLE IF EXISTS kyc_prod.encrypted_images CASCADE;
DROP TABLE IF EXISTS kyc_prod.decisions CASCADE;

-- 4. Drop Schemas
DROP SCHEMA IF EXISTS ml_training CASCADE;
DROP SCHEMA IF EXISTS kyc_prod CASCADE;
