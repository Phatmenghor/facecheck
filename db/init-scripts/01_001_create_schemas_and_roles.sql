-- Migration 001: Create Schemas and Base Extensions
-- Idempotent & Reversible

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create distinct database schemas for production KYC and ML training
CREATE SCHEMA IF NOT EXISTS kyc_prod;
CREATE SCHEMA IF NOT EXISTS ml_training;

-- Revoke all privileges from public on both schemas
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA kyc_prod FROM PUBLIC;
REVOKE ALL ON SCHEMA ml_training FROM PUBLIC;
