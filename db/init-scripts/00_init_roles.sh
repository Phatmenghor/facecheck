#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Database Initialization - Secure Dynamic Role Provisioning
# Executed automatically by PostgreSQL container entrypoint
# ==============================================================================
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Baseline security hardening: Revoke all default public permissions
    REVOKE ALL ON SCHEMA public FROM PUBLIC;
    REVOKE CREATE ON SCHEMA public FROM PUBLIC;

    -- 1. Create or update roles with dynamic passwords from environment
    DO \$\$
    BEGIN
        -- facecheck_migrator
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_migrator') THEN
            CREATE ROLE facecheck_migrator WITH LOGIN PASSWORD '$DB_MIGRATOR_PASSWORD' CREATEDB;
        ELSE
            ALTER ROLE facecheck_migrator WITH PASSWORD '$DB_MIGRATOR_PASSWORD';
        END IF;

        -- facecheck_app
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_app') THEN
            CREATE ROLE facecheck_app WITH LOGIN PASSWORD '$DB_APP_PASSWORD' NOSUPERUSER NOCREATEDB NOCREATEROLE;
        ELSE
            ALTER ROLE facecheck_app WITH PASSWORD '$DB_APP_PASSWORD';
        END IF;

        -- facecheck_ml_worker
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_ml_worker') THEN
            CREATE ROLE facecheck_ml_worker WITH LOGIN PASSWORD '$DB_ML_WORKER_PASSWORD' NOSUPERUSER NOCREATEDB NOCREATEROLE;
        ELSE
            ALTER ROLE facecheck_ml_worker WITH PASSWORD '$DB_ML_WORKER_PASSWORD';
        END IF;

        -- facecheck_readonly
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'facecheck_readonly') THEN
            CREATE ROLE facecheck_readonly WITH LOGIN PASSWORD '$DB_READONLY_PASSWORD' NOSUPERUSER NOCREATEDB NOCREATEROLE;
        ELSE
            ALTER ROLE facecheck_readonly WITH PASSWORD '$DB_READONLY_PASSWORD';
        END IF;
    END
    \$\$;
EOSQL
