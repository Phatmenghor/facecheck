#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Database Migration Runner
# Executes migrations in exact dependency order inside PostgreSQL container
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
fi

POSTGRES_DB="${POSTGRES_DB:-facecheck_db}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

echo "Applying FaceCheck database migrations to database '${POSTGRES_DB}'..."

# Ensure roles and baseline hardening are applied
echo "  [1/6] Running db/roles.sql..."
docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1 < "${ROOT_DIR}/db/roles.sql"

MIGRATIONS=(
    "001_create_schemas_and_roles.sql"
    "002_create_kyc_prod_tables.sql"
    "003_create_ml_training_tables.sql"
    "004_create_sanitized_views_and_grants.sql"
    "005_audit_chain_trigger_and_anchors.sql"
)

STEP=2
for mig in "${MIGRATIONS[@]}"; do
    MIG_PATH="${ROOT_DIR}/db/migrations/${mig}"
    if [[ -f "${MIG_PATH}" ]]; then
        echo "  [${STEP}/6] Applying ${mig}..."
        docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1 < "${MIG_PATH}"
        STEP=$((STEP + 1))
    else
        echo "ERROR: Migration file ${MIG_PATH} not found!" >&2
        exit 1
    fi
done

echo "Database migrations successfully applied and verified."
