#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Test Suite Runner (Inside Docker Container)
# Executes all unit, integration, security, and DB role grant tests
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

TEST_TARGET="${1:-tests/}"

echo "=========================================================================="
echo "          Running FaceCheck Test Suite Inside Docker API Container"
echo "=========================================================================="
echo "Target: ${TEST_TARGET}"

docker compose exec -T api python -m pytest "${TEST_TARGET}" -v

echo ""
echo "=========================================================================="
echo "          Direct PostgreSQL Live Role & GRANT Verification"
echo "=========================================================================="

if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
fi

POSTGRES_DB="${POSTGRES_DB:-facecheck_db}"

# Test 1: facecheck_app cannot write to ml_training
echo -n "  1. Verifying facecheck_app is BLOCKED from writing to ml_training..."
APP_ERR=$(docker compose exec -T -e PGPASSWORD="${DB_APP_PASSWORD}" postgres psql -U facecheck_app -d "${POSTGRES_DB}" -c "INSERT INTO ml_training.consent_records (consent_id, subject_pseudonym, consent_status, consented_purposes, signature_hmac) VALUES ('x', 'y', 'ACTIVE', '[\"KYC\"]', 'h');" 2>&1 || true)
if echo "${APP_ERR}" | grep -q "permission denied"; then
    echo " PASSED (Permission denied as expected)"
else
    echo " FAILED! Expected permission denied, got: ${APP_ERR}" >&2
    exit 1
fi

# Test 2: facecheck_ml_worker cannot write to kyc_prod.decisions
echo -n "  2. Verifying facecheck_ml_worker is BLOCKED from modifying kyc_prod.decisions..."
WORKER_ERR=$(docker compose exec -T -e PGPASSWORD="${DB_ML_WORKER_PASSWORD}" postgres psql -U facecheck_ml_worker -d "${POSTGRES_DB}" -c "DELETE FROM kyc_prod.decisions;" 2>&1 || true)
if echo "${WORKER_ERR}" | grep -q "permission denied"; then
    echo " PASSED (Permission denied as expected)"
else
    echo " FAILED! Expected permission denied, got: ${WORKER_ERR}" >&2
    exit 1
fi

# Test 3: facecheck_readonly cannot write to any table
echo -n "  3. Verifying facecheck_readonly is strictly READ-ONLY..."
RO_ERR=$(docker compose exec -T -e PGPASSWORD="${DB_READONLY_PASSWORD}" postgres psql -U facecheck_readonly -d "${POSTGRES_DB}" -c "INSERT INTO kyc_prod.retention_policies (policy_name, retention_days, description) VALUES ('test', 1, 'test');" 2>&1 || true)
if echo "${RO_ERR}" | grep -q "permission denied"; then
    echo " PASSED (Permission denied as expected)"
else
    echo " FAILED! Expected permission denied, got: ${RO_ERR}" >&2
    exit 1
fi

echo "=========================================================================="
echo "          All In-Container Tests & Role Grants PASSED!"
echo "=========================================================================="
