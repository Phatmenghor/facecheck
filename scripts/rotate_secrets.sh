#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Secret Rotation Runner
# Generates new credentials, updates database roles, writes to .env, and restarts services
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

ENV_FILE="${ROOT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env file not found." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${ENV_FILE}"

POSTGRES_DB="${POSTGRES_DB:-facecheck_db}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

generate_random_hex() {
    local bytes="$1"
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex "${bytes}"
    else
        python -c "import secrets; print(secrets.token_hex(${bytes}))"
    fi
}

echo "=========================================================================="
echo "          FaceCheck Automated Credential Rotation"
echo "=========================================================================="

echo "[1/4] Generating new high-entropy secrets..."
NEW_APP_PASS="$(generate_random_hex 24)"
NEW_WORKER_PASS="$(generate_random_hex 24)"
NEW_READONLY_PASS="$(generate_random_hex 24)"
NEW_MIGRATOR_PASS="$(generate_random_hex 24)"
NEW_API_KEY="fc-live-$(generate_random_hex 24)"
NEW_HMAC_KEY="$(generate_random_hex 32)"

echo "[2/4] Updating credentials in live PostgreSQL database..."
docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1 <<-EOSQL
    ALTER ROLE facecheck_app WITH PASSWORD '${NEW_APP_PASS}';
    ALTER ROLE facecheck_ml_worker WITH PASSWORD '${NEW_WORKER_PASS}';
    ALTER ROLE facecheck_readonly WITH PASSWORD '${NEW_READONLY_PASS}';
    ALTER ROLE facecheck_migrator WITH PASSWORD '${NEW_MIGRATOR_PASS}';
EOSQL
echo "  Live database role passwords updated."

echo "[3/4] Updating local .env configuration..."
# Update keys in .env
sed -i.bak \
    -e "s|^DB_APP_PASSWORD=.*|DB_APP_PASSWORD=${NEW_APP_PASS}|" \
    -e "s|^DB_ML_WORKER_PASSWORD=.*|DB_ML_WORKER_PASSWORD=${NEW_WORKER_PASS}|" \
    -e "s|^DB_READONLY_PASSWORD=.*|DB_READONLY_PASSWORD=${NEW_READONLY_PASS}|" \
    -e "s|^DB_MIGRATOR_PASSWORD=.*|DB_MIGRATOR_PASSWORD=${NEW_MIGRATOR_PASS}|" \
    -e "s|^FACECHECK_API_KEY=.*|FACECHECK_API_KEY=${NEW_API_KEY}|" \
    -e "s|^AUDIT_HMAC_KEY=.*|AUDIT_HMAC_KEY=${NEW_HMAC_KEY}|" \
    "${ENV_FILE}"
rm -f "${ENV_FILE}.bak"
chmod 600 "${ENV_FILE}" 2>/dev/null || true
echo "  .env file updated and secured."

echo "[4/4] Restarting dependent services (api, ml-worker)..."
docker compose restart api ml-worker

echo "Verifying health..."
sleep 3
docker compose ps

echo "=========================================================================="
echo "          Secret Rotation Completed Successfully!"
echo "  * New API Key: ${NEW_API_KEY}"
echo "=========================================================================="
