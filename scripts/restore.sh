#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Encrypted Database Restore Runner
# Decrypts in-memory stream and restores into PostgreSQL container
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
fi

BACKUP_FILE="${1:-}"

if [[ -z "${BACKUP_FILE}" || ! -f "${BACKUP_FILE}" ]]; then
    echo "ERROR: Valid encrypted backup file required as first argument." >&2
    echo "Usage: $0 backups/facecheck_backup_YYYYMMDD_HHMMSSZ.sql.enc" >&2
    exit 1
fi

POSTGRES_DB="${POSTGRES_DB:-facecheck_db}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"

ENC_KEY="${ENCRYPTION_MASTER_KEY:-}"
if [[ -z "${ENC_KEY}" ]]; then
    echo "ERROR: ENCRYPTION_MASTER_KEY is not set in environment or .env." >&2
    exit 1
fi

# Verify checksum if companion file exists
CHECKSUM_FILE="${BACKUP_FILE}.sha256"
if [[ -f "${CHECKSUM_FILE}" ]]; then
    echo "Verifying SHA-256 integrity checksum..."
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "${CHECKSUM_FILE}"
    else
        python -c "
import hashlib, sys
exp = open(sys.argv[2]).read().split()[0]
actual = hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
if exp != actual:
    sys.exit(1)
" "${BACKUP_FILE}" "${CHECKSUM_FILE}"
    fi
    echo "  Checksum verified OK."
fi

echo "Restoring database from encrypted archive: ${BACKUP_FILE}..."

# Decrypt in-memory and pipe directly into psql (zero decrypted plaintext written to disk)
openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 -pass "pass:${ENC_KEY}" -in "${BACKUP_FILE}" \
    | docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1

echo "Database restore completed successfully."
