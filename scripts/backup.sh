#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Encrypted Database Backup Runner
# Exports schemas kyc_prod & ml_training, and encrypts using AES-256-CBC (PBKDF2)
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
BACKUP_DIR="${ROOT_DIR}/backups"
TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
BACKUP_FILE="${BACKUP_DIR}/facecheck_backup_${TIMESTAMP}.sql.enc"
CHECKSUM_FILE="${BACKUP_FILE}.sha256"

mkdir -p "${BACKUP_DIR}"

echo "Starting encrypted backup for FaceCheck database '${POSTGRES_DB}'..."

ENC_KEY="${ENCRYPTION_MASTER_KEY:-}"
if [[ -z "${ENC_KEY}" ]]; then
    echo "ERROR: ENCRYPTION_MASTER_KEY is not set in environment or .env." >&2
    exit 1
fi

# Stream pg_dump directly into OpenSSL encryption pipeline (zero plaintext on disk)
docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres pg_dump \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    --schema=kyc_prod \
    --schema=ml_training \
    --format=plain \
    --no-owner \
    --no-privileges \
    | openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 -pass "pass:${ENC_KEY}" -out "${BACKUP_FILE}"

chmod 600 "${BACKUP_FILE}"

# Compute SHA-256 checksum of encrypted file
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${BACKUP_FILE}" > "${CHECKSUM_FILE}"
else
    python -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest() + '  ' + sys.argv[1])" "${BACKUP_FILE}" > "${CHECKSUM_FILE}"
fi

FILE_SIZE="$(du -h "${BACKUP_FILE}" | cut -f1)"

echo "Encrypted backup successfully created:"
echo "  File:     ${BACKUP_FILE}"
echo "  Size:     ${FILE_SIZE}"
echo "  Checksum: ${CHECKSUM_FILE}"
