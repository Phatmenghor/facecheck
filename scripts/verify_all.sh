#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Complete Verification Runner
# Orchestrates Setup, Migrations, In-Container Tests, Security Scans & Backup
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "=========================================================================="
echo "          FaceCheck End-to-End System Verification Suite"
echo "=========================================================================="

STATUS_SETUP="FAILED"
STATUS_MIGRATIONS="FAILED"
STATUS_TESTS="FAILED"
STATUS_SECURITY="FAILED"
STATUS_BACKUP="FAILED"

# Stage 1: Setup & Health
echo ">>> [STAGE 1/5] Executing scripts/setup.sh..."
if "${SCRIPT_DIR}/setup.sh"; then
    STATUS_SETUP="PASSED"
fi

# Stage 2: Migrations & Roles
echo ">>> [STAGE 2/5] Verifying migrations..."
if "${SCRIPT_DIR}/migrate.sh"; then
    STATUS_MIGRATIONS="PASSED"
fi

# Stage 3: In-Docker Test Suite & Role Permissions
echo ">>> [STAGE 3/5] Running in-container test suite..."
if "${SCRIPT_DIR}/test.sh"; then
    STATUS_TESTS="PASSED"
fi

# Stage 4: Automated Security Scans
echo ">>> [STAGE 4/5] Running security audit..."
if "${SCRIPT_DIR}/security_scan.sh"; then
    STATUS_SECURITY="PASSED"
fi

# Stage 5: Encrypted Backup & Restore Test
echo ">>> [STAGE 5/5] Testing encrypted backup..."
if "${SCRIPT_DIR}/backup.sh"; then
    STATUS_BACKUP="PASSED"
fi

echo ""
echo "=========================================================================="
echo "          FaceCheck Verification Summary Matrix"
echo "=========================================================================="
printf "| %-35s | %-12s |\n" "Verification Stage" "Status"
echo "|-------------------------------------|--------------|"
printf "| %-35s | %-12s |\n" "1. Docker Stack Setup & Health" "${STATUS_SETUP}"
printf "| %-35s | %-12s |\n" "2. DB Schema Migrations & Roles" "${STATUS_MIGRATIONS}"
printf "| %-35s | %-12s |\n" "3. In-Container Test Suite & Roles" "${STATUS_TESTS}"
printf "| %-35s | %-12s |\n" "4. Security Audit & Secrets Scan" "${STATUS_SECURITY}"
printf "| %-35s | %-12s |\n" "5. Encrypted Backup Integrity" "${STATUS_BACKUP}"
echo "=========================================================================="

if [[ "${STATUS_SETUP}" == "PASSED" && \
      "${STATUS_MIGRATIONS}" == "PASSED" && \
      "${STATUS_TESTS}" == "PASSED" && \
      "${STATUS_SECURITY}" == "PASSED" && \
      "${STATUS_BACKUP}" == "PASSED" ]]; then
    echo "VERDICT: ALL VERIFICATION STAGES PASSED SUCCESSFULLY."
    exit 0
else
    echo "VERDICT: ONE OR MORE VERIFICATION STAGES FAILED." >&2
    exit 1
fi
