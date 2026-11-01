#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Development Reset Script
# Strictly DEVELOPMENT only - Permanently tears down containers and persistent volumes
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1091
    source "${ENV_FILE}"
fi

CURRENT_ENV="${ENV:-development}"

# Safety Gate 1: Refuse non-development environment
if [[ "${CURRENT_ENV}" != "development" ]]; then
    echo "==========================================================================" >&2
    echo "FATAL: reset_dev.sh REFUSED!" >&2
    echo "Current environment is '${CURRENT_ENV}'. Development reset is ONLY allowed when ENV=development." >&2
    echo "==========================================================================" >&2
    exit 1
fi

echo "=========================================================================="
echo "          FaceCheck Development Environment Reset"
echo "=========================================================================="
echo "WARNING: This will DESTROY all local database containers, Docker volumes,"
echo "and synthetic test state (postgres_data, model_store, etc.)."
echo ""

CONFIRMED=0
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
    CONFIRMED=1
else
    read -r -p "Are you sure you want to proceed with reset? (Type 'yes' to confirm): " response
    if [[ "${response}" == "yes" || "${response}" == "y" ]]; then
        CONFIRMED=1
    fi
fi

if [[ ${CONFIRMED} -eq 0 ]]; then
    echo "Reset aborted by user."
    exit 0
fi

echo "Stopping containers and wiping volumes..."
docker compose down -v --remove-orphans

echo "Development environment has been completely reset."
echo "To re-initialize from scratch, execute: scripts/setup.sh"
