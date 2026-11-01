#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Stack Stopper
# Stops running containers gracefully
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "Stopping FaceCheck Docker services..."
docker compose stop "$@"

echo "Services stopped."
