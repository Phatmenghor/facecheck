#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Stack Restarter
# Restarts all or specified containers
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "Restarting FaceCheck Docker services..."
docker compose restart "$@"

echo "Current status:"
docker compose ps
