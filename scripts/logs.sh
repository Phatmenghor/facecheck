#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Stack Logs Viewer
# Follows logs for all or specified services (e.g. scripts/logs.sh api)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

docker compose logs -f "$@"
