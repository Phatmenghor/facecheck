#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Stack Starter
# Starts all containers in detached mode
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "Starting FaceCheck Docker services..."
docker compose start "$@" || docker compose up -d "$@"

echo "Services started. Current status:"
docker compose ps
