#!/usr/bin/env bash
# ==============================================================================
# FaceCheck API - Generate CycloneDX SBOM (Software Bill of Materials)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

if ! command -v "${PYTHON_EXEC}" &>/dev/null; then
    PYTHON_EXEC="python"
fi

"${PYTHON_EXEC}" "${SCRIPT_DIR}/make_sbom.py" "$@"
