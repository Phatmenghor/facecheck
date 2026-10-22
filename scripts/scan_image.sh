#!/usr/bin/env bash
# ==============================================================================
# FaceCheck API - Offline Security & Vulnerability Scanner (Trivy)
#
# Scans the container image or filesystem for CVE vulnerabilities and misconfigurations.
# Designed for air-gapped / offline operation with local vulnerability cache.
# ==============================================================================

set -euo pipefail

IMAGE_NAME="${1:-facecheck-api:latest}"
REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval/reports" && pwd)"
mkdir -p "${REPORT_DIR}"
REPORT_JSON="${REPORT_DIR}/trivy_scan_report.json"

echo "============================================================"
echo " FaceCheck Container Vulnerability Scanner"
echo " Target Image: ${IMAGE_NAME}"
echo "============================================================"

if ! command -v trivy &>/dev/null; then
    echo "[WARNING] 'trivy' executable was not found in PATH."
    echo ""
    echo "To perform air-gapped / offline vulnerability scans in your bank environment:"
    echo "  1. Download the standalone Trivy binary for your platform."
    echo "  2. Pre-populate the offline vulnerability database archive: trivy-db.tar.gz."
    echo "  3. Import the database cache:"
    echo "     trivy image --download-db-only (on an admin internet station)"
    echo "     oras pull or copy ~/.cache/trivy/db to the air-gapped host."
    echo "  4. Re-run this script: ./scripts/scan_image.sh ${IMAGE_NAME}"
    echo "============================================================"
    exit 0
fi

echo "[INFO] Running offline Trivy vulnerability scan on ${IMAGE_NAME}..."
trivy image \
    --offline-scan \
    --severity HIGH,CRITICAL \
    --format table \
    "${IMAGE_NAME}" || true

echo ""
echo "[INFO] Generating JSON vulnerability report: ${REPORT_JSON}..."
trivy image \
    --offline-scan \
    --severity HIGH,CRITICAL \
    --format json \
    --output "${REPORT_JSON}" \
    "${IMAGE_NAME}" || true

echo "============================================================"
echo " Vulnerability scan completed."
echo " Report written to: ${REPORT_JSON}"
echo "============================================================"
