# ==============================================================================
# FaceCheck Stack Starter (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

docker compose up -d $args
docker compose ps
