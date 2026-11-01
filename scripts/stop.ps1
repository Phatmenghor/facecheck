# ==============================================================================
# FaceCheck Stack Stopper (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

docker compose stop $args
Write-Host "Services stopped." -ForegroundColor Green
