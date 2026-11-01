# ==============================================================================
# FaceCheck In-Docker Test Suite Runner (PowerShell)
# ==============================================================================
param(
    [string]$Target = "tests/"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "          Running FaceCheck Test Suite Inside Docker API Container" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target"

docker compose exec -T api python -m pytest $Target -v
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "          Direct PostgreSQL Live Role & GRANT Verification" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

if (Test-Path "$rootDir\.env") {
    Get-Content "$rootDir\.env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

$appPw = $env:DB_APP_PASSWORD
$workerPw = $env:DB_ML_WORKER_PASSWORD
$roPw = $env:DB_READONLY_PASSWORD

# Test 1: facecheck_app cannot write to ml_training
Write-Host -NoNewline "  1. Verifying facecheck_app is BLOCKED from writing to ml_training..."
$appErr = docker compose exec -T -e PGPASSWORD="$appPw" postgres psql -U facecheck_app -d facecheck_db -c "INSERT INTO ml_training.consent_records (consent_id, subject_pseudonym, consent_status, consented_purposes, signature_hmac) VALUES ('x', 'y', 'ACTIVE', '[\"KYC\"]', 'h');" 2>&1
if ($appErr -like "*permission denied*") {
    Write-Host " PASSED (Permission denied as expected)" -ForegroundColor Green
} else {
    Write-Host " FAILED! Expected permission denied, got: $appErr" -ForegroundColor Red
    exit 1
}

# Test 2: facecheck_ml_worker cannot write to kyc_prod.decisions
Write-Host -NoNewline "  2. Verifying facecheck_ml_worker is BLOCKED from modifying kyc_prod.decisions..."
$workerErr = docker compose exec -T -e PGPASSWORD="$workerPw" postgres psql -U facecheck_ml_worker -d facecheck_db -c "DELETE FROM kyc_prod.decisions;" 2>&1
if ($workerErr -like "*permission denied*") {
    Write-Host " PASSED (Permission denied as expected)" -ForegroundColor Green
} else {
    Write-Host " FAILED! Expected permission denied, got: $workerErr" -ForegroundColor Red
    exit 1
}

# Test 3: facecheck_readonly cannot write to any table
Write-Host -NoNewline "  3. Verifying facecheck_readonly is strictly READ-ONLY..."
$roErr = docker compose exec -T -e PGPASSWORD="$roPw" postgres psql -U facecheck_readonly -d facecheck_db -c "INSERT INTO kyc_prod.retention_policies (policy_name, retention_days, description) VALUES ('test', 1, 'test');" 2>&1
if ($roErr -like "*permission denied*") {
    Write-Host " PASSED (Permission denied as expected)" -ForegroundColor Green
} else {
    Write-Host " FAILED! Expected permission denied, got: $roErr" -ForegroundColor Red
    exit 1
}

Write-Host "==========================================================================" -ForegroundColor Green
Write-Host "          All In-Container Tests & Role Grants PASSED!" -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
