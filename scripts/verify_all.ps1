# ==============================================================================
# FaceCheck Verification Runner (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "          FaceCheck End-to-End System Verification (PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

$statusSetup = "FAILED"
$statusMigrations = "FAILED"
$statusTests = "FAILED"
$statusSecurity = "FAILED"

Write-Host ">>> [STAGE 1/4] Executing scripts/setup.ps1..." -ForegroundColor Yellow
try {
    & "$rootDir\scripts\setup.ps1"
    $statusSetup = "PASSED"
} catch {
    Write-Host "Setup failed: $_" -ForegroundColor Red
}

Write-Host ">>> [STAGE 2/4] Verifying migrations..." -ForegroundColor Yellow
try {
    & "$rootDir\scripts\migrate.ps1"
    $statusMigrations = "PASSED"
} catch {
    Write-Host "Migrations failed: $_" -ForegroundColor Red
}

Write-Host ">>> [STAGE 3/4] Running in-container test suite..." -ForegroundColor Yellow
try {
    & "$rootDir\scripts\test.ps1"
    $statusTests = "PASSED"
} catch {
    Write-Host "Tests failed: $_" -ForegroundColor Red
}

Write-Host ">>> [STAGE 4/4] Running security scan..." -ForegroundColor Yellow
try {
    python "$rootDir\scripts\scan_git_history.py"
    $statusSecurity = "PASSED"
} catch {
    Write-Host "Security scan failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "          FaceCheck Verification Summary Matrix" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
"{0,-35} | {1,-12}" -f "Verification Stage", "Status"
"------------------------------------+-------------"
"{0,-35} | {1,-12}" -f "1. Docker Stack Setup & Health", $statusSetup
"{0,-35} | {1,-12}" -f "2. DB Schema Migrations & Roles", $statusMigrations
"{0,-35} | {1,-12}" -f "3. In-Container Test Suite & Roles", $statusTests
"{0,-35} | {1,-12}" -f "4. Security Audit & Secrets Scan", $statusSecurity
Write-Host "==========================================================================" -ForegroundColor Cyan

if ($statusSetup -eq "PASSED" -and $statusMigrations -eq "PASSED" -and $statusTests -eq "PASSED" -and $statusSecurity -eq "PASSED") {
    Write-Host "VERDICT: ALL VERIFICATION STAGES PASSED SUCCESSFULLY." -ForegroundColor Green
    exit 0
} else {
    Write-Host "VERDICT: ONE OR MORE VERIFICATION STAGES FAILED." -ForegroundColor Red
    exit 1
}
