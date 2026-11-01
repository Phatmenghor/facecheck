# ==============================================================================
# FaceCheck Database Migration Runner (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

Write-Host "Applying FaceCheck database migrations..." -ForegroundColor Cyan

if (Test-Path "$rootDir\.env") {
    Get-Content "$rootDir\.env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

$pgPw = $env:POSTGRES_PASSWORD

# 1. db/roles.sql
Write-Host "  [1/6] Running db/roles.sql..."
Get-Content "$rootDir\db\roles.sql" -Raw | docker compose exec -T -e PGPASSWORD="$pgPw" postgres psql -U postgres -d facecheck_db -v ON_ERROR_STOP=1
if ($LASTEXITCODE -ne 0) { exit 1 }

$migrations = @(
    "001_create_schemas_and_roles.sql",
    "002_create_kyc_prod_tables.sql",
    "003_create_ml_training_tables.sql",
    "004_create_sanitized_views_and_grants.sql",
    "005_audit_chain_trigger_and_anchors.sql"
)

$step = 2
foreach ($m in $migrations) {
    Write-Host "  [$step/6] Applying $m..."
    Get-Content "$rootDir\db\migrations\$m" -Raw | docker compose exec -T -e PGPASSWORD="$pgPw" postgres psql -U postgres -d facecheck_db -v ON_ERROR_STOP=1
    if ($LASTEXITCODE -ne 0) { exit 1 }
    $step++
}

Write-Host "Database migrations successfully applied and verified." -ForegroundColor Green
