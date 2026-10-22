# FaceCheck Offline Asset Downloader (PowerShell for Windows)
# RUN ON AN INTERNET-CONNECTED MACHINE ONLY

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$wheelhouseDir = Join-Path $rootDir "wheelhouse"
$modelsDir = Join-Path $rootDir "models"
$detectorDir = Join-Path $modelsDir "deepfake_detector"
$checksumFile = Join-Path $modelsDir "CHECKSUMS.txt"

Write-Host "=== 1. Creating directories ==="
New-Item -ItemType Directory -Force -Path $wheelhouseDir | Out-Null
New-Item -ItemType Directory -Force -Path $detectorDir | Out-Null

Write-Host "=== 2. Downloading Python wheels into ./wheelhouse ==="
py -m pip download -r "$rootDir\requirements.txt" -d $wheelhouseDir

Write-Host "=== 3. Downloading YuNet ONNX Face Detector (HuggingFace Mirror) ==="
$yunetUrl = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
$yunetFile = Join-Path $modelsDir "face_detection_yunet_2023mar.onnx"
if (-not (Test-Path $yunetFile)) {
    Invoke-WebRequest -Uri $yunetUrl -OutFile $yunetFile
    Write-Host "Downloaded: $yunetFile"
} else {
    Write-Host "Already present: $yunetFile"
}

Write-Host "=== 4. Downloading Hugging Face Detector (SafeTensors) ==="
$hfBase = "https://huggingface.co/umm-maybe/AI-image-detector/resolve/main"
$hfFiles = @("config.json", "preprocessor_config.json", "model.safetensors")

foreach ($file in $hfFiles) {
    $target = Join-Path $detectorDir $file
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $file..."
        Invoke-WebRequest -Uri "$hfBase/$file" -OutFile $target
    } else {
        Write-Host "Already present: $target"
    }
}

Write-Host "=== 5. Writing SHA256 Checksums ==="
$yunetHash = (Get-FileHash -Algorithm SHA256 $yunetFile).Hash.ToLower()
$modelHash = (Get-FileHash -Algorithm SHA256 (Join-Path $detectorDir "model.safetensors")).Hash.ToLower()

$checksumContent = @"
# Official SHA-256 Checksums for FaceCheck Offline Models
$yunetHash  models/face_detection_yunet_2023mar.onnx
$modelHash  models/deepfake_detector/model.safetensors
"@

Set-Content -Path $checksumFile -Value $checksumContent -Encoding UTF8
Write-Host "=== SUCCESS: Assets downloaded and checksums written to $checksumFile ==="
