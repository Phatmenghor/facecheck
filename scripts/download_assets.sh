#!/usr/bin/env bash
# ==============================================================================
# FaceCheck Offline Asset Downloader
# RUN THIS ON AN INTERNET-CONNECTED MACHINE ONLY
# Downloads Python wheels and pretrained model weights into local directories.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE_DIR="$ROOT_DIR/wheelhouse"
MODELS_DIR="$ROOT_DIR/models"
CHECKSUMS_FILE="$MODELS_DIR/CHECKSUMS.txt"

echo "=== 1. Creating directories ==="
mkdir -p "$WHEELHOUSE_DIR"
mkdir -p "$MODELS_DIR/deepfake_detector"

echo "=== 2. Downloading Python wheels into ./wheelhouse ==="
python3 -m pip install --upgrade pip
python3 -m pip download \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 311 \
  --implementation cp \
  -r "$ROOT_DIR/requirements.txt" \
  -d "$WHEELHOUSE_DIR"

echo "=== 3. Downloading YuNet ONNX Face Detector (HuggingFace Mirror) ==="
YUNET_URL="https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_FILE="$MODELS_DIR/face_detection_yunet_2023mar.onnx"

if [ ! -f "$YUNET_FILE" ]; then
    curl -L --fail "$YUNET_URL" -o "$YUNET_FILE"
    echo "Downloaded: $YUNET_FILE"
else
    echo "Already present: $YUNET_FILE"
fi

echo "=== 4. Downloading Hugging Face Detector (SafeTensors, MIT/Apache) ==="
HF_REPO="umm-maybe/AI-image-detector"
HF_BASE="https://huggingface.co/$HF_REPO/resolve/main"

for FILE in "config.json" "preprocessor_config.json" "model.safetensors"; do
    TARGET="$MODELS_DIR/deepfake_detector/$FILE"
    if [ ! -f "$TARGET" ]; then
        echo "Downloading $FILE..."
        curl -L --fail "$HF_BASE/$FILE" -o "$TARGET"
    else
        echo "Already present: $TARGET"
    fi
done

echo "=== 5. Generating SHA256 Checksums ==="
cd "$ROOT_DIR"
cat << 'EOF' > "$CHECKSUMS_FILE"
# Official SHA-256 Checksums for FaceCheck Offline Models
# Format: <sha256_hash>  <relative_filepath>
EOF

if command -v sha256sum &> /dev/null; then
    sha256sum models/face_detection_yunet_2023mar.onnx >> "$CHECKSUMS_FILE"
    sha256sum models/deepfake_detector/model.safetensors >> "$CHECKSUMS_FILE"
elif command -v shasum &> /dev/null; then
    shasum -a 256 models/face_detection_yunet_2023mar.onnx >> "$CHECKSUMS_FILE"
    shasum -a 256 models/deepfake_detector/model.safetensors >> "$CHECKSUMS_FILE"
fi

echo "=== SUCCESS: Offline assets downloaded successfully! ==="
echo "Models placed in: $MODELS_DIR"
echo "Wheels placed in: $WHEELHOUSE_DIR"
echo "Checksums saved to: $CHECKSUMS_FILE"
echo ""
echo "Next step: Transfer this directory (or build docker image) to the offline server."
