# FaceCheck Offline Models Directory

This directory stores offline pretrained model weights. **Never commit large binary model files to git.**

## Required Models

### 1. OpenCV YuNet Face Detector (ONNX)
- **License**: Apache-2.0 (Open-Source, Commercial Friendly)
- **Format**: ONNX
- **Target Path**: `models/face_detection_yunet_2023mar.onnx`
- **Official Download URL**:
  `https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- **SHA-256**: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

### 2. Deepfake / AI Face Detector (HuggingFace Safetensors or ONNX)
- **License**: MIT / Apache-2.0
- **Format**: `.safetensors` (Pickle-based `.bin` or `.pt` are strictly prohibited)
- **Target Folder**: `models/deepfake_detector/`
- **Files in Folder**:
  - `config.json`
  - `preprocessor_config.json`
  - `model.safetensors`
- **Recommended Open Models**:
  - `umm-maybe/AI-image-detector` (ViT-based, MIT License)
  - `Organika/sdxl-detector` (SafeTensors, Apache-2.0)
- **Alternative ONNX Model**:
  - `models/deepfake_detector.onnx`

## Automatic Download
Run the helper script on an internet-connected machine:
```bash
bash scripts/download_assets.sh
```
Or on Windows:
```powershell
.\scripts\download_assets.ps1
```
