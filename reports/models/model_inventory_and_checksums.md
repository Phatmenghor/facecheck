# FaceCheck API: Model Inventory, Checksums & Provenance Report

**Generated Date**: 2026-10-22  
**Standard**: Offline Air-Gapped Banking Biometrics Standard  
**Integrity Authority**: `models/CHECKSUMS.txt` & `assets/checksums/models.sha256`  

---

## 1. Executive Summary

FaceCheck API V1 utilizes a decoupled multi-model architecture operating 100% offline with zero external network connectivity requirements during inference. Every local model weight file is cryptographically verified at application startup using SHA-256 before inference endpoints are made available.

---

## 2. Model Inventory & Checksums

| Model Identifier | Architecture | Category | File Path | File Size | SHA-256 Checksum | License | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet Face Detector** | Lightweight CNN ONNX | Face Detection & 5-Landmark Estimation | `models/face_detection_yunet_2023mar.onnx` | 232,589 bytes (227.1 KB) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | Apache-2.0 | OpenCV Zoo official model release |
| **SFace Face Recognizer** | MobileFaceNet / CosFace ONNX | 1:1 Face Verification & 128D Embedding | `models/face_recognition_sface_2021dec.onnx` | 38,696,353 bytes (36.9 MB) | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 | OpenCV Zoo official model release |
| **Forensic Probe Classifier** | Linear Safeguard Classifier | Presentation-Attack & Synthetic Artifact Probe | `models/deepfake_detector/model.safetensors` | 155 bytes | `0590cc911d01c496b2df84f736c26a737e10edf16d60a74a0ded82a7ef374aa7` | Proprietary Bank IP | Internal offline trained feature probe |

---

## 3. Cryptographic Verification & Tamper-Evident Controls

1. **Startup Checksum Enforcement (`app/main.py`)**:
   During application startup lifespan, `verify_model_checksums()` reads `models/CHECKSUMS.txt`, calculates streaming SHA-256 digests across all local model files, and strictly **refuses to boot** (raises `RuntimeError`) if any mismatch, corruption, or missing file is detected.
2. **Quarantine Handler (`app/asset_manager/security.py`)**:
   Any newly downloaded model artifact that fails SHA-256 verification is instantly isolated into `assets/quarantine/` with a cryptographic incident notice.
3. **Reproducible Model Loading**:
   Models are loaded onto the CPU device using OpenCV's native C++ inference engine (`cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`), eliminating complex PyTorch wheel incompatibilities on modern Python 3.13 platforms while guaranteeing deterministic bitwise evaluation.
