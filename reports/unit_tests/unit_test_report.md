# FaceCheck API: Automated Unit Test Report

**Evaluation Date**: 2026-10-22  
**Test Framework**: Pytest 9.1.1 (`pytest-asyncio` strict mode)  
**Execution Runtime**: 3.52 seconds  
**Total Unit Tests Executed**: 83  
**Passing**: 83 (100.0%)  
**Failing**: 0 (0.0%)  
**Skipped**: 0 (0.0%)  

---

## 1. Unit Test Module Summary

| Test Module | Coverage Scope | Tests | Passing | Failing | Execution Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_face_matching.py` | SFace embedding normalization, cosine similarity, 5-point alignment, compare_faces decision calibration, missing model fallback | 10 | 10 | 0 | 0.42s | ✅ PASSED |
| `tests/test_anti_spoofing.py` | Texture print attack detection, 2D Moiré screen replay, PAD unified service decision, Interactive Challenge Manager session & trajectory verification | 9 | 9 | 0 | 0.38s | ✅ PASSED |
| `tests/test_asset_manager.py` | SHA-256 calculation, disk space checks, ZipSlip path traversal protection, quarantine handler, manifest generation | 6 | 6 | 0 | 0.18s | ✅ PASSED |
| `tests/test_face_detector.py` | YuNet face presence detection, fallback cascade, margin cropping bounds | 3 | 3 | 0 | 0.12s | ✅ PASSED |
| `tests/test_quality.py` | Resolution, Laplacian variance blur, underexposure/overexposure luminance, face bounding box size | 6 | 6 | 0 | 0.15s | ✅ PASSED |
| `tests/test_step1_face_check.py` | Single face requirement, multiple face rejection, boundary clipping, head pose yaw/pitch/roll | 10 | 10 | 0 | 0.35s | ✅ PASSED |
| `tests/test_validator.py` | JPEG, PNG, WebP validation, magic byte mismatch, corrupt files, empty uploads, file size limits | 8 | 8 | 0 | 0.22s | ✅ PASSED |
| `tests/test_hardening.py` | Polyglot embedded PDF, ZIP/APK payload rejection, JPEG/PNG trailing data, oversized dimensions | 6 | 6 | 0 | 0.16s | ✅ PASSED |
| `tests/test_auth.py` | Constant-time API key validation, missing key 401, rate limiter token bucket | 4 | 4 | 0 | 0.11s | ✅ PASSED |
| `tests/test_audit_chain.py` | SHA-256 hash chaining, cryptographic record verification, tampering detection | 2 | 2 | 0 | 0.08s | ✅ PASSED |
| `tests/test_eval.py` | ROC-AUC computation (perfect vs random), threshold metric evaluation | 4 | 4 | 0 | 0.10s | ✅ PASSED |
| `tests/test_fusion.py` | Decision score boundaries (Likely Real, Uncertain, Likely Fake), score clamping | 5 | 5 | 0 | 0.09s | ✅ PASSED |
| `tests/test_metadata.py` | EXIF extraction, generative AI signature detection (Stable Diffusion, Midjourney, C2PA) | 4 | 4 | 0 | 0.09s | ✅ PASSED |
| `tests/test_metrics.py` | Prometheus metrics collection, latency recording, authorized vs unauthorized access | 3 | 3 | 0 | 0.08s | ✅ PASSED |
| `tests/test_gender.py` | Biometric gender fallback and synthetic portrait handling | 3 | 3 | 0 | 0.07s | ✅ PASSED |
| **Total** | | **83** | **83** | **0** | **~2.60s** | **100% PASSED** |
