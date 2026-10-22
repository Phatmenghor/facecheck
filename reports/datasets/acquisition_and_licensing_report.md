# FaceCheck API: Dataset Acquisition, Licensing & Provenance Report

**Generated Date**: 2026-10-22T08:48:49.754031+00:00  
**Configurable Sample Target (`TARGET_SAMPLES`)**: 10,000  
**Storage Volume**: `D:\CP Bank\account-online\account_online\facecheck_api\assets` (251.24 GB available disk space)  
**Offline Guarantee**: Offline-ready; all models and acquired fixtures operate air-gapped without remote calls.

---

## 1. Executive Summary & Policy Compliance

Biometric facial imagery constitutes sensitive personal data requiring strict legal basis, explicit consent, and licensing compliance. In accordance with **Phase 4** specifications:

1. **Anti-Scraping / No-Bypass Rule**: Automated scraping that bypasses authentication, CAPTCHAs, or usage agreements is strictly prohibited.
2. **Research-Only Restriction**: Datasets restricted to non-commercial academic research (e.g., CelebA-Spoof, OULU-NPU, Idiap Replay-Attack) **must not** be used in a production banking workflow without formal bilateral licensing.
3. **Manual Approval Tracking**: Any dataset requiring formal registration, signing a release form, or manual approval has its download halted and status marked **BLOCKED_MANUAL_APPROVAL_REQUIRED**.
4. **Target Status Disclosure**: Because candidate external datasets require manual institutional review and agreements, the target of 10,000 external licensed samples is explicitly documented as **UNMET (Awaiting Formal Data Release Agreements)**.
5. **No Label Fabrication**: Ordinary unlabelled web images are never substituted to fabricate artificial accuracy claims.

---

## 2. Candidate Dataset Audit & Terms Inventory

| Dataset | Official Source | License / Terms | Commercial Allowed? | Reachability | Acquisition Status | Required Manual Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **celeba_spoof** | [celeba_spoof](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html) | Non-Commercial / Research-Only | ❌ No (Research Only) | HTTP 200 | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires signing agreement with CUHK MMLab. Commercial use strictly prohibited. |
| **oulu_npu** | [oulu_npu](https://sites.google.com/site/oulunpudatabase/) | Non-Commercial / Academic Research | ❌ No (Research Only) | HTTP 200 | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires formal written request to University of Oulu. Automated download not permitted. |
| **faceforensics_plus_plus** | [faceforensics_plus_plus](https://github.com/ondyari/faceforensics) | TUM Research License | ❌ No (Research Only) | HTTP 200 | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires completing Google Form request to Technical University of Munich. Automated scraping blocked. |
| **casia_face_anti_spoofing** | [casia_face_anti_spoofing](http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp) | Academic Research Agreement | ❌ No (Research Only) | HTTP 200 | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires signed release agreement to CBSR CASIA. Not for commercial systems. |
| **replay_attack** | [replay_attack](https://www.idiap.ch/en/scientific-research/data/replayattack) | Idiap Research Data License | ❌ No (Research Only) | Err: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)> | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires registration and user agreement with Idiap Research Institute. |
| **siw_mv2** | [siw_mv2](https://cvlab.cse.msu.edu/siw-mv2-dataset.html) | MSU Research Agreement | ❌ No (Research Only) | Err: <urlopen error timed out> | `BLOCKED_MANUAL_APPROVAL_REQUIRED` | Requires signed release agreement to Michigan State University CVLAB. |

---

## 3. Manual Steps Required to Obtain Blocked Datasets

### A. CelebA-Spoof (CUHK MMLab)
- **Source**: https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html
- **Terms**: Non-commercial academic research only.
- **Manual Steps**:
  1. Complete institutional agreement with The Chinese University of Hong Kong (CUHK).
  2. Designate principal investigator (PI) and institutional email domain.
  3. Obtain Google Drive / Baidu Wangpan access credentials.
  4. Place extracted archives in `assets/datasets/celeba_spoof/` and verify SHA-256 with `AssetManager`.

### B. OULU-NPU Database (University of Oulu)
- **Source**: https://sites.google.com/site/oulunpudatabase/
- **Terms**: Academic research only.
- **Manual Steps**:
  1. Download and sign the OULU-NPU Database Release Agreement.
  2. Send scanned PDF from institutional email to dataset curators at University of Oulu.
  3. Receive download link and decryption key.
  4. Place files in `assets/datasets/oulu_npu/`.

### C. Replay-Attack (Idiap Research Institute)
- **Source**: https://www.idiap.ch/en/scientific-research/data/replayattack
- **Terms**: Idiap Research Data Agreement.
- **Manual Steps**:
  1. Register account on Idiap Research portal.
  2. Accept dataset user agreement regarding biometric privacy.
  3. Download tar archives via Idiap authorized tokens.
  4. Verify archive with `AssetManager.verify_all_assets()`.

### D. FaceForensics++ (TUM)
- **Source**: https://github.com/ondyari/faceforensics
- **Terms**: Technical University of Munich research license.
- **Manual Steps**:
  1. Submit the official Google Form application specifying research intent.
  2. Receive Python download script with temporary download tokens.
  3. Place downloaded sequences in `assets/datasets/faceforensics/`.

---

## 4. Benchmark Dataset Strategy for Phase 10 Automated Testing

To ensure 100% reproducible, verifiable testing across face detection, image quality, face matching (genuine & impostor pairs), and presentation attack detection (print & screen attacks) without violating data privacy or copyright terms:

1. **Deterministic Benchmark Suite**: `assets/datasets/facecheck_v1_benchmark/` contains labeled, deterministic synthetic and permissible test subjects.
2. **Ground-Truth Manifest**: Every test file has an entry in `manifest.json` recording SHA-256 hash, category, identity ID, attack status, expected face count, and quality parameters.
3. **Partitioning**: Training, validation, and test splits are strictly separated to prevent subject leakage.
4. **Streaming Batch Evaluation**: Evaluated using `scripts/evaluate_dataset.py` with configurable batch size (`--batch-size 32`) and sample limit (`--target-samples`).
