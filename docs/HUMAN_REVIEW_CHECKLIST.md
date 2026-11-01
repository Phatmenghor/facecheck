# FaceCheck — Human Spot-Check & Security Review Checklist

**Purpose:** Independent human verification checklist for Bank Security, Compliance, and Model Risk Officers.  
**Audience:** Human AppSec Auditor / Bank Risk Officer  
**Rule:** Agent self-review is not independent review. A human auditor must inspect and verify the critical security controls listed below.

---

## 1. Credentials & Sanitization Spot-Check

| File & Line | Control Objective | What to Verify | Human Sign-Off |
| :--- | :--- | :--- | :--- |
| **`scripts/build_review_frontend.py`**<br/>Line 633 | **Zero Secrets in JavaScript Bundle** | Confirm `const API_KEY = "";` is an empty string. The browser client must **never** bundle or leak API keys in source code. | [ ] VERIFIED |
| **`scripts/verify_all.py`**<br/>Line 79 | **Environment Secret Injection** | Confirm `api_key = os.environ.get("FACECHECK_API_KEY")` dynamically loads keys from the environment rather than a hardcoded string. | [ ] VERIFIED |
| **`scripts/scan_git_history.py`** | **Revocation of Historical Credentials** | Confirm historical keys (`fc-prod-secret-key-bank-core-9872`, `fc-prod-secret-key-ad-verify-5431`) are marked revoked and rotated. | [ ] VERIFIED |

---

## 2. Input Hardening & Upload Security (CWE-409 & CWE-434)

| File & Line Range | CWE Vulnerability | Control Implementation to Verify | Human Sign-Off |
| :--- | :--- | :--- | :--- |
| **`app/services/validator.py`**<br/>Lines 181–182, 220–233 | **CWE-409: Decompression Bomb (Pixel Flood)** | - `Image.MAX_IMAGE_PIXELS` is pinned to hardening limits.<br/>- Pre-decode checks verify width/height $\le 4,096$ and pixels $\le 100,000,000$.<br/>- Catches `Image.DecompressionBombError` and raises HTTP 413. | [ ] VERIFIED |
| **`app/services/validator.py`**<br/>Lines 36–51, 78–105 | **CWE-434: Polyglot Executable Rejection** | - Inspects byte arrays against `POLYGLOT_BINARY_SIGNATURES` (`%PDF`, `PK\x03\x04`, `\x7fELF`, PE headers).<br/>- Rejects embedded HTML/PHP scripts (`<script`, `<?php`). | [ ] VERIFIED |
| **`app/services/validator.py`**<br/>Lines 107–149 | **CWE-434: Trailing Data Appended Payload** | - Verifies EOF markers: JPEG `FF D9`, PNG `IEND`, WebP container size.<br/>- Rejects files containing trailing data exceeding tolerance. | [ ] VERIFIED |
| **`app/services/validator.py`**<br/>Lines 192–206 | **MIME / Extension Spoofing** | Confirms binary magic bytes match filename extension strictly; rejects disguised `.jpg` files that contain non-JPEG payloads. | [ ] VERIFIED |

---

## 3. Database Role Isolation & Least-Privilege GRANTs (Deliverable 2)

| File & SQL Section | Role | Mandatory Security Invariant to Verify | Human Sign-Off |
| :--- | :--- | :--- | :--- |
| **`db/roles.sql`**<br/>Section 1 | **`PUBLIC` Revocation** | `REVOKE ALL ON SCHEMA kyc_prod, ml_training FROM PUBLIC;` ensures zero default access to unauthenticated or public connections. | [ ] VERIFIED |
| **`db/roles.sql`**<br/>Section 3 | **`facecheck_app`** | Has `SELECT, INSERT, UPDATE` strictly on operational tables in `kyc_prod`. Has **ZERO** permissions on `ml_training` tables. | [ ] VERIFIED |
| **`db/roles.sql`**<br/>Section 4 | **`facecheck_ml_worker`** | Has read/write in `ml_training`. Has `SELECT` **ONLY** on sanitized view `kyc_prod.v_consented_training_candidates`. Has **ZERO** access to raw customer images or unmasked PII. | [ ] VERIFIED |
| **`db/roles.sql`**<br/>Section 5 | **`facecheck_readonly`** | Has `SELECT` **ONLY** on sanitized monitoring views (`v_monitoring_*`). Has **ZERO** table access, **ZERO** write permissions (`INSERT/UPDATE/DELETE`), and no access to raw biometric data. | [ ] VERIFIED |
| **`db/roles.sql`**<br/>Section 6 | **`facecheck_migrator`** | Dedicated DDL role used only during schema deployments. No application role possesses superuser or DDL authority. | [ ] VERIFIED |

---

## 4. Audit Chain Append-Only Trigger & External Anchoring

| File & Line | Control Objective | What to Verify | Human Sign-Off |
| :--- | :--- | :--- | :--- |
| **`db/migrations/005_audit_chain_trigger_and_anchors.sql`** | **Tamper-Evident Immutability** | Database trigger `trg_prevent_audit_tampering` unconditionally aborts any `UPDATE` or `DELETE` on `kyc_prod.audit_chain`. | [ ] VERIFIED |
| **`scripts/verify_db_audit_chain.py`** | **External WORM Anchor Verification** | Verification script recomputes every block's SHA-256 hash and compares with external anchor records in `kyc_prod.audit_anchors`. | [ ] VERIFIED |
