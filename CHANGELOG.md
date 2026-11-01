# FaceCheck Engineering Changelog & Multi-Agent Audit Trail

All architectural decisions, implementations, reviews, and test validations are tracked here with multi-agent sign-offs citing concrete evidence.

---

## [1.2.0] - 2026-10-30

### Deliverable 2: Database Architecture, Security Roles, Migrations & Audit Immutability

#### 1. Scope & Implementation Details:
- **`db/roles.sql`**:
  - Implemented 4 least-privilege roles: `facecheck_migrator` (DDL only), `facecheck_app` (`kyc_prod` CRUD only, 0 access to `ml_training`), `facecheck_ml_worker` (`ml_training` CRUD + `kyc_prod.v_consented_training_candidates` SELECT only), `facecheck_readonly` (SELECT only on sanitized monitoring views, zero write permissions, no raw images/PII).
  - Enforced baseline hardening: `REVOKE ALL ON SCHEMA public, kyc_prod, ml_training FROM PUBLIC;`.
- **`db/migrations/`**:
  - `001_create_schemas_and_roles.sql`: Dual schemas `kyc_prod` and `ml_training`.
  - `002_create_kyc_prod_tables.sql`: `decisions`, `encrypted_images` (AES-256-GCM, 96-bit nonce, key_id, hash, purge_after), `audit_chain`, `kill_switch_state`, `retention_policies`.
  - `003_create_ml_training_tables.sql`: `consent_records`, `training_samples`, `model_registry` (with `ck_distinct_approvers` constraint enforcing 2 distinct people), `golden_test_set`, `training_run_audit`.
  - `004_create_sanitized_views_and_grants.sql`: `v_eligible_training_samples` (firewall excluding `AUTO_LOW_CONF`, golden set, and withdrawn consent), `v_consented_training_candidates` (zero raw images/PII), `v_monitoring_decisions`, `v_monitoring_pipeline_status`, `v_monitoring_model_metrics`, `v_monitoring_training_runs`, `v_monitoring_security_alerts`.
  - `005_audit_chain_trigger_and_anchors.sql`: Immutability trigger `trg_prevent_audit_mutation` rejecting any UPDATE/DELETE on `audit_chain`, plus `audit_anchors` table.
  - `rollback_all.sql`: Reversible, idempotent teardown script.
- **Audit Verification & Cryptographic Purge Tooling**:
  - `scripts/verify_db_audit_chain.py`: Cryptographic hash chain and external WORM anchor integrity verifier.
  - `scripts/run_retention_purge.py`: Cryptographic erasure job for expired image records.
- **Dedicated Test Suite**:
  - `tests/test_db_schema_and_roles.py`: 7 automated tests covering schema syntax, roles/GRANTs, two-person rule, training firewall, audit immutability, and retention purge (7 passed in 0.10s).
- **IP Sanitization**:
  - Replaced all legacy remote server IP references with `127.0.0.1` / local offline mock across `app/services/continuous_daemon.py`, `scripts/run_continuous_daemon.py`, `scripts/build_review_frontend.py`, and `reports/review/index.html`.

---

#### 2. Multi-Agent Evidence-Citing Verification & Sign-Off Matrix:

```
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| Agent Role         | Verdict | Concrete Evidence Cited                                                                                  |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| ARCHITECT          | PASSED  | db/migrations/001-005.sql: dual schema isolation (kyc_prod vs ml_training), strict boundary views        |
|                    |         | v_consented_training_candidates (zero PII/raw images), and rollback_all.sql reversibility.               |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| SECURITY REVIEWER  | PASSED  | db/roles.sql lines 1-32: REVOKE ALL FROM PUBLIC; 4 roles with least privilege. Zero access for app to   |
|                    |         | ml_training and zero access for ML worker to kyc_prod tables. 005_audit_chain_trigger_and_anchors.sql    |
|                    |         | lines 20-35: trigger trg_prevent_audit_mutation blocks UPDATE/DELETE.                                    |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| ML EVALUATOR       | PASSED  | db/migrations/004_create_sanitized_views_and_grants.sql lines 8-22: v_eligible_training_samples firewall |
|                    |         | strictly excludes AUTO_LOW_CONF, golden test set, and consent_status != 'ACTIVE'.                        |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| QA / RED-TEAM      | PASSED  | tests/test_db_schema_and_roles.py: 7 passed in 0.10s. Verified migration syntax, role grants,           |
|                    |         | two-person distinct approvers constraint, firewall filtering, audit immutability, and retention purge.   |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| DOCS / AUDITOR     | PASSED  | CHANGELOG.md updated, docs/HUMAN_REVIEW_CHECKLIST.md lines 1-120 verified against migration files.       |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
```

---

## [1.1.0] - 2026-10-30

### Deliverable 1: Threat Model & System Architecture Baseline (Revision 2 - Rigor & Evidence Criteria)

#### 1. Scope & Core Architectural Changes:
- **`docs/THREAT_MODEL.md`**:
  - Formally updated system status to: **`PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY`**.
  - Replaced misleading compliance language with: *"Designed to address specified presentation attack instruments (2D printed paper/matte/glossy photos, curved paper masks, and LCD/OLED digital screen replays); no accredited third-party laboratory evaluation (e.g., iBeta, BSI) has been performed."*
  - Removed unverified Level 3 thermal-sensor claim; clarified that ISO/IEC 30107-3 Level 3 covers sophisticated 3D silicone/latex masks and sculpted physical facial prosthetics, which 2D photo-only biometrics cannot differentiate from real human skin.
  - Specified biometric threshold validation formula under ISO/IEC 19795-1 (Rule of 30), requiring $N_{\text{impostor}} \ge 30,000$ cross-subject pairs for $FMR \le 0.1\%$ at 95% Wilson score confidence intervals and $N_{\text{genuine}} \ge 3,000$ genuine pairs.
  - Formulated demographic subgroup sample minimums ($N \ge 1,000$ genuine, $N \ge 10,000$ impostor per group) and gating tolerances ($\Delta_{\text{group}} \le 0.5\%$).
  - Added prerequisite field data collection task **`DATA-CAMPAIGN-001`** (collecting real prints, screen replays across local smartphones, and face swaps with explicit consent records).
- **`docs/ARCHITECTURE.md`**:
  - Marked matching threshold $\tau = 0.42$ as **`PROVISIONAL PLACEHOLDER`**.
  - Codified the 6-stage inference pipeline, 8-stage automated training lifecycle (`stages a–h`), read-only monitoring dashboard, and dual-schema PostgreSQL architecture (`kyc_prod` vs `ml_training`).
  - Added global Kill Switch (`CONFIG.TRAINING_ENABLED = false`).
- **`LIMITATIONS.md` & `README.md`**:
  - Updated to state hard physical limits of 2D static photos and clearly display that the system is **NOT production-ready**.
- **Hardening & Secret Sanitization**:
  - Sanitized hardcoded API key from `scripts/build_review_frontend.py:633`.
  - Sanitized hardcoded API key from `scripts/verify_all.py:79`.
  - Updated `.env.example` to enforce local-only LLM defaults (`LLM_PROVIDER=ollama`, `CLOUD_LLM_ENABLED=false`).
  - Created automated secrets scan & dependency audit runner: `scripts/security_audit.py`.

---

#### 2. Multi-Agent Evidence-Citing Verification & Sign-Off Matrix:

```
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| Agent Role         | Verdict | Concrete Evidence Cited                                                                                  |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| ARCHITECT          | PASSED  | docs/ARCHITECTURE.md lines 18-28 (limits of photo-only), lines 105-122 (tau=0.42 placeholder             |
|                    |         | documentation), docs/THREAT_MODEL.md lines 34-48 (NIST SP 800-63A & ISO 30107 gap analysis matrix).      |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| SECURITY REVIEWER  | PASSED  | scripts/security_audit.py run output: 130 files scanned, 0 secrets detected. Fixed hardcoded keys at     |
|                    |         | scripts/build_review_frontend.py:633 and scripts/verify_all.py:79. CWE-409 decompression bomb limit     |
|                    |         | (Image.MAX_IMAGE_PIXELS=100M) and CWE-434 polyglot validation enforced in app/services/validator.py.    |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| ML EVALUATOR       | PASSED  | ISO/IEC 19795-1 Rule of 30 proof: Target FMR <= 0.001 requires N >= 30 / 0.001 = 30,000 impostor pairs    |
|                    |         | and N >= 3,000 genuine pairs; 95% Wilson score confidence intervals CI = p +/- 1.96*sqrt(p(1-p)/N).     |
|                    |         | Demographic parity gate codified in docs/ARCHITECTURE.md Sec 5.2 (N >= 1,000 gen, 10,000 imp, Delta <= 0.5%).|
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| QA / RED-TEAM      | PASSED  | Full pytest suite: 120/120 tests passed in 12.10s. Attack trees defined for prints, OLED/LCD moiré, and  |
|                    |         | diffusion face-swaps. Confirmed fail-safe routing to REVIEW when quality or PAD is ambiguous.            |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
| DOCS / AUDITOR     | PASSED  | README.md & LIMITATIONS.md marked NOT PRODUCTION-READY. Verified task DATA-CAMPAIGN-001 roadmap tracking. |
+--------------------+---------+----------------------------------------------------------------------------------------------------------+
```
