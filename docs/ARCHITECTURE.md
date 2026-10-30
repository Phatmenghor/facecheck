# Architecture Specification — FaceCheck Digital Account Opening (Photo-Only)

**System Classification:** Tier-1 Bank Onboarding Identity Verification System  
**Document Version:** 1.1.0  
**Status:** **PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY** (Pending consented real-attack golden set collection and accredited biometric calibration)  
**Author Role:** ARCHITECT (Reviewed with evidence by SECURITY REVIEWER, ML EVALUATOR, QA / RED-TEAM, DOCS/AUDITOR)

---

## 1. System Philosophy & Design Principles

FaceCheck is built for bank digital account opening with a photo-only input model (1 ID portrait photo + 1 selfie photo, without video or depth stream). The architecture is governed by five non-negotiable principles:

1. **Fail-Safe to Human Review:** When biometric evidence, image quality, or presentation attack risk is borderline or uncertain, the decision engine **never outputs `ACCEPT`**. It safely routes the transaction to `REVIEW`.
2. **Separation of Evaluation and Decision:** The machine learning models and heuristics produce quantitative scores and categorical detections. The decision engine evaluates these against bank risk appetite policies. The LLM reviewer **never decides**; it is solely an explainer that translates structured numerical outputs into human-readable rationale.
3. **Defense-in-Depth Pipeline:** Each stage operates as an independent filter. An applicant must pass upload validation, quality gates, presentation attack detection (PAD), and facial matching in sequence.
4. **Air-Gapped & Offline by Default:** All vision models (YuNet, SFace, PyTorch PAD classifier) run strictly on local CPU/GPU runtimes with hardcoded SHA-256 integrity checks. External network calls and cloud LLMs are blocked by default.
5. **Strict Data Segregation & Cryptographic Non-Repudiation:** Production customer KYC data is separated by schema and credentials from ML training data. Every transaction and operational mutation is immutably appended to a cryptographic SHA-256 hash-chained audit ledger.

---

## 2. Production Readiness Gate: Field Data Collection Campaign

> [!CAUTION]
> **MANDATORY PREREQUISITE: PROJECT IS NOT PRODUCTION-READY**  
> FaceCheck is strictly prohibited from being deployed into live customer account opening until **Task DATA-CAMPAIGN-001** is completed and approved:
> - **Field Collection Campaign (`DATA-CAMPAIGN-001`):** Must capture a field-validated golden benchmark dataset of physical presentation attacks across prevalent local consumer devices (e.g. iPhone 13-16, Samsung Galaxy S/A series, Xiaomi, Oppo displays) and physical paper prints (inkjet, color laser, matte, glossy cardstock), with documented customer consent records (`consent_biometric_training = TRUE`).
> - **Biometric Calibration Requirement:** Operating threshold $\tau = 0.42$ is currently an uncalibrated **PLACEHOLDER**. True operational thresholds must be derived from measured ROC/DET curves on $\ge 30,000$ impostor pairs with 95% confidence intervals.

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Integration Layer"]
        App["Mobile / Web Banking Client"]
    end

    subgraph SecurityBorder["Security Border & Ingestion Gateway"]
        AuthFilter["Auth & Rate Limiting\n(Per-user roles: caller, auditor, admin)"]
        UploadVal["Upload Hardening & Decompression Defense\n(MIME check, max 10MB, polyglot reject, EXIF strip)"]
    end

    subgraph InferenceEngine["Biometric Inference Engine (In-Memory Processing)"]
        direction TB
        subgraph Pipeline["6-Stage Processing Pipeline"]
            S1["Stage 1: Ingestion & Volatile Buffer Allocation"]
            S2["Stage 2: Pre-Flight Quality Gate\n(YuNet Face Count=1, Blur, Pose, Brightness, IED >= 60px)"]
            S3["Stage 3: Multi-Signal PAD Classifier\n(Genuine vs Print vs Replay vs Synthetic + Risk 0-100)"]
            S4["Stage 4: 1:1 SFace Biometric Matching\n(128-d L2 Cosine Sim; Threshold tau=0.42 PLACEHOLDER)"]
            S5["Stage 5: Configurable Decision Policy\n(ACCEPT / REVIEW / REJECT)"]
            S6["Stage 6: Structured LLM Reviewer & Explainer\n(Local Ollama/Hermes, Schema-Validated JSON)"]
            
            S1 --> S2 --> S3 --> S4 --> S5 --> S6
        end
    end

    subgraph StorageLayer["Data & Audit Storage (PostgreSQL)"]
        ProdSchema[("Schema: kyc_prod\n- decisions\n- encrypted_images (KMS AES-256)\n- audit_chain (SHA-256 Chained)")]
        TrainSchema[("Schema: ml_training\n- consented_samples\n- model_registry\n- golden_test_set")]
    end

    subgraph MLOpsWorker["Automated Training Worker (Separate Process)"]
        WorkerSched["Scheduled MLOps Worker\n(Ingest -> Auto-Label -> Dedupe -> Train -> Eval -> Gate -> Register -> Shadow)"]
        KillSwitch{"Global Kill Switch\n(CONFIG.TRAINING_ENABLED)"}
    end

    subgraph MonitoringUI["Read-Only Monitoring Web"]
        WebDashboard["Read-Only Web Dashboard\n(Strictly GET Endpoints, SSE 5s refresh, DB Role: facecheck_readonly)"]
    end

    App -->|POST /verify| AuthFilter
    AuthFilter --> UploadVal
    UploadVal --> S1
    S5 -->|Write Decision & Hash Record| ProdSchema
    ProdSchema -.->|Read Consented Data Only| WorkerSched
    WorkerSched --> KillSwitch
    WorkerSched --> TrainSchema
    TrainSchema -.->|SELECT Views Only| WebDashboard
    ProdSchema -.->|SELECT Views Only (Masked PII)| WebDashboard
```

---

## 4. Biometric Inference Pipeline (Detailed Stages)

### 4.1 Stage 1: Upload Validation & Input Hardening
- **Allowed MIME Types:** `image/jpeg`, `image/png`, `image/webp`. Rejects animated formats (`image/gif`, animated WebP, APNG) and multi-frame TIFFs.
- **Size Limits:** Strict maximum file size of 10 MB per image. Pre-allocated memory caps.
- **Decompression Bomb Protection:** Pillow `Image.MAX_IMAGE_PIXELS = 100_000_000`. Maximum spatial dimensions restricted to $4096 \times 4096$ pixels (CWE-409).
- **Polyglot & Trailing Byte Rejection:** File magic bytes must match HTTP Content-Type. The binary stream is inspected to ensure no executable headers (PHP, ELF, PE) or non-image payloads are concatenated past standard EOF markers (`FF D9` for JPEG, `IEND` for PNG) (CWE-434).
- **EXIF Stripping:** All EXIF metadata, GPS coordinates, and camera device serials are parsed for forensic risk signals and immediately stripped before raw image arrays enter memory.

### 4.2 Stage 2: Pre-Flight Facial Quality Gate
Evaluates both the ID crop and selfie to ensure high-fidelity biometrics before invoking heavy neural models:

| Quality Dimension | Metric / Algorithm | Minimum Acceptance Threshold | Failure Reason Code |
| :--- | :--- | :--- | :--- |
| **Face Count** | YuNet multi-scale face detector | Exactly $1$ primary face | `FACE_COUNT_ZERO` or `FACE_COUNT_MULTIPLE` |
| **Inter-Eye Distance (IED)** | Euclidean distance between eye landmarks | $\ge 60$ pixels | `FACE_IED_TOO_SMALL` |
| **Face Crop Resolution** | Spatial bounding box dimensions | $\ge 120 \times 120$ pixels | `FACE_RESOLUTION_TOO_LOW` |
| **Blur / Sharpness** | Modified 2D Laplacian variance | Variance $\ge 60.0$ | `IMAGE_BLURRY` |
| **Exposure / Brightness** | Mean luminance of face ROI ($Y$ channel in YCrCb) | $40 \le \bar{Y} \le 220$ | `FACE_UNDEREXPOSED` or `FACE_OVEREXPOSED` |
| **Head Pose (Selfie)** | YuNet 5-point landmark homography solve | Yaw $\in [-25^\circ, +25^\circ]$, Pitch $\in [-20^\circ, +20^\circ]$, Roll $\in [-20^\circ, +20^\circ]$ | `HEAD_POSE_EXTREME` |
| **Occlusion** | Landmark confidence & edge uniformity across nose/mouth | Occlusion score $< 0.35$ | `FACE_OCCLUSION_DETECTED` |

### 4.3 Stage 3: Presentation Attack Detection (PAD)
- **Attack Types Addressed:** Engineered to detect 2D printed paper/matte/glossy photos, curved paper masks, and LCD/OLED digital screen replays. **No accredited third-party lab evaluation (e.g. iBeta, BSI) has been performed.**
- **Known Inherent Gaps:** Single static 2D photos cannot detect sophisticated 3D silicone/latex masks, sculpted prosthetics, or virtual camera OS driver injection.
- **Signals:** PyTorch/ONNX deep classifier + 2D FFT moiré peak detection + GLCM/LBP texture analysis + boundary gradient analysis.
- **Calibrated Risk Score ($0 - 100$):** Platt-calibrated where $\le 30$ indicates genuine bona fide capture, $31 - 70$ indicates uncertain/borderline, and $\ge 71$ indicates suspected attack.

### 4.4 Stage 4: Biometric 1:1 Facial Matching & Threshold Specification
- **Feature Extractor:** OpenCV SFace (Deep Facial Recognition with SphereFace architecture).
- **Embedding:** 128-dimensional L2-normalized unit vector ($||\mathbf{e}||_2 = 1.0$).
- **Metric:** Cosine similarity:
  $$\text{Similarity}(\mathbf{e}_{\text{ID}}, \mathbf{e}_{\text{Selfie}}) = \mathbf{e}_{\text{ID}} \cdot \mathbf{e}_{\text{Selfie}}$$
- **Threshold Status: $\tau = 0.42$ [PROVISIONAL PLACEHOLDER]**
  - The threshold $\tau = 0.42$ is a provisional development seed. It **must not be used for production** until measured against the field dataset.
  - **Required Empirical Sizing for Target $\text{FMR} \le 0.1\%$ ($1 \times 10^{-3}$):**
    - Under ISO/IEC 19795-1 (Rule of 30), establishing statistical validity at 95% confidence with $\le 30\%$ relative error requires observing at least 30 false match errors:
      $$N_{\text{impostor pairs}} \ge \frac{30}{\text{FMR}} = \frac{30}{0.001} = 30,000 \text{ cross-subject impostor pairs}$$
    - Genuine pair volume must satisfy:
      $$N_{\text{genuine pairs}} \ge 3,000 \text{ genuine comparison pairs}$$
    - Wilson score confidence intervals must be computed and reported:
      $$\text{CI}_{95\%} = \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$$

### 4.5 Stage 5: Configurable Decision Policy Engine
1. **`ACCEPT` (Strict All-Pass):**
   - Quality Gate: Passed ($100\%$).
   - PAD Risk Score: $\le 30$ (Low risk / Genuine).
   - Match Similarity: $\ge \tau_{\text{calibrated}}$ (High biometric confidence).
2. **`REJECT` (Clear Blatant Failure):**
   - PAD Risk Score $\ge 85$ (Definitive spoof detected), OR
   - Match Similarity $< 0.20$ (Complete facial mismatch), OR
   - Severe non-recoverable tamper attempt (e.g. decompression bomb, corrupt payload).
3. **`REVIEW` (Default Safe Harbor for Borderline / Uncertain):**
   - All transactions that are neither strict `ACCEPT` nor blatant `REJECT`.
   - **Guaranteed Invariant:** The system *never* outputs `ACCEPT` when evidence is ambiguous.

### 4.6 Stage 6: Explainability Reviewer (LLM-Assisted)
- **Architectural Boundary:** The LLM is an **advisory explainer**, not an arbiter.
- **Input Isolation:** Receives **only** structured JSON telemetry (scores, quality metrics, reason codes). No raw pixels, no embeddings, zero customer PII.
- **Execution Runtime:** Local LLM (Ollama) by default. Cloud LLMs are hard-disabled in configuration.
- **Output Schema Validation:** Pydantic validated. Treated as untrusted text. Falls back to deterministic rule template on schema deviation or timeout ($> 2.5\text{s}$).

---

## 5. Automated Training & MLOps Pipeline (Stages a–h)

```
[a. INGEST] ───► [b. AUTO-LABEL] ───► [c. DEDUPE & SPLIT] ───► [d. TRAIN]
                                                                  │
                                                                  ▼
[h. PROMOTE/SHADOW] ◄── [g. REGISTER] ◄── [f. GATE] ◄─── [e. EVALUATE]
```

### 5.1 Stages a–e: Ingest, Label, Dedupe, Train, Evaluate
- **a. INGEST:** Only synthetic in-house, licensed public benchmarks, or consented bank data (`consent_biometric_training = TRUE`). Tracks source, license, and consent UUID per sample.
- **b. AUTO-LABEL:** Tags `HUMAN`, `AUTO_HIGH_CONF`, `AUTO_LOW_CONF`. Samples tagged `AUTO_LOW_CONF` are never used for training. 5–10% of `AUTO_HIGH_CONF` data is audited by human reviewers; if disagreement exceeds **3.0%**, training automatically pauses.
- **c. DEDUPE & SPLIT:** pHash/dHash deduplication (Hamming $\le 4$). Identity-disjoint splits. Immutable, versioned golden test set.
- **d. TRAIN:** Trains candidate PAD classifier (PyTorch) or reviewer LoRA.
- **e. EVALUATE:** Computes APCER, BPCER, FMR, FNMR, per-group breakdown, and ECE calibration against the golden set.

### 5.2 Stage f: Demographic Parity Gate & Minimum Sample Sizes
A candidate model is automatically discarded unless it passes all hard gates:
1. **Metric Improvement:** Candidate must achieve equal or lower APCER/BPCER compared to production.
2. **Demographic Group Sample Size Requirements:**
   - Evaluated across subgroups: Gender (Male, Female), Age (18–25, 26–40, 41–60, 60+), Skin Tone (Fitzpatrick I–II, III–IV, V–VI).
   - Each subgroup requires at least **$N \ge 1,000$ genuine pairs** and **$N \ge 10,000$ impostor pairs**.
   - Evaluated using 95% Wilson score confidence intervals.
3. **Zero Demographic Regression:**
   $$\Delta_{\text{FMR, group}} = \left| \text{FMR}_{\text{group, upper 95\% CI}} - \text{FMR}_{\text{population}} \right| \le 0.05\%$$
   $$\Delta_{\text{FNMR, group}} = \left| \text{FNMR}_{\text{group, upper 95\% CI}} - \text{FNMR}_{\text{population}} \right| \le 0.50\%$$
4. **Adversarial Test Pack Pass:** $\ge 98\%$ detection on red-team test samples.

### 5.3 Stages g–h: Registration, Shadow Mode, Promotion & Kill Switch
- **g. REGISTER:** Signs model weights, logs SHA-256 hash, dataset manifest, and git commit in `ml_training.model_registry`. Never overwrites production weights.
- **h. PROMOTE:** Shadow mode execution beside production. Two-person rule CLI authorization. 5% canary rollout with auto-rollback on drift.
- **Global Kill Switch:** `CONFIG.TRAINING_ENABLED = false` immediately halts all training, evaluation, and canary deployments.

---

## 6. Read-Only Monitoring Web Architecture

- **Zero Write Endpoints:** HTTP GET endpoints only. No mutation endpoints exist.
- **Database Role Isolation:** Connects with PostgreSQL role `facecheck_readonly` with `SELECT` permissions strictly limited to sanitized views (`v_monitoring_pipeline_status`, `v_monitoring_model_metrics`, `v_monitoring_training_runs`, `v_monitoring_security_alerts`).
- **Zero Access to Raw KYC Images & PII:** Masked IDs (`usr_***4a8f`), aggregated scores only.
- **Real-Time Push (SSE):** Server-Sent Events stream updates every 5 seconds.
- **Authentication:** Per-user JWT with roles `auditor` or `admin`, TLS 1.3, strict Content Security Policy.

---

## 7. Cryptographic Design & Storage

### 7.1 Application-Level Image Encryption (AES-256-GCM)
- Images retained for regulatory compliance are encrypted at the application tier before database persistence:
  - Cipher: AES-256-GCM authenticated encryption.
  - KMS-managed Key Encryption Keys (KEK) wrapping local Data Encryption Keys (DEK).
  - Unique 96-bit IV nonce per image.
  - Automated cryptographic erasure on `purge_after` timestamp expiry.

### 7.2 Tamper-Evident Hash Chain Ledger (`audit_chain`)
Every decision and operational event generates a SHA-256 hash linking to the preceding entry:
$$\text{Hash}_n = \text{SHA-256}\left(\text{Hash}_{n-1} \,\|\, \text{CheckID} \,\|\, \text{Timestamp} \,\|\, \text{Decision} \,\|\, \text{MatchScore} \,\|\, \text{PADScore} \,\|\, \text{ReasonCodes}\right)$$
Continuous background validation reports status `OK` or `BROKEN`.

---

## 8. Role Sign-Off & Review Matrix (Evidence-Citing)

| Agent Role | Review Verdict | Concrete Evidence Cited |
| :--- | :--- | :--- |
| **ARCHITECT** | **APPROVED** | Updated architecture to mark $\tau=0.42$ as PLACEHOLDER, specified ISO/IEC 19795-1 sample size formulas ($N \ge 30,000$), revised ISO 30107 wording, added demographic CIs, and added task `DATA-CAMPAIGN-001`. |
| **SECURITY REVIEWER** | **APPROVED** | Verified CWE-409/CWE-434 mitigations, dual-schema database isolation, AES-256-GCM image encryption, and tamper-evident SHA-256 hash chaining. |
| **ML EVALUATOR** | **APPROVED** | Verified Rule of 30 sample size derivation ($N=30,000$ for $FMR \le 0.1\%$), Wilson score 95% confidence intervals, and demographic subgroup minimums ($N \ge 1,000$ genuine / $N \ge 10,000$ impostor). |
| **QA / RED-TEAM** | **APPROVED** | Verified attack coverage (prints, OLED/LCD replays, face swaps), confirmed fail-safe routing to `REVIEW`, and validated task `DATA-CAMPAIGN-001`. |
| **DOCS / AUDITOR** | **APPROVED** | Verified all documentation links, marked project as NOT production-ready, and recorded evidence in `CHANGELOG.md`. |
