# Architecture Specification — FaceCheck Digital Account Opening (Photo-Only)

**System Classification:** Tier-1 Bank Onboarding Identity Verification System  
**Document Version:** 1.2.0  
**Status:** **PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY** (Pending consented real-attack golden set collection and accredited biometric calibration)  
**Author Role:** ARCHITECT (Reviewed with evidence by SECURITY REVIEWER, ML EVALUATOR, QA / RED-TEAM, DOCS/AUDITOR)

---

## 1. System Philosophy & Design Principles

FaceCheck is built for bank digital account opening with a photo-only input model (1 ID portrait photo + 1 selfie photo, without video or depth stream). The architecture is governed by five non-negotiable principles:

1. **Fail-Safe to Human Review:** When biometric evidence, image quality, or presentation attack risk is borderline or uncertain, the decision engine **never outputs `ACCEPT`**. It safely routes the transaction to `REVIEW`.
2. **Separation of Evaluation and Decision:** The machine learning models and heuristics produce quantitative scores and categorical detections. The decision engine evaluates these against bank risk appetite policies. The LLM reviewer **never decides**; it is solely an explainer that translates structured numerical outputs into human-readable rationale.
3. **Defense-in-Depth Pipeline:** Each stage operates as an independent filter. An applicant must pass upload validation, quality gates, presentation attack detection (PAD), and facial matching in sequence.
4. **Air-Gapped & Offline by Default:** All vision models (YuNet, SFace, SafeTensors PAD classifier) run strictly on local CPU/GPU runtimes with hardcoded SHA-256 integrity checks. Unsafe deserialization (`torch.load` without `weights_only=True` or arbitrary pickles) is strictly prohibited. External network calls and cloud LLMs are blocked by default.
5. **Strict Data Segregation & Cryptographic Non-Repudiation:** Production customer KYC data is separated by schema and credentials from ML training data. Every transaction and operational mutation is immutably appended to a cryptographic SHA-256 hash-chained audit ledger.

---

## 2. Hard Limitations of Photo-Only Biometrics & Standards Language

> [!WARNING]
> **Strict Operational Scope & Standards Disclaimer:**  
> - **Attack Scope:** Designed to address 2D print and screen-replay presentation attacks. 3D silicone/latex masks and injection attacks are **OUT OF SCOPE** for photo-only verification. No accredited laboratory evaluation has been performed.  
> - **NIST Alignment:** Partially addresses NIST SP 800-63A IAL2 biometric guidance; operational process and fraud controls required by 800-63A are **not covered by this software**.  
> - **Production Readiness:** System status is **NOT PRODUCTION-READY** pending completion of task `DATA-CAMPAIGN-001`.

---

## 3. Biometric Matching & Statistical Calibration Standards

### 3.1 Threshold Status: $\tau = 0.42$ [PROVISIONAL PLACEHOLDER]
- The operational threshold $\tau = 0.42$ is a provisional development seed. It **must not be used for production** until measured against the field dataset.
- **Statistical Validity & The Rule of 30:**
  - The classic Rule of 30 establishes that observing at least 30 false match events provides roughly $\pm 30\%$ relative error at **90% confidence** (not 95%).
  - For target $\text{FMR} \le 0.1\%$ ($1 \times 10^{-3}$):
    $$N_{\text{impostor pairs}} \ge \frac{30}{\text{FMR}} = \frac{30}{0.001} = 30,000 \text{ comparison pairs}$$
  - **Subject Independence & Cluster Bootstrap:** Comparison pairs formed from repeated captures of the same people are **not independent**. The evaluation protocol requires:
    - Minimum distinct subjects: $N_{\text{distinct subjects}} \ge 1,000$ unique individuals.
    - Confidence intervals must **NOT** use the naive Wald formula ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/N}$, which performs poorly near zero).
    - Confidence intervals must be computed using **subject-level cluster bootstrap** (resampling subjects with replacement) or the **true Wilson score interval**:
      $$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

### 3.2 Canonical Demographic Parity Gate & Subgroup Sizing
To prevent disparate impact across applicant demographics, model gating enforces one canonical rule for FMR and one canonical rule for FNMR:

1. **Canonical FMR Gate Rule:**
   $$\text{FMR}_{\text{group, upper 95\% CI}} \le 2.0 \times \text{FMR}_{\text{target}}$$
   For target $\text{FMR} = 0.10\%$, the upper bound of the 95% bootstrap confidence interval for any demographic group must satisfy:
   $$\text{FMR}_{\text{group, upper 95\% CI}} \le 0.20\%$$
2. **Canonical FNMR Gate Rule:**
   $$\text{FNMR}_{\text{group, upper 95\% CI}} \le 1.5 \times \text{FNMR}_{\text{population}}$$
3. **Minimum Subgroup Sample Sizing & Fail-Closed Invariant:**
   Each demographic subgroup (Gender, Age, Fitzpatrick skin tone) must contain at least:
   - $N_{\text{distinct subjects, group}} \ge 100$
   - $N_{\text{genuine pairs, group}} \ge 500$
   - $N_{\text{impostor pairs, group}} \ge 10,000$
   If any demographic subgroup fails to satisfy these sample minimums, it is marked **`INSUFFICIENT_DATA`**, and the gate **fails closed** (the candidate model is automatically rejected and cannot be promoted).

---

## 4. End-to-End System Architecture

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
            S3["Stage 3: Multi-Signal PAD Classifier\n(Genuine vs Print vs Replay + Risk 0-100)"]
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
