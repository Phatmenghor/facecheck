# FaceCheck: Bank Digital Account Opening Biometric Verification

**FaceCheck** is a hardened, offline biometric verification system designed specifically for **bank digital account opening** under a photo-only operational model (comparing one applicant selfie photo against one government-issued ID portrait photo, without video or depth stream).

**SYSTEM STATUS:** `PROTOTYPE / PRE-PRODUCTION` — **STRICTLY NOT PRODUCTION-READY** until the field data collection campaign (`DATA-CAMPAIGN-001`) is completed and biometric thresholds are empirically calibrated.

The system outputs a deterministic compliance decision in `{ACCEPT, REVIEW, REJECT}`, fine-grained risk and similarity scores, detailed reason codes, and an immutable cryptographic audit record. **The system never outputs `ACCEPT` when evidence is uncertain; it safely routes all borderline cases to human `REVIEW`.**

---

## Hard Limitations of Photo-Only Biometrics

> [!CAUTION]
> **CRITICAL OPERATIONAL LIMITATION: Photo-Only Cannot Reliably Stop Advanced Spoofs.**  
> A photo-only verification pipeline (evaluating a single static 2D selfie against an ID photo without video or depth stream) **CANNOT reliably detect hardware/OS camera injection attacks, high-quality generative deepfakes, or custom 3D silicone/latex masks**.
>
> To mitigate these inherent physical vulnerabilities, FaceCheck evaluates orthogonal risk signals:
> - **Image metadata anomalies & software editing signatures**
> - **Re-compression and double-JPEG quantization grid mismatches**
> - **Screen-moiré and high-frequency Fourier grid harmonics**
> - **Print texture rasterization and halftone pattern degradation**
> - **Face-swap blending discontinuities and skin boundary artifacts**
>
> **Operational Rule:** Any borderline, ambiguous, or suspicious capture is routed to **human `REVIEW`**. The system **never outputs `ACCEPT` on weak or conflicting evidence**.

### Standards Alignment & Compliance Gaps

FaceCheck **does not claim compliance** with NIST SP 800-63A IAL3 / AAL3 or ISO/IEC 30107 Level 1/2/3.

| Standard & Target Level | System Status | Documented Gaps & Compensating Controls |
| :--- | :--- | :--- |
| **NIST SP 800-63A IAL2** (Identity Assurance Level 2) | **Aligned with guidance** | Standard permits remote biometric comparison if paired with evidence validation. **Gap:** Remote physical document security features (holograms, tactile relief) cannot be verified from a 2D photo alone. **Compensating control:** Quality gate verifies resolution and edge clarity, routing suspect documents to branch or video KYC. |
| **NIST SP 800-63A IAL3** (Identity Assurance Level 3) | **NOT Compliant** | **Gap:** Requires in-person physical presence or supervised remote attended video session with cryptographic hardware tokens. Static photo submission is explicitly insufficient for IAL3. |
| **ISO/IEC 30107 Presentation Attack Detection (PAD)** | **Designed to address specified attack types; NO accredited lab evaluation performed** | System is engineered to address specified presentation attack instruments (2D printed paper/matte/glossy photos, curved paper masks, and LCD/OLED digital screen replays). **No accredited third-party laboratory evaluation (e.g., iBeta, BSI) has been performed.** |
| **ISO/IEC 30107-3 Level 3** (Advanced Presentation Attacks) | **NOT Compliant** | **Gap:** Level 3 encompasses sophisticated 3D silicone/latex masks and sculpted physical facial prosthetics. Single 2D static photos cannot differentiate medical-grade silicone skin from human tissue. |

*For complete threat modeling and mitigation proofs, see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) and [LIMITATIONS.md](LIMITATIONS.md).*

---

## System Architecture

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

*For complete technical design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 6-Stage Biometric Inference Pipeline

1. **Upload Validation & Hardening:** Enforces 10 MB file cap, allowed MIME types (`image/jpeg`, `image/png`, `image/webp`), decompression bomb protection (`MAX_IMAGE_PIXELS = 100_000_000`, max dimensions $4096 \times 4096$), polyglot and trailing payload rejection, and automatic EXIF stripping.
2. **Quality Gate:** Requires exactly 1 primary face detected via OpenCV YuNet ONNX, minimum inter-eye distance (IED $\ge 60\text{px}$), bounding box resolution $\ge 120 \times 120\text{px}$, Laplacian blur variance $\ge 60.0$, balanced luminance ($40 \le \bar{Y} \le 220$), head pose within $[-25^\circ, +25^\circ]$ yaw, and occlusion check. Returns specific machine-readable reason codes.
3. **Presentation Attack Detection (PAD):** PyTorch/ONNX deep classifier categorizing genuine vs print vs screen-replay vs synthetic faces, combined with FFT moiré peak detection, GLCM texture metrics, and color-gradient boundary heuristics to yield a calibrated risk score ($0 - 100$).
4. **1:1 Facial Matching (SFace):** Generates 128-dimensional L2-normalized facial embeddings using OpenCV SFace. The operating threshold $\tau = 0.42$ is a **PLACEHOLDER** pending empirical validation on $\ge 30,000$ cross-subject impostor pairs targeting $FMR \le 0.1\%$ with 95% Wilson score confidence intervals.
5. **Configurable Decision Policy:**
   - **`ACCEPT`**: Quality Gate $100\%$ pass, PAD Risk $\le 30$, SFace Cosine Sim $\ge \tau_{\text{calibrated}}$.
   - **`REJECT`**: Clear failure (PAD Risk $\ge 85$, Cosine Sim $< 0.20$, or severe malformed payload).
   - **`REVIEW`**: All uncertain, borderline, or conflicting transactions.
6. **LLM Reviewer & Explainer:** Local offline model (Ollama) receiving **only** structured scores and reason codes (zero raw images, zero PII). Generates human-readable compliance summaries strictly validated against a Pydantic JSON schema. The LLM is an advisor; it **never makes the decision**.

---

## Production Prerequisites & Roadmap

| Task ID | Milestone Description | Prerequisite For | Status |
| :--- | :--- | :--- | :--- |
| **`DATA-CAMPAIGN-001`** | Collect consented real-attack golden benchmark (paper prints, smartphone replays, face swaps across local mobile devices) | Production Deployment | **PENDING** |
| **`CALIBRATE-TAU-001`** | Measure empirical ROC/DET curves on $\ge 30,000$ impostor pairs to establish calibrated $\tau$ with 95% Wilson CIs | Production Deployment | **PENDING** |
| **`GATING-DEMO-001`** | Benchmark demographic subgroup sample sizes ($N \ge 1,000$ genuine, $N \ge 10,000$ impostor) with parity tolerance $\Delta \le 0.5\%$ | Production Deployment | **PENDING** |

---

## Read-Only Monitoring Web Dashboard

The operational interface is strictly read-only:
- **GET Endpoints Only:** Zero mutation/write endpoints exist.
- **Database Role Isolation:** Connects with PostgreSQL role `facecheck_readonly` with `SELECT` access strictly on sanitized views.
- **Privacy Firewall:** No access to raw customer images or PII (masked IDs, aggregated scores only).
- **Live Push (SSE):** Server-Sent Events stream updates every 5 seconds covering pipeline status, agent activity, training metrics, model metrics (APCER, BPCER, FMR, FNMR, per-group), decisions, and security alerts.

---

## Key Deliverables & Documentation Index

1. [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md): Comprehensive STRIDE analysis, attack trees, photo-only limitations, and NIST/ISO gap matrix.
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): End-to-end component design, pipeline stages, MLOps lifecycle, and database schema isolation.
3. [LIMITATIONS.md](LIMITATIONS.md): Operational disclaimers and guidance for bank model risk officers.
4. [CHANGELOG.md](CHANGELOG.md): Multi-agent audit trail and evidence-citing review sign-offs.
5. [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): Open-source license inventory.
