# Threat Model — FaceCheck Digital Account Opening (Photo-Only)

**System Classification:** Tier-1 Bank Onboarding Identity Verification  
**Operational Scope:** Photo-only Verification (1 ID Document Portrait Photo + 1 Selfie Photo; No Video Stream)  
**Document Version:** 1.1.0  
**Status:** **PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY** (Pending consented real-attack golden set collection and lab benchmarking)  
**Author Role:** ARCHITECT (Reviewed with evidence by SECURITY REVIEWER, QA / RED-TEAM, ML EVALUATOR)

---

## 1. Executive Summary & Mandatory Production Readiness Disclaimer

> [!CAUTION]
> **CRITICAL OPERATIONAL NOTICE — PROJECT IS NOT PRODUCTION-READY**  
> FaceCheck is currently a prototype/development system. **It is strictly NOT approved for live production banking account opening** until the mandatory field collection milestone is completed:
> - **Field Data Collection Campaign**: Must collect a consented real-attack golden test set comprising physical prints (inkjet, laser, matte, glossy), screen replays across prevalent local consumer smartphones (iPhone, Samsung, Xiaomi/Oppo), and simulated face-swaps under authentic branch/ambient lighting conditions with documented customer consent records (`consent_biometric_training = TRUE`).
> - **Threshold Calibration**: Operational matching threshold $\tau$ is currently a **PLACEHOLDER ($\tau = 0.42$)** and must be re-calibrated on verified validation data with rigorous confidence intervals.

The **FaceCheck** system evaluates applicant identity during bank digital account opening by comparing an applicant's government-issued identity document (ID photo) against a live-captured selfie photo.

Because the system operates in a **photo-only mode** (single static 2D selfie and single static 2D ID capture, without active video challenge-response or depth sensing), it faces fundamental physical constraints. This threat model identifies all threat actors, attack vectors, vulnerabilities, and residual risks under STRIDE, documents strict alignment boundaries with NIST and ISO standards, and establishes architectural controls ensuring the system **fails safely to human REVIEW** whenever evidence is inconclusive.

---

## 2. Hard Limitations of Photo-Only Biometrics

> [!WARNING]
> **Strict Operational Limitation: Photo-Only Biometrics Cannot Guarantee Vitality.**  
> A single 2D static photograph lacks temporal micro-motions, 3D volumetric depth, corneal dynamic reflection, and physiological vitality markers (e.g., photoplethysmography/blood pulse). Consequently, photo-only verification **cannot reliably stop** high-end hardware/OS camera injection attacks, hyper-realistic diffusion deepfakes, or custom 3D silicone/latex masks.

### 2.1 Standards Alignment & Gap Matrix

FaceCheck **does not claim compliance** with NIST SP 800-63A IAL3 / AAL3 or ISO/IEC 30107 Level 1/2/3.

```
+----------------------------------------------------------------------------------------------------+
| Standard           | Formal Status     | Technical Reality & Documented Gaps                       |
+--------------------+-------------------+-----------------------------------------------------------+
| NIST SP 800-63A    | Aligned with      | Remote document comparison permitted if evidence is       |
| IAL2               | Guidance          | validated. Remote 2D photo cannot verify physical security|
|                    |                   | features (holograms, microprint, tactile relief).         |
+--------------------+-------------------+-----------------------------------------------------------+
| NIST SP 800-63A    | NOT Compliant     | Requires in-person physical presence or supervised        |
| IAL3               |                   | remote attended video session with cryptographic tokens.  |
|                    |                   | Static photo submission is explicitly insufficient.       |
+--------------------+-------------------+-----------------------------------------------------------+
| ISO/IEC 30107      | Designed to       | Engineered to detect 2D printed paper/matte/glossy        |
| Presentation       | address specified | photos, curved paper masks, and LCD/OLED digital          |
| Attack Detection   | attack types;     | screen replays via Fourier moiré, texture analysis, and   |
| (PAD)              | NO accredited lab | deep neural classification. NO accredited third-party     |
|                    | evaluation        | lab evaluation (e.g., iBeta, FRSVT, BSI) has been         |
|                    | performed         | performed.                                                |
+--------------------+-------------------+-----------------------------------------------------------+
| ISO/IEC 30107-3    | NOT Compliant     | Level 3 covers sophisticated 3D silicone/latex masks and  |
| Level 3            |                   | sculpted physical facial prosthetics. Single 2D static    |
|                    |                   | photos cannot differentiate medical-grade silicone skin.  |
+--------------------+-------------------+-----------------------------------------------------------+
```

### 2.2 Mitigation Strategy for Photo-Only Deficits

1. **Multi-Signal Risk Fusion**: Rather than relying on a single binary score, the system extracts independent orthogonal signals:
   - Metadata anomalies (EXIF tampering, software editing tags, missing camera hardware markers).
   - Re-compression artifacts (double-JPEG quantization grid mismatches).
   - Screen-moiré and high-frequency Fourier grid harmonics.
   - Print artifacts (color gamut compression, halftone rasterization patterns).
   - Face-swap artifacts (boundary blending discontinuities, color-temperature disparity between face crop and background).
2. **Conservative Fail-Safe Policy**:
   - `ACCEPT` is emitted **ONLY** when: Quality gate passes 100%, PAD risk score is strictly low ($\le 30$), and 1:1 SFace cosine similarity exceeds the calibrated threshold.
   - `REJECT` is emitted **ONLY** for blatant non-matches or severe detected spoofs.
   - **Everything else defaults to `REVIEW`**. When in doubt, a human compliance officer makes the determination.

---

## 3. Threat Actor Profiles & Motivation

| Threat Actor | Motivation | Technical Sophistication | Resources | Attack Vectors |
| :--- | :--- | :--- | :--- | :--- |
| **Casual Impersonator** | Unauthorized account opening using friend/family ID | Low | Off-the-shelf smartphone, printed photo | Holds printed paper photo or iPad in front of camera |
| **Organized Fraud Syndicate (Mule Rings)** | High-volume synthetic identity fraud for money laundering | High | Botnets, generative AI toolkits (Midjourney, Flux, FaceSwap) | Automated batch attacks using deepfake selfies matched to stolen IDs |
| **Compromised Device / Malicious App** | Financial theft via malware on client smartphone | High | Virtual camera drivers, API reverse engineering | Bypasses camera sensor by injecting digital bitmap directly into API upload stream |
| **Disgruntled Insider** | Unauthorized loan approval or identity clearance | Medium | Internal bank network credentials | Tampering with audit logs, manipulating training data or decision thresholds |
| **Adversarial ML Researcher / Attacker** | Evasion of automated biometric classifiers | Very High | Gradient-based perturbation algorithms, surrogate models | Adds imperceptible noise ($\epsilon \le 8/255$) to selfie to force false match or spoof bypass |

---

## 4. Biometric Threshold Calibration & Statistical Requirements

### 4.1 SFace Matching Threshold ($\tau$ = 0.42 Placeholder)
- The baseline threshold $\tau = 0.42$ is currently a **PLACEHOLDER**.
- Under ISO/IEC 19795-1 (Biometric Performance Testing and Reporting), operational thresholds cannot be adopted from default vendor values or small internal tests.
- **Required Statistical Test Volume (Rule of 30 & Wilson Score Intervals):**
  - **Target False Match Rate (FMR):** $\le 0.1\%$ ($1 \times 10^{-3}$).
  - To achieve statistical significance at a 95% confidence interval with $\le 30\%$ relative error bound, the validation dataset must yield at least **30 false match events**, which requires:
    $$N_{\text{impostor pairs}} \ge \frac{30}{\text{FMR}} = \frac{30}{0.001} = 30,000 \text{ independent impostor comparison pairs}$$
  - **Target False Non-Match Rate (FNMR):** Validation requires at least:
    $$N_{\text{genuine pairs}} \ge 3,000 \text{ genuine comparison pairs}$$
    captured across varied lighting, capture devices, and poses.
  - Confidence intervals must be reported using **Clopper-Pearson exact** or **Wilson score intervals**:
    $$\text{CI}_{95\%} = \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$$

### 4.2 Demographic Parity Gate & Subgroup Sample Sizing
To prevent disparate impact across applicant demographics, model gating enforces strict subgroup sample minimums:
- **Demographic Slices:**
  - *Sex / Gender:* Male, Female.
  - *Age Brackets:* 18–25, 26–40, 41–60, 60+.
  - *Skin Tone / Complexion:* Fitzpatrick Scales I–II, III–IV, V–VI.
- **Minimum Subgroup Sample Sizing:** Each demographic group must have at least:
  - $N_{\text{genuine, group}} \ge 1,000$ verified comparison pairs.
  - $N_{\text{impostor, group}} \ge 10,000$ cross-subject comparison pairs.
- **Gating Disparity Criterion:**
  $$\Delta_{\text{FMR, group}} = \left| \text{FMR}_{\text{group, upper 95\% CI}} - \text{FMR}_{\text{population}} \right| \le 0.05\%$$
  $$\Delta_{\text{FNMR, group}} = \left| \text{FNMR}_{\text{group, upper 95\% CI}} - \text{FNMR}_{\text{population}} \right| \le 0.50\%$$
  Any candidate model exhibiting a statistically significant error rate regression in any subgroup beyond these tolerances is automatically discarded at Stage f.

---

## 5. STRIDE Threat Analysis & Specific Mitigations

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 STRIDE THREAT VECTORS                  │
                  └────────────────────────────────────────────────────────┘
                       │                  │                  │
        ┌──────────────┴───────┐   ┌──────┴──────────┐   ┌───┴────────────────┐
        │       Spoofing       │   │    Tampering    │   │  Repudiation / DoS │
        ├──────────────────────┤   ├─────────────────┤   ├────────────────────┤
        │ - Paper print attack │   │ - EXIF forgery  │   │ - Deny onboarding  │
        │ - Screen replay      │   │ - Polyglot file │   │ - Decompression    │
        │ - Deepfake/Face-swap │   │ - Weight poison │   │   bomb / OOM       │
        │ - Virtual camera     │   │ - Audit rewrite │   │ - Pipeline hang    │
        └──────────────────────┘   └─────────────────┘   └────────────────────┘
```

### 5.1 Spoofing (Presentation & Identity Attacks)
- **Threat S-1: Printed Paper Presentation Attack.**
  - *Scenario:* Attacker prints victim's photo on paper/cardstock and holds it before the camera.
  - *Mitigations:* Laplacian texture variance, 2D high-frequency spectral roll-off analysis, specular reflection inconsistency, YuNet multi-point landmark geometry. Borderline cases route to `REVIEW`.
- **Threat S-2: Screen Replay Attack (LCD / OLED Monitor / Smartphone).**
  - *Scenario:* Attacker replays victim's video/photo on an iPad or smartphone pointed at the camera.
  - *Mitigations:* Detect 2D periodic moiré grid patterns via Fourier Transform (FFT) peak detection; screen border/bezel detection; luminance chromaticity clipping.
- **Threat S-3: Deepfake & Face-Swap Synthesis.**
  - *Scenario:* Attacker generates an AI selfie using a diffusion model or swaps victim's face onto an accomplice's head.
  - *Mitigations:* PyTorch PAD classifier trained on synthetic generative artifacts; boundary blending gradient detection; hair-to-skin boundary discontinuity metrics.
- **Threat S-4: Virtual Camera Injection.**
  - *Scenario:* Attacker utilizes an Android/iOS root hook (e.g. Frida, VirtualCam) to bypass the physical camera and stream a pre-rendered digital file.
  - *Mitigations:* In photo-only mode, the backend cannot verify hardware sensor attestation directly. **Compensating control:** Backend inspects EXIF metadata for virtual camera signatures, verifies software quantization tables, and enforces stringent PAD risk thresholds. Any sign of re-compression or unnatural sensor noise defaults to `REVIEW`.

### 5.2 Tampering (Data, Model & Audit Manipulation)
- **Threat T-1: Decompression Bomb / Image Payload Exploits.**
  - *Scenario:* Attacker uploads an image with nested expansion headers unpacking to tens of gigabytes in memory (CWE-409).
  - *Mitigations:* Pillow `Image.MAX_IMAGE_PIXELS = 100_000_000`; maximum dimension check ($4096 \times 4096$); raw file size capped at 10 MB.
- **Threat T-2: Polyglot File Upload (JPEG + Script Payload).**
  - *Scenario:* Attacker embeds executable scripts inside JPEG comment fields or trailing bytes (CWE-434).
  - *Mitigations:* Decode and re-encode all images through Pillow/OpenCV into clean in-memory buffers; strip all EXIF and metadata; reject files where magic bytes disagree with MIME or contain trailing data past end-of-image markers.
- **Threat T-3: Model Weight Tampering in Storage.**
  - *Scenario:* Malicious actor modifies `sface.onnx` or `pad_classifier.pt` to force fraudulent approvals.
  - *Mitigations:* Hardcoded SHA-256 checksums verified at startup; read-only container file system; application fails to boot if any weight hash deviates.
- **Threat T-4: Audit Record Tampering.**
  - *Scenario:* Rogue employee deletes or modifies an audit row to conceal fraudulent account approval.
  - *Mitigations:* Tamper-evident cryptographic SHA-256 hash chaining: every audit entry contains `prev_hash` binding the previous row, current row timestamp, decision, scores, and UUID. Verified continuously by automated audit worker.

### 5.3 Repudiation
- **Threat R-1: Applicant Denies Account Opening or Inconclusive Audit Trail.**
  - *Mitigations:* Immutable audit record stores SHA-256 hashes of input images, exact model versions, raw decision metrics, timestamp, client IP, and authorization principal. Data is written to `kyc_prod.audit_log` before returning the HTTP response.

### 5.4 Information Disclosure
- **Threat I-1: Biometric Data Leakage in Logs or Unencrypted Disk.**
  - *Scenario:* Facial photos or high-dimensional embeddings leak into debug log files or persistent temporary directories (CWE-532).
  - *Mitigations:* No image paths or biometric raw data are ever logged. Images are processed in volatile memory and encrypted at rest using AES-256-GCM envelope encryption with KMS-managed keys.
- **Threat I-2: PII Leakage via LLM Reviewer.**
  - *Scenario:* Customer name, national ID number, or raw facial photos are sent to a cloud or local LLM.
  - *Mitigations:* **Architectural Firewall:** The LLM receives **ONLY** structured telemetry (scores, quality metrics, reason codes). It never sees raw pixels, names, or national IDs. Cloud LLM providers are disabled by default. Local offline LLM (Ollama) is used exclusively unless explicit data residency approval is documented.

### 5.5 Denial of Service (DoS)
- **Threat D-1: CPU/GPU Exhaustion via Flooding.**
  - *Scenario:* Attacker floods `/check` or `/verify` with high-resolution images to exhaust inference threads.
  - *Mitigations:* Ingestion rate limiting (sliding window per API key and client IP); worker process timeouts (5.0s hard cutoff); maximum queue depth.

### 5.6 Elevation of Privilege
- **Threat E-1: Training Worker Accesses Unauthorized Production Customer Data.**
  - *Scenario:* Automated training worker inspects raw KYC databases without customer consent.
  - *Mitigations:* Separate PostgreSQL schemas (`kyc_prod` vs `ml_training`) with distinct database user credentials. Training worker has `SELECT` access only on `ml_training.training_samples` where `consent_verified = TRUE` and `license_type` is valid.

---

## 6. Mandatory Roadmap Task: Consented Real-Attack Golden Set Collection

Before FaceCheck can transition from **PROTOTYPE** to **PRODUCTION-READY**, the following data collection task must be executed:
- **Task ID:** `DATA-CAMPAIGN-001`
- **Objective:** Collect $N \ge 1,000$ genuine subjects and $N \ge 500$ presentation attack instruments across local target demographics.
- **Attack Variations:**
  1. *Print Attacks:* High-resolution inkjet and laser prints on matte and glossy photo stock, curved face wraps, cut-out eye holes.
  2. *Replay Attacks:* High-definition displays (OLED, IPS LCD) across prevalent local devices (Apple iPhone 13-16, Samsung Galaxy S/A series, Xiaomi/Redmi, Oppo/Vivo).
  3. *Synthetic / Face Swaps:* Deepfake face-swaps generated with contemporary diffusion and autoencoder models matched to target ID portraits.
- **Consent Governance:** Every participant must sign an explicit biometric processing and storage consent form with recorded consent UUID, retention expiry, and right to erasure.

---

## 7. Approval & Review Sign-Off (Evidence-Citing)

- **ARCHITECT:** Updated threat model to document provisional $\tau=0.42$ placeholder status, statistical validation pair requirements ($N \ge 30,000$), strict ISO 30107 language, demographic CI formulas, and field collection prerequisite.
- **SECURITY REVIEWER:** Verified STRIDE controls, CWE-409/CWE-434 mitigations, and dual-schema DB boundary.
- **ML EVALUATOR:** Confirmed ISO/IEC 19795-1 test volume specifications and Clopper-Pearson / Wilson score CI criteria.
- **DOCS / AUDITOR:** Verified updated disclaimer and marked system as NOT production-ready pending `DATA-CAMPAIGN-001`.
