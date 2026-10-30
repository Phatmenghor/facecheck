# FaceCheck API — Model Limitations & Operational Disclaimers

**Document Classification:** Internal Banking Risk & Compliance  
**Audience:** Model Risk Officers, Biometric Governance, Compliance & Audit  
**System Scope:** Bank Digital Account Opening (Photo-Only: 1 ID Photo + 1 Selfie Photo; No Video Stream)  
**System Status:** **PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY**

---

## 1. Executive Summary & Mandatory Production Readiness Notice

> [!CAUTION]
> **CRITICAL OPERATIONAL NOTICE: SYSTEM IS NOT PRODUCTION-READY.**  
> FaceCheck is an engineering prototype. **It is NOT certified for live customer account opening.** Live deployment is gated upon the successful completion of:
> 1. **Field Data Collection (`DATA-CAMPAIGN-001`):** Collection of a consented real-attack golden test set including printed paper photos (matte, glossy, laser, inkjet), screen replays across common local smartphones (iPhone, Samsung Galaxy, Xiaomi/Oppo displays), and simulated face swaps under local lighting conditions with recorded explicit GDPR/banking consent.
> 2. **Empirical Threshold Calibration:** Matching threshold $\tau = 0.42$ is currently a **PLACEHOLDER**. A valid operational threshold must be measured from a validation set of $\ge 30,000$ cross-subject impostor pairs and $\ge 3,000$ genuine pairs to guarantee $FMR \le 0.1\%$ at 95% confidence intervals.

---

## 2. Hard Physical Limitations of Photo-Only Biometrics

> [!WARNING]
> **HARD LIMITATION: Photo-Only Biometrics Cannot Reliably Detect Advanced Spoofs.**  
> A photo-only verification pipeline (evaluating a single static 2D selfie against an ID photo without video or depth stream) **CANNOT reliably detect hardware/OS camera injection attacks, high-quality generative deepfakes, or custom 3D silicone/latex masks**.
>
> To mitigate these inherent physical vulnerabilities, FaceCheck deploys orthogonal risk signals:
> - Image metadata anomalies & software editing signatures
> - Re-compression and double-JPEG quantization artifacts
> - Screen-moiré and high-frequency Fourier grid harmonics
> - Print texture rasterization and halftone pattern degradation
> - Face-swap blending discontinuities and skin boundary artifacts
>
> **Operational Rule:** Any borderline, ambiguous, or suspicious capture is routed to **human `REVIEW`**. The system **never outputs `ACCEPT` when uncertain**.

---

## 3. Standards Alignment & Compliance Gaps

FaceCheck **does not claim compliance** with NIST SP 800-63A IAL3 / AAL3 or ISO/IEC 30107 Level 1/2/3.

### Gap Analysis Matrix

| Standard & Target Level | System Status | Documented Gaps & Compensating Controls |
| :--- | :--- | :--- |
| **NIST SP 800-63A IAL2** (Identity Assurance Level 2) | **Aligned with guidance** | Standard permits remote biometric comparison if paired with evidence validation. **Gap:** Remote physical document security features (holograms, tactile relief) cannot be verified from a 2D photo alone. **Compensating control:** Quality gate verifies resolution and edge clarity, routing suspect documents to branch or video KYC. |
| **NIST SP 800-63A IAL3** (Identity Assurance Level 3) | **NOT Compliant** | **Gap:** Requires in-person physical presence or supervised remote attended video session with cryptographic hardware tokens. Static photo submission is explicitly insufficient for IAL3. |
| **ISO/IEC 30107 Presentation Attack Detection (PAD)** | **Designed to address specified attack types; NO accredited lab evaluation performed** | System is engineered to address specified presentation attack instruments (2D printed paper/matte/glossy photos, curved paper masks, and LCD/OLED digital screen replays). **No accredited third-party laboratory evaluation (e.g., iBeta, BSI) has been performed.** |
| **ISO/IEC 30107-3 Level 3** (Advanced Presentation Attacks) | **NOT Compliant** | **Gap:** Level 3 encompasses sophisticated 3D silicone/latex masks and sculpted physical facial prosthetics. Single 2D static photos cannot differentiate medical-grade silicone skin from human tissue. |

---

## 4. Technical Vulnerabilities of 2D Static Images

### 4.1 Absence of Physiological Vitality & Motion
A single 2D static photograph lacks:
- Temporal micro-motion (eyeblink, micro-saccades, pupillary light reflex).
- Volumetric depth curvature (3D structure of the nasal bridge and facial contours).
- Blood flow vitality (remote photoplethysmography / rPPG pulse signal).
- Corneal specular light reflections corresponding to dynamic environmental illumination.

### 4.2 Camera Sensor & Injection Bypass
- A malicious applicant utilizing a rooted Android/iOS device or virtual camera driver can stream a pristine pre-rendered digital file directly to the application layer.
- Static photo analysis cannot detect sensor attestation without hardware-backed cryptographic attestation (e.g. Android KeyStore / Apple Secure Enclave attestation).

### 4.3 Generative AI & High-End Deepfakes
- State-of-the-art diffusion models (e.g. Midjourney v6+, Flux, Stable Diffusion XL) can generate ultra-realistic human faces with natural skin pores and simulated sensor noise.
- While frequency-domain analysis catches low-tier synthetic faces, hyper-realistic diffusion models require conservative PAD scoring and fail-safe routing to human audit.

---

## 5. Recommended Bank Operating Policies & Thresholds

| Operating Policy | Quality Gate | PAD Risk Threshold | Match Cosine Sim ($\tau$) | Action on Borderline |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Account Opening** | 100% Pass | $\le 30$ (Low Risk) | $\ge \tau_{\text{calibrated}}$ (0.42 PLACEHOLDER) | Safe route to human `REVIEW` |
| **High-Risk Product Opening** (e.g. Unsecured Credit) | 100% Pass | $\le 20$ (Very Low Risk) | $\ge \tau_{\text{high-sec}}$ (TBD on empirical ROC) | Route to attended Video KYC |

---

## 6. Architectural Separation & Role of LLM Reviewer

- **The LLM is an Explainer, Never a Decider:** The LLM reviewer translates structured telemetry into human-readable audit summaries. It **never makes or overrides decisions**.
- **Privacy Firewall:** The LLM receives **zero raw images** and **zero customer personal data (PII)**.
- **Untrusted Output:** LLM responses are schema-validated. Any deviation or failure falls back to deterministic rule-based output.
