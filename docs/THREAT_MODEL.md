# Threat Model — FaceCheck Digital Account Opening (Photo-Only)

**System Classification:** Tier-1 Bank Onboarding Identity Verification  
**Operational Scope:** Photo-only Verification (1 ID Document Portrait Photo + 1 Selfie Photo; No Video Stream)  
**Document Version:** 1.2.0  
**Status:** **PROTOTYPE / PRE-PRODUCTION — NOT PRODUCTION-READY** (Pending consented real-attack golden set collection and accredited biometric calibration)  
**Author Role:** ARCHITECT (Reviewed with evidence by SECURITY REVIEWER, QA / RED-TEAM, ML EVALUATOR)

---

## 1. Executive Summary & Mandatory Production Readiness Disclaimer

> [!CAUTION]
> **CRITICAL OPERATIONAL NOTICE — PROJECT IS NOT PRODUCTION-READY**  
> FaceCheck is currently a prototype/development system. **It is strictly NOT approved for live production banking account opening** until the mandatory field collection milestone is completed:
> - **Field Data Collection Campaign (`DATA-CAMPAIGN-001`):** Must collect a consented real-attack golden test set comprising physical prints (inkjet, laser, matte, glossy), screen replays across prevalent local consumer smartphones (iPhone, Samsung, Xiaomi/Oppo), and simulated face-swaps under authentic branch/ambient lighting conditions with documented customer consent records (`consent_biometric_training = TRUE`).
> - **Threshold Calibration:** Operational matching threshold $\tau$ is currently a **PLACEHOLDER ($\tau = 0.42$)** and must be re-calibrated on verified validation data with subject-level bootstrap confidence intervals.

The **FaceCheck** system evaluates applicant identity during bank digital account opening by comparing an applicant's government-issued identity document (ID photo) against a live-captured selfie photo.

Because the system operates in a **photo-only mode** (single static 2D selfie and single static 2D ID capture, without active video challenge-response or depth sensing), it faces fundamental physical constraints. This threat model identifies all threat actors, attack vectors, vulnerabilities, and residual risks under STRIDE, documents strict alignment boundaries with NIST and ISO standards, and establishes architectural controls ensuring the system **fails safely to human REVIEW** whenever evidence is inconclusive.

---

## 2. Hard Limitations of Photo-Only Biometrics

> [!WARNING]
> **Strict Operational Limitation: Photo-Only Biometrics Cannot Guarantee Vitality.**  
> A single 2D static photograph lacks temporal micro-motions, 3D volumetric depth, corneal dynamic reflection, and physiological vitality markers (e.g., photoplethysmography/blood pulse). Consequently, photo-only verification **cannot reliably stop** high-end hardware/OS camera injection attacks, hyper-realistic diffusion deepfakes, or custom 3D silicone/latex masks.

### 2.1 Standards Alignment & Gap Matrix

FaceCheck **does not claim compliance** with NIST SP 800-63A or ISO/IEC 30107.

```
+--------------------+------------------------------------+-----------------------------------------------------------+
| Standard           | Formal Status                      | Technical Reality & Documented Gaps                       |
+--------------------+------------------------------------+-----------------------------------------------------------+
| NIST SP 800-63A    | Partially addresses guidance       | Partially addresses NIST SP 800-63A IAL2 biometric        |
| (Identity          |                                    | guidance; operational process and fraud controls required |
| Assurance)         |                                    | by 800-63A are NOT covered by this software.              |
+--------------------+------------------------------------+-----------------------------------------------------------+
| ISO/IEC 30107      | Designed to address specified      | Designed to address 2D print and screen-replay            |
| (Presentation      | attack types; NO accredited lab    | presentation attacks. 3D silicone/latex masks and         |
| Attack Detection)  | evaluation performed               | injection attacks are OUT OF SCOPE for photo-only         |
|                    |                                    | verification. No accredited laboratory evaluation         |
|                    |                                    | (e.g. iBeta, BSI) has been performed.                     |
+--------------------+------------------------------------+-----------------------------------------------------------+
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

## 3. Biometric Threshold Calibration & Statistical Requirements

### 3.1 SFace Matching Threshold ($\tau = 0.42$ Placeholder)
- The baseline threshold $\tau = 0.42$ is currently a **PLACEHOLDER**.
- Under ISO/IEC 19795-1 (Biometric Performance Testing and Reporting), operational thresholds cannot be adopted from default vendor values or small internal tests.
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

## 4. Mandatory Roadmap Task: Consented Real-Attack Golden Set Collection

Before FaceCheck can transition from **PROTOTYPE** to **PRODUCTION-READY**, the following data collection task must be executed:
- **Task ID:** `DATA-CAMPAIGN-001`
- **Owner:** ML Engineering Lead + Compliance Data Protection Officer
- **Consent Template:** [docs/CONSENT_FORM_TEMPLATE.md](CONSENT_FORM_TEMPLATE.md)
- **Objective:** Collect $N \ge 1,000$ genuine subjects and $N \ge 500$ presentation attack instruments across local target demographics.
- **Attack Variations:**
  1. *Print Attacks:* High-resolution inkjet and laser prints on matte and glossy photo stock, curved face wraps, cut-out eye holes.
  2. *Replay Attacks:* High-definition displays (OLED, IPS LCD) across prevalent local devices (Apple iPhone 13-16, Samsung Galaxy S/A series, Xiaomi/Redmi, Oppo/Vivo).
  3. *Synthetic / Face Swaps:* Deepfake face-swaps generated with contemporary diffusion models matched to target ID portraits.
- **Consent Governance:** Explicit biometric processing and storage consent form with recorded consent UUID, retention expiry, and right to erasure.
