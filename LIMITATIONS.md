# FaceCheck API — Model Limitations & Operational Disclaimers

**Document Classification:** Internal Banking Risk & Compliance  
**Audience:** Model Risk Officers, Ad Verification Teams, Compliance & Audit  

---

## 1. Executive Summary & Legal Disclaimer

> [!WARNING]
> **Risk Indicator Only, Not Final Proof.**  
> The FaceCheck API calculates a statistical risk probability score (1–100) based on facial biometrics, frequency-domain artifacts, texture consistency, and metadata heuristics. This score does **NOT** constitute definitive biometric proof or legal certainty of authenticity or fraud. It is designed solely as an automated preliminary risk indicator.

For any advertising approval decision involving customer accounts, credit products, or brand reputation, bank operational procedures **must mandate a human review workflow** when scores fall in the `UNCERTAIN` (31–70) or borderline `LIKELY_FAKE` (71–85) categories.

---

## 2. Technical Limitations

### 2.1 Image Quality & Resolution Sensitivities
- **Low Resolution:** Images smaller than $128 \times 128$ pixels or detected face crops smaller than $48 \times 48$ pixels lack sufficient high-frequency spatial detail for reliable micro-texture analysis.
- **Heavy JPEG Compression:** Lossy compression (compression ratio $< 0.012$ or severe $8 \times 8$ block boundary discontinuities) destroys subtle synthetic generation artifacts.
- **Extreme Exposure & Blur:** Excessive Gaussian blur (Laplacian variance $< 50$) or severe over/under-exposure suppresses edge and gradient cues.
- **Mitigation:** The API evaluates input quality pre-inference and outputs a `"quality"` object (`status: "LOW_QUALITY"`) with specific warnings when confidence is impaired.

### 2.2 Stylized Art, Cartoons, and Illustrations
- **Non-Photorealistic Artworks:** Vector illustrations, 3D CGI renderings, anime characters, and hand-drawn cartoon faces have synthetic texture distributions and sharp artificial edge boundaries that differ fundamentally from natural camera sensors.
- **Expected Behavior:** Stylized face images are evaluated by the detector as synthetic (`LIKELY_FAKE` or high `UNCERTAIN`), because they lack natural skin pores, camera optical aberrations, and sensor noise.
- **Mitigation:** Ad submission workflows should allow tagging creative submissions as "Illustrated / Artistic Creative" to route them to appropriate policy tracks.

### 2.3 Evasion and Adversarial Attacks
- **Screenshot of a Screen (Moiré):** Photographing an AI image displayed on an LCD monitor introduces physical camera sensor artifacts and moiré patterns, which can mask generative signatures.
- **Re-compression & Post-Processing:** Applying strong analog grain filters, heavy blur, or noise addition can alter high-frequency Fourier spectra.
- **Adversarial Perturbations:** Sophisticated attackers who deliberately craft imperceptible pixel perturbations can attempt to manipulate deep neural network activations.

### 2.4 Biometric Face Verification Limitations (1:1 Matching)
- **Separate Architectural Components:** Face matching, image quality assessment, and anti-spoofing operate as completely independent modules because they answer fundamentally different questions. An authentic face match does NOT establish physical liveness, and a high-quality photograph does NOT establish identity.
- **False Match & False Non-Match Realities:** Biometric 1:1 face matching is probabilistic. Operating at an FMR (False Match Rate) of $0.1\%$ entails an operational trade-off with FNMR (False Non-Match Rate). Appearance changes due to major age progression, trauma, medical conditions, dramatic weight changes, or extreme lighting can cause false non-matches.
- **Non-Match Does Not Prove Fraud:** A non-matching face score indicates identity discrepancy between the two presented photos, not intentional fraud or malicious impersonation.

### 2.5 Presentation-Attack Detection (PAD) & Static Image Liveness
- **Static Photos Never Prove Liveness:** A single static 2D photograph alone CANNOT prove that a human being is physically live and present. Static PAD checks evaluate surface artifacts (paper texture, halftone dots, moiré grids, screen bezels, specular reflection) only.
- **Interactive Verification Mandate:** For high-assurance KYC transactions (e.g. account opening, large fund transfers), an interactive challenge-response sequence (randomized head pose challenges) or video sequence is required.

### 2.6 Demographic Diversity & Parity
- Face detection utilizes the OpenCV YuNet ONNX architecture, which was benchmarked across diverse lighting, poses, and international facial features.
- However, model performance across varying demographic groups, skin tones, age groups, and religious or cultural headwear (e.g., turbans, hijabs) must be continuously benchmarked using the bank's internal test set via `scripts/evaluate.py` to ensure parity.

---

## 3. Recommended Operational Thresholds

Based on bank risk appetite:

| Operating Mode | Real Range (1–100) | Uncertain Range | Fake Range | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Default Balanced** | 1 to 30 | 31 to 70 | 71 to 100 | Standard digital advertising clearance |
| **High Security (Zero Fake Tolerance)** | 1 to 20 | 21 to 55 | 56 to 100 | High-value celebrity endorsement ads, executive spokesperson verification |
| **High Throughput (Low False Alarms)** | 1 to 40 | 41 to 80 | 81 to 100 | High-volume merchant marketplace creatives |

---

## 4. Model Maintenance & Lifecycle Protocol

Because the FaceCheck service operates 100% offline in air-gapped bank environments:
1. **Periodic Evaluation:** Run `python scripts/evaluate.py` monthly on a freshly curated bank evaluation dataset (`eval/data/real/` and `eval/data/fake/`).
2. **Retraining & Calibration:** If newly released generative models (e.g. Midjourney updates, newer diffusion architectures) cause false negative rates to exceed 5%, collect representative examples and execute offline retraining via `python scripts/train_classifier.py`.
3. **Integrity Verification:** Always update and verify cryptographic SHA-256 hashes in `models/CHECKSUMS.txt` using `python scripts/verify_checksums.py` prior to deploying any retrained model into production.
