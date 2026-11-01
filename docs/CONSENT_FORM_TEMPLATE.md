# Biometric Data Collection & Processing Consent Form (Template)

**Campaign Identifier:** `DATA-CAMPAIGN-001`  
**Governing Institution:** CP Bank Biometric Research & Risk Governance  
**Data Protection Officer Contact:** `dpo@cpbank.local`  
**Regulatory Alignment:** General Data Protection Regulation (GDPR Art. 9), National Banking Privacy Act

---

## 1. Purpose of Data Collection

The purpose of this biometric data collection is to build and benchmark presentation attack detection (PAD) and biometric face verification models for digital banking account opening. The data will be used strictly within the bank's isolated, air-gapped infrastructure to benchmark resistance against:
- High-resolution photographic prints on matte, glossy, and cardstock paper.
- Digital screen replays across consumer smartphone displays (iPhone, Samsung Galaxy, Xiaomi/Oppo).
- Simulated facial synthetic manipulation (deepfake and face-swap detection).

---

## 2. Collected Biometric Categories

1. **Frontal Portrait (Selfie):** Captured using a smartphone camera under varied indoor and natural ambient lighting.
2. **Identification Document Portrait:** Digital capture of applicant identification document photo.
3. **Presentation Attack Instruments (PAIs):** Secondary captures of printed photos and screen replays of the portrait.
4. **Metadata:** Device make and model, timestamp, ambient lux estimate. **No financial or credit records are collected.**

---

## 3. Storage, Encryption & Data Isolation

- **Application-Level Encryption:** All biometric images are encrypted prior to persistence using AES-256-GCM authenticated encryption. Encryption keys are managed by the bank's internal Key Management Service (KMS).
- **Environment Isolation:** Raw captures are stored in an isolated internal database schema (`ml_training.training_samples`) inaccessible from the public internet.
- **Zero Third-Party Cloud Sharing:** Biometric data is never shared with third-party cloud AI vendors or external commercial brokers.

---

## 4. Participant Rights & Withdrawal Protocol

- **Voluntary Participation:** Participation is 100% voluntary.
- **Right to Erasure / Withdrawal:** You may withdraw consent at any time without penalty by contacting `dpo@cpbank.local` citing your unique **Consent UUID**. Upon receipt, all stored raw images and derived embeddings will be permanently cryptographically erased within 7 business days.
- **Mandatory Retention Limit:** Data collected under this campaign will be automatically purged 24 months from the date of capture.

---

## 5. Formal Consent Acknowledgement

By signing below, you explicitly consent to the collection, cryptographic storage, and model benchmarking of your facial biometric data:

- **Participant Full Name:** ________________________________________________
- **Unique Consent ID (UUID):** `cst-____________________________________`
- **National ID / Employee ID (Hashed):** ____________________________________
- **Date:** ________________________
- **Signature:** ________________________________________

*Bank Verification Officer Signature:* _______________________________________
