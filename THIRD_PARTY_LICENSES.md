# Third-Party Software & Model Licenses

This document lists the open-source software packages, models, and bundled assets used in the **FaceCheck API** along with their respective license types and copyright notices.

---

## 1. Runtime Python Dependencies

| Package | Version | License | Copyright / Origin |
| :--- | :--- | :--- | :--- |
| **FastAPI** | `>=0.100.0` | MIT | Tiangolo / Sebastián Ramírez |
| **Starlette** | `>=0.27.0` | BSD-3-Clause | Encode OSS Ltd |
| **Uvicorn** | `>=0.22.0` | BSD-3-Clause | Encode OSS Ltd |
| **Pydantic** | `>=2.0.0` | MIT | Samuel Colvin & Pydantic Contributors |
| **Pillow (PIL)** | `>=9.5.0` | HPND | Secret Labs AB, Fredrik Lundh, Alex Clark |
| **PyYAML** | `>=6.0` | MIT | Kirill Simonov |
| **Cryptography** | `>=41.0.0` | Apache-2.0 / BSD-3-Clause | The Cryptography Developers |
| **NumPy** | `>=1.24.0` | BSD-3-Clause | NumPy Developers / Travis Oliphant |
| **ExifRead** | `>=3.0.0` | BSD-3-Clause | Gene Cash, Ianaré Sévi |
| **Python-Multipart** | `>=0.0.6` | Apache-2.0 | Andrew Dunham |
| **HTTPX** | `>=0.24.0` | BSD-3-Clause | Encode OSS Ltd |
| **Pytest** | `>=7.0.0` | MIT | Holger Krekel & Pytest Developers |

---

## 2. Bundled AI Neural Network Models

### 2.1 YuNet Face Detector (`models/face_detection_yunet_2023mar.onnx`)
- **Origin**: OpenCV Model Zoo / Shiqi Yu (Southern University of Science and Technology)
- **License**: Apache License 2.0
- **Integrity Checksum (SHA-256)**: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- **Format**: ONNX (Open Neural Network Exchange), no executable code.

### 2.2 Deepfake Texture & Frequency Classifier (`models/deepfake_detector/model.safetensors`)
- **Origin**: Trained offline on bank advertising verification benchmarks.
- **License**: Proprietary / Apache-2.0 Compatible
- **Integrity Checksum (SHA-256)**: `d9f590e6ac162783598d6c9830209167b350bb47ea7dd911e28603dac5c5b020`
- **Format**: Hugging Face `.safetensors` (pure numeric weights tensor file, zero pickle code execution risk).

---

## 3. Bundled Frontend Assets (Air-Gapped Swagger UI)

### 3.1 Swagger UI (`app/static/swagger/`)
- **Origin**: SmartBear Software
- **License**: Apache License 2.0
- **Files**: `swagger-ui-bundle.js`, `swagger-ui.css`
- **Offline Guarantee**: Bundled locally into static folder to prevent any outbound CDN or Google Fonts traffic.

---

## 4. Key License Texts Summary

### Apache License Version 2.0
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at `http://www.apache.org/licenses/LICENSE-2.0`.

### MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

### BSD-3-Clause License
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the copyright notice and this list of conditions and disclaimer are preserved.
