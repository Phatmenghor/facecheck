# FaceCheck Operations Runbook

**System Classification:** Biometric Authentication & Presentation Attack Detection System  
**Deployment Target:** Local / Air-Gapped Banking Server Environment  
**Stack Architecture:** Docker Compose (PostgreSQL 15.6, FastAPI API, PyTorch ML Worker, Nginx Read-Only Monitor, Ollama Offline LLM)  
**Security Baseline:** OWASP ASVS Level 2, NIST SP 800-63B Alignment, Zero-Secrets, Dual-Network Isolation  

---

## 1. Quick Start & Prerequisites

### 1.1 Prerequisites
- **Host Machine:** Windows 10/11 with Docker Desktop & WSL2 backend, or Linux server with Docker Engine $\ge 24.0$ and Docker Compose v2.
- **Host Dependencies:** **Zero required on host except Docker and Docker Compose.**
- **Network Boundaries:**
  - `internal`: Air-gapped backend network (PostgreSQL, Ollama, ML Worker). Zero external internet gateway.
  - `frontend`: Reverse proxy ingress (FastAPI API and Nginx Read-Only Monitor).
  - Ports published to host on `127.0.0.1` **ONLY** (never `0.0.0.0`).

### 1.2 Initial Provisioning (One-Command Setup)
From project root:

**Using Git Bash / WSL2 / Linux:**
```bash
bash scripts/setup.sh
# or using Makefile:
make setup
```

**Using Native Windows PowerShell:**
```powershell
.\scripts\setup.ps1
```

`setup.sh` automatically performs:
1. Validates Docker daemon connectivity and container runtime version.
2. Generates new cryptographic high-entropy secrets for every database role (`facecheck_app`, `facecheck_ml_worker`, `facecheck_readonly`, `facecheck_migrator`, `POSTGRES_PASSWORD`, API key, AES encryption master key, HMAC key) and writes them to `.env`.
3. Builds hardened multi-stage Docker images (`facecheck-api`, `facecheck-ml-worker`, `facecheck-monitor-web`).
4. Starts services in detached mode on private subnets.
5. Waits for healthchecks to report `healthy` across all 5 containers.
6. Executes migrations `001` through `005` and applies role boundaries.
7. Verifies local offline Ollama model `llama3.2`.
8. Emits service status and local URL endpoints.

---

## 2. Standard Operational Procedures

### 2.1 Starting the Stack
```bash
bash scripts/start.sh
# or: make start
# or PowerShell: .\scripts\start.ps1
```

### 2.2 Stopping the Stack
```bash
bash scripts/stop.sh
# or: make stop
# or PowerShell: .\scripts\stop.ps1
```

### 2.3 Restarting the Stack
```bash
bash scripts/restart.sh
# or: make restart
```

### 2.4 Monitoring Logs
```bash
# Follow all container logs
bash scripts/logs.sh

# Follow specific service
bash scripts/logs.sh api
bash scripts/logs.sh ml-worker
bash scripts/logs.sh postgres
bash scripts/logs.sh monitor-web
bash scripts/logs.sh ollama
```

---

## 3. Database Administration & Migrations

### 3.1 Applying Migrations
Migrations are fully idempotent and versioned under `db/migrations/`:
```bash
bash scripts/migrate.sh
# or: make migrate
# or PowerShell: .\scripts\migrate.ps1
```
Execution order:
1. `db/roles.sql`: Revokes public schema access; defines `facecheck_migrator`, `facecheck_app`, `facecheck_ml_worker`, `facecheck_readonly`.
2. `001_create_schemas_and_roles.sql`: Creates `kyc_prod` and `ml_training` schemas.
3. `002_create_kyc_prod_tables.sql`: `decisions`, `encrypted_images`, `audit_chain`, `kill_switch_state`, `retention_policies`.
4. `003_create_ml_training_tables.sql`: `consent_records`, `training_samples`, `model_registry`, `golden_test_set`, `training_run_audit`.
5. `004_create_sanitized_views_and_grants.sql`: Firewall views and sanitized monitoring views with least-privilege `GRANT` statements.
6. `005_audit_chain_trigger_and_anchors.sql`: Immutability trigger `trg_prevent_audit_mutation` and external WORM anchor registry.

---

## 4. Testing & Verification

### 4.1 Running All In-Docker Tests
All tests run inside the hardened API container against the real PostgreSQL container:
```bash
bash scripts/test.sh
# or: make test
# or PowerShell: .\scripts\test.ps1
```
This suite automatically verifies:
- API endpoint contracts and input validation.
- Biometric facial matching and PAD pipeline logic.
- Database role boundaries (verifies `facecheck_app` cannot write to `ml_training`, `facecheck_ml_worker` cannot delete from `kyc_prod`, and `facecheck_readonly` cannot mutate any table).
- Immutability trigger rejecting any SQL UPDATE or DELETE on `kyc_prod.audit_chain`.
- SafeTensors model loading (verifies zero unsafe pickle `torch.load` calls).

### 4.2 Comprehensive System Verification
```bash
bash scripts/verify_all.sh
# or: make verify-all
# or PowerShell: .\scripts\verify_all.ps1
```

---

## 5. Security & Secret Management

### 5.1 Automated Security & Vulnerability Scans
```bash
bash scripts/security_scan.sh
# or: make scan
```
The scan executes:
1. **Full Git History Audit:** Scans all commits and diff branches using `scripts/scan_git_history.py` for credentials or secrets.
2. **Dependency Audit:** Runs `pip-audit` against installed packages for CVEs.
3. **Pickle / SafeTensors Inspection:** Verifies absence of arbitrary code execution vectors.
4. **Container Hardening Check:** Verifies non-root UIDs, capability dropping, and read-only filesystem immutability.

### 5.2 Secret Rotation Runbook (Zero Downtime)
To rotate all database role passwords, application API keys, and HMAC signatures:
```bash
bash scripts/rotate_secrets.sh
# or: make rotate-secrets
```
Procedure executed by `rotate_secrets.sh`:
1. Generates 6 new cryptographically random credentials.
2. Connects to PostgreSQL as superuser and updates passwords via `ALTER ROLE ... WITH PASSWORD`.
3. Updates `.env` configuration file with restricted file permissions (`600`).
4. Gracefully restarts `api` and `ml-worker` containers to consume the new credentials.
5. Verifies container health checks.

---

## 6. Backup & Disaster Recovery

### 6.1 Encrypted Database Backup
Exports both `kyc_prod` and `ml_training` schemas directly into an AES-256-CBC encrypted archive with PBKDF2 key derivation (100,000 iterations):
```bash
bash scripts/backup.sh
# or: make backup
```
- Output: `backups/facecheck_backup_<TIMESTAMP>.sql.enc`
- Companion: `backups/facecheck_backup_<TIMESTAMP>.sql.enc.sha256`
- **Security Invariant:** Plaintext SQL is never written to disk. OpenSSL encrypts the live `pg_dump` standard output stream in memory.

### 6.2 Restoring from Encrypted Backup
```bash
bash scripts/restore.sh backups/facecheck_backup_<TIMESTAMP>.sql.enc
# or: make restore file=backups/facecheck_backup_<TIMESTAMP>.sql.enc
```
- Decrypts the stream directly into the PostgreSQL input pipe without creating any decrypted temporary files on disk.

---

## 7. Incident Response & Emergency Procedures

### 7.1 Emergency Training Kill Switch
If model degradation, label drift, or poisoning is suspected in automated training:
```bash
# Option A: Engage via REST API
curl -X POST http://127.0.0.1:8000/api/v1/learning/kill-switch/engage \
     -H "X-API-Key: <FACECHECK_API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{"reason": "Suspected label drift detected during audit", "initiated_by": "secops_admin"}'

# Option B: Stop ml-worker container directly
docker compose stop ml-worker
```

### 7.2 Resetting Development Stack
To completely destroy all containers, local test state, and Docker volumes for a clean slate:
```bash
bash scripts/reset_dev.sh
# or: make reset-dev
# With automated force flag:
bash scripts/reset_dev.sh --force
```
*Note: `reset_dev.sh` validates that `ENV=development` and aborts immediately if executed on non-development configurations.*

---

## 8. Network & Port Allocation Map

| Service Name | Container Name | Internal Network | Frontend Network | Published Host Port | Protocol | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `postgres` | `facecheck_postgres` | Yes | No | None (Internal only) | TCP / PostgreSQL | Enterprise data store |
| `ollama` | `facecheck_ollama` | Yes | No | `127.0.0.1:11434` | HTTP / REST | Local offline explainer |
| `api` | `facecheck_api` | Yes | Yes | `127.0.0.1:8000` | HTTP / REST | FastAPI core biometric inference |
| `ml-worker` | `facecheck_ml_worker` | Yes | No | None (Daemon only) | None | Autonomous background training worker |
| `monitor-web` | `facecheck_monitor_web` | No | Yes | `127.0.0.1:8080` | HTTP / HTML | Read-only web monitoring dashboard |
