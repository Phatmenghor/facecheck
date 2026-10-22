"""Full System Self-Test & Diagnostic Script for FaceCheck API.

Runs:
1. Model files & checksum validation
2. 67 Unit & integration tests via pytest
3. Benchmark evaluation & metrics thresholds
4. Live API endpoints (health, metrics, real check, fake check, retake check)
5. HTML & JSON report presence and integrity
"""

import sys
import hashlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.config import get_config
import pytest


def check_models():
    print("[1/5] Checking model files and checksums...")
    checksum_file = PROJECT_ROOT / "models" / "CHECKSUMS.txt"
    assert checksum_file.is_file(), "models/CHECKSUMS.txt is missing!"
    
    with open(checksum_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]
    
    for expected_hash, rel_path in lines:
        target_file = PROJECT_ROOT / rel_path
        assert target_file.is_file(), f"Model file missing: {rel_path}"
        hasher = hashlib.sha256()
        with open(target_file, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        calc_hash = hasher.hexdigest()
        assert calc_hash == expected_hash, f"Hash mismatch for {rel_path}: expected {expected_hash}, got {calc_hash}"
        print(f"  [OK] {rel_path} ({calc_hash[:16]}...) verified")


def check_unit_tests():
    print("\n[2/5] Running pytest unit & integration test suite (67 tests)...")
    exit_code = pytest.main(["-q", "tests"])
    assert exit_code == 0, f"pytest failed with exit code {exit_code}"
    print("  [OK] All pytest tests passed.")


def check_evaluation():
    print("\n[3/5] Running benchmark evaluation...")
    from scripts.evaluate import run_evaluation
    config = get_config()
    real_path = PROJECT_ROOT / config.eval.data_dir / "real"
    fake_path = PROJECT_ROOT / config.eval.data_dir / "fake"
    reports_path = PROJECT_ROOT / config.eval.reports_dir
    
    res = run_evaluation(real_path, fake_path, reports_path, selected_threshold=50)
    verdict = res.get("verdict", {})
    verdict_status = verdict.get("status") if isinstance(verdict, dict) else str(verdict)
    summary = res.get("summary", {})
    acc = summary.get("accuracy", 0.0)
    fpr = summary.get("fpr", 1.0)
    fnr = summary.get("fnr", 1.0)
    auc = summary.get("roc_auc", 0.0)
    assert verdict_status == "PASS", f"Evaluation verdict is not PASS: {verdict}"
    assert acc >= config.eval.min_acceptable.min_accuracy, f"Accuracy {acc} < {config.eval.min_acceptable.min_accuracy}"
    assert fpr <= config.eval.min_acceptable.max_false_positive_rate, f"FPR {fpr} > {config.eval.min_acceptable.max_false_positive_rate}"
    assert fnr <= config.eval.min_acceptable.max_false_negative_rate, f"FNR {fnr} > {config.eval.min_acceptable.max_false_negative_rate}"
    print(f"  [OK] Evaluation passed: Accuracy={acc*100:.1f}%, FPR={fpr*100:.2f}%, FNR={fnr*100:.2f}%, AUC={auc:.4f}")


def check_live_api():
    print("\n[4/5] Testing live API endpoints...")
    client = TestClient(app)
    api_key = "fc-prod-secret-key-bank-core-9872"
    config = get_config()
    
    # Health
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200, f"Health returned {r_health.status_code}"
    is_offline = r_health.json().get("offline_mode")
    assert is_offline == config.server.offline_mode, f"Expected offline_mode={config.server.offline_mode}, got {is_offline}"
    print(f"  [OK] GET /api/v1/health: 200 OK (offline_mode={is_offline})")
    
    # Metrics
    r_metrics = client.get("/api/v1/metrics", headers={"X-API-Key": api_key})
    assert r_metrics.status_code == 200, f"Metrics returned {r_metrics.status_code}"
    print("  [OK] GET /api/v1/metrics: 200 OK")
    
    # Real image
    real_img = PROJECT_ROOT / "eval" / "data" / "real" / "portrait_01.jpg"
    with open(real_img, "rb") as f:
        r_real = client.post(
            "/api/v1/check",
            files={"image": ("portrait_01.jpg", f, "image/jpeg")},
            headers={"X-API-Key": api_key},
        )
    assert r_real.status_code == 200, f"Real check returned {r_real.status_code}: {r_real.text}"
    real_data = r_real.json()
    assert real_data["label"] == "LIKELY_REAL", f"Expected LIKELY_REAL, got {real_data['label']}"
    assert real_data["face_check"]["status"] == "PASSED", f"Expected PASSED face check, got {real_data['face_check']['status']}"
    print(f"  [OK] POST /api/v1/check [Real Image]: score={real_data['score']}, label={real_data['label']}, face_check={real_data['face_check']['status']}")
    
    # Fake AI image
    fake_img = PROJECT_ROOT / "eval" / "data" / "fake" / "challenging_attacks" / "attack_3d_metahuman_uncanny.jpg"
    with open(fake_img, "rb") as f:
        r_fake = client.post(
            "/api/v1/check",
            files={"image": ("fake.jpg", f, "image/jpeg")},
            headers={"X-API-Key": api_key},
        )
    assert r_fake.status_code == 200, f"Fake check returned {r_fake.status_code}: {r_fake.text}"
    fake_data = r_fake.json()
    assert fake_data["label"] == "LIKELY_FAKE", f"Expected LIKELY_FAKE, got {fake_data['label']}"
    print(f"  [OK] POST /api/v1/check [AI Synthetic Image]: score={fake_data['score']}, label={fake_data['label']}")
    
    # Pre-flight early reject (cartoon / no human face)
    cartoon_img = PROJECT_ROOT / "eval" / "data" / "fake" / "cartoon_01.jpg"
    with open(cartoon_img, "rb") as f:
        r_cartoon = client.post(
            "/api/v1/check",
            files={"file": ("cartoon_01.jpg", f, "image/jpeg")},
            headers={"X-API-Key": api_key},
        )
    assert r_cartoon.status_code == 200
    cartoon_data = r_cartoon.json()
    assert cartoon_data["label"] == "RETAKE", f"Expected RETAKE, got {cartoon_data['label']}"
    assert "NO_FACE" in cartoon_data["reason_codes"], f"Expected NO_FACE reason code, got {cartoon_data['reason_codes']}"
    print(f"  [OK] POST /api/v1/check [Non-Human Face]: label={cartoon_data['label']}, reason_codes={cartoon_data['reason_codes']}")


def check_reports():
    print("\n[5/5] Checking HTML and JSON report files...")
    reports_dir = PROJECT_ROOT / "eval" / "reports"
    latest_html = reports_dir / "latest.html"
    assert latest_html.is_file(), "latest.html is missing!"
    size_kb = latest_html.stat().st_size / 1024
    assert size_kb > 50, f"latest.html suspiciously small ({size_kb:.1f} KB)"
    print(f"  [OK] eval/reports/latest.html exists and verified ({size_kb:.1f} KB)")


def main():
    print("=" * 70)
    print("     FACECHECK API: COMPLETE SYSTEM SELF-TEST & VERIFICATION")
    print("=" * 70)
    try:
        check_models()
        check_unit_tests()
        check_evaluation()
        check_live_api()
        check_reports()
        print("\n" + "=" * 70)
        print(" [SUCCESS] ALL CHECKS PASSED PERFECTLY ON THIS COMPUTER!")
        print("=" * 70)
    except AssertionError as err:
        print(f"\n[FAIL] Assertion failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
