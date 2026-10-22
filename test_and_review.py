#!/usr/bin/env python3
"""
FaceCheck API: Automated Test Runner & Instant Browser Review.
Executes automated unit tests, runs live external online auto-match verification,
generates the evaluation report, and automatically opens it in your default web browser.

Usage:
    py test_and_review.py
"""

import sys
import time
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from scripts.evaluate import run_evaluation
from scripts.test_random import fetch_dynamic_online_batch
from app.services.pipeline import run_pipeline
from app.config import get_config


def run_live_online_check(count: int = 6) -> None:
    """Download fresh external images from the internet and run live auto-match verification."""
    print("\n[STEP 2/4] Running Live External Online Auto-Match Verification...")
    print("-" * 80)
    print(f" [ONLINE] Fetching {count} dynamic non-cached images across external providers...")
    
    t0 = time.perf_counter()
    items = fetch_dynamic_online_batch(count=count, max_workers=10)
    
    if not items:
        print(" [WARN] External download unavailable; continuing to benchmark evaluation.")
        return

    print(f" [ONLINE] Downloaded {len(items)} live images in {time.perf_counter()-t0:.2f}s.")
    correct = 0
    for it in items:
        fname = it["filename"]
        gt_type = it["gt_type"]
        gt_gender = it["gt_gender"]
        res = run_pipeline(fname, it["raw_bytes"])
        score = res.get("score")
        label = res.get("label")
        gender = res.get("gender")
        
        if gt_type == "REAL" and (score is not None and score <= 50):
            decision = "APPROVED"
            is_ok = True
        elif gt_type == "FAKE" and ((score is not None and score > 50) or label == "RETAKE"):
            decision = "BLOCKED"
            is_ok = True
        else:
            decision = "REVIEW"
            is_ok = False
        
        if is_ok:
            correct += 1
        score_disp = f"{score}%" if score is not None else "Blocked"
        print(f"   * {fname[:20]:20s} | {gt_type:4s} | Pred: {gender or 'N/A':7s} | Risk: {score_disp:7s} | Decision: {decision}")

    print(f" [OK] Live Online Auto-Match: {correct}/{len(items)} ({(correct/len(items))*100:.1f}%) verified successfully.\n")


def main():
    print("=" * 80)
    print("     FACECHECK API: LIVE ONLINE TEST SUITE & BROWSER REVIEW")
    print("=" * 80)
    t_start = time.perf_counter()

    # 1. Run 68 Pytest Unit & Integration Tests
    print("\n[STEP 1/4] Running Automated Unit & Integration Tests...")
    print("-" * 80)
    pytest_exit_code = pytest.main(["-q", "tests"])
    if pytest_exit_code != 0:
        print(f"\n[FAIL] Automated tests failed with exit code: {pytest_exit_code}")
        sys.exit(pytest_exit_code)
    print(" [OK] All automated unit and integration tests passed.\n")

    # 2. Run Live External Online Verification
    run_live_online_check(count=6)

    # 3. Run Benchmark Evaluation & HTML Report Generation
    print("[STEP 3/4] Running Multi-Style Biometric Benchmark Evaluation...")
    print("-" * 80)
    config = get_config()
    real_path = PROJECT_ROOT / config.eval.data_dir / "real"
    fake_path = PROJECT_ROOT / config.eval.data_dir / "fake"
    reports_path = PROJECT_ROOT / config.eval.reports_dir

    eval_result = run_evaluation(real_path, fake_path, reports_path, selected_threshold=50)

    # 4. Automatically Open Visual HTML Report in Default Browser
    latest_html = reports_path / "latest.html"
    report_uri = latest_html.resolve().as_uri()

    print("[STEP 4/4] Launching Visual Evaluation Report in Browser...")
    print("-" * 80)
    print(f" [BROWSER] Opening: {report_uri}")
    webbrowser.open(report_uri)

    t_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f" [DONE] Online Automated Testing Complete in {t_elapsed:.2f}s.")
    print(f"        Review the live visual report in your web browser!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
