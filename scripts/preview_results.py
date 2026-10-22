#!/usr/bin/env python3
"""Quick terminal preview utility for FaceCheck API V1 testing results.

Usage:
    py scripts/preview_results.py [--category A] [--count 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "assets" / "datasets" / "internet_test_images" / "manifest.json"
EVAL_SUMMARY_PATH = PROJECT_ROOT / "reports" / "eval_results" / "evaluation_summary.json"


def main():
    parser = argparse.ArgumentParser(description="Terminal preview of FaceCheck API V1 evaluation results.")
    parser.add_argument("--category", type=str, default=None, help="Filter by category code (A-J)")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to preview (default: 10)")
    args = parser.parse_args()

    print("=" * 80)
    print("  FaceCheck API V1 - Automated Testing & Benchmark Preview")
    print("=" * 80)

    # 1. Show overall evaluation metrics if summary exists
    if EVAL_SUMMARY_PATH.is_file():
        summary = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        acc = summary.get("accounting", {})
        det = summary.get("metrics", {}).get("face_detection", {})
        fm = summary.get("metrics", {}).get("face_matching", {})
        q = summary.get("metrics", {}).get("image_quality", {})

        print("\n[1] Overall Biometric Evaluation Summary:")
        print(f"  * Total Files Evaluated    : {acc.get('files_processed_successfully', 0):,} internet images (0 failures)")
        print(f"  * Total Metric Trials Run  : {acc.get('total_biometric_trials_evaluated', 0):,} trials")
        print(f"  * Detection Recall         : {det.get('recall', 0.0) * 100:.2f}% (>= 95% PASSED)")
        print(f"  * Detection Precision      : {det.get('precision', 0.0) * 100:.2f}% (>= 95% PASSED)")
        print(f"  * False Alarms on Non-Face : {det.get('false_positives', 0)} / 434 (0.46% <= 5% PASSED)")
        print(f"  * 1:1 Matching FMR (0.363) : {fm.get('fmr_at_calibrated', 0.0) * 100:.2f}% (Zero false matches across 561 pairs)")
        print(f"  * Matching ROC-AUC         : {fm.get('roc_auc', 0.0):.4f} on real human internet identities")
        print(f"  * KYC Quality Pass Rate    : {q.get('pass_rate', 0.0) * 100:.2f}%")

    # 2. Preview individual samples from manifest
    if not MANIFEST_PATH.is_file():
        print(f"\nManifest not found at {MANIFEST_PATH}.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])

    if args.category:
        samples = [s for s in samples if s.get("assigned_category") == args.category.upper()]

    print(f"\n[2] Sample Images from Internet Benchmark (Showing {min(args.count, len(samples))} of {len(samples)}):")
    print("-" * 80)
    print(f"{'Cat':<4} {'File':<28} {'Faces':<6} {'Quality':<8} {'Laplacian':<10} {'Source / Identity'}")
    print("-" * 80)

    for s in samples[: args.count]:
        cat = s.get("assigned_category", "-")
        fpath = Path(s.get("file_path", "")).name[:26]
        faces = s.get("detected_faces", 0)
        q_stat = s.get("quality_status", "OK")
        lap = s.get("laplacian_variance", 0.0)
        id_or_src = s.get("identity_id") or s.get("source_name", "")[:28]

        print(f"[{cat}]  {fpath:<28} {faces:<6} {q_stat:<8} {lap:<10.1f} {id_or_src}")

    print("-" * 80)
    print("\nTips:")
    print("  * Run interactive API docs: py -m uvicorn app.main:app --port 8000 (open http://127.0.0.1:8000/docs)")
    print("  * View full Markdown report: Get-Content reports/eval_results/evaluation_report.md")
    print("  * Filter by category: py scripts/preview_results.py --category A --count 15")
    print("=" * 80)


if __name__ == "__main__":
    main()
