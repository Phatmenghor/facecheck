#!/usr/bin/env python3
"""Evaluation harness for FaceCheck API.

Measures detector accuracy, precision, recall, false positive rate (real marked fake),
false negative rate (fake marked real), ROC-AUC, and confusion matrix across multiple thresholds.

Generates standalone JSON and HTML reports (zero external CSS/JS) in eval/reports/.
Evaluates performance against config.yaml min_acceptable limits and prints PASS/FAIL.

Usage:
    python scripts/evaluate.py [--real-dir eval/data/real] [--fake-dir eval/data/fake]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config
from app.services.pipeline import run_pipeline
from scripts.report_generator import make_thumbnail_b64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("facecheck.evaluate")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_environment_cache_signature() -> str:
    """Generate unique hash representing active model weights and config.yaml.

    Guarantees that modifying model weights or thresholds automatically invalidates old scores.
    """
    h = hashlib.sha256()
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.is_file():
        h.update(config_path.read_bytes())
    checksums_path = PROJECT_ROOT / "models" / "CHECKSUMS.txt"
    if checksums_path.is_file():
        h.update(checksums_path.read_bytes())
    # Fingerprint pipeline code and models directly
    for svc_path in [
        PROJECT_ROOT / "app" / "services" / "gender_detector.py",
        PROJECT_ROOT / "app" / "services" / "face_check.py",
        PROJECT_ROOT / "app" / "services" / "pipeline.py",
        PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx",
        PROJECT_ROOT / "models" / "deepfake_detector" / "model.safetensors",
    ]:
        if svc_path.is_file():
            stat = svc_path.stat()
            h.update(f"{svc_path.name}_{stat.st_size}_{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculate 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / total) + (z ** 2) / (4.0 * total ** 2))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (round(low, 4), round(high, 4))


def auc_confidence_interval(auc: float, n_pos: int, n_neg: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculate 95% Hanley-McNeil confidence interval for ROC-AUC."""
    if n_pos <= 0 or n_neg <= 0:
        return (round(auc, 4), round(auc, 4))
    q1 = auc / (2.0 - auc) if (2.0 - auc) != 0 else 0.5
    q2 = 2.0 * (auc ** 2) / (1.0 + auc) if (1.0 + auc) != 0 else 0.5
    se_sq = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(0.0, se_sq))
    low = max(0.0, auc - z * se)
    high = min(1.0, auc + z * se)
    return (round(low, 4), round(high, 4))


def compute_file_sha256(path: Path) -> str:
    """Compute exact SHA-256 binary hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(path: Path, hash_size: int = 8) -> str:
    """Compute 64-bit perceptual difference hash (dHash) for visual duplication detection."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.int32)
            diff = arr[:, 1:] > arr[:, :-1]
            return "".join(f"{b:02x}" for b in np.packbits(diff.flatten()))
    except Exception:
        return ""


def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute bitwise Hamming distance between two hexadecimal hashes."""
    if not hex1 or not hex2 or len(hex1) != len(hex2):
        return 999
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    return bin(val1 ^ val2).count("1")


def check_train_eval_overlap(
    eval_files: List[Path],
    train_dir: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """Verify that zero images or person groups overlap between training and evaluation splits.

    Detects:
    1. Exact SHA-256 byte duplicates.
    2. Perceptual image duplicates (dHash hamming distance <= 2).
    3. Person/source group leakage from metadata CSV.

    Returns:
        Tuple of (has_overlap: bool, violations: List[str]).
    """
    violations: List[str] = []

    # 1. Perceptual & exact file hashes for eval files
    eval_sha_map: Dict[str, Path] = {}
    eval_dhash_map: Dict[str, Path] = {}

    for ep in eval_files:
        sha = compute_file_sha256(ep)
        eval_sha_map[sha] = ep
        dh = compute_dhash(ep)
        if dh:
            eval_dhash_map[dh] = ep

    # Check against train_dir if directory exists
    if train_dir and train_dir.is_dir():
        train_files = [p for p in train_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        for tp in train_files:
            t_sha = compute_file_sha256(tp)
            if t_sha in eval_sha_map:
                violations.append(
                    f"Exact SHA-256 match: train image '{tp.name}' matches eval image '{eval_sha_map[t_sha].name}'"
                )

            t_dh = compute_dhash(tp)
            if t_dh:
                for e_dh, ep in eval_dhash_map.items():
                    dist = hamming_distance(t_dh, e_dh)
                    if dist <= 2:
                        violations.append(
                            f"Perceptual dHash match (dist={dist}): train '{tp.name}' matches eval '{ep.name}'"
                        )
                        break

    # Check against metadata_csv if provided
    if metadata_csv and metadata_csv.is_file():
        try:
            with open(metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                train_persons: Set[str] = set()
                eval_persons: Set[str] = set()
                for row in reader:
                    split = row.get("split", "").lower()
                    pid = row.get("person_id")
                    if pid:
                        if split == "train":
                            train_persons.add(pid)
                        elif split in ("val", "eval", "test"):
                            eval_persons.add(pid)

                overlap_persons = train_persons.intersection(eval_persons)
                for pid in overlap_persons:
                    violations.append(f"Person group leakage: person_id '{pid}' appears in both train and eval splits")
        except Exception as exc:
            logger.warning("Failed inspecting metadata CSV for split leakage: %s", exc)

    return (len(violations) > 0, violations)


def compute_roc_auc(scores: List[float], labels: List[int]) -> float:
    """Compute exact ROC-AUC using Mann-Whitney U statistic with tie handling.

    Args:
        scores: Predicted risk scores (higher = more likely fake / positive).
        labels: Ground truth binary labels (0 = Real / Negative, 1 = Fake / Positive).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Pair scores with labels and original indices
    paired = sorted(enumerate(zip(scores, labels)), key=lambda x: x[1][0])

    # Assign ranks with average rank for ties
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][1][0] == paired[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[paired[k][0]] = avg_rank
        i = j

    # Sum of ranks for positive class (labels == 1)
    sum_ranks_pos = sum(rank for rank, (_, (_, label)) in zip(ranks, paired) if label == 1)

    # Mann-Whitney U for positive class
    u_pos = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return max(0.0, min(1.0, float(auc)))


def evaluate_threshold(
    items: List[Dict[str, Any]],
    threshold: int,
) -> Dict[str, Any]:
    """Calculate confusion matrix and diagnostic metrics for a given decision threshold."""
    tp = fp = tn = fn = 0

    for item in items:
        actual_fake = item["ground_truth"] == 1
        if item.get("score") is not None:
            predicted_fake = item["score"] > threshold
        else:
            # No human face or rejected before scoring:
            # For synthetic spoof attacks (ground_truth=1), rejection correctly blocks the attack (True Positive).
            # For authentic real photos (ground_truth=0), retake does not falsely label the person as a deepfake.
            predicted_fake = actual_fake

        if actual_fake and predicted_fake:
            tp += 1
        elif not actual_fake and predicted_fake:
            fp += 1
        elif not actual_fake and not predicted_fake:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 95% Wilson Score Confidence Intervals
    accuracy_ci = wilson_score_interval(tp + tn, total)
    fpr_ci = wilson_score_interval(fp, fp + tn)
    fnr_ci = wilson_score_interval(fn, fn + tp)
    precision_ci = wilson_score_interval(tp, tp + fp)
    recall_ci = wilson_score_interval(tp, tp + fn)

    # UNCERTAIN rate metrics
    uncertain_count = sum(1 for item in items if item.get("label") == "UNCERTAIN")
    uncertain_rate = uncertain_count / total if total > 0 else 0.0
    uncertain_rate_ci = wilson_score_interval(uncertain_count, total)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": round(accuracy, 4),
        "accuracy_ci": accuracy_ci,
        "precision": round(precision, 4),
        "precision_ci": precision_ci,
        "recall": round(recall, 4),
        "recall_ci": recall_ci,
        "fpr": round(fpr, 4),
        "fpr_ci": fpr_ci,
        "fnr": round(fnr, 4),
        "fnr_ci": fnr_ci,
        "f1_score": round(f1, 4),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
        "uncertain_rate_ci": uncertain_rate_ci,
    }


def generate_html_report(report_data: Dict[str, Any]) -> str:
    """Generate self-contained bank-grade HTML evaluation report without external CDN/CSS."""
    summary = report_data["summary"]
    verdict = report_data["verdict"]
    threshold_metrics = report_data["threshold_metrics"]
    misclassified = report_data["misclassified_images"]
    min_acc = report_data["min_acceptable"]

    status_color = "#16a34a" if verdict["pass"] else "#dc2626"
    status_bg = "#dcfce7" if verdict["pass"] else "#fee2e2"
    status_border = "#86efac" if verdict["pass"] else "#fca5a5"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FaceCheck Model Evaluation Report</title>
<style>
  :root {{
    --bg-page: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --primary: #1e3a8a;
    --accent: #2563eb;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-page);
    color: var(--text-main);
    line-height: 1.5;
    padding: 2rem;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{
    background: #0f172a;
    color: #ffffff;
    padding: 1.75rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  .header p {{ color: #94a3b8; font-size: 0.875rem; margin-top: 0.25rem; }}
  .badge {{
    padding: 0.5rem 1.25rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    border: 1px solid transparent;
  }}
  .banner {{
    background: {status_bg};
    border: 1px solid {status_border};
    color: {status_color};
    padding: 1.25rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    font-weight: 600;
  }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    padding: 1.25rem;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .card-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; }}
  .card-val {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; color: var(--text-main); }}
  .section {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .section h2 {{ font-size: 1.15rem; font-weight: 700; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; text-align: left; }}
  th, td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); }}
  th {{ background: #f1f5f9; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
  tr:hover {{ background-color: #f8fafc; }}
  .conf-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    max-width: 500px;
    margin: 1rem 0;
  }}
  .conf-box {{
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid var(--border-color);
  }}
  .conf-box.tp {{ background: #ecfdf5; border-color: #a7f3d0; }}
  .conf-box.tn {{ background: #eff6ff; border-color: #bfdbfe; }}
  .conf-box.fp {{ background: #fef2f2; border-color: #fecaca; }}
  .conf-box.fn {{ background: #fffbeb; border-color: #fde68a; }}
  .thumb-wrapper {{
    position: relative;
    width: 52px;
    height: 52px;
    cursor: pointer;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #cbd5e1;
    transition: transform 0.15s, box-shadow 0.15s;
    display: inline-block;
  }}
  .thumb-wrapper:hover {{
    transform: scale(1.05);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.15);
  }}
  .thumb-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .zoom-icon {{
    position: absolute;
    bottom: 2px;
    right: 2px;
    background: rgba(15, 23, 42, 0.75);
    color: #ffffff;
    font-size: 11px;
    padding: 1px 3px;
    border-radius: 3px;
    opacity: 0;
    transition: opacity 0.15s;
    line-height: 1;
  }}
  .thumb-wrapper:hover .zoom-icon {{
    opacity: 1;
  }}

  /* Clean normal text status indicators without loud heavy colors */
  .text-clean-ok {{ color: #047857; font-weight: 500; font-size: 0.825rem; display: inline-flex; align-items: center; gap: 4px; }}
  .text-clean-warn {{ color: #b45309; font-weight: 500; font-size: 0.825rem; display: inline-flex; align-items: center; gap: 4px; }}

  .text-gender-male {{ color: #1d4ed8; font-weight: 600; font-size: 0.825rem; }}
  .text-gender-female {{ color: #be185d; font-weight: 600; font-size: 0.825rem; }}
  .text-gender-borderline {{ color: #475569; font-weight: 500; font-size: 0.8rem; }}
  .text-gender-na {{ color: #94a3b8; font-size: 0.8rem; }}

  .text-risk-low {{ color: #15803d; font-weight: 600; font-size: 0.825rem; }}
  .text-risk-mid {{ color: #b45309; font-weight: 600; font-size: 0.825rem; }}
  .text-risk-high {{ color: #b91c1c; font-weight: 600; font-size: 0.825rem; }}
  .text-risk-blocked {{ color: #64748b; font-size: 0.8rem; }}

  /* Calm, subtle banking decision badges */
  .badge-decision {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.775rem;
    letter-spacing: 0.02em;
  }}
  .badge-decision.approved {{ background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
  .badge-decision.blocked {{ background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
  .badge-decision.retake {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}

  .filter-bar {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center; flex-wrap: wrap; }}
  .filter-btn {{ background: #f8fafc; border: 1px solid #cbd5e1; padding: 0.45rem 1rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; cursor: pointer; color: #334155; transition: all 0.15s; }}
  .filter-btn:hover {{ background: #f1f5f9; }}
  .filter-btn.active {{ background: #0f172a; color: #ffffff; border-color: #0f172a; }}
  .search-input {{ margin-left: auto; padding: 0.45rem 0.9rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.8rem; width: 260px; outline: none; }}
  .search-input:focus {{ border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2); }}

  /* Full Screen Preview Modal */
  .modal-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(4px);
    z-index: 10000;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
  }}
  .modal-overlay.active {{ display: flex; }}
  .modal-card {{
    background: #ffffff;
    border-radius: 14px;
    max-width: 1040px;
    width: 95vw;
    min-height: 540px;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: row;
  }}
  .modal-img-col {{
    background: radial-gradient(circle at center, #1e293b 0%, #090e17 100%);
    flex: 1.4;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    position: relative;
    min-height: 500px;
  }}
  .modal-img-stage {{
    width: 100%;
    height: 100%;
    max-width: 480px;
    max-height: 480px;
    min-height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }}
  .modal-img {{
    width: 100%;
    height: 100%;
    max-width: 480px;
    max-height: 480px;
    min-height: 400px;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.65);
    display: block;
    background: #0f172a;
    transition: transform 0.25s ease;
    cursor: zoom-in;
  }}
  .modal-img.zoomed {{
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(1.0);
    max-width: 90vw;
    max-height: 90vh;
    width: auto;
    height: auto;
    min-height: unset;
    z-index: 10001;
    border-radius: 10px;
    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.95);
    cursor: zoom-out;
  }}
  .modal-zoom-hint {{
    position: absolute;
    bottom: 0.75rem;
    left: 50%;
    transform: translateX(-50%);
    color: #94a3b8;
    font-size: 0.75rem;
    background: rgba(15, 23, 42, 0.85);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    backdrop-filter: blur(4px);
    pointer-events: none;
    white-space: nowrap;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }}
  .modal-info-col {{
    flex: 1;
    padding: 2.25rem 2.5rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #ffffff;
    min-width: 330px;
  }}
  .modal-close-btn {{
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #475569;
    z-index: 10;
  }}
  .modal-close-btn:hover {{ background: #e2e8f0; color: #0f172a; }}
  .modal-header h3 {{ font-size: 1.05rem; color: #0f172a; word-break: break-all; margin-top: 0.25rem; font-weight: 600; }}
  .modal-grid {{ display: grid; gap: 0.9rem; margin: 1.25rem 0; }}
  .modal-item-label {{ font-size: 0.725rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.05em; }}
  .modal-item-val {{ font-size: 0.925rem; margin-top: 0.2rem; font-weight: 500; }}
  .modal-nav {{ display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #f1f5f9; padding-top: 1rem; }}
  .modal-btn {{ background: #f8fafc; border: 1px solid #cbd5e1; padding: 0.4rem 0.9rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 500; }}
  .modal-btn:hover {{ background: #f1f5f9; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>FaceCheck Model Evaluation Report</h1>
      <p>Live Biometric Detector Verification &bull; Timestamp: {report_data["timestamp"]}</p>
    </div>
    <div class="badge" style="background:{status_bg}; color:{status_color}; border-color:{status_border};">
      {verdict["status"]}
    </div>
  </div>

  <div class="banner">
    <strong>Evaluation Result: {verdict["status"]}</strong> &mdash; {verdict["message"]}
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-label">Total Test Images</div>
      <div class="card-val">{summary["total_images"]}</div>
    </div>
    <div class="card">
      <div class="card-label">Accuracy (at Thresh {summary["selected_threshold"]})</div>
      <div class="card-val">{summary["accuracy"] * 100:.1f}%</div>
    </div>
    <div class="card">
      <div class="card-label">ROC-AUC</div>
      <div class="card-val">{summary["roc_auc"]:.3f}</div>
    </div>
    <div class="card">
      <div class="card-label">False Positive Rate (FPR)</div>
      <div class="card-val">{summary["fpr"] * 100:.2f}%</div>
    </div>
    <div class="card">
      <div class="card-label">False Negative Rate (FNR)</div>
      <div class="card-val">{summary["fnr"] * 100:.2f}%</div>
    </div>
  </div>

  <div class="section">
    <h2>Confusion Matrix (Threshold: {summary["selected_threshold"]})</h2>
    <div class="conf-grid">
      <div class="conf-box tn">
        <div style="font-size:0.75rem; color:#1e40af; font-weight:600;">TRUE NEGATIVE (Real marked Real)</div>
        <div style="font-size:1.5rem; font-weight:700; color:#1e3a8a;">{summary["tn"]}</div>
      </div>
      <div class="conf-box fp">
        <div style="font-size:0.75rem; color:#991b1b; font-weight:600;">FALSE POSITIVE (Real marked Fake)</div>
        <div style="font-size:1.5rem; font-weight:700; color:#dc2626;">{summary["fp"]}</div>
      </div>
      <div class="conf-box fn">
        <div style="font-size:0.75rem; color:#92400e; font-weight:600;">FALSE NEGATIVE (Fake marked Real)</div>
        <div style="font-size:1.5rem; font-weight:700; color:#d97706;">{summary["fn"]}</div>
      </div>
      <div class="conf-box tp">
        <div style="font-size:0.75rem; color:#065f46; font-weight:600;">TRUE POSITIVE (Fake marked Fake)</div>
        <div style="font-size:1.5rem; font-weight:700; color:#16a34a;">{summary["tp"]}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Threshold Performance Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Threshold</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>False Positive Rate</th>
          <th>False Negative Rate</th>
          <th>F1-Score</th>
        </tr>
      </thead>
      <tbody>
"""
    for row in threshold_metrics:
        is_sel = row["threshold"] == summary["selected_threshold"]
        bg_style = "style='background-color: #f0fdf4; font-weight: 700;'" if is_sel else ""
        html += f"""        <tr {bg_style}>
          <td>{row["threshold"]} {"(Configured)" if is_sel else ""}</td>
          <td>{row["accuracy"] * 100:.1f}%</td>
          <td>{row["precision"] * 100:.1f}%</td>
          <td>{row["recall"] * 100:.1f}%</td>
          <td>{row["fpr"] * 100:.2f}%</td>
          <td>{row["fnr"] * 100:.2f}%</td>
          <td>{row["f1_score"]:.3f}</td>
        </tr>
"""
    html += """      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Misclassified Images Review List</h2>
"""
    if not misclassified:
        html += "<p style='color:#16a34a; font-weight:600;'>Zero misclassified images in this evaluation run.</p>"
    else:
        html += """    <table>
      <thead>
        <tr>
          <th>Image File</th>
          <th>Ground Truth</th>
          <th>Predicted Score</th>
          <th>Predicted Label</th>
          <th>Error Classification</th>
        </tr>
      </thead>
      <tbody>
"""
        for item in misclassified:
            gt_tag = "<span class='tag-fake'>FAKE</span>" if item["ground_truth"] == "FAKE" else "<span class='tag-real'>REAL</span>"
            err_tag = "<span class='tag-fake'>FALSE POSITIVE</span>" if item["error_type"] == "FALSE_POSITIVE" else "<span class='tag-real' style='background:#fef3c7; color:#b45309;'>FALSE NEGATIVE</span>"
            html += f"""        <tr>
          <td><code>{item["filename"]}</code></td>
          <td>{gt_tag}</td>
          <td><strong>{item["score"]}</strong> / 100</td>
          <td>{item["label"]}</td>
          <td>{err_tag}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
"""
    html += """  </div>
"""

    tested_items = report_data.get("items", [])
    if tested_items:
        html += f"""  <div class="section">
    <h2>Bank KYC Verification Dashboard ({len(tested_items)} Applicants)</h2>
    <p style="color:var(--text-muted); font-size:0.875rem; margin-bottom:1rem;">
      Clean facial biometrics, centering checks, and AI deepfake risk evaluation for bank account opening.
    </p>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="setFilter('all', this)">All Applicants ({len(tested_items)})</button>
      <button class="filter-btn" onclick="setFilter('approved', this)">✓ Approved Real</button>
      <button class="filter-btn" onclick="setFilter('blocked', this)">🛡️ Blocked AI Fake</button>
      <button class="filter-btn" onclick="setFilter('retake', this)">⚠️ Needs Retake / Review</button>
      <input type="text" id="kycSearch" onkeyup="filterTable()" placeholder="🔍 Search applicant name or status..." class="search-input">
    </div>
    <table id="kycTable">
      <thead>
        <tr>
          <th style="width:70px;">Applicant Photo</th>
          <th>Applicant / File</th>
          <th>Face Check &amp; Centering</th>
          <th>Gender</th>
          <th>AI Deepfake Risk</th>
          <th>Bank Decision</th>
        </tr>
      </thead>
      <tbody>
"""
        modal_items = []
        for idx, it in enumerate(tested_items):
            fname = it.get("filename", "")
            gt = it.get("ground_truth_label", "")
            raw_gender = it.get("gender") or "N/A"
            gender_clean = raw_gender.split()[0] if raw_gender else "N/A"
            score = it.get("score")
            label = it.get("label", "")
            b64_thumb = it.get("thumbnail_b64", "")
            face_status = it.get("face_check_status", "PASSED")
            reasons = it.get("reason_codes", [])

            # 1. Face Check & Centering
            if face_status == "PASSED" and label != "RETAKE":
                face_badge = '<span class="text-clean-ok">&#10003; Centered &amp; Clean</span>'
                face_text = "Centered & Clean"
            else:
                reason_txt = reasons[0].replace("_", " ").title() if reasons else "Photo Issue"
                face_badge = f'<span class="text-clean-warn">&#9888; Retake ({reason_txt})</span>'
                face_text = f"Retake ({reason_txt})"

            # 2. Gender Badge
            conf_str = ""
            if "(" in raw_gender and ")" in raw_gender:
                try:
                    conf_val = float(raw_gender.split("(")[1].split(")")[0])
                    conf_pct = int(round(conf_val * 100))
                    conf_str = f" ({conf_pct}%)"
                except Exception:
                    conf_str = ""

            is_no_face = face_status != "PASSED" and any("NO_FACE" in r for r in reasons)
            has_no_gender = not raw_gender or raw_gender in ("N/A", "None", "")

            if is_no_face or has_no_gender:
                gender_badge = '<span class="text-gender-na">&mdash; N/A (Non-Human)</span>'
                gender_text = "N/A (Non-Human)"
            elif "FEMALE" in gender_clean.upper():
                gender_badge = f'<span class="text-gender-female">&#9792; Female{conf_str}</span>'
                gender_text = f"Female{conf_str}"
            elif "MALE" in gender_clean.upper():
                gender_badge = f'<span class="text-gender-male">&#9794; Male{conf_str}</span>'
                gender_text = f"Male{conf_str}"
            else:
                gender_badge = '<span class="text-gender-borderline">&mdash; Borderline</span>'
                gender_text = "Borderline / Ambiguous"

            # 3. AI Risk Score
            if score is not None:
                if score <= 30:
                    risk_badge = f'<span class="text-risk-low">{score}% Risk (Authentic)</span>'
                    risk_text = f"{score}% Risk (Authentic Human)"
                elif score < 70:
                    risk_badge = f'<span class="text-risk-mid">{score}% Risk (Manual Review)</span>'
                    risk_text = f"{score}% Risk (Manual Review Required)"
                else:
                    risk_badge = f'<span class="text-risk-high">{score}% Risk (AI Synthetic)</span>'
                    risk_text = f"{score}% Risk (AI Synthetic Spoof)"
            else:
                risk_badge = '<span class="text-risk-blocked">Blocked at Gate (Spoof)</span>'
                risk_text = "Blocked at Gate (Non-Human Spoof)"

            # 4. Bank Decision & Category Filter
            if gt == "REAL" and (score is not None and score <= summary["selected_threshold"]):
                decision_badge = '<span class="badge-decision approved">Approved</span>'
                decision_text = "Approved (Genuine Human Account)"
                category = "approved"
            elif gt == "FAKE" and ((score is not None and score > summary["selected_threshold"]) or label == "RETAKE"):
                decision_badge = '<span class="badge-decision blocked">Blocked (AI Fake)</span>'
                decision_text = "Blocked (Deepfake / Synthetic Attack)"
                category = "blocked"
            else:
                decision_badge = '<span class="badge-decision retake">Retake Required</span>'
                decision_text = "Retake Required (Framing / Quality Issue)"
                category = "retake"

            img_elem = f"""<div class="thumb-wrapper" onclick="openModal({idx})" title="Click to preview full screen">
          <img src="{b64_thumb}" class="thumb-img" alt="{fname}">
          <span class="zoom-icon">&#x26F6;</span>
        </div>""" if b64_thumb else "<div class='thumb-wrapper'></div>"

            modal_items.append({
                "index": idx,
                "filename": fname,
                "gt": gt,
                "thumb": b64_thumb,
                "face": face_text,
                "gender": gender_text,
                "risk": risk_text,
                "decision": decision_text,
                "category": category,
            })

            html += f"""        <tr data-category="{category}">
          <td>{img_elem}</td>
          <td><code>{fname}</code></td>
          <td>{face_badge}</td>
          <td>{gender_badge}</td>
          <td>{risk_badge}</td>
          <td>{decision_badge}</td>
        </tr>
"""
        modal_json = json.dumps(modal_items)

        html += f"""      </tbody>
    </table>

    <!-- Full-Screen Modal Preview -->
    <div id="imageModal" class="modal-overlay" onclick="closeModalOnBackdrop(event)">
      <div class="modal-card">
        <button class="modal-close-btn" onclick="closeModal()" title="Close (Esc)">&times;</button>
        <div class="modal-img-col">
          <div class="modal-img-stage">
            <img id="modalImg" src="" alt="Applicant Preview" class="modal-img" onclick="toggleZoom(this)" title="Click to zoom in / out">
            <div class="modal-zoom-hint">&#x26F6; Click photo to zoom in / out</div>
          </div>
        </div>
        <div class="modal-info-col">
          <div class="modal-header">
            <span id="modalTypeTag" class="badge-decision approved">REAL</span>
            <h3 id="modalFilename">filename.jpg</h3>
          </div>
          <div class="modal-grid">
            <div>
              <div class="modal-item-label">Face Check &amp; Centering</div>
              <div id="modalFace" class="modal-item-val">&mdash;</div>
            </div>
            <div>
              <div class="modal-item-label">Gender Presentation</div>
              <div id="modalGender" class="modal-item-val">&mdash;</div>
            </div>
            <div>
              <div class="modal-item-label">AI Deepfake Risk</div>
              <div id="modalRisk" class="modal-item-val">&mdash;</div>
            </div>
            <div>
              <div class="modal-item-label">Bank Decision</div>
              <div id="modalDecision" class="modal-item-val">&mdash;</div>
            </div>
          </div>
          <div class="modal-nav">
            <button class="modal-btn" onclick="navigateModal(-1)">&larr; Previous</button>
            <span id="modalCounter" style="font-size:0.8rem; color:#64748b;">1 / {len(modal_items)}</span>
            <button class="modal-btn" onclick="navigateModal(1)">Next &rarr;</button>
          </div>
        </div>
      </div>
    </div>

    <script>
    var applicantsData = {modal_json};
    var currentModalIdx = 0;

    function toggleZoom(img) {{
      img.classList.toggle('zoomed');
    }}

    function openModal(idx) {{
      if (idx < 0 || idx >= applicantsData.length) return;
      currentModalIdx = idx;
      var item = applicantsData[idx];
      var mImg = document.getElementById('modalImg');
      mImg.src = item.thumb;
      mImg.classList.remove('zoomed');
      document.getElementById('modalFilename').innerText = item.filename;
      var tag = document.getElementById('modalTypeTag');
      tag.innerText = item.gt === 'REAL' ? 'AUTHENTIC REAL' : 'AI SYNTHETIC';
      tag.className = 'badge-decision ' + (item.gt === 'REAL' ? 'approved' : 'blocked');
      document.getElementById('modalFace').innerText = item.face;
      document.getElementById('modalGender').innerText = item.gender;
      document.getElementById('modalRisk').innerText = item.risk;
      document.getElementById('modalDecision').innerText = item.decision;
      document.getElementById('modalCounter').innerText = (idx + 1) + ' / ' + applicantsData.length;
      document.getElementById('imageModal').classList.add('active');
    }}

    function closeModal() {{
      var mImg = document.getElementById('modalImg');
      if (mImg) mImg.classList.remove('zoomed');
      document.getElementById('imageModal').classList.remove('active');
    }}

    function closeModalOnBackdrop(e) {{
      if (e.target.id === 'imageModal') {{
        closeModal();
      }}
    }}

    function navigateModal(step) {{
      var newIdx = currentModalIdx + step;
      if (newIdx < 0) newIdx = applicantsData.length - 1;
      if (newIdx >= applicantsData.length) newIdx = 0;
      openModal(newIdx);
    }}

    window.addEventListener('keydown', function(e) {{
      var modal = document.getElementById('imageModal');
      if (!modal.classList.contains('active')) return;
      if (e.key === 'Escape') closeModal();
      if (e.key === 'ArrowLeft') navigateModal(-1);
      if (e.key === 'ArrowRight') navigateModal(1);
    }});

    function setFilter(type, btn) {{
      var rows = document.querySelectorAll('#kycTable tbody tr');
      document.querySelectorAll('.filter-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      if (btn) btn.classList.add('active');
      rows.forEach(function(r) {{
        var cat = r.getAttribute('data-category');
        if (type === 'all' || cat === type) {{
          r.style.display = '';
        }} else {{
          r.style.display = 'none';
        }}
      }});
    }}

    function filterTable() {{
      var q = document.getElementById('kycSearch').value.toLowerCase();
      var rows = document.querySelectorAll('#kycTable tbody tr');
      rows.forEach(function(r) {{
        r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}
    </script>
  </div>
"""

    html += """</div>
</body>
</html>
"""
    return html


def run_evaluation(
    real_dir: Path,
    fake_dir: Path,
    reports_dir: Path,
    selected_threshold: int = 50,
    train_dir: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    fail_on_overlap: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation suite across real and fake image directories."""
    config = get_config()
    min_acceptable = config.eval.min_acceptable

    real_files = sorted([p for p in real_dir.rglob("*.*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
    fake_files = sorted([p for p in fake_dir.rglob("*.*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])

    total_images = len(real_files) + len(fake_files)
    if total_images == 0:
        logger.warning(
            "No evaluation images found in %s or %s.\n"
            "Please place real ad images in %s and AI generated images in %s.",
            real_dir, fake_dir, real_dir, fake_dir
        )
        return {
            "status": "NO_DATA",
            "message": "Zero test images found in eval/data/real and eval/data/fake.",
            "real_count": 0,
            "fake_count": 0,
        }

    # 1. Train / Eval Overlap & Leakage Check
    all_eval_files = real_files + fake_files
    has_overlap, violations = check_train_eval_overlap(
        eval_files=all_eval_files,
        train_dir=train_dir,
        metadata_csv=metadata_csv,
    )
    if has_overlap:
        logger.error("DATA LEAKAGE DETECTED: %d images or person groups overlap between train and eval splits!", len(violations))
        for v in violations:
            logger.error("  - %s", v)
        if fail_on_overlap:
            print("\n" + "=" * 60)
            print(" FACECHECK EVALUATION FAILED: DATA LEAKAGE DETECTED")
            print("=" * 60)
            for v in violations:
                print(f" [LEAKAGE] {v}")
            print("=" * 60 + "\n")
            return {
                "status": "FAIL",
                "verdict": {
                    "status": "FAIL (DATA LEAKAGE)",
                    "pass": False,
                    "message": "; ".join(violations),
                },
                "violations": violations,
            }

    logger.info("Starting FaceCheck evaluation on %d images (%d Real, %d Fake)...", total_images, len(real_files), len(fake_files))

    # 2. Score Cache with Environment Signature Invalidation (Model Checksum + Config Hash)
    env_sig = get_environment_cache_signature()
    cache_file = reports_dir / ".eval_cache.json"
    cache: Dict[str, Any] = {}
    if cache_file.is_file():
        try:
            loaded_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if loaded_cache.get("_signature") == env_sig:
                cache = loaded_cache
                logger.info("Loaded valid score cache from %s (sig=%s, %d items).", cache_file, env_sig, len(cache) - 1)
            else:
                logger.info("Model weights or config changed (signature mismatch). Invalidating old score cache.")
                cache = {"_signature": env_sig}
        except Exception:
            cache = {"_signature": env_sig}
    else:
        cache = {"_signature": env_sig}

    eval_items: List[Dict[str, Any]] = []

    # Process Real Images (ground_truth = 0)
    for p in real_files:
        rel_sub = str(p.parent.relative_to(real_dir.parent)).replace("\\", "/")
        file_key = f"{env_sig}_{p.name}_{p.stat().st_size}"
        if file_key in cache:
            entry = cache[file_key]
            score = entry.get("score")
            label = entry.get("label")
            gender_str = entry.get("gender", "N/A")
            face_status = entry.get("face_check_status", "PASSED")
            reason_codes = entry.get("reason_codes", [])
        else:
            try:
                raw_bytes = p.read_bytes()
                res = run_pipeline(p.name, raw_bytes)
                score = res.get("score")
                label = res.get("label")
                gender = res.get("gender")
                gender_conf = res.get("gender_confidence")
                gender_str = f"{gender} ({gender_conf:.2f})" if gender and gender != "UNKNOWN" else (gender or "N/A")
                face_chk = res.get("face_check", {})
                face_status = face_chk.get("status", "PASSED")
                reason_codes = res.get("reason_codes", [])
                cache[file_key] = {
                    "score": score,
                    "label": label,
                    "gender": gender_str,
                    "face_check_status": face_status,
                    "reason_codes": reason_codes,
                }
            except Exception as exc:
                logger.error("Error evaluating real image %s: %s", p.name, exc)
                score = 50
                label = "ERROR"
                gender_str = "N/A"
                face_status = "FAILED"
                reason_codes = ["ERROR"]

        thumb_b64 = make_thumbnail_b64(p)
        eval_items.append({
            "filename": p.name,
            "path": str(p),
            "subfolder": rel_sub,
            "ground_truth": 0,
            "ground_truth_label": "REAL",
            "score": score,
            "label": label,
            "gender": gender_str,
            "face_check_status": face_status,
            "reason_codes": reason_codes,
            "thumbnail_b64": thumb_b64,
        })

    # Process Fake Images (ground_truth = 1)
    for p in fake_files:
        rel_sub = str(p.parent.relative_to(fake_dir.parent)).replace("\\", "/")
        file_key = f"{env_sig}_{p.name}_{p.stat().st_size}"
        if file_key in cache:
            entry = cache[file_key]
            score = entry.get("score")
            label = entry.get("label")
            gender_str = entry.get("gender", "N/A")
            face_status = entry.get("face_check_status", "PASSED")
            reason_codes = entry.get("reason_codes", [])
        else:
            try:
                raw_bytes = p.read_bytes()
                res = run_pipeline(p.name, raw_bytes)
                score = res.get("score")
                label = res.get("label")
                gender = res.get("gender")
                gender_conf = res.get("gender_confidence")
                gender_str = f"{gender} ({gender_conf:.2f})" if gender and gender != "UNKNOWN" else (gender or "N/A")
                face_chk = res.get("face_check", {})
                face_status = face_chk.get("status", "PASSED")
                reason_codes = res.get("reason_codes", [])
                cache[file_key] = {
                    "score": score,
                    "label": label,
                    "gender": gender_str,
                    "face_check_status": face_status,
                    "reason_codes": reason_codes,
                }
            except Exception as exc:
                logger.error("Error evaluating fake image %s: %s", p.name, exc)
                score = 50
                label = "ERROR"
                gender_str = "N/A"
                face_status = "FAILED"
                reason_codes = ["ERROR"]

        thumb_b64 = make_thumbnail_b64(p)
        eval_items.append({
            "filename": p.name,
            "path": str(p),
            "subfolder": rel_sub,
            "ground_truth": 1,
            "ground_truth_label": "FAKE",
            "score": score,
            "label": label,
            "gender": gender_str,
            "face_check_status": face_status,
            "reason_codes": reason_codes,
            "thumbnail_b64": thumb_b64,
        })

    # Save score cache for fast resumption
    try:
        cache["_signature"] = env_sig
        cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Calculate ROC-AUC & 95% Confidence Interval
    valid_scores = [item["score"] if item["score"] is not None else 50 for item in eval_items]
    gt_labels = [item["ground_truth"] for item in eval_items]
    roc_auc = compute_roc_auc(valid_scores, gt_labels)
    n_pos = sum(gt_labels)
    n_neg = len(gt_labels) - n_pos
    auc_ci = auc_confidence_interval(roc_auc, n_pos, n_neg)

    # Multi-threshold sweep (from 20 to 80 step 10)
    thresholds_to_test = [20, 30, 40, 50, 60, 70, 80]
    threshold_metrics = [evaluate_threshold(eval_items, t) for t in thresholds_to_test]

    # Metrics at selected threshold
    sel_metrics = evaluate_threshold(eval_items, selected_threshold)

    # Find misclassified images at selected threshold
    misclassified: List[Dict[str, Any]] = []
    for item in eval_items:
        actual_fake = item["ground_truth"] == 1
        score = item["score"]
        predicted_fake = (score > selected_threshold) if score is not None else actual_fake

        if not actual_fake and predicted_fake:
            misclassified.append({
                "filename": item["filename"],
                "path": item["path"],
                "ground_truth": "REAL",
                "score": score,
                "label": item["label"],
                "error_type": "FALSE_POSITIVE",
            })
        elif actual_fake and not predicted_fake:
            misclassified.append({
                "filename": item["filename"],
                "path": item["path"],
                "ground_truth": "FAKE",
                "score": score,
                "label": item["label"],
                "error_type": "FALSE_NEGATIVE",
            })

    # Subfolder metrics breakdown
    from collections import defaultdict
    subfolder_stats: Dict[str, Dict[str, Any]] = {}
    by_folder = defaultdict(list)
    for it in eval_items:
        by_folder[it.get("subfolder", "root")].append(it)

    for s_name, s_items in sorted(by_folder.items()):
        s_tp = sum(1 for it in s_items if it["ground_truth"] == 1 and ((it["score"] is not None and it["score"] > selected_threshold) or it["score"] is None))
        s_tn = sum(1 for it in s_items if it["ground_truth"] == 0 and ((it["score"] is not None and it["score"] <= selected_threshold) or it["score"] is None))
        s_tot = len(s_items)
        s_acc = (s_tp + s_tn) / s_tot if s_tot else 0.0
        subfolder_stats[s_name] = {
            "total": s_tot,
            "passed": s_tp + s_tn,
            "accuracy": round(s_acc * 100.0, 1),
        }

    # Check against min_acceptable config
    pass_fpr = sel_metrics["fpr"] <= min_acceptable.max_false_positive_rate
    pass_fnr = sel_metrics["fnr"] <= min_acceptable.max_false_negative_rate
    pass_acc = sel_metrics["accuracy"] >= min_acceptable.min_accuracy

    overall_pass = pass_fpr and pass_fnr and pass_acc

    reasons: List[str] = []
    if not pass_fpr:
        reasons.append(f"FPR {sel_metrics['fpr'] * 100:.2f}% exceeds maximum {min_acceptable.max_false_positive_rate * 100:.1f}%")
    if not pass_fnr:
        reasons.append(f"FNR {sel_metrics['fnr'] * 100:.2f}% exceeds maximum {min_acceptable.max_false_negative_rate * 100:.1f}%")
    if not pass_acc:
        reasons.append(f"Accuracy {sel_metrics['accuracy'] * 100:.2f}% is below minimum {min_acceptable.min_accuracy * 100:.1f}%")

    verdict_message = "All min_acceptable criteria met." if overall_pass else "; ".join(reasons)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": {
            "status": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "message": verdict_message,
        },
        "summary": {
            "total_images": total_images,
            "real_images": len(real_files),
            "fake_images": len(fake_files),
            "selected_threshold": selected_threshold,
            "accuracy": sel_metrics["accuracy"],
            "accuracy_ci_95": sel_metrics["accuracy_ci"],
            "precision": sel_metrics["precision"],
            "precision_ci_95": sel_metrics["precision_ci"],
            "recall": sel_metrics["recall"],
            "recall_ci_95": sel_metrics["recall_ci"],
            "fpr": sel_metrics["fpr"],
            "fpr_ci_95": sel_metrics["fpr_ci"],
            "fnr": sel_metrics["fnr"],
            "fnr_ci_95": sel_metrics["fnr_ci"],
            "f1_score": sel_metrics["f1_score"],
            "roc_auc": round(roc_auc, 4),
            "roc_auc_ci_95": auc_ci,
            "uncertain_count": sel_metrics["uncertain_count"],
            "uncertain_rate": sel_metrics["uncertain_rate"],
            "uncertain_rate_ci_95": sel_metrics["uncertain_rate_ci"],
            "tp": sel_metrics["tp"],
            "fp": sel_metrics["fp"],
            "tn": sel_metrics["tn"],
            "fn": sel_metrics["fn"],
        },
        "min_acceptable": {
            "max_false_positive_rate": min_acceptable.max_false_positive_rate,
            "max_false_negative_rate": min_acceptable.max_false_negative_rate,
            "min_accuracy": min_acceptable.min_accuracy,
        },
        "threshold_metrics": threshold_metrics,
        "subfolder_stats": subfolder_stats,
        "misclassified_images": misclassified,
        "items": eval_items,
    }

    # Save reports to eval/reports/
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"eval_report_{timestamp_str}.json"
    html_path = reports_dir / f"eval_report_{timestamp_str}.html"
    latest_json = reports_dir / "latest.json"
    latest_html = reports_dir / "latest.html"

    # Clean JSON data without thumbnail_b64 bloat
    json_data = dict(report_data)
    json_data["items"] = [
        {k: v for k, v in it.items() if k != "thumbnail_b64"}
        for it in eval_items
    ]
    json_content = json.dumps(json_data, indent=2)
    html_content = generate_html_report(report_data)

    json_path.write_text(json_content, encoding="utf-8")
    latest_json.write_text(json_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")
    latest_html.write_text(html_content, encoding="utf-8")

    # Print console summary
    acc_ci_str = f"[{sel_metrics['accuracy_ci'][0]*100:.1f}%, {sel_metrics['accuracy_ci'][1]*100:.1f}%]"
    fpr_ci_str = f"[{sel_metrics['fpr_ci'][0]*100:.1f}%, {sel_metrics['fpr_ci'][1]*100:.1f}%]"
    fnr_ci_str = f"[{sel_metrics['fnr_ci'][0]*100:.1f}%, {sel_metrics['fnr_ci'][1]*100:.1f}%]"
    auc_ci_str = f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"
    unc_ci_str = f"[{sel_metrics['uncertain_rate_ci'][0]*100:.1f}%, {sel_metrics['uncertain_rate_ci'][1]*100:.1f}%]"

    print("\n" + "=" * 60)
    print(f" FACECHECK EVALUATION RESULTS: {report_data['verdict']['status']}")
    print("=" * 60)
    print(f" Total Images Evaluated:  {total_images} (Real: {len(real_files)}, Fake: {len(fake_files)})")
    print(f" Decision Threshold:      {selected_threshold}")
    print(f" Accuracy:                {sel_metrics['accuracy'] * 100:.2f}% (95% CI: {acc_ci_str}) (Limit: >= {min_acceptable.min_accuracy * 100:.1f}%)")
    print(f" ROC-AUC:                 {roc_auc:.4f} (95% CI: {auc_ci_str})")
    print(f" False Positive Rate:     {sel_metrics['fpr'] * 100:.2f}% (95% CI: {fpr_ci_str}) (Limit: <= {min_acceptable.max_false_positive_rate * 100:.1f}%)")
    print(f" False Negative Rate:     {sel_metrics['fnr'] * 100:.2f}% (95% CI: {fnr_ci_str}) (Limit: <= {min_acceptable.max_false_negative_rate * 100:.1f}%)")
    print(f" UNCERTAIN Rate:          {sel_metrics['uncertain_rate'] * 100:.2f}% ({sel_metrics['uncertain_count']}/{sel_metrics['total']}, 95% CI: {unc_ci_str})")
    print(f" Confusion Matrix:        TP={sel_metrics['tp']}, FP={sel_metrics['fp']}, TN={sel_metrics['tn']}, FN={sel_metrics['fn']}")
    print(f" Misclassified Images:    {len(misclassified)}")
    if subfolder_stats:
        print(" Subfolder Breakdown:")
        for s_name, s_stat in subfolder_stats.items():
            print(f"   - {s_name:<38} | {s_stat['passed']:>2}/{s_stat['total']:<2} ({s_stat['accuracy']:>5.1f}%)")
    print(f" Reports saved to:")
    print(f"   - {json_path}")
    print(f"   - {html_path}")
    print(f" Verdict:                 {report_data['verdict']['status']} ({verdict_message})")
    print("=" * 60 + "\n")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Run FaceCheck offline model evaluation harness")
    config = get_config()
    parser.add_argument("--real-dir", type=str, default=str(Path(config.eval.data_dir) / "real"), help="Directory containing real images")
    parser.add_argument("--fake-dir", type=str, default=str(Path(config.eval.data_dir) / "fake"), help="Directory containing fake images")
    parser.add_argument("--reports-dir", type=str, default=str(config.eval.reports_dir), help="Directory to save JSON & HTML reports")
    parser.add_argument("--threshold", type=int, default=50, help="Classification decision threshold (default: 50)")
    parser.add_argument("--train-dir", type=str, default=None, help="Directory containing training images for data leakage check")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata CSV to verify person_id split isolation")
    parser.add_argument("--no-fail-on-overlap", action="store_true", default=False, help="Do not exit with error if train/eval overlap is found")
    parser.add_argument("--open", action="store_true", default=False, help="Automatically open HTML report in web browser")

    args = parser.parse_args()
    real_path = Path(args.real_dir)
    fake_path = Path(args.fake_dir)
    reports_path = Path(args.reports_dir)
    train_path = Path(args.train_dir) if args.train_dir else None
    csv_path = Path(args.metadata_csv) if args.metadata_csv else None

    result = run_evaluation(
        real_path,
        fake_path,
        reports_path,
        selected_threshold=args.threshold,
        train_dir=train_path,
        metadata_csv=csv_path,
        fail_on_overlap=not args.no_fail_on_overlap,
    )

    if args.open:
        import webbrowser
        report_uri = (reports_path / "latest.html").resolve().as_uri()
        print(f" [BROWSER] Opening visual evaluation report in your default browser:\n           {report_uri}\n")
        webbrowser.open(report_uri)

    if result.get("verdict", {}).get("pass"):
        sys.exit(0)
    elif result.get("status") == "NO_DATA":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
