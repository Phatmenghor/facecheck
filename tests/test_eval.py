"""Unit tests for Evaluation Test Harness and Metrics."""

from __future__ import annotations

import io
import json
import pytest
from pathlib import Path
from PIL import Image

from scripts.evaluate import (
    compute_roc_auc,
    evaluate_threshold,
    run_evaluation,
)


def test_compute_roc_auc_perfect():
    scores = [10.0, 20.0, 80.0, 90.0]
    labels = [0, 0, 1, 1]
    auc = compute_roc_auc(scores, labels)
    assert auc == 1.0


def test_compute_roc_auc_random():
    scores = [50.0, 50.0, 50.0, 50.0]
    labels = [0, 0, 1, 1]
    auc = compute_roc_auc(scores, labels)
    assert auc == 0.5


def test_evaluate_threshold_metrics():
    items = [
        {"ground_truth": 0, "score": 20},  # TN at 50
        {"ground_truth": 0, "score": 60},  # FP at 50 (real marked fake)
        {"ground_truth": 1, "score": 40},  # FN at 50 (fake marked real)
        {"ground_truth": 1, "score": 80},  # TP at 50
    ]
    res = evaluate_threshold(items, threshold=50)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 1
    assert res["fn"] == 1
    assert res["accuracy"] == 0.5
    assert res["fpr"] == 0.5
    assert res["fnr"] == 0.5


def test_full_evaluation_flow(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    reports_dir = tmp_path / "reports"

    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    # Generate 1 simple real test image and 1 fake test image
    img_real = Image.new("RGB", (150, 150), color=(120, 120, 120))
    img_real.save(real_dir / "test_real.jpg", format="JPEG")

    img_fake = Image.new("RGB", (150, 150), color=(200, 50, 50))
    img_fake.save(fake_dir / "test_fake.jpg", format="JPEG")

    result = run_evaluation(
        real_dir=real_dir,
        fake_dir=fake_dir,
        reports_dir=reports_dir,
        selected_threshold=50,
    )

    assert result["summary"]["total_images"] == 2
    assert "verdict" in result
    assert "threshold_metrics" in result

    # Check generated files
    assert (reports_dir / "latest.json").is_file()
    assert (reports_dir / "latest.html").is_file()

    # Verify HTML is self-contained without external scripts
    html_text = (reports_dir / "latest.html").read_text(encoding="utf-8")
    assert "<html" in html_text
    assert "FaceCheck Model Evaluation Report" in html_text
    assert "http://" not in html_text and "https://" not in html_text  # Zero external CDNs!
