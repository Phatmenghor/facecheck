"""Visual bank-grade HTML and JSON evaluation report generator with embedded image thumbnails.

Zero external CDN dependencies (zero external CSS/JS/images).
All thumbnails are encoded in base64 data URIs for 100% air-gapped viewing.
"""

from __future__ import annotations

import base64
import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def make_thumbnail_b64(file_path: Path, max_size=(800, 800)) -> str:
    """Generate high-clarity Base64 JPEG data URI for modal review & table thumbnails."""
    try:
        with Image.open(file_path) as im:
            im = im.convert("RGB")
            # Only downscale if original exceeds max_size; preserve full sharpness
            if im.size[0] > max_size[0] or im.size[1] > max_size[1]:
                im.thumbnail(max_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
            b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return ""


def generate_visual_html_report(
    tested_items: List[Dict[str, Any]],
    group_stats: Dict[str, Dict[str, Any]],
    overall_acc: float,
    duration_s: float,
    timestamp_str: Optional[str] = None,
) -> str:
    """Generate self-contained bank-grade HTML evaluation report with image previews."""
    if not timestamp_str:
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_tested = len(tested_items)
    total_passed = sum(1 for it in tested_items if it.get("passed"))
    real_count = sum(1 for it in tested_items if it.get("category") == "REAL")
    fake_count = sum(1 for it in tested_items if it.get("category") == "FAKE")
    non_face_count = sum(1 for it in tested_items if it.get("category") == "NON_FACE")

    avg_ms = (
        sum(it.get("elapsed_ms", 0.0) for it in tested_items) / total_tested
        if total_tested
        else 0.0
    )

    status_pass = overall_acc >= 90.0
    status_text = "PASSED (>= 90%)" if status_pass else "FAILED (< 90%)"
    status_bg = "#dcfce7" if status_pass else "#fee2e2"
    status_color = "#16a34a" if status_pass else "#dc2626"
    status_border = "#86efac" if status_pass else "#fca5a5"

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
    --primary: #0284c7;
    --success: #16a34a;
    --danger: #dc2626;
    --warning: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-page);
    color: var(--text-main);
    line-height: 1.5;
    padding: 24px;
  }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}
  .header {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 4px;
  }}
  .header p {{
    font-size: 13px;
    color: var(--text-muted);
  }}
  .badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 700;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .kpi-card .label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 6px;
    font-weight: 600;
  }}
  .kpi-card .value {{
    font-size: 26px;
    font-weight: 700;
    color: var(--text-main);
  }}
  .kpi-card .sub {{
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }}
  .section {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .section h2 {{
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  tr:hover {{ background: #f8fafc; }}
  .thumb-img {{
    width: 60px;
    height: 60px;
    border-radius: 8px;
    object-fit: cover;
    display: block;
    border: 2px solid #cbd5e1;
    background: #e2e8f0;
  }}
  .thumb-real {{ border-color: #86efac; }}
  .thumb-fake {{ border-color: #fca5a5; }}
  .thumb-nonface {{ border-color: #93c5fd; }}
  .tag {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }}
  .tag-real {{ background: #dcfce7; color: #15803d; }}
  .tag-fake {{ background: #fee2e2; color: #b91c1c; }}
  .tag-uncertain {{ background: #fef3c7; color: #b45309; }}
  .tag-nonface {{ background: #e0f2fe; color: #0369a1; }}
  .verdict-pass {{ color: #16a34a; font-weight: 700; }}
  .verdict-fail {{ color: #dc2626; font-weight: 700; }}
  .folder-pill {{
    background: #f1f5f9;
    color: #475569;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-family: monospace;
  }}
  .score-bar-bg {{
    width: 80px;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
  }}
  .score-bar-fill {{
    height: 100%;
    border-radius: 4px;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div>
      <h1>FaceCheck Model Evaluation Report</h1>
      <p>Live Online Multi-Style Biometric Integrity &bull; Generated: {timestamp_str}</p>
    </div>
    <div class="badge" style="background:{status_bg}; color:{status_color}; border-color:{status_border};">
      {status_text}
    </div>
  </div>

  <!-- Key Metrics Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="label">System Accuracy</div>
      <div class="value" style="color:{status_color};">{overall_acc:.1f}%</div>
      <div class="sub">Target: &ge; 90.0%</div>
    </div>
    <div class="kpi-card">
      <div class="label">Benchmark Images</div>
      <div class="value">{total_tested}</div>
      <div class="sub">{total_passed} Passed &bull; {total_tested - total_passed} Failed</div>
    </div>
    <div class="kpi-card">
      <div class="label">Average Latency</div>
      <div class="value">{avg_ms:.1f} ms</div>
      <div class="sub">Total Duration: {duration_s:.2f}s</div>
    </div>
    <div class="kpi-card">
      <div class="label">Class Distribution</div>
      <div class="value" style="font-size:16px; margin-top:8px;">
        <span class="tag tag-real">Real: {real_count}</span>
        <span class="tag tag-fake">Fake: {fake_count}</span>
        <span class="tag tag-nonface">Docs: {non_face_count}</span>
      </div>
      <div class="sub">Across {len(group_stats)} Group Subfolders</div>
    </div>
  </div>

  <!-- Category Breakdown Section -->
  <div class="section">
    <h2>Category & Group Folder Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Group Folder Category</th>
          <th>Images Tested</th>
          <th>Passed</th>
          <th>Accuracy</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
"""

    for g_name, stat in sorted(group_stats.items()):
        v_class = "verdict-pass" if stat["accuracy"] >= 90.0 else "verdict-fail"
        v_txt = "PASSED" if stat["accuracy"] >= 90.0 else "REVIEW"
        html += f"""        <tr>
          <td><span class="folder-pill">{g_name}</span></td>
          <td>{stat["total"]}</td>
          <td>{stat["passed"]}</td>
          <td><strong>{stat["accuracy"]:.1f}%</strong></td>
          <td><span class="{v_class}">{v_txt}</span></td>
        </tr>
"""

    html += f"""        <tr style="background:#f8fafc; font-weight:700;">
          <td>OVERALL SYSTEM ACCURACY</td>
          <td>{total_tested}</td>
          <td>{total_passed}</td>
          <td>{overall_acc:.1f}%</td>
          <td><span class="{"verdict-pass" if status_pass else "verdict-fail"}">{status_text}</span></td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Detailed Tested Images Gallery / Table -->
  <div class="section">
    <h2>Tested Benchmark Images Gallery ({total_tested} Images)</h2>
    <table>
      <thead>
        <tr>
          <th style="width:70px;">Image</th>
          <th>Filename & Folder</th>
          <th>Category</th>
          <th>Gender</th>
          <th>AI Risk Score</th>
          <th>Prediction Label</th>
          <th>Latency</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
"""

    for item in tested_items:
        fname = item.get("filename", "")
        folder = item.get("folder", "")
        cat = item.get("category", "")
        gender = item.get("gender") or "N/A"
        score = item.get("score")
        label = item.get("label", "")
        ms = item.get("elapsed_ms", 0.0)
        passed = item.get("passed", False)
        b64_thumb = item.get("thumbnail_b64", "")

        # Border color & tag styling
        if cat == "REAL":
            border_cls = "thumb-real"
            cat_tag = "<span class='tag tag-real'>REAL</span>"
        elif cat == "FAKE":
            border_cls = "thumb-fake"
            cat_tag = "<span class='tag tag-fake'>FAKE</span>"
        else:
            border_cls = "thumb-nonface"
            cat_tag = "<span class='tag tag-nonface'>NON-FACE</span>"

        if label == "LIKELY_REAL":
            label_tag = "<span class='tag tag-real'>LIKELY_REAL</span>"
        elif label == "LIKELY_FAKE":
            label_tag = "<span class='tag tag-fake'>LIKELY_FAKE</span>"
        elif label == "UNCERTAIN":
            label_tag = "<span class='tag tag-uncertain'>UNCERTAIN</span>"
        else:
            label_tag = "<span class='tag tag-nonface'>NO_FACE_FOUND</span>"

        # Score bar
        if score is not None:
            bar_color = "#16a34a" if score <= 30 else ("#d97706" if score <= 65 else "#dc2626")
            score_bar = f"<div class='score-bar-bg'><div class='score-bar-fill' style='width:{score}%; background:{bar_color};'></div></div><strong>{score}</strong>/100"
        else:
            score_bar = "<span style='color:#94a3b8;'>null</span>"

        verdict_str = "<span class='verdict-pass'>PASS</span>" if passed else "<span class='verdict-fail'>FAIL</span>"

        img_elem = (
            f"<img src='{b64_thumb}' class='thumb-img {border_cls}' alt='{fname}'>"
            if b64_thumb
            else "<div class='thumb-img'></div>"
        )

        html += f"""        <tr>
          <td>{img_elem}</td>
          <td>
            <strong>{fname}</strong><br>
            <span class="folder-pill">{folder}</span>
          </td>
          <td>{cat_tag}</td>
          <td><strong>{gender}</strong></td>
          <td>{score_bar}</td>
          <td>{label_tag}</td>
          <td style="color:#64748b;">{ms:.1f} ms</td>
          <td>{verdict_str}</td>
        </tr>
"""

    html += """      </tbody>
    </table>
  </div>

</div>
</body>
</html>
"""
    return html


def save_evaluation_reports(
    reports_dir: Path,
    tested_items: List[Dict[str, Any]],
    group_stats: Dict[str, Dict[str, Any]],
    overall_acc: float,
    duration_s: float,
) -> Tuple[Path, Path]:
    """Save both timestamped and latest HTML/JSON report files."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_display = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = generate_visual_html_report(
        tested_items=tested_items,
        group_stats=group_stats,
        overall_acc=overall_acc,
        duration_s=duration_s,
        timestamp_str=ts_display,
    )

    json_data = {
        "timestamp": ts_display,
        "overall_accuracy": overall_acc,
        "total_tested": len(tested_items),
        "total_passed": sum(1 for it in tested_items if it.get("passed")),
        "duration_seconds": duration_s,
        "group_stats": group_stats,
        "items": [
            {k: v for k, v in it.items() if k != "thumbnail_b64"}
            for it in tested_items
        ],
    }
    json_content = json.dumps(json_data, indent=2)

    # Write latest files
    latest_html = reports_dir / "latest.html"
    latest_json = reports_dir / "latest.json"
    latest_html.write_text(html_content, encoding="utf-8")
    latest_json.write_text(json_content, encoding="utf-8")

    # Write timestamped archival files
    arch_html = reports_dir / f"eval_report_{ts}.html"
    arch_json = reports_dir / f"eval_report_{ts}.json"
    arch_html.write_text(html_content, encoding="utf-8")
    arch_json.write_text(json_content, encoding="utf-8")

    return latest_html, latest_json
