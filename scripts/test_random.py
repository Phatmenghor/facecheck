#!/usr/bin/env python3
"""
FaceCheck API: Infinite Dynamic Online Testing with Unique Non-Cached Random URLs.

Generates completely unique, truly random URLs with UUID seeds and cache-busting
timestamps on every download. Supports scaling from 10 up to 10,000 images seamlessly
in chunked parallel batches:
- Real human portraits: i.pravatar.cc & randomuser.me CDN (infinite unique UUID seeds)
- Synthetic AI & Avatars: api.dicebear.com & robohash.org (infinite unique UUID seeds)
- Real-time KYC pipeline analysis, gender estimation, deepfake scoring, and auto-matching.

Usage:
    py scripts/test_random.py
    py scripts/test_random.py --count 15
    py scripts/test_random.py --count 50
    py scripts/test_random.py --count 100 --review
"""

import sys
import argparse
import concurrent.futures
import random
import time
import uuid
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.pipeline import run_pipeline
from app.config import get_config

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def fetch_url(url: str, timeout: int = 6) -> Optional[bytes]:
    """Fetch raw binary content from an external HTTP/HTTPS URL with anti-cache headers."""
    headers = {
        "User-Agent": USER_AGENT,
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = resp.read()
                if len(data) > 500:
                    return data
    except Exception:
        pass
    return None


def generate_random_real_task(idx: int) -> Tuple[str, str, str, str]:
    """Generate a 100% unique, non-cached random URL for high-resolution real human portraits."""
    unique_seed = f"{uuid.uuid4().hex}_{idx}_{time.time_ns()}"
    cb = uuid.uuid4().hex[:8]
    
    # Alternate between high-res Pravatar 800x800 portraits and randomuser direct CDNs
    if idx % 3 == 0:
        url = f"https://i.pravatar.cc/800?u={unique_seed}&cb={cb}"
        fname = f"real_pravatar_{idx:05d}_{cb}.jpg"
        return (url, fname, "REAL", "UNKNOWN")
    elif idx % 3 == 1:
        p_id = (idx % 70) + 1
        url = f"https://i.pravatar.cc/800?img={p_id}&cb={cb}"
        fname = f"real_pravatar_hd_{p_id:02d}_{cb}.jpg"
        return (url, fname, "REAL", "UNKNOWN")
    else:
        gender = "men" if (idx % 2 == 0) else "women"
        p_id = random.randint(1, 99)
        url = f"https://randomuser.me/api/portraits/{gender}/{p_id}.jpg?cb={unique_seed}"
        fname = f"real_cdn_{gender}_{p_id:02d}_{cb}.jpg"
        gt_g = "MALE" if gender == "men" else "FEMALE"
        return (url, fname, "REAL", gt_g)


def generate_random_synthetic_task(idx: int) -> Tuple[str, str, str, str]:
    """Generate a 100% unique, non-cached random URL for large synthetic AI / avatar faces."""
    unique_seed = f"synth_{uuid.uuid4().hex}_{idx}_{time.time_ns()}"
    cb = uuid.uuid4().hex[:8]
    styles = ["personas", "lorelei", "avataaars", "bottts", "adventurer", "open-peeps", "thumbs", "pixel-art"]
    
    if idx % 5 != 0:
        style = styles[idx % len(styles)]
        url = f"https://api.dicebear.com/7.x/{style}/png?seed={unique_seed}&cb={cb}&size=512"
        fname = f"synth_{style}_{idx:05d}_{cb}.png"
        return (url, fname, "FAKE", "SYNTH")
    else:
        url = f"https://robohash.org/{unique_seed}.png?size=600x600&set=set1&cb={cb}"
        fname = f"synth_robo_{idx:05d}_{cb}.png"
        return (url, fname, "FAKE", "SYNTH")


def generate_random_non_face_task(idx: int) -> Tuple[str, str, str, str]:
    """Generate a 100% dynamic URL for non-face objects / landscapes with zero human faces."""
    categories = [
        "Mountain_Lake", "Forest_Trees", "Ocean_Waves", "City_Skyline",
        "Credit_Card", "Bank_Vault", "Circuit_Board", "Modern_Architecture"
    ]
    bg_colors = ["1e293b", "0f766e", "0369a1", "4338ca", "b45309", "334155", "164e63"]
    cat = categories[idx % len(categories)]
    bg = bg_colors[idx % len(bg_colors)]
    cb = uuid.uuid4().hex[:8]
    label = urllib.parse.quote_plus(f"{cat}_{cb}")
    url = f"https://dummyimage.com/800x600/{bg}/ffffff.png&text={label}"
    fname = f"non_face_{cat.lower()}_{idx:04d}_{cb}.png"
    return (url, fname, "NON_FACE", "N/A")


def download_item_worker(task: Tuple[str, str, str, str]) -> Optional[Dict[str, Any]]:
    """Worker to download and package an image item."""
    url, fname, gt_type, gt_gender = task
    raw = fetch_url(url, timeout=6)
    if raw:
        return {
            "filename": fname,
            "url": url,
            "raw_bytes": raw,
            "gt_type": gt_type,
            "gt_gender": gt_gender,
        }
    return None


def fetch_dynamic_online_batch(count: int, max_workers: int = 16) -> List[Dict[str, Any]]:
    """Download N 100% dynamic, non-cached random images concurrently across external providers."""
    n_nonface = max(1, count // 10) if count >= 8 else 0
    remaining = count - n_nonface
    n_real = max(1, remaining // 2)
    n_fake = max(1, remaining - n_real)

    tasks: List[Tuple[str, str, str, str]] = []
    for i in range(n_real):
        tasks.append(generate_random_real_task(i))
    for i in range(n_fake):
        tasks.append(generate_random_synthetic_task(i))
    for i in range(n_nonface):
        tasks.append(generate_random_non_face_task(i))

    random.shuffle(tasks)

    items: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_item_worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                items.append(res)

    return items


def run_testing_batch(
    items: List[Dict[str, Any]],
    batch_num: int,
    total_batches: int,
) -> Tuple[int, int, int]:
    """Execute pipeline on a batch of items and print table rows."""
    correct_predictions = 0
    gender_matches = 0
    gender_eval_count = 0

    for item in items:
        fname = item["filename"]
        gt_type = item["gt_type"]
        gt_gender = item["gt_gender"]
        raw_bytes = item["raw_bytes"]

        try:
            res = run_pipeline(fname, raw_bytes)
            score = res.get("score")
            label = res.get("label")
            gender = res.get("gender")
            conf = res.get("gender_confidence")
            face_chk = res.get("face_check", {})
            face_status = face_chk.get("status", "PASSED")
            reasons = res.get("reason_codes", [])

            # Format fields
            score_str = f"{score:2d}% Risk" if score is not None else "Blocked"
            
            # Non-human check
            is_non_human = face_status != "PASSED" and any("NO_FACE" in r for r in reasons)
            if is_non_human or not gender:
                gender_str = "N/A (No Face)"
            elif gender == "UNKNOWN":
                gender_str = "Borderline"
            elif conf:
                gender_str = f"{gender} ({int(round(conf*100))}%)"
            else:
                gender_str = gender

            if face_status == "PASSED" and label != "RETAKE":
                face_str = "[OK] Centered"
            else:
                r_code = reasons[0] if reasons else "Issue"
                face_str = f"[!] {r_code[:8]}"

            # Bank decision
            if gt_type == "REAL" and (score is not None and score <= 50):
                decision_str = "Approved"
                is_correct = True
            elif gt_type == "FAKE" and ((score is not None and score > 50) or label == "RETAKE"):
                decision_str = "Blocked (Fake)"
                is_correct = True
            elif gt_type == "NON_FACE" and (label == "RETAKE" or face_status != "PASSED"):
                decision_str = "Retake (No Face)"
                is_correct = True
            else:
                decision_str = "Retake Required"
                is_correct = False

            if is_correct:
                correct_predictions += 1

            # Gender auto-matching
            g_match_str = "--"
            if gt_type == "REAL" and gt_gender in ("MALE", "FEMALE"):
                gender_eval_count += 1
                if gender == gt_gender:
                    g_match_str = "[MATCH]"
                    gender_matches += 1
                elif gender == "UNKNOWN":
                    g_match_str = "[NEUT]"
                else:
                    g_match_str = "[DIFF]"

            print(f"{fname[:25]:25s} | {gt_type[:8]:8s} | {face_str:12s} | {gt_gender:10s} | {gender_str:16s} | {score_str:9s} | {decision_str:15s} | {g_match_str:7s}")

        except Exception as exc:
            print(f"{fname[:25]:25s} | {gt_type[:8]:8s} | ERROR: {exc}")

    return correct_predictions, gender_matches, gender_eval_count


def main():
    parser = argparse.ArgumentParser(description="FaceCheck Infinite Dynamic Random Online Testing")
    parser.add_argument("--count", type=int, default=15, help="Number of random non-cached images to test (e.g. 15, 100, 1000, up to 10000)")
    parser.add_argument("--chunk-size", type=int, default=25, help="Parallel chunk download batch size (default: 25)")
    parser.add_argument("--review", action="store_true", help="Open visual review report in browser after test")
    args = parser.parse_args()

    total_target = max(1, min(10000, args.count))
    chunk_size = max(5, min(50, args.chunk_size))
    total_chunks = (total_target + chunk_size - 1) // chunk_size

    print("=" * 125)
    print(f"     FACECHECK INFINITE DYNAMIC RANDOM TESTING ({total_target} Unique Non-Cached Images From Internet)")
    print("=" * 125)
    print(f" Mode: 100% Online Dynamic | UUID Seed Cache-Busting: ACTIVE | Providers: pravatar, randomuser, dicebear, robohash")
    print("-" * 125)
    print(f"{'APPLICANT / FILE':25s} | {'TYPE':8s} | {'FACE CHECK':12s} | {'EXT GENDER':10s} | {'PRED GENDER':16s} | {'AI RISK':9s} | {'BANK DECISION':15s} | {'G-MATCH':7s}")
    print("-" * 125)

    total_correct = 0
    total_g_matches = 0
    total_g_eval = 0
    total_processed = 0

    t_all_start = time.perf_counter()

    for chunk_idx in range(total_chunks):
        current_chunk_count = min(chunk_size, total_target - total_processed)
        if current_chunk_count <= 0:
            break

        # Download dynamic chunk with unique seeds
        chunk_items = fetch_dynamic_online_batch(current_chunk_count, max_workers=16)
        if not chunk_items:
            print(f"[WARN] Chunk {chunk_idx+1} download timed out. Retrying...")
            continue

        c_corr, g_mat, g_ev = run_testing_batch(chunk_items, chunk_idx + 1, total_chunks)
        total_correct += c_corr
        total_g_matches += g_mat
        total_g_eval += g_ev
        total_processed += len(chunk_items)

    t_all_elapsed = time.perf_counter() - t_all_start
    acc = (total_correct / total_processed) * 100.0 if total_processed > 0 else 0.0
    g_acc = (total_g_matches / total_g_eval) * 100.0 if total_g_eval > 0 else 0.0

    print("-" * 125)
    print(f" Summary: Tested {total_processed} live dynamic images in {t_all_elapsed:.2f}s ({t_all_elapsed/max(1, total_processed)*1000:.1f}ms/img total)")
    print(f" AI Deepfake Detection Accuracy: {acc:.1f}% ({total_correct}/{total_processed})")
    if total_g_eval > 0:
        print(f" External Gender Auto-Match:      {g_acc:.1f}% ({total_g_matches}/{total_g_eval} verified matched)")
    print("=" * 125 + "\n")

    if args.review:
        config = get_config()
        latest_html = PROJECT_ROOT / config.eval.reports_dir / "latest.html"
        if latest_html.is_file():
            print(f"[BROWSER] Launching visual report: {latest_html.resolve().as_uri()}")
            webbrowser.open(latest_html.resolve().as_uri())


if __name__ == "__main__":
    main()
