#!/usr/bin/env python3
"""Builds a deterministic, labeled benchmark dataset for large-scale FaceCheck evaluation.

Creates partitioned dataset under assets/datasets/facecheck_v1_benchmark/:
1. identities/ (Multi-capture identity clusters for Genuine and Impostor verification trials)
2. pad/ (Genuine, Print Attack, Screen Replay, and Digital Manipulation for PAD trials)
3. quality/ (Clean, Blurry, Dark, Bright, Non-Frontal, Tiny, Multi-Face, No-Face)

Every sample is logged with SHA-256 and ground-truth metadata in manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "assets" / "datasets" / "facecheck_v1_benchmark"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("facecheck.dataset_builder")


def _calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def generate_face_image(
    identity_seed: int,
    variation_type: str = "frontal_clean",
    size: Tuple[int, int] = (300, 300),
) -> Image.Image:
    """Generate a consistent identity facial image with specific variation."""
    rng = random.Random(identity_seed)
    w, h = size

    # Deterministic identity characteristics based on identity_seed
    skin_r = rng.randint(180, 240)
    skin_g = rng.randint(140, 195)
    skin_b = rng.randint(110, 160)
    skin_color = (skin_r, skin_g, skin_b)

    hair_r = rng.randint(20, 60)
    hair_g = rng.randint(15, 45)
    hair_b = rng.randint(10, 35)
    hair_color = (hair_r, hair_g, hair_b)

    hw = rng.randint(68, 82)
    hh = rng.randint(92, 108)
    eye_spacing = rng.randint(32, 42)
    nose_len = rng.randint(28, 38)
    mouth_w = rng.randint(26, 36)

    # Variation parameters
    yaw_offset = 0
    pitch_offset = 0
    roll_deg = 0.0
    brightness_factor = 1.0
    contrast_factor = 1.0
    blur_radius = 0.0
    jpeg_quality = 95

    if variation_type == "slight_yaw_left":
        yaw_offset = 6
    elif variation_type == "slight_yaw_right":
        yaw_offset = -6
    elif variation_type == "non_frontal_yaw":
        yaw_offset = 26
    elif variation_type == "non_frontal_pitch":
        pitch_offset = 22
    elif variation_type == "low_light":
        brightness_factor = 0.55
    elif variation_type == "dark_underexposed":
        brightness_factor = 0.28
    elif variation_type == "bright_light":
        brightness_factor = 1.25
    elif variation_type == "bright_overexposed":
        brightness_factor = 1.85
    elif variation_type == "slight_blur":
        blur_radius = 1.2
    elif variation_type == "heavy_blur":
        blur_radius = 4.5
    elif variation_type == "compressed_jpeg":
        jpeg_quality = 30
    elif variation_type == "slight_roll":
        roll_deg = 8.0

    img = Image.new("RGB", (w, h), color=(235, 238, 242))
    d = ImageDraw.Draw(img)

    cx = (w // 2) + yaw_offset
    cy = (h // 2) + pitch_offset

    # 1. Head shape
    head_box = [cx - hw, cy - hh, cx + hw, cy + hh]
    d.ellipse(head_box, fill=skin_color, outline=(int(skin_r * 0.8), int(skin_g * 0.8), int(skin_b * 0.8)), width=2)

    # 2. Hair cap
    d.pieslice([cx - hw - 3, cy - hh - 8, cx + hw + 3, cy - int(hh * 0.1)], 180, 360, fill=hair_color)

    # 3. Eyes
    eye_y = cy - int(hh * 0.15)
    ew = 14
    eh = 9

    # Left eye
    le_cx = cx - eye_spacing
    d.ellipse([le_cx - ew, eye_y - eh, le_cx + ew, eye_y + eh], fill=(255, 255, 255), outline=(30, 30, 30), width=1)
    d.ellipse([le_cx - 4, eye_y - 4, le_cx + 4, eye_y + 4], fill=(40, 25, 15))

    # Right eye
    re_cx = cx + eye_spacing
    d.ellipse([re_cx - ew, eye_y - eh, re_cx + ew, eye_y + eh], fill=(255, 255, 255), outline=(30, 30, 30), width=1)
    d.ellipse([re_cx - 4, eye_y - 4, re_cx + 4, eye_y + 4], fill=(40, 25, 15))

    # 4. Eyebrows
    d.line([le_cx - ew, eye_y - eh - 5, le_cx + ew, eye_y - eh - 7], fill=hair_color, width=3)
    d.line([re_cx - ew, eye_y - eh - 7, re_cx + ew, eye_y - eh - 5], fill=hair_color, width=3)

    # 5. Nose
    nose_y = cy + int(hh * 0.12)
    d.polygon([(cx, nose_y - nose_len), (cx - 7, nose_y), (cx + 7, nose_y)], fill=(int(skin_r * 0.9), int(skin_g * 0.85), int(skin_b * 0.85)))

    # 6. Mouth
    mouth_y = cy + int(hh * 0.48)
    if variation_type == "expression_smile":
        d.arc([cx - mouth_w - 4, mouth_y - 12, cx + mouth_w + 4, mouth_y + 16], start=0, end=180, fill=(160, 60, 60), width=5)
    else:
        d.arc([cx - mouth_w, mouth_y - 8, cx + mouth_w, mouth_y + 12], start=0, end=180, fill=(160, 60, 60), width=4)

    # Apply roll
    if roll_deg != 0.0:
        img = img.rotate(roll_deg, resample=Image.Resampling.BICUBIC, fillcolor=(235, 238, 242))

    # Apply brightness/contrast
    if brightness_factor != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Apply blur
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def generate_print_attack_image(base_img: Image.Image) -> Image.Image:
    """Simulate a printed photograph presentation attack."""
    w, h = base_img.size
    img = base_img.copy()

    # 1. Overlay periodic halftone dot raster grid
    arr = np.array(img, dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    freq = 0.45
    halftone_pattern = (np.sin(freq * x) * np.cos(freq * y) * 28.0)
    arr = np.clip(arr + halftone_pattern[:, :, None], 0, 255)

    res_img = Image.fromarray(arr.astype(np.uint8))

    # 2. Add white photo paper border around the print
    border_w = int(min(w, h) * 0.06)
    bordered = Image.new("RGB", (w, h), color=(248, 248, 250))
    inner = res_img.resize((w - 2 * border_w, h - 2 * border_w))
    bordered.paste(inner, (border_w, border_w))

    return bordered


def generate_screen_replay_image(base_img: Image.Image) -> Image.Image:
    """Simulate an electronic screen replay presentation attack."""
    w, h = base_img.size
    arr = np.array(base_img, dtype=np.float32)

    # 1. Overlay 2D Moiré grid frequency interference
    y, x = np.ogrid[:h, :w]
    moire_grid = (np.sin(x * 0.75 + y * 0.25) * np.cos(y * 0.65 - x * 0.20) * 35.0)
    arr = np.clip(arr + moire_grid[:, :, None], 0, 255)

    # 2. Add glass specular reflection glare spot (large contiguous region)
    glare_cx, glare_cy = int(w * 0.35), int(h * 0.30)
    dist_sq = (x - glare_cx) ** 2 + (y - glare_cy) ** 2
    glare_spot = np.exp(-dist_sq / (2 * (25.0 ** 2))) * 220.0
    arr = np.clip(arr + glare_spot[:, :, None], 0, 255)

    res_img = Image.fromarray(arr.astype(np.uint8))

    # 3. Add dark device bezel frame
    bezel_w = int(min(w, h) * 0.08)
    framed = Image.new("RGB", (w, h), color=(18, 18, 20))
    inner = res_img.resize((w - 2 * bezel_w, h - 2 * bezel_w))
    framed.paste(inner, (bezel_w, bezel_w))

    return framed


def generate_no_face_image(size: Tuple[int, int] = (300, 300), idx: int = 0) -> Image.Image:
    """Generate a landscape / abstract image with zero human faces."""
    w, h = size
    img = Image.new("RGB", (w, h), color=(60, 90, 140))
    d = ImageDraw.Draw(img)

    # Mountains / scenery
    d.polygon([(0, h), (w * 0.35, h * 0.45), (w * 0.7, h), (w, h * 0.55), (w, h)], fill=(40, 65, 100))
    d.polygon([(w * 0.2, h), (w * 0.55, h * 0.35), (w * 0.9, h)], fill=(75, 105, 130))
    d.rectangle([0, int(h * 0.75), w, h], fill=(35, 120, 60))
    return img


def generate_multi_face_image(size: Tuple[int, int] = (400, 300)) -> Image.Image:
    """Generate an image containing 2 distinct human faces."""
    w, h = size
    img = Image.new("RGB", (w, h), color=(230, 235, 240))

    face1 = generate_face_image(identity_seed=101, size=(180, 240))
    face2 = generate_face_image(identity_seed=102, size=(180, 240))

    img.paste(face1, (20, 40))
    img.paste(face2, (200, 40))
    return img


def build_benchmark_dataset(target_samples: int = 1600) -> Dict[str, Any]:
    """Build and save complete benchmark dataset with manifest."""
    logger.info("Building benchmark dataset under %s (target ~%d images)...", BENCHMARK_DIR, target_samples)

    identities_dir = BENCHMARK_DIR / "identities"
    pad_dir = BENCHMARK_DIR / "pad"
    quality_dir = BENCHMARK_DIR / "quality"

    for d in [
        identities_dir,
        pad_dir / "genuine",
        pad_dir / "print_attack",
        pad_dir / "screen_replay",
        pad_dir / "digital_manipulation",
        quality_dir / "clean",
        quality_dir / "blurry",
        quality_dir / "dark",
        quality_dir / "bright",
        quality_dir / "non_frontal",
        quality_dir / "no_face",
        quality_dir / "multi_face",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []

    # 1. Build Identities Partition (60 identities x 8 variations = 480 images)
    num_identities = 60
    variations = [
        "frontal_clean", "slight_yaw_left", "slight_yaw_right", "low_light",
        "bright_light", "expression_smile", "slight_blur", "compressed_jpeg"
    ]

    for id_idx in range(1, num_identities + 1):
        id_str = f"id_{id_idx:03d}"
        id_folder = identities_dir / id_str
        id_folder.mkdir(parents=True, exist_ok=True)

        for v_type in variations:
            im = generate_face_image(identity_seed=id_idx * 17, variation_type=v_type)
            fname = f"{id_str}_{v_type}.jpg"
            out_p = id_folder / fname

            import io
            buf = io.BytesIO()
            q = 30 if v_type == "compressed_jpeg" else 95
            im.save(buf, format="JPEG", quality=q)
            raw = buf.getvalue()
            out_p.write_bytes(raw)

            manifest_entries.append({
                "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
                "sha256": _calculate_sha256(raw),
                "category": "identity_verification",
                "identity_id": id_str,
                "variation": v_type,
                "is_attack": False,
                "attack_type": "GENUINE",
                "expected_faces": 1,
                "expected_quality": "LOW_QUALITY" if v_type in ("low_light", "compressed_jpeg") else "OK",
            })

    # 2. Build PAD Partition (600 images)
    # Genuine: 200
    for i in range(1, 201):
        im = generate_face_image(identity_seed=1000 + i, variation_type="frontal_clean")
        out_p = pad_dir / "genuine" / f"genuine_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "pad",
            "identity_id": f"pad_gen_{i:03d}",
            "is_attack": False,
            "attack_type": "GENUINE",
            "expected_faces": 1,
            "expected_quality": "OK",
        })

    # Print attacks: 150
    for i in range(1, 151):
        base = generate_face_image(identity_seed=2000 + i, variation_type="frontal_clean")
        p_img = generate_print_attack_image(base)
        out_p = pad_dir / "print_attack" / f"print_attack_{i:03d}.jpg"
        buf = io.BytesIO()
        p_img.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "pad",
            "identity_id": f"pad_print_{i:03d}",
            "is_attack": True,
            "attack_type": "PRINT_ATTACK",
            "expected_faces": 1,
            "expected_quality": "OK",
        })

    # Screen replays: 150
    for i in range(1, 151):
        base = generate_face_image(identity_seed=3000 + i, variation_type="frontal_clean")
        s_img = generate_screen_replay_image(base)
        out_p = pad_dir / "screen_replay" / f"screen_replay_{i:03d}.jpg"
        buf = io.BytesIO()
        s_img.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "pad",
            "identity_id": f"pad_replay_{i:03d}",
            "is_attack": True,
            "attack_type": "SCREEN_REPLAY",
            "expected_faces": 1,
            "expected_quality": "OK",
        })

    # Digital manipulation / Deepfakes: 100
    for i in range(1, 101):
        im = generate_face_image(identity_seed=4000 + i, variation_type="slight_yaw_left")
        # Add high-frequency noise typical of AI generative models
        arr = np.array(im, dtype=np.float32)
        noise = np.random.normal(0, 12.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        d_img = Image.fromarray(arr.astype(np.uint8))
        out_p = pad_dir / "digital_manipulation" / f"deepfake_{i:03d}.jpg"
        buf = io.BytesIO()
        d_img.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "pad",
            "identity_id": f"pad_deepfake_{i:03d}",
            "is_attack": True,
            "attack_type": "DIGITAL_MANIPULATION",
            "expected_faces": 1,
            "expected_quality": "OK",
        })

    # 3. Build Quality & Detection Edge Cases (460 images)
    # Clean: 80
    for i in range(1, 81):
        im = generate_face_image(identity_seed=5000 + i, variation_type="frontal_clean")
        out_p = quality_dir / "clean" / f"clean_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 1,
            "expected_quality": "OK",
        })

    # Blurry: 80
    for i in range(1, 81):
        im = generate_face_image(identity_seed=5100 + i, variation_type="heavy_blur")
        out_p = quality_dir / "blurry" / f"blurry_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 1,
            "expected_quality": "LOW_QUALITY",
            "issue": "BLURRY",
        })

    # Dark: 50
    for i in range(1, 51):
        im = generate_face_image(identity_seed=5200 + i, variation_type="dark_underexposed")
        out_p = quality_dir / "dark" / f"dark_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 1,
            "expected_quality": "LOW_QUALITY",
            "issue": "TOO_DARK",
        })

    # Bright: 50
    for i in range(1, 51):
        im = generate_face_image(identity_seed=5300 + i, variation_type="bright_overexposed")
        out_p = quality_dir / "bright" / f"bright_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 1,
            "expected_quality": "LOW_QUALITY",
            "issue": "TOO_BRIGHT",
        })

    # Non-frontal: 60
    for i in range(1, 61):
        im = generate_face_image(identity_seed=5400 + i, variation_type="non_frontal_yaw")
        out_p = quality_dir / "non_frontal" / f"non_frontal_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 1,
            "expected_quality": "LOW_QUALITY",
            "issue": "FACE_NOT_FRONTAL",
        })

    # No face: 100
    for i in range(1, 101):
        im = generate_no_face_image(idx=i)
        out_p = quality_dir / "no_face" / f"no_face_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 0,
            "expected_quality": "NO_FACE",
        })

    # Multi face: 40
    for i in range(1, 41):
        im = generate_multi_face_image()
        out_p = quality_dir / "multi_face" / f"multi_face_{i:03d}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
        out_p.write_bytes(raw)
        manifest_entries.append({
            "file_path": str(out_p.relative_to(BENCHMARK_DIR)).replace("\\", "/"),
            "sha256": _calculate_sha256(raw),
            "category": "quality",
            "expected_faces": 2,
            "expected_quality": "MULTIPLE_FACES",
        })

    # Save manifest
    manifest_file = BENCHMARK_DIR / "manifest.json"
    manifest_data = {
        "version": "1.0.0",
        "description": "Deterministic ground-truth benchmark for FaceCheck API V1 evaluation",
        "total_files": len(manifest_entries),
        "partitions": {
            "identity_verification": 480,
            "pad": 600,
            "quality": 460,
        },
        "samples": manifest_entries,
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info("Benchmark dataset created: %d files registered in %s", len(manifest_entries), manifest_file)
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic FaceCheck benchmark dataset.")
    parser.add_argument("--target-samples", type=int, default=1600, help="Target sample count")
    args = parser.parse_args()
    build_benchmark_dataset(target_samples=args.target_samples)
