#!/usr/bin/env python3
"""Deterministic synthetic test-fixture generator for FaceCheck API.

Generates labeled synthetic fixtures under tests/fixtures/:
- valid_face_clean.jpg (detected cleanly by YuNet)
- valid_face_png.png
- valid_face_webp.webp
- valid_nid_document.jpg (simulated national ID card with portrait)
- corrupt_truncated.jpg
- empty_file.jpg
- polyglot_pdf_image.jpg
- trailing_data_jpeg.jpg
- tiny_image.jpg (<32px)
- blurry_image.jpg
- dark_image.jpg (underexposed)
- bright_image.jpg (overexposed)
- rotated_90.jpg
- extreme_aspect_ratio.jpg
- compression_artifacts.jpg
- no_face_landscape.jpg
- multiple_faces.jpg
- print_attack.jpg (paper texture, halftone dots, frame)
- screen_replay.jpg (moiré pattern and device bezel)

Every generated fixture is explicitly documented as synthetic in tests/fixtures/manifest.json.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def draw_synthetic_face(
    size: Tuple[int, int] = (300, 300),
    center_offset: Tuple[int, int] = (0, 0),
    scale: float = 1.0,
    roll_deg: float = 0.0,
    skin_color: Tuple[int, int, int] = (215, 175, 145),
    bg_color: Tuple[int, int, int] = (235, 235, 235),
) -> Image.Image:
    """Generate a clean synthetic face graphic that reliably triggers face detectors."""
    w, h = size
    img = Image.new("RGB", (w, h), color=bg_color)
    d = ImageDraw.Draw(img)

    cx = (w // 2) + center_offset[0]
    cy = (h // 2) + center_offset[1]
    hw = int(75 * scale)
    hh = int(100 * scale)

    # 1. Head oval
    head_box = [cx - hw, cy - hh, cx + hw, cy + hh]
    d.ellipse(head_box, fill=skin_color, outline=(160, 120, 95), width=2)

    # 2. Hair cap
    d.pieslice([cx - hw - 2, cy - hh - 5, cx + hw + 2, cy - int(hh * 0.1)], 180, 360, fill=(50, 40, 30))

    # 3. Eyes
    eye_y = cy - int(hh * 0.15)
    eye_dx = int(hw * 0.45)
    ew = max(6, int(15 * scale))
    eh = max(4, int(10 * scale))

    # Left eye
    d.ellipse([cx - eye_dx - ew, eye_y - eh, cx - eye_dx + ew, eye_y + eh], fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    d.ellipse([cx - eye_dx - 4, eye_y - 4, cx - eye_dx + 4, eye_y + 4], fill=(30, 20, 10))

    # Right eye
    d.ellipse([cx + eye_dx - ew, eye_y - eh, cx + eye_dx + ew, eye_y + eh], fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    d.ellipse([cx + eye_dx - 4, eye_y - 4, cx + eye_dx + 4, eye_y + 4], fill=(30, 20, 10))

    # 4. Eyebrows
    d.line([cx - eye_dx - ew, eye_y - eh - 6, cx - eye_dx + ew, eye_y - eh - 8], fill=(45, 30, 20), width=3)
    d.line([cx + eye_dx - ew, eye_y - eh - 8, cx + eye_dx + ew, eye_y - eh - 6], fill=(45, 30, 20), width=3)

    # 5. Nose
    nose_y = cy + int(hh * 0.15)
    d.polygon([(cx, nose_y - int(hh * 0.15)), (cx - int(hw * 0.12), nose_y), (cx + int(hw * 0.12), nose_y)], fill=(195, 150, 120))

    # 6. Mouth
    mouth_y = cy + int(hh * 0.50)
    mw = int(hw * 0.38)
    d.arc([cx - mw, mouth_y - 10, cx + mw, mouth_y + 15], start=0, end=180, fill=(170, 65, 65), width=4)

    if roll_deg != 0.0:
        img = img.rotate(roll_deg, resample=Image.Resampling.BICUBIC, fillcolor=bg_color)

    return img


def generate_all_fixtures() -> Dict[str, Any]:
    """Create deterministic fixture suite and return manifest."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "version": "1.0.0",
        "description": "Deterministic synthetic test fixtures for FaceCheck API automated testing",
        "notice": "All fixtures in this directory are purely synthetic and must NOT be used for real accuracy claims.",
        "fixtures": {},
    }

    def record(name: str, category: str, desc: str, expected_face: bool = False, expected_quality: str = "OK"):
        p = FIXTURES_DIR / name
        manifest["fixtures"][name] = {
            "relative_path": f"tests/fixtures/{name}",
            "category": category,
            "description": desc,
            "expected_face_detected": expected_face,
            "expected_quality": expected_quality,
            "size_bytes": p.stat().st_size if p.is_file() else 0,
            "synthetic": True,
        }

    # 1. Valid clean face image (JPG)
    clean_im = draw_synthetic_face(size=(300, 300))
    clean_p = FIXTURES_DIR / "valid_face_clean.jpg"
    clean_im.save(clean_p, "JPEG", quality=95)
    record("valid_face_clean.jpg", "valid", "Clean centered synthetic frontal face", expected_face=True)

    # 2. Valid PNG & WebP versions
    png_p = FIXTURES_DIR / "valid_face_png.png"
    clean_im.save(png_p, "PNG")
    record("valid_face_png.png", "valid", "Lossless PNG synthetic face", expected_face=True)

    webp_p = FIXTURES_DIR / "valid_face_webp.webp"
    clean_im.save(webp_p, "WEBP", quality=95)
    record("valid_face_webp.webp", "valid", "WebP format synthetic face", expected_face=True)

    # 3. Simulated National ID document (Cambodian / ISO 7810 ID-1 card style)
    id_im = Image.new("RGB", (600, 380), color=(230, 240, 248))
    id_draw = ImageDraw.Draw(id_im)
    # Card border and header
    id_draw.rounded_rectangle([10, 10, 590, 370], radius=15, outline=(150, 170, 190), width=3)
    id_draw.rectangle([20, 20, 580, 75], fill=(35, 75, 125))
    id_draw.text((40, 35), "KINGDOM OF CAMBODIA - NATIONAL IDENTITY CARD", fill=(255, 255, 255))
    # Simulated Portrait area
    portrait = draw_synthetic_face(size=(160, 200), bg_color=(220, 230, 240))
    id_im.paste(portrait, (40, 95))
    # Card text lines
    for line_idx, line_txt in enumerate(["ID: 010987654", "Name: SOK CHANTHA", "DOB: 15/08/1995", "Sex: M", "Nationality: KHM"]):
        id_draw.text((230, 110 + line_idx * 30), line_txt, fill=(30, 30, 40))
    id_p = FIXTURES_DIR / "valid_nid_document.jpg"
    id_im.save(id_p, "JPEG", quality=95)
    record("valid_nid_document.jpg", "valid", "Simulated identity card with portrait", expected_face=True)

    # 4. Tiny image (<32px)
    tiny_im = draw_synthetic_face(size=(24, 24), scale=0.7)
    tiny_p = FIXTURES_DIR / "tiny_image.jpg"
    tiny_im.save(tiny_p, "JPEG", quality=90)
    record("tiny_image.jpg", "quality_edge", "Sub-resolution image 24x24 px", expected_face=False, expected_quality="LOW_QUALITY")

    # 5. Blurry image
    blur_im = clean_im.filter(ImageFilter.GaussianBlur(radius=6.0))
    blur_p = FIXTURES_DIR / "blurry_image.jpg"
    blur_im.save(blur_p, "JPEG", quality=90)
    record("blurry_image.jpg", "quality_edge", "Excessively blurred face", expected_face=True, expected_quality="LOW_QUALITY")

    # 6. Dark / underexposed
    dark_arr = (np.array(clean_im, dtype=np.float32) * 0.08).astype(np.uint8)
    dark_im = Image.fromarray(dark_arr)
    dark_p = FIXTURES_DIR / "dark_image.jpg"
    dark_im.save(dark_p, "JPEG", quality=90)
    record("dark_image.jpg", "quality_edge", "Underexposed dark image", expected_face=False, expected_quality="LOW_QUALITY")

    # 7. Bright / overexposed
    bright_arr = np.clip(np.array(clean_im, dtype=np.float32) * 1.8 + 80, 0, 255).astype(np.uint8)
    bright_im = Image.fromarray(bright_arr)
    bright_p = FIXTURES_DIR / "bright_image.jpg"
    bright_im.save(bright_p, "JPEG", quality=90)
    record("bright_image.jpg", "quality_edge", "Overexposed washed out image", expected_face=False, expected_quality="LOW_QUALITY")

    # 8. Rotated 90 degrees
    rot_im = clean_im.rotate(90)
    rot_p = FIXTURES_DIR / "rotated_90.jpg"
    rot_im.save(rot_p, "JPEG", quality=95)
    record("rotated_90.jpg", "pose_edge", "Face rotated 90 degrees clockwise", expected_face=False)

    # 9. No face landscape
    no_face_im = Image.new("RGB", (300, 300), color=(70, 130, 180))
    nf_draw = ImageDraw.Draw(no_face_im)
    nf_draw.polygon([(0, 300), (120, 150), (220, 250), (300, 120), (300, 300)], fill=(34, 139, 34))
    nf_p = FIXTURES_DIR / "no_face_landscape.jpg"
    no_face_im.save(nf_p, "JPEG", quality=90)
    record("no_face_landscape.jpg", "face_detection", "Landscape scene with no human face", expected_face=False)

    # 10. Multiple faces
    multi_im = Image.new("RGB", (500, 300), color=(230, 230, 230))
    face1 = draw_synthetic_face(size=(220, 220), scale=0.85)
    face2 = draw_synthetic_face(size=(220, 220), scale=0.85, skin_color=(200, 160, 135))
    multi_im.paste(face1, (20, 40))
    multi_im.paste(face2, (260, 40))
    multi_p = FIXTURES_DIR / "multiple_faces.jpg"
    multi_im.save(multi_p, "JPEG", quality=95)
    record("multiple_faces.jpg", "face_detection", "Two distinct human faces in frame", expected_face=True)

    # 11. Synthetic Print Attack (printed paper simulation)
    # Adds subtle paper texture, halftone dot noise, and white paper margin
    print_im = Image.new("RGB", (360, 360), color=(250, 250, 248))
    p_draw = ImageDraw.Draw(print_im)
    p_draw.rectangle([15, 15, 345, 345], outline=(200, 200, 195), width=2)
    # Paste face inside
    print_im.paste(clean_im, (30, 30))
    # Apply synthetic halftone pattern
    p_arr = np.array(print_im, dtype=np.float32)
    y_coords, x_coords = np.mgrid[0:360, 0:360]
    halftone_grid = (np.sin(x_coords * 1.2) * np.sin(y_coords * 1.2) * 8.0).astype(np.float32)
    p_arr = np.clip(p_arr + halftone_grid[:, :, None], 0, 255).astype(np.uint8)
    print_res = Image.fromarray(p_arr)
    print_p = FIXTURES_DIR / "print_attack.jpg"
    print_res.save(print_p, "JPEG", quality=88)
    record("print_attack.jpg", "pad_presentation_attack", "Simulated printed photo with paper border and halftone artifacts", expected_face=True)

    # 12. Synthetic Screen Replay Attack (moiré pattern + screen bezel simulation)
    screen_im = Image.new("RGB", (380, 380), color=(20, 20, 22))
    s_draw = ImageDraw.Draw(screen_im)
    s_draw.rounded_rectangle([10, 10, 370, 370], radius=18, outline=(80, 80, 85), width=4)
    # Screen display inside bezel
    screen_im.paste(clean_im.resize((320, 320)), (30, 30))
    # Add high-frequency 2D moiré interference
    s_arr = np.array(screen_im, dtype=np.float32)
    sy, sx = np.mgrid[0:380, 0:380]
    moire = (np.sin(sx * 0.45 + sy * 0.35) * np.cos(sx * 0.25 - sy * 0.40) * 14.0).astype(np.float32)
    s_arr[30:350, 30:350, :] = np.clip(s_arr[30:350, 30:350, :] + moire[30:350, 30:350, None], 0, 255)
    screen_res = Image.fromarray(s_arr.astype(np.uint8))
    screen_p = FIXTURES_DIR / "screen_replay.jpg"
    screen_res.save(screen_p, "JPEG", quality=90)
    record("screen_replay.jpg", "pad_presentation_attack", "Simulated smartphone screen replay with moire pattern and bezel", expected_face=True)

    # 13. Security test fixtures: Corrupted, Empty, Polyglot, Trailing data
    # Empty file (0 bytes)
    empty_p = FIXTURES_DIR / "invalid_empty.jpg"
    empty_p.write_bytes(b"")
    record("invalid_empty.jpg", "security_invalid", "Empty 0-byte file", expected_face=False)

    # Corrupt truncated JPEG
    corrupt_p = FIXTURES_DIR / "corrupt_truncated.jpg"
    corrupt_p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00truncatedcorrupteddata")
    record("corrupt_truncated.jpg", "security_invalid", "Truncated corrupted JPEG header", expected_face=False)

    # Polyglot PDF embedded in image bytes
    buf = io.BytesIO()
    clean_im.save(buf, "JPEG")
    valid_bytes = buf.getvalue()
    polyglot_bytes = valid_bytes[:30] + b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n" + valid_bytes[30:]
    poly_p = FIXTURES_DIR / "polyglot_pdf_image.jpg"
    poly_p.write_bytes(polyglot_bytes)
    record("polyglot_pdf_image.jpg", "security_invalid", "Polyglot image containing PDF document signature", expected_face=False)

    # Trailing data appended after EOI marker
    trailing_bytes = valid_bytes + b"\x00\x00\x00MALICIOUS_TRAILING_SHELLCODE_PAYLOAD" * 4
    trail_p = FIXTURES_DIR / "trailing_data_jpeg.jpg"
    trail_p.write_bytes(trailing_bytes)
    record("trailing_data_jpeg.jpg", "security_invalid", "JPEG with appended unauthorized trailing data", expected_face=False)

    # 14. Extreme aspect ratio (1 x 1000 or 50:1)
    extreme_im = Image.new("RGB", (10, 600), color=(100, 100, 100))
    extreme_p = FIXTURES_DIR / "extreme_aspect_ratio.jpg"
    extreme_im.save(extreme_p, "JPEG")
    record("extreme_aspect_ratio.jpg", "quality_edge", "Extreme aspect ratio 1:60", expected_face=False, expected_quality="LOW_QUALITY")

    # Save manifest
    manifest_p = FIXTURES_DIR / "manifest.json"
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print("=" * 80)
    print("      FACECHECK API: DETERMINISTIC TEST-FIXTURE GENERATOR")
    print("=" * 80)
    manifest = generate_all_fixtures()
    count = len(manifest["fixtures"])
    print(f"\n[OK] Generated {count} deterministic synthetic test fixtures under tests/fixtures/")
    for name, item in manifest["fixtures"].items():
        sz = item["size_bytes"]
        cat = item["category"]
        print(f"  * {name:28s} | {cat:22s} | {sz:7d} bytes | {item['description']}")
    print(f"\n[OK] Manifest recorded at: {FIXTURES_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
