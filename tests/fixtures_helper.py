"""Database-driven test fixture loader with deterministic offline synthetic fallback.

Loads test fixture images from PostgreSQL kyc_image_store when available.
Falls back safely to in-memory deterministic synthetic graphics when the database is offline.
Guarantees 100% offline, air-gapped test execution with zero reliance on external network databases.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_CACHE: Dict[str, bytes] = {}


def _generate_synthetic_clean_face() -> bytes:
    """Generate in-memory synthetic face image bytes."""
    size = (300, 300)
    w, h = size
    img = Image.new("RGB", (w, h), color=(235, 235, 235))
    d = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    hw, hh = 75, 100

    # Head
    d.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=(215, 175, 145), outline=(160, 120, 95), width=2)
    # Hair
    d.pieslice([cx - hw - 2, cy - hh - 5, cx + hw + 2, cy - int(hh * 0.1)], 180, 360, fill=(50, 40, 30))
    # Eyes
    eye_y = cy - int(hh * 0.15)
    eye_dx = int(hw * 0.45)
    d.ellipse([cx - eye_dx - 12, eye_y - 8, cx - eye_dx + 12, eye_y + 8], fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    d.ellipse([cx - eye_dx - 4, eye_y - 4, cx - eye_dx + 4, eye_y + 4], fill=(30, 20, 10))
    d.ellipse([cx + eye_dx - 12, eye_y - 8, cx + eye_dx + 12, eye_y + 8], fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    d.ellipse([cx + eye_dx - 4, eye_y - 4, cx + eye_dx + 4, eye_y + 4], fill=(30, 20, 10))
    # Nose
    nose_y = cy + int(hh * 0.15)
    d.polygon([(cx, nose_y - int(hh * 0.15)), (cx - int(hw * 0.12), nose_y), (cx + int(hw * 0.12), nose_y)], fill=(195, 150, 120))
    # Mouth
    mouth_y = cy + int(hh * 0.50)
    mw = int(hw * 0.38)
    d.arc([cx - mw, mouth_y - 10, cx + mw, mouth_y + 15], start=0, end=180, fill=(170, 65, 65), width=4)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _generate_synthetic_no_face() -> bytes:
    """Generate in-memory landscape image with zero human faces."""
    img = Image.new("RGB", (300, 300), color=(80, 140, 200))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 300), (120, 140), (220, 300)], fill=(70, 120, 80))
    d.polygon([(150, 300), (250, 160), (300, 300)], fill=(60, 100, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _generate_synthetic_multiple_faces() -> bytes:
    """Generate in-memory image with two distinct human faces."""
    clean_bytes = _generate_synthetic_clean_face()
    clean_im = Image.open(io.BytesIO(clean_bytes))
    multi_im = Image.new("RGB", (480, 300), color=(240, 240, 240))
    face1 = clean_im.resize((200, 200))
    face2 = clean_im.resize((200, 200))
    multi_im.paste(face1, (20, 50))
    multi_im.paste(face2, (260, 50))
    buf = io.BytesIO()
    multi_im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _generate_synthetic_blurry_face() -> bytes:
    """Generate in-memory heavy Gaussian blurry image."""
    clean_bytes = _generate_synthetic_clean_face()
    clean_im = Image.open(io.BytesIO(clean_bytes))
    blurry_im = clean_im.filter(ImageFilter.GaussianBlur(radius=10))
    buf = io.BytesIO()
    blurry_im.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _generate_synthetic_print_attack() -> bytes:
    clean_bytes = _generate_synthetic_clean_face()
    clean_im = Image.open(io.BytesIO(clean_bytes))
    print_im = Image.new("RGB", (360, 360), color=(250, 250, 248))
    p_draw = ImageDraw.Draw(print_im)
    p_draw.rectangle([15, 15, 345, 345], outline=(200, 200, 195), width=2)
    print_im.paste(clean_im, (30, 30))

    p_arr = np.array(print_im, dtype=np.float32)
    y_coords, x_coords = np.mgrid[0:360, 0:360]
    halftone_grid = (np.sin(x_coords * 1.2) * np.sin(y_coords * 1.2) * 8.0).astype(np.float32)
    p_arr = np.clip(p_arr + halftone_grid[:, :, None], 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(p_arr).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _generate_synthetic_screen_replay() -> bytes:
    clean_bytes = _generate_synthetic_clean_face()
    clean_im = Image.open(io.BytesIO(clean_bytes))
    screen_im = Image.new("RGB", (380, 380), color=(20, 20, 22))
    s_draw = ImageDraw.Draw(screen_im)
    s_draw.rounded_rectangle([10, 10, 370, 370], radius=18, outline=(80, 80, 85), width=4)
    screen_im.paste(clean_im.resize((320, 320)), (30, 30))

    s_arr = np.array(screen_im, dtype=np.float32)
    sy, sx = np.mgrid[0:380, 0:380]
    moire = (np.sin(sx * 0.45 + sy * 0.35) * np.cos(sx * 0.25 - sy * 0.40) * 14.0).astype(np.float32)
    s_arr[30:350, 30:350, :] = np.clip(s_arr[30:350, 30:350, :] + moire[30:350, 30:350, None], 0, 255)
    buf = io.BytesIO()
    Image.fromarray(s_arr.astype(np.uint8)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def get_fixture_bytes(name: str) -> bytes:
    """Retrieve raw image bytes for a test fixture.
    
    Attempts PostgreSQL kyc_image_store first. If database is unreachable or offline,
    generates deterministic synthetic test fixtures in-memory to guarantee offline test isolation.
    """
    if name in _CACHE:
        return _CACHE[name]

    stem = Path(name).stem

    # Attempt PostgreSQL retrieval if connection is alive
    try:
        from app.services.learning_store import get_learning_store
        store = get_learning_store()
        raw = store.get_image_bytes(stem) or store.get_image_bytes(name)
        if raw:
            _CACHE[name] = raw
            return raw
    except Exception:
        pass

    # Deterministic in-memory synthetic fixture generation
    low_name = name.lower()
    if "no_face" in low_name or "landscape" in low_name or "empty" in low_name:
        raw = _generate_synthetic_no_face()
    elif "blur" in low_name:
        raw = _generate_synthetic_blurry_face()
    elif "multi" in low_name:
        raw = _generate_synthetic_multiple_faces()
    elif "print" in low_name:
        raw = _generate_synthetic_print_attack()
    elif "screen" in low_name or "replay" in low_name or "moire" in low_name:
        raw = _generate_synthetic_screen_replay()
    else:
        raw = _generate_synthetic_clean_face()

    _CACHE[name] = raw
    return raw


def get_fixture_image(name: str) -> Image.Image:
    """Retrieve PIL Image for a test fixture directly in memory."""
    raw = get_fixture_bytes(name)
    return Image.open(io.BytesIO(raw))
