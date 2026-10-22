"""Unit tests for Step 2: Metadata and AI signature forensics."""

import io
from PIL import Image, PngImagePlugin
from app.services.metadata import inspect_metadata


def test_missing_exif_returns_weak_signal_score():
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    result = inspect_metadata(raw, img)
    assert result["score"] == 50
    assert "No camera EXIF" in result["detail"]
    assert result["tool_detected"] is None


def test_stable_diffusion_png_chunk_detected():
    img = Image.new("RGB", (100, 100), color=(120, 120, 120))
    png_info = PngImagePlugin.PngInfo()
    # Typical Automatic1111 parameter text
    sd_prompt = "A bank teller at a modern counter\nSteps: 25, Sampler: Euler a, CFG scale: 7, Seed: 1234567, Model: sd_xl_base_1.0.safetensors"
    png_info.add_text("parameters", sd_prompt)

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=png_info)
    raw = buf.getvalue()

    loaded_img = Image.open(io.BytesIO(raw))
    result = inspect_metadata(raw, loaded_img)

    assert result["score"] >= 80
    assert "AI signature detected" in result["detail"]
    assert "Stable Diffusion" in result["tool_detected"]


def test_midjourney_binary_flag_detected():
    img = Image.new("RGB", (100, 100), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    # Append Midjourney metadata flag into raw bytes
    raw = buf.getvalue() + b"\x00Midjourney prompt: portrait of executive --v 6 --ar 16:9\x00"

    result = inspect_metadata(raw, img)
    assert result["score"] >= 80
    assert result["tool_detected"] is not None
    assert "Midjourney" in result["tool_detected"]


def test_c2pa_algorithmic_media_detected():
    img = Image.new("RGB", (100, 100), color=(90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    # Simulate C2PA JUMBF algorithmic media claim
    raw = buf.getvalue() + b"c2pa.actions: c2pa.created with digitalSourceType: trainedAlgorithmicMedia claim_generator: Adobe Firefly"

    result = inspect_metadata(raw, img)
    assert result["score"] >= 80
    assert result["tool_detected"] is not None
    assert "Adobe Firefly" in result["tool_detected"] or "C2PA" in result["tool_detected"]
