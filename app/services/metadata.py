"""EXIF and C2PA Content Credentials inspection for AI generation signatures."""

from __future__ import annotations

import io
import re
from typing import Dict, Any, Optional, Tuple
import exifread
from PIL import Image

AI_SIGNATURE_PATTERNS = [
    (r"steps:\s*\d+,\s*sampler:\s*[\w\+]+,\s*cfg\s*scale:\s*[\d\.]+", "Stable Diffusion Parameters"),
    (r"model:\s*[\w\-_\.]+(\.safetensors|\.ckpt)", "Stable Diffusion Model Hash"),
    (r"negative\s*prompt:\s*", "Stable Diffusion Negative Prompt"),
    (r"\"class_type\":\s*\"(KSampler|CheckpointLoaderSimple|VAEDecode)\"", "ComfyUI Workflow Metadata"),
    (r"invokeai", "InvokeAI Metadata"),
    (r"--v\s+(4|5|5\.1|5\.2|6|6\.1)", "Midjourney Version Flag"),
    (r"--ar\s+\d+:\d+", "Midjourney Aspect Ratio Flag"),
    (r"midjourney", "Midjourney Tag"),
    (r"dall-e|dalle", "DALL-E Signature"),
    (r"adobe\s*firefly", "Adobe Firefly"),
    (r"trainedalgorithmicmedia", "IPTC Algorithmic/Generative Media"),
    (r"c2pa\.actions.*c2pa\.created", "C2PA Synthetic Generation Action"),
    (r"prompt:\s*[\w\s]{10,}", "Synthetic Prompt Parameter"),
]


def extract_c2pa(raw_bytes: bytes) -> Tuple[bool, Optional[str], bool]:
    """Inspect image bytes offline for C2PA manifests.

    Returns:
        (c2pa_present, claim_generator, is_ai_generated)
    """
    try:
        import c2pa
        reader = c2pa.Reader.from_stream("image/jpeg", io.BytesIO(raw_bytes))
        manifest = reader.json()
        if manifest:
            is_ai = "trainedAlgorithmicMedia" in manifest or "c2pa.created" in manifest
            claim_gen = reader.active_manifest().get("claim_generator")
            return True, claim_gen, is_ai
    except Exception:
        pass

    # Offline JUMBF binary inspection
    lower_bytes = raw_bytes[:100000].lower() + raw_bytes[-50000:].lower() if len(raw_bytes) > 150000 else raw_bytes.lower()
    if b"c2pa" in lower_bytes or b"jumbf" in lower_bytes:
        is_ai = (
            b"trainedalgorithmicmedia" in lower_bytes
            or b"c2pa.created" in lower_bytes
            or b"generative" in lower_bytes
        )
        gen_match = re.search(rb"claim_generator[\"':\s]+([a-zA-Z0-9_\-\.\s]{3,40})", lower_bytes)
        gen_str = gen_match.group(1).decode("utf-8", errors="ignore") if gen_match else "C2PA Manifest"
        return True, gen_str, is_ai

    return False, None, False


def inspect_metadata(raw_bytes: bytes, pil_image: Image.Image) -> Dict[str, Any]:
    """Step 2: Read EXIF and C2PA; compute metadata score (1-100) and detail string.

    Returns:
        Dictionary with 'score' (1-100), 'detail' (str), 'tool_detected' (Optional[str]).
    """
    text_blobs = []

    # Check PNG info
    if hasattr(pil_image, "info"):
        for k, v in pil_image.info.items():
            if isinstance(v, str):
                text_blobs.append(f"{k}: {v}")

    # Check EXIF
    has_camera_exif = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None

    try:
        tags = exifread.process_file(io.BytesIO(raw_bytes), details=False)
        if tags:
            make_tag = tags.get("Image Make")
            model_tag = tags.get("Image Model")
            if make_tag or model_tag:
                has_camera_exif = True
                camera_make = str(make_tag).strip() if make_tag else ""
                camera_model = str(model_tag).strip() if model_tag else ""

            for t_name, t_val in tags.items():
                text_blobs.append(f"{t_name}: {t_val}")
    except Exception:
        pass

    # Check for AI signatures in collected text
    joined_text = "\n".join(text_blobs)
    ai_found = False
    detected_tool: Optional[str] = None

    for pattern, tool_name in AI_SIGNATURE_PATTERNS:
        if re.search(pattern, joined_text, re.IGNORECASE):
            ai_found = True
            detected_tool = tool_name
            break

    # Also search binary headers if not in parsed strings
    if not ai_found:
        header_sample = raw_bytes[:65536].decode("latin-1", errors="ignore")
        for pattern, tool_name in AI_SIGNATURE_PATTERNS:
            if re.search(pattern, header_sample, re.IGNORECASE):
                ai_found = True
                detected_tool = tool_name
                break

    # Check C2PA Content Credentials
    c2pa_present, c2pa_gen, c2pa_ai = extract_c2pa(raw_bytes)
    if c2pa_ai:
        ai_found = True
        detected_tool = f"C2PA Algorithmic Media ({c2pa_gen or 'Unknown'})"

    # Score calculation (1 to 100)
    if ai_found:
        score = 85
        detail = f"AI signature detected ({detected_tool})"
    elif has_camera_exif:
        score = 15
        cam_info = f"{camera_make} {camera_model}".strip()
        detail = f"Verified camera hardware EXIF ({cam_info})"
    else:
        # Missing EXIF is a weak signal, common online
        score = 50
        detail = "No camera EXIF (weak signal, metadata stripped or absent)"

    return {
        "score": score,
        "detail": detail,
        "tool_detected": detected_tool,
        "has_camera_exif": has_camera_exif,
    }
