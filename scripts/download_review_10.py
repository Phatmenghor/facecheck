"""Download exactly 10 curated test images from the internet for KYC biometric review.

Fetches 10 high-resolution, diverse biometric review images covering:
1. Bona fide clean portrait reference (Barack Obama 2009 Official Headshot)
2. Bona fide genuine match pair (Barack Obama Alternate Official Headshot)
3. Bona fide impostor probe (Steve Jobs 2010 Headshot)
4. Bona fide clean female portrait (Angela Merkel 2019 Headshot)
5. Multi-face group photograph (Street market crowd, >1 face policy flag)
6. No-face landscape photograph (Mount Rainier scenery, 0 face rejection)
7. Defocus/motion blur portrait (Gabriel motion blur, Laplacian blur rejection)
8. Severely underexposed low-key portrait (Young man low-key, Luminance rejection)
9. Presentation attack halftone print artifact (Walter Wyman halftone print attack)
10. Presentation attack screen moiré artifact (Liz Truss video conference screen replay)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_review_10")

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = BASE_DIR / "assets" / "review_10"
IMAGES_DIR = REVIEW_DIR / "images"
MANIFEST_PATH = REVIEW_DIR / "manifest.json"

CURATED_ITEMS = [
    {
        "id": "IMG_01",
        "filename": "img_01_barack_obama_ref.jpg",
        "title": "Barack Obama 2009 Official Presidential Portrait (Primary Reference)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e9/Official_portrait_of_Barack_Obama.jpg",
        "source": "Wikimedia Commons",
        "license": "Public Domain (Official White House Photo by Pete Souza)",
        "target_category": "BONA_FIDE_CLEAN",
        "scenario": "Reference KYC Portrait",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "LIVE",
        "notes": "Primary reference portrait for 1:1 facial recognition baseline.",
    },
    {
        "id": "IMG_02",
        "filename": "img_02_barack_obama_match.jpg",
        "title": "Barack Obama Official Portrait Variant (Genuine 1:1 Match Target)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c6/Official_portrait_of_Barack_Obama-2.jpg",
        "source": "Wikimedia Commons",
        "license": "Public Domain (Official White House Photo by Pete Souza)",
        "target_category": "BONA_FIDE_CLEAN",
        "scenario": "Genuine Match Pair (Same Person)",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "LIVE",
        "notes": "Secondary headshot of the same person; genuine 1:1 match target (similarity > 0.363).",
    },
    {
        "id": "IMG_03",
        "filename": "img_03_steve_jobs_impostor.jpg",
        "title": "Steve Jobs Headshot 2010 (Impostor Cross-Identity Probe)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f5/Steve_Jobs_Headshot_2010-CROP2.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY 2.0 (Bernard Gotfryd / edit)",
        "target_category": "BONA_FIDE_CLEAN",
        "scenario": "Impostor Match Pair (Different Person)",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "LIVE",
        "notes": "Different identity high-quality portrait; probe for 1:1 match impostor rejection (similarity < 0.363).",
    },
    {
        "id": "IMG_04",
        "filename": "img_04_angela_merkel_clean.jpg",
        "title": "Angela Merkel Portrait 2019 (Female Bona Fide)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0f/Angela_Merkel_2019_cropped.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 3.0 de (Bundesarchiv, B 145 Bild-F089033-0004)",
        "target_category": "BONA_FIDE_CLEAN",
        "scenario": "Female Bona Fide Portrait",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "LIVE",
        "notes": "Clean studio-lit portrait of female subject; evaluates demographic diversity and high-quality baseline.",
    },
    {
        "id": "IMG_05",
        "filename": "img_05_multiface_crowd.jpg",
        "title": "Street Market Crowd (Multi-Face Scene)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/34/Crowds_at_Walking_Street_Market_on_Th_Talad_Gao_-_Lampang_-_Thailand_%2835175487956%29.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 2.0 (BerryJ)",
        "target_category": "MULTI_FACE",
        "scenario": "Multi-Face Policy Exception",
        "expected_faces": -1,  # >1 faces
        "expected_quality": "WARNING",
        "expected_pad": "LIVE",
        "notes": "Multiple individuals in frame; triggers bank KYC policy requirement for single subject capture.",
    },
    {
        "id": "IMG_06",
        "filename": "img_06_noface_nature.jpg",
        "title": "Mount Rainier National Park (No-Face Scene)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mount_Rainier_from_Sourdough_Ridge_Trail%2C_Mount_Rainier_National_Park%2C_Ashford%2C_WA.jpg",
        "source": "Wikimedia Commons",
        "license": "CC0 / Public Domain",
        "target_category": "NO_FACE",
        "scenario": "Zero-Face Exception",
        "expected_faces": 0,
        "expected_quality": "FAIL",
        "expected_pad": "N/A",
        "notes": "Landscape scenery without human subjects; tests zero-face rejection gate.",
    },
    {
        "id": "IMG_07",
        "filename": "img_07_defocus_blurry.jpg",
        "title": "Gabriel Portrait with Motion Blur (Defocus / Motion Blur)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2a/Gabriel_portrait_with_motion_blur%2C_Alameda_Afonso_Henriques%2C_Lisbon%2C_Portugal_%28approx._GPS_location%29_julesvernex2.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 4.0 (julesvernex2)",
        "target_category": "BLURRY",
        "scenario": "Severe Motion Blur Quality Failure",
        "expected_faces": 1,
        "expected_quality": "FAIL",
        "expected_pad": "LIVE",
        "notes": "Natural optical/motion blur; tests Laplacian variance rejection (< 50.0 threshold).",
    },
    {
        "id": "IMG_08",
        "filename": "img_08_dark_underexposed.jpg",
        "title": "Low-Key Young Man Portrait (Underexposed Lighting)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5d/Low-key_photography_-_portrait_of_a_young_man.jpg",
        "source": "Wikimedia Commons",
        "license": "CC BY-SA 4.0 (Radosław Botev)",
        "target_category": "UNDEREXPOSED",
        "scenario": "Severe Underexposure Quality Failure",
        "expected_faces": 1,
        "expected_quality": "FAIL",
        "expected_pad": "LIVE",
        "notes": "Severely underexposed low-key portrait; tests mean luminance rejection (< 25.0 threshold).",
    },
    {
        "id": "IMG_09",
        "filename": "img_09_print_attack_halftone.jpg",
        "title": "Historical Halftone Print Portrait (Print Attack Artifact)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Walter_Wyman%2C_photograph_by_Stalee._halftone.jpg",
        "source": "Wikimedia Commons",
        "license": "Public Domain (Stalee / NLM)",
        "target_category": "PRINT_ATTACK",
        "scenario": "Presentation Attack - Halftone Paper Print",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "SPOOF",
        "notes": "Authentic halftone ink print dot structure; tests Presentation Attack Detection for paper prints.",
    },
    {
        "id": "IMG_10",
        "filename": "img_10_screen_replay_moire.jpg",
        "title": "Liz Truss Video Call Screen Display (Screen Replay Artifact)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Prime_Minister_Liz_Truss_G7_Ukraine_Video_Call_%2852419927357%29.jpg",
        "source": "Wikimedia Commons",
        "license": "Open Government Licence v3.0 (Simon Dawson / No 10 Downing Street)",
        "target_category": "SCREEN_REPLAY",
        "scenario": "Presentation Attack - Video Screen Replay",
        "expected_faces": 1,
        "expected_quality": "PASS",
        "expected_pad": "SPOOF",
        "notes": "Display screen capture with digital monitor refresh and moiré raster; tests Presentation Attack Detection for digital screens.",
    },
]


def download_curated_10() -> List[Dict[str, Any]]:
    """Download the 10 curated test images from the internet into assets/review_10/images."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "FaceCheck-Bot/1.0 (https://github.com/cpbank/facecheck; biometric-qa@cpbank.com) httpx/0.28.1"
    }

    client = httpx.Client(verify=False, headers=headers, timeout=30.0, follow_redirects=True)
    results = []

    logger.info("Downloading 10 curated review images...")

    for item in CURATED_ITEMS:
        filename = item["filename"]
        target_path = IMAGES_DIR / filename
        url = item["url"]

        logger.info(f"Downloading [{item['id']}] {filename} from {url[:60]}...")
        try:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content

            # Normalize image to RGB JPEG to ensure universal OpenCV compatibility
            pil_img = Image.open(io.BytesIO(content))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Constrain extreme sizes (e.g. 7000px down to max 2400px) for efficient CPU inference and UI display
            max_dimension = 2400
            if max(pil_img.size) > max_dimension:
                pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # Save normalized image
            pil_img.save(target_path, format="JPEG", quality=95)

            # Re-read from disk to calculate exact checksum and dimensions
            disk_bytes = target_path.read_bytes()
            sha256 = hashlib.sha256(disk_bytes).hexdigest()
            width, height = pil_img.size

            meta = dict(item)
            meta["file_path"] = str(target_path.relative_to(BASE_DIR))
            meta["file_size_bytes"] = len(disk_bytes)
            meta["sha256"] = sha256
            meta["width"] = width
            meta["height"] = height
            meta["download_status"] = "SUCCESS"
            results.append(meta)
            logger.info(f"  -> Saved {filename} ({len(disk_bytes)} bytes, {width}x{height}, sha256={sha256[:12]}...)")
        except Exception as exc:
            logger.error(f"  -> FAILED to download {filename}: {exc}")
            meta = dict(item)
            meta["download_status"] = f"ERROR: {exc}"
            results.append(meta)

    manifest_data = {
        "target_samples": 10,
        "downloaded_samples": len([r for r in results if r.get("download_status") == "SUCCESS"]),
        "items": results,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Manifest written to {MANIFEST_PATH}")
    return results


if __name__ == "__main__":
    download_curated_10()
