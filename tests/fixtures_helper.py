"""Database-driven test fixture loader.

Loads test fixture images directly from PostgreSQL kyc_image_store.
Guarantees ZERO image files on local filesystem.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
from app.services.learning_store import get_learning_store

_CACHE: dict[str, bytes] = {}


def get_fixture_bytes(name: str) -> bytes:
    """Retrieve raw image bytes for a test fixture directly from PostgreSQL kyc_image_store."""
    if name in _CACHE:
        return _CACHE[name]

    stem = Path(name).stem
    store = get_learning_store()
    raw = store.get_image_bytes(stem)
    if not raw:
        raw = store.get_image_bytes(name)

    if raw:
        _CACHE[name] = raw
        return raw

    raise FileNotFoundError(f"Fixture '{name}' not found in PostgreSQL kyc_image_store")


def get_fixture_image(name: str) -> Image.Image:
    """Retrieve PIL Image for a test fixture directly from PostgreSQL kyc_image_store."""
    raw = get_fixture_bytes(name)
    return Image.open(io.BytesIO(raw))
