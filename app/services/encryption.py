"""Optional biometric encryption service (AES-256-GCM) with metadata stripping."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional
from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config import get_config


class BiometricEncryption:
    """AES-256-GCM encryption for saving images only when explicitly configured."""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or self._resolve_or_create_key()
        self._aesgcm = AESGCM(self._key)

    @classmethod
    def _resolve_or_create_key(cls) -> bytes:
        config = get_config()
        env_key = os.getenv(config.storage.encryption_key_env)
        if env_key:
            if len(env_key) == 64:
                return bytes.fromhex(env_key)
            elif len(env_key) == 32:
                return env_key.encode("utf-8")

        key_file = Path(config.storage.encryption_key_file)
        if key_file.is_file():
            content = key_file.read_bytes().strip()
            if len(content) == 32:
                return content
            elif len(content) == 64:
                return bytes.fromhex(content.decode("utf-8"))

        key_file.parent.mkdir(parents=True, exist_ok=True)
        new_key = AESGCM.generate_key(bit_length=256)
        key_file.write_bytes(new_key)
        return new_key

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt_bytes(self, payload: bytes) -> bytes:
        if len(payload) < 28:
            raise ValueError("Encrypted payload too short")
        nonce = payload[:12]
        ciphertext = payload[12:]
        return self._aesgcm.decrypt(nonce, ciphertext, None)

    def save_encrypted(
        self,
        raw_bytes: bytes,
        request_id: str,
        clean_image: Optional[Image.Image] = None,
    ) -> Optional[Path]:
        """Save encrypted bytes if configured, otherwise discard.

        If clean_image is provided, strips all metadata and re-encodes to a clean PNG
        prior to encryption, protecting privacy and preventing polyglot persistence.
        """
        config = get_config()
        if not config.storage.save_encrypted_images:
            return None  # Discarded immediately in memory

        if clean_image is not None:
            # Re-encode clean sanitized image with zero EXIF or device metadata
            buffer = io.BytesIO()
            clean_image.save(buffer, format="PNG", optimize=True)
            payload_to_encrypt = buffer.getvalue()
        else:
            payload_to_encrypt = raw_bytes

        dest_dir = Path(config.storage.encrypted_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / f"{request_id}.enc"
        file_path.write_bytes(self.encrypt_bytes(payload_to_encrypt))
        return file_path


_encryption_singleton: Optional[BiometricEncryption] = None


def get_encryption() -> BiometricEncryption:
    global _encryption_singleton
    if _encryption_singleton is None:
        _encryption_singleton = BiometricEncryption()
    return _encryption_singleton
