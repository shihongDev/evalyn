"""
Encrypted storage: at-rest encryption for sensitive trace and evaluation data.

Provides XOR-based encryption (no external dependencies) for string fields
in dicts, with key generation and derivation utilities.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EncryptionConfig:
    """Configuration for at-rest encryption."""

    enabled: bool = False
    algorithm: str = "fernet"  # "fernet" / "xor_simple"
    key_source: str = "env"  # "env" / "file" / "config"
    key_env_var: str = "EVALYN_ENCRYPTION_KEY"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm,
            "key_source": self.key_source,
            "key_env_var": self.key_env_var,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptionConfig:
        return cls(
            enabled=data.get("enabled", False),
            algorithm=data.get("algorithm", "fernet"),
            key_source=data.get("key_source", "env"),
            key_env_var=data.get("key_env_var", "EVALYN_ENCRYPTION_KEY"),
        )


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""

    original_size: int = 0
    encrypted_size: int = 0
    algorithm: str = ""
    success: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_size": self.original_size,
            "encrypted_size": self.encrypted_size,
            "algorithm": self.algorithm,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Key Management
# ---------------------------------------------------------------------------


def generate_key() -> str:
    """Generate a random 32-byte key, base64-encoded."""
    raw = secrets.token_bytes(32)
    return base64.b64encode(raw).decode("utf-8")


def derive_key(password: str, salt: str = "evalyn") -> str:
    """Derive a key from a password using iterated SHA-256.

    Hashes password+salt through 100_000 rounds of SHA-256
    (PBKDF2-like stretching without external deps). Returns base64-encoded.
    """
    current = (password + salt).encode("utf-8")
    for _ in range(100_000):
        current = hashlib.sha256(current).digest()
    return base64.b64encode(current).decode("utf-8")


# ---------------------------------------------------------------------------
# XOR Encryption
# ---------------------------------------------------------------------------


def encrypt_text(text: str, key: str) -> str:
    """XOR each byte of text with cycling key bytes. Return base64-encoded."""
    text_bytes = text.encode("utf-8")
    key_bytes = key.encode("utf-8")
    if not key_bytes:
        return base64.b64encode(text_bytes).decode("utf-8")
    encrypted = bytes(
        tb ^ key_bytes[i % len(key_bytes)] for i, tb in enumerate(text_bytes)
    )
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_text(encrypted: str, key: str) -> str:
    """Reverse of encrypt_text: base64-decode then XOR with cycling key."""
    encrypted_bytes = base64.b64decode(encrypted)
    key_bytes = key.encode("utf-8")
    if not key_bytes:
        return encrypted_bytes.decode("utf-8")
    decrypted = bytes(
        eb ^ key_bytes[i % len(key_bytes)] for i, eb in enumerate(encrypted_bytes)
    )
    return decrypted.decode("utf-8")


# ---------------------------------------------------------------------------
# Dict-level Encryption
# ---------------------------------------------------------------------------


def is_encrypted(value: str) -> bool:
    """Check if a string value is encrypted (starts with 'ENC:')."""
    return isinstance(value, str) and value.startswith("ENC:")


def encrypt_dict(
    data: dict[str, Any], key: str, fields: list[str] | None = None
) -> dict[str, Any]:
    """Encrypt specified string fields in a dict.

    Returns a new dict with encrypted values prefixed with 'ENC:'.
    Non-string fields and fields not in the list are preserved unchanged.
    If fields is empty or None, no fields are encrypted.
    """
    if fields is None:
        fields = []
    result = dict(data)
    for f in fields:
        if f in result and isinstance(result[f], str):
            result[f] = "ENC:" + encrypt_text(result[f], key)
    return result


def decrypt_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Decrypt all fields that start with 'ENC:'.

    Returns a new dict with decrypted values.
    """
    result = dict(data)
    for f, v in result.items():
        if isinstance(v, str) and is_encrypted(v):
            result[f] = decrypt_text(v[4:], key)
    return result
