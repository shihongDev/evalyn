"""
Tests for evalyn_sdk.storage.encryption.

Run with: uv run pytest tests/test_storage_encryption.py -v
"""

from __future__ import annotations

import base64

import pytest

from evalyn_sdk.storage.encryption import (
    EncryptionConfig,
    EncryptionResult,
    decrypt_dict,
    decrypt_text,
    derive_key,
    encrypt_dict,
    encrypt_text,
    generate_key,
    is_encrypted,
)


# ---------------------------------------------------------------------------
# generate_key
# ---------------------------------------------------------------------------


def test_generate_key_returns_base64():
    key = generate_key()
    raw = base64.b64decode(key)
    assert len(raw) == 32


def test_generate_key_unique():
    k1 = generate_key()
    k2 = generate_key()
    assert k1 != k2


# ---------------------------------------------------------------------------
# derive_key
# ---------------------------------------------------------------------------


def test_derive_key_deterministic():
    k1 = derive_key("password123", "salt1")
    k2 = derive_key("password123", "salt1")
    assert k1 == k2


def test_derive_key_different_password():
    k1 = derive_key("alpha")
    k2 = derive_key("beta")
    assert k1 != k2


def test_derive_key_different_salt():
    k1 = derive_key("password", "salt_a")
    k2 = derive_key("password", "salt_b")
    assert k1 != k2


def test_derive_key_returns_base64():
    key = derive_key("test")
    raw = base64.b64decode(key)
    assert len(raw) == 32


# ---------------------------------------------------------------------------
# encrypt_text / decrypt_text
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip():
    key = generate_key()
    original = "hello world"
    encrypted = encrypt_text(original, key)
    assert decrypt_text(encrypted, key) == original


def test_encrypt_text_different_key_different_output():
    k1 = generate_key()
    k2 = generate_key()
    encrypted1 = encrypt_text("same text", k1)
    encrypted2 = encrypt_text("same text", k2)
    assert encrypted1 != encrypted2


def test_encrypt_text_output_is_base64():
    key = generate_key()
    encrypted = encrypt_text("test data", key)
    # Should not raise
    base64.b64decode(encrypted)


def test_encrypt_decrypt_empty_text():
    key = generate_key()
    encrypted = encrypt_text("", key)
    assert decrypt_text(encrypted, key) == ""


def test_encrypt_decrypt_empty_key():
    encrypted = encrypt_text("hello", "")
    assert decrypt_text(encrypted, "") == "hello"


def test_encrypt_decrypt_unicode():
    key = generate_key()
    original = "cafe \u00e9\u00e8\u00ea - \u6d4b\u8bd5"
    encrypted = encrypt_text(original, key)
    assert decrypt_text(encrypted, key) == original


def test_encrypt_decrypt_long_text():
    key = generate_key()
    original = "x" * 10_000
    encrypted = encrypt_text(original, key)
    assert decrypt_text(encrypted, key) == original


# ---------------------------------------------------------------------------
# encrypt_dict / decrypt_dict
# ---------------------------------------------------------------------------


def test_encrypt_dict_encrypts_specified_fields():
    key = generate_key()
    data = {"name": "Alice", "secret": "s3cret", "count": 42}
    result = encrypt_dict(data, key, fields=["secret"])
    assert result["secret"].startswith("ENC:")
    assert result["name"] == "Alice"
    assert result["count"] == 42


def test_decrypt_dict_restores_values():
    key = generate_key()
    data = {"name": "Alice", "secret": "s3cret", "count": 42}
    encrypted = encrypt_dict(data, key, fields=["secret"])
    decrypted = decrypt_dict(encrypted, key)
    assert decrypted["secret"] == "s3cret"
    assert decrypted["name"] == "Alice"
    assert decrypted["count"] == 42


def test_encrypt_dict_multiple_fields():
    key = generate_key()
    data = {"a": "val_a", "b": "val_b", "c": "val_c"}
    result = encrypt_dict(data, key, fields=["a", "c"])
    assert result["a"].startswith("ENC:")
    assert result["b"] == "val_b"
    assert result["c"].startswith("ENC:")
    restored = decrypt_dict(result, key)
    assert restored == data


def test_encrypt_dict_no_fields():
    key = generate_key()
    data = {"x": "y"}
    result = encrypt_dict(data, key, fields=[])
    assert result == data


def test_encrypt_dict_none_fields():
    key = generate_key()
    data = {"x": "y"}
    result = encrypt_dict(data, key)
    assert result == data


def test_encrypt_dict_nonstring_preserved():
    key = generate_key()
    data = {"num": 123, "flag": True, "items": [1, 2, 3]}
    result = encrypt_dict(data, key, fields=["num", "flag", "items"])
    # Non-string fields should be unchanged
    assert result["num"] == 123
    assert result["flag"] is True
    assert result["items"] == [1, 2, 3]


def test_encrypt_dict_missing_field():
    key = generate_key()
    data = {"a": "val"}
    result = encrypt_dict(data, key, fields=["nonexistent"])
    assert result == data


def test_encrypt_dict_does_not_mutate_original():
    key = generate_key()
    data = {"secret": "hidden"}
    original_copy = dict(data)
    encrypt_dict(data, key, fields=["secret"])
    assert data == original_copy


# ---------------------------------------------------------------------------
# is_encrypted
# ---------------------------------------------------------------------------


def test_is_encrypted_true():
    assert is_encrypted("ENC:abc123") is True


def test_is_encrypted_false_plain():
    assert is_encrypted("plain text") is False


def test_is_encrypted_false_partial():
    assert is_encrypted("EN:abc") is False


def test_is_encrypted_false_non_string():
    assert is_encrypted(123) is False  # type: ignore[arg-type]


def test_is_encrypted_empty():
    assert is_encrypted("") is False


# ---------------------------------------------------------------------------
# EncryptionConfig
# ---------------------------------------------------------------------------


def test_encryption_config_defaults():
    cfg = EncryptionConfig()
    assert cfg.enabled is False
    assert cfg.algorithm == "fernet"
    assert cfg.key_source == "env"
    assert cfg.key_env_var == "EVALYN_ENCRYPTION_KEY"


def test_encryption_config_as_dict():
    cfg = EncryptionConfig(enabled=True, algorithm="xor_simple")
    d = cfg.as_dict()
    assert d["enabled"] is True
    assert d["algorithm"] == "xor_simple"


def test_encryption_config_from_dict():
    d = {"enabled": True, "algorithm": "xor_simple", "key_source": "file"}
    cfg = EncryptionConfig.from_dict(d)
    assert cfg.enabled is True
    assert cfg.algorithm == "xor_simple"
    assert cfg.key_source == "file"
    assert cfg.key_env_var == "EVALYN_ENCRYPTION_KEY"


def test_encryption_config_from_dict_empty():
    cfg = EncryptionConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.algorithm == "fernet"


def test_encryption_config_roundtrip():
    cfg = EncryptionConfig(enabled=True, key_source="config", key_env_var="MY_KEY")
    restored = EncryptionConfig.from_dict(cfg.as_dict())
    assert restored.enabled == cfg.enabled
    assert restored.key_source == cfg.key_source
    assert restored.key_env_var == cfg.key_env_var


# ---------------------------------------------------------------------------
# EncryptionResult
# ---------------------------------------------------------------------------


def test_encryption_result_defaults():
    r = EncryptionResult()
    assert r.original_size == 0
    assert r.success is True


def test_encryption_result_as_dict():
    r = EncryptionResult(original_size=100, encrypted_size=120, algorithm="xor_simple")
    d = r.as_dict()
    assert d["original_size"] == 100
    assert d["encrypted_size"] == 120
    assert d["algorithm"] == "xor_simple"
    assert d["success"] is True
