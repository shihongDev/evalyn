"""Tests for API key rotation support module."""

from __future__ import annotations

from evalyn_sdk.key_rotation import (
    KeyConfig,
    KeyRotationResult,
    RotationManager,
    format_rotation_status,
    load_keys_from_env,
    mask_key,
)


# --- mask_key ---


class TestMaskKey:
    def test_normal_key(self):
        # last 4 chars of "sk-abc123xyz9" are "xyz9"
        assert mask_key("sk-abc123xyz9") == "****xyz9"

    def test_short_key_4_chars(self):
        assert mask_key("abcd") == "****"

    def test_short_key_3_chars(self):
        assert mask_key("abc") == "***"

    def test_short_key_1_char(self):
        assert mask_key("x") == "*"

    def test_empty_key(self):
        assert mask_key("") == ""

    def test_exactly_5_chars(self):
        assert mask_key("12345") == "****2345"

    def test_long_key(self):
        key = "sk-proj-very-long-key-value-1234"
        result = mask_key(key)
        assert result == "****1234"
        assert "very" not in result


# --- KeyConfig ---


class TestKeyConfig:
    def test_create(self):
        config = KeyConfig(
            provider="openai", keys=["sk-key1", "sk-key2"]
        )
        assert config.provider == "openai"
        assert config.active_index == 0

    def test_as_dict_roundtrip(self):
        config = KeyConfig(
            provider="openai",
            keys=["sk-abc1234", "sk-xyz5678"],
            active_index=1,
        )
        d = config.as_dict()
        restored = KeyConfig.from_dict(d)
        assert restored.keys == config.keys
        assert restored.provider == config.provider
        assert restored.active_index == config.active_index

    def test_as_safe_dict_masks_keys(self):
        config = KeyConfig(
            provider="openai",
            keys=["sk-abc1234", "sk-xyz5678"],
            active_index=1,
        )
        d = config.as_safe_dict()
        assert d["provider"] == "openai"
        assert d["active_index"] == 1
        # Keys should be masked
        assert d["keys"][0] == "****1234"
        assert d["keys"][1] == "****5678"

    def test_as_safe_dict_never_leaks_full_keys(self):
        config = KeyConfig(
            provider="test", keys=["super-secret-key-abcd"]
        )
        d = config.as_safe_dict()
        assert "super-secret" not in str(d)
        assert "abcd" in d["keys"][0]

    def test_from_dict(self):
        d = {
            "provider": "anthropic",
            "keys": ["key-a", "key-b"],
            "active_index": 1,
        }
        config = KeyConfig.from_dict(d)
        assert config.provider == "anthropic"
        assert config.keys == ["key-a", "key-b"]
        assert config.active_index == 1

    def test_from_dict_defaults(self):
        d = {"provider": "test"}
        config = KeyConfig.from_dict(d)
        assert config.keys == []
        assert config.active_index == 0


# --- KeyRotationResult ---


class TestKeyRotationResult:
    def test_create(self):
        result = KeyRotationResult(
            provider="openai",
            old_key_suffix="1234",
            new_key_suffix="5678",
            success=True,
        )
        assert result.error == ""

    def test_as_dict(self):
        result = KeyRotationResult(
            provider="openai",
            old_key_suffix="1234",
            new_key_suffix="5678",
            success=True,
        )
        d = result.as_dict()
        assert d["provider"] == "openai"
        assert d["success"] is True
        assert d["error"] == ""

    def test_from_dict(self):
        d = {
            "provider": "openai",
            "old_key_suffix": "1234",
            "new_key_suffix": "5678",
            "success": False,
            "error": "failed",
        }
        result = KeyRotationResult.from_dict(d)
        assert result.success is False
        assert result.error == "failed"

    def test_roundtrip(self):
        original = KeyRotationResult(
            provider="test",
            old_key_suffix="aaaa",
            new_key_suffix="bbbb",
            success=True,
            error="",
        )
        restored = KeyRotationResult.from_dict(original.as_dict())
        assert restored.provider == original.provider
        assert restored.old_key_suffix == original.old_key_suffix
        assert restored.success == original.success


# --- RotationManager ---


class TestRotationManager:
    def test_empty_manager(self):
        mgr = RotationManager()
        assert mgr.list_providers() == []

    def test_register_provider(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["key-a", "key-b"])
        assert "openai" in mgr.list_providers()

    def test_get_active_key(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["key-a", "key-b"])
        assert mgr.get_active_key("openai") == "key-a"

    def test_get_active_key_unknown_provider(self):
        mgr = RotationManager()
        assert mgr.get_active_key("unknown") is None

    def test_get_active_key_empty_keys(self):
        mgr = RotationManager()
        mgr.register_provider("empty", [])
        assert mgr.get_active_key("empty") is None

    def test_rotate_basic(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["key-a", "key-b"])
        result = mgr.rotate("openai")
        assert result.success is True
        assert mgr.get_active_key("openai") == "key-b"

    def test_rotate_wraps_around(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["key-a", "key-b"])
        mgr.rotate("openai")  # a -> b
        result = mgr.rotate("openai")  # b -> a (wrap)
        assert result.success is True
        assert mgr.get_active_key("openai") == "key-a"

    def test_rotate_three_keys(self):
        mgr = RotationManager()
        mgr.register_provider("p", ["k1", "k2", "k3"])
        mgr.rotate("p")  # k1 -> k2
        assert mgr.get_active_key("p") == "k2"
        mgr.rotate("p")  # k2 -> k3
        assert mgr.get_active_key("p") == "k3"
        mgr.rotate("p")  # k3 -> k1
        assert mgr.get_active_key("p") == "k1"

    def test_rotate_unknown_provider(self):
        mgr = RotationManager()
        result = mgr.rotate("missing")
        assert result.success is False
        assert "not registered" in result.error

    def test_rotate_single_key_fails(self):
        mgr = RotationManager()
        mgr.register_provider("one", ["only-key"])
        result = mgr.rotate("one")
        assert result.success is False
        assert "Only one key" in result.error

    def test_rotate_empty_keys_fails(self):
        mgr = RotationManager()
        mgr.register_provider("empty", [])
        result = mgr.rotate("empty")
        assert result.success is False
        assert "No keys" in result.error

    def test_rotate_returns_suffixes(self):
        mgr = RotationManager()
        mgr.register_provider("p", ["sk-abc1234", "sk-xyz5678"])
        result = mgr.rotate("p")
        assert result.old_key_suffix == "1234"
        assert result.new_key_suffix == "5678"

    def test_handle_auth_error_rotates(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["key-a", "key-b"])
        result = mgr.handle_auth_error("openai")
        assert result.success is True
        assert mgr.get_active_key("openai") == "key-b"

    def test_handle_auth_error_unknown_provider(self):
        mgr = RotationManager()
        result = mgr.handle_auth_error("missing")
        assert result.success is False

    def test_list_providers_multiple(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["k1"])
        mgr.register_provider("anthropic", ["k2"])
        providers = mgr.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_get_key_status(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["sk-test1234", "sk-test5678"])
        status = mgr.get_key_status("openai")
        assert status["provider"] == "openai"
        assert status["num_keys"] == 2
        assert status["active_index"] == 0
        assert status["active_key_suffix"] == "1234"

    def test_get_key_status_after_rotation(self):
        mgr = RotationManager()
        mgr.register_provider("p", ["key-aaaa", "key-bbbb"])
        mgr.rotate("p")
        status = mgr.get_key_status("p")
        assert status["active_index"] == 1
        assert status["active_key_suffix"] == "bbbb"

    def test_get_key_status_unknown_provider(self):
        mgr = RotationManager()
        status = mgr.get_key_status("missing")
        assert status["num_keys"] == 0
        assert status["active_index"] == -1
        assert status["active_key_suffix"] == ""

    def test_reregister_provider_resets(self):
        mgr = RotationManager()
        mgr.register_provider("p", ["k1", "k2"])
        mgr.rotate("p")
        assert mgr.get_active_key("p") == "k2"
        mgr.register_provider("p", ["k3", "k4"])
        assert mgr.get_active_key("p") == "k3"


# --- load_keys_from_env ---


class TestLoadKeysFromEnv:
    def test_single_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
        keys = load_keys_from_env("openai")
        assert keys == ["sk-test1234"]

    def test_numbered_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-base")
        monkeypatch.setenv("OPENAI_API_KEY_1", "sk-first")
        monkeypatch.setenv("OPENAI_API_KEY_2", "sk-second")
        keys = load_keys_from_env("openai")
        assert keys == ["sk-base", "sk-first", "sk-second"]

    def test_no_keys_set(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)
        keys = load_keys_from_env("openai")
        assert keys == []

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-custom")
        keys = load_keys_from_env("openai", env_prefix="MY_CUSTOM_KEY")
        assert keys == ["sk-custom"]

    def test_custom_prefix_with_numbered(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-base")
        monkeypatch.setenv("MY_KEY_1", "sk-one")
        monkeypatch.setenv("MY_KEY_2", "sk-two")
        keys = load_keys_from_env("anything", env_prefix="MY_KEY")
        assert keys == ["sk-base", "sk-one", "sk-two"]

    def test_numbered_only_no_base(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY_1", "sk-one")
        monkeypatch.setenv("ANTHROPIC_API_KEY_2", "sk-two")
        keys = load_keys_from_env("anthropic")
        assert keys == ["sk-one", "sk-two"]

    def test_gap_in_numbering_stops(self, monkeypatch):
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        monkeypatch.setenv("TEST_API_KEY_1", "sk-one")
        monkeypatch.delenv("TEST_API_KEY_2", raising=False)
        monkeypatch.setenv("TEST_API_KEY_3", "sk-three")
        keys = load_keys_from_env("test")
        # Stops at gap (2 missing), so only picks up 1
        assert keys == ["sk-one"]

    def test_provider_name_uppercased(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
        keys = load_keys_from_env("gemini")
        assert keys == ["sk-gemini"]


# --- format_rotation_status ---


class TestFormatRotationStatus:
    def test_no_providers(self):
        mgr = RotationManager()
        text = format_rotation_status(mgr)
        assert "No providers registered" in text

    def test_single_provider(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["sk-test1234"])
        text = format_rotation_status(mgr)
        assert "openai" in text
        assert "Keys: 1" in text
        assert "****1234" in text

    def test_multiple_providers(self):
        mgr = RotationManager()
        mgr.register_provider("openai", ["sk-aaaa1111"])
        mgr.register_provider("anthropic", ["sk-bbbb2222"])
        text = format_rotation_status(mgr)
        assert "openai" in text
        assert "anthropic" in text

    def test_header_present(self):
        mgr = RotationManager()
        mgr.register_provider("test", ["key-abcd"])
        text = format_rotation_status(mgr)
        assert "Key Rotation Status" in text
