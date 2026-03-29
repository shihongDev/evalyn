"""Tests for the secrets backend integration module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.secrets_backend import (
    SecretReference,
    SecretResult,
    SecretsBackend,
    EnvBackend,
    DictBackend,
    SecretsManager,
    parse_secret_reference,
    create_default_manager,
    format_secrets_status,
)


# ---------------------------------------------------------------------------
# SecretReference
# ---------------------------------------------------------------------------


class TestSecretReference:
    def test_basic_construction(self):
        ref = SecretReference(backend="env", path="MY_KEY")
        assert ref.backend == "env"
        assert ref.path == "MY_KEY"
        assert ref.key_field == ""
        assert ref.version == "latest"

    def test_full_construction(self):
        ref = SecretReference(
            backend="aws", path="arn:aws:secret:123", key_field="api_key", version="2"
        )
        assert ref.backend == "aws"
        assert ref.key_field == "api_key"
        assert ref.version == "2"

    def test_as_dict(self):
        ref = SecretReference(backend="gcp", path="projects/x/secrets/y")
        d = ref.as_dict()
        assert d["backend"] == "gcp"
        assert d["path"] == "projects/x/secrets/y"
        assert d["key_field"] == ""
        assert d["version"] == "latest"

    def test_from_dict_minimal(self):
        ref = SecretReference.from_dict({"backend": "vault", "path": "secret/data/k"})
        assert ref.backend == "vault"
        assert ref.path == "secret/data/k"
        assert ref.key_field == ""
        assert ref.version == "latest"

    def test_from_dict_full(self):
        ref = SecretReference.from_dict({
            "backend": "aws",
            "path": "my-arn",
            "key_field": "token",
            "version": "3",
        })
        assert ref.key_field == "token"
        assert ref.version == "3"

    def test_roundtrip(self):
        original = SecretReference(
            backend="gcp", path="p/s", key_field="f", version="5"
        )
        restored = SecretReference.from_dict(original.as_dict())
        assert restored.backend == original.backend
        assert restored.path == original.path
        assert restored.key_field == original.key_field
        assert restored.version == original.version


# ---------------------------------------------------------------------------
# SecretResult
# ---------------------------------------------------------------------------


class TestSecretResult:
    def test_basic_construction(self):
        res = SecretResult(value="sk-abc123xyz", source="env")
        assert res.value == "sk-abc123xyz"
        assert res.source == "env"
        assert res.cached is False
        assert res.error == ""

    def test_as_dict_roundtrip(self):
        res = SecretResult(value="sk-abc123xyz", source="env")
        d = res.as_dict()
        restored = SecretResult.from_dict(d)
        assert restored.value == res.value
        assert restored.source == res.source

    def test_as_safe_dict_masks_value(self):
        res = SecretResult(value="sk-abc123xyz", source="env")
        d = res.as_safe_dict()
        assert d["value"] == "***3xyz"
        assert "sk-abc" not in d["value"]

    def test_as_safe_dict_masks_short_value(self):
        res = SecretResult(value="ab", source="env")
        d = res.as_safe_dict()
        assert d["value"] == "***"

    def test_as_safe_dict_masks_exactly_four(self):
        res = SecretResult(value="abcd", source="env")
        d = res.as_safe_dict()
        assert d["value"] == "***abcd"

    def test_as_dict_includes_metadata(self):
        res = SecretResult(value="secret", source="dict", cached=True, error="oops")
        d = res.as_dict()
        assert d["source"] == "dict"
        assert d["cached"] is True
        assert d["error"] == "oops"

    def test_from_dict(self):
        res = SecretResult.from_dict({
            "value": "val",
            "source": "env",
            "cached": True,
            "error": "",
        })
        assert res.value == "val"
        assert res.cached is True

    def test_from_dict_defaults(self):
        res = SecretResult.from_dict({})
        assert res.value == ""
        assert res.source == ""
        assert res.cached is False
        assert res.error == ""

    def test_error_result(self):
        res = SecretResult(value="", source="env", error="not found")
        assert res.error == "not found"


# ---------------------------------------------------------------------------
# EnvBackend
# ---------------------------------------------------------------------------


class TestEnvBackend:
    def test_name(self):
        assert EnvBackend().name == "env"

    def test_resolve_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_KEY", "my-api-key-1234")
        backend = EnvBackend()
        ref = SecretReference(backend="env", path="TEST_SECRET_KEY")
        result = backend.resolve(ref)
        assert result.value == "my-api-key-1234"
        assert result.source == "env"
        assert result.error == ""

    def test_resolve_missing_var(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        backend = EnvBackend()
        ref = SecretReference(backend="env", path="NONEXISTENT_VAR_12345")
        result = backend.resolve(ref)
        assert result.value == ""
        assert "not set" in result.error

    def test_resolve_empty_var(self, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR", "")
        backend = EnvBackend()
        ref = SecretReference(backend="env", path="EMPTY_VAR")
        result = backend.resolve(ref)
        assert result.error != ""

    def test_is_subclass_of_secrets_backend(self):
        assert issubclass(EnvBackend, SecretsBackend)


# ---------------------------------------------------------------------------
# DictBackend
# ---------------------------------------------------------------------------


class TestDictBackend:
    def test_name(self):
        assert DictBackend().name == "dict"

    def test_resolve_existing_key(self):
        backend = DictBackend({"my_secret": "value123"})
        ref = SecretReference(backend="dict", path="my_secret")
        result = backend.resolve(ref)
        assert result.value == "value123"
        assert result.source == "dict"
        assert result.error == ""

    def test_resolve_missing_key(self):
        backend = DictBackend({})
        ref = SecretReference(backend="dict", path="missing")
        result = backend.resolve(ref)
        assert result.value == ""
        assert "not found" in result.error

    def test_empty_init(self):
        backend = DictBackend()
        ref = SecretReference(backend="dict", path="anything")
        result = backend.resolve(ref)
        assert result.error != ""

    def test_is_subclass_of_secrets_backend(self):
        assert issubclass(DictBackend, SecretsBackend)

    def test_multiple_keys(self):
        backend = DictBackend({"a": "1", "b": "2", "c": "3"})
        assert backend.resolve(SecretReference(backend="dict", path="a")).value == "1"
        assert backend.resolve(SecretReference(backend="dict", path="b")).value == "2"
        assert backend.resolve(SecretReference(backend="dict", path="c")).value == "3"


# ---------------------------------------------------------------------------
# SecretsManager
# ---------------------------------------------------------------------------


class TestSecretsManager:
    def test_empty_manager(self):
        mgr = SecretsManager()
        assert mgr.list_backends() == []

    def test_register_backend(self):
        mgr = SecretsManager()
        mgr.register_backend(EnvBackend())
        assert "env" in mgr.list_backends()

    def test_register_multiple_backends(self):
        mgr = SecretsManager()
        mgr.register_backend(EnvBackend())
        mgr.register_backend(DictBackend({"k": "v"}))
        backends = mgr.list_backends()
        assert "env" in backends
        assert "dict" in backends

    def test_resolve_dispatches_to_correct_backend(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"secret": "found_it"}))
        ref = SecretReference(backend="dict", path="secret")
        result = mgr.resolve(ref)
        assert result.value == "found_it"

    def test_resolve_unknown_backend(self):
        mgr = SecretsManager()
        ref = SecretReference(backend="unknown", path="x")
        result = mgr.resolve(ref)
        assert result.error != ""
        assert "no backend registered" in result.error

    def test_resolve_all(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"a": "1", "b": "2"}))
        refs = [
            SecretReference(backend="dict", path="a"),
            SecretReference(backend="dict", path="b"),
        ]
        results = mgr.resolve_all(refs)
        assert len(results) == 2
        assert results[0].value == "1"
        assert results[1].value == "2"

    def test_resolve_all_empty(self):
        mgr = SecretsManager()
        assert mgr.resolve_all([]) == []

    def test_resolve_all_mixed_success_failure(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"exists": "val"}))
        refs = [
            SecretReference(backend="dict", path="exists"),
            SecretReference(backend="dict", path="missing"),
        ]
        results = mgr.resolve_all(refs)
        assert results[0].error == ""
        assert results[1].error != ""

    def test_list_backends_sorted(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend())
        mgr.register_backend(EnvBackend())
        # "dict" < "env" alphabetically
        assert mgr.list_backends() == ["dict", "env"]

    def test_replace_backend(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"k": "old"}))
        mgr.register_backend(DictBackend({"k": "new"}))
        ref = SecretReference(backend="dict", path="k")
        assert mgr.resolve(ref).value == "new"


# ---------------------------------------------------------------------------
# parse_secret_reference
# ---------------------------------------------------------------------------


class TestParseSecretReference:
    def test_env_scheme(self):
        ref = parse_secret_reference("env://MY_API_KEY")
        assert ref.backend == "env"
        assert ref.path == "MY_API_KEY"
        assert ref.key_field == ""

    def test_aws_scheme(self):
        ref = parse_secret_reference("aws://arn:aws:secretsmanager:us-east-1:123:secret:prod")
        assert ref.backend == "aws"
        assert ref.path == "arn:aws:secretsmanager:us-east-1:123:secret:prod"

    def test_gcp_scheme(self):
        ref = parse_secret_reference("gcp://projects/my-proj/secrets/api-key")
        assert ref.backend == "gcp"
        assert ref.path == "projects/my-proj/secrets/api-key"

    def test_vault_scheme(self):
        ref = parse_secret_reference("vault://secret/data/myapp")
        assert ref.backend == "vault"
        assert ref.path == "secret/data/myapp"

    def test_with_key_field(self):
        ref = parse_secret_reference("aws://my-secret#api_key")
        assert ref.backend == "aws"
        assert ref.path == "my-secret"
        assert ref.key_field == "api_key"

    def test_default_backend_is_env(self):
        ref = parse_secret_reference("MY_VAR")
        assert ref.backend == "env"
        assert ref.path == "MY_VAR"

    def test_bare_path_with_field(self):
        ref = parse_secret_reference("MY_JSON#token")
        assert ref.backend == "env"
        assert ref.path == "MY_JSON"
        assert ref.key_field == "token"

    def test_empty_string(self):
        ref = parse_secret_reference("")
        assert ref.backend == "env"
        assert ref.path == ""


# ---------------------------------------------------------------------------
# create_default_manager
# ---------------------------------------------------------------------------


class TestCreateDefaultManager:
    def test_has_env_backend(self):
        mgr = create_default_manager()
        assert "env" in mgr.list_backends()

    def test_can_resolve_env_vars(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_MGR_TEST", "hello")
        mgr = create_default_manager()
        ref = SecretReference(backend="env", path="DEFAULT_MGR_TEST")
        result = mgr.resolve(ref)
        assert result.value == "hello"


# ---------------------------------------------------------------------------
# format_secrets_status
# ---------------------------------------------------------------------------


class TestFormatSecretsStatus:
    def test_basic_format(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"key1": "secret1234"}))
        refs = [SecretReference(backend="dict", path="key1")]
        output = format_secrets_status(mgr, refs)
        assert "Secrets status (1 references):" in output
        assert "dict://key1" in output
        assert "resolved" in output
        # Value should be masked
        assert "secret1234" not in output
        assert "1234" in output

    def test_error_format(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({}))
        refs = [SecretReference(backend="dict", path="missing")]
        output = format_secrets_status(mgr, refs)
        assert "ERROR" in output

    def test_multiple_refs(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"a": "value_a1234", "b": "value_b5678"}))
        refs = [
            SecretReference(backend="dict", path="a"),
            SecretReference(backend="dict", path="b"),
        ]
        output = format_secrets_status(mgr, refs)
        assert "2 references" in output
        assert "dict://a" in output
        assert "dict://b" in output

    def test_key_field_shown(self):
        mgr = SecretsManager()
        mgr.register_backend(DictBackend({"s": "val12345"}))
        refs = [SecretReference(backend="dict", path="s", key_field="token")]
        output = format_secrets_status(mgr, refs)
        assert "dict://s#token" in output

    def test_empty_refs(self):
        mgr = SecretsManager()
        output = format_secrets_status(mgr, [])
        assert "0 references" in output

    def test_unregistered_backend_error(self):
        mgr = SecretsManager()
        refs = [SecretReference(backend="aws", path="arn")]
        output = format_secrets_status(mgr, refs)
        assert "ERROR" in output
        assert "no backend registered" in output
