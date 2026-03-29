"""Tests for the provider instrumentation registry and specs.

Run with: uv run python -m pytest tests/test_provider_instrumentors.py -v
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from evalyn_sdk.provider_instrumentors import (
    BUILTIN_PROVIDERS,
    InstrumentationHook,
    ProviderRegistry,
    ProviderSpec,
    create_default_registry,
    format_provider_status,
)


# ---------------------------------------------------------------------------
# ProviderSpec - construction and fields
# ---------------------------------------------------------------------------


class TestProviderSpecConstruction:
    def test_basic_fields(self):
        spec = ProviderSpec(
            provider_id="test",
            name="Test Provider",
            module_path="test_mod",
            client_class="TestClient",
            methods_to_wrap=["run"],
            env_key="TEST_KEY",
        )
        assert spec.provider_id == "test"
        assert spec.name == "Test Provider"
        assert spec.module_path == "test_mod"
        assert spec.client_class == "TestClient"
        assert spec.methods_to_wrap == ["run"]
        assert spec.env_key == "TEST_KEY"

    def test_multiple_methods(self):
        spec = ProviderSpec(
            provider_id="multi",
            name="Multi",
            module_path="m",
            client_class="C",
            methods_to_wrap=["chat", "generate", "embed"],
            env_key="M_KEY",
        )
        assert len(spec.methods_to_wrap) == 3

    def test_empty_methods_list(self):
        spec = ProviderSpec(
            provider_id="empty",
            name="Empty",
            module_path="e",
            client_class="E",
            methods_to_wrap=[],
            env_key="E_KEY",
        )
        assert spec.methods_to_wrap == []


# ---------------------------------------------------------------------------
# ProviderSpec - as_dict / from_dict
# ---------------------------------------------------------------------------


class TestProviderSpecSerialization:
    def test_as_dict_keys(self):
        spec = ProviderSpec(
            provider_id="a",
            name="A",
            module_path="a_mod",
            client_class="AC",
            methods_to_wrap=["x"],
            env_key="A_KEY",
        )
        d = spec.as_dict()
        assert set(d.keys()) == {
            "provider_id",
            "name",
            "module_path",
            "client_class",
            "methods_to_wrap",
            "env_key",
        }

    def test_roundtrip(self):
        spec = ProviderSpec(
            provider_id="rt",
            name="Roundtrip",
            module_path="rt_mod",
            client_class="RTC",
            methods_to_wrap=["a", "b"],
            env_key="RT_KEY",
        )
        restored = ProviderSpec.from_dict(spec.as_dict())
        assert restored.provider_id == spec.provider_id
        assert restored.name == spec.name
        assert restored.module_path == spec.module_path
        assert restored.client_class == spec.client_class
        assert restored.methods_to_wrap == spec.methods_to_wrap
        assert restored.env_key == spec.env_key

    def test_from_dict_missing_methods(self):
        d = {
            "provider_id": "x",
            "name": "X",
            "module_path": "xm",
            "client_class": "XC",
            "env_key": "X_KEY",
        }
        spec = ProviderSpec.from_dict(d)
        assert spec.methods_to_wrap == []

    def test_as_dict_methods_is_copy(self):
        spec = ProviderSpec(
            provider_id="cp",
            name="Copy",
            module_path="cp_mod",
            client_class="CPC",
            methods_to_wrap=["m1"],
            env_key="CP_KEY",
        )
        d = spec.as_dict()
        d["methods_to_wrap"].append("injected")
        assert "injected" not in spec.methods_to_wrap


# ---------------------------------------------------------------------------
# InstrumentationHook - construction and fields
# ---------------------------------------------------------------------------


class TestInstrumentationHookConstruction:
    def test_basic_fields(self):
        hook = InstrumentationHook(
            hook_id="h1",
            provider_id="p1",
            method_name="chat",
        )
        assert hook.hook_id == "h1"
        assert hook.provider_id == "p1"
        assert hook.method_name == "chat"
        assert hook.before_fn_name == ""
        assert hook.after_fn_name == ""

    def test_with_callbacks(self):
        hook = InstrumentationHook(
            hook_id="h2",
            provider_id="p1",
            method_name="generate",
            before_fn_name="on_before_generate",
            after_fn_name="on_after_generate",
        )
        assert hook.before_fn_name == "on_before_generate"
        assert hook.after_fn_name == "on_after_generate"

    def test_defaults(self):
        hook = InstrumentationHook(hook_id="h", provider_id="p", method_name="m")
        assert hook.before_fn_name == ""
        assert hook.after_fn_name == ""


# ---------------------------------------------------------------------------
# InstrumentationHook - as_dict / from_dict
# ---------------------------------------------------------------------------


class TestInstrumentationHookSerialization:
    def test_as_dict_keys(self):
        hook = InstrumentationHook(hook_id="h", provider_id="p", method_name="m")
        d = hook.as_dict()
        assert set(d.keys()) == {
            "hook_id",
            "provider_id",
            "method_name",
            "before_fn_name",
            "after_fn_name",
        }

    def test_roundtrip(self):
        hook = InstrumentationHook(
            hook_id="h3",
            provider_id="p2",
            method_name="embed",
            before_fn_name="before",
            after_fn_name="after",
        )
        restored = InstrumentationHook.from_dict(hook.as_dict())
        assert restored.hook_id == hook.hook_id
        assert restored.provider_id == hook.provider_id
        assert restored.method_name == hook.method_name
        assert restored.before_fn_name == hook.before_fn_name
        assert restored.after_fn_name == hook.after_fn_name

    def test_from_dict_defaults(self):
        d = {"hook_id": "h", "provider_id": "p", "method_name": "m"}
        hook = InstrumentationHook.from_dict(d)
        assert hook.before_fn_name == ""
        assert hook.after_fn_name == ""


# ---------------------------------------------------------------------------
# ProviderRegistry - register / get / list
# ---------------------------------------------------------------------------


class TestProviderRegistryBasics:
    def test_empty_registry(self):
        reg = ProviderRegistry()
        assert reg.list_providers() == []
        assert reg.get("nonexistent") is None

    def test_register_and_get(self):
        reg = ProviderRegistry()
        spec = ProviderSpec(
            provider_id="test",
            name="Test",
            module_path="t",
            client_class="T",
            methods_to_wrap=["run"],
            env_key="T_KEY",
        )
        reg.register(spec)
        assert reg.get("test") is spec

    def test_list_providers_order(self):
        reg = ProviderRegistry()
        specs = [
            ProviderSpec("a", "A", "a", "A", ["x"], "A_KEY"),
            ProviderSpec("b", "B", "b", "B", ["y"], "B_KEY"),
            ProviderSpec("c", "C", "c", "C", ["z"], "C_KEY"),
        ]
        for s in specs:
            reg.register(s)
        listed = reg.list_providers()
        assert [s.provider_id for s in listed] == ["a", "b", "c"]

    def test_overwrite_existing(self):
        reg = ProviderRegistry()
        v1 = ProviderSpec("dup", "V1", "m1", "C1", ["a"], "K1")
        v2 = ProviderSpec("dup", "V2", "m2", "C2", ["b"], "K2")
        reg.register(v1)
        reg.register(v2)
        assert reg.get("dup").name == "V2"
        assert len(reg.list_providers()) == 1

    def test_get_nonexistent_returns_none(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("x", "X", "x", "X", [], "X_KEY"))
        assert reg.get("missing") is None


# ---------------------------------------------------------------------------
# ProviderRegistry - is_available / get_available
# ---------------------------------------------------------------------------


class TestProviderRegistryAvailability:
    def test_is_available_with_env_set(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "T", "t", "T", ["r"], "TEST_AVAIL_KEY"))
        with patch.dict(os.environ, {"TEST_AVAIL_KEY": "secret123"}):
            assert reg.is_available("t") is True

    def test_is_available_with_env_unset(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "T", "t", "T", ["r"], "DEFINITELY_UNSET_KEY_XYZ"))
        env = os.environ.copy()
        env.pop("DEFINITELY_UNSET_KEY_XYZ", None)
        with patch.dict(os.environ, env, clear=True):
            assert reg.is_available("t") is False

    def test_is_available_unknown_provider(self):
        reg = ProviderRegistry()
        assert reg.is_available("ghost") is False

    def test_is_available_empty_env_value(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "T", "t", "T", ["r"], "EMPTY_VAL_KEY"))
        with patch.dict(os.environ, {"EMPTY_VAL_KEY": ""}):
            assert reg.is_available("t") is False

    def test_get_available_mixed(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("yes", "Yes", "y", "Y", ["a"], "YES_KEY"))
        reg.register(ProviderSpec("no", "No", "n", "N", ["b"], "NO_KEY"))
        env = {"YES_KEY": "val"}
        # Make sure NO_KEY is not in environment
        with patch.dict(os.environ, env, clear=True):
            avail = reg.get_available()
            ids = [s.provider_id for s in avail]
            assert "yes" in ids
            assert "no" not in ids

    def test_get_available_none(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("a", "A", "a", "A", [], "NOPE_KEY_1"))
        with patch.dict(os.environ, {}, clear=True):
            assert reg.get_available() == []

    def test_get_available_all(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("a", "A", "a", "A", [], "ALL_KEY_1"))
        reg.register(ProviderSpec("b", "B", "b", "B", [], "ALL_KEY_2"))
        with patch.dict(os.environ, {"ALL_KEY_1": "v", "ALL_KEY_2": "v"}, clear=True):
            assert len(reg.get_available()) == 2


# ---------------------------------------------------------------------------
# BUILTIN_PROVIDERS
# ---------------------------------------------------------------------------


class TestBuiltinProviders:
    def test_count(self):
        assert len(BUILTIN_PROVIDERS) == 7

    def test_ids_unique(self):
        ids = [s.provider_id for s in BUILTIN_PROVIDERS]
        assert len(ids) == len(set(ids))

    def test_env_keys_nonempty(self):
        for spec in BUILTIN_PROVIDERS:
            assert spec.env_key, f"{spec.provider_id} has empty env_key"

    def test_methods_nonempty(self):
        for spec in BUILTIN_PROVIDERS:
            assert spec.methods_to_wrap, f"{spec.provider_id} has no methods"

    def test_cohere_spec(self):
        cohere = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "cohere")
        assert cohere.env_key == "COHERE_API_KEY"
        assert cohere.client_class == "Client"
        assert "chat" in cohere.methods_to_wrap
        assert "generate" in cohere.methods_to_wrap

    def test_mistral_spec(self):
        mistral = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "mistral")
        assert mistral.env_key == "MISTRAL_API_KEY"
        assert mistral.client_class == "Mistral"

    def test_bedrock_spec(self):
        bedrock = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "bedrock")
        assert bedrock.env_key == "AWS_ACCESS_KEY_ID"
        assert "invoke_model" in bedrock.methods_to_wrap

    def test_azure_openai_spec(self):
        azure = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "azure_openai")
        assert azure.env_key == "AZURE_OPENAI_API_KEY"
        assert azure.client_class == "AzureOpenAI"

    def test_groq_spec(self):
        groq = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "groq")
        assert groq.env_key == "GROQ_API_KEY"

    def test_together_spec(self):
        together = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "together")
        assert together.env_key == "TOGETHER_API_KEY"

    def test_replicate_spec(self):
        replicate = next(s for s in BUILTIN_PROVIDERS if s.provider_id == "replicate")
        assert replicate.env_key == "REPLICATE_API_TOKEN"
        assert replicate.client_class == "Client"


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_returns_registry(self):
        reg = create_default_registry()
        assert isinstance(reg, ProviderRegistry)

    def test_contains_all_builtins(self):
        reg = create_default_registry()
        for spec in BUILTIN_PROVIDERS:
            assert reg.get(spec.provider_id) is not None

    def test_count_matches(self):
        reg = create_default_registry()
        assert len(reg.list_providers()) == len(BUILTIN_PROVIDERS)

    def test_independent_instances(self):
        r1 = create_default_registry()
        r2 = create_default_registry()
        r1.register(ProviderSpec("extra", "Extra", "e", "E", [], "E_KEY"))
        assert r2.get("extra") is None


# ---------------------------------------------------------------------------
# format_provider_status
# ---------------------------------------------------------------------------


class TestFormatProviderStatus:
    def test_empty_registry(self):
        reg = ProviderRegistry()
        output = format_provider_status(reg)
        assert output == "No providers registered."

    def test_header_present(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "Test", "t", "T", ["r"], "T_KEY"))
        with patch.dict(os.environ, {}, clear=True):
            output = format_provider_status(reg)
            assert "Provider" in output
            assert "Status" in output
            assert "Env Var" in output

    def test_available_status(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "Test", "t", "T", ["r"], "FMT_KEY"))
        with patch.dict(os.environ, {"FMT_KEY": "secret"}):
            output = format_provider_status(reg)
            assert "available" in output

    def test_not_configured_status(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "Test", "t", "T", ["r"], "NOPE_FMT_KEY"))
        with patch.dict(os.environ, {}, clear=True):
            output = format_provider_status(reg)
            assert "not configured" in output

    def test_separator_line(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "Test", "t", "T", ["r"], "S_KEY"))
        with patch.dict(os.environ, {}, clear=True):
            output = format_provider_status(reg)
            lines = output.split("\n")
            assert len(lines) >= 3
            # Second line should be dashes
            assert set(lines[1].strip().replace(" ", "")) == {"-"}

    def test_multiple_providers(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("a", "Alpha", "a", "A", [], "A_KEY_FMT"))
        reg.register(ProviderSpec("b", "Beta", "b", "B", [], "B_KEY_FMT"))
        with patch.dict(os.environ, {"A_KEY_FMT": "v"}, clear=True):
            output = format_provider_status(reg)
            assert "Alpha" in output
            assert "Beta" in output
            assert "available" in output
            assert "not configured" in output

    def test_env_var_column(self):
        reg = ProviderRegistry()
        reg.register(ProviderSpec("t", "T", "t", "T", [], "MY_SPECIAL_KEY"))
        with patch.dict(os.environ, {}, clear=True):
            output = format_provider_status(reg)
            assert "MY_SPECIAL_KEY" in output
