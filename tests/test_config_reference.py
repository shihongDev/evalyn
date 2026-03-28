"""Tests for evalyn_sdk.config_reference module."""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.config_reference import (
    KNOWN_OPTIONS,
    ConfigOption,
    ConfigReference,
    build_reference,
    filter_reference,
    format_reference_markdown,
    format_reference_plain,
    validate_config,
)


# ---------------------------------------------------------------------------
# Sample test data
# ---------------------------------------------------------------------------

SAMPLE_OPTIONS = [
    ConfigOption(
        key="provider",
        type_name="str",
        default="openai",
        description="LLM provider",
        env_var="EVALYN_PROVIDER",
        cli_flag="--provider",
        valid_values=["openai", "anthropic", "google"],
        required=False,
        section="provider",
    ),
    ConfigOption(
        key="model",
        type_name="str",
        default="gpt-4o",
        description="Model name for LLM judge",
        env_var="EVALYN_MODEL",
        cli_flag="--model",
        valid_values=[],
        required=False,
        section="provider",
    ),
    ConfigOption(
        key="dataset_path",
        type_name="str",
        default="./data",
        description="Path to dataset directory",
        env_var="EVALYN_DATASET_PATH",
        cli_flag="--dataset",
        valid_values=[],
        required=True,
        section="data",
    ),
    ConfigOption(
        key="parallel_workers",
        type_name="int",
        default=4,
        description="Number of parallel workers",
        env_var="EVALYN_PARALLEL_WORKERS",
        cli_flag="--workers",
        valid_values=[],
        required=False,
        section="runtime",
    ),
    ConfigOption(
        key="confidence_threshold",
        type_name="float",
        default=0.7,
        description="Minimum confidence threshold",
        env_var="",
        cli_flag="",
        valid_values=[],
        required=False,
        section="judge",
    ),
    ConfigOption(
        key="cache_enabled",
        type_name="bool",
        default=True,
        description="Enable caching",
        env_var="",
        cli_flag="--cache/--no-cache",
        valid_values=[],
        required=False,
        section="cache",
    ),
    ConfigOption(
        key="log_level",
        type_name="str",
        default="info",
        description="Logging level",
        env_var="EVALYN_LOG_LEVEL",
        cli_flag="--log-level",
        valid_values=["debug", "info", "warning", "error"],
        required=False,
        section="runtime",
    ),
]


@pytest.fixture
def ref():
    return build_reference(SAMPLE_OPTIONS)


# ---------------------------------------------------------------------------
# ConfigOption tests
# ---------------------------------------------------------------------------


class TestConfigOption:
    def test_as_dict_roundtrip(self):
        opt = SAMPLE_OPTIONS[0]
        d = opt.as_dict()
        restored = ConfigOption.from_dict(d)
        assert restored.key == opt.key
        assert restored.type_name == opt.type_name
        assert restored.default == opt.default
        assert restored.valid_values == opt.valid_values
        assert restored.required == opt.required

    def test_from_dict_defaults(self):
        opt = ConfigOption.from_dict({"key": "test"})
        assert opt.type_name == "str"
        assert opt.default is None
        assert opt.description == ""
        assert opt.env_var == ""
        assert opt.cli_flag == ""
        assert opt.valid_values == []
        assert opt.required is False
        assert opt.section == ""

    def test_as_dict_keys(self):
        opt = ConfigOption(key="k", type_name="str", default="d", description="desc")
        d = opt.as_dict()
        expected = {
            "key",
            "type_name",
            "default",
            "description",
            "env_var",
            "cli_flag",
            "valid_values",
            "required",
            "section",
        }
        assert set(d.keys()) == expected

    def test_valid_values_is_copy(self):
        opt = ConfigOption(
            key="k",
            type_name="str",
            default="a",
            description="d",
            valid_values=["a", "b"],
        )
        d = opt.as_dict()
        d["valid_values"].append("c")
        assert len(opt.valid_values) == 2


# ---------------------------------------------------------------------------
# ConfigReference tests
# ---------------------------------------------------------------------------


class TestConfigReference:
    def test_as_dict_roundtrip(self, ref):
        d = ref.as_dict()
        restored = ConfigReference.from_dict(d)
        assert len(restored.options) == len(ref.options)
        assert restored.sections == ref.sections
        assert restored.generated_at == ref.generated_at

    def test_from_dict_defaults(self):
        cr = ConfigReference.from_dict({})
        assert cr.options == []
        assert cr.sections == []
        assert cr.generated_at == ""

    def test_json_serializable(self, ref):
        d = ref.as_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        restored = ConfigReference.from_dict(loaded)
        assert len(restored.options) == len(ref.options)

    def test_sections_sorted(self, ref):
        assert ref.sections == sorted(ref.sections)


# ---------------------------------------------------------------------------
# KNOWN_OPTIONS tests
# ---------------------------------------------------------------------------


class TestKnownOptions:
    def test_has_entries(self):
        assert len(KNOWN_OPTIONS) >= 15

    def test_all_have_key(self):
        for opt in KNOWN_OPTIONS:
            assert opt.key, f"Option missing key: {opt}"

    def test_all_have_section(self):
        for opt in KNOWN_OPTIONS:
            assert opt.section, f"Option {opt.key} missing section"

    def test_all_have_type(self):
        for opt in KNOWN_OPTIONS:
            assert opt.type_name in ("str", "int", "float", "bool"), (
                f"Option {opt.key} has unexpected type: {opt.type_name}"
            )

    def test_unique_keys(self):
        keys = [o.key for o in KNOWN_OPTIONS]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# build_reference tests
# ---------------------------------------------------------------------------


class TestBuildReference:
    def test_uses_known_options_by_default(self):
        ref = build_reference()
        assert len(ref.options) == len(KNOWN_OPTIONS)

    def test_custom_options(self):
        opts = [
            ConfigOption(key="x", type_name="str", default="", description="test", section="s1"),
            ConfigOption(key="y", type_name="int", default=0, description="test", section="s2"),
        ]
        ref = build_reference(opts)
        assert len(ref.options) == 2
        assert ref.sections == ["s1", "s2"]

    def test_generated_at(self):
        ref = build_reference()
        assert ref.generated_at != ""

    def test_sections_derived(self, ref):
        expected = sorted({"provider", "data", "runtime", "judge", "cache"})
        assert ref.sections == expected

    def test_empty_options(self):
        ref = build_reference([])
        assert ref.options == []
        assert ref.sections == []


# ---------------------------------------------------------------------------
# filter_reference tests
# ---------------------------------------------------------------------------


class TestFilterReference:
    def test_filter_by_section(self, ref):
        result = filter_reference(ref, section="provider")
        assert len(result.options) == 2
        assert all(o.section == "provider" for o in result.options)

    def test_filter_by_search_key(self, ref):
        result = filter_reference(ref, search="cache")
        assert len(result.options) == 1
        assert result.options[0].key == "cache_enabled"

    def test_filter_by_search_description(self, ref):
        result = filter_reference(ref, search="parallel")
        assert len(result.options) == 1

    def test_filter_case_insensitive(self, ref):
        result = filter_reference(ref, search="LLM")
        assert len(result.options) >= 1

    def test_filter_combined(self, ref):
        result = filter_reference(ref, section="runtime", search="log")
        assert len(result.options) == 1
        assert result.options[0].key == "log_level"

    def test_filter_no_match(self, ref):
        result = filter_reference(ref, section="nonexistent")
        assert len(result.options) == 0

    def test_filter_preserves_generated_at(self, ref):
        result = filter_reference(ref, section="data")
        assert result.generated_at == ref.generated_at

    def test_filter_sections_updated(self, ref):
        result = filter_reference(ref, section="provider")
        assert result.sections == ["provider"]

    def test_filter_no_args_returns_all(self, ref):
        result = filter_reference(ref)
        assert len(result.options) == len(ref.options)


# ---------------------------------------------------------------------------
# format_reference_markdown tests
# ---------------------------------------------------------------------------


class TestFormatReferenceMarkdown:
    def test_has_title(self, ref):
        md = format_reference_markdown(ref)
        assert "# Configuration Reference" in md

    def test_has_sections(self, ref):
        md = format_reference_markdown(ref)
        assert "## provider" in md
        assert "## data" in md

    def test_has_option_entries(self, ref):
        md = format_reference_markdown(ref)
        assert "`provider`" in md
        assert "`dataset_path`" in md

    def test_required_marked(self, ref):
        md = format_reference_markdown(ref)
        assert "**(required)**" in md

    def test_has_type(self, ref):
        md = format_reference_markdown(ref)
        assert "`str`" in md
        assert "`int`" in md

    def test_has_env_var(self, ref):
        md = format_reference_markdown(ref)
        assert "EVALYN_PROVIDER" in md

    def test_has_cli_flag(self, ref):
        md = format_reference_markdown(ref)
        assert "--provider" in md

    def test_has_valid_values(self, ref):
        md = format_reference_markdown(ref)
        assert "`openai`" in md
        assert "`anthropic`" in md

    def test_empty_ref(self):
        ref = ConfigReference(generated_at="now")
        md = format_reference_markdown(ref)
        assert "# Configuration Reference" in md


# ---------------------------------------------------------------------------
# format_reference_plain tests
# ---------------------------------------------------------------------------


class TestFormatReferencePlain:
    def test_has_title(self, ref):
        txt = format_reference_plain(ref)
        assert "Configuration Reference" in txt

    def test_has_sections(self, ref):
        txt = format_reference_plain(ref)
        assert "[provider]" in txt
        assert "[data]" in txt

    def test_has_options(self, ref):
        txt = format_reference_plain(ref)
        assert "provider" in txt
        assert "dataset_path" in txt

    def test_required_marked(self, ref):
        txt = format_reference_plain(ref)
        assert "(required)" in txt

    def test_has_type_info(self, ref):
        txt = format_reference_plain(ref)
        assert "Type:" in txt

    def test_has_default(self, ref):
        txt = format_reference_plain(ref)
        assert "Default:" in txt

    def test_empty_ref_plain(self):
        ref = ConfigReference(generated_at="now")
        txt = format_reference_plain(ref)
        assert "Configuration Reference" in txt


# ---------------------------------------------------------------------------
# validate_config tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self, ref):
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "dataset_path": "./data",
            "parallel_workers": 4,
            "confidence_threshold": 0.8,
            "cache_enabled": True,
            "log_level": "info",
        }
        errors = validate_config(config, ref)
        assert errors == []

    def test_unknown_key(self, ref):
        config = {"dataset_path": "./data", "nonexistent_key": "value"}
        errors = validate_config(config, ref)
        assert any("Unknown" in e and "nonexistent_key" in e for e in errors)

    def test_missing_required(self, ref):
        config = {"provider": "openai"}
        errors = validate_config(config, ref)
        assert any("required" in e.lower() and "dataset_path" in e for e in errors)

    def test_wrong_type_str_for_int(self, ref):
        config = {"dataset_path": "./data", "parallel_workers": "not_a_number"}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "parallel_workers" in e for e in errors)

    def test_wrong_type_int_for_str(self, ref):
        config = {"dataset_path": "./data", "provider": 123}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "provider" in e for e in errors)

    def test_wrong_type_str_for_bool(self, ref):
        config = {"dataset_path": "./data", "cache_enabled": "yes"}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "cache_enabled" in e for e in errors)

    def test_wrong_type_bool_for_int(self, ref):
        config = {"dataset_path": "./data", "parallel_workers": True}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "parallel_workers" in e for e in errors)

    def test_wrong_type_str_for_float(self, ref):
        config = {"dataset_path": "./data", "confidence_threshold": "high"}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "confidence_threshold" in e for e in errors)

    def test_int_accepted_for_float(self, ref):
        config = {"dataset_path": "./data", "confidence_threshold": 1}
        errors = validate_config(config, ref)
        type_errors = [e for e in errors if "Wrong type" in e and "confidence_threshold" in e]
        assert len(type_errors) == 0

    def test_float_rejected_for_int(self, ref):
        config = {"dataset_path": "./data", "parallel_workers": 4.5}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "parallel_workers" in e for e in errors)

    def test_invalid_value(self, ref):
        config = {"dataset_path": "./data", "provider": "unknown_provider"}
        errors = validate_config(config, ref)
        assert any("Invalid value" in e and "provider" in e for e in errors)

    def test_invalid_log_level(self, ref):
        config = {"dataset_path": "./data", "log_level": "verbose"}
        errors = validate_config(config, ref)
        assert any("Invalid value" in e and "log_level" in e for e in errors)

    def test_valid_log_level(self, ref):
        config = {"dataset_path": "./data", "log_level": "debug"}
        errors = validate_config(config, ref)
        val_errors = [e for e in errors if "Invalid value" in e and "log_level" in e]
        assert len(val_errors) == 0

    def test_empty_config_missing_required(self, ref):
        errors = validate_config({}, ref)
        assert any("required" in e.lower() for e in errors)

    def test_multiple_errors(self, ref):
        config = {
            "unknown_key": "x",
            "provider": 999,
            "parallel_workers": "bad",
        }
        errors = validate_config(config, ref)
        assert len(errors) >= 3

    def test_none_value_skips_type_check(self, ref):
        config = {"dataset_path": "./data", "provider": None}
        errors = validate_config(config, ref)
        type_errors = [e for e in errors if "Wrong type" in e and "provider" in e]
        assert len(type_errors) == 0

    def test_bool_rejected_for_float(self, ref):
        config = {"dataset_path": "./data", "confidence_threshold": True}
        errors = validate_config(config, ref)
        assert any("Wrong type" in e and "confidence_threshold" in e for e in errors)

    def test_valid_values_not_checked_when_empty(self, ref):
        config = {"dataset_path": "./data", "model": "any-model-name"}
        errors = validate_config(config, ref)
        val_errors = [e for e in errors if "Invalid value" in e and "model" in e]
        assert len(val_errors) == 0

    def test_all_required_present_no_required_errors(self, ref):
        config = {"dataset_path": "./data"}
        errors = validate_config(config, ref)
        required_errors = [e for e in errors if "required" in e.lower()]
        assert len(required_errors) == 0
