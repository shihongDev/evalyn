"""Tests for the config validation module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.config_validation import (
    ConfigError,
    ConfigSchema,
    ValidationResult,
    DEFAULT_SCHEMA,
    validate_config,
    suggest_fix,
    validate_config_file,
    format_validation_result,
)


# ---------------------------------------------------------------------------
# ConfigError dataclass
# ---------------------------------------------------------------------------


class TestConfigError:
    def test_as_dict(self):
        e = ConfigError(
            field="foo", error_type="unknown_key",
            message="Unknown key", suggestion="Remove it",
        )
        d = e.as_dict()
        assert d["field"] == "foo"
        assert d["error_type"] == "unknown_key"
        assert d["message"] == "Unknown key"
        assert d["suggestion"] == "Remove it"

    def test_from_dict(self):
        d = {
            "field": "bar",
            "error_type": "missing_required",
            "message": "Missing",
            "suggestion": "Add it",
        }
        e = ConfigError.from_dict(d)
        assert e.field == "bar"
        assert e.suggestion == "Add it"

    def test_from_dict_default_suggestion(self):
        d = {"field": "x", "error_type": "unknown_key", "message": "msg"}
        e = ConfigError.from_dict(d)
        assert e.suggestion == ""

    def test_roundtrip(self):
        e = ConfigError(field="f", error_type="type_mismatch", message="m", suggestion="s")
        assert ConfigError.from_dict(e.as_dict()).as_dict() == e.as_dict()


# ---------------------------------------------------------------------------
# ConfigSchema dataclass
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_as_dict(self):
        s = ConfigSchema(
            required_fields=["a"],
            optional_fields=["b"],
            field_types={"a": "str"},
            deprecated_fields={"old": "new"},
        )
        d = s.as_dict()
        assert d["required_fields"] == ["a"]
        assert d["deprecated_fields"] == {"old": "new"}

    def test_from_dict(self):
        d = {
            "required_fields": ["x"],
            "optional_fields": ["y"],
            "field_types": {"x": "int"},
            "deprecated_fields": {},
        }
        s = ConfigSchema.from_dict(d)
        assert s.required_fields == ["x"]
        assert s.field_types == {"x": "int"}

    def test_from_dict_defaults(self):
        s = ConfigSchema.from_dict({})
        assert s.required_fields == []
        assert s.optional_fields == []
        assert s.field_types == {}
        assert s.deprecated_fields == {}

    def test_all_known_fields(self):
        s = ConfigSchema(
            required_fields=["a", "b"],
            optional_fields=["c"],
            deprecated_fields={"old": "a"},
        )
        known = s.all_known_fields
        assert known == {"a", "b", "c", "old"}

    def test_roundtrip(self):
        s = ConfigSchema(
            required_fields=["r"], optional_fields=["o"],
            field_types={"r": "str"}, deprecated_fields={"d": "r"},
        )
        assert ConfigSchema.from_dict(s.as_dict()).as_dict() == s.as_dict()


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_as_dict_empty(self):
        vr = ValidationResult()
        d = vr.as_dict()
        assert d["errors"] == []
        assert d["warnings"] == []
        assert d["valid"] is True

    def test_from_dict(self):
        d = {
            "errors": [{"field": "f", "error_type": "unknown_key", "message": "m"}],
            "warnings": [],
            "valid": False,
        }
        vr = ValidationResult.from_dict(d)
        assert len(vr.errors) == 1
        assert vr.valid is False

    def test_roundtrip(self):
        vr = ValidationResult(
            errors=[ConfigError(field="x", error_type="missing_required", message="m")],
            warnings=[ConfigError(field="y", error_type="deprecated", message="w", suggestion="s")],
            valid=False,
        )
        d = vr.as_dict()
        vr2 = ValidationResult.from_dict(d)
        assert vr2.as_dict() == d


# ---------------------------------------------------------------------------
# DEFAULT_SCHEMA
# ---------------------------------------------------------------------------


class TestDefaultSchema:
    def test_required_fields(self):
        assert "provider" in DEFAULT_SCHEMA.required_fields
        assert "model" in DEFAULT_SCHEMA.required_fields

    def test_optional_fields(self):
        assert "dataset_path" in DEFAULT_SCHEMA.optional_fields
        assert "cache_enabled" in DEFAULT_SCHEMA.optional_fields

    def test_field_types(self):
        assert DEFAULT_SCHEMA.field_types["provider"] == "str"
        assert DEFAULT_SCHEMA.field_types["parallel_workers"] == "int"
        assert DEFAULT_SCHEMA.field_types["cache_enabled"] == "bool"

    def test_deprecated_fields(self):
        assert DEFAULT_SCHEMA.deprecated_fields["llm_provider"] == "provider"
        assert DEFAULT_SCHEMA.deprecated_fields["max_retries"] == "retry_limit"


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_minimal_config(self):
        config = {"provider": "openai", "model": "gpt-4"}
        result = validate_config(config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_full_config(self):
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "dataset_path": "data/",
            "parallel_workers": 4,
            "cache_enabled": True,
        }
        result = validate_config(config)
        assert result.valid is True

    def test_missing_required(self):
        config = {"model": "gpt-4"}
        result = validate_config(config)
        assert result.valid is False
        missing = [e for e in result.errors if e.error_type == "missing_required"]
        assert len(missing) == 1
        assert missing[0].field == "provider"

    def test_missing_all_required(self):
        config = {"dataset_path": "data/"}
        result = validate_config(config)
        assert result.valid is False
        missing = [e for e in result.errors if e.error_type == "missing_required"]
        assert len(missing) == 2

    def test_unknown_key(self):
        config = {"provider": "openai", "model": "gpt-4", "bogus_field": "value"}
        result = validate_config(config)
        assert result.valid is False
        unknown = [e for e in result.errors if e.error_type == "unknown_key"]
        assert len(unknown) == 1
        assert unknown[0].field == "bogus_field"

    def test_type_mismatch_int(self):
        config = {"provider": "openai", "model": "gpt-4", "parallel_workers": "not_a_number"}
        result = validate_config(config)
        assert result.valid is False
        tm = [e for e in result.errors if e.error_type == "type_mismatch"]
        assert len(tm) == 1
        assert tm[0].field == "parallel_workers"

    def test_type_mismatch_bool(self):
        config = {"provider": "openai", "model": "gpt-4", "cache_enabled": "maybe"}
        result = validate_config(config)
        assert result.valid is False

    def test_bool_string_accepted(self):
        config = {"provider": "openai", "model": "gpt-4", "cache_enabled": "true"}
        result = validate_config(config)
        assert result.valid is True

    def test_int_string_accepted(self):
        config = {"provider": "openai", "model": "gpt-4", "parallel_workers": "8"}
        result = validate_config(config)
        assert result.valid is True

    def test_deprecated_field_warning(self):
        config = {"provider": "openai", "model": "gpt-4", "llm_provider": "openai"}
        result = validate_config(config)
        # Deprecated produces a warning, not an error
        assert result.valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].error_type == "deprecated"
        assert "provider" in result.warnings[0].message

    def test_multiple_deprecated(self):
        config = {
            "provider": "openai", "model": "gpt-4",
            "llm_provider": "openai", "max_retries": 3,
        }
        result = validate_config(config)
        assert len(result.warnings) == 2

    def test_custom_schema(self):
        schema = ConfigSchema(
            required_fields=["name"],
            optional_fields=["desc"],
            field_types={"name": "str"},
            deprecated_fields={},
        )
        config = {"name": "test"}
        result = validate_config(config, schema)
        assert result.valid is True

    def test_custom_schema_missing_required(self):
        schema = ConfigSchema(required_fields=["name"])
        config = {}
        result = validate_config(config, schema)
        assert result.valid is False

    def test_empty_config(self):
        result = validate_config({})
        assert result.valid is False
        missing = [e for e in result.errors if e.error_type == "missing_required"]
        assert len(missing) == 2

    def test_provider_type_mismatch(self):
        config = {"provider": 123, "model": "gpt-4"}
        result = validate_config(config)
        assert result.valid is False
        tm = [e for e in result.errors if e.error_type == "type_mismatch"]
        assert any(e.field == "provider" for e in tm)


# ---------------------------------------------------------------------------
# suggest_fix
# ---------------------------------------------------------------------------


class TestSuggestFix:
    def test_unknown_key(self):
        e = ConfigError(field="xyz", error_type="unknown_key", message="")
        s = suggest_fix(e)
        assert "xyz" in s

    def test_missing_required(self):
        e = ConfigError(field="provider", error_type="missing_required", message="")
        s = suggest_fix(e)
        assert "provider" in s

    def test_type_mismatch(self):
        e = ConfigError(field="timeout", error_type="type_mismatch", message="")
        s = suggest_fix(e)
        assert "timeout" in s

    def test_deprecated(self):
        e = ConfigError(field="old_name", error_type="deprecated", message="")
        s = suggest_fix(e)
        assert "old_name" in s

    def test_unknown_error_type(self):
        e = ConfigError(field="f", error_type="custom_thing", message="")
        s = suggest_fix(e)
        assert s == ""


# ---------------------------------------------------------------------------
# validate_config_file
# ---------------------------------------------------------------------------


class TestValidateConfigFile:
    def test_file_not_found(self, tmp_path):
        result = validate_config_file(str(tmp_path / "missing.yaml"))
        assert result.valid is False
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].message

    def test_valid_file(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("provider: openai\nmodel: gpt-4\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True

    def test_missing_required_in_file(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("model: gpt-4\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is False

    def test_comments_ignored(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("# comment\nprovider: openai\nmodel: gpt-4\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True

    def test_blank_lines_ignored(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("provider: openai\n\n\nmodel: gpt-4\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True

    def test_int_parsing(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("provider: openai\nmodel: gpt-4\nparallel_workers: 8\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True

    def test_bool_parsing(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("provider: openai\nmodel: gpt-4\ncache_enabled: true\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True

    def test_deprecated_in_file(self, tmp_path):
        cfg_file = tmp_path / "evalyn.yaml"
        cfg_file.write_text("provider: openai\nmodel: gpt-4\nllm_provider: openai\n")
        result = validate_config_file(str(cfg_file))
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_custom_schema_on_file(self, tmp_path):
        schema = ConfigSchema(required_fields=["name"], optional_fields=["desc"])
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text("name: test\ndesc: a description\n")
        result = validate_config_file(str(cfg_file), schema)
        assert result.valid is True


# ---------------------------------------------------------------------------
# format_validation_result
# ---------------------------------------------------------------------------


class TestFormatValidationResult:
    def test_passed_no_warnings(self):
        result = ValidationResult(errors=[], warnings=[], valid=True)
        txt = format_validation_result(result)
        assert "PASSED" in txt
        assert "No errors" in txt

    def test_passed_with_warnings(self):
        result = ValidationResult(
            errors=[],
            warnings=[ConfigError(field="old", error_type="deprecated", message="dep", suggestion="fix")],
            valid=True,
        )
        txt = format_validation_result(result)
        assert "PASSED" in txt
        assert "warnings" in txt.lower()

    def test_failed_shows_errors(self):
        result = ValidationResult(
            errors=[ConfigError(field="x", error_type="unknown_key", message="Unknown x")],
            warnings=[],
            valid=False,
        )
        txt = format_validation_result(result)
        assert "FAILED" in txt
        assert "Unknown x" in txt

    def test_shows_suggestions(self):
        result = ValidationResult(
            errors=[ConfigError(
                field="y", error_type="missing_required",
                message="Missing y", suggestion="Add y",
            )],
            warnings=[],
            valid=False,
        )
        txt = format_validation_result(result)
        assert "Add y" in txt

    def test_error_count_in_output(self):
        errors = [
            ConfigError(field="a", error_type="unknown_key", message="a"),
            ConfigError(field="b", error_type="unknown_key", message="b"),
        ]
        result = ValidationResult(errors=errors, warnings=[], valid=False)
        txt = format_validation_result(result)
        assert "Errors (2)" in txt

    def test_warning_count_in_output(self):
        warnings = [
            ConfigError(field="w", error_type="deprecated", message="w", suggestion="s"),
        ]
        result = ValidationResult(errors=[], warnings=warnings, valid=True)
        txt = format_validation_result(result)
        assert "Warnings (1)" in txt
