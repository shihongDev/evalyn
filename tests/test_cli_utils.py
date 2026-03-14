"""Tests for CLI utility modules: formatters, validation, config, command_common."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# formatters.py
# ---------------------------------------------------------------------------

from evalyn_sdk.cli.utils.formatters import (
    format_cost,
    is_json_format,
    output_json,
    print_error_if_table,
    print_if_table,
    print_table,
    print_token_usage_summary,
)


class TestFormatCost:
    def test_sub_dollar_uses_four_decimals(self):
        assert format_cost(0.0042) == "$0.0042"

    def test_dollar_and_above_uses_two_decimals(self):
        assert format_cost(3.1415) == "$3.14"

    def test_exactly_one_dollar(self):
        assert format_cost(1.0) == "$1.00"

    def test_zero_cost(self):
        assert format_cost(0.0) == "$0.0000"

    def test_large_cost(self):
        assert format_cost(123.456) == "$123.46"


class TestIsJsonFormat:
    def test_json_format(self):
        args = type("Args", (), {"format": "json"})()
        assert is_json_format(args) is True

    def test_table_format(self):
        args = type("Args", (), {"format": "table"})()
        assert is_json_format(args) is False

    def test_no_format_attr_defaults_table(self):
        args = type("Args", (), {})()
        assert is_json_format(args) is False


class TestPrintIfTable:
    def test_prints_when_table_format(self, capsys):
        args = type("Args", (), {"format": "table"})()
        print_if_table(args, "hello")
        assert "hello" in capsys.readouterr().out

    def test_silent_when_json_format(self, capsys):
        args = type("Args", (), {"format": "json"})()
        print_if_table(args, "hello")
        assert capsys.readouterr().out == ""


class TestPrintErrorIfTable:
    def test_prints_to_stderr_when_table(self, capsys):
        args = type("Args", (), {"format": "table"})()
        print_error_if_table(args, "error msg")
        assert "error msg" in capsys.readouterr().err

    def test_silent_when_json(self, capsys):
        args = type("Args", (), {"format": "json"})()
        print_error_if_table(args, "error msg")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestOutputJson:
    def test_valid_json_output(self, capsys):
        output_json({"key": "value", "num": 42})
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42


class TestPrintTable:
    def test_basic_table(self, capsys):
        headers = ["Name", "Score"]
        rows = [["Alice", "95"], ["Bob", "87"]]
        print_table(headers, rows)
        out = capsys.readouterr().out
        assert "Alice" in out
        assert "Bob" in out
        assert "Name" in out
        assert "Score" in out
        # Should have separator line
        assert "---" in out

    def test_empty_rows(self, capsys):
        print_table(["Col1"], [])
        out = capsys.readouterr().out
        assert "Col1" in out

    def test_auto_width_expands_for_long_values(self, capsys):
        headers = ["ID"]
        rows = [["abcdefghij1234567890"]]
        print_table(headers, rows)
        out = capsys.readouterr().out
        assert "abcdefghij1234567890" in out


class TestPrintTokenUsageSummary:
    def test_empty_usage_prints_nothing(self, capsys):
        print_token_usage_summary({})
        assert capsys.readouterr().out == ""

    def test_zero_tokens_prints_nothing(self, capsys):
        print_token_usage_summary({"total_tokens": 0})
        assert capsys.readouterr().out == ""

    def test_basic_usage_summary(self, capsys):
        usage = {
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_tokens": 1500,
            "total_cost_usd": 0.0035,
            "models_used": ["gpt-4o-mini"],
        }
        print_token_usage_summary(usage)
        out = capsys.readouterr().out
        assert "1,000" in out
        assert "500" in out
        assert "1,500" in out
        assert "$0.0035" in out
        assert "gpt-4o-mini" in out

    def test_unknown_pricing_shows_asterisk(self, capsys):
        usage = {
            "total_tokens": 100,
            "total_cost_usd": 0.01,
            "has_unknown_pricing": True,
        }
        print_token_usage_summary(usage)
        out = capsys.readouterr().out
        assert "*" in out
        assert "approximate" in out.lower()

    def test_verbose_cost_breakdown(self, capsys):
        usage = {
            "total_tokens": 500,
            "total_input_tokens": 300,
            "total_output_tokens": 200,
            "total_cost_usd": 0.10,
            "cost_by_metric": {
                "helpfulness_accuracy": 0.07,
                "toxicity_safety": 0.03,
            },
        }
        print_token_usage_summary(usage, verbose=True)
        out = capsys.readouterr().out
        assert "helpfulness_accuracy" in out
        assert "toxicity_safety" in out


# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------

from evalyn_sdk.cli.utils.validation import extract_project_id


class TestExtractProjectId:
    def test_project_id_key(self):
        assert extract_project_id({"project_id": "proj-123"}) == "proj-123"

    def test_project_key(self):
        assert extract_project_id({"project": "my-project"}) == "my-project"

    def test_project_name_key(self):
        assert extract_project_id({"project_name": "named-proj"}) == "named-proj"

    def test_priority_order(self):
        # project_id takes precedence
        meta = {"project_id": "first", "project": "second", "project_name": "third"}
        assert extract_project_id(meta) == "first"

    def test_empty_dict(self):
        assert extract_project_id({}) is None

    def test_non_dict_input(self):
        assert extract_project_id("not a dict") is None
        assert extract_project_id(None) is None

    def test_none_values_fall_through(self):
        meta = {"project_id": None, "project": "fallback"}
        assert extract_project_id(meta) == "fallback"


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

from evalyn_sdk.cli.utils.config import (
    _expand_env_vars,
    get_config_default,
    resolve_dataset_path,
)


class TestExpandEnvVars:
    def test_dollar_brace_syntax(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            assert _expand_env_vars("prefix-${MY_VAR}-suffix") == "prefix-hello-suffix"

    def test_dollar_prefix_syntax(self):
        with patch.dict(os.environ, {"FOO": "bar"}):
            assert _expand_env_vars("$FOO") == "bar"

    def test_unset_var_unchanged(self):
        result = _expand_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert "${NONEXISTENT_VAR_XYZ}" in result

    def test_dict_recursive(self):
        with patch.dict(os.environ, {"DB_HOST": "localhost"}):
            config = {"database": {"host": "$DB_HOST", "port": 5432}}
            expanded = _expand_env_vars(config)
            assert expanded["database"]["host"] == "localhost"
            assert expanded["database"]["port"] == 5432

    def test_list_recursive(self):
        with patch.dict(os.environ, {"ITEM": "val"}):
            result = _expand_env_vars(["$ITEM", "literal"])
            assert result == ["val", "literal"]

    def test_non_string_passthrough(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(True) is True


class TestGetConfigDefault:
    def test_nested_key_access(self):
        config = {"a": {"b": {"c": "deep"}}}
        assert get_config_default(config, "a", "b", "c") == "deep"

    def test_missing_key_returns_default(self):
        config = {"a": {"b": 1}}
        assert get_config_default(config, "a", "x", default="fallback") == "fallback"

    def test_none_intermediate_returns_default(self):
        config = {"a": None}
        assert get_config_default(config, "a", "b", default=42) == 42

    def test_empty_config(self):
        assert get_config_default({}, "x", default="empty") == "empty"


class TestResolveDatasetPath:
    def test_explicit_dir_path(self, tmp_path):
        dataset_dir = tmp_path / "my-dataset"
        dataset_dir.mkdir()
        result = resolve_dataset_path(str(dataset_dir))
        assert result == dataset_dir

    def test_explicit_file_path(self, tmp_path):
        dataset_file = tmp_path / "data" / "dataset.jsonl"
        dataset_file.parent.mkdir(parents=True)
        dataset_file.touch()
        result = resolve_dataset_path(str(dataset_file))
        assert result == dataset_file.parent

    def test_none_arg_returns_none(self):
        result = resolve_dataset_path(None)
        assert result is None

    def test_config_default_dataset(self, tmp_path):
        config = {"defaults": {"dataset": str(tmp_path / "from-config")}}
        result = resolve_dataset_path(None, config=config)
        assert result == tmp_path / "from-config"


# ---------------------------------------------------------------------------
# command_common.py
# ---------------------------------------------------------------------------

from evalyn_sdk.cli.utils.command_common import (
    resolve_call_id,
    try_resolve_call_id,
)


class TestResolveCallId:
    def test_with_resolver_returns_resolved(self):
        class MockStorage:
            def resolve_call_id(self, input_id):
                if input_id == "abc":
                    return "abc-full-uuid-1234"
                return None
        assert resolve_call_id(MockStorage(), "abc") == "abc-full-uuid-1234"

    def test_no_resolver_returns_input(self):
        class NoResolverStorage:
            pass
        assert resolve_call_id(NoResolverStorage(), "raw-id") == "raw-id"

    def test_unresolved_returns_input(self):
        class MockStorage:
            def resolve_call_id(self, input_id):
                return None
        assert resolve_call_id(MockStorage(), "unknown") == "unknown"


class TestTryResolveCallId:
    def test_resolved(self):
        class MockStorage:
            def resolve_call_id(self, input_id):
                return "full-id"
        assert try_resolve_call_id(MockStorage(), "short") == "full-id"

    def test_not_found_returns_none(self):
        class MockStorage:
            def resolve_call_id(self, input_id):
                return None
        assert try_resolve_call_id(MockStorage(), "nope") is None

    def test_no_resolver_returns_input(self):
        class PlainStorage:
            pass
        assert try_resolve_call_id(PlainStorage(), "id") == "id"
