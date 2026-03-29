"""Tests for batch_script module."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from evalyn_sdk.batch_script import (
    BatchConfig,
    BatchResult,
    CommandResult,
    execute_batch,
    format_batch_report,
    load_script_file,
    parse_script,
    substitute_variables,
)


# =============================================================================
# BatchConfig
# =============================================================================


class TestBatchConfig:
    def test_defaults(self):
        c = BatchConfig()
        assert c.stop_on_error is True
        assert c.dry_run is False
        assert c.variables == {}

    def test_custom_values(self):
        c = BatchConfig(stop_on_error=False, dry_run=True, variables={"k": "v"})
        assert c.stop_on_error is False
        assert c.dry_run is True
        assert c.variables == {"k": "v"}

    def test_as_dict(self):
        c = BatchConfig(variables={"x": "1"})
        d = c.as_dict()
        assert d["stop_on_error"] is True
        assert d["dry_run"] is False
        assert d["variables"] == {"x": "1"}

    def test_from_dict_full(self):
        c = BatchConfig.from_dict({"stop_on_error": False, "dry_run": True, "variables": {"a": "b"}})
        assert c.stop_on_error is False
        assert c.dry_run is True
        assert c.variables == {"a": "b"}

    def test_from_dict_empty(self):
        c = BatchConfig.from_dict({})
        assert c.stop_on_error is True
        assert c.dry_run is False
        assert c.variables == {}

    def test_roundtrip(self):
        original = BatchConfig(stop_on_error=False, variables={"env": "prod"})
        restored = BatchConfig.from_dict(original.as_dict())
        assert restored.stop_on_error == original.stop_on_error
        assert restored.dry_run == original.dry_run
        assert restored.variables == original.variables

    def test_variables_dict_is_copy(self):
        """as_dict should return a copy of variables, not the same object."""
        c = BatchConfig(variables={"a": "1"})
        d = c.as_dict()
        d["variables"]["b"] = "2"
        assert "b" not in c.variables


# =============================================================================
# CommandResult
# =============================================================================


class TestCommandResult:
    def test_defaults(self):
        r = CommandResult(command="echo hi")
        assert r.exit_code == 0
        assert r.output == ""
        assert r.duration_seconds == 0.0
        assert r.skipped is False

    def test_custom_values(self):
        r = CommandResult(command="fail", exit_code=1, output="err", duration_seconds=1.5, skipped=True)
        assert r.exit_code == 1
        assert r.output == "err"
        assert r.duration_seconds == 1.5
        assert r.skipped is True

    def test_as_dict(self):
        r = CommandResult(command="test", exit_code=2, output="out")
        d = r.as_dict()
        assert d["command"] == "test"
        assert d["exit_code"] == 2
        assert d["output"] == "out"

    def test_from_dict(self):
        r = CommandResult.from_dict({"command": "run", "exit_code": 0, "output": "ok"})
        assert r.command == "run"
        assert r.exit_code == 0

    def test_from_dict_minimal(self):
        r = CommandResult.from_dict({"command": "x"})
        assert r.command == "x"
        assert r.exit_code == 0
        assert r.skipped is False

    def test_roundtrip(self):
        original = CommandResult(command="cmd", exit_code=3, output="data", duration_seconds=0.5)
        restored = CommandResult.from_dict(original.as_dict())
        assert restored.command == original.command
        assert restored.exit_code == original.exit_code
        assert restored.output == original.output
        assert restored.duration_seconds == original.duration_seconds


# =============================================================================
# BatchResult
# =============================================================================


class TestBatchResult:
    def test_defaults(self):
        b = BatchResult()
        assert b.results == []
        assert b.total_commands == 0
        assert b.succeeded == 0
        assert b.failed == 0
        assert b.skipped == 0

    def test_as_dict(self):
        r = CommandResult(command="echo 1")
        b = BatchResult(results=[r], total_commands=1, succeeded=1)
        d = b.as_dict()
        assert len(d["results"]) == 1
        assert d["total_commands"] == 1
        assert d["succeeded"] == 1

    def test_from_dict(self):
        data = {
            "results": [{"command": "a"}, {"command": "b", "exit_code": 1}],
            "total_commands": 2,
            "succeeded": 1,
            "failed": 1,
            "skipped": 0,
        }
        b = BatchResult.from_dict(data)
        assert len(b.results) == 2
        assert b.results[0].command == "a"
        assert b.results[1].exit_code == 1

    def test_from_dict_empty(self):
        b = BatchResult.from_dict({})
        assert b.results == []
        assert b.total_commands == 0

    def test_roundtrip(self):
        results = [CommandResult(command="a"), CommandResult(command="b", exit_code=1)]
        original = BatchResult(results=results, total_commands=2, succeeded=1, failed=1, skipped=0)
        restored = BatchResult.from_dict(original.as_dict())
        assert len(restored.results) == 2
        assert restored.total_commands == original.total_commands
        assert restored.succeeded == original.succeeded


# =============================================================================
# parse_script
# =============================================================================


class TestParseScript:
    def test_simple(self):
        content = "echo hello\necho world"
        assert parse_script(content) == ["echo hello", "echo world"]

    def test_skip_blank_lines(self):
        content = "echo a\n\n\necho b\n"
        assert parse_script(content) == ["echo a", "echo b"]

    def test_skip_comments(self):
        content = "# comment\necho a\n  # indented comment\necho b"
        assert parse_script(content) == ["echo a", "echo b"]

    def test_strip_whitespace(self):
        content = "  echo a  \n\techo b\t"
        assert parse_script(content) == ["echo a", "echo b"]

    def test_empty_content(self):
        assert parse_script("") == []

    def test_only_comments(self):
        assert parse_script("# just\n# comments") == []

    def test_only_blank_lines(self):
        assert parse_script("\n\n\n") == []

    def test_mixed(self):
        content = """# Script
echo start

# step 1
run step1
# step 2
run step2

echo done
"""
        assert parse_script(content) == ["echo start", "run step1", "run step2", "echo done"]


# =============================================================================
# substitute_variables
# =============================================================================


class TestSubstituteVariables:
    def test_dollar_var(self):
        assert substitute_variables("echo $NAME", {"NAME": "alice"}) == "echo alice"

    def test_braced_var(self):
        assert substitute_variables("echo ${NAME}", {"NAME": "bob"}) == "echo bob"

    def test_multiple_vars(self):
        result = substitute_variables("$A and $B", {"A": "1", "B": "2"})
        assert result == "1 and 2"

    def test_unknown_var_left_intact(self):
        assert substitute_variables("$UNKNOWN", {}) == "$UNKNOWN"

    def test_unknown_braced_var_left_intact(self):
        assert substitute_variables("${UNKNOWN}", {}) == "${UNKNOWN}"

    def test_builtin_date(self):
        result = substitute_variables("date=$DATE", {})
        # Should be a YYYY-MM-DD format date
        import re
        assert re.match(r"date=\d{4}-\d{2}-\d{2}$", result)

    def test_builtin_timestamp(self):
        result = substitute_variables("ts=$TIMESTAMP", {})
        assert result.startswith("ts=")
        assert "T" in result  # ISO format has T separator

    def test_override_builtin(self):
        result = substitute_variables("$DATE", {"DATE": "custom"})
        assert result == "custom"

    def test_no_vars(self):
        assert substitute_variables("plain text", {"X": "1"}) == "plain text"

    def test_adjacent_vars(self):
        assert substitute_variables("$A$B", {"A": "x", "B": "y"}) == "xy"

    def test_var_in_middle(self):
        assert substitute_variables("pre-$VAR-post", {"VAR": "mid"}) == "pre-mid-post"

    def test_braced_and_unbraced_mix(self):
        result = substitute_variables("${A} $B", {"A": "1", "B": "2"})
        assert result == "1 2"


# =============================================================================
# execute_batch
# =============================================================================


def _ok_executor(cmd: str) -> tuple[int, str]:
    return (0, f"ran: {cmd}")


def _fail_executor(cmd: str) -> tuple[int, str]:
    return (1, "error")


def _mixed_executor(cmd: str) -> tuple[int, str]:
    if "fail" in cmd:
        return (1, "failed")
    return (0, "ok")


class TestExecuteBatch:
    def test_all_succeed(self):
        result = execute_batch(["a", "b", "c"], _ok_executor)
        assert result.total_commands == 3
        assert result.succeeded == 3
        assert result.failed == 0
        assert result.skipped == 0

    def test_stop_on_error(self):
        result = execute_batch(
            ["ok", "fail", "after"],
            _mixed_executor,
            BatchConfig(stop_on_error=True),
        )
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.skipped == 1
        assert result.results[2].skipped is True

    def test_continue_on_error(self):
        result = execute_batch(
            ["ok", "fail", "ok2"],
            _mixed_executor,
            BatchConfig(stop_on_error=False),
        )
        assert result.succeeded == 2
        assert result.failed == 1
        assert result.skipped == 0

    def test_dry_run(self):
        calls = []

        def track(cmd: str) -> tuple[int, str]:
            calls.append(cmd)
            return (0, "")

        result = execute_batch(
            ["a", "b"],
            track,
            BatchConfig(dry_run=True),
        )
        assert calls == []
        assert result.skipped == 2
        assert result.succeeded == 0

    def test_variable_substitution(self):
        captured = []

        def capture(cmd: str) -> tuple[int, str]:
            captured.append(cmd)
            return (0, "")

        execute_batch(
            ["echo $ENV"],
            capture,
            BatchConfig(variables={"ENV": "prod"}),
        )
        assert captured == ["echo prod"]

    def test_empty_commands(self):
        result = execute_batch([], _ok_executor)
        assert result.total_commands == 0
        assert result.succeeded == 0

    def test_default_config(self):
        result = execute_batch(["a"], _ok_executor)
        assert result.total_commands == 1
        assert result.succeeded == 1

    def test_duration_recorded(self):
        result = execute_batch(["a"], _ok_executor)
        assert result.results[0].duration_seconds >= 0

    def test_output_captured(self):
        result = execute_batch(["cmd"], _ok_executor)
        assert result.results[0].output == "ran: cmd"

    def test_multiple_failures_stop_on_error(self):
        """With stop_on_error, only the first failure is recorded, rest skipped."""
        result = execute_batch(
            ["fail1", "fail2", "fail3"],
            _fail_executor,
            BatchConfig(stop_on_error=True),
        )
        assert result.failed == 1
        assert result.skipped == 2

    def test_all_fail_no_stop(self):
        result = execute_batch(
            ["a", "b", "c"],
            _fail_executor,
            BatchConfig(stop_on_error=False),
        )
        assert result.failed == 3
        assert result.skipped == 0


# =============================================================================
# load_script_file
# =============================================================================


class TestLoadScriptFile:
    def test_load_from_file(self, tmp_path):
        script = tmp_path / "test.sh"
        script.write_text("# header\necho hello\necho world\n", encoding="utf-8")
        commands = load_script_file(str(script))
        assert commands == ["echo hello", "echo world"]

    def test_load_empty_file(self, tmp_path):
        script = tmp_path / "empty.sh"
        script.write_text("", encoding="utf-8")
        assert load_script_file(str(script)) == []

    def test_load_comments_only(self, tmp_path):
        script = tmp_path / "comments.sh"
        script.write_text("# only comments\n# here\n", encoding="utf-8")
        assert load_script_file(str(script)) == []

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_script_file("/nonexistent/path/script.sh")


# =============================================================================
# format_batch_report
# =============================================================================


class TestFormatBatchReport:
    def test_empty_report(self):
        report = format_batch_report(BatchResult())
        assert "Total: 0" in report
        assert "Succeeded: 0" in report

    def test_success_report(self):
        result = BatchResult(
            results=[CommandResult(command="echo hi", exit_code=0, output="hi")],
            total_commands=1,
            succeeded=1,
        )
        report = format_batch_report(result)
        assert "OK" in report
        assert "echo hi" in report
        assert "Succeeded: 1" in report

    def test_failure_report(self):
        result = BatchResult(
            results=[CommandResult(command="bad", exit_code=2, output="err")],
            total_commands=1,
            failed=1,
        )
        report = format_batch_report(result)
        assert "FAIL" in report
        assert "exit 2" in report

    def test_skipped_report(self):
        result = BatchResult(
            results=[CommandResult(command="skipped_cmd", skipped=True)],
            total_commands=1,
            skipped=1,
        )
        report = format_batch_report(result)
        assert "SKIP" in report

    def test_duration_shown(self):
        result = BatchResult(
            results=[CommandResult(command="slow", duration_seconds=1.234)],
            total_commands=1,
            succeeded=1,
        )
        report = format_batch_report(result)
        assert "1.234s" in report

    def test_multiline_output(self):
        result = BatchResult(
            results=[CommandResult(command="multi", output="line1\nline2")],
            total_commands=1,
            succeeded=1,
        )
        report = format_batch_report(result)
        assert "line1" in report
        assert "line2" in report

    def test_header_present(self):
        report = format_batch_report(BatchResult())
        assert "Batch Execution Report" in report

    def test_numbering(self):
        results = [
            CommandResult(command="a"),
            CommandResult(command="b"),
            CommandResult(command="c"),
        ]
        report = format_batch_report(BatchResult(results=results, total_commands=3, succeeded=3))
        assert "[1]" in report
        assert "[2]" in report
        assert "[3]" in report
