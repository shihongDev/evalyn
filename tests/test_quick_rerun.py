"""Tests for CLI quick rerun."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.quick_rerun import (
    RerunConfig,
    RerunRequest,
    build_rerun_request,
    format_rerun_preview,
    get_last_command,
    is_destructive_command,
    merge_args,
)


# ---------------------------------------------------------------------------
# RerunConfig dataclass
# ---------------------------------------------------------------------------


class TestRerunConfig:
    def test_defaults(self):
        c = RerunConfig()
        assert c.history_path == ".evalyn/history.jsonl"
        assert c.allow_destructive is False

    def test_custom_values(self):
        c = RerunConfig(history_path="/tmp/h.jsonl", allow_destructive=True)
        assert c.history_path == "/tmp/h.jsonl"
        assert c.allow_destructive is True

    def test_as_dict(self):
        c = RerunConfig()
        d = c.as_dict()
        assert d["history_path"] == ".evalyn/history.jsonl"
        assert d["allow_destructive"] is False

    def test_from_dict(self):
        c = RerunConfig.from_dict({"history_path": "/x", "allow_destructive": True})
        assert c.history_path == "/x"
        assert c.allow_destructive is True

    def test_from_dict_defaults(self):
        c = RerunConfig.from_dict({})
        assert c.history_path == ".evalyn/history.jsonl"
        assert c.allow_destructive is False

    def test_roundtrip(self):
        original = RerunConfig(history_path="/custom/path.jsonl", allow_destructive=True)
        restored = RerunConfig.from_dict(original.as_dict())
        assert restored.history_path == original.history_path
        assert restored.allow_destructive == original.allow_destructive


# ---------------------------------------------------------------------------
# RerunRequest dataclass
# ---------------------------------------------------------------------------


class TestRerunRequest:
    def test_basic_creation(self):
        r = RerunRequest(
            original_command="run",
            original_args=["--fast"],
            override_args=["--slow"],
            merged_command="evalyn run --slow",
        )
        assert r.original_command == "run"
        assert r.original_args == ["--fast"]

    def test_as_dict(self):
        r = RerunRequest(
            original_command="analyze",
            original_args=["--format", "json"],
            override_args=["--format", "csv"],
            merged_command="evalyn analyze --format csv",
        )
        d = r.as_dict()
        assert d["original_command"] == "analyze"
        assert d["override_args"] == ["--format", "csv"]

    def test_from_dict(self):
        data = {
            "original_command": "run",
            "original_args": ["a"],
            "override_args": ["b"],
            "merged_command": "evalyn run b",
        }
        r = RerunRequest.from_dict(data)
        assert r.original_command == "run"
        assert r.merged_command == "evalyn run b"

    def test_from_dict_defaults(self):
        r = RerunRequest.from_dict({})
        assert r.original_command == ""
        assert r.original_args == []
        assert r.override_args == []
        assert r.merged_command == ""

    def test_roundtrip(self):
        original = RerunRequest(
            original_command="compare",
            original_args=["--run", "abc"],
            override_args=["--run", "xyz"],
            merged_command="evalyn compare --run xyz",
        )
        restored = RerunRequest.from_dict(original.as_dict())
        assert restored.original_command == original.original_command
        assert restored.original_args == original.original_args
        assert restored.override_args == original.override_args
        assert restored.merged_command == original.merged_command


# ---------------------------------------------------------------------------
# get_last_command
# ---------------------------------------------------------------------------


class TestGetLastCommand:
    def test_missing_file(self, tmp_path):
        result = get_last_command(str(tmp_path / "nonexistent.jsonl"))
        assert result is None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text("")
        assert get_last_command(str(p)) is None

    def test_blank_lines_only(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text("\n\n\n")
        assert get_last_command(str(p)) is None

    def test_single_entry(self, tmp_path):
        p = tmp_path / "history.jsonl"
        entry = {"command": "run", "args": ["--fast"]}
        p.write_text(json.dumps(entry) + "\n")
        result = get_last_command(str(p))
        assert result == ("run", ["--fast"])

    def test_multiple_entries_returns_last(self, tmp_path):
        p = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"command": "run", "args": ["--v1"]}),
            json.dumps({"command": "analyze", "args": ["--v2"]}),
            json.dumps({"command": "compare", "args": ["--v3"]}),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = get_last_command(str(p))
        assert result == ("compare", ["--v3"])

    def test_entry_without_args(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text(json.dumps({"command": "status"}) + "\n")
        result = get_last_command(str(p))
        assert result == ("status", [])

    def test_trailing_blank_lines(self, tmp_path):
        p = tmp_path / "history.jsonl"
        entry = json.dumps({"command": "run", "args": []})
        p.write_text(entry + "\n\n\n")
        result = get_last_command(str(p))
        assert result == ("run", [])


# ---------------------------------------------------------------------------
# merge_args
# ---------------------------------------------------------------------------


class TestMergeArgs:
    def test_no_overrides(self):
        assert merge_args(["--fast", "--verbose"], []) == ["--fast", "--verbose"]

    def test_no_originals(self):
        assert merge_args([], ["--fast"]) == ["--fast"]

    def test_replace_flag_with_equals(self):
        result = merge_args(["--format=json"], ["--format=csv"])
        assert result == ["--format=csv"]

    def test_replace_flag_value_pair(self):
        result = merge_args(["--format", "json"], ["--format", "csv"])
        assert result == ["--format", "csv"]

    def test_replace_equals_with_pair(self):
        result = merge_args(["--format=json"], ["--format", "csv"])
        assert result == ["--format", "csv"]

    def test_replace_pair_with_equals(self):
        result = merge_args(["--format", "json"], ["--format=csv"])
        assert result == ["--format=csv"]

    def test_append_new_flag(self):
        result = merge_args(["--fast"], ["--verbose"])
        assert result == ["--fast", "--verbose"]

    def test_append_new_flag_with_value(self):
        result = merge_args(["--fast"], ["--output", "file.txt"])
        assert result == ["--fast", "--output", "file.txt"]

    def test_mixed_replace_and_append(self):
        result = merge_args(
            ["--format", "json", "--verbose"],
            ["--format", "csv", "--output", "out.txt"],
        )
        assert "--format" in result
        assert "csv" in result
        assert "--verbose" in result
        assert "--output" in result
        assert "out.txt" in result

    def test_positional_args_preserved(self):
        result = merge_args(["path/to/data"], ["--fast"])
        assert result == ["path/to/data", "--fast"]

    def test_short_flags(self):
        result = merge_args(["-v"], ["-q"])
        assert result == ["-v", "-q"]

    def test_replace_short_flag_value(self):
        result = merge_args(["-f", "json"], ["-f", "csv"])
        assert result == ["-f", "csv"]

    def test_boolean_flag_override(self):
        result = merge_args(["--verbose", "--fast"], ["--verbose"])
        # --verbose already present, so it replaces itself (same)
        assert result.count("--verbose") == 1
        assert "--fast" in result

    def test_empty_both(self):
        assert merge_args([], []) == []

    def test_multiple_replacements(self):
        result = merge_args(
            ["--format", "json", "--output", "a.txt"],
            ["--format", "csv", "--output", "b.txt"],
        )
        assert result == ["--format", "csv", "--output", "b.txt"]


# ---------------------------------------------------------------------------
# build_rerun_request
# ---------------------------------------------------------------------------


class TestBuildRerunRequest:
    def test_no_history(self, tmp_path):
        result = build_rerun_request(str(tmp_path / "nope.jsonl"), ["--fast"])
        assert result is None

    def test_basic_rerun(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text(json.dumps({"command": "run", "args": ["--format", "json"]}) + "\n")
        req = build_rerun_request(str(p), ["--format", "csv"])
        assert req is not None
        assert req.original_command == "run"
        assert req.original_args == ["--format", "json"]
        assert req.override_args == ["--format", "csv"]
        assert "evalyn run" in req.merged_command
        assert "csv" in req.merged_command

    def test_no_overrides(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text(json.dumps({"command": "status", "args": []}) + "\n")
        req = build_rerun_request(str(p), [])
        assert req is not None
        assert req.merged_command == "evalyn status"

    def test_adds_new_flags(self, tmp_path):
        p = tmp_path / "history.jsonl"
        p.write_text(json.dumps({"command": "run", "args": []}) + "\n")
        req = build_rerun_request(str(p), ["--verbose"])
        assert req is not None
        assert "--verbose" in req.merged_command


# ---------------------------------------------------------------------------
# format_rerun_preview
# ---------------------------------------------------------------------------


class TestFormatRerunPreview:
    def test_basic_format(self):
        req = RerunRequest(
            original_command="run",
            original_args=["--fast"],
            override_args=[],
            merged_command="evalyn run --fast",
        )
        assert format_rerun_preview(req) == "Rerunning: evalyn run --fast"

    def test_empty_merged(self):
        req = RerunRequest(
            original_command="status",
            original_args=[],
            override_args=[],
            merged_command="evalyn status",
        )
        assert format_rerun_preview(req) == "Rerunning: evalyn status"

    def test_complex_command(self):
        req = RerunRequest(
            original_command="analyze",
            original_args=["--format", "json", "--verbose"],
            override_args=["--format", "csv"],
            merged_command="evalyn analyze --format csv --verbose",
        )
        preview = format_rerun_preview(req)
        assert preview.startswith("Rerunning: ")
        assert "evalyn analyze" in preview


# ---------------------------------------------------------------------------
# is_destructive_command
# ---------------------------------------------------------------------------


class TestIsDestructiveCommand:
    def test_gc(self):
        assert is_destructive_command("gc") is True

    def test_delete(self):
        assert is_destructive_command("delete") is True

    def test_reset(self):
        assert is_destructive_command("reset") is True

    def test_case_insensitive(self):
        assert is_destructive_command("GC") is True
        assert is_destructive_command("Delete") is True
        assert is_destructive_command("RESET") is True

    def test_safe_commands(self):
        assert is_destructive_command("run") is False
        assert is_destructive_command("analyze") is False
        assert is_destructive_command("compare") is False
        assert is_destructive_command("status") is False
        assert is_destructive_command("") is False

    def test_partial_match_not_destructive(self):
        assert is_destructive_command("garbage") is False
        assert is_destructive_command("deleted") is False
