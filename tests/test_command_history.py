"""Tests for CLI command history."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.command_history import (
    HistoryEntry,
    HistoryLog,
    append_history,
    create_history_entry,
    filter_history,
    format_history,
    get_replay_sequence,
    read_history,
)


# ---------------------------------------------------------------------------
# HistoryEntry dataclass
# ---------------------------------------------------------------------------


class TestHistoryEntry:
    def test_basic_creation(self):
        e = HistoryEntry(command="run", args=["--fast"], timestamp="2026-01-01T00:00:00+00:00")
        assert e.command == "run"
        assert e.args == ["--fast"]
        assert e.exit_code == 0
        assert e.duration_seconds == 0.0

    def test_custom_exit_code(self):
        e = HistoryEntry(command="run", args=[], timestamp="t", exit_code=1)
        assert e.exit_code == 1

    def test_custom_duration(self):
        e = HistoryEntry(command="run", args=[], timestamp="t", duration_seconds=3.5)
        assert e.duration_seconds == 3.5

    def test_as_dict(self):
        e = HistoryEntry(
            command="run",
            args=["--fast"],
            timestamp="2026-01-01T00:00:00+00:00",
            exit_code=0,
            duration_seconds=1.2,
        )
        d = e.as_dict()
        assert d["command"] == "run"
        assert d["args"] == ["--fast"]
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["exit_code"] == 0
        assert d["duration_seconds"] == 1.2

    def test_from_dict(self):
        d = {
            "command": "analyze",
            "args": ["--run", "abc"],
            "timestamp": "2026-03-01T12:00:00+00:00",
            "exit_code": 0,
            "duration_seconds": 2.0,
        }
        e = HistoryEntry.from_dict(d)
        assert e.command == "analyze"
        assert e.args == ["--run", "abc"]

    def test_from_dict_defaults(self):
        d = {"command": "run", "timestamp": "t"}
        e = HistoryEntry.from_dict(d)
        assert e.args == []
        assert e.exit_code == 0
        assert e.duration_seconds == 0.0

    def test_roundtrip(self):
        original = HistoryEntry(
            command="eval", args=["--workers", "4"], timestamp="2026-06-01T00:00:00+00:00",
            exit_code=2, duration_seconds=5.5,
        )
        restored = HistoryEntry.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# HistoryLog dataclass
# ---------------------------------------------------------------------------


class TestHistoryLog:
    def test_empty_log(self):
        log = HistoryLog()
        assert log.entries == []
        assert log.log_path == ""

    def test_as_dict(self):
        entry = HistoryEntry(command="run", args=[], timestamp="t")
        log = HistoryLog(entries=[entry], log_path="/tmp/h.jsonl")
        d = log.as_dict()
        assert len(d["entries"]) == 1
        assert d["log_path"] == "/tmp/h.jsonl"

    def test_from_dict(self):
        d = {
            "entries": [{"command": "run", "args": [], "timestamp": "t"}],
            "log_path": "/tmp/h.jsonl",
        }
        log = HistoryLog.from_dict(d)
        assert len(log.entries) == 1
        assert log.log_path == "/tmp/h.jsonl"

    def test_from_dict_defaults(self):
        log = HistoryLog.from_dict({})
        assert log.entries == []
        assert log.log_path == ""

    def test_roundtrip(self):
        entry = HistoryEntry(command="run", args=["a"], timestamp="t", exit_code=1)
        log = HistoryLog(entries=[entry], log_path="/p")
        restored = HistoryLog.from_dict(log.as_dict())
        assert len(restored.entries) == 1
        assert restored.entries[0].command == "run"
        assert restored.log_path == "/p"


# ---------------------------------------------------------------------------
# append_history / read_history (file I/O)
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_append_creates_file(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        entry = HistoryEntry(command="run", args=[], timestamp="2026-01-01T00:00:00+00:00")
        append_history(log_path, entry)
        assert Path(log_path).exists()

    def test_append_creates_parent_dirs(self, tmp_path):
        log_path = str(tmp_path / "deep" / "nested" / "history.jsonl")
        entry = HistoryEntry(command="run", args=[], timestamp="t")
        append_history(log_path, entry)
        assert Path(log_path).exists()

    def test_read_missing_file(self, tmp_path):
        log_path = str(tmp_path / "missing.jsonl")
        log = read_history(log_path)
        assert log.entries == []
        assert log.log_path == log_path

    def test_write_then_read(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        entry = HistoryEntry(command="run", args=["--fast"], timestamp="2026-01-01T00:00:00+00:00")
        append_history(log_path, entry)
        log = read_history(log_path)
        assert len(log.entries) == 1
        assert log.entries[0].command == "run"
        assert log.entries[0].args == ["--fast"]

    def test_multiple_appends(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        for i in range(5):
            entry = HistoryEntry(command=f"cmd{i}", args=[], timestamp=f"2026-01-0{i+1}T00:00:00+00:00")
            append_history(log_path, entry)
        log = read_history(log_path)
        assert len(log.entries) == 5

    def test_read_limit(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        for i in range(10):
            entry = HistoryEntry(command=f"cmd{i}", args=[], timestamp=f"t{i}")
            append_history(log_path, entry)
        log = read_history(log_path, limit=3)
        assert len(log.entries) == 3
        # Should be the last 3 entries
        assert log.entries[0].command == "cmd7"
        assert log.entries[2].command == "cmd9"

    def test_read_limit_larger_than_entries(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        entry = HistoryEntry(command="run", args=[], timestamp="t")
        append_history(log_path, entry)
        log = read_history(log_path, limit=100)
        assert len(log.entries) == 1

    def test_json_lines_format(self, tmp_path):
        log_path = str(tmp_path / "history.jsonl")
        entry = HistoryEntry(command="run", args=["a", "b"], timestamp="t")
        append_history(log_path, entry)
        with open(log_path) as f:
            line = f.readline().strip()
        data = json.loads(line)
        assert data["command"] == "run"
        assert data["args"] == ["a", "b"]


# ---------------------------------------------------------------------------
# filter_history
# ---------------------------------------------------------------------------


class TestFilterHistory:
    def _make_log(self):
        entries = [
            HistoryEntry(command="run", args=[], timestamp="2026-01-01T00:00:00+00:00"),
            HistoryEntry(command="analyze", args=[], timestamp="2026-01-02T00:00:00+00:00"),
            HistoryEntry(command="run", args=["--fast"], timestamp="2026-01-03T00:00:00+00:00"),
            HistoryEntry(command="status", args=[], timestamp="2026-01-04T00:00:00+00:00"),
            HistoryEntry(command="run", args=[], timestamp="2026-01-05T00:00:00+00:00"),
        ]
        return HistoryLog(entries=entries, log_path="/tmp/h.jsonl")

    def test_no_filters(self):
        log = self._make_log()
        result = filter_history(log)
        assert len(result.entries) == 5

    def test_filter_by_command(self):
        log = self._make_log()
        result = filter_history(log, command="run")
        assert len(result.entries) == 3
        assert all(e.command == "run" for e in result.entries)

    def test_filter_by_command_no_match(self):
        log = self._make_log()
        result = filter_history(log, command="nonexistent")
        assert len(result.entries) == 0

    def test_filter_after(self):
        log = self._make_log()
        result = filter_history(log, after="2026-01-03T00:00:00+00:00")
        assert len(result.entries) == 3

    def test_filter_before(self):
        log = self._make_log()
        result = filter_history(log, before="2026-01-02T00:00:00+00:00")
        assert len(result.entries) == 2

    def test_filter_date_range(self):
        log = self._make_log()
        result = filter_history(
            log,
            after="2026-01-02T00:00:00+00:00",
            before="2026-01-04T00:00:00+00:00",
        )
        assert len(result.entries) == 3

    def test_filter_combined_command_and_date(self):
        log = self._make_log()
        result = filter_history(
            log,
            command="run",
            after="2026-01-02T00:00:00+00:00",
        )
        assert len(result.entries) == 2

    def test_preserves_log_path(self):
        log = self._make_log()
        result = filter_history(log, command="run")
        assert result.log_path == "/tmp/h.jsonl"

    def test_empty_log(self):
        log = HistoryLog(entries=[], log_path="")
        result = filter_history(log, command="run")
        assert len(result.entries) == 0


# ---------------------------------------------------------------------------
# get_replay_sequence
# ---------------------------------------------------------------------------


class TestGetReplaySequence:
    def test_from_start(self):
        entries = [
            HistoryEntry(command="a", args=[], timestamp="2026-01-01T00:00:00+00:00"),
            HistoryEntry(command="b", args=[], timestamp="2026-01-02T00:00:00+00:00"),
            HistoryEntry(command="c", args=[], timestamp="2026-01-03T00:00:00+00:00"),
        ]
        log = HistoryLog(entries=entries)
        seq = get_replay_sequence(log, "2026-01-01T00:00:00+00:00")
        assert len(seq) == 3

    def test_from_middle(self):
        entries = [
            HistoryEntry(command="a", args=[], timestamp="2026-01-01T00:00:00+00:00"),
            HistoryEntry(command="b", args=[], timestamp="2026-01-02T00:00:00+00:00"),
            HistoryEntry(command="c", args=[], timestamp="2026-01-03T00:00:00+00:00"),
        ]
        log = HistoryLog(entries=entries)
        seq = get_replay_sequence(log, "2026-01-02T00:00:00+00:00")
        assert len(seq) == 2
        assert seq[0].command == "b"

    def test_from_future(self):
        entries = [
            HistoryEntry(command="a", args=[], timestamp="2026-01-01T00:00:00+00:00"),
        ]
        log = HistoryLog(entries=entries)
        seq = get_replay_sequence(log, "2027-01-01T00:00:00+00:00")
        assert len(seq) == 0

    def test_empty_log(self):
        log = HistoryLog(entries=[])
        seq = get_replay_sequence(log, "2026-01-01T00:00:00+00:00")
        assert len(seq) == 0


# ---------------------------------------------------------------------------
# format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    def test_empty_log(self):
        log = HistoryLog(entries=[])
        assert format_history(log) == ""

    def test_header_present(self):
        entry = HistoryEntry(command="run", args=[], timestamp="2026-01-01T00:00:00+00:00")
        log = HistoryLog(entries=[entry])
        result = format_history(log)
        assert "TIMESTAMP" in result
        assert "COMMAND" in result
        assert "EXIT" in result
        assert "DURATION" in result
        assert "ARGS" in result

    def test_separator_line(self):
        entry = HistoryEntry(command="run", args=[], timestamp="t")
        log = HistoryLog(entries=[entry])
        result = format_history(log)
        lines = result.split("\n")
        assert len(lines) >= 3
        assert set(lines[1].strip()) == {"-"}

    def test_entry_values(self):
        entry = HistoryEntry(
            command="analyze", args=["--run", "abc"],
            timestamp="2026-03-01T12:00:00+00:00",
            exit_code=0, duration_seconds=1.5,
        )
        log = HistoryLog(entries=[entry])
        result = format_history(log)
        assert "analyze" in result
        assert "--run abc" in result
        assert "1.5s" in result

    def test_multiple_entries(self):
        entries = [
            HistoryEntry(command="run", args=[], timestamp="t1"),
            HistoryEntry(command="analyze", args=[], timestamp="t2"),
        ]
        log = HistoryLog(entries=entries)
        result = format_history(log)
        lines = result.split("\n")
        # header + separator + 2 data lines
        assert len(lines) == 4

    def test_nonzero_exit_code(self):
        entry = HistoryEntry(command="run", args=[], timestamp="t", exit_code=1)
        log = HistoryLog(entries=[entry])
        result = format_history(log)
        assert "1" in result


# ---------------------------------------------------------------------------
# create_history_entry
# ---------------------------------------------------------------------------


class TestCreateHistoryEntry:
    def test_auto_timestamp(self):
        entry = create_history_entry("run", ["--fast"])
        assert entry.command == "run"
        assert entry.args == ["--fast"]
        # Timestamp should be a valid ISO string parseable by datetime
        dt = datetime.fromisoformat(entry.timestamp)
        assert dt.tzinfo is not None

    def test_default_exit_code(self):
        entry = create_history_entry("run", [])
        assert entry.exit_code == 0

    def test_default_duration(self):
        entry = create_history_entry("run", [])
        assert entry.duration_seconds == 0.0

    def test_custom_exit_code(self):
        entry = create_history_entry("run", [], exit_code=1)
        assert entry.exit_code == 1

    def test_custom_duration(self):
        entry = create_history_entry("run", [], duration=3.7)
        assert entry.duration_seconds == 3.7

    def test_timestamp_is_recent(self):
        before = datetime.now(timezone.utc)
        entry = create_history_entry("run", [])
        after = datetime.now(timezone.utc)
        ts = datetime.fromisoformat(entry.timestamp)
        assert before <= ts <= after
