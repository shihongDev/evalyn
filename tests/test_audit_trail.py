"""Tests for audit_trail module."""

from __future__ import annotations

import json
import os

import pytest

from evalyn_sdk.audit_trail import (
    AuditEntry,
    AuditLog,
    append_entry,
    compute_config_hash,
    create_audit_entry,
    filter_entries,
    format_audit_log,
    get_audit_stats,
    read_audit_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entry():
    return AuditEntry(
        entry_id="abc-123",
        user="testuser",
        timestamp="2026-01-15T10:00:00+00:00",
        command="run-eval",
        args={"dataset": "data/test"},
        config_hash="aabbcc",
        result_summary="5 metrics evaluated",
        evalyn_version="0.1.0",
    )


@pytest.fixture
def sample_log(sample_entry):
    entry2 = AuditEntry(
        entry_id="def-456",
        user="otheruser",
        timestamp="2026-01-16T12:00:00+00:00",
        command="analyze",
        args={"run": "latest"},
        config_hash="ddeeff",
        result_summary="analysis complete",
        evalyn_version="0.1.0",
    )
    return AuditLog(entries=[sample_entry, entry2], log_path="/tmp/test.jsonl")


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


class TestAuditEntry:
    def test_as_dict(self, sample_entry):
        d = sample_entry.as_dict()
        assert d["entry_id"] == "abc-123"
        assert d["command"] == "run-eval"
        assert d["args"]["dataset"] == "data/test"

    def test_from_dict_roundtrip(self, sample_entry):
        d = sample_entry.as_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.entry_id == sample_entry.entry_id
        assert restored.user == sample_entry.user
        assert restored.command == sample_entry.command

    def test_from_dict_defaults(self):
        e = AuditEntry.from_dict({
            "entry_id": "x",
            "user": "u",
            "timestamp": "t",
            "command": "c",
        })
        assert e.args == {}
        assert e.config_hash == ""
        assert e.evalyn_version == "unknown"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_as_dict(self, sample_log):
        d = sample_log.as_dict()
        assert len(d["entries"]) == 2
        assert d["log_path"] == "/tmp/test.jsonl"

    def test_from_dict_roundtrip(self, sample_log):
        d = sample_log.as_dict()
        restored = AuditLog.from_dict(d)
        assert len(restored.entries) == 2
        assert restored.log_path == sample_log.log_path

    def test_from_dict_defaults(self):
        log = AuditLog.from_dict({})
        assert log.entries == []
        assert log.log_path == ""

    def test_empty_log(self):
        log = AuditLog()
        assert log.entries == []


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_deterministic(self):
        config = {"model": "gpt-4", "temperature": 0.5}
        h1 = compute_config_hash(config)
        h2 = compute_config_hash(config)
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs_different_hash(self):
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2

    def test_empty_config(self):
        h = compute_config_hash({})
        assert len(h) == 64  # SHA256 hex

    def test_returns_hex_string(self):
        h = compute_config_hash({"x": "y"})
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# create_audit_entry
# ---------------------------------------------------------------------------


class TestCreateAuditEntry:
    def test_basic(self):
        entry = create_audit_entry(
            command="run-eval",
            args={"dataset": "test"},
            config={"model": "gpt-4"},
        )
        assert entry.command == "run-eval"
        assert entry.args["dataset"] == "test"
        assert len(entry.entry_id) > 0
        assert len(entry.timestamp) > 0
        assert len(entry.config_hash) == 64

    def test_custom_user(self):
        entry = create_audit_entry(
            command="analyze", args={}, config={}, user="alice"
        )
        assert entry.user == "alice"

    def test_auto_user(self):
        entry = create_audit_entry(command="test", args={}, config={})
        # Should pick up USER or USERNAME from env
        assert len(entry.user) > 0

    def test_result_summary(self):
        entry = create_audit_entry(
            command="eval", args={}, config={},
            result_summary="10 items evaluated",
        )
        assert entry.result_summary == "10 items evaluated"

    def test_unique_ids(self):
        e1 = create_audit_entry(command="a", args={}, config={})
        e2 = create_audit_entry(command="b", args={}, config={})
        assert e1.entry_id != e2.entry_id

    def test_version_populated(self):
        entry = create_audit_entry(command="x", args={}, config={})
        assert entry.evalyn_version != ""


# ---------------------------------------------------------------------------
# append_entry / read_audit_log
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_append_and_read(self, tmp_path, sample_entry):
        log_file = str(tmp_path / "audit.jsonl")
        append_entry(log_file, sample_entry)
        log = read_audit_log(log_file)
        assert len(log.entries) == 1
        assert log.entries[0].entry_id == "abc-123"

    def test_multiple_appends(self, tmp_path, sample_entry):
        log_file = str(tmp_path / "audit.jsonl")
        append_entry(log_file, sample_entry)
        append_entry(log_file, sample_entry)
        log = read_audit_log(log_file)
        assert len(log.entries) == 2

    def test_creates_parent_dirs(self, tmp_path, sample_entry):
        log_file = str(tmp_path / "deep" / "nested" / "audit.jsonl")
        append_entry(log_file, sample_entry)
        log = read_audit_log(log_file)
        assert len(log.entries) == 1

    def test_read_missing_file(self, tmp_path):
        log = read_audit_log(str(tmp_path / "nonexistent.jsonl"))
        assert log.entries == []

    def test_read_empty_file(self, tmp_path):
        log_file = str(tmp_path / "empty.jsonl")
        open(log_file, "w").close()
        log = read_audit_log(log_file)
        assert log.entries == []

    def test_log_path_preserved(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        log = read_audit_log(log_file)
        assert log.log_path == log_file

    def test_jsonl_format(self, tmp_path, sample_entry):
        log_file = str(tmp_path / "audit.jsonl")
        append_entry(log_file, sample_entry)
        with open(log_file) as f:
            content = f.read()
        # Each entry is one line of valid JSON
        lines = content.strip().split("\n")
        assert len(lines) == 1
        json.loads(lines[0])  # should not raise


# ---------------------------------------------------------------------------
# filter_entries
# ---------------------------------------------------------------------------


class TestFilterEntries:
    def test_filter_by_command(self, sample_log):
        result = filter_entries(sample_log, command="run-eval")
        assert len(result.entries) == 1
        assert result.entries[0].command == "run-eval"

    def test_filter_by_user(self, sample_log):
        result = filter_entries(sample_log, user="otheruser")
        assert len(result.entries) == 1
        assert result.entries[0].user == "otheruser"

    def test_filter_by_after(self, sample_log):
        result = filter_entries(sample_log, after="2026-01-16T00:00:00+00:00")
        assert len(result.entries) == 1
        assert result.entries[0].entry_id == "def-456"

    def test_filter_by_before(self, sample_log):
        result = filter_entries(sample_log, before="2026-01-16T00:00:00+00:00")
        assert len(result.entries) == 1
        assert result.entries[0].entry_id == "abc-123"

    def test_filter_no_match(self, sample_log):
        result = filter_entries(sample_log, command="nonexistent")
        assert len(result.entries) == 0

    def test_filter_multiple_criteria(self, sample_log):
        result = filter_entries(sample_log, command="run-eval", user="testuser")
        assert len(result.entries) == 1

    def test_filter_preserves_log_path(self, sample_log):
        result = filter_entries(sample_log, command="run-eval")
        assert result.log_path == sample_log.log_path

    def test_no_filters(self, sample_log):
        result = filter_entries(sample_log)
        assert len(result.entries) == 2


# ---------------------------------------------------------------------------
# format_audit_log
# ---------------------------------------------------------------------------


class TestFormatAuditLog:
    def test_basic(self, sample_log):
        text = format_audit_log(sample_log)
        assert "abc-123" in text
        assert "testuser" in text
        assert "run-eval" in text

    def test_empty_log(self):
        text = format_audit_log(AuditLog())
        assert "No audit entries" in text

    def test_limit(self, sample_log):
        text = format_audit_log(sample_log, limit=1)
        # Should show only the last entry
        lines = [l for l in text.strip().split("\n") if l.strip() and not l.startswith("-")]
        # header + 1 data line
        assert len(lines) == 2

    def test_contains_header(self, sample_log):
        text = format_audit_log(sample_log)
        assert "ID" in text
        assert "User" in text
        assert "Command" in text


# ---------------------------------------------------------------------------
# get_audit_stats
# ---------------------------------------------------------------------------


class TestGetAuditStats:
    def test_basic(self, sample_log):
        stats = get_audit_stats(sample_log)
        assert stats["total_entries"] == 2
        assert stats["unique_users"] == 2
        assert stats["unique_commands"] == 2
        assert stats["date_range"]["earliest"] is not None
        assert stats["most_used_command"] is not None

    def test_empty_log(self):
        stats = get_audit_stats(AuditLog())
        assert stats["total_entries"] == 0
        assert stats["unique_users"] == 0
        assert stats["most_used_command"] is None

    def test_most_used_command(self):
        entries = [
            AuditEntry(
                entry_id=str(i), user="u", timestamp="2026-01-01",
                command="run-eval" if i < 3 else "analyze",
                args={}, config_hash="", result_summary="", evalyn_version="0.1.0",
            )
            for i in range(5)
        ]
        log = AuditLog(entries=entries)
        stats = get_audit_stats(log)
        assert stats["most_used_command"] == "run-eval"

    def test_date_range(self, sample_log):
        stats = get_audit_stats(sample_log)
        assert stats["date_range"]["earliest"] <= stats["date_range"]["latest"]
