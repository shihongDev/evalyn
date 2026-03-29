"""Tests for the CLI execution audit log module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.execution_audit import (
    AuditConfig,
    AuditRecord,
    append_audit,
    compute_audit_stats,
    create_audit_record,
    filter_audit,
    format_audit_log,
    read_audit_log,
    verify_audit_integrity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    command: str = "analyze",
    args: list | None = None,
    timestamp: str = "2026-03-29T10:00:00+00:00",
    exit_code: int = 0,
    duration: float = 1.5,
    user: str = "tester",
    cwd: str = "/tmp",
) -> AuditRecord:
    return AuditRecord(
        command=command,
        args=args or [],
        timestamp=timestamp,
        exit_code=exit_code,
        duration_seconds=duration,
        user=user,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# AuditConfig tests
# ---------------------------------------------------------------------------


class TestAuditConfig:
    def test_defaults(self):
        cfg = AuditConfig()
        assert cfg.log_path == ".evalyn/audit.jsonl"
        assert cfg.enabled is True
        assert cfg.include_output is False

    def test_custom_values(self):
        cfg = AuditConfig(log_path="/tmp/a.jsonl", enabled=False, include_output=True)
        assert cfg.log_path == "/tmp/a.jsonl"
        assert cfg.enabled is False
        assert cfg.include_output is True

    def test_as_dict(self):
        cfg = AuditConfig(log_path="x.jsonl", enabled=False, include_output=True)
        d = cfg.as_dict()
        assert d == {"log_path": "x.jsonl", "enabled": False, "include_output": True}

    def test_from_dict(self):
        d = {"log_path": "p.jsonl", "enabled": False, "include_output": True}
        cfg = AuditConfig.from_dict(d)
        assert cfg.log_path == "p.jsonl"
        assert cfg.enabled is False
        assert cfg.include_output is True

    def test_from_dict_defaults(self):
        cfg = AuditConfig.from_dict({})
        assert cfg.log_path == ".evalyn/audit.jsonl"
        assert cfg.enabled is True
        assert cfg.include_output is False

    def test_roundtrip(self):
        cfg = AuditConfig(log_path="rt.jsonl", enabled=True, include_output=True)
        assert AuditConfig.from_dict(cfg.as_dict()) == cfg


# ---------------------------------------------------------------------------
# AuditRecord tests
# ---------------------------------------------------------------------------


class TestAuditRecord:
    def test_creation(self):
        r = _make_record()
        assert r.command == "analyze"
        assert r.exit_code == 0
        assert r.duration_seconds == 1.5

    def test_as_dict(self):
        r = _make_record(command="run", args=["--verbose"])
        d = r.as_dict()
        assert d["command"] == "run"
        assert d["args"] == ["--verbose"]
        assert d["exit_code"] == 0

    def test_from_dict(self):
        d = {
            "command": "eval",
            "args": ["-n", "5"],
            "timestamp": "2026-01-01T00:00:00+00:00",
            "exit_code": 1,
            "duration_seconds": 2.0,
            "user": "bob",
            "cwd": "/home",
        }
        r = AuditRecord.from_dict(d)
        assert r.command == "eval"
        assert r.args == ["-n", "5"]
        assert r.exit_code == 1

    def test_from_dict_defaults(self):
        d = {"command": "x", "timestamp": "2026-01-01T00:00:00+00:00"}
        r = AuditRecord.from_dict(d)
        assert r.args == []
        assert r.exit_code == 0
        assert r.duration_seconds == 0.0
        assert r.user == ""
        assert r.cwd == ""

    def test_roundtrip(self):
        r = _make_record(command="test", args=["a", "b"], exit_code=2, duration=3.14)
        assert AuditRecord.from_dict(r.as_dict()) == r

    def test_args_are_copied(self):
        original_args = ["a", "b"]
        d = {
            "command": "x",
            "args": original_args,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        r = AuditRecord.from_dict(d)
        original_args.append("c")
        assert r.args == ["a", "b"]


# ---------------------------------------------------------------------------
# create_audit_record tests
# ---------------------------------------------------------------------------


class TestCreateAuditRecord:
    def test_auto_timestamp(self):
        r = create_audit_record("analyze")
        assert r.timestamp  # not empty
        assert "T" in r.timestamp  # ISO format

    def test_auto_user(self):
        r = create_audit_record("run")
        assert r.user  # not empty

    def test_auto_cwd(self):
        r = create_audit_record("run")
        assert r.cwd == os.getcwd()

    def test_command_and_args(self):
        r = create_audit_record("eval", args=["--dataset", "test"])
        assert r.command == "eval"
        assert r.args == ["--dataset", "test"]

    def test_default_exit_code(self):
        r = create_audit_record("x")
        assert r.exit_code == 0

    def test_custom_exit_code(self):
        r = create_audit_record("x", exit_code=42)
        assert r.exit_code == 42

    def test_default_duration(self):
        r = create_audit_record("x")
        assert r.duration_seconds == 0.0

    def test_none_args_becomes_empty_list(self):
        r = create_audit_record("x", args=None)
        assert r.args == []


# ---------------------------------------------------------------------------
# append_audit / read_audit_log tests
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_append_creates_file(self, tmp_path):
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        r = _make_record()
        append_audit(cfg, r)
        assert Path(log).exists()

    def test_append_creates_parent_dirs(self, tmp_path):
        log = str(tmp_path / "deep" / "nested" / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        append_audit(cfg, _make_record())
        assert Path(log).exists()

    def test_disabled_does_not_write(self, tmp_path):
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log, enabled=False)
        append_audit(cfg, _make_record())
        assert not Path(log).exists()

    def test_read_after_append(self, tmp_path):
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        for i in range(3):
            append_audit(cfg, _make_record(command=f"cmd_{i}"))
        records = read_audit_log(log)
        assert len(records) == 3
        assert records[0].command == "cmd_0"
        assert records[2].command == "cmd_2"

    def test_read_missing_file(self, tmp_path):
        records = read_audit_log(str(tmp_path / "nonexistent.jsonl"))
        assert records == []

    def test_read_with_limit(self, tmp_path):
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        for i in range(10):
            append_audit(cfg, _make_record(command=f"cmd_{i}"))
        records = read_audit_log(log, limit=3)
        assert len(records) == 3
        # Last 3 records
        assert records[0].command == "cmd_7"

    def test_read_skips_malformed_lines(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log.write_text(
            '{"command":"ok","args":[],"timestamp":"T","exit_code":0,'
            '"duration_seconds":0,"user":"u","cwd":"/"}\n'
            "NOT VALID JSON\n"
            '{"command":"ok2","args":[],"timestamp":"T2","exit_code":0,'
            '"duration_seconds":0,"user":"u","cwd":"/"}\n'
        )
        records = read_audit_log(str(log))
        assert len(records) == 2

    def test_read_skips_empty_lines(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log.write_text(
            '{"command":"a","args":[],"timestamp":"T","exit_code":0,'
            '"duration_seconds":0,"user":"u","cwd":"/"}\n'
            "\n"
            '{"command":"b","args":[],"timestamp":"T","exit_code":0,'
            '"duration_seconds":0,"user":"u","cwd":"/"}\n'
        )
        records = read_audit_log(str(log))
        assert len(records) == 2

    def test_jsonl_format(self, tmp_path):
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        append_audit(cfg, _make_record())
        with open(log) as f:
            line = f.readline().strip()
        data = json.loads(line)
        assert "command" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# filter_audit tests
# ---------------------------------------------------------------------------


class TestFilterAudit:
    def _records(self):
        return [
            _make_record(command="analyze", timestamp="2026-03-29T10:00:00+00:00", exit_code=0),
            _make_record(command="run", timestamp="2026-03-29T11:00:00+00:00", exit_code=1),
            _make_record(command="analyze", timestamp="2026-03-29T12:00:00+00:00", exit_code=0),
            _make_record(command="export", timestamp="2026-03-29T13:00:00+00:00", exit_code=0),
        ]

    def test_filter_by_command(self):
        result = filter_audit(self._records(), command="analyze")
        assert len(result) == 2
        assert all(r.command == "analyze" for r in result)

    def test_filter_by_exit_code(self):
        result = filter_audit(self._records(), exit_code=1)
        assert len(result) == 1
        assert result[0].command == "run"

    def test_filter_by_after(self):
        result = filter_audit(self._records(), after="2026-03-29T11:30:00+00:00")
        assert len(result) == 2

    def test_filter_combined(self):
        result = filter_audit(
            self._records(),
            command="analyze",
            after="2026-03-29T10:30:00+00:00",
        )
        assert len(result) == 1
        assert result[0].timestamp == "2026-03-29T12:00:00+00:00"

    def test_filter_no_match(self):
        result = filter_audit(self._records(), command="nonexistent")
        assert result == []

    def test_filter_no_criteria(self):
        records = self._records()
        result = filter_audit(records)
        assert len(result) == len(records)


# ---------------------------------------------------------------------------
# compute_audit_stats tests
# ---------------------------------------------------------------------------


class TestComputeAuditStats:
    def test_empty_records(self):
        stats = compute_audit_stats([])
        assert stats["total_commands"] == 0
        assert stats["unique_commands"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["most_used"] == ""
        assert stats["avg_duration"] == 0.0

    def test_basic_stats(self):
        records = [
            _make_record(command="analyze", duration=2.0, exit_code=0),
            _make_record(command="analyze", duration=3.0, exit_code=0),
            _make_record(command="run", duration=1.0, exit_code=1),
        ]
        stats = compute_audit_stats(records)
        assert stats["total_commands"] == 3
        assert stats["unique_commands"] == 2
        assert stats["most_used"] == "analyze"
        assert abs(stats["avg_duration"] - 2.0) < 0.01
        assert abs(stats["error_rate"] - 1 / 3) < 0.01

    def test_single_record(self):
        stats = compute_audit_stats([_make_record(command="x", duration=5.0)])
        assert stats["total_commands"] == 1
        assert stats["unique_commands"] == 1
        assert stats["most_used"] == "x"
        assert stats["avg_duration"] == 5.0
        assert stats["error_rate"] == 0.0

    def test_all_errors(self):
        records = [
            _make_record(exit_code=1),
            _make_record(exit_code=2),
        ]
        stats = compute_audit_stats(records)
        assert stats["error_rate"] == 1.0


# ---------------------------------------------------------------------------
# format_audit_log tests
# ---------------------------------------------------------------------------


class TestFormatAuditLog:
    def test_empty_records(self):
        result = format_audit_log([])
        assert result == "No audit records."

    def test_header_present(self):
        result = format_audit_log([_make_record()])
        assert "Timestamp" in result
        assert "Command" in result
        assert "Exit" in result
        assert "Duration" in result

    def test_record_values_present(self):
        result = format_audit_log([_make_record(command="analyze", exit_code=0)])
        assert "analyze" in result

    def test_multiple_records(self):
        records = [_make_record(command=f"cmd_{i}") for i in range(3)]
        result = format_audit_log(records)
        lines = result.strip().split("\n")
        # header + separator + 3 records
        assert len(lines) == 5

    def test_args_in_output(self):
        r = _make_record(args=["--verbose", "--dry-run"])
        result = format_audit_log([r])
        assert "--verbose" in result
        assert "--dry-run" in result


# ---------------------------------------------------------------------------
# verify_audit_integrity tests
# ---------------------------------------------------------------------------


class TestVerifyAuditIntegrity:
    def test_empty_list(self):
        assert verify_audit_integrity([]) is True

    def test_single_record(self):
        assert verify_audit_integrity([_make_record()]) is True

    def test_increasing_timestamps(self):
        records = [
            _make_record(timestamp="2026-03-29T10:00:00+00:00"),
            _make_record(timestamp="2026-03-29T11:00:00+00:00"),
            _make_record(timestamp="2026-03-29T12:00:00+00:00"),
        ]
        assert verify_audit_integrity(records) is True

    def test_equal_timestamps(self):
        records = [
            _make_record(timestamp="2026-03-29T10:00:00+00:00"),
            _make_record(timestamp="2026-03-29T10:00:00+00:00"),
        ]
        assert verify_audit_integrity(records) is True

    def test_decreasing_timestamp_fails(self):
        records = [
            _make_record(timestamp="2026-03-29T12:00:00+00:00"),
            _make_record(timestamp="2026-03-29T10:00:00+00:00"),
        ]
        assert verify_audit_integrity(records) is False

    def test_out_of_order_in_middle(self):
        records = [
            _make_record(timestamp="2026-03-29T10:00:00+00:00"),
            _make_record(timestamp="2026-03-29T12:00:00+00:00"),
            _make_record(timestamp="2026-03-29T11:00:00+00:00"),
        ]
        assert verify_audit_integrity(records) is False

    def test_integrity_after_append(self, tmp_path):
        """Records written via append_audit should have monotonic timestamps."""
        log = str(tmp_path / "audit.jsonl")
        cfg = AuditConfig(log_path=log)
        for _ in range(5):
            r = create_audit_record("test")
            append_audit(cfg, r)
        records = read_audit_log(log)
        assert verify_audit_integrity(records) is True
