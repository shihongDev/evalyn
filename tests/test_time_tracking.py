"""Tests for CLI time tracking module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.time_tracking import (
    TimingEntry,
    TimingReport,
    TimingStats,
    build_timing_report,
    compute_stats,
    create_timing_entry,
    format_timing_report,
    identify_slowest,
    log_timing,
    read_timing_log,
)


# ---------------------------------------------------------------------------
# TimingEntry dataclass
# ---------------------------------------------------------------------------


class TestTimingEntry:
    def test_basic_creation(self):
        e = TimingEntry(command="run", duration_seconds=1.5, timestamp="2026-01-01T00:00:00+00:00")
        assert e.command == "run"
        assert e.duration_seconds == 1.5
        assert e.timestamp == "2026-01-01T00:00:00+00:00"

    def test_as_dict(self):
        e = TimingEntry(command="run", duration_seconds=2.0, timestamp="t1")
        d = e.as_dict()
        assert d == {"command": "run", "duration_seconds": 2.0, "timestamp": "t1"}

    def test_from_dict(self):
        d = {"command": "analyze", "duration_seconds": 0.5, "timestamp": "t2"}
        e = TimingEntry.from_dict(d)
        assert e.command == "analyze"
        assert e.duration_seconds == 0.5
        assert e.timestamp == "t2"

    def test_roundtrip(self):
        e = TimingEntry(command="compare", duration_seconds=3.3, timestamp="t3")
        e2 = TimingEntry.from_dict(e.as_dict())
        assert e == e2

    def test_zero_duration(self):
        e = TimingEntry(command="noop", duration_seconds=0.0, timestamp="t")
        assert e.duration_seconds == 0.0

    def test_large_duration(self):
        e = TimingEntry(command="slow", duration_seconds=9999.99, timestamp="t")
        assert e.duration_seconds == 9999.99


# ---------------------------------------------------------------------------
# TimingStats dataclass
# ---------------------------------------------------------------------------


class TestTimingStats:
    def test_basic_creation(self):
        s = TimingStats(command="run", total_seconds=10.0, count=5, avg_seconds=2.0, min_seconds=1.0, max_seconds=3.0)
        assert s.command == "run"
        assert s.count == 5

    def test_as_dict(self):
        s = TimingStats(command="x", total_seconds=6.0, count=3, avg_seconds=2.0, min_seconds=1.0, max_seconds=3.0)
        d = s.as_dict()
        assert d["command"] == "x"
        assert d["total_seconds"] == 6.0
        assert d["count"] == 3

    def test_from_dict(self):
        d = {"command": "y", "total_seconds": 4.0, "count": 2, "avg_seconds": 2.0, "min_seconds": 1.0, "max_seconds": 3.0}
        s = TimingStats.from_dict(d)
        assert s.command == "y"
        assert s.avg_seconds == 2.0

    def test_roundtrip(self):
        s = TimingStats(command="z", total_seconds=9.0, count=3, avg_seconds=3.0, min_seconds=1.0, max_seconds=5.0)
        s2 = TimingStats.from_dict(s.as_dict())
        assert s == s2

    def test_single_count(self):
        s = TimingStats(command="once", total_seconds=1.0, count=1, avg_seconds=1.0, min_seconds=1.0, max_seconds=1.0)
        assert s.min_seconds == s.max_seconds == s.avg_seconds


# ---------------------------------------------------------------------------
# TimingReport dataclass
# ---------------------------------------------------------------------------


class TestTimingReport:
    def test_basic_creation(self):
        r = TimingReport(stats=[], total_commands=0, total_time_seconds=0.0)
        assert r.total_commands == 0
        assert r.stats == []

    def test_as_dict_empty(self):
        r = TimingReport(stats=[], total_commands=0, total_time_seconds=0.0)
        d = r.as_dict()
        assert d["stats"] == []
        assert d["total_commands"] == 0

    def test_from_dict_empty(self):
        d = {"stats": [], "total_commands": 0, "total_time_seconds": 0.0}
        r = TimingReport.from_dict(d)
        assert r.stats == []

    def test_roundtrip_with_stats(self):
        s = TimingStats(command="a", total_seconds=2.0, count=1, avg_seconds=2.0, min_seconds=2.0, max_seconds=2.0)
        r = TimingReport(stats=[s], total_commands=1, total_time_seconds=2.0)
        r2 = TimingReport.from_dict(r.as_dict())
        assert r2.total_commands == 1
        assert r2.stats[0].command == "a"

    def test_from_dict_defaults(self):
        r = TimingReport.from_dict({})
        assert r.total_commands == 0
        assert r.total_time_seconds == 0.0
        assert r.stats == []


# ---------------------------------------------------------------------------
# create_timing_entry
# ---------------------------------------------------------------------------


class TestCreateTimingEntry:
    def test_auto_timestamp(self):
        e = create_timing_entry("run", 1.0)
        assert e.command == "run"
        assert e.duration_seconds == 1.0
        # timestamp should be ISO format with timezone
        assert "T" in e.timestamp
        assert "+" in e.timestamp or "Z" in e.timestamp

    def test_different_commands(self):
        e1 = create_timing_entry("analyze", 2.0)
        e2 = create_timing_entry("compare", 3.0)
        assert e1.command != e2.command

    def test_zero_duration(self):
        e = create_timing_entry("fast", 0.0)
        assert e.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# log_timing / read_timing_log
# ---------------------------------------------------------------------------


class TestLogAndRead:
    def test_log_and_read_single(self, tmp_path):
        log = str(tmp_path / "timing.jsonl")
        e = TimingEntry(command="run", duration_seconds=1.0, timestamp="t1")
        log_timing(log, e)
        entries = read_timing_log(log)
        assert len(entries) == 1
        assert entries[0].command == "run"

    def test_log_multiple(self, tmp_path):
        log = str(tmp_path / "timing.jsonl")
        for i in range(5):
            log_timing(log, TimingEntry(command=f"cmd{i}", duration_seconds=float(i), timestamp=f"t{i}"))
        entries = read_timing_log(log)
        assert len(entries) == 5

    def test_read_nonexistent(self, tmp_path):
        log = str(tmp_path / "missing.jsonl")
        entries = read_timing_log(log)
        assert entries == []

    def test_log_creates_directories(self, tmp_path):
        log = str(tmp_path / "sub" / "dir" / "timing.jsonl")
        e = TimingEntry(command="run", duration_seconds=1.0, timestamp="t1")
        log_timing(log, e)
        assert os.path.exists(log)

    def test_append_preserves_existing(self, tmp_path):
        log = str(tmp_path / "timing.jsonl")
        log_timing(log, TimingEntry(command="first", duration_seconds=1.0, timestamp="t1"))
        log_timing(log, TimingEntry(command="second", duration_seconds=2.0, timestamp="t2"))
        entries = read_timing_log(log)
        assert entries[0].command == "first"
        assert entries[1].command == "second"

    def test_jsonl_format(self, tmp_path):
        log = str(tmp_path / "timing.jsonl")
        log_timing(log, TimingEntry(command="x", duration_seconds=1.0, timestamp="t"))
        with open(log) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["command"] == "x"

    def test_read_skips_blank_lines(self, tmp_path):
        log = str(tmp_path / "timing.jsonl")
        with open(log, "w") as f:
            f.write('{"command":"a","duration_seconds":1.0,"timestamp":"t"}\n')
            f.write("\n")
            f.write('{"command":"b","duration_seconds":2.0,"timestamp":"t"}\n')
        entries = read_timing_log(log)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def _make_entries(self, data):
        """Helper: data is list of (command, duration) tuples."""
        return [TimingEntry(command=c, duration_seconds=d, timestamp="t") for c, d in data]

    def test_empty(self):
        assert compute_stats([]) == []

    def test_single_command(self):
        entries = self._make_entries([("run", 2.0)])
        stats = compute_stats(entries)
        assert len(stats) == 1
        assert stats[0].command == "run"
        assert stats[0].total_seconds == 2.0
        assert stats[0].count == 1
        assert stats[0].avg_seconds == 2.0
        assert stats[0].min_seconds == 2.0
        assert stats[0].max_seconds == 2.0

    def test_multiple_same_command(self):
        entries = self._make_entries([("run", 1.0), ("run", 3.0), ("run", 2.0)])
        stats = compute_stats(entries)
        assert len(stats) == 1
        s = stats[0]
        assert s.count == 3
        assert s.total_seconds == 6.0
        assert s.avg_seconds == 2.0
        assert s.min_seconds == 1.0
        assert s.max_seconds == 3.0

    def test_sorted_by_total_descending(self):
        entries = self._make_entries([("fast", 1.0), ("slow", 10.0), ("mid", 5.0)])
        stats = compute_stats(entries)
        commands = [s.command for s in stats]
        assert commands == ["slow", "mid", "fast"]

    def test_two_commands(self):
        entries = self._make_entries([("a", 1.0), ("b", 2.0), ("a", 3.0)])
        stats = compute_stats(entries)
        assert len(stats) == 2
        by_cmd = {s.command: s for s in stats}
        assert by_cmd["a"].total_seconds == 4.0
        assert by_cmd["b"].total_seconds == 2.0

    def test_min_max_distinct(self):
        entries = self._make_entries([("x", 1.0), ("x", 5.0), ("x", 3.0)])
        stats = compute_stats(entries)
        assert stats[0].min_seconds == 1.0
        assert stats[0].max_seconds == 5.0


# ---------------------------------------------------------------------------
# build_timing_report
# ---------------------------------------------------------------------------


class TestBuildTimingReport:
    def _make_entries(self, data):
        return [TimingEntry(command=c, duration_seconds=d, timestamp="t") for c, d in data]

    def test_empty(self):
        r = build_timing_report([])
        assert r.total_commands == 0
        assert r.total_time_seconds == 0.0
        assert r.stats == []

    def test_basic(self):
        entries = self._make_entries([("run", 1.0), ("run", 2.0), ("analyze", 3.0)])
        r = build_timing_report(entries)
        assert r.total_commands == 3
        assert r.total_time_seconds == 6.0
        assert len(r.stats) == 2

    def test_total_time_matches_sum(self):
        entries = self._make_entries([("a", 1.5), ("b", 2.5), ("c", 3.0)])
        r = build_timing_report(entries)
        assert r.total_time_seconds == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# identify_slowest
# ---------------------------------------------------------------------------


class TestIdentifySlowest:
    def _make_stats(self, data):
        """data: list of (command, avg_seconds)"""
        return [
            TimingStats(command=c, total_seconds=a, count=1, avg_seconds=a, min_seconds=a, max_seconds=a)
            for c, a in data
        ]

    def test_empty(self):
        assert identify_slowest([]) == []

    def test_fewer_than_n(self):
        stats = self._make_stats([("a", 1.0), ("b", 2.0)])
        result = identify_slowest(stats, n=5)
        assert len(result) == 2
        assert result[0].command == "b"

    def test_top_n(self):
        stats = self._make_stats([("a", 1.0), ("b", 5.0), ("c", 3.0), ("d", 4.0), ("e", 2.0)])
        result = identify_slowest(stats, n=3)
        assert len(result) == 3
        assert [r.command for r in result] == ["b", "d", "c"]

    def test_n_equals_one(self):
        stats = self._make_stats([("a", 10.0), ("b", 20.0)])
        result = identify_slowest(stats, n=1)
        assert len(result) == 1
        assert result[0].command == "b"

    def test_default_n(self):
        stats = self._make_stats([(f"cmd{i}", float(i)) for i in range(10)])
        result = identify_slowest(stats)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# format_timing_report
# ---------------------------------------------------------------------------


class TestFormatTimingReport:
    def test_empty_report(self):
        r = TimingReport(stats=[], total_commands=0, total_time_seconds=0.0)
        text = format_timing_report(r)
        assert "Timing Report" in text
        assert "Total: 0 commands" in text

    def test_with_stats(self):
        s = TimingStats(command="run", total_seconds=6.0, count=3, avg_seconds=2.0, min_seconds=1.0, max_seconds=3.0)
        r = TimingReport(stats=[s], total_commands=3, total_time_seconds=6.0)
        text = format_timing_report(r)
        assert "run" in text
        assert "6.000" in text
        assert "2.000" in text

    def test_multiple_stats(self):
        s1 = TimingStats(command="run", total_seconds=6.0, count=3, avg_seconds=2.0, min_seconds=1.0, max_seconds=3.0)
        s2 = TimingStats(command="analyze", total_seconds=4.0, count=2, avg_seconds=2.0, min_seconds=1.0, max_seconds=3.0)
        r = TimingReport(stats=[s1, s2], total_commands=5, total_time_seconds=10.0)
        text = format_timing_report(r)
        assert "run" in text
        assert "analyze" in text
        assert "Total: 5 commands" in text

    def test_header_present(self):
        r = TimingReport(stats=[], total_commands=0, total_time_seconds=0.0)
        text = format_timing_report(r)
        assert "Command" in text
        assert "Count" in text
        assert "Total(s)" in text
        assert "Avg(s)" in text
        assert "Min(s)" in text
        assert "Max(s)" in text
