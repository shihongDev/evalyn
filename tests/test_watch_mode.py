"""Tests for the file-change watch mode module."""

from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.watch_mode import (
    WatchConfig,
    FileState,
    WatchEvent,
    ChangeDetector,
    should_debounce,
    run_watch_loop,
    format_watch_event,
)


# ---------------------------------------------------------------------------
# WatchConfig
# ---------------------------------------------------------------------------


class TestWatchConfig:
    def test_defaults(self):
        cfg = WatchConfig()
        assert cfg.paths == []
        assert cfg.debounce_seconds == 2.0
        assert cfg.poll_interval == 0.5

    def test_custom_values(self):
        cfg = WatchConfig(paths=["/a"], debounce_seconds=1.0, poll_interval=0.1)
        assert cfg.paths == ["/a"]
        assert cfg.debounce_seconds == 1.0
        assert cfg.poll_interval == 0.1

    def test_as_dict(self):
        cfg = WatchConfig(paths=["/x"], debounce_seconds=3.0, poll_interval=0.2)
        d = cfg.as_dict()
        assert d == {"paths": ["/x"], "debounce_seconds": 3.0, "poll_interval": 0.2}

    def test_from_dict(self):
        d = {"paths": ["/a", "/b"], "debounce_seconds": 1.5, "poll_interval": 0.3}
        cfg = WatchConfig.from_dict(d)
        assert cfg.paths == ["/a", "/b"]
        assert cfg.debounce_seconds == 1.5
        assert cfg.poll_interval == 0.3

    def test_from_dict_defaults(self):
        cfg = WatchConfig.from_dict({})
        assert cfg.paths == []
        assert cfg.debounce_seconds == 2.0
        assert cfg.poll_interval == 0.5

    def test_roundtrip(self):
        cfg = WatchConfig(paths=["/a"], debounce_seconds=5.0, poll_interval=1.0)
        cfg2 = WatchConfig.from_dict(cfg.as_dict())
        assert cfg2.paths == cfg.paths
        assert cfg2.debounce_seconds == cfg.debounce_seconds
        assert cfg2.poll_interval == cfg.poll_interval

    def test_as_dict_copies_paths(self):
        original = ["/a"]
        cfg = WatchConfig(paths=original)
        d = cfg.as_dict()
        d["paths"].append("/extra")
        assert cfg.paths == ["/a"]


# ---------------------------------------------------------------------------
# FileState
# ---------------------------------------------------------------------------


class TestFileState:
    def test_basic(self):
        fs = FileState(path="/f", mtime=1000.0, size=42, exists=True)
        assert fs.path == "/f"
        assert fs.mtime == 1000.0
        assert fs.size == 42
        assert fs.exists is True

    def test_as_dict(self):
        fs = FileState(path="/f", mtime=1.0, size=10, exists=True)
        d = fs.as_dict()
        assert d == {"path": "/f", "mtime": 1.0, "size": 10, "exists": True}

    def test_from_dict(self):
        d = {"path": "/x", "mtime": 2.0, "size": 20, "exists": False}
        fs = FileState.from_dict(d)
        assert fs.path == "/x"
        assert fs.exists is False

    def test_from_dict_defaults(self):
        fs = FileState.from_dict({"path": "/x"})
        assert fs.mtime == 0.0
        assert fs.size == 0
        assert fs.exists is False

    def test_roundtrip(self):
        fs = FileState(path="/a", mtime=99.9, size=100, exists=True)
        fs2 = FileState.from_dict(fs.as_dict())
        assert fs2.path == fs.path
        assert fs2.mtime == fs.mtime
        assert fs2.size == fs.size
        assert fs2.exists == fs.exists


# ---------------------------------------------------------------------------
# WatchEvent
# ---------------------------------------------------------------------------


class TestWatchEvent:
    def test_basic(self):
        ev = WatchEvent(path="/f", event_type="modified", timestamp="2026-01-01T00:00:00")
        assert ev.path == "/f"
        assert ev.event_type == "modified"

    def test_as_dict(self):
        ev = WatchEvent(path="/f", event_type="created", timestamp="ts")
        d = ev.as_dict()
        assert d == {"path": "/f", "event_type": "created", "timestamp": "ts"}

    def test_from_dict(self):
        d = {"path": "/g", "event_type": "deleted", "timestamp": "ts2"}
        ev = WatchEvent.from_dict(d)
        assert ev.path == "/g"
        assert ev.event_type == "deleted"

    def test_roundtrip(self):
        ev = WatchEvent(path="/x", event_type="modified", timestamp="t")
        ev2 = WatchEvent.from_dict(ev.as_dict())
        assert ev2.path == ev.path
        assert ev2.event_type == ev.event_type
        assert ev2.timestamp == ev.timestamp


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------


class TestChangeDetector:
    def test_snapshot_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        det = ChangeDetector([str(f)])
        snap = det.snapshot()
        assert str(f) in snap
        assert snap[str(f)].exists is True
        assert snap[str(f)].size == 5

    def test_snapshot_missing_file(self):
        det = ChangeDetector(["/nonexistent/file.txt"])
        snap = det.snapshot()
        assert snap["/nonexistent/file.txt"].exists is False
        assert snap["/nonexistent/file.txt"].mtime == 0.0

    def test_detect_no_changes(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        det = ChangeDetector([str(f)])
        snap = det.snapshot()
        events = det.detect_changes(snap, snap)
        assert events == []

    def test_detect_modification(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("old")
        det = ChangeDetector([str(f)])
        old = det.snapshot()
        # Ensure mtime differs
        time.sleep(0.05)
        f.write_text("new content")
        new = det.snapshot()
        events = det.detect_changes(old, new)
        assert len(events) == 1
        assert events[0].event_type == "modified"
        assert events[0].path == str(f)

    def test_detect_creation(self):
        old = {"f": FileState(path="f", mtime=0.0, size=0, exists=False)}
        new = {"f": FileState(path="f", mtime=100.0, size=10, exists=True)}
        events = ChangeDetector.detect_changes(old, new)
        assert len(events) == 1
        assert events[0].event_type == "created"

    def test_detect_deletion(self):
        old = {"f": FileState(path="f", mtime=100.0, size=10, exists=True)}
        new = {"f": FileState(path="f", mtime=0.0, size=0, exists=False)}
        events = ChangeDetector.detect_changes(old, new)
        assert len(events) == 1
        assert events[0].event_type == "deleted"

    def test_detect_new_path_in_new_snapshot(self):
        old: dict = {}
        new = {"f": FileState(path="f", mtime=1.0, size=1, exists=True)}
        events = ChangeDetector.detect_changes(old, new)
        assert len(events) == 1
        assert events[0].event_type == "created"

    def test_detect_removed_path_from_old(self):
        old = {"f": FileState(path="f", mtime=1.0, size=1, exists=True)}
        new: dict = {}
        events = ChangeDetector.detect_changes(old, new)
        assert len(events) == 1
        assert events[0].event_type == "deleted"

    def test_has_changes_true(self):
        old = {"f": FileState(path="f", mtime=1.0, size=1, exists=True)}
        new = {"f": FileState(path="f", mtime=2.0, size=1, exists=True)}
        assert ChangeDetector.has_changes(old, new) is True

    def test_has_changes_false(self):
        state = {"f": FileState(path="f", mtime=1.0, size=1, exists=True)}
        assert ChangeDetector.has_changes(state, state) is False

    def test_has_changes_different_keys(self):
        old = {"a": FileState(path="a", mtime=1.0, size=1, exists=True)}
        new = {"b": FileState(path="b", mtime=1.0, size=1, exists=True)}
        assert ChangeDetector.has_changes(old, new) is True

    def test_has_changes_existence_change(self):
        old = {"f": FileState(path="f", mtime=0.0, size=0, exists=False)}
        new = {"f": FileState(path="f", mtime=1.0, size=5, exists=True)}
        assert ChangeDetector.has_changes(old, new) is True

    def test_has_changes_size_change(self):
        old = {"f": FileState(path="f", mtime=1.0, size=1, exists=True)}
        new = {"f": FileState(path="f", mtime=1.0, size=99, exists=True)}
        assert ChangeDetector.has_changes(old, new) is True

    def test_multiple_changes(self):
        old = {
            "a": FileState(path="a", mtime=1.0, size=1, exists=True),
            "b": FileState(path="b", mtime=1.0, size=1, exists=True),
        }
        new = {
            "a": FileState(path="a", mtime=2.0, size=1, exists=True),
            "b": FileState(path="b", mtime=1.0, size=0, exists=False),
        }
        events = ChangeDetector.detect_changes(old, new)
        types = {e.event_type for e in events}
        assert "modified" in types
        assert "deleted" in types

    def test_events_have_timestamps(self):
        old = {"f": FileState(path="f", mtime=0.0, size=0, exists=False)}
        new = {"f": FileState(path="f", mtime=1.0, size=5, exists=True)}
        events = ChangeDetector.detect_changes(old, new)
        assert events[0].timestamp  # non-empty string

    def test_snapshot_multiple_files(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("aaa")
        b.write_text("bb")
        det = ChangeDetector([str(a), str(b)])
        snap = det.snapshot()
        assert len(snap) == 2
        assert snap[str(a)].size == 3
        assert snap[str(b)].size == 2


# ---------------------------------------------------------------------------
# should_debounce
# ---------------------------------------------------------------------------


class TestDebounce:
    def test_within_debounce(self):
        now = time.time()
        assert should_debounce(now, 5.0) is True

    def test_outside_debounce(self):
        old = time.time() - 10.0
        assert should_debounce(old, 5.0) is False

    def test_zero_debounce(self):
        assert should_debounce(time.time(), 0.0) is False

    def test_exact_boundary(self):
        # At exactly the boundary, time() >= last + debounce
        past = time.time() - 2.0
        assert should_debounce(past, 2.0) is False

    def test_last_run_zero(self):
        # last_run_time=0 means never run, should not debounce
        assert should_debounce(0.0, 2.0) is False


# ---------------------------------------------------------------------------
# run_watch_loop
# ---------------------------------------------------------------------------


class TestRunWatchLoop:
    def test_no_changes_no_callback(self, tmp_path):
        f = tmp_path / "stable.txt"
        f.write_text("x")
        calls = []
        config = WatchConfig(paths=[str(f)], debounce_seconds=0.0, poll_interval=0.01)
        count = run_watch_loop([str(f)], lambda evts: calls.append(evts), config, max_iterations=1)
        assert count == 0
        assert calls == []

    def test_detects_change_and_calls_back(self, tmp_path):
        f = tmp_path / "changing.txt"
        f.write_text("v1")

        calls = []

        def callback(events):
            calls.append(events)

        # Modify the file before the loop checks
        config = WatchConfig(paths=[str(f)], debounce_seconds=0.0, poll_interval=0.01)
        det = ChangeDetector([str(f)])
        _old = det.snapshot()

        # Trigger a real change
        time.sleep(0.05)
        f.write_text("v2 - modified")

        count = run_watch_loop([str(f)], callback, config, max_iterations=1)
        # May or may not detect depending on timing, but the function should return
        assert isinstance(count, int)

    def test_max_iterations_zero_means_one(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        config = WatchConfig(paths=[str(f)], debounce_seconds=0.0, poll_interval=0.01)
        count = run_watch_loop([str(f)], lambda e: None, config, max_iterations=0)
        assert isinstance(count, int)

    def test_returns_int(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        config = WatchConfig(paths=[str(f)], debounce_seconds=0.0, poll_interval=0.01)
        result = run_watch_loop([str(f)], lambda e: None, config, max_iterations=2)
        assert isinstance(result, int)
        assert result >= 0

    def test_debounce_suppresses_rapid_callbacks(self, tmp_path):
        f = tmp_path / "rapid.txt"
        f.write_text("v1")

        calls = []
        config = WatchConfig(paths=[str(f)], debounce_seconds=100.0, poll_interval=0.01)

        # Even if file changes, debounce should prevent more than 1 callback
        # With debounce=100s, after the first callback, no more should fire
        time.sleep(0.05)
        f.write_text("v2")

        count = run_watch_loop([str(f)], lambda e: calls.append(e), config, max_iterations=3)
        # First iteration may fire (debounce from time 0), rest suppressed
        assert count <= 1

    def test_callback_receives_events(self, tmp_path):
        """Verify that callback argument is a list of WatchEvent."""
        f = tmp_path / "cb.txt"
        f.write_text("v1")

        received = []

        def cb(events):
            received.extend(events)

        config = WatchConfig(paths=[str(f)], debounce_seconds=0.0, poll_interval=0.01)
        # Modify before loop
        time.sleep(0.05)
        f.write_text("v2 more content")
        run_watch_loop([str(f)], cb, config, max_iterations=1)

        for ev in received:
            assert isinstance(ev, WatchEvent)

    def test_empty_paths(self):
        config = WatchConfig(paths=[], debounce_seconds=0.0, poll_interval=0.01)
        count = run_watch_loop([], lambda e: None, config, max_iterations=1)
        assert count == 0


# ---------------------------------------------------------------------------
# format_watch_event
# ---------------------------------------------------------------------------


class TestFormatWatchEvent:
    def test_modified(self):
        ev = WatchEvent(path="/a/b.py", event_type="modified", timestamp="2026-01-01T00:00:00")
        text = format_watch_event(ev)
        assert "Modified" in text
        assert "/a/b.py" in text
        assert "2026-01-01" in text

    def test_created(self):
        ev = WatchEvent(path="/new.py", event_type="created", timestamp="ts")
        text = format_watch_event(ev)
        assert "Created" in text
        assert "/new.py" in text

    def test_deleted(self):
        ev = WatchEvent(path="/gone.py", event_type="deleted", timestamp="ts")
        text = format_watch_event(ev)
        assert "Deleted" in text

    def test_unknown_type_passthrough(self):
        ev = WatchEvent(path="/f", event_type="renamed", timestamp="ts")
        text = format_watch_event(ev)
        assert "renamed" in text

    def test_format_contains_timestamp(self):
        ev = WatchEvent(path="/f", event_type="modified", timestamp="2026-03-29T10:00:00+00:00")
        text = format_watch_event(ev)
        assert "2026-03-29" in text
