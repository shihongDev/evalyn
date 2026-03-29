"""Tests for checkpoint_inspection module."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.checkpoint_inspection import (
    CheckpointInfo,
    CheckpointSummary,
    _format_age,
    _format_size,
    delete_checkpoint,
    delete_stale_checkpoints,
    format_checkpoint_detail,
    format_checkpoint_list,
    get_checkpoint_progress,
    read_checkpoint,
    scan_checkpoints,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ago_iso(days: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()


def _make_checkpoint(
    dirpath: str,
    name: str = "checkpoint.json",
    run_id: str = "run-1",
    processed: int = 5,
    total: int = 10,
    created_at: str = "",
) -> str:
    """Write a checkpoint file and return its path."""
    if not created_at:
        created_at = _now_iso()
    path = os.path.join(dirpath, name)
    content = {
        "run_id": run_id,
        "processed_count": processed,
        "total_count": total,
        "created_at": created_at,
    }
    with open(path, "w") as f:
        json.dump(content, f)
    return path


# ---------------------------------------------------------------------------
# CheckpointInfo dataclass
# ---------------------------------------------------------------------------


class TestCheckpointInfo:
    def test_create(self):
        info = CheckpointInfo(
            path="/tmp/cp.json",
            run_id="r1",
            processed_count=3,
            total_count=10,
            created_at="2025-01-01T00:00:00+00:00",
            size_bytes=256,
        )
        assert info.run_id == "r1"
        assert info.processed_count == 3

    def test_as_dict(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=1,
            total_count=2, created_at="t", size_bytes=100,
        )
        d = info.as_dict()
        assert d["path"] == "p"
        assert d["size_bytes"] == 100

    def test_from_dict(self):
        d = {
            "path": "p",
            "run_id": "r",
            "processed_count": 5,
            "total_count": 10,
            "created_at": "t",
            "size_bytes": 200,
        }
        info = CheckpointInfo.from_dict(d)
        assert info.total_count == 10

    def test_from_dict_defaults(self):
        d = {"path": "p"}
        info = CheckpointInfo.from_dict(d)
        assert info.run_id == ""
        assert info.processed_count == 0
        assert info.total_count == 0
        assert info.size_bytes == 0

    def test_roundtrip(self):
        info = CheckpointInfo(
            path="a", run_id="b", processed_count=3,
            total_count=5, created_at="c", size_bytes=99,
        )
        assert CheckpointInfo.from_dict(info.as_dict()) == info


# ---------------------------------------------------------------------------
# CheckpointSummary dataclass
# ---------------------------------------------------------------------------


class TestCheckpointSummary:
    def test_empty(self):
        s = CheckpointSummary()
        assert s.checkpoints == []
        assert s.total_count == 0
        assert s.total_size_bytes == 0

    def test_as_dict(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=1,
            total_count=2, created_at="t", size_bytes=50,
        )
        s = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=50)
        d = s.as_dict()
        assert d["total_count"] == 1
        assert len(d["checkpoints"]) == 1

    def test_from_dict(self):
        d = {
            "checkpoints": [
                {
                    "path": "p",
                    "run_id": "r",
                    "processed_count": 1,
                    "total_count": 2,
                    "created_at": "t",
                    "size_bytes": 50,
                }
            ],
            "total_count": 1,
            "total_size_bytes": 50,
        }
        s = CheckpointSummary.from_dict(d)
        assert len(s.checkpoints) == 1
        assert s.total_size_bytes == 50

    def test_from_dict_empty(self):
        s = CheckpointSummary.from_dict({})
        assert s.checkpoints == []
        assert s.total_count == 0

    def test_roundtrip(self):
        info = CheckpointInfo(
            path="a", run_id="b", processed_count=3,
            total_count=5, created_at="c", size_bytes=99,
        )
        s = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=99)
        s2 = CheckpointSummary.from_dict(s.as_dict())
        assert s2.total_count == s.total_count
        assert s2.checkpoints[0] == info


# ---------------------------------------------------------------------------
# scan_checkpoints
# ---------------------------------------------------------------------------


class TestScanCheckpoints:
    def test_empty_dir(self, tmp_path):
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 0
        assert s.checkpoints == []

    def test_nonexistent_dir(self):
        s = scan_checkpoints("/nonexistent/path/xyz")
        assert s.total_count == 0

    def test_finds_checkpoint(self, tmp_path):
        _make_checkpoint(str(tmp_path))
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 1
        assert s.checkpoints[0].run_id == "run-1"

    def test_ignores_non_checkpoint(self, tmp_path):
        # Write a file that does not start with "checkpoint"
        other = tmp_path / "results.json"
        other.write_text("{}")
        _make_checkpoint(str(tmp_path))
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 1

    def test_multiple_checkpoints(self, tmp_path):
        _make_checkpoint(str(tmp_path), name="checkpoint_1.json", run_id="r1")
        _make_checkpoint(str(tmp_path), name="checkpoint_2.json", run_id="r2")
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 2

    def test_nested_directories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_checkpoint(str(sub), name="checkpoint.json")
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 1

    def test_total_size(self, tmp_path):
        _make_checkpoint(str(tmp_path), name="checkpoint_a.json")
        _make_checkpoint(str(tmp_path), name="checkpoint_b.json")
        s = scan_checkpoints(str(tmp_path))
        assert s.total_size_bytes > 0
        total = sum(c.size_bytes for c in s.checkpoints)
        assert s.total_size_bytes == total

    def test_invalid_json_skipped(self, tmp_path):
        bad = tmp_path / "checkpoint_bad.json"
        bad.write_text("not json {{{")
        _make_checkpoint(str(tmp_path), name="checkpoint_good.json")
        s = scan_checkpoints(str(tmp_path))
        assert s.total_count == 1

    def test_parses_fields(self, tmp_path):
        _make_checkpoint(
            str(tmp_path),
            name="checkpoint.json",
            run_id="abc",
            processed=7,
            total=20,
        )
        s = scan_checkpoints(str(tmp_path))
        info = s.checkpoints[0]
        assert info.run_id == "abc"
        assert info.processed_count == 7
        assert info.total_count == 20


# ---------------------------------------------------------------------------
# read_checkpoint
# ---------------------------------------------------------------------------


class TestReadCheckpoint:
    def test_reads_json(self, tmp_path):
        path = _make_checkpoint(str(tmp_path))
        data = read_checkpoint(path)
        assert data["run_id"] == "run-1"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_checkpoint("/nonexistent/checkpoint.json")


# ---------------------------------------------------------------------------
# get_checkpoint_progress
# ---------------------------------------------------------------------------


class TestGetCheckpointProgress:
    def test_half(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=5,
            total_count=10, created_at="", size_bytes=0,
        )
        assert get_checkpoint_progress(info) == pytest.approx(0.5)

    def test_complete(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=10,
            total_count=10, created_at="", size_bytes=0,
        )
        assert get_checkpoint_progress(info) == pytest.approx(1.0)

    def test_zero_total(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=0,
            total_count=0, created_at="", size_bytes=0,
        )
        assert get_checkpoint_progress(info) == 0.0

    def test_negative_total(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=0,
            total_count=-1, created_at="", size_bytes=0,
        )
        assert get_checkpoint_progress(info) == 0.0

    def test_partial(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=3,
            total_count=7, created_at="", size_bytes=0,
        )
        assert get_checkpoint_progress(info) == pytest.approx(3 / 7)


# ---------------------------------------------------------------------------
# delete_checkpoint
# ---------------------------------------------------------------------------


class TestDeleteCheckpoint:
    def test_deletes_existing(self, tmp_path):
        path = _make_checkpoint(str(tmp_path))
        assert os.path.exists(path)
        assert delete_checkpoint(path) is True
        assert not os.path.exists(path)

    def test_nonexistent_returns_false(self):
        assert delete_checkpoint("/nonexistent/file.json") is False


# ---------------------------------------------------------------------------
# delete_stale_checkpoints
# ---------------------------------------------------------------------------


class TestDeleteStaleCheckpoints:
    def test_deletes_old(self, tmp_path):
        old_ts = _ago_iso(10)
        path = _make_checkpoint(str(tmp_path), created_at=old_ts)
        info = CheckpointInfo(
            path=path, run_id="r", processed_count=0,
            total_count=0, created_at=old_ts, size_bytes=0,
        )
        summary = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=0)
        count = delete_stale_checkpoints(summary, max_age_days=7.0)
        assert count == 1
        assert not os.path.exists(path)

    def test_keeps_recent(self, tmp_path):
        recent_ts = _now_iso()
        path = _make_checkpoint(str(tmp_path), created_at=recent_ts)
        info = CheckpointInfo(
            path=path, run_id="r", processed_count=0,
            total_count=0, created_at=recent_ts, size_bytes=0,
        )
        summary = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=0)
        count = delete_stale_checkpoints(summary, max_age_days=7.0)
        assert count == 0
        assert os.path.exists(path)

    def test_empty_created_at_skipped(self, tmp_path):
        path = _make_checkpoint(str(tmp_path))
        info = CheckpointInfo(
            path=path, run_id="r", processed_count=0,
            total_count=0, created_at="", size_bytes=0,
        )
        summary = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=0)
        count = delete_stale_checkpoints(summary, max_age_days=1.0)
        assert count == 0

    def test_custom_max_age(self, tmp_path):
        ts = _ago_iso(2)
        path = _make_checkpoint(str(tmp_path), created_at=ts)
        info = CheckpointInfo(
            path=path, run_id="r", processed_count=0,
            total_count=0, created_at=ts, size_bytes=0,
        )
        summary = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=0)
        count = delete_stale_checkpoints(summary, max_age_days=1.0)
        assert count == 1

    def test_mixed_old_and_new(self, tmp_path):
        old_ts = _ago_iso(10)
        new_ts = _now_iso()
        old_path = _make_checkpoint(str(tmp_path), name="checkpoint_old.json", created_at=old_ts)
        new_path = _make_checkpoint(str(tmp_path), name="checkpoint_new.json", created_at=new_ts)
        infos = [
            CheckpointInfo(path=old_path, run_id="r1", processed_count=0, total_count=0, created_at=old_ts, size_bytes=0),
            CheckpointInfo(path=new_path, run_id="r2", processed_count=0, total_count=0, created_at=new_ts, size_bytes=0),
        ]
        summary = CheckpointSummary(checkpoints=infos, total_count=2, total_size_bytes=0)
        count = delete_stale_checkpoints(summary, max_age_days=7.0)
        assert count == 1
        assert not os.path.exists(old_path)
        assert os.path.exists(new_path)


# ---------------------------------------------------------------------------
# _format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        result = _format_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = _format_size(2 * 1024 * 1024)
        assert "MB" in result

    def test_zero(self):
        assert _format_size(0) == "0 B"


# ---------------------------------------------------------------------------
# _format_age
# ---------------------------------------------------------------------------


class TestFormatAge:
    def test_empty_string(self):
        assert _format_age("") == "unknown"

    def test_recent(self):
        ts = _now_iso()
        result = _format_age(ts)
        assert "s ago" in result or "0" in result

    def test_invalid(self):
        assert _format_age("not-a-date") == "unknown"


# ---------------------------------------------------------------------------
# format_checkpoint_list
# ---------------------------------------------------------------------------


class TestFormatCheckpointList:
    def test_empty(self):
        s = CheckpointSummary()
        text = format_checkpoint_list(s)
        assert "No checkpoints" in text

    def test_with_checkpoints(self):
        info = CheckpointInfo(
            path="/data/checkpoint.json",
            run_id="r1",
            processed_count=5,
            total_count=10,
            created_at=_now_iso(),
            size_bytes=1024,
        )
        s = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=1024)
        text = format_checkpoint_list(s)
        assert "checkpoint" in text.lower()
        assert "1" in text

    def test_long_path_truncated(self):
        long_path = "/very/long/path/" + "x" * 50 + "/checkpoint.json"
        info = CheckpointInfo(
            path=long_path, run_id="r", processed_count=1,
            total_count=2, created_at=_now_iso(), size_bytes=100,
        )
        s = CheckpointSummary(checkpoints=[info], total_count=1, total_size_bytes=100)
        text = format_checkpoint_list(s)
        assert "..." in text


# ---------------------------------------------------------------------------
# format_checkpoint_detail
# ---------------------------------------------------------------------------


class TestFormatCheckpointDetail:
    def test_basic(self):
        info = CheckpointInfo(
            path="/data/checkpoint.json",
            run_id="run-42",
            processed_count=7,
            total_count=20,
            created_at=_now_iso(),
            size_bytes=4096,
        )
        text = format_checkpoint_detail(info)
        assert "run-42" in text
        assert "7/20" in text
        assert "35%" in text

    def test_unknown_created_at(self):
        info = CheckpointInfo(
            path="p", run_id="r", processed_count=0,
            total_count=0, created_at="", size_bytes=0,
        )
        text = format_checkpoint_detail(info)
        assert "unknown" in text

    def test_contains_all_fields(self):
        info = CheckpointInfo(
            path="/x/y.json", run_id="r1", processed_count=10,
            total_count=10, created_at=_now_iso(), size_bytes=512,
        )
        text = format_checkpoint_detail(info)
        assert "Run ID" in text
        assert "Progress" in text
        assert "Size" in text
        assert "Age" in text
