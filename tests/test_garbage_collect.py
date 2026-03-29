"""Tests for the garbage collection module."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.garbage_collect import (
    GCConfig,
    GCResult,
    OrphanedItem,
    _format_bytes,
    _get_size,
    collect_garbage,
    find_orphaned_checkpoints,
    find_orphaned_runs,
    find_temp_files,
    format_gc_report,
    remove_item,
)


# ---------------------------------------------------------------------------
# OrphanedItem
# ---------------------------------------------------------------------------


class TestOrphanedItem:
    def test_create(self):
        item = OrphanedItem(
            path="/tmp/checkpoint_001.json",
            item_type="checkpoint",
            size_bytes=1024,
            reason="no run dir",
        )
        assert item.path == "/tmp/checkpoint_001.json"
        assert item.item_type == "checkpoint"
        assert item.size_bytes == 1024
        assert item.reason == "no run dir"

    def test_as_dict(self):
        item = OrphanedItem("a.json", "checkpoint", 512, "orphaned")
        d = item.as_dict()
        assert d["path"] == "a.json"
        assert d["item_type"] == "checkpoint"
        assert d["size_bytes"] == 512
        assert d["reason"] == "orphaned"

    def test_from_dict(self):
        d = {"path": "b.json", "item_type": "run", "size_bytes": 2048, "reason": "no dataset"}
        item = OrphanedItem.from_dict(d)
        assert item.path == "b.json"
        assert item.item_type == "run"
        assert item.size_bytes == 2048

    def test_from_dict_defaults(self):
        item = OrphanedItem.from_dict({})
        assert item.path == ""
        assert item.item_type == "temp"
        assert item.size_bytes == 0
        assert item.reason == ""

    def test_roundtrip(self):
        original = OrphanedItem("/x/y", "temp", 999, "old file")
        restored = OrphanedItem.from_dict(original.as_dict())
        assert restored.path == original.path
        assert restored.item_type == original.item_type
        assert restored.size_bytes == original.size_bytes
        assert restored.reason == original.reason


# ---------------------------------------------------------------------------
# GCConfig
# ---------------------------------------------------------------------------


class TestGCConfig:
    def test_defaults(self):
        config = GCConfig()
        assert config.data_dir == "data"
        assert config.temp_dir == ".evalyn"
        assert config.dry_run is True
        assert config.max_age_days == 30.0

    def test_custom_values(self):
        config = GCConfig(data_dir="/custom", temp_dir="/tmp/ev", dry_run=False, max_age_days=7.0)
        assert config.data_dir == "/custom"
        assert config.dry_run is False
        assert config.max_age_days == 7.0

    def test_as_dict(self):
        config = GCConfig()
        d = config.as_dict()
        assert d["data_dir"] == "data"
        assert d["dry_run"] is True

    def test_from_dict(self):
        d = {"data_dir": "mydata", "temp_dir": "mytmp", "dry_run": False, "max_age_days": 5.0}
        config = GCConfig.from_dict(d)
        assert config.data_dir == "mydata"
        assert config.dry_run is False
        assert config.max_age_days == 5.0

    def test_from_dict_defaults(self):
        config = GCConfig.from_dict({})
        assert config.data_dir == "data"
        assert config.dry_run is True

    def test_roundtrip(self):
        original = GCConfig("x", "y", False, 15.0)
        restored = GCConfig.from_dict(original.as_dict())
        assert restored.data_dir == original.data_dir
        assert restored.temp_dir == original.temp_dir
        assert restored.dry_run == original.dry_run
        assert restored.max_age_days == original.max_age_days


# ---------------------------------------------------------------------------
# GCResult
# ---------------------------------------------------------------------------


class TestGCResult:
    def test_defaults(self):
        result = GCResult()
        assert result.items_found == []
        assert result.items_removed == 0
        assert result.bytes_freed == 0
        assert result.dry_run is True

    def test_as_dict(self):
        item = OrphanedItem("a", "temp", 100, "old")
        result = GCResult(items_found=[item], items_removed=1, bytes_freed=100, dry_run=False)
        d = result.as_dict()
        assert len(d["items_found"]) == 1
        assert d["items_removed"] == 1
        assert d["bytes_freed"] == 100
        assert d["dry_run"] is False

    def test_from_dict(self):
        d = {
            "items_found": [{"path": "x", "item_type": "checkpoint", "size_bytes": 50, "reason": "r"}],
            "items_removed": 1,
            "bytes_freed": 50,
            "dry_run": False,
        }
        result = GCResult.from_dict(d)
        assert len(result.items_found) == 1
        assert result.items_found[0].path == "x"
        assert result.items_removed == 1

    def test_from_dict_defaults(self):
        result = GCResult.from_dict({})
        assert result.items_found == []
        assert result.items_removed == 0
        assert result.dry_run is True

    def test_roundtrip(self):
        item = OrphanedItem("z", "run", 200, "missing dataset")
        original = GCResult([item], 1, 200, False)
        restored = GCResult.from_dict(original.as_dict())
        assert len(restored.items_found) == 1
        assert restored.items_removed == 1
        assert restored.bytes_freed == 200
        assert restored.dry_run is False


# ---------------------------------------------------------------------------
# find_orphaned_checkpoints
# ---------------------------------------------------------------------------


class TestFindOrphanedCheckpoints:
    def test_no_data_dir(self, tmp_path):
        result = find_orphaned_checkpoints(str(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_dir(self, tmp_path):
        result = find_orphaned_checkpoints(str(tmp_path))
        assert result == []

    def test_checkpoint_with_run_dir_not_orphaned(self, tmp_path):
        (tmp_path / "checkpoint_001.json").write_text("{}")
        (tmp_path / "run_001").mkdir()
        result = find_orphaned_checkpoints(str(tmp_path))
        assert result == []

    def test_checkpoint_without_run_dir_is_orphaned(self, tmp_path):
        (tmp_path / "checkpoint_001.json").write_text("{}")
        result = find_orphaned_checkpoints(str(tmp_path))
        assert len(result) == 1
        assert result[0].item_type == "checkpoint"
        assert "checkpoint_001.json" in result[0].path

    def test_multiple_checkpoints_mixed(self, tmp_path):
        # Sub-dir A has both checkpoint and run dir
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "checkpoint_a.json").write_text("{}")
        (dir_a / "run_a").mkdir()

        # Sub-dir B has only checkpoint
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        (dir_b / "checkpoint_b.json").write_text("{}")

        result = find_orphaned_checkpoints(str(tmp_path))
        assert len(result) == 1
        assert "checkpoint_b.json" in result[0].path

    def test_nested_checkpoint_orphaned(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        nested.mkdir(parents=True)
        (nested / "checkpoint_x.json").write_text("{}")
        result = find_orphaned_checkpoints(str(tmp_path))
        assert len(result) == 1

    def test_size_reported(self, tmp_path):
        content = '{"data": "some checkpoint content"}'
        (tmp_path / "checkpoint_big.json").write_text(content)
        result = find_orphaned_checkpoints(str(tmp_path))
        assert result[0].size_bytes > 0


# ---------------------------------------------------------------------------
# find_orphaned_runs
# ---------------------------------------------------------------------------


class TestFindOrphanedRuns:
    def test_no_data_dir(self, tmp_path):
        result = find_orphaned_runs(str(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_dir(self, tmp_path):
        result = find_orphaned_runs(str(tmp_path))
        assert result == []

    def test_run_with_dataset_not_orphaned(self, tmp_path):
        (tmp_path / "dataset.jsonl").write_text('{"id": "1"}\n')
        (tmp_path / "run_001").mkdir()
        result = find_orphaned_runs(str(tmp_path))
        assert result == []

    def test_run_without_dataset_is_orphaned(self, tmp_path):
        (tmp_path / "run_001").mkdir()
        result = find_orphaned_runs(str(tmp_path))
        assert len(result) == 1
        assert result[0].item_type == "run"
        assert "run_001" in result[0].path

    def test_run_file_not_dir_ignored(self, tmp_path):
        (tmp_path / "run_001").write_text("not a dir")
        result = find_orphaned_runs(str(tmp_path))
        assert result == []

    def test_multiple_runs_mixed(self, tmp_path):
        # Good: dataset dir with run
        good = tmp_path / "good_dataset"
        good.mkdir()
        (good / "dataset.jsonl").write_text('{"id": "1"}\n')
        (good / "run_001").mkdir()

        # Bad: run without dataset
        bad = tmp_path / "orphan_dir"
        bad.mkdir()
        (bad / "run_002").mkdir()

        result = find_orphaned_runs(str(tmp_path))
        assert len(result) == 1
        assert "run_002" in result[0].path

    def test_nested_orphaned_run(self, tmp_path):
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "run_deep").mkdir()
        result = find_orphaned_runs(str(tmp_path))
        assert len(result) == 1

    def test_size_of_run_dir(self, tmp_path):
        run_dir = tmp_path / "run_big"
        run_dir.mkdir()
        (run_dir / "results.json").write_text('{"big": "data"}')
        result = find_orphaned_runs(str(tmp_path))
        assert result[0].size_bytes > 0


# ---------------------------------------------------------------------------
# find_temp_files
# ---------------------------------------------------------------------------


class TestFindTempFiles:
    def test_no_temp_dir(self, tmp_path):
        result = find_temp_files(str(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_dir(self, tmp_path):
        result = find_temp_files(str(tmp_path))
        assert result == []

    def test_recent_file_not_found(self, tmp_path):
        (tmp_path / "recent.tmp").write_text("data")
        result = find_temp_files(str(tmp_path), max_age_days=30.0)
        assert result == []

    def test_old_file_found(self, tmp_path):
        old_file = tmp_path / "old.tmp"
        old_file.write_text("stale data")
        # Set mtime to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(str(old_file), (old_time, old_time))
        result = find_temp_files(str(tmp_path), max_age_days=30.0)
        assert len(result) == 1
        assert result[0].item_type == "temp"
        assert "old.tmp" in result[0].path

    def test_max_age_zero_finds_all(self, tmp_path):
        (tmp_path / "any.tmp").write_text("x")
        # Files just created have age > 0 seconds, so max_age_days=0 catches them
        result = find_temp_files(str(tmp_path), max_age_days=0.0)
        assert len(result) == 1

    def test_subdirectory_files(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "nested.cache"
        f.write_text("cached")
        old_time = time.time() - (45 * 86400)
        os.utime(str(f), (old_time, old_time))
        result = find_temp_files(str(tmp_path), max_age_days=30.0)
        assert len(result) == 1

    def test_reason_includes_age(self, tmp_path):
        f = tmp_path / "aged.tmp"
        f.write_text("old")
        old_time = time.time() - (90 * 86400)
        os.utime(str(f), (old_time, old_time))
        result = find_temp_files(str(tmp_path), max_age_days=30.0)
        assert "days old" in result[0].reason

    def test_directories_excluded(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = find_temp_files(str(tmp_path), max_age_days=0.0)
        # Only files, not directories
        for item in result:
            assert Path(item.path).is_file()


# ---------------------------------------------------------------------------
# remove_item
# ---------------------------------------------------------------------------


class TestRemoveItem:
    def test_remove_file(self, tmp_path):
        f = tmp_path / "to_delete.json"
        f.write_text("{}")
        item = OrphanedItem(str(f), "checkpoint", 2, "test")
        assert remove_item(item) is True
        assert not f.exists()

    def test_remove_directory(self, tmp_path):
        d = tmp_path / "run_to_delete"
        d.mkdir()
        (d / "inner.json").write_text("{}")
        item = OrphanedItem(str(d), "run", 100, "test")
        assert remove_item(item) is True
        assert not d.exists()

    def test_remove_nonexistent_returns_false(self, tmp_path):
        item = OrphanedItem(str(tmp_path / "gone"), "temp", 0, "test")
        assert remove_item(item) is False

    def test_remove_file_permission_error(self, tmp_path):
        f = tmp_path / "locked.json"
        f.write_text("{}")
        item = OrphanedItem(str(f), "checkpoint", 2, "test")
        with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
            assert remove_item(item) is False


# ---------------------------------------------------------------------------
# collect_garbage
# ---------------------------------------------------------------------------


class TestCollectGarbage:
    def test_dry_run_does_not_delete(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "checkpoint_001.json").write_text("{}")
        temp_dir = tmp_path / ".evalyn"
        temp_dir.mkdir()

        config = GCConfig(
            data_dir=str(data_dir),
            temp_dir=str(temp_dir),
            dry_run=True,
        )
        result = collect_garbage(config)
        assert result.dry_run is True
        assert len(result.items_found) >= 1
        assert result.items_removed == 0
        # File still exists
        assert (data_dir / "checkpoint_001.json").exists()

    def test_actual_cleanup(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "checkpoint_orphan.json").write_text('{"x": 1}')

        temp_dir = tmp_path / ".evalyn"
        temp_dir.mkdir()

        config = GCConfig(
            data_dir=str(data_dir),
            temp_dir=str(temp_dir),
            dry_run=False,
        )
        result = collect_garbage(config)
        assert result.dry_run is False
        assert result.items_removed >= 1
        assert result.bytes_freed > 0
        assert not (data_dir / "checkpoint_orphan.json").exists()

    def test_combines_all_types(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Orphaned checkpoint in a sub-dir with no run_* sibling
        ck_dir = data_dir / "ck_only"
        ck_dir.mkdir()
        (ck_dir / "checkpoint_001.json").write_text("{}")
        # Orphaned run in a different sub-dir with no dataset.jsonl
        run_dir = data_dir / "runs_only"
        run_dir.mkdir()
        (run_dir / "run_orphan").mkdir()

        temp_dir = tmp_path / ".evalyn"
        temp_dir.mkdir()
        old_file = temp_dir / "old.tmp"
        old_file.write_text("stale")
        old_time = time.time() - (60 * 86400)
        os.utime(str(old_file), (old_time, old_time))

        config = GCConfig(
            data_dir=str(data_dir),
            temp_dir=str(temp_dir),
            dry_run=True,
        )
        result = collect_garbage(config)
        types_found = {item.item_type for item in result.items_found}
        assert "checkpoint" in types_found
        assert "run" in types_found
        assert "temp" in types_found

    def test_empty_dirs_no_errors(self, tmp_path):
        config = GCConfig(
            data_dir=str(tmp_path / "nodata"),
            temp_dir=str(tmp_path / "notemp"),
            dry_run=True,
        )
        result = collect_garbage(config)
        assert result.items_found == []
        assert result.items_removed == 0


# ---------------------------------------------------------------------------
# format_gc_report
# ---------------------------------------------------------------------------


class TestFormatGCReport:
    def test_empty_report(self):
        result = GCResult(dry_run=True)
        report = format_gc_report(result)
        assert "No orphaned items found" in report
        assert "DRY RUN" in report

    def test_dry_run_report(self):
        items = [
            OrphanedItem("/a/checkpoint_1.json", "checkpoint", 1024, "no run"),
            OrphanedItem("/b/run_old", "run", 4096, "no dataset"),
        ]
        result = GCResult(items_found=items, dry_run=True)
        report = format_gc_report(result)
        assert "DRY RUN" in report
        assert "Items found: 2" in report
        assert "Would remove: 2" in report
        assert "checkpoint" in report
        assert "run" in report

    def test_cleanup_report(self):
        items = [OrphanedItem("/x/y", "temp", 500, "old")]
        result = GCResult(items_found=items, items_removed=1, bytes_freed=500, dry_run=False)
        report = format_gc_report(result)
        assert "CLEANUP" in report
        assert "Removed: 1" in report
        assert "Freed:" in report

    def test_report_grouped_by_type(self):
        items = [
            OrphanedItem("a", "checkpoint", 10, "r1"),
            OrphanedItem("b", "checkpoint", 20, "r2"),
            OrphanedItem("c", "run", 30, "r3"),
        ]
        result = GCResult(items_found=items, dry_run=True)
        report = format_gc_report(result)
        assert "checkpoint (2 items" in report
        assert "run (1 items" in report


# ---------------------------------------------------------------------------
# _format_bytes
# ---------------------------------------------------------------------------


class TestFormatBytes:
    def test_bytes(self):
        assert _format_bytes(0) == "0 B"
        assert _format_bytes(500) == "500 B"
        assert _format_bytes(1023) == "1023 B"

    def test_kilobytes(self):
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_bytes(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert _format_bytes(1024 * 1024 * 1024) == "1.0 GB"


# ---------------------------------------------------------------------------
# _get_size
# ---------------------------------------------------------------------------


class TestGetSize:
    def test_file_size(self, tmp_path):
        f = tmp_path / "sized.txt"
        content = "hello world"
        f.write_text(content)
        assert _get_size(str(f)) == len(content)

    def test_dir_size(self, tmp_path):
        d = tmp_path / "sized_dir"
        d.mkdir()
        (d / "a.txt").write_text("aaa")
        (d / "b.txt").write_text("bbbbb")
        assert _get_size(str(d)) == 8

    def test_nonexistent(self, tmp_path):
        assert _get_size(str(tmp_path / "nope")) == 0

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _get_size(str(d)) == 0
