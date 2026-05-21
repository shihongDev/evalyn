"""Tests for evaluation hooks and dataset versioning."""

import json
import pytest
from pathlib import Path
from evalyn_sdk.evaluation.hooks import EvalHooks
from evalyn_sdk.datasets.versioning import (
    create_snapshot, list_versions, rollback_to_version,
    compute_version_hash, DatasetVersion,
)


# =============================================================================
# Event callback hooks tests
# =============================================================================

class TestEvalHooks:
    def test_no_hooks_by_default(self):
        hooks = EvalHooks()
        assert not hooks.has_hooks

    def test_register_item_complete(self):
        hooks = EvalHooks()
        called = []
        hooks.on_item_complete(lambda item_id, results: called.append(item_id))
        hooks.fire_item_complete("item-1", [])
        assert called == ["item-1"]

    def test_register_metric_complete(self):
        hooks = EvalHooks()
        called = []
        hooks.on_metric_complete(lambda mid, iid, r: called.append((mid, iid)))
        hooks.fire_metric_complete("help", "item-1", None)
        assert called == [("help", "item-1")]

    def test_register_run_complete(self):
        hooks = EvalHooks()
        runs = []
        hooks.on_run_complete(lambda run: runs.append(run))
        hooks.fire_run_complete("mock_run")
        assert runs == ["mock_run"]

    def test_register_run_start(self):
        hooks = EvalHooks()
        starts = []
        hooks.on_run_start(lambda rid, ic, mc: starts.append((rid, ic, mc)))
        hooks.fire_run_start("run-1", 100, 5)
        assert starts == [("run-1", 100, 5)]

    def test_register_error(self):
        hooks = EvalHooks()
        errors = []
        hooks.on_error(lambda mid, iid, err: errors.append(str(err)))
        hooks.fire_error("help", "i1", ValueError("test error"))
        assert errors == ["test error"]

    def test_multiple_callbacks(self):
        hooks = EvalHooks()
        calls_a, calls_b = [], []
        hooks.on_item_complete(lambda iid, r: calls_a.append(iid))
        hooks.on_item_complete(lambda iid, r: calls_b.append(iid))
        hooks.fire_item_complete("item-1", [])
        assert calls_a == ["item-1"]
        assert calls_b == ["item-1"]

    def test_callback_exception_caught(self):
        hooks = EvalHooks()
        hooks.on_item_complete(lambda iid, r: 1 / 0)  # will raise
        # Should not raise
        hooks.fire_item_complete("item-1", [])

    def test_has_hooks(self):
        hooks = EvalHooks()
        assert not hooks.has_hooks
        hooks.on_run_complete(lambda r: None)
        assert hooks.has_hooks

    def test_summary(self):
        hooks = EvalHooks()
        hooks.on_item_complete(lambda iid, r: None)
        hooks.on_run_complete(lambda r: None)
        hooks.on_run_complete(lambda r: None)
        s = hooks.summary()
        assert s["on_item_complete"] == 1
        assert s["on_run_complete"] == 2
        assert s["on_metric_complete"] == 0

    def test_fire_without_hooks(self):
        hooks = EvalHooks()
        # Should not raise
        hooks.fire_item_complete("i1", [])
        hooks.fire_metric_complete("m1", "i1", None)
        hooks.fire_run_complete(None)
        hooks.fire_run_start("r1", 0, 0)
        hooks.fire_error("m1", "i1", ValueError("x"))


# =============================================================================
# Dataset versioning tests
# =============================================================================

def _write_dataset(dataset_dir: Path, items: list) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / "dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return dataset_file


class TestComputeVersionHash:
    def test_deterministic(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1", "input": "hello"}])
        h1 = compute_version_hash(tmp_path / "dataset.jsonl")
        h2 = compute_version_hash(tmp_path / "dataset.jsonl")
        assert h1 == h2

    def test_different_content(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        _write_dataset(d1, [{"id": "1", "input": "hello"}])
        _write_dataset(d2, [{"id": "1", "input": "world"}])
        assert compute_version_hash(d1 / "dataset.jsonl") != compute_version_hash(d2 / "dataset.jsonl")

    def test_missing_file(self, tmp_path):
        assert compute_version_hash(tmp_path / "nonexistent.jsonl") == ""

    def test_hash_length(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}])
        assert len(compute_version_hash(tmp_path / "dataset.jsonl")) == 16


class TestCreateSnapshot:
    def test_creates_snapshot(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}, {"id": "2"}])
        version = create_snapshot(tmp_path, description="initial")
        assert version is not None
        assert version.item_count == 2
        assert version.description == "initial"
        assert (tmp_path / ".versions").exists()

    def test_no_dataset(self, tmp_path):
        assert create_snapshot(tmp_path) is None

    def test_duplicate_snapshot_skipped(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}])
        v1 = create_snapshot(tmp_path)
        v2 = create_snapshot(tmp_path)  # same content
        assert v1 is not None
        assert v2 is None  # already exists

    def test_different_content_new_snapshot(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}])
        v1 = create_snapshot(tmp_path)
        _write_dataset(tmp_path, [{"id": "1"}, {"id": "2"}])  # add item
        v2 = create_snapshot(tmp_path)
        assert v1 is not None
        assert v2 is not None
        assert v1.version_hash != v2.version_hash


class TestListVersions:
    def test_list_versions(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}])
        create_snapshot(tmp_path, "v1")
        _write_dataset(tmp_path, [{"id": "1"}, {"id": "2"}])
        create_snapshot(tmp_path, "v2")

        versions = list_versions(tmp_path)
        assert len(versions) == 2
        assert versions[0].description == "v2"  # newest first
        assert versions[1].description == "v1"

    def test_empty(self, tmp_path):
        assert list_versions(tmp_path) == []


class TestRollback:
    def test_rollback_to_previous(self, tmp_path):
        # Create v1
        _write_dataset(tmp_path, [{"id": "1"}])
        v1 = create_snapshot(tmp_path, "v1")
        v1_hash = v1.version_hash

        # Create v2 (different content)
        _write_dataset(tmp_path, [{"id": "1"}, {"id": "2"}])
        create_snapshot(tmp_path, "v2")

        # Rollback to v1
        assert rollback_to_version(tmp_path, v1_hash) is True

        # Verify content matches v1
        dataset_file = tmp_path / "dataset.jsonl"
        lines = dataset_file.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_rollback_nonexistent(self, tmp_path):
        assert rollback_to_version(tmp_path, "nonexistent") is False

    def test_rollback_creates_safety_snapshot(self, tmp_path):
        _write_dataset(tmp_path, [{"id": "1"}])
        v1 = create_snapshot(tmp_path)
        _write_dataset(tmp_path, [{"id": "1"}, {"id": "2"}])

        rollback_to_version(tmp_path, v1.version_hash)

        # Should have auto-snapshot of the v2 content
        versions = list_versions(tmp_path)
        assert len(versions) >= 2


class TestDatasetVersionSerialization:
    def test_roundtrip(self):
        v = DatasetVersion(
            version_hash="abc123", item_count=50,
            created_at="2026-03-28T00:00:00", description="test",
        )
        d = v.as_dict()
        restored = DatasetVersion.from_dict(d)
        assert restored.version_hash == "abc123"
        assert restored.item_count == 50
