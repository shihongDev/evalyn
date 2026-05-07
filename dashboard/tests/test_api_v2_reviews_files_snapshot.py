"""Regression tests for the on-disk snapshot of ``_reviews_files_for_root``.

The snapshot pattern mirrors the run-dirs one: persist the result of a
cold filesystem walk to ``.evalyn/cache/reviews_files/<hash>.json`` so
the first request after a fresh dashboard launch skips the multi-second
WSL/NTFS glob walk over hundreds of placeholder ``reviews/`` dirs.

These tests pin the contract:
  - cold walk persists; warm boot restores from snapshot
  - mtime change invalidates the snapshot
  - vanished files are silently skipped (defensive)
  - corrupt / mismatched snapshots fall back to walk
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from evalyn_dashboard.api.v2 import _shared


def _seed_dataset_with_reviews(root: Path, name: str, files: list[str]) -> None:
    """Create ``root/name/reviews/<files>``."""
    reviews = root / name / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    for f in files:
        (reviews / f).write_text("[]")


def test_reviews_files_snapshot_persists_and_restores(tmp_path: Path) -> None:
    """Cold walk writes the snapshot; a fresh in-memory cache restores
    from the on-disk one without re-walking."""
    root = tmp_path / "datasets"
    root.mkdir()
    _seed_dataset_with_reviews(root, "ds_alpha", ["a.jsonl", "b.jsonl"])
    _seed_dataset_with_reviews(root, "ds_beta", ["x.jsonl"])
    # Empty dataset (no reviews/) — exercises the placeholder-skip path.
    (root / "ds_empty").mkdir()

    # Force snapshots into the test's tmp_path so we don't pollute cwd.
    snap_dir = tmp_path / "snap"
    with patch.object(
        _shared, "_reviews_files_snapshot_dir", lambda: snap_dir
    ):
        _shared._clear_caches_for_tests()

        # First call: cold walk + snapshot write.
        out1 = _shared._reviews_files_for_root(root)
        names1 = sorted(p.name for p, _ in out1)
        assert names1 == ["a.jsonl", "b.jsonl", "x.jsonl"]
        # Snapshot file should exist.
        snap_path = _shared._reviews_files_snapshot_path(root)
        assert snap_path.is_file()

        # Drop in-memory cache; second call should hit the disk snapshot
        # and return the same answer WITHOUT re-walking.
        _shared._reviews_dirs_cache.clear()
        with patch.object(_shared, "_walk_reviews_files") as fake_walk:
            out2 = _shared._reviews_files_for_root(root)
            assert fake_walk.call_count == 0, "snapshot path should NOT walk"
        assert sorted(p.name for p, _ in out2) == names1


def test_reviews_files_snapshot_invalidated_on_mtime_change(
    tmp_path: Path,
) -> None:
    """Adding a dataset bumps the root mtime, which invalidates the
    snapshot. The next call must walk the filesystem."""
    root = tmp_path / "datasets"
    root.mkdir()
    _seed_dataset_with_reviews(root, "ds1", ["a.jsonl"])

    snap_dir = tmp_path / "snap"
    with patch.object(
        _shared, "_reviews_files_snapshot_dir", lambda: snap_dir
    ):
        _shared._clear_caches_for_tests()
        first = _shared._reviews_files_for_root(root)
        assert len(first) == 1

        # Bump the root: add a second dataset with reviews.
        _seed_dataset_with_reviews(root, "ds2", ["b.jsonl"])
        # Touch the root explicitly (mkdir-of-child should bump it on
        # most FSes; force it for determinism).
        os.utime(root, None)

        # Drop in-memory cache so the next call is forced through the
        # snapshot/walk decision.
        _shared._reviews_dirs_cache.clear()
        # Also need to drop the dataset_dirs cache so the walk sees ds2.
        _shared._dataset_dirs_cache.clear()
        second = _shared._reviews_files_for_root(root)
        names = sorted(p.name for p, _ in second)
        assert names == ["a.jsonl", "b.jsonl"], (
            "mtime bump must trigger a fresh walk"
        )


def test_reviews_files_snapshot_skips_vanished_files(tmp_path: Path) -> None:
    """If a file disappears between snapshot write and read (developer
    cleanup, branch switch), the loader silently skips it rather than
    poisoning the cache with a missing path."""
    root = tmp_path / "datasets"
    root.mkdir()
    _seed_dataset_with_reviews(root, "ds1", ["a.jsonl", "b.jsonl"])

    snap_dir = tmp_path / "snap"
    with patch.object(
        _shared, "_reviews_files_snapshot_dir", lambda: snap_dir
    ):
        _shared._clear_caches_for_tests()
        first = _shared._reviews_files_for_root(root)
        assert len(first) == 2

        # Delete one file directly. Snapshot was written before the
        # delete; we want the next load to skip it without crashing.
        (root / "ds1" / "reviews" / "b.jsonl").unlink()

        _shared._reviews_dirs_cache.clear()
        loaded = _shared._load_reviews_files_snapshot(
            root, root.stat().st_mtime,
        )
        # Only the surviving file should come back.
        assert loaded is not None
        assert [p.name for p, _ in loaded] == ["a.jsonl"]


def test_reviews_files_snapshot_corrupt_falls_back_to_walk(
    tmp_path: Path,
) -> None:
    """A poisoned JSON file returns None from the loader so the caller
    falls back to the walk + rewrite pattern."""
    root = tmp_path / "datasets"
    root.mkdir()
    _seed_dataset_with_reviews(root, "ds1", ["a.jsonl"])

    snap_dir = tmp_path / "snap"
    snap_dir.mkdir()
    with patch.object(
        _shared, "_reviews_files_snapshot_dir", lambda: snap_dir
    ):
        # Pre-create a malformed snapshot at the expected path.
        bad_path = _shared._reviews_files_snapshot_path(root)
        bad_path.write_text("{not json")

        loaded = _shared._load_reviews_files_snapshot(
            root, root.stat().st_mtime,
        )
        assert loaded is None

        # The full helper should fall back to walk + rewrite cleanly.
        _shared._reviews_dirs_cache.clear()
        out = _shared._reviews_files_for_root(root)
        assert [p.name for p, _ in out] == ["a.jsonl"]
        # The corrupt snapshot was overwritten with a valid one.
        rewritten = json.loads(bad_path.read_text(encoding="utf-8"))
        assert rewritten["root"] == str(root)
        assert isinstance(rewritten["files"], list)
