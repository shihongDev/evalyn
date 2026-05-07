"""Regression tests for the cold-path FS walk in ``api.v2._shared``.

The scandir refactor changed the implementation of
``list_dataset_dirs``/``_walk_run_dirs`` from ``Path.iterdir() +
Path.is_dir()`` (two syscalls per entry on POSIX) to ``os.scandir`` with
cached ``DirEntry.is_dir()`` (one syscall on the listing). These tests
pin the contract that mattered: shape, sort order, and tolerance for
common error states (missing dirs, files where dirs were expected).
"""

from __future__ import annotations

from pathlib import Path

from evalyn_dashboard.api.v2._shared import (
    _scandir_subdirs,
    _walk_run_dirs,
    list_dataset_dirs,
)


def test_scandir_subdirs_returns_only_directories_sorted(tmp_path: Path) -> None:
    """Files are filtered out; directory order is name-ascending."""
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "m_dir").mkdir()
    (tmp_path / "regular_file.txt").write_text("hi")
    out = _scandir_subdirs(tmp_path)
    assert [p.name for p in out] == ["a_dir", "m_dir", "z_dir"]


def test_scandir_subdirs_missing_root_is_empty(tmp_path: Path) -> None:
    """A nonexistent root returns [] rather than raising."""
    assert _scandir_subdirs(tmp_path / "nope") == []


def test_list_dataset_dirs_skips_files(tmp_path: Path) -> None:
    """``list_dataset_dirs`` honors the same dir-only contract."""
    (tmp_path / "ds1").mkdir()
    (tmp_path / "ds2").mkdir()
    (tmp_path / "stray.json").write_text("{}")
    out = list_dataset_dirs(tmp_path)
    names = [p.name for p in out]
    assert names == ["ds1", "ds2"]


def test_walk_run_dirs_skips_datasets_without_eval_runs(tmp_path: Path) -> None:
    """The classic cold-start case: many datasets, most without an
    eval_runs subdir. The walk must skip them silently and only return
    the inner run dirs from the few that have data."""
    # ds1 has eval_runs with two runs.
    ds1 = tmp_path / "ds1"
    (ds1 / "eval_runs" / "run_a").mkdir(parents=True)
    (ds1 / "eval_runs" / "run_b").mkdir(parents=True)
    # ds2 has eval_runs but it's EMPTY.
    ds2 = tmp_path / "ds2"
    (ds2 / "eval_runs").mkdir(parents=True)
    # ds3 has NO eval_runs subdir at all (the placeholder case).
    ds3 = tmp_path / "ds3"
    ds3.mkdir()
    (ds3 / "metadata.json").write_text("{}")
    # ds4 has eval_runs as a FILE (corrupt state) - must not crash.
    ds4 = tmp_path / "ds4"
    ds4.mkdir()
    (ds4 / "eval_runs").write_text("oops")

    out = _walk_run_dirs(tmp_path)
    names = sorted(p.name for p in out)
    assert names == ["run_a", "run_b"]


def test_walk_run_dirs_includes_only_subdirectories_inside_eval_runs(
    tmp_path: Path,
) -> None:
    """Files inside ``eval_runs/`` are skipped; only run subdirs returned."""
    ds = tmp_path / "ds1"
    (ds / "eval_runs" / "run_x").mkdir(parents=True)
    (ds / "eval_runs" / "stray.json").write_text("{}")
    out = _walk_run_dirs(tmp_path)
    assert [p.name for p in out] == ["run_x"]
