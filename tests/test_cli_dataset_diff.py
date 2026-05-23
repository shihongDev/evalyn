"""Unit tests for `evalyn dataset-diff` CLI command.

Covers:
- _resolve_dataset_file: directory + direct file inputs, missing path errors
- cmd_dataset_diff with no differences (identical datasets)
- cmd_dataset_diff with added items only
- cmd_dataset_diff with removed items only
- cmd_dataset_diff with added + removed + modified
- --format json produces valid JSON
- --show-items caps output at 10 entries and reports the truncated tail

Run with: uv run pytest tests/test_cli_dataset_diff.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make sure the SDK is importable independent of test collection order.
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.cli.commands import dataset_diff as dataset_diff_mod
from evalyn_sdk.cli.commands.dataset_diff import (
    _SHOW_ITEMS_LIMIT,
    _render_json,
    _render_table,
    _resolve_dataset_file,
    cmd_dataset_diff,
    register_commands,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_dataset(path: Path, items: list[dict]) -> Path:
    """Write a list of dict items as a JSONL dataset file. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return path


def _make_item(item_id: str, *, prompt: str = "hello", answer: str = "world") -> dict:
    """Build a minimal DatasetItem-compatible payload."""
    return {
        "id": item_id,
        "input": {"prompt": prompt},
        "output": answer,
        "human_label": None,
        "metadata": {},
    }


def _args(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with sane defaults for cmd_dataset_diff."""
    defaults = {
        "dataset_a": None,
        "dataset_b": None,
        "format": "table",
        "show_items": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _resolve_dataset_file
# ---------------------------------------------------------------------------


class TestResolveDatasetFile:
    def test_direct_jsonl_file(self, tmp_path):
        f = _write_dataset(tmp_path / "dataset.jsonl", [_make_item("a")])
        assert _resolve_dataset_file(str(f)) == f

    def test_directory_with_dataset_jsonl(self, tmp_path):
        f = _write_dataset(tmp_path / "ds" / "dataset.jsonl", [_make_item("a")])
        assert _resolve_dataset_file(str(tmp_path / "ds")) == f

    def test_directory_with_single_other_jsonl(self, tmp_path):
        f = _write_dataset(tmp_path / "ds" / "custom.jsonl", [_make_item("a")])
        assert _resolve_dataset_file(str(tmp_path / "ds")) == f

    def test_missing_path_calls_fatal_error(self):
        with pytest.raises(SystemExit) as excinfo:
            _resolve_dataset_file("/this/path/does/not/exist/at/all")
        assert excinfo.value.code == 1

    def test_directory_with_no_dataset_calls_fatal_error(self, tmp_path):
        # Empty directory: no .jsonl, no .json, no dataset.jsonl. Must exit.
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit) as excinfo:
            _resolve_dataset_file(str(tmp_path / "empty"))
        assert excinfo.value.code == 1

    def test_directory_with_dataset_jsonl_wins_over_other_jsonl(self, tmp_path):
        # If both dataset.jsonl and another *.jsonl exist, dataset.jsonl wins.
        d = tmp_path / "ds"
        primary = _write_dataset(d / "dataset.jsonl", [_make_item("a")])
        _write_dataset(d / "other.jsonl", [_make_item("b")])
        assert _resolve_dataset_file(str(d)) == primary


# ---------------------------------------------------------------------------
# cmd_dataset_diff: identical / added / removed / mixed
# ---------------------------------------------------------------------------


class TestCmdDatasetDiffTable:
    def test_identical_datasets_show_zero_changes(self, tmp_path, capsys):
        items = [_make_item("a"), _make_item("b"), _make_item("c")]
        a = _write_dataset(tmp_path / "a.jsonl", items)
        b = _write_dataset(tmp_path / "b.jsonl", items)

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b)))
        out = capsys.readouterr().out

        assert "Added:     0" in out
        assert "Removed:   0" in out
        assert "Modified:  0" in out
        assert "Unchanged: 3" in out
        assert "identical" in out.lower()

    def test_added_only(self, tmp_path, capsys):
        a_items = [_make_item("a")]
        b_items = [_make_item("a"), _make_item("b"), _make_item("c")]
        a = _write_dataset(tmp_path / "a.jsonl", a_items)
        b = _write_dataset(tmp_path / "b.jsonl", b_items)

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b)))
        out = capsys.readouterr().out

        assert "Added:     2" in out
        assert "Removed:   0" in out
        assert "Modified:  0" in out
        assert "Unchanged: 1" in out

    def test_removed_only(self, tmp_path, capsys):
        a_items = [_make_item("a"), _make_item("b"), _make_item("c")]
        b_items = [_make_item("a")]
        a = _write_dataset(tmp_path / "a.jsonl", a_items)
        b = _write_dataset(tmp_path / "b.jsonl", b_items)

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b)))
        out = capsys.readouterr().out

        assert "Added:     0" in out
        assert "Removed:   2" in out
        assert "Modified:  0" in out
        assert "Unchanged: 1" in out

    def test_added_removed_and_modified(self, tmp_path, capsys):
        a_items = [
            _make_item("a", answer="old"),
            _make_item("b"),
            _make_item("c"),
        ]
        b_items = [
            _make_item("a", answer="new"),  # modified
            _make_item("b"),  # unchanged
            _make_item("d"),  # added (and c removed)
        ]
        a = _write_dataset(tmp_path / "a.jsonl", a_items)
        b = _write_dataset(tmp_path / "b.jsonl", b_items)

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b)))
        out = capsys.readouterr().out

        assert "Added:     1" in out
        assert "Removed:   1" in out
        assert "Modified:  1" in out
        assert "Unchanged: 1" in out

    def test_resolves_dataset_directory(self, tmp_path, capsys):
        # Pass directories rather than direct files. Must auto-find dataset.jsonl.
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        _write_dataset(a_dir / "dataset.jsonl", [_make_item("x")])
        _write_dataset(b_dir / "dataset.jsonl", [_make_item("x"), _make_item("y")])

        cmd_dataset_diff(_args(dataset_a=str(a_dir), dataset_b=str(b_dir)))
        out = capsys.readouterr().out

        assert "Added:     1" in out
        assert "Unchanged: 1" in out


# ---------------------------------------------------------------------------
# --format json
# ---------------------------------------------------------------------------


class TestCmdDatasetDiffJson:
    def test_json_output_is_valid(self, tmp_path, capsys):
        a = _write_dataset(tmp_path / "a.jsonl", [_make_item("a")])
        b = _write_dataset(
            tmp_path / "b.jsonl", [_make_item("a"), _make_item("b")]
        )

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b), format="json"))
        out = capsys.readouterr().out

        payload = json.loads(out)
        assert payload["summary"]["added"] == 1
        assert payload["summary"]["removed"] == 0
        assert payload["summary"]["modified"] == 0
        assert payload["summary"]["unchanged"] == 1
        assert payload["has_changes"] is True
        assert payload["a"].endswith("a.jsonl")
        assert payload["b"].endswith("b.jsonl")
        # show_items defaults to False so id lists should NOT be present.
        assert "added_ids" not in payload

    def test_json_with_show_items_includes_ids(self, tmp_path, capsys):
        a = _write_dataset(tmp_path / "a.jsonl", [_make_item("a")])
        b = _write_dataset(
            tmp_path / "b.jsonl", [_make_item("a"), _make_item("b")]
        )

        cmd_dataset_diff(
            _args(dataset_a=str(a), dataset_b=str(b), format="json", show_items=True)
        )
        out = capsys.readouterr().out

        payload = json.loads(out)
        assert payload["added_ids"] == ["b"]
        assert payload["removed_ids"] == []
        assert payload["modified_ids"] == []
        assert payload["truncated"] == {"added": 0, "removed": 0, "modified": 0}

    def test_json_identical_has_changes_false(self, tmp_path, capsys):
        items = [_make_item("a"), _make_item("b")]
        a = _write_dataset(tmp_path / "a.jsonl", items)
        b = _write_dataset(tmp_path / "b.jsonl", items)

        cmd_dataset_diff(_args(dataset_a=str(a), dataset_b=str(b), format="json"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["has_changes"] is False
        assert payload["summary"]["unchanged"] == 2


# ---------------------------------------------------------------------------
# --show-items truncation
# ---------------------------------------------------------------------------


class TestShowItemsLimit:
    def test_show_items_caps_at_limit_in_table(self, tmp_path, capsys):
        # B has _SHOW_ITEMS_LIMIT + 5 new items; output must list only first 10
        # and mention the remaining 5 as "... and N more".
        extra = _SHOW_ITEMS_LIMIT + 5
        a = _write_dataset(tmp_path / "a.jsonl", [_make_item("seed")])
        b_items = [_make_item("seed")] + [_make_item(f"new-{i:03d}") for i in range(extra)]
        b = _write_dataset(tmp_path / "b.jsonl", b_items)

        cmd_dataset_diff(
            _args(dataset_a=str(a), dataset_b=str(b), show_items=True)
        )
        out = capsys.readouterr().out

        # Header reports the true total.
        assert f"Added:     {extra}" in out

        # Body lists only the first _SHOW_ITEMS_LIMIT IDs (alphabetical sort
        # from the underlying diff means new-000 .. new-009).
        for i in range(_SHOW_ITEMS_LIMIT):
            assert f"new-{i:03d}" in out
        # The 11th item must NOT be in the body.
        assert f"new-{_SHOW_ITEMS_LIMIT:03d}" not in out
        # And the truncation tail must be present.
        assert f"... and {extra - _SHOW_ITEMS_LIMIT} more" in out

    def test_show_items_caps_at_limit_in_json(self, tmp_path, capsys):
        extra = _SHOW_ITEMS_LIMIT + 3
        a = _write_dataset(tmp_path / "a.jsonl", [_make_item("seed")])
        b_items = [_make_item("seed")] + [_make_item(f"new-{i:03d}") for i in range(extra)]
        b = _write_dataset(tmp_path / "b.jsonl", b_items)

        cmd_dataset_diff(
            _args(
                dataset_a=str(a),
                dataset_b=str(b),
                format="json",
                show_items=True,
            )
        )
        payload = json.loads(capsys.readouterr().out)

        assert len(payload["added_ids"]) == _SHOW_ITEMS_LIMIT
        assert payload["truncated"]["added"] == 3
        assert payload["summary"]["added"] == extra


# ---------------------------------------------------------------------------
# Renderer unit tests (pure functions, no IO)
# ---------------------------------------------------------------------------


class TestRenderers:
    def _diff(self, items_a: list, items_b: list):
        from evalyn_sdk.datasets.merge import diff_datasets
        from evalyn_sdk.models import DatasetItem

        a = [DatasetItem.from_payload(it) for it in items_a]
        b = [DatasetItem.from_payload(it) for it in items_b]
        return diff_datasets(a, b)

    def test_render_table_no_changes_mentions_identical(self):
        diff = self._diff([_make_item("a")], [_make_item("a")])
        out = _render_table(diff, label_a="A", label_b="B", show_items=False)
        assert "Unchanged: 1" in out
        assert "identical" in out.lower()

    def test_render_table_labels_appear(self):
        diff = self._diff([_make_item("a")], [_make_item("a")])
        out = _render_table(
            diff,
            label_a="path/to/a.jsonl",
            label_b="path/to/b.jsonl",
            show_items=False,
        )
        assert "path/to/a.jsonl" in out
        assert "path/to/b.jsonl" in out

    def test_render_json_summary_matches(self):
        diff = self._diff([_make_item("a")], [_make_item("a"), _make_item("b")])
        out = _render_json(diff, label_a="A", label_b="B", show_items=False)
        payload = json.loads(out)
        assert payload["summary"] == {
            "added": 1,
            "removed": 0,
            "modified": 0,
            "unchanged": 1,
        }


# ---------------------------------------------------------------------------
# register_commands wiring
# ---------------------------------------------------------------------------


class TestRegisterCommands:
    def test_register_attaches_func(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        register_commands(subparsers)

        args = parser.parse_args(["dataset-diff", "a.jsonl", "b.jsonl"])
        assert args.func is dataset_diff_mod.cmd_dataset_diff
        assert args.dataset_a == "a.jsonl"
        assert args.dataset_b == "b.jsonl"
        assert args.format == "table"
        assert args.show_items is False

    def test_register_accepts_flags(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        register_commands(subparsers)

        args = parser.parse_args(
            ["dataset-diff", "a", "b", "--format", "json", "--show-items"]
        )
        assert args.format == "json"
        assert args.show_items is True
