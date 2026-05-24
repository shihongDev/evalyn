"""Tests for `evalyn dataset-stats`."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evalyn_sdk.cli.commands import dataset_stats
from evalyn_sdk.models import DatasetItem


def _make_args(dataset: str, **overrides) -> argparse.Namespace:
    ns = argparse.Namespace(dataset=dataset, format="table")
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _make_items(n: int = 5, with_labels: int = 2) -> list[DatasetItem]:
    items = []
    for i in range(n):
        items.append(
            DatasetItem(
                id=f"item-{i}",
                input={"text": "hello world " * (i + 1)},
                output="response text " * (i + 1),
                human_label={"score": 1.0} if i < with_labels else None,
                metadata={"source": "test", "batch": str(i % 2)},
            )
        )
    return items


# ----------------------------------------------------------------------
# compute_stats
# ----------------------------------------------------------------------


def test_compute_stats_basic():
    items = _make_items(n=5, with_labels=2)
    stats = dataset_stats.compute_stats(items)
    assert stats["item_count"] == 5
    assert stats["labeled_count"] == 2
    assert stats["labeled_pct"] == pytest.approx(40.0)
    assert stats["input_length"]["min"] > 0
    assert stats["input_length"]["max"] >= stats["input_length"]["min"]


def test_compute_stats_empty():
    stats = dataset_stats.compute_stats([])
    assert stats["item_count"] == 0
    assert stats["labeled_count"] == 0
    assert stats["labeled_pct"] == 0.0
    assert stats["input_length"]["min"] == 0
    assert stats["input_length"]["max"] == 0


def test_compute_stats_all_labeled():
    items = _make_items(n=3, with_labels=3)
    stats = dataset_stats.compute_stats(items)
    assert stats["labeled_count"] == 3
    assert stats["labeled_pct"] == pytest.approx(100.0)


def test_compute_stats_metadata_frequencies():
    items = _make_items(n=4)
    stats = dataset_stats.compute_stats(items)
    freq = stats["metadata_key_freq"]
    assert freq["source"] == 4
    assert freq["batch"] == 4


def test_length_stats_p95_calculation():
    lengths = list(range(1, 101))  # 1 to 100
    stats = dataset_stats._length_stats(lengths)
    assert stats["min"] == 1
    assert stats["max"] == 100
    assert stats["mean"] == 50.5
    assert stats["median"] == 50.5
    assert stats["p95"] == 95


def test_length_stats_p95_small_n():
    # Regression for review finding: previous off-by-one returned p67 for
    # n=3. Nearest-rank should give the max for tiny samples.
    assert dataset_stats._length_stats([10])["p95"] == 10
    assert dataset_stats._length_stats([10, 20])["p95"] == 20
    assert dataset_stats._length_stats([10, 20, 30])["p95"] == 30
    assert dataset_stats._length_stats([1, 2, 3, 4, 5])["p95"] == 5


# ----------------------------------------------------------------------
# CLI behavior
# ----------------------------------------------------------------------


def _write_dataset_jsonl(tmp_path: Path, items: list[DatasetItem]) -> Path:
    """Write items as JSONL to a temp file."""
    p = tmp_path / "dataset.jsonl"
    with p.open("w") as f:
        for item in items:
            f.write(json.dumps(item.as_dict()) + "\n")
    return p


def test_cli_table_output(tmp_path):
    items = _make_items(n=3)
    path = _write_dataset_jsonl(tmp_path, items)

    args = _make_args(str(path))
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        dataset_stats.cmd_dataset_stats(args)
    output = buf.getvalue()
    assert "Items:" in output
    assert "3" in output
    assert "Input length" in output
    assert "Output length" in output


def test_cli_json_output(tmp_path):
    items = _make_items(n=3, with_labels=1)
    path = _write_dataset_jsonl(tmp_path, items)

    args = _make_args(str(path), format="json")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        dataset_stats.cmd_dataset_stats(args)
    payload = json.loads(buf.getvalue())
    assert payload["item_count"] == 3
    assert payload["labeled_count"] == 1
    assert "input_length" in payload
    assert "output_length" in payload


def test_cli_missing_dataset_exits(tmp_path):
    args = _make_args(str(tmp_path / "nonexistent.jsonl"))
    buf = io.StringIO()
    with patch("sys.stdout", new=buf), pytest.raises(SystemExit) as exc:
        dataset_stats.cmd_dataset_stats(args)
    assert exc.value.code == 2


def test_cli_empty_dataset(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    args = _make_args(str(p))
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        dataset_stats.cmd_dataset_stats(args)
    assert "empty" in buf.getvalue().lower()


# ----------------------------------------------------------------------
# Argparse wiring
# ----------------------------------------------------------------------


def test_register_commands_wires_subparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    dataset_stats.register_commands(subparsers)

    args = parser.parse_args(["dataset-stats", "data/foo.jsonl", "--format", "json"])
    assert args.dataset == "data/foo.jsonl"
    assert args.format == "json"
    assert hasattr(args, "func")
