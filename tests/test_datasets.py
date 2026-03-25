"""
Tests for evalyn_sdk.datasets: stream_dataset, load_dataset, save_dataset.

Run with: pytest tests/test_datasets.py -v
"""
from __future__ import annotations

import json
import logging

import pytest

from evalyn_sdk.datasets import load_dataset, stream_dataset, save_dataset
from evalyn_sdk.models import DatasetItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, rows):
    """Write rows as JSONL."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json_array(path, rows):
    """Write rows as a JSON array."""
    with open(path, "w") as f:
        json.dump(rows, f)


def _sample_row(idx=0):
    return {"id": f"item-{idx}", "input": {"q": f"question {idx}"}, "output": f"answer {idx}"}


# ---------------------------------------------------------------------------
# stream_dataset
# ---------------------------------------------------------------------------

class TestStreamDataset:
    def test_stream_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_sample_row(i) for i in range(5)])
        items = list(stream_dataset(p))
        assert len(items) == 5
        assert items[0].id == "item-0"
        assert items[4].input == {"q": "question 4"}

    def test_stream_json_array(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json_array(p, [_sample_row(i) for i in range(3)])
        items = list(stream_dataset(p))
        assert len(items) == 3
        assert items[2].id == "item-2"

    def test_stream_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        items = list(stream_dataset(p))
        assert items == []

    def test_stream_whitespace_only_file(self, tmp_path):
        p = tmp_path / "ws.jsonl"
        p.write_text("  \n  \n  \n")
        items = list(stream_dataset(p))
        assert items == []

    def test_stream_skips_malformed_lines(self, tmp_path, caplog):
        p = tmp_path / "bad.jsonl"
        lines = [
            json.dumps(_sample_row(0)),
            "this is not valid json",
            json.dumps(_sample_row(2)),
            "{broken: json",
        ]
        p.write_text("\n".join(lines) + "\n")

        with caplog.at_level(logging.WARNING, logger="evalyn_sdk.datasets"):
            items = list(stream_dataset(p))

        assert len(items) == 2
        assert items[0].id == "item-0"
        assert items[1].id == "item-2"
        assert len(caplog.records) == 2

    def test_stream_skips_blank_lines_in_jsonl(self, tmp_path):
        p = tmp_path / "gaps.jsonl"
        content = json.dumps(_sample_row(0)) + "\n\n\n" + json.dumps(_sample_row(1)) + "\n"
        p.write_text(content)
        items = list(stream_dataset(p))
        assert len(items) == 2

    def test_stream_is_lazy_generator(self, tmp_path):
        """stream_dataset returns a generator, not a list."""
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_sample_row(0)])
        result = stream_dataset(p)
        import types
        assert isinstance(result, types.GeneratorType)

    def test_stream_old_format_inputs_expected(self, tmp_path):
        """Old format with 'inputs' and 'expected' still works."""
        p = tmp_path / "old.jsonl"
        row = {"id": "old-1", "inputs": {"q": "hello"}, "expected": "world"}
        _write_jsonl(p, [row])
        items = list(stream_dataset(p))
        assert len(items) == 1
        assert items[0].input == {"q": "hello"}
        assert items[0].output == "world"

    def test_stream_empty_json_array(self, tmp_path):
        p = tmp_path / "empty_arr.json"
        p.write_text("[]")
        items = list(stream_dataset(p))
        assert items == []


# ---------------------------------------------------------------------------
# load_dataset (wraps stream_dataset)
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_returns_list(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_sample_row(i) for i in range(3)])
        result = load_dataset(p)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_load_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert load_dataset(p) == []


# ---------------------------------------------------------------------------
# save_dataset roundtrip
# ---------------------------------------------------------------------------

class TestSaveDatasetRoundtrip:
    def test_save_then_load_roundtrip(self, tmp_path):
        items = [
            DatasetItem(id="a", input={"q": "hi"}, output="hey"),
            DatasetItem(id="b", input={"q": "bye"}, output="later"),
        ]
        p = tmp_path / "out.jsonl"
        save_dataset(items, p)
        loaded = load_dataset(p)
        assert len(loaded) == 2
        assert loaded[0].id == "a"
        assert loaded[1].output == "later"
