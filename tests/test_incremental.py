"""Tests for incremental dataset build."""

import json
import pytest
from pathlib import Path
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets_incremental import (
    incremental_build,
    load_build_state,
    save_build_state,
    IncrementalBuildResult,
)

# =============================================================================
# Incremental dataset build tests
# =============================================================================

def _item(item_id: str, input_text: str = "test") -> DatasetItem:
    return DatasetItem(id=item_id, input={"query": input_text}, output=f"answer-{item_id}")


class TestIncrementalBuild:
    def test_append_new_items(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello"), _item("2", "world")]
        new = [_item("3", "foo"), _item("4", "bar")]

        # Write existing dataset
        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 2
        assert result.duplicates_skipped == 0
        assert result.total_items == 4

    def test_deduplicate(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello"), _item("2", "world")]
        new = [_item("3", "hello"), _item("4", "new")]  # "hello" is duplicate

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 1  # only "new" added
        assert result.duplicates_skipped == 1

    def test_all_duplicates(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello")]
        new = [_item("2", "hello")]  # same input

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 0
        assert result.duplicates_skipped == 1
        assert result.total_items == 1

    def test_empty_new_items(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1")]

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, [], dataset_path)
        assert result.new_items_added == 0
        assert result.total_items == 1

    def test_empty_existing(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        new = [_item("1", "hello"), _item("2", "world")]

        # Create empty file
        dataset_path.write_text("")

        result = incremental_build([], new, dataset_path)
        assert result.new_items_added == 2
        assert result.total_items == 2

    def test_build_state_saved(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        from evalyn_sdk.datasets import save_dataset
        save_dataset([_item("1")], dataset_path)

        incremental_build([_item("1")], [_item("2", "new")], dataset_path)
        state = load_build_state(tmp_path)
        assert state is not None

    def test_result_as_dict(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        from evalyn_sdk.datasets import save_dataset
        save_dataset([], dataset_path)

        result = incremental_build([], [_item("1")], dataset_path)
        d = result.as_dict()
        assert "new_items_added" in d
        assert "total_items" in d
        assert "last_build_timestamp" in d


class TestBuildState:
    def test_save_and_load(self, tmp_path):
        save_build_state(tmp_path, "2026-03-28T12:00:00Z")
        state = load_build_state(tmp_path)
        assert state == "2026-03-28T12:00:00Z"

    def test_load_missing(self, tmp_path):
        assert load_build_state(tmp_path) is None

    def test_load_corrupt(self, tmp_path):
        (tmp_path / "meta.json").write_text("not json{{{")
        assert load_build_state(tmp_path) is None

    def test_preserves_existing_meta(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"project": "test", "version": "v1"}))
        save_build_state(tmp_path, "2026-03-28T12:00:00Z")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["project"] == "test"
        assert meta["last_incremental_build"] == "2026-03-28T12:00:00Z"
