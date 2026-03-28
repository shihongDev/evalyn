"""Tests for differential evaluation."""

import pytest
from evalyn_sdk.evaluation.differential import (
    ItemChange,
    DiffResult,
    hash_item,
    compute_diff,
    carry_forward_results,
    build_differential_plan,
)


# --- hash_item ---


class TestHashItem:
    def test_deterministic(self):
        h1 = hash_item("id1", "hello", "world")
        h2 = hash_item("id1", "hello", "world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = hash_item("id1", "hello", "world")
        h2 = hash_item("id1", "hello", "other")
        assert h1 != h2

    def test_different_id_different_hash(self):
        h1 = hash_item("id1", "hello")
        h2 = hash_item("id2", "hello")
        assert h1 != h2

    def test_empty_output(self):
        h = hash_item("id1", "input")
        assert len(h) == 64  # SHA-256 hex digest length

    def test_returns_hex_string(self):
        h = hash_item("x", "y", "z")
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_strings(self):
        h = hash_item("", "", "")
        assert len(h) == 64


# --- compute_diff ---


class TestComputeDiff:
    def test_all_new_items(self):
        old: dict = {}
        new = {"a": "h1", "b": "h2"}
        diff = compute_diff(old, new)
        assert diff.added_count == 2
        assert diff.removed_count == 0
        assert diff.modified_count == 0
        assert diff.unchanged_count == 0
        assert diff.total_items == 2

    def test_all_removed(self):
        old = {"a": "h1", "b": "h2"}
        new: dict = {}
        diff = compute_diff(old, new)
        assert diff.removed_count == 2
        assert diff.added_count == 0
        assert diff.total_items == 0

    def test_no_changes(self):
        items = {"a": "h1", "b": "h2", "c": "h3"}
        diff = compute_diff(items, items)
        assert diff.unchanged_count == 3
        assert diff.added_count == 0
        assert diff.removed_count == 0
        assert diff.modified_count == 0

    def test_mixed_changes(self):
        old = {"a": "h1", "b": "h2", "c": "h3"}
        new = {"a": "h1", "b": "h2_changed", "d": "h4"}
        diff = compute_diff(old, new)
        assert diff.unchanged_count == 1  # a
        assert diff.modified_count == 1  # b
        assert diff.removed_count == 1  # c
        assert diff.added_count == 1  # d
        assert diff.total_items == 3  # new has 3 items

    def test_empty_dicts(self):
        diff = compute_diff({}, {})
        assert diff.added_count == 0
        assert diff.removed_count == 0
        assert diff.modified_count == 0
        assert diff.unchanged_count == 0
        assert diff.total_items == 0
        assert diff.changes == []

    def test_changes_sorted_by_id(self):
        old = {"c": "1", "a": "2"}
        new = {"b": "3", "a": "2"}
        diff = compute_diff(old, new)
        ids = [c.item_id for c in diff.changes]
        assert ids == sorted(ids)


# --- items_to_evaluate property ---


class TestItemsToEvaluate:
    def test_includes_added_and_modified(self):
        changes = [
            ItemChange("a", "added", new_hash="h1"),
            ItemChange("b", "modified", old_hash="h2", new_hash="h3"),
            ItemChange("c", "unchanged", old_hash="h4", new_hash="h4"),
            ItemChange("d", "removed", old_hash="h5"),
        ]
        diff = DiffResult(
            changes=changes,
            added_count=1,
            removed_count=1,
            modified_count=1,
            unchanged_count=1,
            total_items=3,
        )
        to_eval = diff.items_to_evaluate
        assert "a" in to_eval
        assert "b" in to_eval
        assert "c" not in to_eval
        assert "d" not in to_eval

    def test_empty_diff(self):
        diff = DiffResult()
        assert diff.items_to_evaluate == []


# --- carry_forward_results ---


class TestCarryForwardResults:
    def test_filters_unchanged_only(self):
        changes = [
            ItemChange("a", "unchanged", old_hash="h1", new_hash="h1"),
            ItemChange("b", "modified", old_hash="h2", new_hash="h3"),
            ItemChange("c", "added", new_hash="h4"),
        ]
        diff = DiffResult(
            changes=changes,
            added_count=1,
            removed_count=0,
            modified_count=1,
            unchanged_count=1,
            total_items=3,
        )
        previous = [
            {"item_id": "a", "score": 0.9},
            {"item_id": "b", "score": 0.5},
        ]
        carried = carry_forward_results(previous, diff)
        assert len(carried) == 1
        assert carried[0]["item_id"] == "a"

    def test_no_unchanged(self):
        diff = DiffResult(
            changes=[ItemChange("a", "modified", old_hash="h1", new_hash="h2")],
            modified_count=1,
            total_items=1,
        )
        previous = [{"item_id": "a", "score": 0.9}]
        carried = carry_forward_results(previous, diff)
        assert carried == []

    def test_empty_previous(self):
        diff = DiffResult(
            changes=[ItemChange("a", "unchanged", old_hash="h1", new_hash="h1")],
            unchanged_count=1,
            total_items=1,
        )
        carried = carry_forward_results([], diff)
        assert carried == []

    def test_preserves_result_data(self):
        diff = DiffResult(
            changes=[ItemChange("x", "unchanged", old_hash="h", new_hash="h")],
            unchanged_count=1,
            total_items=1,
        )
        previous = [{"item_id": "x", "score": 0.75, "details": {"key": "val"}}]
        carried = carry_forward_results(previous, diff)
        assert carried[0]["score"] == 0.75
        assert carried[0]["details"] == {"key": "val"}


# --- build_differential_plan ---


class TestBuildDifferentialPlan:
    def test_delegates_to_compute_diff(self):
        old = {"a": "h1", "b": "h2"}
        new = {"a": "h1", "c": "h3"}
        plan = build_differential_plan(old, new)
        assert plan.unchanged_count == 1
        assert plan.removed_count == 1
        assert plan.added_count == 1

    def test_empty(self):
        plan = build_differential_plan({}, {})
        assert plan.total_items == 0


# --- DiffResult serialization ---


class TestDiffResultSerialization:
    def test_format_text(self):
        diff = DiffResult(
            changes=[],
            added_count=3,
            removed_count=1,
            modified_count=2,
            unchanged_count=10,
            total_items=15,
        )
        text = diff.format_text()
        assert "Differential Evaluation Plan" in text
        assert "Added: 3" in text
        assert "Removed: 1" in text
        assert "Modified: 2" in text
        assert "Unchanged: 10" in text
        assert "Total items: 15" in text

    def test_as_dict_from_dict_roundtrip(self):
        change = ItemChange("item1", "modified", old_hash="aaa", new_hash="bbb")
        original = DiffResult(
            changes=[change],
            added_count=0,
            removed_count=0,
            modified_count=1,
            unchanged_count=0,
            total_items=1,
        )
        data = original.as_dict()
        restored = DiffResult.from_dict(data)
        assert restored.modified_count == 1
        assert restored.total_items == 1
        assert len(restored.changes) == 1
        assert restored.changes[0].item_id == "item1"
        assert restored.changes[0].change_type == "modified"
        assert restored.changes[0].old_hash == "aaa"
        assert restored.changes[0].new_hash == "bbb"

    def test_from_dict_defaults(self):
        data: dict = {}
        result = DiffResult.from_dict(data)
        assert result.changes == []
        assert result.added_count == 0
        assert result.total_items == 0

    def test_items_to_evaluate_in_format_text(self):
        diff = DiffResult(
            changes=[
                ItemChange("a", "added", new_hash="h1"),
                ItemChange("b", "modified", old_hash="h2", new_hash="h3"),
            ],
            added_count=1,
            modified_count=1,
            total_items=2,
        )
        text = diff.format_text()
        assert "Items to evaluate: 2" in text
