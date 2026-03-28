"""Tests for evalyn_sdk.datasets_snapshot_comparison."""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_snapshot_comparison import (
    ItemDiff,
    SnapshotComparison,
    compare_snapshots,
    compute_change_rate,
    compute_item_diff,
    filter_diffs,
    render_diff_summary,
)


# ---------------------------------------------------------------------------
# compare_snapshots
# ---------------------------------------------------------------------------


class TestCompareSnapshots:
    def test_identical_items(self):
        items = [
            {"id": "1", "input": "hello", "output": "world"},
            {"id": "2", "input": "foo", "output": "bar"},
        ]
        result = compare_snapshots(items, items)
        assert result.added_count == 0
        assert result.removed_count == 0
        assert result.modified_count == 0
        assert result.unchanged_count == 2

    def test_with_adds(self):
        items_a = [{"id": "1", "input": "hello", "output": "world"}]
        items_b = [
            {"id": "1", "input": "hello", "output": "world"},
            {"id": "2", "input": "foo", "output": "bar"},
        ]
        result = compare_snapshots(items_a, items_b)
        assert result.added_count == 1
        assert result.unchanged_count == 1

    def test_with_removes(self):
        items_a = [
            {"id": "1", "input": "hello", "output": "world"},
            {"id": "2", "input": "foo", "output": "bar"},
        ]
        items_b = [{"id": "1", "input": "hello", "output": "world"}]
        result = compare_snapshots(items_a, items_b)
        assert result.removed_count == 1
        assert result.unchanged_count == 1

    def test_with_modifications(self):
        items_a = [{"id": "1", "input": "hello", "output": "world"}]
        items_b = [{"id": "1", "input": "hello changed", "output": "world"}]
        result = compare_snapshots(items_a, items_b)
        assert result.modified_count == 1
        assert result.unchanged_count == 0

    def test_mixed_changes(self):
        items_a = [
            {"id": "1", "input": "same", "output": "same"},
            {"id": "2", "input": "old", "output": "old"},
            {"id": "3", "input": "gone", "output": "gone"},
        ]
        items_b = [
            {"id": "1", "input": "same", "output": "same"},
            {"id": "2", "input": "new", "output": "old"},
            {"id": "4", "input": "fresh", "output": "fresh"},
        ]
        result = compare_snapshots(items_a, items_b)
        assert result.unchanged_count == 1
        assert result.modified_count == 1
        assert result.removed_count == 1
        assert result.added_count == 1

    def test_custom_versions(self):
        result = compare_snapshots([], [], version_a="baseline", version_b="candidate")
        assert result.version_a == "baseline"
        assert result.version_b == "candidate"

    def test_empty_both_lists(self):
        result = compare_snapshots([], [])
        assert result.added_count == 0
        assert result.removed_count == 0
        assert result.modified_count == 0
        assert result.unchanged_count == 0
        assert result.diffs == []

    def test_empty_a_nonempty_b(self):
        items_b = [{"id": "1", "input": "hello", "output": "world"}]
        result = compare_snapshots([], items_b)
        assert result.added_count == 1
        assert result.removed_count == 0

    def test_nonempty_a_empty_b(self):
        items_a = [{"id": "1", "input": "hello", "output": "world"}]
        result = compare_snapshots(items_a, [])
        assert result.removed_count == 1
        assert result.added_count == 0

    def test_output_modification_detected(self):
        items_a = [{"id": "1", "input": "same", "output": "old_answer"}]
        items_b = [{"id": "1", "input": "same", "output": "new_answer"}]
        result = compare_snapshots(items_a, items_b)
        assert result.modified_count == 1


# ---------------------------------------------------------------------------
# compute_item_diff
# ---------------------------------------------------------------------------


class TestComputeItemDiff:
    def test_generates_diff_text(self):
        old = {"id": "1", "input": "hello world", "output": "answer"}
        new = {"id": "1", "input": "hello universe", "output": "answer"}
        diff = compute_item_diff(old, new)
        assert diff.change_type == "modified"
        assert diff.item_id == "1"
        assert "hello world" in diff.diff_text or "hello universe" in diff.diff_text
        assert diff.diff_text != ""

    def test_unchanged_item(self):
        item = {"id": "1", "input": "hello", "output": "world"}
        diff = compute_item_diff(item, item)
        assert diff.change_type == "unchanged"
        assert diff.diff_text == ""

    def test_output_only_change(self):
        old = {"id": "1", "input": "same", "output": "old"}
        new = {"id": "1", "input": "same", "output": "new"}
        diff = compute_item_diff(old, new)
        assert diff.change_type == "modified"
        assert diff.old_output == "old"
        assert diff.new_output == "new"

    def test_multiline_diff(self):
        old = {"id": "1", "input": "line1\nline2\nline3", "output": ""}
        new = {"id": "1", "input": "line1\nchanged\nline3", "output": ""}
        diff = compute_item_diff(old, new)
        assert diff.change_type == "modified"
        assert diff.diff_text != ""


# ---------------------------------------------------------------------------
# filter_diffs
# ---------------------------------------------------------------------------


class TestFilterDiffs:
    def test_filter_by_added(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[
                ItemDiff(item_id="1", change_type="added"),
                ItemDiff(item_id="2", change_type="removed"),
                ItemDiff(item_id="3", change_type="added"),
            ],
        )
        result = filter_diffs(comp, "added")
        assert len(result) == 2
        assert all(d.change_type == "added" for d in result)

    def test_filter_by_removed(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[
                ItemDiff(item_id="1", change_type="removed"),
                ItemDiff(item_id="2", change_type="unchanged"),
            ],
        )
        result = filter_diffs(comp, "removed")
        assert len(result) == 1
        assert result[0].item_id == "1"

    def test_filter_returns_empty_on_no_match(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[ItemDiff(item_id="1", change_type="unchanged")],
        )
        result = filter_diffs(comp, "modified")
        assert result == []

    def test_filter_unchanged(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[
                ItemDiff(item_id="1", change_type="unchanged"),
                ItemDiff(item_id="2", change_type="modified"),
            ],
        )
        result = filter_diffs(comp, "unchanged")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# render_diff_summary
# ---------------------------------------------------------------------------


class TestRenderDiffSummary:
    def test_output_contains_counts(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=2,
            removed_count=1,
            modified_count=3,
            unchanged_count=4,
        )
        text = render_diff_summary(comp)
        assert "v1" in text
        assert "v2" in text
        assert "Added" in text
        assert "Removed" in text
        assert "Modified" in text
        assert "Unchanged" in text
        assert "2" in text
        assert "1" in text
        assert "3" in text
        assert "4" in text

    def test_change_rate_in_summary(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=1,
            removed_count=0,
            modified_count=0,
            unchanged_count=1,
        )
        text = render_diff_summary(comp)
        assert "Change rate" in text
        assert "50.0%" in text


# ---------------------------------------------------------------------------
# compute_change_rate
# ---------------------------------------------------------------------------


class TestComputeChangeRate:
    def test_all_changed(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=2,
            removed_count=1,
            modified_count=1,
            unchanged_count=0,
        )
        assert compute_change_rate(comp) == pytest.approx(1.0)

    def test_none_changed(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=0,
            removed_count=0,
            modified_count=0,
            unchanged_count=5,
        )
        assert compute_change_rate(comp) == pytest.approx(0.0)

    def test_partial_change(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=1,
            removed_count=0,
            modified_count=1,
            unchanged_count=2,
        )
        assert compute_change_rate(comp) == pytest.approx(0.5)

    def test_empty_comparison(self):
        comp = SnapshotComparison(version_a="v1", version_b="v2")
        assert compute_change_rate(comp) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SnapshotComparison format_text
# ---------------------------------------------------------------------------


class TestSnapshotComparisonFormatText:
    def test_format_text_contains_header(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            added_count=1,
            removed_count=0,
            modified_count=0,
            unchanged_count=1,
        )
        text = comp.format_text()
        assert "Snapshot Comparison" in text
        assert "v1" in text
        assert "v2" in text

    def test_format_text_shows_changes(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[
                ItemDiff(item_id="item-1", change_type="added"),
                ItemDiff(item_id="item-2", change_type="removed"),
            ],
            added_count=1,
            removed_count=1,
        )
        text = comp.format_text()
        assert "ADDED" in text
        assert "REMOVED" in text
        assert "item-1" in text
        assert "item-2" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_item_diff_as_dict(self):
        diff = ItemDiff(
            item_id="1",
            change_type="modified",
            old_input="old",
            new_input="new",
            old_output="a",
            new_output="b",
            diff_text="some diff",
        )
        d = diff.as_dict()
        assert d["item_id"] == "1"
        assert d["change_type"] == "modified"
        assert d["old_input"] == "old"
        assert d["new_input"] == "new"
        assert d["diff_text"] == "some diff"

    def test_snapshot_comparison_roundtrip(self):
        comp = SnapshotComparison(
            version_a="v1",
            version_b="v2",
            diffs=[
                ItemDiff(item_id="1", change_type="added", new_input="hello"),
            ],
            added_count=1,
            removed_count=0,
            modified_count=0,
            unchanged_count=0,
        )
        d = comp.as_dict()
        restored = SnapshotComparison.from_dict(d)
        assert restored.version_a == "v1"
        assert restored.version_b == "v2"
        assert restored.added_count == 1
        assert len(restored.diffs) == 1
        assert restored.diffs[0].item_id == "1"
        assert restored.diffs[0].change_type == "added"

    def test_from_dict_empty(self):
        comp = SnapshotComparison.from_dict({})
        assert comp.version_a == ""
        assert comp.version_b == ""
        assert comp.diffs == []
        assert comp.added_count == 0

    def test_from_dict_preserves_all_fields(self):
        d = {
            "version_a": "base",
            "version_b": "head",
            "diffs": [
                {
                    "item_id": "x",
                    "change_type": "removed",
                    "old_input": "hi",
                    "new_input": "",
                    "old_output": "bye",
                    "new_output": "",
                    "diff_text": "",
                }
            ],
            "added_count": 0,
            "removed_count": 1,
            "modified_count": 0,
            "unchanged_count": 0,
        }
        comp = SnapshotComparison.from_dict(d)
        assert comp.version_a == "base"
        assert comp.removed_count == 1
        assert comp.diffs[0].old_input == "hi"
