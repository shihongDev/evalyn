"""Tests for eval_diff module."""

from __future__ import annotations

from evalyn_sdk.eval_diff import (
    ItemDiff,
    RunDiff,
    TextDiff,
    compute_score_diff,
    compute_text_diff,
    filter_diffs,
    format_score_diff,
    format_text_diff,
    summarize_diff,
)


class TestItemDiff:
    def test_as_dict(self):
        d = ItemDiff("i1", "acc", 0.8, 0.9, 0.1, "improved").as_dict()
        assert d["item_id"] == "i1"
        assert d["delta"] == 0.1

    def test_from_dict_roundtrip(self):
        orig = ItemDiff("i1", "acc", 0.8, 0.9, 0.1, "improved")
        restored = ItemDiff.from_dict(orig.as_dict())
        assert restored.direction == "improved"


class TestRunDiff:
    def test_as_dict(self):
        rd = RunDiff("a", "b", [], 0, 0, 0, 0)
        d = rd.as_dict()
        assert d["run_a_id"] == "a"

    def test_from_dict_roundtrip(self):
        diff = ItemDiff("i1", "m", 0.5, 0.6, 0.1, "improved")
        rd = RunDiff("a", "b", [diff], 1, 0, 0, 1)
        restored = RunDiff.from_dict(rd.as_dict())
        assert len(restored.item_diffs) == 1


class TestTextDiff:
    def test_as_dict(self):
        td = TextDiff("i1", "output", "hello world", "hello there", ["there"], ["world"])
        d = td.as_dict()
        assert d["added_words"] == ["there"]

    def test_from_dict_roundtrip(self):
        td = TextDiff("i1", "out", "a", "b", ["b"], ["a"])
        restored = TextDiff.from_dict(td.as_dict())
        assert restored.removed_words == ["a"]


class TestComputeScoreDiff:
    def test_basic(self):
        a = {"i1": {"acc": 0.8}}
        b = {"i1": {"acc": 0.9}}
        rd = compute_score_diff(a, b)
        assert rd.improved_count == 1
        assert abs(rd.item_diffs[0].delta - 0.1) < 1e-9

    def test_regression(self):
        a = {"i1": {"acc": 0.9}}
        b = {"i1": {"acc": 0.7}}
        rd = compute_score_diff(a, b)
        assert rd.regressed_count == 1

    def test_unchanged(self):
        a = {"i1": {"acc": 0.8}}
        b = {"i1": {"acc": 0.805}}
        rd = compute_score_diff(a, b, threshold=0.01)
        assert rd.unchanged_count == 1

    def test_multiple_metrics(self):
        a = {"i1": {"acc": 0.8, "lat": 0.5}}
        b = {"i1": {"acc": 0.9, "lat": 0.3}}
        rd = compute_score_diff(a, b)
        assert len(rd.item_diffs) == 2

    def test_multiple_items(self):
        a = {"i1": {"acc": 0.8}, "i2": {"acc": 0.6}}
        b = {"i1": {"acc": 0.9}, "i2": {"acc": 0.7}}
        rd = compute_score_diff(a, b)
        assert rd.total_items == 2
        assert rd.improved_count == 2

    def test_missing_in_b(self):
        a = {"i1": {"acc": 0.8}}
        b = {}
        rd = compute_score_diff(a, b)
        assert len(rd.item_diffs) == 1
        assert rd.item_diffs[0].score_b == 0.0

    def test_new_in_b(self):
        a = {}
        b = {"i1": {"acc": 0.9}}
        rd = compute_score_diff(a, b)
        assert rd.improved_count == 1

    def test_empty(self):
        rd = compute_score_diff({}, {})
        assert rd.total_items == 0
        assert len(rd.item_diffs) == 0

    def test_custom_threshold(self):
        a = {"i1": {"acc": 0.80}}
        b = {"i1": {"acc": 0.82}}
        rd = compute_score_diff(a, b, threshold=0.05)
        assert rd.unchanged_count == 1

    def test_custom_run_ids(self):
        rd = compute_score_diff({}, {}, run_a_id="v1", run_b_id="v2")
        assert rd.run_a_id == "v1"
        assert rd.run_b_id == "v2"


class TestComputeTextDiff:
    def test_basic(self):
        a = {"i1": "hello world"}
        b = {"i1": "hello there"}
        diffs = compute_text_diff(a, b)
        assert len(diffs) == 1
        assert "there" in diffs[0].added_words
        assert "world" in diffs[0].removed_words

    def test_identical(self):
        a = {"i1": "same text"}
        b = {"i1": "same text"}
        diffs = compute_text_diff(a, b)
        assert len(diffs) == 0

    def test_only_common_items(self):
        a = {"i1": "hello", "i2": "world"}
        b = {"i1": "hi", "i3": "other"}
        diffs = compute_text_diff(a, b)
        assert all(d.item_id == "i1" for d in diffs)

    def test_empty(self):
        diffs = compute_text_diff({}, {})
        assert len(diffs) == 0

    def test_custom_field(self):
        a = {"i1": "hello"}
        b = {"i1": "world"}
        diffs = compute_text_diff(a, b, field_name="input")
        assert diffs[0].field == "input"


class TestFilterDiffs:
    def _make_run_diff(self):
        diffs = [
            ItemDiff("i1", "acc", 0.8, 0.9, 0.1, "improved"),
            ItemDiff("i2", "acc", 0.9, 0.7, -0.2, "regressed"),
            ItemDiff("i3", "lat", 0.5, 0.5, 0.0, "unchanged"),
            ItemDiff("i4", "acc", 0.6, 0.65, 0.05, "improved"),
        ]
        return RunDiff("a", "b", diffs, 2, 1, 1, 4)

    def test_filter_by_direction(self):
        rd = self._make_run_diff()
        improved = filter_diffs(rd, direction="improved")
        assert len(improved) == 2

    def test_filter_by_metric(self):
        rd = self._make_run_diff()
        acc = filter_diffs(rd, metric_id="acc")
        assert len(acc) == 3

    def test_filter_by_min_delta(self):
        rd = self._make_run_diff()
        big = filter_diffs(rd, min_delta=0.1)
        assert len(big) == 2

    def test_combined_filters(self):
        rd = self._make_run_diff()
        result = filter_diffs(rd, direction="improved", min_delta=0.08)
        assert len(result) == 1
        assert result[0].item_id == "i1"

    def test_no_filters(self):
        rd = self._make_run_diff()
        assert len(filter_diffs(rd)) == 4


class TestFormatScoreDiff:
    def test_contains_header(self):
        rd = RunDiff("v1", "v2", [], 0, 0, 0, 0)
        text = format_score_diff(rd)
        assert "v1" in text and "v2" in text

    def test_shows_diffs(self):
        diffs = [ItemDiff("i1", "acc", 0.8, 0.9, 0.1, "improved")]
        rd = RunDiff("a", "b", diffs, 1, 0, 0, 1)
        text = format_score_diff(rd)
        assert "i1" in text
        assert "0.8000" in text


class TestFormatTextDiff:
    def test_shows_changes(self):
        diffs = [TextDiff("i1", "output", "a b", "a c", ["c"], ["b"])]
        text = format_text_diff(diffs)
        assert "removed: b" in text
        assert "added: c" in text

    def test_empty(self):
        assert format_text_diff([]) == ""


class TestSummarizeDiff:
    def test_basic(self):
        diffs = [
            ItemDiff("i1", "acc", 0.8, 0.9, 0.1, "improved"),
            ItemDiff("i2", "acc", 0.9, 0.7, -0.2, "regressed"),
        ]
        rd = RunDiff("a", "b", diffs, 1, 1, 0, 2)
        summary = summarize_diff(rd)
        assert summary["improved_count"] == 1
        assert summary["regressed_count"] == 1
        assert abs(summary["mean_delta"] - (-0.05)) < 1e-9
        assert summary["most_improved"]["item_id"] == "i1"
        assert summary["most_regressed"]["item_id"] == "i2"

    def test_empty(self):
        rd = RunDiff("a", "b", [], 0, 0, 0, 0)
        summary = summarize_diff(rd)
        assert summary["mean_delta"] == 0.0
        assert summary["most_improved"] is None
