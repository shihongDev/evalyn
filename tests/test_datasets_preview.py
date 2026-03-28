"""Tests for datasets_preview module."""

import pytest
from evalyn_sdk.datasets_preview import (
    PreviewStats,
    DatasetPreview,
    compute_preview_stats,
    sample_items,
    build_preview,
    render_preview_table,
    detect_data_quality_issues,
)


def _items(n: int, with_output: bool = False, category: str = "") -> list:
    """Helper to build item dicts."""
    result = []
    for i in range(n):
        item = {"input": f"input text number {i}"}
        if with_output:
            item["output"] = f"output {i}"
        if category:
            item["category"] = category
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# compute_preview_stats
# ---------------------------------------------------------------------------


class TestComputePreviewStats:
    def test_basic(self):
        items = [{"input": "hello"}, {"input": "world!"}]
        stats = compute_preview_stats(items)
        assert stats.total_items == 2
        assert stats.avg_input_length == pytest.approx(5.5)
        assert stats.has_outputs is False

    def test_with_outputs(self):
        items = [
            {"input": "abc", "output": "xy"},
            {"input": "defg", "output": "z"},
        ]
        stats = compute_preview_stats(items)
        assert stats.has_outputs is True
        assert stats.avg_output_length == pytest.approx(1.5)

    def test_input_length_range(self):
        items = [{"input": "a"}, {"input": "abcdef"}]
        stats = compute_preview_stats(items)
        assert stats.min_input_length == 1
        assert stats.max_input_length == 6

    def test_categories_counted(self):
        items = [
            {"input": "a", "category": "math"},
            {"input": "b", "category": "math"},
            {"input": "c", "category": "science"},
        ]
        stats = compute_preview_stats(items)
        assert stats.categories == {"math": 2, "science": 1}

    def test_empty_items(self):
        stats = compute_preview_stats([])
        assert stats.total_items == 0
        assert stats.avg_input_length == 0.0
        assert stats.min_input_length == 0

    def test_single_item(self):
        items = [{"input": "only one"}]
        stats = compute_preview_stats(items)
        assert stats.total_items == 1
        assert stats.min_input_length == 8
        assert stats.max_input_length == 8

    def test_no_category_key(self):
        items = [{"input": "abc"}]
        stats = compute_preview_stats(items)
        assert stats.categories == {}

    def test_empty_output_string(self):
        items = [{"input": "abc", "output": ""}]
        stats = compute_preview_stats(items)
        assert stats.has_outputs is False


# ---------------------------------------------------------------------------
# sample_items
# ---------------------------------------------------------------------------


class TestSampleItems:
    def test_correct_count(self):
        items = _items(20)
        sampled = sample_items(items, n=5)
        assert len(sampled) == 5

    def test_deterministic_with_seed(self):
        items = _items(20)
        a = sample_items(items, n=5, seed=42)
        b = sample_items(items, n=5, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        items = _items(50)
        a = sample_items(items, n=5, seed=1)
        b = sample_items(items, n=5, seed=2)
        assert a != b

    def test_fewer_items_than_n(self):
        items = _items(3)
        sampled = sample_items(items, n=10)
        assert len(sampled) == 3

    def test_empty_items(self):
        sampled = sample_items([], n=5)
        assert sampled == []

    def test_n_equals_length(self):
        items = _items(5)
        sampled = sample_items(items, n=5)
        assert len(sampled) == 5


# ---------------------------------------------------------------------------
# build_preview
# ---------------------------------------------------------------------------


class TestBuildPreview:
    def test_full(self):
        items = _items(10, with_output=True)
        preview = build_preview(items, sample_size=3, seed=42)
        assert preview.total_count == 10
        assert preview.sample_count == 3
        assert preview.stats is not None
        assert preview.stats.total_items == 10
        assert preview.stats.has_outputs is True

    def test_empty(self):
        preview = build_preview([])
        assert preview.total_count == 0
        assert preview.sample_count == 0
        assert preview.stats is not None
        assert preview.stats.total_items == 0

    def test_deterministic(self):
        items = _items(20, with_output=True)
        a = build_preview(items, sample_size=5, seed=99)
        b = build_preview(items, sample_size=5, seed=99)
        assert a.sample_items == b.sample_items


# ---------------------------------------------------------------------------
# render_preview_table
# ---------------------------------------------------------------------------


class TestRenderPreviewTable:
    def test_output_contains_items(self):
        items = [{"input": "hello world", "output": "answer"}]
        preview = DatasetPreview(sample_items=items, sample_count=1, total_count=1)
        table = render_preview_table(preview)
        assert "hello world" in table
        assert "answer" in table

    def test_truncation(self):
        long_text = "x" * 200
        items = [{"input": long_text, "output": "short"}]
        preview = DatasetPreview(sample_items=items, sample_count=1, total_count=1)
        table = render_preview_table(preview, max_text_length=40)
        assert "..." in table

    def test_empty_preview(self):
        preview = DatasetPreview()
        table = render_preview_table(preview)
        assert "no sample items" in table

    def test_multiple_rows(self):
        items = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ]
        preview = DatasetPreview(sample_items=items, sample_count=2, total_count=2)
        table = render_preview_table(preview)
        assert "q1" in table
        assert "q2" in table


# ---------------------------------------------------------------------------
# detect_data_quality_issues
# ---------------------------------------------------------------------------


class TestDetectDataQualityIssues:
    def test_short_inputs(self):
        stats = PreviewStats(total_items=5, avg_input_length=5.0)
        issues = detect_data_quality_issues(stats)
        assert any("short inputs" in i.lower() for i in issues)

    def test_no_outputs(self):
        stats = PreviewStats(total_items=5, avg_input_length=50.0, has_outputs=False)
        issues = detect_data_quality_issues(stats)
        assert any("no outputs" in i.lower() for i in issues)

    def test_healthy_dataset(self):
        stats = PreviewStats(
            total_items=10,
            avg_input_length=100.0,
            avg_output_length=50.0,
            has_outputs=True,
            max_input_length=200,
        )
        issues = detect_data_quality_issues(stats)
        assert issues == []

    def test_empty_stats(self):
        stats = PreviewStats()
        issues = detect_data_quality_issues(stats)
        # no items means no per-item issues flagged
        assert any("no outputs" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# PreviewStats format_text
# ---------------------------------------------------------------------------


class TestPreviewStatsFormatText:
    def test_basic_format(self):
        stats = PreviewStats(total_items=3, avg_input_length=12.5, has_outputs=True)
        text = stats.format_text()
        assert "Total items: 3" in text
        assert "12.5" in text
        assert "Has outputs: True" in text

    def test_categories_shown(self):
        stats = PreviewStats(
            total_items=2, categories={"math": 1, "science": 1}
        )
        text = stats.format_text()
        assert "math: 1" in text
        assert "science: 1" in text


# ---------------------------------------------------------------------------
# DatasetPreview format_text
# ---------------------------------------------------------------------------


class TestDatasetPreviewFormatText:
    def test_basic(self):
        preview = build_preview(
            [{"input": "hello", "output": "world"}], sample_size=1
        )
        text = preview.format_text()
        assert "1 of 1" in text
        assert "hello" in text

    def test_empty(self):
        preview = DatasetPreview()
        text = preview.format_text()
        assert "0 of 0" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_preview_stats_as_dict(self):
        stats = PreviewStats(total_items=5, avg_input_length=20.0, has_outputs=True)
        d = stats.as_dict()
        assert d["total_items"] == 5
        assert d["has_outputs"] is True

    def test_dataset_preview_roundtrip(self):
        original = build_preview(
            _items(10, with_output=True), sample_size=3, seed=42
        )
        d = original.as_dict()
        restored = DatasetPreview.from_dict(d)
        assert restored.total_count == original.total_count
        assert restored.sample_count == original.sample_count
        assert restored.stats is not None
        assert restored.stats.total_items == original.stats.total_items
        assert restored.sample_items == original.sample_items

    def test_from_dict_empty(self):
        restored = DatasetPreview.from_dict({})
        assert restored.total_count == 0
        assert restored.stats is None
        assert restored.sample_items == []
