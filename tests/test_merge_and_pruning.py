"""Tests for dataset merge/diff and metric correlation pruning."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets.merge import (
    diff_datasets, merge_datasets, DatasetDiff,
)
from evalyn_sdk.analysis.correlation_pruning import (
    compute_correlations, recommend_pruning,
    CorrelationPair, PruningRecommendation, _pearson_r,
)


def _item(item_id: str, input_text: str = "test", output_text: str = "answer") -> DatasetItem:
    return DatasetItem(id=item_id, input=input_text, output=output_text)


# =============================================================================
# Dataset diff tests
# =============================================================================

class TestDiffDatasets:
    def test_identical(self):
        a = [_item("1", "hello", "world")]
        b = [_item("1", "hello", "world")]
        diff = diff_datasets(a, b)
        assert not diff.has_changes
        assert diff.unchanged == 1

    def test_added(self):
        a = [_item("1")]
        b = [_item("1"), _item("2")]
        diff = diff_datasets(a, b)
        assert len(diff.added) == 1
        assert diff.added[0].id == "2"

    def test_removed(self):
        a = [_item("1"), _item("2")]
        b = [_item("1")]
        diff = diff_datasets(a, b)
        assert len(diff.removed) == 1
        assert diff.removed[0].id == "2"

    def test_modified(self):
        a = [_item("1", "hello", "world")]
        b = [_item("1", "hello", "universe")]
        diff = diff_datasets(a, b)
        assert len(diff.modified) == 1
        assert diff.unchanged == 0

    def test_complex_diff(self):
        a = [_item("1", "a"), _item("2", "b"), _item("3", "c")]
        b = [_item("1", "a"), _item("2", "b_changed"), _item("4", "d")]
        diff = diff_datasets(a, b)
        assert diff.summary == {"added": 1, "removed": 1, "modified": 1, "unchanged": 1}

    def test_empty_datasets(self):
        diff = diff_datasets([], [])
        assert not diff.has_changes

    def test_format_text(self):
        a = [_item("1")]
        b = [_item("1"), _item("2")]
        diff = diff_datasets(a, b)
        text = diff.format_text()
        assert "Added:" in text
        assert "1" in text


# =============================================================================
# Dataset merge tests
# =============================================================================

class TestMergeDatasets:
    def test_basic_merge(self):
        a = [_item("1", "hello")]
        b = [_item("2", "world")]
        merged, dupes = merge_datasets(a, b)
        assert len(merged) == 2
        assert dupes == 0

    def test_merge_dedup(self):
        a = [_item("1", "same input")]
        b = [_item("2", "same input")]  # same content, different ID
        merged, dupes = merge_datasets(a, b, deduplicate=True)
        assert len(merged) == 1
        assert dupes == 1

    def test_merge_no_dedup(self):
        a = [_item("1", "same")]
        b = [_item("2", "same")]
        merged, dupes = merge_datasets(a, b, deduplicate=False)
        assert len(merged) == 2
        assert dupes == 0

    def test_merge_prefer_b(self):
        a = [_item("1", "old")]
        b = [_item("1", "new")]
        merged, _ = merge_datasets(a, b, prefer="b")
        assert merged[0].input == "new"

    def test_merge_prefer_a(self):
        a = [_item("1", "old")]
        b = [_item("1", "new")]
        merged, _ = merge_datasets(a, b, prefer="a")
        assert merged[0].input == "old"

    def test_merge_empty(self):
        merged, dupes = merge_datasets([], [])
        assert len(merged) == 0


# =============================================================================
# Correlation pruning tests
# =============================================================================

class TestPearsonR:
    def test_perfect_positive(self):
        r = _pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0)

    def test_perfect_negative(self):
        r = _pearson_r([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r == pytest.approx(-1.0)

    def test_no_correlation(self):
        r = _pearson_r([1, 2, 3, 4, 5], [3, 1, 4, 1, 5])
        assert r is not None
        assert abs(r) < 0.5

    def test_constant_returns_none(self):
        r = _pearson_r([1, 1, 1], [2, 3, 4])
        assert r is None

    def test_too_few_returns_none(self):
        assert _pearson_r([1], [2]) is None


class TestComputeCorrelations:
    def test_basic(self):
        scores = {
            "a": [0.8, 0.9, 0.7, 0.6, 0.8],
            "b": [0.7, 0.8, 0.6, 0.5, 0.7],  # correlated with a
        }
        pairs = compute_correlations(scores)
        assert len(pairs) == 1
        assert pairs[0].correlation > 0.9

    def test_redundant_classification(self):
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "b": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
        pairs = compute_correlations(scores)
        assert pairs[0].relationship == "redundant"

    def test_tradeoff_classification(self):
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "b": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
        pairs = compute_correlations(scores)
        assert pairs[0].relationship == "tradeoff"

    def test_too_few_samples_skipped(self):
        scores = {"a": [0.5, 0.6], "b": [0.7, 0.8]}
        pairs = compute_correlations(scores, min_samples=5)
        assert len(pairs) == 0

    def test_empty(self):
        assert compute_correlations({}) == []


class TestRecommendPruning:
    def test_no_redundant(self):
        pairs = [CorrelationPair("a", "b", 0.3, "independent")]
        rec = recommend_pruning(pairs)
        assert len(rec.drop) == 0
        assert "a" in rec.keep and "b" in rec.keep

    def test_drop_redundant(self):
        pairs = [CorrelationPair("a", "b", 0.95, "redundant")]
        rec = recommend_pruning(pairs)
        assert len(rec.drop) == 1
        assert len(rec.keep) == 1

    def test_prefer_objective(self):
        pairs = [CorrelationPair("obj", "subj", 0.9, "redundant")]
        types = {"obj": "objective", "subj": "subjective"}
        rec = recommend_pruning(pairs, metric_types=types)
        assert "subj" in rec.drop  # drop the expensive one
        assert "obj" in rec.keep

    def test_cost_savings(self):
        pairs = [CorrelationPair("a", "b", 0.9, "redundant")]
        types = {"a": "objective", "b": "subjective"}
        rec = recommend_pruning(pairs, metric_types=types)
        assert rec.estimated_cost_savings == 1.0  # 1 of 1 subjective dropped

    def test_format_text(self):
        pairs = [CorrelationPair("help", "accuracy", 0.92, "redundant")]
        rec = recommend_pruning(pairs)
        text = rec.format_text()
        assert "Pruning" in text
        assert "Drop:" in text

    def test_empty_correlations(self):
        rec = recommend_pruning([])
        assert len(rec.drop) == 0

    def test_as_dict(self):
        pairs = [CorrelationPair("a", "b", 0.9, "redundant")]
        rec = recommend_pruning(pairs)
        d = rec.as_dict()
        assert "keep" in d
        assert "drop" in d
        assert "redundant_pairs" in d
