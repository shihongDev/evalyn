"""Tests for K-fold cross-validation evaluation.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.evaluation.cross_validation import (
    CVReport,
    FoldResult,
    aggregate_fold_results,
    build_cv_report,
    create_folds,
    detect_high_variance_items,
)


# ---------------------------------------------------------------------------
# create_folds
# ---------------------------------------------------------------------------


class TestCreateFolds:
    def test_correct_count(self):
        items = list(range(20))
        folds = create_folds(items, num_folds=5, seed=42)
        assert len(folds) == 5

    def test_all_items_covered(self):
        items = list(range(20))
        folds = create_folds(items, num_folds=5, seed=42)
        all_test = []
        for _train, test in folds:
            all_test.extend(test)
        assert sorted(all_test) == sorted(items)

    def test_no_overlap_between_test_sets(self):
        items = list(range(20))
        folds = create_folds(items, num_folds=5, seed=42)
        seen = set()
        for _train, test in folds:
            for item in test:
                assert item not in seen, f"Item {item} appears in multiple test sets"
                seen.add(item)

    def test_train_test_disjoint(self):
        items = list(range(20))
        folds = create_folds(items, num_folds=5, seed=42)
        for train, test in folds:
            assert not set(train) & set(test)

    def test_deterministic_with_seed(self):
        items = list(range(20))
        folds1 = create_folds(items, num_folds=5, seed=42)
        folds2 = create_folds(items, num_folds=5, seed=42)
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            assert train1 == train2
            assert test1 == test2

    def test_different_seeds_differ(self):
        items = list(range(50))
        folds1 = create_folds(items, num_folds=5, seed=1)
        folds2 = create_folds(items, num_folds=5, seed=2)
        # At least one fold should differ
        any_different = False
        for (_, t1), (_, t2) in zip(folds1, folds2):
            if t1 != t2:
                any_different = True
                break
        assert any_different

    def test_stratified(self):
        items = [("a", 1), ("a", 2), ("a", 3), ("a", 4),
                 ("b", 5), ("b", 6), ("b", 7), ("b", 8)]
        folds = create_folds(items, num_folds=2, seed=0, stratify_key=lambda x: x[0])
        # Each test fold should have items from both strata
        for _train, test in folds:
            labels = {x[0] for x in test}
            assert len(labels) == 2, f"Expected both strata in test set, got {labels}"

    def test_fewer_items_than_folds(self):
        items = [1, 2]
        folds = create_folds(items, num_folds=5, seed=42)
        assert len(folds) == 5
        # Some folds will have empty test sets
        total_test = sum(len(test) for _, test in folds)
        assert total_test == 2

    def test_single_fold(self):
        items = list(range(10))
        folds = create_folds(items, num_folds=1, seed=0)
        assert len(folds) == 1
        train, test = folds[0]
        assert len(train) == 0
        assert sorted(test) == sorted(items)

    def test_invalid_num_folds(self):
        with pytest.raises(ValueError):
            create_folds([1, 2, 3], num_folds=0)

    def test_empty_items(self):
        folds = create_folds([], num_folds=3, seed=42)
        assert len(folds) == 3
        for train, test in folds:
            assert train == []
            assert test == []


# ---------------------------------------------------------------------------
# aggregate_fold_results
# ---------------------------------------------------------------------------


class TestAggregateFoldResults:
    def test_mean_correct(self):
        folds = [
            FoldResult(0, 8, 2, {"accuracy": 0.8}),
            FoldResult(1, 8, 2, {"accuracy": 0.9}),
            FoldResult(2, 8, 2, {"accuracy": 1.0}),
        ]
        means, _ = aggregate_fold_results(folds)
        assert abs(means["accuracy"] - 0.9) < 1e-9

    def test_std_dev_correct(self):
        folds = [
            FoldResult(0, 8, 2, {"accuracy": 0.8}),
            FoldResult(1, 8, 2, {"accuracy": 0.9}),
            FoldResult(2, 8, 2, {"accuracy": 1.0}),
        ]
        _, std_devs = aggregate_fold_results(folds)
        # Population std dev of [0.8, 0.9, 1.0]: mean=0.9, var = (0.01+0+0.01)/3
        expected = math.sqrt(0.02 / 3)
        assert abs(std_devs["accuracy"] - expected) < 1e-9

    def test_multiple_metrics(self):
        folds = [
            FoldResult(0, 8, 2, {"a": 0.5, "b": 1.0}),
            FoldResult(1, 8, 2, {"a": 0.7, "b": 0.8}),
        ]
        means, std_devs = aggregate_fold_results(folds)
        assert "a" in means
        assert "b" in means
        assert abs(means["a"] - 0.6) < 1e-9
        assert abs(means["b"] - 0.9) < 1e-9

    def test_empty_folds(self):
        means, std_devs = aggregate_fold_results([])
        assert means == {}
        assert std_devs == {}

    def test_single_fold_zero_std(self):
        folds = [FoldResult(0, 5, 5, {"accuracy": 0.85})]
        _, std_devs = aggregate_fold_results(folds)
        assert std_devs["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# detect_high_variance_items
# ---------------------------------------------------------------------------


class TestDetectHighVariance:
    def test_flags_high_variance(self):
        folds = [
            FoldResult(0, 8, 2, {"stable": 0.9, "unstable": 0.5}),
            FoldResult(1, 8, 2, {"stable": 0.9, "unstable": 1.0}),
        ]
        flagged = detect_high_variance_items(folds, threshold=0.1)
        assert "unstable" in flagged
        assert "stable" not in flagged

    def test_none_flagged(self):
        folds = [
            FoldResult(0, 8, 2, {"a": 0.9}),
            FoldResult(1, 8, 2, {"a": 0.9}),
        ]
        assert detect_high_variance_items(folds, threshold=0.1) == []


# ---------------------------------------------------------------------------
# build_cv_report
# ---------------------------------------------------------------------------


class TestBuildCVReport:
    def test_report_structure(self):
        folds = [
            FoldResult(0, 8, 2, {"accuracy": 0.8}),
            FoldResult(1, 8, 2, {"accuracy": 0.9}),
        ]
        report = build_cv_report(folds)
        assert report.num_folds == 2
        assert "accuracy" in report.aggregate_scores
        assert "accuracy" in report.score_std_devs

    def test_format_text_contains_header(self):
        folds = [
            FoldResult(0, 8, 2, {"accuracy": 0.8}),
            FoldResult(1, 8, 2, {"accuracy": 0.9}),
        ]
        report = build_cv_report(folds)
        text = report.format_text()
        assert "Cross-Validation Report" in text
        assert "accuracy" in text

    def test_format_text_high_variance(self):
        folds = [
            FoldResult(0, 8, 2, {"wobbly": 0.3}),
            FoldResult(1, 8, 2, {"wobbly": 0.9}),
        ]
        report = build_cv_report(folds, high_variance_threshold=0.1)
        text = report.format_text()
        assert "High variance" in text
        assert "wobbly" in text


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_fold_result_as_dict(self):
        fr = FoldResult(0, 10, 5, {"accuracy": 0.85})
        d = fr.as_dict()
        assert d["fold_index"] == 0
        assert d["metric_scores"]["accuracy"] == 0.85

    def test_cv_report_roundtrip(self):
        folds = [
            FoldResult(0, 8, 2, {"accuracy": 0.8}),
            FoldResult(1, 8, 2, {"accuracy": 0.9}),
        ]
        report = build_cv_report(folds)
        d = report.as_dict()
        restored = CVReport.from_dict(d)
        assert restored.num_folds == 2
        assert len(restored.fold_results) == 2
        assert abs(restored.aggregate_scores["accuracy"] - 0.85) < 1e-3

    def test_cv_report_from_dict_defaults(self):
        report = CVReport.from_dict({})
        assert report.num_folds == 0
        assert report.fold_results == []
