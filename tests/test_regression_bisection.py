"""Tests for the regression bisection module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.regression_bisection import (
    BisectionStep,
    BisectionResult,
    bisect_items,
    compute_pass_rate,
    identify_regression_items,
    rank_by_regression_impact,
    build_bisection_report,
)


# ---------------------------------------------------------------------------
# compute_pass_rate
# ---------------------------------------------------------------------------


class TestComputePassRate:
    def test_basic(self):
        items = [
            {"id": "a", "score": 0.8},
            {"id": "b", "score": 0.3},
            {"id": "c", "score": 0.6},
        ]
        rate = compute_pass_rate(items, threshold=0.5)
        assert abs(rate - 2 / 3) < 1e-9

    def test_all_pass(self):
        items = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.9}]
        assert compute_pass_rate(items, threshold=0.5) == 1.0

    def test_all_fail(self):
        items = [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.2}]
        assert compute_pass_rate(items, threshold=0.5) == 0.0

    def test_empty_items(self):
        assert compute_pass_rate([], threshold=0.5) == 0.0

    def test_custom_score_field(self):
        items = [{"id": "a", "quality": 0.9}, {"id": "b", "quality": 0.1}]
        rate = compute_pass_rate(items, score_field="quality", threshold=0.5)
        assert rate == 0.5

    def test_threshold_boundary(self):
        """Score exactly at threshold should pass."""
        items = [{"id": "a", "score": 0.5}]
        assert compute_pass_rate(items, threshold=0.5) == 1.0

    def test_threshold_just_below(self):
        items = [{"id": "a", "score": 0.499}]
        assert compute_pass_rate(items, threshold=0.5) == 0.0

    def test_missing_score_field_defaults_zero(self):
        items = [{"id": "a"}]
        assert compute_pass_rate(items, threshold=0.5) == 0.0


# ---------------------------------------------------------------------------
# bisect_items
# ---------------------------------------------------------------------------


class TestBisectItems:
    def test_finds_culprit_half(self):
        """Items where the second half has lower scores should be identified."""
        items = [
            {"id": f"good_{i}", "score": 0.9} for i in range(4)
        ] + [
            {"id": f"bad_{i}", "score": 0.1} for i in range(4)
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.1)
        assert result.total_steps > 0
        assert len(result.culprit_items) > 0
        # Culprits should be from the bad items
        assert any("bad" in cid for cid in result.culprit_items)

    def test_single_item(self):
        items = [{"id": "only", "score": 0.1}]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn)
        assert result.culprit_items == ["only"]

    def test_two_items(self):
        items = [
            {"id": "good", "score": 0.9},
            {"id": "bad", "score": 0.1},
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn)
        assert len(result.culprit_items) <= 2

    def test_empty_items(self):
        def score_fn(subset):
            return 0.0

        result = bisect_items([], score_fn)
        assert result.culprit_items == []
        assert result.total_steps == 0

    def test_all_same_score(self):
        """When all items have identical scores, bisection should still terminate."""
        items = [{"id": f"item_{i}", "score": 0.5} for i in range(8)]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.1)
        # Should terminate without infinite loop
        assert result.total_steps > 0
        assert result.original_pass_rate == 1.0

    def test_original_pass_rate_recorded(self):
        items = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.9},
            {"id": "c", "score": 0.1},
            {"id": "d", "score": 0.1},
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn)
        assert result.original_pass_rate == 0.5

    def test_steps_recorded(self):
        items = [
            {"id": f"item_{i}", "score": 0.9 if i < 4 else 0.1}
            for i in range(8)
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.05)
        assert len(result.steps) > 0
        for step in result.steps:
            assert step.step_number > 0
            assert step.subset_size > 0

    def test_regression_magnitude(self):
        """Regression magnitude should reflect the difference."""
        items = [
            {"id": "good", "score": 0.9},
            {"id": "bad1", "score": 0.1},
            {"id": "bad2", "score": 0.1},
            {"id": "bad3", "score": 0.1},
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.05)
        assert result.regression_magnitude >= 0.0


# ---------------------------------------------------------------------------
# identify_regression_items
# ---------------------------------------------------------------------------


class TestIdentifyRegressionItems:
    def test_finds_regressed(self):
        items_a = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.8},
            {"id": "3", "score": 0.7},
        ]
        items_b = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.3},
            {"id": "3", "score": 0.7},
        ]
        regressed = identify_regression_items(items_a, items_b)
        assert regressed == ["2"]

    def test_no_regression(self):
        items_a = [{"id": "1", "score": 0.5}]
        items_b = [{"id": "1", "score": 0.8}]
        assert identify_regression_items(items_a, items_b) == []

    def test_missing_in_b(self):
        """Items only in A should not appear as regressed."""
        items_a = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]
        items_b = [{"id": "1", "score": 0.5}]
        regressed = identify_regression_items(items_a, items_b)
        assert "2" not in regressed
        assert "1" in regressed

    def test_custom_score_field(self):
        items_a = [{"id": "1", "quality": 0.9}]
        items_b = [{"id": "1", "quality": 0.3}]
        regressed = identify_regression_items(items_a, items_b, score_field="quality")
        assert regressed == ["1"]

    def test_empty_lists(self):
        assert identify_regression_items([], []) == []

    def test_equal_scores_not_regressed(self):
        items_a = [{"id": "1", "score": 0.5}]
        items_b = [{"id": "1", "score": 0.5}]
        assert identify_regression_items(items_a, items_b) == []


# ---------------------------------------------------------------------------
# rank_by_regression_impact
# ---------------------------------------------------------------------------


class TestRankByRegressionImpact:
    def test_sorted_by_drop(self):
        items = [
            {"id": "small_drop", "score": 0.7},
            {"id": "big_drop", "score": 0.1},
            {"id": "no_drop", "score": 1.0},
        ]
        baseline = {"small_drop": 0.8, "big_drop": 0.9, "no_drop": 0.5}
        ranked = rank_by_regression_impact(items, baseline)
        assert len(ranked) == 2
        assert ranked[0][0] == "big_drop"
        assert ranked[1][0] == "small_drop"

    def test_no_regression(self):
        items = [{"id": "a", "score": 0.9}]
        baseline = {"a": 0.5}
        ranked = rank_by_regression_impact(items, baseline)
        assert ranked == []

    def test_not_in_baseline(self):
        """Items not in baseline should be skipped."""
        items = [{"id": "new", "score": 0.1}]
        baseline = {"old": 0.9}
        ranked = rank_by_regression_impact(items, baseline)
        assert ranked == []

    def test_empty_items(self):
        ranked = rank_by_regression_impact([], {"a": 0.5})
        assert ranked == []

    def test_drop_magnitude_correct(self):
        items = [{"id": "a", "score": 0.3}]
        baseline = {"a": 0.8}
        ranked = rank_by_regression_impact(items, baseline)
        assert len(ranked) == 1
        assert abs(ranked[0][1] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# BisectionResult format_text
# ---------------------------------------------------------------------------


class TestBisectionResultFormatText:
    def test_format_contains_key_info(self):
        result = BisectionResult(
            culprit_items=["item_a", "item_b"],
            total_steps=3,
            steps=[
                BisectionStep(step_number=1, subset_size=4, pass_rate=0.5, direction="left"),
            ],
            original_pass_rate=0.75,
            regression_magnitude=0.25,
        )
        text = result.format_text()
        assert "BISECTION RESULT" in text
        assert "item_a" in text
        assert "item_b" in text
        assert "75.00%" in text
        assert "Steps:" in text

    def test_format_empty_result(self):
        result = BisectionResult()
        text = result.format_text()
        assert "BISECTION RESULT" in text
        assert "Culprit items: 0" in text


# ---------------------------------------------------------------------------
# BisectionStep as_dict
# ---------------------------------------------------------------------------


class TestBisectionStepAsDict:
    def test_as_dict(self):
        step = BisectionStep(
            step_number=1, subset_size=4, pass_rate=0.75,
            direction="left", item_ids=["a", "b"],
        )
        d = step.as_dict()
        assert d["step_number"] == 1
        assert d["subset_size"] == 4
        assert d["pass_rate"] == 0.75
        assert d["direction"] == "left"
        assert d["item_ids"] == ["a", "b"]


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_bisection_result_roundtrip(self):
        result = BisectionResult(
            culprit_items=["x", "y"],
            total_steps=2,
            steps=[
                BisectionStep(step_number=1, subset_size=4, pass_rate=0.5, direction="right"),
                BisectionStep(step_number=2, subset_size=2, pass_rate=0.0, direction="found"),
            ],
            original_pass_rate=0.8,
            regression_magnitude=0.3,
        )
        d = result.as_dict()
        restored = BisectionResult.from_dict(d)
        assert restored.culprit_items == result.culprit_items
        assert restored.total_steps == result.total_steps
        assert len(restored.steps) == len(result.steps)
        assert abs(restored.original_pass_rate - result.original_pass_rate) < 1e-9
        assert abs(restored.regression_magnitude - result.regression_magnitude) < 1e-9

    def test_from_dict_defaults(self):
        result = BisectionResult.from_dict({})
        assert result.culprit_items == []
        assert result.total_steps == 0
        assert result.steps == []


# ---------------------------------------------------------------------------
# build_bisection_report
# ---------------------------------------------------------------------------


class TestBuildBisectionReport:
    def test_returns_format_text(self):
        result = BisectionResult(
            culprit_items=["c1"],
            total_steps=1,
            original_pass_rate=0.9,
            regression_magnitude=0.4,
        )
        text = build_bisection_report(result)
        assert "BISECTION RESULT" in text
        assert "c1" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_bisect_large_set(self):
        """Bisection should handle larger item sets without error."""
        items = [{"id": f"i{i}", "score": 0.9 if i < 50 else 0.1} for i in range(100)]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.05)
        assert result.total_steps > 0
        assert len(result.culprit_items) > 0

    def test_bisect_three_items(self):
        """Odd number of items: one half is larger."""
        items = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.1},
            {"id": "c", "score": 0.1},
        ]

        def score_fn(subset):
            return compute_pass_rate(subset, threshold=0.5)

        result = bisect_items(items, score_fn, target_drop=0.05)
        assert result.total_steps > 0
