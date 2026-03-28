"""Tests for the item difficulty estimation module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.item_difficulty import (
    DifficultyReport,
    ItemDifficulty,
    classify_difficulty,
    compute_difficulty_variance,
    compute_item_fail_rate,
    estimate_all_difficulties,
    estimate_difficulty,
    sort_by_difficulty,
)


# ---------------------------------------------------------------------------
# compute_item_fail_rate
# ---------------------------------------------------------------------------


class TestComputeItemFailRate:
    def test_all_pass(self):
        """All passing results -> 0.0 fail rate."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": True},
        ]
        assert compute_item_fail_rate("a", results) == 0.0

    def test_all_fail(self):
        """All failing results -> 1.0 fail rate."""
        results = [
            {"item_id": "a", "passed": False},
            {"item_id": "a", "passed": False},
        ]
        assert compute_item_fail_rate("a", results) == 1.0

    def test_mixed(self):
        """Mixed results compute correct ratio."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": False},
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": False},
        ]
        assert compute_item_fail_rate("a", results) == 0.5

    def test_filters_by_item_id(self):
        """Only counts results matching the given item_id."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
            {"item_id": "a", "passed": False},
        ]
        assert compute_item_fail_rate("a", results) == 0.5
        assert compute_item_fail_rate("b", results) == 1.0

    def test_no_matching_results(self):
        """No matching item_id -> 0.0."""
        results = [{"item_id": "b", "passed": False}]
        assert compute_item_fail_rate("a", results) == 0.0

    def test_empty_results(self):
        """Empty results list -> 0.0."""
        assert compute_item_fail_rate("a", []) == 0.0

    def test_single_pass(self):
        """Single passing result -> 0.0."""
        results = [{"item_id": "a", "passed": True}]
        assert compute_item_fail_rate("a", results) == 0.0

    def test_single_fail(self):
        """Single failing result -> 1.0."""
        results = [{"item_id": "a", "passed": False}]
        assert compute_item_fail_rate("a", results) == 1.0

    def test_missing_passed_defaults_true(self):
        """Missing 'passed' key defaults to True (pass)."""
        results = [{"item_id": "a"}]
        assert compute_item_fail_rate("a", results) == 0.0


# ---------------------------------------------------------------------------
# classify_difficulty
# ---------------------------------------------------------------------------


class TestClassifyDifficulty:
    def test_easy(self):
        """fail_rate < 0.1 -> easy."""
        assert classify_difficulty(0.0) == "easy"
        assert classify_difficulty(0.05) == "easy"
        assert classify_difficulty(0.09) == "easy"

    def test_medium(self):
        """0.1 <= fail_rate < 0.4 -> medium."""
        assert classify_difficulty(0.1) == "medium"
        assert classify_difficulty(0.25) == "medium"
        assert classify_difficulty(0.39) == "medium"

    def test_hard(self):
        """0.4 <= fail_rate < 0.7 -> hard."""
        assert classify_difficulty(0.4) == "hard"
        assert classify_difficulty(0.5) == "hard"
        assert classify_difficulty(0.69) == "hard"

    def test_very_hard(self):
        """fail_rate >= 0.7 -> very_hard."""
        assert classify_difficulty(0.7) == "very_hard"
        assert classify_difficulty(0.85) == "very_hard"
        assert classify_difficulty(1.0) == "very_hard"

    def test_boundary_0_1(self):
        """Exact boundary at 0.1."""
        assert classify_difficulty(0.1) == "medium"

    def test_boundary_0_4(self):
        """Exact boundary at 0.4."""
        assert classify_difficulty(0.4) == "hard"

    def test_boundary_0_7(self):
        """Exact boundary at 0.7."""
        assert classify_difficulty(0.7) == "very_hard"


# ---------------------------------------------------------------------------
# compute_difficulty_variance
# ---------------------------------------------------------------------------


class TestComputeDifficultyVariance:
    def test_all_pass(self):
        """All True -> variance 0.0."""
        assert compute_difficulty_variance([True, True, True]) == 0.0

    def test_all_fail(self):
        """All False -> variance 0.0."""
        assert compute_difficulty_variance([False, False, False]) == 0.0

    def test_mixed_half(self):
        """Half pass, half fail -> variance 0.25."""
        v = compute_difficulty_variance([True, False])
        assert abs(v - 0.25) < 1e-9

    def test_empty(self):
        """Empty list -> 0.0."""
        assert compute_difficulty_variance([]) == 0.0

    def test_single_element(self):
        """Single element -> variance 0.0."""
        assert compute_difficulty_variance([True]) == 0.0
        assert compute_difficulty_variance([False]) == 0.0

    def test_mostly_pass(self):
        """3 pass, 1 fail -> specific variance."""
        v = compute_difficulty_variance([True, True, True, False])
        # mean = 0.75, var = ((0.25)^2 * 3 + (0.75)^2 * 1) / 4 = (0.1875 + 0.5625) / 4 = 0.1875
        assert abs(v - 0.1875) < 1e-9


# ---------------------------------------------------------------------------
# estimate_difficulty
# ---------------------------------------------------------------------------


class TestEstimateDifficulty:
    def test_full_estimation(self):
        """Full estimation produces correct fields."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": False},
            {"item_id": "a", "passed": True},
            {"item_id": "a", "passed": True},
        ]
        d = estimate_difficulty("a", results)
        assert d.item_id == "a"
        assert d.fail_rate == 0.25
        assert d.num_runs == 4
        assert d.difficulty_label == "medium"
        assert d.variance > 0

    def test_no_matching_runs(self):
        """No matching runs -> easy, 0 runs."""
        d = estimate_difficulty("x", [])
        assert d.fail_rate == 0.0
        assert d.num_runs == 0
        assert d.difficulty_label == "easy"

    def test_all_fail_very_hard(self):
        """All failures -> very_hard."""
        results = [
            {"item_id": "a", "passed": False},
            {"item_id": "a", "passed": False},
        ]
        d = estimate_difficulty("a", results)
        assert d.difficulty_label == "very_hard"
        assert d.fail_rate == 1.0


# ---------------------------------------------------------------------------
# estimate_all_difficulties
# ---------------------------------------------------------------------------


class TestEstimateAllDifficulties:
    def test_basic_report(self):
        """Basic report with mixed items."""
        results = [
            {"item_id": "easy", "passed": True},
            {"item_id": "easy", "passed": True},
            {"item_id": "hard", "passed": False},
            {"item_id": "hard", "passed": False},
        ]
        report = estimate_all_difficulties(["easy", "hard"], results)
        assert len(report.items) == 2
        assert report.avg_difficulty == 0.5
        assert "easy" in report.distribution or "very_hard" in report.distribution

    def test_empty_items(self):
        """Empty items list -> empty report."""
        report = estimate_all_difficulties([], [])
        assert report.items == []
        assert report.avg_difficulty == 0.0

    def test_hardest_and_easiest(self):
        """Hardest and easiest items are populated correctly."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
            {"item_id": "c", "passed": False},
        ]
        report = estimate_all_difficulties(["a", "b", "c"], results)
        assert "a" in report.easiest_items
        # b and c both fail, should be in hardest
        assert report.hardest_items[0] in ("b", "c")

    def test_distribution_counts(self):
        """Distribution sums to total items."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
        ]
        report = estimate_all_difficulties(["a", "b"], results)
        total = sum(report.distribution.values())
        assert total == 2

    def test_single_item(self):
        """Single item report."""
        results = [{"item_id": "x", "passed": True}]
        report = estimate_all_difficulties(["x"], results)
        assert len(report.items) == 1
        assert report.avg_difficulty == 0.0


# ---------------------------------------------------------------------------
# sort_by_difficulty
# ---------------------------------------------------------------------------


class TestSortByDifficulty:
    def test_hardest_first_default(self):
        """Default sort: hardest first (highest fail_rate)."""
        items = [
            ItemDifficulty(item_id="easy", fail_rate=0.1),
            ItemDifficulty(item_id="hard", fail_rate=0.9),
            ItemDifficulty(item_id="mid", fail_rate=0.5),
        ]
        sorted_items = sort_by_difficulty(items)
        assert sorted_items[0].item_id == "hard"
        assert sorted_items[-1].item_id == "easy"

    def test_ascending(self):
        """Ascending sort: easiest first."""
        items = [
            ItemDifficulty(item_id="easy", fail_rate=0.1),
            ItemDifficulty(item_id="hard", fail_rate=0.9),
        ]
        sorted_items = sort_by_difficulty(items, ascending=True)
        assert sorted_items[0].item_id == "easy"

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert sort_by_difficulty([]) == []

    def test_single_item(self):
        """Single item returns same."""
        items = [ItemDifficulty(item_id="a", fail_rate=0.5)]
        sorted_items = sort_by_difficulty(items)
        assert len(sorted_items) == 1
        assert sorted_items[0].item_id == "a"


# ---------------------------------------------------------------------------
# DifficultyReport format_text
# ---------------------------------------------------------------------------


class TestDifficultyReportFormatText:
    def test_format_text_header(self):
        """format_text includes item count."""
        results = [{"item_id": "a", "passed": True}]
        report = estimate_all_difficulties(["a"], results)
        text = report.format_text()
        assert "1 items" in text

    def test_format_text_avg(self):
        """format_text includes average fail rate."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
        ]
        report = estimate_all_difficulties(["a", "b"], results)
        text = report.format_text()
        assert "Average fail rate" in text

    def test_format_text_distribution(self):
        """format_text lists distribution labels."""
        results = [{"item_id": "a", "passed": True}]
        report = estimate_all_difficulties(["a"], results)
        text = report.format_text()
        assert "Distribution" in text

    def test_format_text_hardest(self):
        """format_text includes hardest items."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
        ]
        report = estimate_all_difficulties(["a", "b"], results)
        text = report.format_text()
        assert "Hardest" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_item_difficulty_as_dict(self):
        """ItemDifficulty.as_dict returns all fields."""
        d = ItemDifficulty(item_id="x", fail_rate=0.5, num_runs=3, difficulty_label="hard", variance=0.2)
        data = d.as_dict()
        assert data["item_id"] == "x"
        assert data["fail_rate"] == 0.5
        assert data["num_runs"] == 3
        assert data["difficulty_label"] == "hard"
        assert data["variance"] == 0.2

    def test_item_difficulty_roundtrip(self):
        """ItemDifficulty survives as_dict -> from_dict."""
        original = ItemDifficulty(item_id="y", fail_rate=0.3, num_runs=5, difficulty_label="medium", variance=0.1)
        restored = ItemDifficulty.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.fail_rate == original.fail_rate
        assert restored.num_runs == original.num_runs
        assert restored.difficulty_label == original.difficulty_label
        assert restored.variance == original.variance

    def test_difficulty_report_roundtrip(self):
        """DifficultyReport survives as_dict -> from_dict."""
        results = [
            {"item_id": "a", "passed": True},
            {"item_id": "b", "passed": False},
        ]
        original = estimate_all_difficulties(["a", "b"], results)
        data = original.as_dict()
        restored = DifficultyReport.from_dict(data)
        assert len(restored.items) == len(original.items)
        assert restored.avg_difficulty == original.avg_difficulty
        assert restored.distribution == original.distribution
        assert restored.hardest_items == original.hardest_items
        assert restored.easiest_items == original.easiest_items

    def test_difficulty_report_from_dict_empty(self):
        """from_dict handles empty dict."""
        report = DifficultyReport.from_dict({})
        assert report.items == []
        assert report.avg_difficulty == 0.0
        assert report.distribution == {}

    def test_item_difficulty_from_dict_defaults(self):
        """from_dict uses defaults for missing optional fields."""
        d = ItemDifficulty.from_dict({"item_id": "z", "fail_rate": 0.5})
        assert d.num_runs == 0
        assert d.difficulty_label == "medium"
        assert d.variance == 0.0
