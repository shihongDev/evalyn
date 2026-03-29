"""Tests for balanced sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.balanced_sampling import (
    BalanceConfig,
    BalanceResult,
    CategoryStats,
    compute_balance_score,
    compute_category_counts,
    format_balance_report,
    oversample,
    proportional_sample,
    run_balanced_sampling,
    suggest_rebalancing,
    undersample,
)


# ---- BalanceConfig ----


class TestBalanceConfig:
    def test_defaults(self):
        c = BalanceConfig(balance_field="topic")
        assert c.balance_field == "topic"
        assert c.strategy == "undersample"
        assert c.target_size == 0
        assert c.seed is None

    def test_custom_values(self):
        c = BalanceConfig(
            balance_field="lang",
            strategy="oversample",
            target_size=50,
            seed=42,
        )
        assert c.strategy == "oversample"
        assert c.target_size == 50

    def test_as_dict(self):
        c = BalanceConfig(balance_field="cat", seed=7)
        d = c.as_dict()
        assert d == {
            "balance_field": "cat",
            "strategy": "undersample",
            "target_size": 0,
            "seed": 7,
        }

    def test_from_dict_full(self):
        d = {
            "balance_field": "topic",
            "strategy": "proportional",
            "target_size": 100,
            "seed": 99,
        }
        c = BalanceConfig.from_dict(d)
        assert c.balance_field == "topic"
        assert c.strategy == "proportional"
        assert c.target_size == 100
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = BalanceConfig.from_dict({"balance_field": "x"})
        assert c.strategy == "undersample"
        assert c.target_size == 0
        assert c.seed is None

    def test_roundtrip(self):
        c = BalanceConfig(
            balance_field="f", strategy="oversample", target_size=10, seed=5
        )
        assert BalanceConfig.from_dict(c.as_dict()) == c


# ---- CategoryStats ----


class TestCategoryStats:
    def test_creation(self):
        s = CategoryStats(category="sports", count=50, sampled_count=20, ratio=0.4)
        assert s.category == "sports"
        assert s.ratio == 0.4

    def test_as_dict(self):
        s = CategoryStats(category="tech", count=30, sampled_count=10, ratio=0.33)
        d = s.as_dict()
        assert d == {
            "category": "tech",
            "count": 30,
            "sampled_count": 10,
            "ratio": 0.33,
        }

    def test_from_dict(self):
        d = {"category": "art", "count": 15, "sampled_count": 5, "ratio": 0.25}
        s = CategoryStats.from_dict(d)
        assert s.category == "art"
        assert s.sampled_count == 5

    def test_roundtrip(self):
        s = CategoryStats(category="x", count=10, sampled_count=3, ratio=0.3)
        assert CategoryStats.from_dict(s.as_dict()) == s


# ---- BalanceResult ----


class TestBalanceResult:
    def test_defaults(self):
        r = BalanceResult()
        assert r.selected_ids == []
        assert r.category_stats == []
        assert r.balance_score == 0.0

    def test_as_dict(self):
        r = BalanceResult(
            selected_ids=["a"],
            category_stats=[
                CategoryStats(category="x", count=5, sampled_count=3, ratio=0.5)
            ],
            total_pool=10,
            total_selected=3,
            balance_score=0.9,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["category_stats"]) == 1
        assert d["category_stats"][0]["category"] == "x"

    def test_from_dict(self):
        d = {
            "selected_ids": ["b", "c"],
            "category_stats": [
                {"category": "y", "count": 8, "sampled_count": 4, "ratio": 0.5}
            ],
            "total_pool": 20,
            "total_selected": 4,
            "balance_score": 0.85,
        }
        r = BalanceResult.from_dict(d)
        assert r.total_selected == 4
        assert r.category_stats[0].category == "y"

    def test_roundtrip(self):
        r = BalanceResult(
            selected_ids=["a"],
            category_stats=[
                CategoryStats(category="z", count=1, sampled_count=1, ratio=1.0)
            ],
            total_pool=1,
            total_selected=1,
            balance_score=1.0,
        )
        assert BalanceResult.from_dict(r.as_dict()) == r


# ---- compute_category_counts ----


class TestComputeCategoryCounts:
    def test_basic_grouping(self):
        items = [
            {"id": "1", "topic": "sports"},
            {"id": "2", "topic": "tech"},
            {"id": "3", "topic": "sports"},
        ]
        groups = compute_category_counts(items, "topic")
        assert set(groups.keys()) == {"sports", "tech"}
        assert groups["sports"] == ["1", "3"]
        assert groups["tech"] == ["2"]

    def test_missing_field(self):
        items = [
            {"id": "1", "topic": "sports"},
            {"id": "2"},
        ]
        groups = compute_category_counts(items, "topic")
        assert "<missing>" in groups
        assert groups["<missing>"] == ["2"]

    def test_empty_items(self):
        groups = compute_category_counts([], "topic")
        assert groups == {}

    def test_single_category(self):
        items = [{"id": str(i), "cat": "only"} for i in range(5)]
        groups = compute_category_counts(items, "cat")
        assert list(groups.keys()) == ["only"]
        assert len(groups["only"]) == 5


# ---- compute_balance_score ----


class TestComputeBalanceScore:
    def test_perfect_balance(self):
        assert compute_balance_score({"a": 10, "b": 10, "c": 10}) == 1.0

    def test_single_category(self):
        assert compute_balance_score({"a": 100}) == 1.0

    def test_empty(self):
        assert compute_balance_score({}) == 1.0

    def test_imbalanced(self):
        score = compute_balance_score({"a": 100, "b": 1})
        assert 0.0 < score < 1.0

    def test_extreme_imbalance(self):
        score = compute_balance_score({"a": 1000, "b": 1, "c": 1})
        assert score < 0.5

    def test_two_equal_categories(self):
        assert compute_balance_score({"a": 50, "b": 50}) == 1.0

    def test_all_zero_counts(self):
        assert compute_balance_score({"a": 0, "b": 0}) == 1.0


# ---- undersample ----


class TestUndersample:
    def test_basic(self):
        groups = {"a": ["1", "2", "3"], "b": ["4", "5"]}
        result = undersample(groups, 2, seed=42)
        # Category a: 2 of 3, category b: 2 of 2
        a_items = [x for x in result if x in ["1", "2", "3"]]
        b_items = [x for x in result if x in ["4", "5"]]
        assert len(a_items) == 2
        assert len(b_items) == 2

    def test_target_exceeds_category(self):
        groups = {"a": ["1"], "b": ["2", "3", "4"]}
        result = undersample(groups, 5, seed=0)
        assert "1" in result
        assert len([x for x in result if x in ["2", "3", "4"]]) == 3

    def test_target_zero(self):
        groups = {"a": ["1", "2"]}
        result = undersample(groups, 0, seed=0)
        assert result == []

    def test_deterministic(self):
        groups = {"a": [str(i) for i in range(20)]}
        r1 = undersample(groups, 5, seed=42)
        r2 = undersample(groups, 5, seed=42)
        assert r1 == r2

    def test_empty_groups(self):
        result = undersample({}, 10)
        assert result == []


# ---- oversample ----


class TestOversample:
    def test_basic(self):
        groups = {"a": ["1"], "b": ["2", "3", "4"]}
        result = oversample(groups, 3, seed=42)
        a_items = [x for x in result if x == "1"]
        b_items = [x for x in result if x in ["2", "3", "4"]]
        assert len(a_items) == 3  # oversampled from 1 to 3
        assert len(b_items) == 3

    def test_no_oversampling_needed(self):
        groups = {"a": ["1", "2", "3"], "b": ["4", "5", "6"]}
        result = oversample(groups, 3, seed=0)
        assert len(result) == 6

    def test_target_zero(self):
        groups = {"a": ["1", "2"]}
        result = oversample(groups, 0, seed=0)
        assert result == []

    def test_deterministic(self):
        groups = {"a": ["1"], "b": ["2", "3"]}
        r1 = oversample(groups, 5, seed=99)
        r2 = oversample(groups, 5, seed=99)
        assert r1 == r2

    def test_empty_groups(self):
        result = oversample({}, 10)
        assert result == []


# ---- proportional_sample ----


class TestProportionalSample:
    def test_basic(self):
        groups = {"a": [str(i) for i in range(80)], "b": [str(i + 80) for i in range(20)]}
        result = proportional_sample(groups, 10, seed=42)
        assert len(result) == 10
        # Category a should get more samples
        a_count = len([x for x in result if int(x) < 80])
        b_count = len([x for x in result if int(x) >= 80])
        assert a_count > b_count

    def test_at_least_one_per_category(self):
        groups = {"a": ["1"] * 100, "b": ["2"], "c": ["3"]}
        result = proportional_sample(groups, 5, seed=42)
        cats_present = set()
        for item in result:
            if item == "1":
                cats_present.add("a")
            elif item == "2":
                cats_present.add("b")
            elif item == "3":
                cats_present.add("c")
        assert cats_present == {"a", "b", "c"}

    def test_total_zero(self):
        groups = {"a": ["1"]}
        result = proportional_sample(groups, 0)
        assert result == []

    def test_empty_groups(self):
        result = proportional_sample({}, 10)
        assert result == []

    def test_deterministic(self):
        groups = {"a": [str(i) for i in range(50)], "b": [str(i + 50) for i in range(50)]}
        r1 = proportional_sample(groups, 10, seed=7)
        r2 = proportional_sample(groups, 10, seed=7)
        assert r1 == r2

    def test_budget_less_than_categories(self):
        groups = {"a": ["1"], "b": ["2"], "c": ["3"], "d": ["4"]}
        result = proportional_sample(groups, 2, seed=42)
        assert len(result) == 2


# ---- run_balanced_sampling ----


class TestRunBalancedSampling:
    def _make_items(self, cats):
        """Helper: create item dicts from a list of (id, category) pairs."""
        return [{"id": item_id, "topic": cat} for item_id, cat in cats]

    def test_undersample_auto(self):
        items = self._make_items(
            [("1", "a"), ("2", "a"), ("3", "a"), ("4", "b"), ("5", "b")]
        )
        config = BalanceConfig(balance_field="topic", strategy="undersample", seed=42)
        result = run_balanced_sampling(items, config)
        # Auto target: min category (b) has 2, so 2 per category = 4 total
        assert result.total_selected == 4
        assert result.total_pool == 5

    def test_oversample_auto(self):
        items = self._make_items(
            [("1", "a"), ("2", "a"), ("3", "a"), ("4", "b")]
        )
        config = BalanceConfig(balance_field="topic", strategy="oversample", seed=42)
        result = run_balanced_sampling(items, config)
        # Auto target: max category (a) has 3, so 3 per category = 6 total
        assert result.total_selected == 6

    def test_proportional(self):
        items = self._make_items(
            [("1", "a"), ("2", "a"), ("3", "a"), ("4", "b")]
        )
        config = BalanceConfig(
            balance_field="topic", strategy="proportional", seed=42
        )
        result = run_balanced_sampling(items, config)
        # Auto total = len(items) = 4
        assert result.total_selected == 4

    def test_empty_items(self):
        config = BalanceConfig(balance_field="topic")
        result = run_balanced_sampling([], config)
        assert result.selected_ids == []
        assert result.balance_score == 1.0

    def test_unknown_strategy(self):
        items = [{"id": "1", "topic": "a"}]
        config = BalanceConfig(balance_field="topic", strategy="unknown")
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_balanced_sampling(items, config)

    def test_balance_score_in_result(self):
        items = self._make_items(
            [("1", "a"), ("2", "a"), ("3", "b"), ("4", "b")]
        )
        config = BalanceConfig(balance_field="topic", strategy="undersample", seed=0)
        result = run_balanced_sampling(items, config)
        assert 0.0 <= result.balance_score <= 1.0

    def test_category_stats_present(self):
        items = self._make_items([("1", "x"), ("2", "y")])
        config = BalanceConfig(balance_field="topic", strategy="undersample", seed=0)
        result = run_balanced_sampling(items, config)
        cats = {s.category for s in result.category_stats}
        assert cats == {"x", "y"}

    def test_with_target_size(self):
        items = self._make_items(
            [("1", "a"), ("2", "a"), ("3", "a"), ("4", "b"), ("5", "b"), ("6", "b")]
        )
        config = BalanceConfig(
            balance_field="topic",
            strategy="undersample",
            target_size=4,
            seed=42,
        )
        result = run_balanced_sampling(items, config)
        assert result.total_selected == 4


# ---- format_balance_report ----


class TestFormatBalanceReport:
    def test_contains_header(self):
        result = BalanceResult(
            selected_ids=["a"],
            category_stats=[
                CategoryStats(category="x", count=5, sampled_count=3, ratio=1.0)
            ],
            total_pool=5,
            total_selected=3,
            balance_score=1.0,
        )
        report = format_balance_report(result)
        assert "Balanced Sampling Report" in report
        assert "=" * 40 in report

    def test_contains_stats(self):
        result = BalanceResult(
            selected_ids=["a", "b"],
            category_stats=[
                CategoryStats(category="sports", count=10, sampled_count=5, ratio=0.5),
                CategoryStats(category="tech", count=10, sampled_count=5, ratio=0.5),
            ],
            total_pool=20,
            total_selected=10,
            balance_score=1.0,
        )
        report = format_balance_report(result)
        assert "Total pool size: 20" in report
        assert "Total selected: 10" in report
        assert "sports" in report
        assert "tech" in report

    def test_contains_selected_ids(self):
        result = BalanceResult(selected_ids=["item_1", "item_2"])
        report = format_balance_report(result)
        assert "item_1" in report
        assert "item_2" in report

    def test_empty_result(self):
        result = BalanceResult()
        report = format_balance_report(result)
        assert "Total selected: 0" in report


# ---- suggest_rebalancing ----


class TestSuggestRebalancing:
    def test_balanced(self):
        stats = [
            CategoryStats(category="a", count=10, sampled_count=5, ratio=0.5),
            CategoryStats(category="b", count=10, sampled_count=5, ratio=0.5),
        ]
        suggestions = suggest_rebalancing(stats)
        assert len(suggestions) == 1
        assert "well-balanced" in suggestions[0]

    def test_over_represented(self):
        stats = [
            CategoryStats(category="big", count=100, sampled_count=90, ratio=0.9),
            CategoryStats(category="small", count=10, sampled_count=10, ratio=0.1),
        ]
        suggestions = suggest_rebalancing(stats)
        over = [s for s in suggestions if "over-represented" in s]
        under = [s for s in suggestions if "under-represented" in s]
        assert len(over) >= 1
        assert len(under) >= 1

    def test_empty_stats(self):
        suggestions = suggest_rebalancing([])
        assert suggestions == []

    def test_single_category(self):
        stats = [CategoryStats(category="only", count=50, sampled_count=50, ratio=1.0)]
        suggestions = suggest_rebalancing(stats)
        # Single category, target is 1.0, ratio is 1.0 - balanced
        assert len(suggestions) == 1
        assert "well-balanced" in suggestions[0]

    def test_three_categories_imbalanced(self):
        stats = [
            CategoryStats(category="a", count=100, sampled_count=80, ratio=0.8),
            CategoryStats(category="b", count=10, sampled_count=10, ratio=0.1),
            CategoryStats(category="c", count=10, sampled_count=10, ratio=0.1),
        ]
        suggestions = suggest_rebalancing(stats)
        # "a" is over, "b" and "c" are under
        texts = " ".join(suggestions)
        assert "over-represented" in texts
        assert "under-represented" in texts
