"""Tests for coverage-aware sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.coverage_sampling import (
    CoverageConfig,
    CoverageResult,
    CoverageScore,
    compute_coverage_rate,
    compute_item_features,
    format_coverage_sampling_report,
    greedy_max_coverage,
    run_coverage_sampling,
    score_marginal_coverage,
)


# ---- CoverageConfig ----


class TestCoverageConfig:
    def test_defaults(self):
        c = CoverageConfig()
        assert c.sample_size == 100
        assert c.diversity_weight == 1.0
        assert c.seed is None

    def test_custom_values(self):
        c = CoverageConfig(sample_size=50, diversity_weight=0.5, seed=42)
        assert c.sample_size == 50
        assert c.diversity_weight == 0.5
        assert c.seed == 42

    def test_as_dict(self):
        c = CoverageConfig(sample_size=10, seed=7)
        d = c.as_dict()
        assert d == {"sample_size": 10, "diversity_weight": 1.0, "seed": 7}

    def test_from_dict_full(self):
        d = {"sample_size": 25, "diversity_weight": 0.8, "seed": 99}
        c = CoverageConfig.from_dict(d)
        assert c.sample_size == 25
        assert c.diversity_weight == 0.8
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = CoverageConfig.from_dict({})
        assert c.sample_size == 100
        assert c.diversity_weight == 1.0
        assert c.seed is None

    def test_roundtrip(self):
        c = CoverageConfig(sample_size=33, diversity_weight=0.1, seed=5)
        assert CoverageConfig.from_dict(c.as_dict()) == c


# ---- CoverageScore ----


class TestCoverageScore:
    def test_creation(self):
        s = CoverageScore(item_id="a1", marginal_coverage=0.5)
        assert s.item_id == "a1"
        assert s.marginal_coverage == 0.5

    def test_as_dict(self):
        s = CoverageScore(item_id="x", marginal_coverage=0.75)
        assert s.as_dict() == {"item_id": "x", "marginal_coverage": 0.75}

    def test_from_dict(self):
        d = {"item_id": "y", "marginal_coverage": 0.33}
        s = CoverageScore.from_dict(d)
        assert s.item_id == "y"
        assert s.marginal_coverage == 0.33

    def test_roundtrip(self):
        s = CoverageScore(item_id="z", marginal_coverage=0.0)
        assert CoverageScore.from_dict(s.as_dict()) == s


# ---- CoverageResult ----


class TestCoverageResult:
    def test_defaults(self):
        r = CoverageResult()
        assert r.selected_ids == []
        assert r.coverage_rate == 0.0
        assert r.total_pool == 0

    def test_as_dict(self):
        r = CoverageResult(
            selected_ids=["a", "b"],
            coverage_rate=0.8,
            total_pool=10,
            vocabulary_covered=40,
            vocabulary_total=50,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert d["coverage_rate"] == 0.8
        assert d["vocabulary_total"] == 50

    def test_from_dict(self):
        d = {
            "selected_ids": ["c"],
            "coverage_rate": 0.5,
            "total_pool": 5,
            "vocabulary_covered": 10,
            "vocabulary_total": 20,
        }
        r = CoverageResult.from_dict(d)
        assert r.selected_ids == ["c"]
        assert r.vocabulary_covered == 10

    def test_roundtrip(self):
        r = CoverageResult(
            selected_ids=["x", "y"],
            coverage_rate=0.9,
            total_pool=100,
            vocabulary_covered=90,
            vocabulary_total=100,
        )
        assert CoverageResult.from_dict(r.as_dict()) == r


# ---- compute_item_features ----


class TestComputeItemFeatures:
    def test_basic_words(self):
        feats = compute_item_features("Hello World")
        assert "hello" in feats
        assert "world" in feats

    def test_filters_short_words(self):
        feats = compute_item_features("I am a cat")
        assert "i" not in feats
        assert "a" not in feats
        assert "am" in feats
        assert "cat" in feats

    def test_lowercases(self):
        feats = compute_item_features("FOO Bar BAZ")
        assert feats == {"foo", "bar", "baz"}

    def test_empty_string(self):
        feats = compute_item_features("")
        assert feats == set()

    def test_only_short_words(self):
        feats = compute_item_features("I a x")
        assert feats == set()

    def test_punctuation_split(self):
        feats = compute_item_features("hello, world! foo-bar")
        assert "hello" in feats
        assert "world" in feats
        assert "foo" in feats
        assert "bar" in feats

    def test_numbers_included(self):
        feats = compute_item_features("test 42 abc123")
        assert "test" in feats
        assert "42" in feats
        assert "abc123" in feats


# ---- compute_coverage_rate ----


class TestComputeCoverageRate:
    def test_full_coverage(self):
        total = {"a", "b", "c"}
        selected = {"a", "b", "c", "d"}
        assert compute_coverage_rate(selected, total) == 1.0

    def test_partial_coverage(self):
        total = {"a", "b", "c", "d"}
        selected = {"a", "b"}
        assert compute_coverage_rate(selected, total) == 0.5

    def test_no_coverage(self):
        total = {"a", "b"}
        selected = {"x", "y"}
        assert compute_coverage_rate(selected, total) == 0.0

    def test_empty_total(self):
        assert compute_coverage_rate({"a"}, set()) == 1.0

    def test_empty_selected(self):
        assert compute_coverage_rate(set(), {"a", "b"}) == 0.0


# ---- score_marginal_coverage ----


class TestScoreMarginalCoverage:
    def test_full_new_coverage(self):
        item = {"a", "b", "c"}
        covered = set()
        total = {"a", "b", "c"}
        assert score_marginal_coverage(item, covered, total) == 1.0

    def test_partial_new_coverage(self):
        item = {"a", "b"}
        covered = {"a"}
        total = {"a", "b", "c"}
        # uncovered = {"b", "c"}, item covers {"b"} -> 1/2
        assert score_marginal_coverage(item, covered, total) == 0.5

    def test_no_new_coverage(self):
        item = {"a"}
        covered = {"a", "b"}
        total = {"a", "b"}
        assert score_marginal_coverage(item, covered, total) == 0.0

    def test_all_covered_already(self):
        item = {"x", "y"}
        covered = {"a", "b", "c"}
        total = {"a", "b", "c"}
        assert score_marginal_coverage(item, covered, total) == 0.0

    def test_empty_item(self):
        assert score_marginal_coverage(set(), set(), {"a"}) == 0.0


# ---- greedy_max_coverage ----


class TestGreedyMaxCoverage:
    def test_empty_items(self):
        result = greedy_max_coverage({}, 10)
        assert result.selected_ids == []
        assert result.total_pool == 0
        assert result.coverage_rate == 1.0

    def test_single_item(self):
        result = greedy_max_coverage({"a": "hello world"}, 5)
        assert result.selected_ids == ["a"]
        assert result.total_pool == 1
        assert result.coverage_rate == 1.0

    def test_selects_diverse_items_first(self):
        items = {
            "a": "cat dog",
            "b": "cat dog",
            "c": "fish bird",
        }
        result = greedy_max_coverage(items, 2, seed=42)
        # Should pick one of a/b (cat dog) and c (fish bird) for max coverage
        selected_set = set(result.selected_ids)
        assert "c" in selected_set or len(selected_set) == 2
        assert result.coverage_rate == 1.0

    def test_sample_size_limits_selection(self):
        items = {f"item{i}": f"word{i}" for i in range(20)}
        result = greedy_max_coverage(items, 5, seed=1)
        assert len(result.selected_ids) == 5
        assert result.total_pool == 20

    def test_sample_size_exceeds_pool(self):
        items = {"a": "foo", "b": "bar"}
        result = greedy_max_coverage(items, 100)
        assert len(result.selected_ids) == 2
        assert result.coverage_rate == 1.0

    def test_deterministic_with_seed(self):
        items = {f"i{i}": f"word{i} common" for i in range(10)}
        r1 = greedy_max_coverage(items, 5, seed=42)
        r2 = greedy_max_coverage(items, 5, seed=42)
        assert r1.selected_ids == r2.selected_ids

    def test_different_seeds_may_differ(self):
        # When there are ties, different seeds should break them differently
        items = {f"i{i}": "same words here" for i in range(10)}
        r1 = greedy_max_coverage(items, 3, seed=1)
        r2 = greedy_max_coverage(items, 3, seed=999)
        # They pick from same-coverage items, so results may differ
        # Just check both are valid
        assert len(r1.selected_ids) == 3
        assert len(r2.selected_ids) == 3

    def test_coverage_rate_computation(self):
        items = {
            "a": "alpha beta",
            "b": "gamma delta",
            "c": "alpha gamma epsilon",
        }
        result = greedy_max_coverage(items, 2, seed=0)
        assert 0.0 < result.coverage_rate <= 1.0
        assert result.vocabulary_covered <= result.vocabulary_total

    def test_greedy_picks_largest_first(self):
        items = {
            "small": "ab",
            "big": "one two three four five six seven eight",
        }
        result = greedy_max_coverage(items, 1, seed=0)
        assert result.selected_ids == ["big"]


# ---- run_coverage_sampling ----


class TestRunCoverageSampling:
    def test_basic_run(self):
        items = {"a": "hello world", "b": "foo bar", "c": "hello foo"}
        config = CoverageConfig(sample_size=2, seed=42)
        result = run_coverage_sampling(items, config)
        assert len(result.selected_ids) == 2
        assert result.total_pool == 3

    def test_uses_config_seed(self):
        items = {f"i{i}": f"w{i}" for i in range(10)}
        c1 = CoverageConfig(sample_size=5, seed=7)
        c2 = CoverageConfig(sample_size=5, seed=7)
        r1 = run_coverage_sampling(items, c1)
        r2 = run_coverage_sampling(items, c2)
        assert r1.selected_ids == r2.selected_ids

    def test_uses_config_sample_size(self):
        items = {f"i{i}": f"w{i}" for i in range(20)}
        config = CoverageConfig(sample_size=3)
        result = run_coverage_sampling(items, config)
        assert len(result.selected_ids) == 3

    def test_empty_items(self):
        config = CoverageConfig(sample_size=10)
        result = run_coverage_sampling({}, config)
        assert result.selected_ids == []
        assert result.total_pool == 0


# ---- format_coverage_sampling_report ----


class TestFormatCoverageSamplingReport:
    def test_contains_header(self):
        result = CoverageResult(
            selected_ids=["a"],
            coverage_rate=0.8,
            total_pool=10,
            vocabulary_covered=40,
            vocabulary_total=50,
        )
        report = format_coverage_sampling_report(result)
        assert "Coverage Sampling Report" in report
        assert "=" * 40 in report

    def test_contains_stats(self):
        result = CoverageResult(
            selected_ids=["x", "y"],
            coverage_rate=0.75,
            total_pool=20,
            vocabulary_covered=30,
            vocabulary_total=40,
        )
        report = format_coverage_sampling_report(result)
        assert "Total pool size: 20" in report
        assert "Selected samples: 2" in report
        assert "75.00%" in report
        assert "30 / 40" in report

    def test_contains_selected_ids(self):
        result = CoverageResult(selected_ids=["item_1", "item_2"])
        report = format_coverage_sampling_report(result)
        assert "item_1" in report
        assert "item_2" in report

    def test_empty_result(self):
        result = CoverageResult()
        report = format_coverage_sampling_report(result)
        assert "Selected samples: 0" in report
