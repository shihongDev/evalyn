"""Tests for score-stratified sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.stratified_sampling import (
    ScoreBin,
    StratifiedConfig,
    StratifiedResult,
    compute_bin_balance,
    create_score_bins,
    format_stratified_report,
    run_stratified_sampling,
    sample_from_bins,
)


# ---- ScoreBin ----


class TestScoreBin:
    def test_construction(self):
        b = ScoreBin(bin_id="b0", low=0.0, high=0.2, item_ids=["a"], count=1)
        assert b.bin_id == "b0"
        assert b.low == 0.0
        assert b.high == 0.2
        assert b.count == 1

    def test_defaults(self):
        b = ScoreBin(bin_id="b0", low=0.0, high=0.5)
        assert b.item_ids == []
        assert b.count == 0

    def test_as_dict(self):
        b = ScoreBin(bin_id="b1", low=0.2, high=0.4, item_ids=["x"], count=1)
        d = b.as_dict()
        assert d == {
            "bin_id": "b1",
            "low": 0.2,
            "high": 0.4,
            "item_ids": ["x"],
            "count": 1,
        }

    def test_from_dict(self):
        d = {"bin_id": "b2", "low": 0.4, "high": 0.6, "item_ids": ["y"], "count": 1}
        b = ScoreBin.from_dict(d)
        assert b.bin_id == "b2"
        assert b.item_ids == ["y"]

    def test_from_dict_defaults(self):
        d = {"bin_id": "b0", "low": 0.0, "high": 0.5}
        b = ScoreBin.from_dict(d)
        assert b.item_ids == []
        assert b.count == 0

    def test_roundtrip(self):
        b = ScoreBin(bin_id="b3", low=0.6, high=0.8, item_ids=["a", "b"], count=2)
        assert ScoreBin.from_dict(b.as_dict()) == b


# ---- StratifiedConfig ----


class TestStratifiedConfig:
    def test_defaults(self):
        c = StratifiedConfig()
        assert c.n_bins == 5
        assert c.sample_per_bin == 10
        assert c.seed is None

    def test_custom_values(self):
        c = StratifiedConfig(n_bins=10, sample_per_bin=5, seed=42)
        assert c.n_bins == 10
        assert c.seed == 42

    def test_as_dict(self):
        c = StratifiedConfig(seed=7)
        d = c.as_dict()
        assert d == {"n_bins": 5, "sample_per_bin": 10, "seed": 7}

    def test_from_dict_full(self):
        d = {"n_bins": 3, "sample_per_bin": 20, "seed": 99}
        c = StratifiedConfig.from_dict(d)
        assert c.n_bins == 3
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = StratifiedConfig.from_dict({})
        assert c.n_bins == 5
        assert c.seed is None

    def test_roundtrip(self):
        c = StratifiedConfig(n_bins=8, seed=11)
        assert StratifiedConfig.from_dict(c.as_dict()) == c


# ---- StratifiedResult ----


class TestStratifiedResult:
    def test_defaults(self):
        r = StratifiedResult()
        assert r.selected_ids == []
        assert r.bins == []
        assert r.total_pool == 0
        assert r.total_selected == 0
        assert r.bin_balance == 0.0

    def test_as_dict(self):
        b = ScoreBin(bin_id="b0", low=0.0, high=0.5, item_ids=["a"], count=1)
        r = StratifiedResult(
            selected_ids=["a"],
            bins=[b],
            total_pool=1,
            total_selected=1,
            bin_balance=1.0,
        )
        d = r.as_dict()
        assert d["total_pool"] == 1
        assert len(d["bins"]) == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "bins": [{"bin_id": "b0", "low": 0.0, "high": 0.5, "item_ids": ["x"], "count": 1}],
            "total_pool": 1,
            "total_selected": 1,
            "bin_balance": 0.9,
        }
        r = StratifiedResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert r.bins[0].bin_id == "b0"

    def test_from_dict_empty(self):
        r = StratifiedResult.from_dict({})
        assert r.selected_ids == []
        assert r.total_pool == 0


# ---- create_score_bins ----


class TestCreateScoreBins:
    def test_five_bins_default(self):
        scores = {"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.7, "e": 0.9}
        bins = create_score_bins(scores, n_bins=5)
        assert len(bins) == 5
        assert bins[0].low == 0.0
        assert bins[4].high == 1.0

    def test_items_assigned_correctly(self):
        scores = {"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.7, "e": 0.9}
        bins = create_score_bins(scores, n_bins=5)
        # bin_0: [0.0, 0.2) should have "a" (0.1)
        assert "a" in bins[0].item_ids
        # bin_1: [0.2, 0.4) should have "b" (0.3)
        assert "b" in bins[1].item_ids

    def test_score_zero(self):
        scores = {"a": 0.0}
        bins = create_score_bins(scores, n_bins=5)
        assert "a" in bins[0].item_ids

    def test_score_one(self):
        scores = {"a": 1.0}
        bins = create_score_bins(scores, n_bins=5)
        assert "a" in bins[4].item_ids

    def test_empty_scores(self):
        bins = create_score_bins({}, n_bins=5)
        assert len(bins) == 5
        assert all(b.count == 0 for b in bins)

    def test_single_bin(self):
        scores = {"a": 0.1, "b": 0.9}
        bins = create_score_bins(scores, n_bins=1)
        assert len(bins) == 1
        assert bins[0].count == 2

    def test_zero_bins(self):
        bins = create_score_bins({"a": 0.5}, n_bins=0)
        assert bins == []

    def test_counts_match_items(self):
        scores = {f"i{i}": i / 10.0 for i in range(11)}
        bins = create_score_bins(scores, n_bins=5)
        total = sum(b.count for b in bins)
        assert total == 11
        for b in bins:
            assert b.count == len(b.item_ids)

    def test_bin_ids_sequential(self):
        bins = create_score_bins({"a": 0.5}, n_bins=3)
        assert [b.bin_id for b in bins] == ["bin_0", "bin_1", "bin_2"]

    def test_many_items(self):
        scores = {f"i{i}": i / 1000.0 for i in range(1000)}
        bins = create_score_bins(scores, n_bins=10)
        total = sum(b.count for b in bins)
        assert total == 1000

    def test_all_same_score(self):
        scores = {f"i{i}": 0.5 for i in range(20)}
        bins = create_score_bins(scores, n_bins=5)
        total = sum(b.count for b in bins)
        assert total == 20


# ---- sample_from_bins ----


class TestSampleFromBins:
    def test_basic_sampling(self):
        bins = [
            ScoreBin(bin_id="b0", low=0.0, high=0.5, item_ids=["a", "b", "c"], count=3),
            ScoreBin(bin_id="b1", low=0.5, high=1.0, item_ids=["d", "e"], count=2),
        ]
        selected = sample_from_bins(bins, sample_per_bin=2, seed=42)
        assert len(selected) == 4  # 2 from b0 + 2 from b1

    def test_sample_more_than_available(self):
        bins = [
            ScoreBin(bin_id="b0", low=0.0, high=0.5, item_ids=["a"], count=1),
        ]
        selected = sample_from_bins(bins, sample_per_bin=10)
        assert selected == ["a"]

    def test_empty_bins(self):
        bins = [
            ScoreBin(bin_id="b0", low=0.0, high=0.5, item_ids=[], count=0),
        ]
        selected = sample_from_bins(bins, sample_per_bin=5)
        assert selected == []

    def test_no_bins(self):
        selected = sample_from_bins([], sample_per_bin=5)
        assert selected == []

    def test_deterministic_with_seed(self):
        bins = [
            ScoreBin(bin_id="b0", low=0.0, high=0.5, item_ids=[f"i{i}" for i in range(20)], count=20),
        ]
        s1 = sample_from_bins(bins, sample_per_bin=5, seed=42)
        s2 = sample_from_bins(bins, sample_per_bin=5, seed=42)
        assert s1 == s2

    def test_different_seeds(self):
        ids = [f"i{i}" for i in range(50)]
        bins = [ScoreBin(bin_id="b0", low=0.0, high=1.0, item_ids=ids, count=50)]
        s1 = sample_from_bins(bins, sample_per_bin=10, seed=1)
        s2 = sample_from_bins(bins, sample_per_bin=10, seed=2)
        assert s1 != s2

    def test_all_items_when_sample_equals_count(self):
        bins = [
            ScoreBin(bin_id="b0", low=0.0, high=1.0, item_ids=["a", "b", "c"], count=3),
        ]
        selected = sample_from_bins(bins, sample_per_bin=3)
        assert set(selected) == {"a", "b", "c"}


# ---- compute_bin_balance ----


class TestComputeBinBalance:
    def test_perfect_balance(self):
        bins = [ScoreBin(bin_id=f"b{i}", low=0.0, high=1.0) for i in range(3)]
        balance = compute_bin_balance(bins, [10, 10, 10])
        assert balance == 1.0

    def test_complete_imbalance(self):
        bins = [ScoreBin(bin_id=f"b{i}", low=0.0, high=1.0) for i in range(3)]
        balance = compute_bin_balance(bins, [30, 0, 0])
        assert balance < 0.5

    def test_empty_counts(self):
        bins = []
        balance = compute_bin_balance(bins, [])
        assert balance == 1.0

    def test_all_zeros(self):
        bins = [ScoreBin(bin_id="b0", low=0.0, high=1.0)]
        balance = compute_bin_balance(bins, [0])
        assert balance == 1.0

    def test_single_bin(self):
        bins = [ScoreBin(bin_id="b0", low=0.0, high=1.0)]
        balance = compute_bin_balance(bins, [5])
        assert balance == 1.0

    def test_moderate_imbalance(self):
        bins = [ScoreBin(bin_id=f"b{i}", low=0.0, high=1.0) for i in range(4)]
        balance = compute_bin_balance(bins, [10, 8, 12, 10])
        assert 0.5 < balance < 1.0

    def test_clamped_to_zero(self):
        # Extreme imbalance: CV > 1.0 should clamp to 0.0
        bins = [ScoreBin(bin_id=f"b{i}", low=0.0, high=1.0) for i in range(5)]
        balance = compute_bin_balance(bins, [100, 0, 0, 0, 0])
        assert balance == 0.0


# ---- run_stratified_sampling ----


class TestRunStratifiedSampling:
    def test_empty_scores(self):
        r = run_stratified_sampling({}, StratifiedConfig())
        assert r.selected_ids == []
        assert r.total_pool == 0
        assert r.bin_balance == 1.0

    def test_basic_run(self):
        scores = {f"i{i}": i / 50.0 for i in range(50)}
        config = StratifiedConfig(n_bins=5, sample_per_bin=5, seed=42)
        r = run_stratified_sampling(scores, config)
        assert len(r.bins) == 5
        assert r.total_pool == 50
        assert r.total_selected == len(r.selected_ids)

    def test_total_selected_matches(self):
        scores = {f"i{i}": i / 20.0 for i in range(20)}
        config = StratifiedConfig(n_bins=4, sample_per_bin=3, seed=1)
        r = run_stratified_sampling(scores, config)
        assert r.total_selected == len(r.selected_ids)

    def test_deterministic_with_seed(self):
        scores = {f"i{i}": i / 100.0 for i in range(100)}
        config = StratifiedConfig(n_bins=5, sample_per_bin=10, seed=42)
        r1 = run_stratified_sampling(scores, config)
        r2 = run_stratified_sampling(scores, config)
        assert r1.selected_ids == r2.selected_ids

    def test_different_seeds(self):
        scores = {f"i{i}": i / 100.0 for i in range(100)}
        r1 = run_stratified_sampling(scores, StratifiedConfig(n_bins=5, sample_per_bin=5, seed=1))
        r2 = run_stratified_sampling(scores, StratifiedConfig(n_bins=5, sample_per_bin=5, seed=2))
        assert r1.selected_ids != r2.selected_ids

    def test_bin_balance_is_set(self):
        scores = {f"i{i}": i / 100.0 for i in range(100)}
        config = StratifiedConfig(n_bins=5, sample_per_bin=10, seed=1)
        r = run_stratified_sampling(scores, config)
        assert 0.0 <= r.bin_balance <= 1.0

    def test_uniform_distribution_high_balance(self):
        # Evenly distributed scores should yield high balance
        scores = {f"i{i}": i / 100.0 for i in range(100)}
        config = StratifiedConfig(n_bins=5, sample_per_bin=10, seed=1)
        r = run_stratified_sampling(scores, config)
        assert r.bin_balance > 0.7

    def test_small_pool_large_sample(self):
        scores = {"a": 0.1, "b": 0.9}
        config = StratifiedConfig(n_bins=5, sample_per_bin=100, seed=1)
        r = run_stratified_sampling(scores, config)
        assert set(r.selected_ids) == {"a", "b"}

    def test_single_bin(self):
        scores = {f"i{i}": i / 10.0 for i in range(10)}
        config = StratifiedConfig(n_bins=1, sample_per_bin=5, seed=1)
        r = run_stratified_sampling(scores, config)
        assert len(r.bins) == 1
        assert r.total_selected == 5


# ---- format_stratified_report ----


class TestFormatStratifiedReport:
    def test_contains_header(self):
        r = StratifiedResult(total_pool=10, total_selected=5, bin_balance=0.9)
        report = format_stratified_report(r)
        assert "Stratified Sampling Report" in report

    def test_contains_stats(self):
        r = StratifiedResult(total_pool=50, total_selected=20, bin_balance=0.85)
        report = format_stratified_report(r)
        assert "Total pool size: 50" in report
        assert "Total selected: 20" in report
        assert "0.8500" in report

    def test_contains_bin_breakdown(self):
        b = ScoreBin(bin_id="bin_0", low=0.0, high=0.2, item_ids=["a"], count=1)
        r = StratifiedResult(
            selected_ids=["a"],
            bins=[b],
            total_pool=1,
            total_selected=1,
            bin_balance=1.0,
        )
        report = format_stratified_report(r)
        assert "bin_0" in report
        assert "[0.00, 0.20)" in report

    def test_contains_selected_ids(self):
        r = StratifiedResult(selected_ids=["alpha", "beta"])
        report = format_stratified_report(r)
        assert "- alpha" in report
        assert "- beta" in report

    def test_empty_result(self):
        r = StratifiedResult()
        report = format_stratified_report(r)
        assert "Stratified Sampling Report" in report
        assert "Total pool size: 0" in report

    def test_number_of_bins_shown(self):
        bins = [ScoreBin(bin_id=f"b{i}", low=0.0, high=1.0) for i in range(3)]
        r = StratifiedResult(bins=bins)
        report = format_stratified_report(r)
        assert "Number of bins: 3" in report
