"""Tests for evaluation warm-up and metric A/B variant testing."""

import pytest
from evalyn_sdk.models import DatasetItem, MetricResult
from evalyn_sdk.evaluation.warmup import (
    WarmupConfig, split_warmup_items, analyze_warmup_effect, WarmupResult,
)
from evalyn_sdk.evaluation.ab_testing import (
    compare_variants, ABTestResult,
)


def _item(iid):
    return DatasetItem(id=iid, input=f"q-{iid}", output=f"a-{iid}")


def _result(mid="m", iid="i1", passed=True, score=1.0):
    return MetricResult(
        metric_id=mid, item_id=iid, call_id="c1",
        score=score, passed=passed, details={},
    )


# =============================================================================
# Warm-up tests
# =============================================================================

class TestWarmupConfig:
    def test_defaults(self):
        cfg = WarmupConfig()
        assert cfg.warmup_count == 5
        assert cfg.discard_warmup is True
        assert cfg.enabled is False

    def test_as_dict(self):
        d = WarmupConfig(warmup_count=10, enabled=True).as_dict()
        assert d["warmup_count"] == 10
        assert d["enabled"] is True


class TestSplitWarmupItems:
    def test_basic_split(self):
        items = [_item(str(i)) for i in range(20)]
        warmup, remaining = split_warmup_items(items, warmup_count=5)
        assert len(warmup) == 5
        assert len(remaining) == 15

    def test_small_dataset(self):
        items = [_item(str(i)) for i in range(3)]
        warmup, remaining = split_warmup_items(items, warmup_count=10)
        assert len(warmup) == 3  # all items
        assert len(remaining) == 3  # also all items

    def test_empty(self):
        warmup, remaining = split_warmup_items([], warmup_count=5)
        assert warmup == []
        assert remaining == []


class TestAnalyzeWarmupEffect:
    def test_basic(self):
        warmup_results = [_result(iid="w1", score=0.3), _result(iid="w2", score=0.4)]
        post_results = [_result(iid="p1", score=0.9), _result(iid="p2", score=0.85)]
        wr = analyze_warmup_effect(warmup_results, post_results)
        assert wr.warmup_results_discarded == 2
        assert wr.warmup_pass_rate == 1.0  # both passed
        assert wr.post_warmup_pass_rate == 1.0

    def test_empty_warmup(self):
        wr = analyze_warmup_effect([], [_result()])
        assert wr.warmup_items_count == 0
        assert wr.warmup_results_discarded == 0

    def test_empty_post(self):
        wr = analyze_warmup_effect([_result()], [])
        assert wr.post_warmup_pass_rate == 0.0

    def test_variance_reduction(self):
        # High variance in warmup, low variance in post
        warmup = [_result(iid=f"w{i}", score=s) for i, s in enumerate([0.1, 0.9, 0.2, 0.8, 0.15])]
        post = [_result(iid=f"p{i}", score=s) for i, s in enumerate([0.85, 0.88, 0.87, 0.86, 0.89])]
        wr = analyze_warmup_effect(warmup, post)
        assert wr.variance_reduction > 0

    def test_as_dict(self):
        wr = analyze_warmup_effect([_result()], [_result()])
        d = wr.as_dict()
        assert "warmup_items_count" in d
        assert "variance_reduction" in d

    def test_format_text(self):
        wr = analyze_warmup_effect([_result()], [_result()])
        text = wr.format_text()
        assert "Warm-Up" in text


# =============================================================================
# A/B testing tests
# =============================================================================

class TestCompareVariants:
    def test_identical_results(self):
        results = [_result(iid=f"i{i}", passed=True) for i in range(5)]
        ab = compare_variants(results, results, "v1", "v2")
        assert ab.agreement_rate == 1.0
        assert ab.winner is None  # tied

    def test_variant_b_better(self):
        a_results = [_result(iid=f"i{i}", passed=(i < 3)) for i in range(10)]
        b_results = [_result(iid=f"i{i}", passed=(i < 8)) for i in range(10)]
        ab = compare_variants(a_results, b_results, "v1", "v2")
        assert ab.b_pass_rate > ab.a_pass_rate
        assert ab.winner == "v2"

    def test_variant_a_better(self):
        a_results = [_result(iid=f"i{i}", passed=True) for i in range(5)]
        b_results = [_result(iid=f"i{i}", passed=False) for i in range(5)]
        ab = compare_variants(a_results, b_results, "v1", "v2")
        assert ab.winner == "v1"

    def test_disagreement_items(self):
        a = [_result(iid="i1", passed=True), _result(iid="i2", passed=True)]
        b = [_result(iid="i1", passed=True), _result(iid="i2", passed=False)]
        ab = compare_variants(a, b)
        assert len(ab.disagreement_items) == 1
        assert "i2" in ab.disagreement_items

    def test_no_common_items(self):
        a = [_result(iid="a1")]
        b = [_result(iid="b1")]
        ab = compare_variants(a, b)
        assert ab.item_count == 0
        assert ab.agreement_rate == 0.0

    def test_score_comparison(self):
        a = [_result(iid="i1", score=0.7), _result(iid="i2", score=0.6)]
        b = [_result(iid="i1", score=0.9), _result(iid="i2", score=0.8)]
        ab = compare_variants(a, b)
        assert ab.b_avg_score > ab.a_avg_score
        assert ab.score_delta > 0

    def test_empty(self):
        ab = compare_variants([], [])
        assert ab.item_count == 0

    def test_as_dict(self):
        a = [_result(iid="i1")]
        b = [_result(iid="i1")]
        d = compare_variants(a, b, "v1", "v2").as_dict()
        assert d["variant_a"] == "v1"
        assert d["variant_b"] == "v2"
        assert "agreement_rate" in d
        assert "winner" in d

    def test_format_text(self):
        a = [_result(iid="i1")]
        b = [_result(iid="i1")]
        text = compare_variants(a, b, "v1", "v2").format_text()
        assert "A/B Test" in text
        assert "v1 vs v2" in text
