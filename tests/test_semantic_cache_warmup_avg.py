"""Tests for semantic caching and warmup averaging."""

import pytest
from evalyn_sdk.models import MetricResult
from evalyn_sdk.evaluation.semantic_cache import (
    SemanticCache, SemanticCacheStats,
)
from evalyn_sdk.evaluation.warmup_averaging import (
    average_results, AveragedResult, WarmupAveragingReport,
)


def _result(mid="m", iid="i1", score=0.8, passed=True):
    return MetricResult(
        metric_id=mid, item_id=iid, call_id="c1",
        score=score, passed=passed, details={},
    )


# =============================================================================
# Semantic cache tests
# =============================================================================

class TestSemanticCache:
    def test_exact_hit(self):
        cache = SemanticCache()
        cache.put("prompt", "input", "output", {"score": 0.9})
        result = cache.get("prompt", "input", "output")
        assert result is not None
        assert result["score"] == 0.9
        assert cache.stats.exact_hits == 1

    def test_exact_miss(self):
        cache = SemanticCache()
        cache.put("prompt", "input", "output", {"score": 0.9})
        result = cache.get("prompt", "different", "output")
        assert result is None
        assert cache.stats.misses == 1

    def test_fuzzy_hit(self):
        cache = SemanticCache(fuzzy_threshold=0.8)
        cache.put("prompt", "hello world today", "response here", {"score": 0.9})
        # Very similar input (1 word different)
        result = cache.get("prompt", "hello world today!", "response here")
        assert result is not None
        assert cache.stats.fuzzy_hits == 1

    def test_fuzzy_miss_below_threshold(self):
        cache = SemanticCache(fuzzy_threshold=0.99)
        cache.put("prompt", "completely different text", "output", {"score": 0.9})
        result = cache.get("prompt", "totally unrelated input", "result")
        assert result is None

    def test_fuzzy_disabled(self):
        cache = SemanticCache(fuzzy_enabled=False)
        cache.put("prompt", "hello world", "out", {"score": 0.9})
        result = cache.get("prompt", "hello world!", "out")
        assert result is None  # exact miss, fuzzy disabled

    def test_size(self):
        cache = SemanticCache()
        assert cache.size == 0
        cache.put("p", "i1", "o1", {"a": 1})
        cache.put("p", "i2", "o2", {"a": 2})
        assert cache.size == 2

    def test_clear(self):
        cache = SemanticCache()
        cache.put("p", "i", "o", {"a": 1})
        cache.clear()
        assert cache.size == 0

    def test_eviction(self):
        cache = SemanticCache(max_entries=2)
        cache.put("p", "i1", "o1", {"a": 1})
        cache.put("p", "i2", "o2", {"a": 2})
        cache.put("p", "i3", "o3", {"a": 3})
        assert cache.size == 2  # one evicted

    def test_stats_hit_rate(self):
        cache = SemanticCache()
        cache.put("p", "i", "o", {"a": 1})
        cache.get("p", "i", "o")  # hit
        cache.get("p", "miss", "o")  # miss
        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_stats_api_calls_saved(self):
        cache = SemanticCache()
        cache.put("p", "i", "o", {"a": 1})
        cache.get("p", "i", "o")
        cache.get("p", "i", "o")
        assert cache.stats.api_calls_saved == 2

    def test_stats_as_dict(self):
        cache = SemanticCache()
        d = cache.stats.as_dict()
        assert "exact_hits" in d
        assert "fuzzy_hits" in d
        assert "hit_rate" in d


# =============================================================================
# Warmup averaging tests
# =============================================================================

class TestAverageResults:
    def test_basic_averaging(self):
        samples = [
            [_result("m", "i1", 0.7, True)],
            [_result("m", "i1", 0.8, True)],
            [_result("m", "i1", 0.9, True)],
        ]
        avg = average_results(samples, "m", "i1")
        assert avg.mean_score == pytest.approx(0.8, abs=0.01)
        assert avg.samples == 3
        assert avg.majority_passed is True
        assert avg.passed_votes == 3

    def test_mixed_votes(self):
        samples = [
            [_result("m", "i1", 0.6, True)],
            [_result("m", "i1", 0.3, False)],
            [_result("m", "i1", 0.7, True)],
        ]
        avg = average_results(samples, "m", "i1")
        assert avg.majority_passed is True
        assert avg.passed_votes == 2
        assert avg.failed_votes == 1
        assert avg.agreement_rate == pytest.approx(2/3)

    def test_all_fail(self):
        samples = [
            [_result("m", "i1", 0.1, False)],
            [_result("m", "i1", 0.2, False)],
        ]
        avg = average_results(samples, "m", "i1")
        assert avg.majority_passed is False
        assert avg.agreement_rate == 1.0

    def test_high_variance(self):
        samples = [
            [_result("m", "i1", 0.1, False)],
            [_result("m", "i1", 0.9, True)],
            [_result("m", "i1", 0.2, False)],
        ]
        avg = average_results(samples, "m", "i1")
        assert avg.high_variance is True

    def test_low_variance(self):
        samples = [
            [_result("m", "i1", 0.80, True)],
            [_result("m", "i1", 0.82, True)],
            [_result("m", "i1", 0.79, True)],
        ]
        avg = average_results(samples, "m", "i1")
        assert avg.high_variance is False

    def test_empty_samples(self):
        avg = average_results([], "m", "i1")
        assert avg.mean_score == 0.0
        assert avg.agreement_rate == 0.0

    def test_as_dict(self):
        samples = [[_result("m", "i1", 0.8)]]
        d = average_results(samples, "m", "i1").as_dict()
        assert "mean_score" in d
        assert "agreement_rate" in d
        assert "high_variance" in d


class TestWarmupAveragingReport:
    def test_basic_report(self):
        results = [
            AveragedResult("m1", "i1", 3, [0.8, 0.85, 0.82], 3, 0),
            AveragedResult("m1", "i2", 3, [0.1, 0.9, 0.3], 1, 2),
        ]
        report = WarmupAveragingReport(results=results)
        assert report.high_variance_count >= 1
        assert report.avg_agreement > 0

    def test_empty_report(self):
        report = WarmupAveragingReport()
        assert report.high_variance_count == 0
        assert report.avg_agreement == 0.0

    def test_as_dict(self):
        report = WarmupAveragingReport(results=[
            AveragedResult("m", "i", 2, [0.8, 0.9], 2, 0),
        ])
        d = report.as_dict()
        assert "total_evaluations" in d
        assert "high_variance_count" in d

    def test_format_text(self):
        report = WarmupAveragingReport(results=[
            AveragedResult("m", "i", 3, [0.1, 0.9, 0.3], 1, 2),
        ])
        text = report.format_text()
        assert "Warmup Averaging" in text
