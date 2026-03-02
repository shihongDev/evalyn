"""Tests for the insights engine."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.core import RunAnalysis, MetricStats, ItemStats
from evalyn_sdk.analysis.insights import (
    compute_metric_correlations,
    CorrelationResult,
)


def _make_run_analysis(
    items: dict[str, dict[str, dict]],
    metric_ids: list[str] | None = None,
) -> RunAnalysis:
    """Helper: build RunAnalysis from {item_id: {metric_id: {passed, score}}} dict."""
    item_stats = {}
    metric_stats_map: dict[str, MetricStats] = {}

    for item_id, metrics in items.items():
        ist = ItemStats(item_id=item_id)
        for metric_id, result in metrics.items():
            ist.metric_results[metric_id] = {
                "passed": result.get("passed", True),
                "score": result.get("score", 1.0),
                "reason": None,
                "details": {},
            }
            if result.get("passed", True):
                ist.metrics_passed += 1
            else:
                ist.metrics_failed += 1

            if metric_id not in metric_stats_map:
                metric_stats_map[metric_id] = MetricStats(
                    metric_id=metric_id, metric_type="objective"
                )
            ms = metric_stats_map[metric_id]
            ms.count += 1
            score = result.get("score", 1.0)
            if score is not None:
                ms.scores.append(score)
            if result.get("passed", True):
                ms.passed += 1
            else:
                ms.failed += 1
            ms.has_pass_fail = True

        item_stats[item_id] = ist

    failed_items = [iid for iid, ist in item_stats.items() if not ist.all_passed]

    return RunAnalysis(
        run_id="test-run",
        dataset_name="test",
        created_at="2026-01-01",
        total_items=len(item_stats),
        total_metrics=len(metric_stats_map),
        metric_stats=metric_stats_map,
        item_stats=item_stats,
        failed_items=failed_items,
    )


class TestMetricCorrelations:
    def test_perfectly_correlated_metrics(self):
        """Two metrics with identical scores should be classified as redundant."""
        items = {
            f"item_{i}": {
                "metric_a": {"score": float(i) / 10, "passed": True},
                "metric_b": {"score": float(i) / 10, "passed": True},
            }
            for i in range(10)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert len(results) == 1
        assert results[0].relationship == "redundant"
        assert results[0].pearson > 0.99

    def test_anticorrelated_metrics(self):
        """Two metrics with inverse scores should be classified as tradeoff."""
        items = {
            f"item_{i}": {
                "metric_a": {"score": float(i) / 10, "passed": True},
                "metric_b": {"score": 1.0 - float(i) / 10, "passed": True},
            }
            for i in range(10)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert len(results) == 1
        assert results[0].relationship == "tradeoff"
        assert results[0].pearson < -0.99

    def test_independent_metrics_filtered(self):
        """Independent metrics (low correlation) should not appear in results."""
        import random
        random.seed(42)
        items = {
            f"item_{i}": {
                "metric_a": {"score": float(i) / 20, "passed": True},
                "metric_b": {"score": random.random(), "passed": True},
            }
            for i in range(20)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        # Should be empty or only contain pairs with |r| > 0.5
        for r in results:
            assert abs(r.pearson) > 0.5

    def test_insufficient_data(self):
        """Fewer than 5 common items should return empty result."""
        items = {
            f"item_{i}": {
                "metric_a": {"score": float(i) / 10, "passed": True},
                "metric_b": {"score": float(i) / 10, "passed": True},
            }
            for i in range(3)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert results == []

    def test_single_metric(self):
        """Single metric should return empty result (no pairs)."""
        items = {
            f"item_{i}": {
                "metric_a": {"score": float(i) / 10, "passed": True},
            }
            for i in range(10)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert results == []

    def test_constant_scores_handled(self):
        """Metrics with constant scores (std=0) should not crash."""
        items = {
            f"item_{i}": {
                "metric_a": {"score": 1.0, "passed": True},
                "metric_b": {"score": float(i) / 10, "passed": True},
            }
            for i in range(10)
        }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert results == []  # r is undefined when one metric is constant

    def test_results_sorted_by_abs_pearson(self):
        """Results should be sorted by absolute Pearson value descending."""
        # Create 3 metrics: a perfectly correlated with b, and moderately with c
        items = {}
        for i in range(15):
            items[f"item_{i}"] = {
                "metric_a": {"score": float(i) / 15, "passed": True},
                "metric_b": {"score": float(i) / 15, "passed": True},  # perfect correlation
                "metric_c": {"score": 1.0 - float(i) / 15, "passed": True},  # perfect anti-correlation
            }
        run = _make_run_analysis(items)
        results = compute_metric_correlations(run)
        assert len(results) >= 2
        # All results should be sorted by abs(pearson) desc
        for i in range(len(results) - 1):
            assert abs(results[i].pearson) >= abs(results[i + 1].pearson)
