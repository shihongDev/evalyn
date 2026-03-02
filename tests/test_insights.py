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
    detect_regressions,
    analyze_input_features,
    analyze_score_distributions,
    generate_recommendations,
    CorrelationResult,
    RegressionAlert,
    FeatureInsight,
    DistributionInsight,
    Recommendation,
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


# ---------------------------------------------------------------------------
# Helpers for regression / distribution tests
# ---------------------------------------------------------------------------


def _make_run_with_pass_rates(rates: dict[str, float], n_items: int = 10) -> RunAnalysis:
    """Build RunAnalysis from {metric_id: pass_rate}. Creates n_items per metric."""
    items: dict[str, dict[str, dict]] = {}
    for i in range(n_items):
        item_metrics: dict[str, dict] = {}
        for metric_id, rate in rates.items():
            passed = i < int(rate * n_items)
            item_metrics[metric_id] = {"passed": passed, "score": 1.0 if passed else 0.0}
        items[f"item_{i}"] = item_metrics
    return _make_run_analysis(items)


def _make_run_from_scores(metric_scores: dict[str, list[float]]) -> RunAnalysis:
    """Build RunAnalysis from {metric_id: [score_list]}."""
    max_items = max(len(scores) for scores in metric_scores.values())
    items: dict[str, dict[str, dict]] = {}
    for i in range(max_items):
        item_metrics: dict[str, dict] = {}
        for metric_id, scores in metric_scores.items():
            if i < len(scores):
                s = scores[i]
                item_metrics[metric_id] = {"passed": s >= 0.5, "score": s}
        items[f"item_{i}"] = item_metrics
    return _make_run_analysis(items)


# ---------------------------------------------------------------------------
# Regression Detection Tests
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    def test_critical_regression(self):
        """Pass rate drop > 15% should be critical."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.9})
        current = _make_run_with_pass_rates({"helpfulness": 0.7})
        alerts = detect_regressions(current, previous)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].metric_id == "helpfulness"
        assert alerts[0].delta < 0

    def test_warning_regression(self):
        """Pass rate drop 5-15% should be warning."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.9})
        current = _make_run_with_pass_rates({"helpfulness": 0.8})
        alerts = detect_regressions(current, previous)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_info_regression(self):
        """Pass rate drop < 5% should be info."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.9}, n_items=100)
        current = _make_run_with_pass_rates({"helpfulness": 0.87}, n_items=100)
        alerts = detect_regressions(current, previous)
        assert len(alerts) == 1
        assert alerts[0].severity == "info"

    def test_no_regression_on_improvement(self):
        """Improvement should produce no alerts."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.7})
        current = _make_run_with_pass_rates({"helpfulness": 0.9})
        alerts = detect_regressions(current, previous)
        assert alerts == []

    def test_no_regression_on_no_change(self):
        """No change should produce no alerts."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.8})
        current = _make_run_with_pass_rates({"helpfulness": 0.8})
        alerts = detect_regressions(current, previous)
        assert alerts == []

    def test_new_metric_ignored(self):
        """Metric only in current run should not produce alert."""
        previous = _make_run_with_pass_rates({"helpfulness": 0.9})
        current = _make_run_with_pass_rates({"helpfulness": 0.9, "new_metric": 0.5})
        alerts = detect_regressions(current, previous)
        assert alerts == []

    def test_multiple_regressions_sorted(self):
        """Multiple regressions sorted by severity, then delta."""
        previous = _make_run_with_pass_rates({
            "metric_a": 0.9,
            "metric_b": 0.8,
            "metric_c": 0.9,
        })
        current = _make_run_with_pass_rates({
            "metric_a": 0.7,
            "metric_b": 0.7,
            "metric_c": 0.6,
        })
        alerts = detect_regressions(current, previous)
        assert len(alerts) == 3
        assert alerts[0].severity == "critical"
        assert alerts[1].severity == "critical"
        assert alerts[2].severity == "warning"
        assert alerts[0].delta <= alerts[1].delta


# ---------------------------------------------------------------------------
# Input Feature Analysis Tests
# ---------------------------------------------------------------------------


class TestInputFeatureAnalysis:
    def test_long_inputs_fail_more(self):
        """Items with longer inputs that fail should produce insight."""
        run_items = {}
        dataset_items = []
        for i in range(20):
            item_id = f"item_{i}"
            is_long = i >= 10
            passed = not is_long
            run_items[item_id] = {
                "quality": {"passed": passed, "score": 1.0 if passed else 0.0}
            }
            input_text = "x" * (500 if is_long else 50)
            dataset_items.append({
                "id": item_id,
                "inputs": {"query": input_text},
                "output": "response",
            })
        run = _make_run_analysis(run_items)
        results = analyze_input_features(dataset_items, run)
        assert len(results) >= 1
        input_insight = [r for r in results if r.feature_name == "input_length"]
        assert len(input_insight) == 1
        assert "fail more" in input_insight[0].finding

    def test_no_significant_difference(self):
        """Similar pass rates across groups should produce no insight."""
        run_items = {}
        dataset_items = []
        for i in range(20):
            item_id = f"item_{i}"
            run_items[item_id] = {"quality": {"passed": True, "score": 1.0}}
            dataset_items.append({
                "id": item_id,
                "inputs": {"query": "x" * (i * 10 + 10)},
                "output": "response",
            })
        run = _make_run_analysis(run_items)
        results = analyze_input_features(dataset_items, run)
        assert results == []

    def test_empty_dataset(self):
        """Empty dataset should return empty result."""
        run = _make_run_analysis({})
        results = analyze_input_features([], run)
        assert results == []

    def test_string_inputs(self):
        """String inputs (not dict) should be handled."""
        run_items = {}
        dataset_items = []
        for i in range(20):
            item_id = f"item_{i}"
            is_long = i >= 10
            passed = not is_long
            run_items[item_id] = {
                "quality": {"passed": passed, "score": 1.0 if passed else 0.0}
            }
            dataset_items.append({
                "id": item_id,
                "input": "x" * (500 if is_long else 50),
                "output": "response",
            })
        run = _make_run_analysis(run_items)
        results = analyze_input_features(dataset_items, run)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Score Distribution Tests
# ---------------------------------------------------------------------------


class TestScoreDistributions:
    def test_cliff_at_one(self):
        """All scores at 1.0 should be detected as cliff."""
        run = _make_run_from_scores({"metric_a": [1.0] * 10})
        results = analyze_score_distributions(run)
        assert len(results) == 1
        assert results[0].shape == "cliff"
        assert "lenient" in results[0].finding

    def test_cliff_at_zero(self):
        """All scores at 0.0 should be detected as cliff."""
        run = _make_run_from_scores({"metric_a": [0.0] * 10})
        results = analyze_score_distributions(run)
        assert len(results) == 1
        assert results[0].shape == "cliff"
        assert "0.0" in results[0].finding

    def test_bimodal_distribution(self):
        """Scores clustered at low and high should be bimodal."""
        run = _make_run_from_scores({"metric_a": [0.1] * 5 + [0.9] * 5})
        results = analyze_score_distributions(run)
        assert len(results) == 1
        assert results[0].shape == "bimodal"

    def test_skewed_low(self):
        """Low mean with variance should be skewed_low."""
        scores = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.1, 0.05]
        run = _make_run_from_scores({"metric_a": scores})
        results = analyze_score_distributions(run)
        assert len(results) == 1
        assert results[0].shape == "skewed_low"

    def test_skewed_high(self):
        """High mean with low variance should be skewed_high."""
        scores = [0.85, 0.9, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89, 0.93, 0.88]
        run = _make_run_from_scores({"metric_a": scores})
        results = analyze_score_distributions(run)
        assert len(results) == 1
        assert results[0].shape == "skewed_high"

    def test_normal_not_reported(self):
        """Normal distribution should not appear in results."""
        scores = [0.4, 0.45, 0.5, 0.55, 0.6, 0.42, 0.48, 0.52, 0.58, 0.5]
        run = _make_run_from_scores({"metric_a": scores})
        results = analyze_score_distributions(run)
        assert results == []

    def test_insufficient_data(self):
        """Fewer than 5 scores should return empty result."""
        run = _make_run_from_scores({"metric_a": [0.0, 0.0, 0.0]})
        results = analyze_score_distributions(run)
        assert results == []


# ---------------------------------------------------------------------------
# Recommendations Engine Tests
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_critical_regression_is_priority_1(self):
        """Critical regressions get highest priority."""
        run = _make_run_with_pass_rates({"helpfulness": 0.7})
        regressions = [RegressionAlert(
            metric_id="helpfulness",
            previous_pass_rate=0.9,
            current_pass_rate=0.7,
            delta=-0.2,
            severity="critical",
        )]
        recs = generate_recommendations(run, [], regressions, [], [])
        assert len(recs) >= 1
        assert recs[0].category == "regression"
        assert recs[0].priority == 1

    def test_redundant_metric_recommendation(self):
        """Redundant metrics should produce remove suggestion."""
        run = _make_run_with_pass_rates({"a": 0.9, "b": 0.9})
        corrs = [CorrelationResult(
            metric_a="a", metric_b="b", pearson=0.95, relationship="redundant",
        )]
        recs = generate_recommendations(run, corrs, [], [], [])
        assert any("removing" in r.message.lower() for r in recs)

    def test_empty_inputs_produce_no_crash(self):
        """Empty inputs should produce empty or minimal recommendations."""
        run = _make_run_with_pass_rates({"a": 0.95})
        recs = generate_recommendations(run, [], [], [], [])
        # Should not crash; may produce 0 or few recs
        assert isinstance(recs, list)

    def test_recommendations_sorted_by_priority(self):
        """Output should be sorted by priority ascending."""
        run = _make_run_with_pass_rates({"a": 0.7})
        regressions = [RegressionAlert(
            metric_id="a", previous_pass_rate=0.9,
            current_pass_rate=0.7, delta=-0.2, severity="critical",
        )]
        corrs = [CorrelationResult(
            metric_a="a", metric_b="b", pearson=0.95, relationship="redundant",
        )]
        recs = generate_recommendations(run, corrs, regressions, [], [])
        for i in range(len(recs) - 1):
            assert recs[i].priority <= recs[i + 1].priority
