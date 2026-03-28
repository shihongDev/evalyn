"""Tests for latency_optimization module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.latency_optimization import (
    LatencyProfile,
    LatencyReport,
    OptimizationSuggestion,
    analyze_latency,
    build_latency_profile,
    compute_percentile,
    estimate_optimization_impact,
    rank_metrics_by_latency,
)


# --- compute_percentile ---


def test_compute_percentile_basic():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_percentile(values, 50) == 30.0


def test_compute_percentile_p0():
    values = [10.0, 20.0, 30.0]
    assert compute_percentile(values, 0) == 10.0


def test_compute_percentile_p100():
    values = [10.0, 20.0, 30.0]
    assert compute_percentile(values, 100) == 30.0


def test_compute_percentile_single_value():
    assert compute_percentile([42.0], 50) == 42.0
    assert compute_percentile([42.0], 99) == 42.0


def test_compute_percentile_empty():
    assert compute_percentile([], 50) == 0.0


def test_compute_percentile_two_values():
    values = [10.0, 20.0]
    assert compute_percentile(values, 50) == 15.0


def test_compute_percentile_unsorted_input():
    values = [50.0, 10.0, 30.0, 20.0, 40.0]
    assert compute_percentile(values, 50) == 30.0


def test_compute_percentile_interpolation():
    values = [0.0, 100.0]
    result = compute_percentile(values, 25)
    assert result == pytest.approx(25.0)


# --- build_latency_profile ---


def test_build_latency_profile_correct_stats():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    profile = build_latency_profile("metric_a", latencies)
    assert profile.metric_id == "metric_a"
    assert profile.avg_latency_ms == pytest.approx(300.0)
    assert profile.p50_ms == pytest.approx(300.0)
    assert profile.call_count == 5


def test_build_latency_profile_single_value():
    profile = build_latency_profile("m1", [42.0])
    assert profile.avg_latency_ms == pytest.approx(42.0)
    assert profile.p50_ms == pytest.approx(42.0)
    assert profile.p95_ms == pytest.approx(42.0)
    assert profile.p99_ms == pytest.approx(42.0)
    assert profile.call_count == 1


def test_build_latency_profile_empty():
    profile = build_latency_profile("empty", [])
    assert profile.avg_latency_ms == 0.0
    assert profile.p50_ms == 0.0
    assert profile.call_count == 0


def test_build_latency_profile_two_values():
    profile = build_latency_profile("m2", [10.0, 20.0])
    assert profile.avg_latency_ms == pytest.approx(15.0)
    assert profile.call_count == 2


# --- analyze_latency ---


def test_analyze_latency_total_counts():
    p1 = build_latency_profile("a", [100.0] * 5)
    p2 = build_latency_profile("b", [200.0] * 6)
    report = analyze_latency([p1, p2])
    assert report.total_calls == 11
    assert report.total_latency_ms == pytest.approx(100.0 * 5 + 200.0 * 6)


def test_analyze_latency_batch_suggestion():
    profiles = [build_latency_profile("m", [100.0] * 10)]
    report = analyze_latency(profiles)
    texts = [s.suggestion for s in report.suggestions]
    assert any("batch" in t.lower() or "Batch" in t for t in texts)


def test_analyze_latency_cheap_model_suggestion():
    profile = LatencyProfile(
        metric_id="slow", avg_latency_ms=6000.0, p50_ms=5500.0,
        p95_ms=7000.0, p99_ms=8000.0, call_count=3,
    )
    report = analyze_latency([profile])
    texts = [s.suggestion for s in report.suggestions]
    assert any("cheaper model" in t.lower() for t in texts)


def test_analyze_latency_caching_suggestion():
    profiles = [build_latency_profile("m", [100.0] * 5)]
    report = analyze_latency(profiles)
    texts = [s.suggestion for s in report.suggestions]
    assert any("caching" in t.lower() for t in texts)


def test_analyze_latency_empty():
    report = analyze_latency([])
    assert report.total_calls == 0
    assert report.total_latency_ms == 0.0
    assert report.suggestions == []


def test_analyze_latency_no_suggestions_for_few_calls():
    profiles = [build_latency_profile("m", [100.0, 200.0])]
    report = analyze_latency(profiles)
    # 2 calls - not enough for batch or cache suggestions
    texts = [s.suggestion for s in report.suggestions]
    assert not any("batch" in t.lower() for t in texts)


# --- estimate_optimization_impact ---


def test_estimate_optimization_caching():
    p = LatencyProfile(metric_id="x")
    assert estimate_optimization_impact(p, "caching") == pytest.approx(0.50)


def test_estimate_optimization_batching():
    p = LatencyProfile(metric_id="x")
    assert estimate_optimization_impact(p, "batching") == pytest.approx(0.30)


def test_estimate_optimization_cheaper_model():
    p = LatencyProfile(metric_id="x")
    assert estimate_optimization_impact(p, "cheaper_model") == pytest.approx(0.40)


def test_estimate_optimization_unknown():
    p = LatencyProfile(metric_id="x")
    assert estimate_optimization_impact(p, "magic") == 0.0


# --- rank_metrics_by_latency ---


def test_rank_metrics_by_latency_sorted():
    p1 = LatencyProfile(metric_id="fast", avg_latency_ms=100.0)
    p2 = LatencyProfile(metric_id="slow", avg_latency_ms=500.0)
    p3 = LatencyProfile(metric_id="mid", avg_latency_ms=300.0)
    ranked = rank_metrics_by_latency([p1, p2, p3])
    assert [p.metric_id for p in ranked] == ["slow", "mid", "fast"]


def test_rank_metrics_by_latency_empty():
    assert rank_metrics_by_latency([]) == []


def test_rank_metrics_by_latency_single():
    p = LatencyProfile(metric_id="only", avg_latency_ms=42.0)
    assert rank_metrics_by_latency([p]) == [p]


# --- LatencyProfile as_dict / from_dict ---


def test_latency_profile_as_dict():
    p = LatencyProfile(
        metric_id="m1", avg_latency_ms=100.0, p50_ms=90.0,
        p95_ms=200.0, p99_ms=300.0, call_count=10,
    )
    d = p.as_dict()
    assert d["metric_id"] == "m1"
    assert d["avg_latency_ms"] == 100.0
    assert d["call_count"] == 10


def test_latency_profile_from_dict():
    d = {
        "metric_id": "m2",
        "avg_latency_ms": 150.0,
        "p50_ms": 140.0,
        "p95_ms": 250.0,
        "p99_ms": 350.0,
        "call_count": 20,
    }
    p = LatencyProfile.from_dict(d)
    assert p.metric_id == "m2"
    assert p.avg_latency_ms == 150.0
    assert p.call_count == 20


def test_latency_profile_roundtrip():
    p = LatencyProfile(
        metric_id="rt", avg_latency_ms=77.0, p50_ms=70.0,
        p95_ms=100.0, p99_ms=120.0, call_count=5,
    )
    assert LatencyProfile.from_dict(p.as_dict()).metric_id == "rt"


# --- LatencyReport format_text ---


def test_latency_report_format_text():
    p = LatencyProfile(
        metric_id="m1", avg_latency_ms=100.0, p50_ms=90.0,
        p95_ms=200.0, p99_ms=300.0, call_count=10,
    )
    s = OptimizationSuggestion(
        suggestion="Enable caching", estimated_savings_pct=50.0, priority="medium",
    )
    report = LatencyReport(
        profiles=[p], suggestions=[s], total_calls=10, total_latency_ms=1000.0,
    )
    text = report.format_text()
    assert "Latency Report" in text
    assert "m1" in text
    assert "Enable caching" in text
    assert "Total calls: 10" in text


def test_latency_report_format_text_empty():
    report = LatencyReport()
    text = report.format_text()
    assert "Latency Report" in text
    assert "Total calls: 0" in text


# --- LatencyReport as_dict / from_dict ---


def test_latency_report_as_dict():
    report = LatencyReport(total_calls=5, total_latency_ms=500.0)
    d = report.as_dict()
    assert d["total_calls"] == 5
    assert d["profiles"] == []


def test_latency_report_from_dict():
    d = {
        "profiles": [{"metric_id": "x", "avg_latency_ms": 10.0}],
        "total_calls": 3,
        "total_latency_ms": 30.0,
    }
    report = LatencyReport.from_dict(d)
    assert report.total_calls == 3
    assert len(report.profiles) == 1
    assert report.profiles[0].metric_id == "x"


# --- OptimizationSuggestion as_dict ---


def test_optimization_suggestion_as_dict():
    s = OptimizationSuggestion(
        suggestion="Do something", estimated_savings_pct=25.0, priority="low",
    )
    d = s.as_dict()
    assert d["suggestion"] == "Do something"
    assert d["priority"] == "low"


# --- Edge cases ---


def test_build_profile_zero_latencies():
    profile = build_latency_profile("zeros", [0.0, 0.0, 0.0])
    assert profile.avg_latency_ms == 0.0
    assert profile.p50_ms == 0.0
    assert profile.call_count == 3


def test_analyze_latency_high_priority_batch():
    profiles = [build_latency_profile("m", [100.0] * 50)]
    report = analyze_latency(profiles)
    batch_suggestions = [
        s for s in report.suggestions if "batch" in s.suggestion.lower()
    ]
    assert len(batch_suggestions) == 1
    assert batch_suggestions[0].priority == "high"
