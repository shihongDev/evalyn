"""Tests for metric compatibility matrix and runtime estimation."""

import pytest
from evalyn_sdk.models import MetricSpec
from evalyn_sdk.metrics.compatibility import (
    build_compatibility_matrix, CompatibilityEntry, CompatibilityMatrix,
)
from evalyn_sdk.analysis.runtime_estimation import (
    estimate_runtime, MetricTimeEstimate, RuntimeEstimateReport,
)


def _spec(mid, mtype="objective", unit_types=None):
    return MetricSpec(id=mid, name=mid, type=mtype,
                     unit_types=unit_types or ["outcome"])


# =============================================================================
# Compatibility matrix tests
# =============================================================================

class TestCompatibilityEntry:
    def test_supports(self):
        e = CompatibilityEntry("m", "objective", ["outcome", "single_turn"])
        assert e.supports("outcome")
        assert e.supports("single_turn")
        assert not e.supports("tool_use")

    def test_as_dict(self):
        e = CompatibilityEntry("m", "objective", ["outcome"])
        d = e.as_dict()
        assert d["metric_id"] == "m"
        assert d["unit_types"] == ["outcome"]


class TestBuildMatrix:
    def test_basic(self):
        specs = [_spec("a"), _spec("b", "subjective", ["outcome", "single_turn"])]
        matrix = build_compatibility_matrix(specs)
        assert len(matrix.entries) == 2

    def test_get_compatible(self):
        specs = [
            _spec("a", unit_types=["outcome"]),
            _spec("b", unit_types=["outcome", "single_turn"]),
            _spec("c", unit_types=["tool_use"]),
        ]
        matrix = build_compatibility_matrix(specs)
        compat = matrix.get_compatible("outcome")
        assert "a" in compat
        assert "b" in compat
        assert "c" not in compat

    def test_get_incompatible(self):
        specs = [_spec("a", unit_types=["outcome"]), _spec("b", unit_types=["tool_use"])]
        matrix = build_compatibility_matrix(specs)
        incompat = matrix.get_incompatible("outcome")
        assert "b" in incompat
        assert "a" not in incompat

    def test_check_selection_all_compatible(self):
        specs = [_spec("a", unit_types=["outcome"]), _spec("b", unit_types=["outcome"])]
        matrix = build_compatibility_matrix(specs)
        result = matrix.check_selection(["a", "b"], "outcome")
        assert result["all_compatible"] is True
        assert len(result["warnings"]) == 0

    def test_check_selection_incompatible(self):
        specs = [_spec("a", unit_types=["outcome"]), _spec("b", unit_types=["tool_use"])]
        matrix = build_compatibility_matrix(specs)
        result = matrix.check_selection(["a", "b"], "outcome")
        assert not result["all_compatible"]
        assert "b" in result["incompatible"]
        assert len(result["warnings"]) == 1

    def test_check_selection_unknown(self):
        matrix = build_compatibility_matrix([_spec("a")])
        result = matrix.check_selection(["a", "unknown_metric"], "outcome")
        assert "unknown_metric" in result["unknown"]

    def test_empty_matrix(self):
        matrix = build_compatibility_matrix([])
        assert len(matrix.entries) == 0
        assert matrix.get_compatible("outcome") == []

    def test_as_dict(self):
        specs = [_spec("a", unit_types=["outcome", "single_turn"])]
        d = build_compatibility_matrix(specs).as_dict()
        assert "matrix" in d
        assert "a" in d["matrix"]
        assert d["matrix"]["a"]["outcome"] is True
        assert d["matrix"]["a"]["tool_use"] is False

    def test_format_text(self):
        specs = [_spec("json_valid"), _spec("help", "subjective")]
        text = build_compatibility_matrix(specs).format_text()
        assert "Compatibility" in text
        assert "json_valid" in text


# =============================================================================
# Runtime estimation tests
# =============================================================================

class TestEstimateRuntime:
    def test_objective_fast(self):
        specs = [_spec("json_valid")]
        report = estimate_runtime(specs, item_count=100)
        assert report.total_sequential_ms < 1000  # < 1 second for 100 items

    def test_subjective_slow(self):
        specs = [_spec("help", "subjective")]
        report = estimate_runtime(specs, item_count=100)
        assert report.total_sequential_ms > 100000  # > 100 seconds

    def test_parallel_reduces_time(self):
        specs = [_spec("help", "subjective")]
        seq = estimate_runtime(specs, item_count=100, max_workers=1)
        par = estimate_runtime(specs, item_count=100, max_workers=4)
        assert par.total_parallel_ms < seq.total_sequential_ms

    def test_historical_times(self):
        specs = [_spec("custom")]
        historical = {"custom": 500.0}  # 500ms per item
        report = estimate_runtime(specs, item_count=10, historical_times=historical)
        assert report.metrics[0].avg_time_ms == 500.0
        assert report.metrics[0].source == "historical"

    def test_default_source(self):
        specs = [_spec("a")]
        report = estimate_runtime(specs, item_count=10)
        assert report.metrics[0].source == "default"

    def test_slowest_metric(self):
        specs = [_spec("fast"), _spec("slow", "subjective")]
        report = estimate_runtime(specs, item_count=100)
        assert report.slowest_metric == "slow"

    def test_empty(self):
        report = estimate_runtime([], item_count=100)
        assert report.total_sequential_ms == 0
        assert report.slowest_metric is None

    def test_zero_items(self):
        specs = [_spec("a", "subjective")]
        report = estimate_runtime(specs, item_count=0)
        assert report.total_sequential_ms == 0

    def test_as_dict(self):
        specs = [_spec("a")]
        d = estimate_runtime(specs, item_count=10).as_dict()
        assert "total_sequential_seconds" in d
        assert "metrics" in d
        assert "slowest_metric" in d

    def test_format_text(self):
        specs = [_spec("a"), _spec("b", "subjective")]
        text = estimate_runtime(specs, item_count=50).format_text()
        assert "Runtime" in text
        assert "Sequential" in text

    def test_metric_estimate_as_dict(self):
        e = MetricTimeEstimate("m", "objective", 5.0, 500.0, "default")
        d = e.as_dict()
        assert d["metric_id"] == "m"
        assert d["avg_time_ms"] == 5.0
