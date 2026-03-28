"""Tests for time series decomposition."""

import pytest
from evalyn_sdk.analysis.time_series import (
    DecompositionResult,
    TimeSeriesReport,
    decompose_metric,
    decompose_all,
    _moving_average,
    _linear_slope,
    _std,
)


# =============================================================================
# DecompositionResult
# =============================================================================

class TestDecompositionResult:
    def test_defaults(self):
        r = DecompositionResult(metric_id="m1")
        assert r.trend_direction == "stable"
        assert r.trend_slope == 0.0
        assert r.noise_level == 0.0
        assert r.values == []

    def test_as_dict(self):
        r = DecompositionResult(
            metric_id="help",
            values=[0.8, 0.9, 0.85],
            trend_direction="improving",
            trend_slope=0.025,
            noise_level=0.031,
        )
        d = r.as_dict()
        assert d["metric_id"] == "help"
        assert d["num_points"] == 3
        assert d["trend_direction"] == "improving"
        assert d["trend_slope"] == 0.025
        assert d["noise_level"] == 0.031

    def test_format_text(self):
        r = DecompositionResult(
            metric_id="safety",
            values=[0.9] * 5,
            trend_direction="stable",
            trend_slope=0.001,
            noise_level=0.02,
        )
        text = r.format_text()
        assert "safety" in text
        assert "stable" in text
        assert "n=5" in text

    def test_from_dict(self):
        d = {"metric_id": "x", "trend_direction": "improving", "trend_slope": 0.05, "noise_level": 0.03}
        r = DecompositionResult.from_dict(d)
        assert r.metric_id == "x"
        assert r.trend_direction == "improving"
        assert r.trend_slope == 0.05
        assert r.noise_level == 0.03
        # Raw arrays not restored from summary
        assert r.values == []
        assert r.trend == []

    def test_from_dict_defaults(self):
        r = DecompositionResult.from_dict({"metric_id": "m"})
        assert r.trend_direction == "stable"
        assert r.trend_slope == 0.0


# =============================================================================
# TimeSeriesReport
# =============================================================================

class TestTimeSeriesReport:
    def _make_result(self, mid, direction, slope=0.0, noise=0.0):
        return DecompositionResult(
            metric_id=mid,
            values=[0.5] * 5,
            trend_direction=direction,
            trend_slope=slope,
            noise_level=noise,
        )

    def test_improving_metrics(self):
        report = TimeSeriesReport(results=[
            self._make_result("a", "improving"),
            self._make_result("b", "stable"),
            self._make_result("c", "improving"),
        ])
        assert report.improving_metrics == ["a", "c"]

    def test_declining_metrics(self):
        report = TimeSeriesReport(results=[
            self._make_result("a", "declining"),
            self._make_result("b", "stable"),
        ])
        assert report.declining_metrics == ["a"]

    def test_noisy_metrics(self):
        report = TimeSeriesReport(results=[
            self._make_result("a", "stable", noise=0.05),
            self._make_result("b", "stable", noise=0.15),
            self._make_result("c", "stable", noise=0.2),
        ])
        assert report.noisy_metrics == ["b", "c"]

    def test_as_dict(self):
        report = TimeSeriesReport(results=[
            self._make_result("a", "improving"),
        ])
        d = report.as_dict()
        assert "improving" in d
        assert "declining" in d
        assert "noisy" in d
        assert len(d["results"]) == 1

    def test_format_text_empty(self):
        report = TimeSeriesReport(results=[])
        text = report.format_text()
        assert "Time Series Analysis" in text

    def test_format_text_with_declining(self):
        report = TimeSeriesReport(results=[
            self._make_result("help", "declining", slope=-0.05),
        ])
        text = report.format_text()
        assert "Declining" in text
        assert "help" in text

    def test_format_text_with_noisy(self):
        report = TimeSeriesReport(results=[
            self._make_result("x", "stable", noise=0.2),
        ])
        text = report.format_text()
        assert "Noisy" in text

    def test_from_dict(self):
        report = TimeSeriesReport(results=[
            self._make_result("a", "improving", slope=0.05),
            self._make_result("b", "stable"),
        ])
        d = report.as_dict()
        restored = TimeSeriesReport.from_dict(d)
        assert len(restored.results) == 2
        assert restored.results[0].metric_id == "a"
        assert restored.results[0].trend_direction == "improving"


# =============================================================================
# decompose_metric
# =============================================================================

class TestDecomposeMetric:
    def test_insufficient_data(self):
        r = decompose_metric("m1", [0.5, 0.6])
        assert r.trend_direction == "insufficient_data"
        assert r.values == [0.5, 0.6]

    def test_single_value(self):
        r = decompose_metric("m1", [0.9])
        assert r.trend_direction == "insufficient_data"

    def test_empty(self):
        r = decompose_metric("m1", [])
        assert r.trend_direction == "insufficient_data"

    def test_improving_trend(self):
        # Steadily increasing values
        values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        r = decompose_metric("up", values)
        assert r.trend_direction == "improving"
        assert r.trend_slope > 0

    def test_declining_trend(self):
        # Steadily decreasing values
        values = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        r = decompose_metric("down", values)
        assert r.trend_direction == "declining"
        assert r.trend_slope < 0

    def test_stable_trend(self):
        # Flat values
        values = [0.8, 0.8, 0.8, 0.8, 0.8]
        r = decompose_metric("flat", values)
        assert r.trend_direction == "stable"
        assert abs(r.trend_slope) < 0.01

    def test_noisy_stable(self):
        # Oscillating but flat on average
        values = [0.7, 0.9, 0.7, 0.9, 0.7, 0.9]
        r = decompose_metric("noisy", values)
        assert r.noise_level > 0
        assert len(r.residual) == len(values)

    def test_residuals_sum_close_to_zero(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        r = decompose_metric("m", values)
        # Residuals should roughly center around zero
        avg_residual = sum(r.residual) / len(r.residual)
        assert abs(avg_residual) < 0.1

    def test_trend_length_matches_values(self):
        values = [0.5, 0.6, 0.7, 0.8]
        r = decompose_metric("m", values)
        assert len(r.trend) == len(values)
        assert len(r.residual) == len(values)

    def test_custom_slope_threshold(self):
        # Small increase that passes default threshold but not a higher one
        values = [0.8, 0.81, 0.82, 0.83, 0.84]
        r1 = decompose_metric("m", values, slope_threshold=0.001)
        r2 = decompose_metric("m", values, slope_threshold=0.1)
        assert r1.trend_direction == "improving"
        assert r2.trend_direction == "stable"

    def test_custom_window(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        r1 = decompose_metric("m", values, window=3)
        r2 = decompose_metric("m", values, window=5)
        # Both should detect improving trend
        assert r1.trend_direction == "improving"
        assert r2.trend_direction == "improving"
        # Different smoothing yields different noise levels
        assert r1.noise_level != r2.noise_level or r1.trend != r2.trend


# =============================================================================
# decompose_all
# =============================================================================

class TestDecomposeAll:
    def test_multiple_metrics(self):
        series = {
            "help": [0.5, 0.6, 0.7, 0.8, 0.9],
            "safety": [0.9, 0.9, 0.9, 0.9, 0.9],
        }
        report = decompose_all(series)
        assert len(report.results) == 2
        # Sorted by metric_id
        assert report.results[0].metric_id == "help"
        assert report.results[1].metric_id == "safety"

    def test_empty(self):
        report = decompose_all({})
        assert len(report.results) == 0

    def test_single_metric(self):
        report = decompose_all({"only": [0.5, 0.6, 0.7]})
        assert len(report.results) == 1

    def test_custom_window(self):
        series = {"m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        report = decompose_all(series, window=5)
        assert report.results[0].trend_direction == "improving"


# =============================================================================
# Internal helpers
# =============================================================================

class TestMovingAverage:
    def test_basic(self):
        result = _moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert len(result) == 5
        # Center point should be exact
        assert result[2] == pytest.approx(3.0)

    def test_window_larger_than_values(self):
        result = _moving_average([1.0, 2.0], window=5)
        assert result == [1.5, 1.5]

    def test_single_value(self):
        result = _moving_average([5.0], window=3)
        assert result == [5.0]

    def test_edges_use_available_data(self):
        # With 5 values and window=3, edges use partial windows
        result = _moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        # First element: avg of [1,2] = 1.5
        assert result[0] == pytest.approx(1.5)
        # Last element: avg of [4,5] = 4.5
        assert result[4] == pytest.approx(4.5)

    def test_n_equals_window_returns_global_avg(self):
        # When n <= window, all values get the global average
        result = _moving_average([1.0, 2.0, 3.0], window=3)
        assert result == [pytest.approx(2.0)] * 3

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="odd"):
            _moving_average([1.0, 2.0, 3.0, 4.0], window=4)


class TestLinearSlope:
    def test_increasing(self):
        slope = _linear_slope([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)

    def test_flat(self):
        slope = _linear_slope([5.0, 5.0, 5.0])
        assert slope == pytest.approx(0.0)

    def test_decreasing(self):
        slope = _linear_slope([4.0, 3.0, 2.0, 1.0])
        assert slope == pytest.approx(-1.0)

    def test_single_value(self):
        assert _linear_slope([1.0]) == 0.0

    def test_empty(self):
        assert _linear_slope([]) == 0.0


class TestStd:
    def test_zero_variance(self):
        assert _std([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_known_std(self):
        # std of [0, 2] with population formula = 1.0
        assert _std([0.0, 2.0]) == pytest.approx(1.0)

    def test_single_value(self):
        assert _std([1.0]) == 0.0

    def test_empty(self):
        assert _std([]) == 0.0
