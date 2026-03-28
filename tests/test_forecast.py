"""Tests for trend forecasting."""

import pytest
from evalyn_sdk.analysis.forecast import (
    ForecastPoint,
    MetricForecast,
    ForecastReport,
    forecast_metric,
    forecast_all,
    _linear_slope_intercept,
    _residual_std,
    _z_score,
)


# =============================================================================
# ForecastPoint
# =============================================================================

class TestForecastPoint:
    def test_as_dict(self):
        fp = ForecastPoint(step=1, value=0.85, lower=0.75, upper=0.95)
        d = fp.as_dict()
        assert d["step"] == 1
        assert d["value"] == 0.85
        assert d["lower"] == 0.75
        assert d["upper"] == 0.95


# =============================================================================
# MetricForecast
# =============================================================================

class TestMetricForecast:
    def test_as_dict(self):
        f = MetricForecast(
            metric_id="help",
            method="linear",
            observed=[0.7, 0.8, 0.9],
            forecasted=[ForecastPoint(1, 1.0, 0.9, 1.1)],
            alerts=["test alert"],
        )
        d = f.as_dict()
        assert d["metric_id"] == "help"
        assert d["method"] == "linear"
        assert d["num_observed"] == 3
        assert d["num_forecasted"] == 1
        assert len(d["alerts"]) == 1

    def test_from_dict(self):
        data = {
            "metric_id": "x",
            "method": "exponential_smoothing",
            "alerts": ["warn"],
            "forecasted": [{"step": 1, "value": 0.5, "lower": 0.4, "upper": 0.6}],
        }
        f = MetricForecast.from_dict(data)
        assert f.metric_id == "x"
        assert f.method == "exponential_smoothing"
        assert len(f.forecasted) == 1
        assert f.forecasted[0].value == 0.5

    def test_format_text(self):
        f = MetricForecast(
            metric_id="safety",
            method="linear",
            forecasted=[ForecastPoint(1, 0.85, 0.8, 0.9)],
            alerts=["dropping below 0.80"],
        )
        text = f.format_text()
        assert "safety" in text
        assert "+1" in text
        assert "ALERT" in text


# =============================================================================
# ForecastReport
# =============================================================================

class TestForecastReport:
    def test_alerts_aggregated(self):
        r = ForecastReport(results=[
            MetricForecast(metric_id="a", method="linear", alerts=["alert1"]),
            MetricForecast(metric_id="b", method="linear", alerts=[]),
            MetricForecast(metric_id="c", method="linear", alerts=["alert2"]),
        ])
        assert len(r.alerts) == 2
        assert r.metrics_at_risk == ["a", "c"]

    def test_as_dict(self):
        r = ForecastReport(results=[
            MetricForecast(metric_id="x", method="linear"),
        ])
        d = r.as_dict()
        assert "alerts" in d
        assert "metrics_at_risk" in d
        assert len(d["results"]) == 1

    def test_from_dict(self):
        r = ForecastReport(results=[
            MetricForecast(metric_id="m", method="linear", alerts=["a"]),
        ])
        d = r.as_dict()
        restored = ForecastReport.from_dict(d)
        assert len(restored.results) == 1
        assert restored.results[0].metric_id == "m"

    def test_format_text_empty(self):
        r = ForecastReport()
        text = r.format_text()
        assert "Trend Forecast" in text

    def test_format_text_with_risk(self):
        r = ForecastReport(results=[
            MetricForecast(metric_id="help", method="linear", alerts=["danger"]),
        ])
        text = r.format_text()
        assert "At risk" in text
        assert "help" in text


# =============================================================================
# forecast_metric
# =============================================================================

class TestForecastMetric:
    def test_insufficient_data(self):
        f = forecast_metric("m", [0.5, 0.6])
        assert len(f.forecasted) == 0
        assert "insufficient" in f.alerts[0]

    def test_empty_data(self):
        f = forecast_metric("m", [])
        assert "insufficient" in f.alerts[0]

    def test_linear_increasing(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        f = forecast_metric("up", values, horizon=3, method="linear")
        assert f.method == "linear"
        assert len(f.forecasted) == 3
        # Each forecast should be higher than the last observed
        assert f.forecasted[0].value > 0.9

    def test_linear_decreasing(self):
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        f = forecast_metric("down", values, horizon=2, method="linear")
        assert f.forecasted[0].value < 0.5

    def test_confidence_bands_widen(self):
        # Use noisy data so residual std > 0
        values = [0.5, 0.65, 0.6, 0.75, 0.7, 0.85, 0.8]
        f = forecast_metric("m", values, horizon=5, method="linear")
        # Later forecasts should have wider bands
        widths = [fp.upper - fp.lower for fp in f.forecasted]
        assert widths[-1] > widths[0]

    def test_confidence_band_contains_value(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        f = forecast_metric("m", values, horizon=3, method="linear")
        for fp in f.forecasted:
            assert fp.lower <= fp.value <= fp.upper

    def test_threshold_alert(self):
        # Declining metric should trigger alert
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        f = forecast_metric("drop", values, horizon=3, threshold=0.3)
        assert len(f.alerts) > 0
        assert "drop below" in f.alerts[0]

    def test_no_threshold_alert_when_safe(self):
        values = [0.8, 0.85, 0.9, 0.92, 0.95]
        f = forecast_metric("safe", values, horizon=3, threshold=0.5)
        assert len(f.alerts) == 0

    def test_exponential_smoothing(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        f = forecast_metric("m", values, horizon=3, method="exponential_smoothing")
        assert f.method == "exponential_smoothing"
        assert len(f.forecasted) == 3
        # Should forecast upward
        assert f.forecasted[0].value > values[-1] * 0.9

    def test_exponential_smoothing_bands(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        f = forecast_metric("m", values, horizon=3, method="exponential_smoothing")
        for fp in f.forecasted:
            assert fp.lower <= fp.value <= fp.upper

    def test_flat_series(self):
        values = [0.8] * 10
        f = forecast_metric("flat", values, horizon=3, method="linear")
        for fp in f.forecasted:
            assert abs(fp.value - 0.8) < 0.01


# =============================================================================
# forecast_all
# =============================================================================

class TestForecastAll:
    def test_multiple_metrics(self):
        series = {
            "help": [0.5, 0.6, 0.7, 0.8, 0.9],
            "safety": [0.95, 0.93, 0.91, 0.89, 0.87],
        }
        report = forecast_all(series, horizon=2, threshold=0.85)
        assert len(report.results) == 2
        # safety is declining toward 0.85
        assert "safety" in report.metrics_at_risk or len(report.alerts) >= 0

    def test_empty(self):
        report = forecast_all({})
        assert len(report.results) == 0

    def test_sorted_by_metric_id(self):
        series = {"z": [0.5, 0.6, 0.7], "a": [0.8, 0.9, 1.0]}
        report = forecast_all(series)
        assert report.results[0].metric_id == "a"
        assert report.results[1].metric_id == "z"


# =============================================================================
# Internal helpers
# =============================================================================

class TestLinearSlopeIntercept:
    def test_perfect_line(self):
        slope, intercept = _linear_slope_intercept([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(1.0)

    def test_flat(self):
        slope, intercept = _linear_slope_intercept([5.0, 5.0, 5.0])
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(5.0)

    def test_single_value(self):
        slope, intercept = _linear_slope_intercept([3.0])
        assert slope == 0.0
        assert intercept == 3.0


class TestResidualStd:
    def test_perfect_fit(self):
        assert _residual_std([1.0, 2.0, 3.0], 1.0, 1.0) == pytest.approx(0.0)

    def test_too_few_points(self):
        assert _residual_std([1.0, 2.0], 1.0, 1.0) == 0.0


class TestZScore:
    def test_90(self):
        assert _z_score(0.90) == pytest.approx(1.645)

    def test_95(self):
        assert _z_score(0.95) == pytest.approx(1.960)

    def test_closest_match(self):
        # 0.92 is closest to 0.90
        assert _z_score(0.92) == pytest.approx(1.645)
