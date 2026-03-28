"""Tests for trend anomaly detection."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.trend_anomaly import (
    AnomalyConfig,
    AnomalyPoint,
    AnomalyReport,
    build_anomaly_report,
    compute_z_scores,
    detect_anomalies,
    detect_sudden_drop,
    detect_trend_break,
    render_anomaly_chart,
)


# ---------------------------------------------------------------------------
# compute_z_scores
# ---------------------------------------------------------------------------


class TestComputeZScores:
    def test_basic_series(self):
        """Z-scores should be computed for a simple series."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        z = compute_z_scores(values, window_size=5)
        assert len(z) == 6
        # The last value (5.0) should have a large positive z-score
        assert z[-1] > 2.0

    def test_constant_values(self):
        """Constant values should all have z-score 0."""
        values = [3.0] * 10
        z = compute_z_scores(values, window_size=5)
        assert all(score == 0.0 for score in z)

    def test_single_value(self):
        """Single value should return z-score 0."""
        z = compute_z_scores([5.0])
        assert z == [0.0]

    def test_empty_list(self):
        """Empty list should return empty list."""
        z = compute_z_scores([])
        assert z == []

    def test_two_values(self):
        """Two values: both have z=0 due to insufficient window."""
        z = compute_z_scores([1.0, 10.0])
        assert z[0] == 0.0
        # Only 1 point in window - insufficient for z-score
        assert z[1] == 0.0

    def test_three_values_with_spike(self):
        """Three values: third point can use 2-point window."""
        z = compute_z_scores([1.0, 1.0, 10.0])
        assert z[0] == 0.0
        assert z[1] == 0.0
        # Third value has 2-point window [1.0, 1.0], std=0, value differs
        assert z[2] > 50  # Gets large z due to zero-std window

    def test_window_size_respected(self):
        """Smaller window should make z-scores more sensitive."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        z_small = compute_z_scores(values, window_size=3)
        z_large = compute_z_scores(values, window_size=8)
        # Both should detect the spike at the end
        assert z_small[-1] > 0
        assert z_large[-1] > 0


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_finds_spikes(self):
        """Should detect a clear spike as anomaly."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
        points = detect_anomalies(values)
        anomalies = [p for p in points if p.is_anomaly]
        assert len(anomalies) >= 1
        # The spike at index 6 should be flagged
        anomaly_indices = {a.index for a in anomalies}
        assert 6 in anomaly_indices

    def test_no_anomalies_smooth(self):
        """Constant values should have no anomalies."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        points = detect_anomalies(values)
        anomalies = [p for p in points if p.is_anomaly]
        assert len(anomalies) == 0

    def test_returns_all_points(self):
        """Should return a point for every input value."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        points = detect_anomalies(values)
        assert len(points) == len(values)

    def test_insufficient_data(self):
        """Should return empty list if fewer than min_data_points."""
        values = [1.0, 2.0, 3.0]
        points = detect_anomalies(values, AnomalyConfig(min_data_points=5))
        assert points == []

    def test_custom_threshold(self):
        """Lower threshold should flag more anomalies."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
        strict = detect_anomalies(values, AnomalyConfig(z_threshold=3.0))
        loose = detect_anomalies(values, AnomalyConfig(z_threshold=0.5))
        strict_count = sum(1 for p in strict if p.is_anomaly)
        loose_count = sum(1 for p in loose if p.is_anomaly)
        assert loose_count >= strict_count

    def test_default_config(self):
        """Should work with default config (None)."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
        points = detect_anomalies(values, None)
        assert len(points) == 6

    def test_anomaly_point_fields(self):
        """Each AnomalyPoint should have correct fields."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
        points = detect_anomalies(values)
        for p in points:
            assert isinstance(p.index, int)
            assert isinstance(p.value, float)
            assert isinstance(p.expected, float)
            assert isinstance(p.z_score, float)
            assert isinstance(p.is_anomaly, bool)


# ---------------------------------------------------------------------------
# detect_trend_break
# ---------------------------------------------------------------------------


class TestDetectTrendBreak:
    def test_finds_reversal(self):
        """Should find the point where an uptrend reverses to downtrend."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 4.0, 3.0]
        idx = detect_trend_break(values, window_size=3)
        assert idx is not None
        # Break should be around the peak
        assert 4 <= idx <= 8

    def test_no_break_monotonic(self):
        """Monotonically increasing series should have no break."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        idx = detect_trend_break(values, window_size=3)
        assert idx is None

    def test_insufficient_data(self):
        """Too few points should return None."""
        values = [1.0, 2.0, 3.0]
        idx = detect_trend_break(values, window_size=5)
        assert idx is None

    def test_constant_values(self):
        """Constant values should have no break."""
        values = [5.0] * 15
        idx = detect_trend_break(values, window_size=3)
        assert idx is None

    def test_v_shape(self):
        """V-shaped series should detect the break at the bottom."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        idx = detect_trend_break(values, window_size=3)
        assert idx is not None


# ---------------------------------------------------------------------------
# detect_sudden_drop
# ---------------------------------------------------------------------------


class TestDetectSuddenDrop:
    def test_finds_drop(self):
        """Should find a 50% drop."""
        values = [1.0, 1.0, 1.0, 0.4, 1.0]
        drops = detect_sudden_drop(values, threshold_pct=0.2)
        assert 3 in drops

    def test_no_drops(self):
        """Gradually changing series should have no drops."""
        values = [1.0, 0.95, 0.90, 0.85, 0.80]
        drops = detect_sudden_drop(values, threshold_pct=0.2)
        assert drops == []

    def test_multiple_drops(self):
        """Should find multiple drops."""
        values = [1.0, 0.5, 1.0, 0.5, 1.0]
        drops = detect_sudden_drop(values, threshold_pct=0.2)
        assert len(drops) == 2
        assert 1 in drops
        assert 3 in drops

    def test_increase_not_flagged(self):
        """Increases should not be flagged."""
        values = [0.5, 1.0, 0.5, 1.0]
        drops = detect_sudden_drop(values, threshold_pct=0.2)
        # Only drops from 1.0 -> 0.5 (indices 2)
        assert 1 not in drops

    def test_zero_previous_skipped(self):
        """Zero previous value should not cause division error."""
        values = [0.0, 1.0, 0.5]
        drops = detect_sudden_drop(values, threshold_pct=0.2)
        # First transition skipped (prev=0), second is a drop
        assert 2 in drops

    def test_empty_list(self):
        """Empty list should return empty."""
        assert detect_sudden_drop([]) == []


# ---------------------------------------------------------------------------
# build_anomaly_report
# ---------------------------------------------------------------------------


class TestBuildAnomalyReport:
    def test_basic_report(self):
        """Should produce a valid report."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
        report = build_anomaly_report("test_metric", values)
        assert report.metric_id == "test_metric"
        assert report.total_points == 6
        assert report.anomaly_count >= 1
        assert report.anomaly_rate > 0

    def test_no_anomalies_report(self):
        """Smooth data should produce report with zero anomalies."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        report = build_anomaly_report("smooth", values)
        assert report.anomaly_count == 0
        assert report.anomaly_rate == 0.0

    def test_empty_values(self):
        """Empty values should produce empty report."""
        report = build_anomaly_report("empty", [])
        assert report.total_points == 0
        assert report.anomaly_count == 0

    def test_with_custom_config(self):
        """Custom config should be respected."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        config = AnomalyConfig(z_threshold=0.5)
        report = build_anomaly_report("custom", values, config)
        assert report.total_points == 6


# ---------------------------------------------------------------------------
# render_anomaly_chart
# ---------------------------------------------------------------------------


class TestRenderAnomalyChart:
    def test_output_has_lines(self):
        """Should produce one line per value."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        chart = render_anomaly_chart(values, [2], width=40)
        lines = chart.strip().split("\n")
        assert len(lines) == 5

    def test_anomaly_markers(self):
        """Anomaly indices should use '!' marker."""
        values = [1.0, 2.0, 3.0]
        chart = render_anomaly_chart(values, [1], width=40)
        lines = chart.strip().split("\n")
        assert "!" in lines[1]
        assert "!" not in lines[0]

    def test_empty_values(self):
        """Empty values should return empty string."""
        assert render_anomaly_chart([], []) == ""

    def test_constant_values(self):
        """Constant values should still produce chart."""
        values = [5.0, 5.0, 5.0]
        chart = render_anomaly_chart(values, [], width=30)
        assert len(chart) > 0


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------


class TestAnomalyConfigSerialization:
    def test_as_dict(self):
        """as_dict should produce correct dict."""
        config = AnomalyConfig(z_threshold=3.0, min_data_points=10, window_size=7)
        d = config.as_dict()
        assert d["z_threshold"] == 3.0
        assert d["min_data_points"] == 10
        assert d["window_size"] == 7

    def test_from_dict(self):
        """from_dict should reconstruct config."""
        d = {"z_threshold": 1.5, "min_data_points": 3, "window_size": 10}
        config = AnomalyConfig.from_dict(d)
        assert config.z_threshold == 1.5
        assert config.min_data_points == 3
        assert config.window_size == 10

    def test_from_dict_defaults(self):
        """from_dict with empty dict should use defaults."""
        config = AnomalyConfig.from_dict({})
        assert config.z_threshold == 2.0
        assert config.min_data_points == 5
        assert config.window_size == 5

    def test_roundtrip(self):
        """as_dict -> from_dict should produce equal config."""
        original = AnomalyConfig(z_threshold=2.5, min_data_points=8, window_size=3)
        restored = AnomalyConfig.from_dict(original.as_dict())
        assert restored.z_threshold == original.z_threshold
        assert restored.min_data_points == original.min_data_points
        assert restored.window_size == original.window_size


class TestAnomalyPointSerialization:
    def test_as_dict(self):
        """as_dict should produce correct dict."""
        point = AnomalyPoint(index=3, value=5.0, expected=2.0, z_score=3.5, is_anomaly=True)
        d = point.as_dict()
        assert d["index"] == 3
        assert d["value"] == 5.0
        assert d["expected"] == 2.0
        assert d["z_score"] == 3.5
        assert d["is_anomaly"] is True

    def test_as_dict_default_not_anomaly(self):
        """Default is_anomaly should be False."""
        point = AnomalyPoint(index=0, value=1.0, expected=1.0, z_score=0.0)
        d = point.as_dict()
        assert d["is_anomaly"] is False


class TestAnomalyReportSerialization:
    def test_as_dict(self):
        """as_dict should include all fields."""
        report = AnomalyReport(
            anomalies=[AnomalyPoint(0, 1.0, 1.0, 0.0, False)],
            total_points=10,
            anomaly_count=1,
            anomaly_rate=0.1,
            metric_id="test",
        )
        d = report.as_dict()
        assert d["metric_id"] == "test"
        assert d["total_points"] == 10
        assert len(d["anomalies"]) == 1

    def test_from_dict(self):
        """from_dict should reconstruct report."""
        d = {
            "anomalies": [
                {"index": 0, "value": 5.0, "expected": 1.0, "z_score": 4.0, "is_anomaly": True}
            ],
            "total_points": 5,
            "anomaly_count": 1,
            "anomaly_rate": 0.2,
            "metric_id": "m1",
        }
        report = AnomalyReport.from_dict(d)
        assert report.metric_id == "m1"
        assert report.anomaly_count == 1
        assert len(report.anomalies) == 1
        assert report.anomalies[0].is_anomaly is True

    def test_roundtrip(self):
        """as_dict -> from_dict roundtrip should preserve data."""
        original = AnomalyReport(
            anomalies=[
                AnomalyPoint(0, 1.0, 1.0, 0.0, False),
                AnomalyPoint(5, 10.0, 1.0, 9.0, True),
            ],
            total_points=10,
            anomaly_count=1,
            anomaly_rate=0.1,
            metric_id="roundtrip",
        )
        restored = AnomalyReport.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.total_points == original.total_points
        assert len(restored.anomalies) == 2

    def test_format_text(self):
        """format_text should produce readable output."""
        report = AnomalyReport(
            anomalies=[AnomalyPoint(3, 10.0, 1.0, 9.0, True)],
            total_points=10,
            anomaly_count=1,
            anomaly_rate=0.1,
            metric_id="format_test",
        )
        text = report.format_text()
        assert "format_test" in text
        assert "10" in text
        assert "Anomalies: 1" in text

    def test_format_text_empty(self):
        """format_text with no anomalies should still work."""
        report = AnomalyReport(metric_id="empty")
        text = report.format_text()
        assert "empty" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_short_series_z_scores(self):
        """Short series should not crash."""
        z = compute_z_scores([1.0, 2.0, 3.0])
        assert len(z) == 3

    def test_short_series_anomalies(self):
        """Short series below min_data_points returns empty."""
        points = detect_anomalies([1.0, 2.0])
        assert points == []

    def test_large_spike(self):
        """Very large spike should definitely be detected."""
        values = [1.0] * 10 + [1000.0]
        report = build_anomaly_report("spike", values)
        assert report.anomaly_count >= 1

    def test_negative_values(self):
        """Negative values should work correctly."""
        values = [-1.0, -1.0, -1.0, -1.0, -1.0, -10.0]
        points = detect_anomalies(values)
        assert len(points) == 6

    def test_alternating_values(self):
        """Alternating high/low should be handled."""
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        z = compute_z_scores(values, window_size=3)
        assert len(z) == 7
