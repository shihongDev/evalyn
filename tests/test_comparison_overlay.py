"""Tests for comparison_overlay module."""

from __future__ import annotations

import pytest

from evalyn_sdk.comparison_overlay import (
    OverlayChart,
    OverlayConfig,
    RunOverlayData,
    compute_deltas,
    generate_delta_table_html,
    generate_dual_bar_svg,
    generate_overlay_dashboard_html,
    generate_radar_svg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_a():
    return RunOverlayData(
        run_id="run-001",
        label="Baseline",
        metrics={"accuracy": 0.85, "latency": 0.72, "safety": 0.95},
        color="#4A90D9",
    )


@pytest.fixture
def run_b():
    return RunOverlayData(
        run_id="run-002",
        label="Improved",
        metrics={"accuracy": 0.92, "latency": 0.68, "safety": 0.97},
        color="#E8553A",
    )


@pytest.fixture
def config():
    return OverlayConfig(width=800, height=500, show_legend=True, show_delta=True)


# ---------------------------------------------------------------------------
# RunOverlayData
# ---------------------------------------------------------------------------


class TestRunOverlayData:
    def test_as_dict(self, run_a):
        d = run_a.as_dict()
        assert d["run_id"] == "run-001"
        assert d["metrics"]["accuracy"] == 0.85
        assert d["color"] == "#4A90D9"

    def test_from_dict_roundtrip(self, run_a):
        d = run_a.as_dict()
        restored = RunOverlayData.from_dict(d)
        assert restored.run_id == run_a.run_id
        assert restored.metrics == run_a.metrics

    def test_default_color(self):
        r = RunOverlayData(run_id="x", label="X", metrics={})
        assert r.color == "#4A90D9"


# ---------------------------------------------------------------------------
# OverlayConfig
# ---------------------------------------------------------------------------


class TestOverlayConfig:
    def test_defaults(self):
        c = OverlayConfig()
        assert c.width == 600
        assert c.height == 400
        assert c.show_legend is True
        assert c.show_delta is True

    def test_as_dict_from_dict(self, config):
        d = config.as_dict()
        restored = OverlayConfig.from_dict(d)
        assert restored.width == 800
        assert restored.height == 500


# ---------------------------------------------------------------------------
# OverlayChart
# ---------------------------------------------------------------------------


class TestOverlayChart:
    def test_as_dict(self):
        c = OverlayChart(chart_type="bar", svg="<svg/>", title="Test")
        d = c.as_dict()
        assert d["chart_type"] == "bar"

    def test_from_dict(self):
        c = OverlayChart.from_dict(
            {"chart_type": "radar", "svg": "<svg/>", "title": "T"}
        )
        assert c.chart_type == "radar"


# ---------------------------------------------------------------------------
# compute_deltas
# ---------------------------------------------------------------------------


class TestComputeDeltas:
    def test_basic(self, run_a, run_b):
        deltas = compute_deltas(run_a, run_b)
        assert abs(deltas["accuracy"] - 0.07) < 1e-9
        assert abs(deltas["latency"] - (-0.04)) < 1e-9
        assert abs(deltas["safety"] - 0.02) < 1e-9

    def test_missing_metric_in_a(self):
        a = RunOverlayData(run_id="a", label="A", metrics={"x": 1.0})
        b = RunOverlayData(run_id="b", label="B", metrics={"x": 2.0, "y": 3.0})
        deltas = compute_deltas(a, b)
        assert deltas["x"] == 1.0
        assert deltas["y"] == 3.0

    def test_missing_metric_in_b(self):
        a = RunOverlayData(run_id="a", label="A", metrics={"x": 1.0, "y": 5.0})
        b = RunOverlayData(run_id="b", label="B", metrics={"x": 2.0})
        deltas = compute_deltas(a, b)
        assert deltas["y"] == -5.0

    def test_empty_metrics(self):
        a = RunOverlayData(run_id="a", label="A", metrics={})
        b = RunOverlayData(run_id="b", label="B", metrics={})
        assert compute_deltas(a, b) == {}

    def test_identical_metrics(self, run_a):
        deltas = compute_deltas(run_a, run_a)
        for v in deltas.values():
            assert v == 0.0


# ---------------------------------------------------------------------------
# generate_dual_bar_svg
# ---------------------------------------------------------------------------


class TestGenerateDualBarSvg:
    def test_basic(self, run_a, run_b):
        svg = generate_dual_bar_svg(run_a, run_b)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "run-a" in svg
        assert "run-b" in svg

    def test_with_config(self, run_a, run_b, config):
        svg = generate_dual_bar_svg(run_a, run_b, config)
        assert 'width="800"' in svg

    def test_empty_metrics(self):
        a = RunOverlayData(run_id="a", label="A", metrics={})
        b = RunOverlayData(run_id="b", label="B", metrics={})
        svg = generate_dual_bar_svg(a, b)
        assert "<svg" in svg

    def test_escapes_labels(self):
        a = RunOverlayData(run_id="a", label='<script>"xss"</script>', metrics={"x": 1.0})
        b = RunOverlayData(run_id="b", label="B", metrics={"x": 2.0})
        svg = generate_dual_bar_svg(a, b)
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg

    def test_legend_present(self, run_a, run_b):
        svg = generate_dual_bar_svg(run_a, run_b, OverlayConfig(show_legend=True))
        assert "Baseline" in svg
        assert "Improved" in svg

    def test_no_legend(self, run_a, run_b):
        cfg = OverlayConfig(show_legend=False)
        svg = generate_dual_bar_svg(run_a, run_b, cfg)
        # Title still appears, but no legend rectangles at bottom
        assert "<svg" in svg

    def test_delta_labels(self, run_a, run_b):
        svg = generate_dual_bar_svg(run_a, run_b, OverlayConfig(show_delta=True))
        assert "+" in svg or "-" in svg


# ---------------------------------------------------------------------------
# generate_radar_svg
# ---------------------------------------------------------------------------


class TestGenerateRadarSvg:
    def test_basic(self, run_a, run_b):
        svg = generate_radar_svg(run_a, run_b)
        assert "<svg" in svg
        assert "polygon" in svg
        assert "run-a" in svg
        assert "run-b" in svg

    def test_fewer_than_3_metrics(self):
        a = RunOverlayData(run_id="a", label="A", metrics={"x": 1.0, "y": 2.0})
        b = RunOverlayData(run_id="b", label="B", metrics={"x": 1.5, "y": 2.5})
        svg = generate_radar_svg(a, b)
        assert "Radar chart requires 3+ metrics" in svg

    def test_escapes_metric_names(self):
        a = RunOverlayData(
            run_id="a", label="A",
            metrics={"<b>bold</b>": 1.0, "metric2": 2.0, "metric3": 3.0},
        )
        b = RunOverlayData(
            run_id="b", label="B",
            metrics={"<b>bold</b>": 1.5, "metric2": 2.5, "metric3": 3.5},
        )
        svg = generate_radar_svg(a, b)
        assert "<b>" not in svg
        assert "&lt;b&gt;" in svg

    def test_grid_lines(self, run_a, run_b):
        svg = generate_radar_svg(run_a, run_b)
        # Should have concentric grid polygons
        assert svg.count("stroke=\"#ddd\"") >= 3


# ---------------------------------------------------------------------------
# generate_delta_table_html
# ---------------------------------------------------------------------------


class TestGenerateDeltaTableHtml:
    def test_basic(self, run_a, run_b):
        html_str = generate_delta_table_html(run_a, run_b)
        assert "<table" in html_str
        assert "Baseline" in html_str
        assert "Improved" in html_str
        assert "accuracy" in html_str

    def test_directions(self, run_a, run_b):
        html_str = generate_delta_table_html(run_a, run_b)
        assert 'data-direction="up"' in html_str or 'data-direction="down"' in html_str

    def test_escapes_labels(self):
        a = RunOverlayData(
            run_id="a", label='<img src=x>', metrics={"m": 1.0}
        )
        b = RunOverlayData(run_id="b", label="B", metrics={"m": 2.0})
        html_str = generate_delta_table_html(a, b)
        assert "<img" not in html_str
        assert "&lt;img" in html_str

    def test_same_values(self):
        a = RunOverlayData(run_id="a", label="A", metrics={"m": 1.0})
        b = RunOverlayData(run_id="b", label="B", metrics={"m": 1.0})
        html_str = generate_delta_table_html(a, b)
        assert 'data-direction="same"' in html_str

    def test_empty_metrics(self):
        a = RunOverlayData(run_id="a", label="A", metrics={})
        b = RunOverlayData(run_id="b", label="B", metrics={})
        html_str = generate_delta_table_html(a, b)
        assert "<table" in html_str
        assert "<tbody></tbody>" in html_str


# ---------------------------------------------------------------------------
# generate_overlay_dashboard_html
# ---------------------------------------------------------------------------


class TestGenerateOverlayDashboardHtml:
    def test_basic(self, run_a, run_b):
        html_str = generate_overlay_dashboard_html(run_a, run_b)
        assert "<!DOCTYPE html>" in html_str
        assert "Comparison:" in html_str
        assert "Bar Chart" in html_str
        assert "Radar Chart" in html_str
        assert "Metric Deltas" in html_str

    def test_toggle_script(self, run_a, run_b):
        html_str = generate_overlay_dashboard_html(run_a, run_b)
        assert "toggleRun" in html_str
        assert "toggle-run-a" in html_str
        assert "toggle-run-b" in html_str

    def test_with_config(self, run_a, run_b, config):
        html_str = generate_overlay_dashboard_html(run_a, run_b, config)
        assert 'width="800"' in html_str

    def test_contains_both_labels(self, run_a, run_b):
        html_str = generate_overlay_dashboard_html(run_a, run_b)
        assert "Baseline" in html_str
        assert "Improved" in html_str

    def test_self_contained(self, run_a, run_b):
        html_str = generate_overlay_dashboard_html(run_a, run_b)
        assert "<style>" in html_str
        assert "<script>" in html_str
        assert "</html>" in html_str
