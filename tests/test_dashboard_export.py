"""Tests for dashboard_export module."""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.dashboard_export import (
    ExportBundle,
    ExportableChart,
    build_export_bundle,
    chart_to_csv,
    chart_to_json,
    generate_copy_button_html,
    generate_download_button_html,
    generate_export_panel_html,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chart():
    return ExportableChart(
        chart_id="metrics_overview",
        title="Metrics Overview",
        data=[
            {"metric": "accuracy", "score": 0.85},
            {"metric": "latency", "score": 0.72},
        ],
        columns=["metric", "score"],
    )


@pytest.fixture
def sample_bundle(sample_chart):
    return ExportBundle(
        charts=[sample_chart],
        summary={"total_metrics": 2, "mean_score": 0.785},
        generated_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# ExportableChart
# ---------------------------------------------------------------------------


class TestExportableChart:
    def test_as_dict(self, sample_chart):
        d = sample_chart.as_dict()
        assert d["chart_id"] == "metrics_overview"
        assert len(d["data"]) == 2
        assert d["columns"] == ["metric", "score"]

    def test_from_dict_roundtrip(self, sample_chart):
        d = sample_chart.as_dict()
        restored = ExportableChart.from_dict(d)
        assert restored.chart_id == sample_chart.chart_id
        assert restored.title == sample_chart.title
        assert len(restored.data) == len(sample_chart.data)

    def test_from_dict_defaults(self):
        c = ExportableChart.from_dict({})
        assert c.chart_id == ""
        assert c.data == []
        assert c.columns == []

    def test_as_dict_copies_data(self, sample_chart):
        d = sample_chart.as_dict()
        d["data"].append({"metric": "new", "score": 0.5})
        assert len(sample_chart.data) == 2


# ---------------------------------------------------------------------------
# ExportBundle
# ---------------------------------------------------------------------------


class TestExportBundle:
    def test_as_dict(self, sample_bundle):
        d = sample_bundle.as_dict()
        assert len(d["charts"]) == 1
        assert d["summary"]["total_metrics"] == 2
        assert d["generated_at"] == "2026-01-01T00:00:00Z"

    def test_from_dict_roundtrip(self, sample_bundle):
        d = sample_bundle.as_dict()
        restored = ExportBundle.from_dict(d)
        assert len(restored.charts) == 1
        assert restored.summary["total_metrics"] == 2

    def test_from_dict_defaults(self):
        b = ExportBundle.from_dict({})
        assert b.charts == []
        assert b.summary == {}


# ---------------------------------------------------------------------------
# chart_to_csv
# ---------------------------------------------------------------------------


class TestChartToCsv:
    def test_basic(self, sample_chart):
        csv_str = chart_to_csv(sample_chart)
        lines = csv_str.strip().splitlines()
        assert lines[0].strip() == "metric,score"
        assert "accuracy" in lines[1]
        assert "0.85" in lines[1]

    def test_empty_data(self):
        chart = ExportableChart(
            chart_id="empty", title="Empty", data=[], columns=["a", "b"]
        )
        csv_str = chart_to_csv(chart)
        assert "a,b" in csv_str
        assert csv_str.strip().count("\n") == 0

    def test_missing_column_in_row(self):
        chart = ExportableChart(
            chart_id="t", title="T",
            data=[{"a": 1}],
            columns=["a", "b"],
        )
        csv_str = chart_to_csv(chart)
        assert "1," in csv_str or "1,\r\n" in csv_str

    def test_special_characters_in_data(self):
        chart = ExportableChart(
            chart_id="t", title="T",
            data=[{"name": 'hello,"world"'}],
            columns=["name"],
        )
        csv_str = chart_to_csv(chart)
        assert "hello" in csv_str


# ---------------------------------------------------------------------------
# chart_to_json
# ---------------------------------------------------------------------------


class TestChartToJson:
    def test_basic(self, sample_chart):
        json_str = chart_to_json(sample_chart)
        parsed = json.loads(json_str)
        assert parsed["chart_id"] == "metrics_overview"
        assert len(parsed["data"]) == 2

    def test_empty_data(self):
        chart = ExportableChart(
            chart_id="empty", title="Empty", data=[], columns=[]
        )
        parsed = json.loads(chart_to_json(chart))
        assert parsed["data"] == []

    def test_valid_json(self, sample_chart):
        json_str = chart_to_json(sample_chart)
        # Should not raise
        json.loads(json_str)


# ---------------------------------------------------------------------------
# generate_download_button_html
# ---------------------------------------------------------------------------


class TestGenerateDownloadButtonHtml:
    def test_csv_format(self, sample_chart):
        html_str = generate_download_button_html(sample_chart, "csv")
        assert "download=" in html_str
        assert "data:text/csv;base64," in html_str
        assert "CSV" in html_str

    def test_json_format(self, sample_chart):
        html_str = generate_download_button_html(sample_chart, "json")
        assert "data:application/json;base64," in html_str
        assert "JSON" in html_str

    def test_escapes_title(self):
        chart = ExportableChart(
            chart_id="t", title='<script>alert("xss")</script>',
            data=[], columns=[],
        )
        html_str = generate_download_button_html(chart)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_contains_href(self, sample_chart):
        html_str = generate_download_button_html(sample_chart)
        assert 'href="data:' in html_str


# ---------------------------------------------------------------------------
# generate_copy_button_html
# ---------------------------------------------------------------------------


class TestGenerateCopyButtonHtml:
    def test_basic(self):
        html_str = generate_copy_button_html("some text", "Copy Me")
        assert "Copy Me" in html_str
        assert "navigator.clipboard.writeText" in html_str

    def test_escapes_content(self):
        html_str = generate_copy_button_html('<script>alert("x")</script>')
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_default_label(self):
        html_str = generate_copy_button_html("data")
        assert "Copy" in html_str


# ---------------------------------------------------------------------------
# generate_export_panel_html
# ---------------------------------------------------------------------------


class TestGenerateExportPanelHtml:
    def test_contains_sections(self, sample_bundle):
        html_str = generate_export_panel_html(sample_bundle)
        assert "export-panel" in html_str
        assert "Data Export" in html_str
        assert "Summary" in html_str

    def test_contains_download_buttons(self, sample_bundle):
        html_str = generate_export_panel_html(sample_bundle)
        assert "CSV" in html_str
        assert "JSON" in html_str

    def test_contains_copy_button(self, sample_bundle):
        html_str = generate_export_panel_html(sample_bundle)
        assert "Copy Summary" in html_str

    def test_empty_bundle(self):
        bundle = ExportBundle(charts=[], summary={}, generated_at="now")
        html_str = generate_export_panel_html(bundle)
        assert "export-panel" in html_str
        assert "Summary" in html_str

    def test_multiple_charts(self):
        charts = [
            ExportableChart(
                chart_id=f"c{i}", title=f"Chart {i}",
                data=[{"x": i}], columns=["x"],
            )
            for i in range(3)
        ]
        bundle = ExportBundle(charts=charts, summary={}, generated_at="now")
        html_str = generate_export_panel_html(bundle)
        assert "Chart 0" in html_str
        assert "Chart 2" in html_str


# ---------------------------------------------------------------------------
# build_export_bundle
# ---------------------------------------------------------------------------


class TestBuildExportBundle:
    def test_basic(self):
        bundle = build_export_bundle(
            metrics={"accuracy": 0.9, "latency": 0.7},
            failed_items=[{"id": "1", "reason": "timeout"}],
            per_metric_data={"accuracy": [{"item": "a", "score": 0.9}]},
        )
        assert len(bundle.charts) == 3  # overview + failed + accuracy
        assert bundle.summary["total_metrics"] == 2
        assert bundle.summary["failed_count"] == 1

    def test_no_failed_items(self):
        bundle = build_export_bundle(
            metrics={"accuracy": 0.9},
            failed_items=[],
            per_metric_data={},
        )
        assert len(bundle.charts) == 1  # only overview

    def test_no_per_metric_data(self):
        bundle = build_export_bundle(
            metrics={"a": 0.5},
            failed_items=[],
            per_metric_data={},
        )
        assert len(bundle.charts) == 1

    def test_generated_at_populated(self):
        bundle = build_export_bundle(
            metrics={"a": 1.0}, failed_items=[], per_metric_data={}
        )
        assert bundle.generated_at != ""

    def test_chart_ids(self):
        bundle = build_export_bundle(
            metrics={"accuracy": 0.9},
            failed_items=[{"id": "1"}],
            per_metric_data={"latency": [{"item": "b", "val": 0.5}]},
        )
        ids = [c.chart_id for c in bundle.charts]
        assert "metrics_overview" in ids
        assert "failed_items" in ids
        assert "metric_latency" in ids

    def test_empty_metrics(self):
        bundle = build_export_bundle(
            metrics={}, failed_items=[], per_metric_data={}
        )
        assert len(bundle.charts) == 1  # overview with no rows
        assert bundle.summary["total_metrics"] == 0
