"""Tests for export format functions and trace utility functions."""
from __future__ import annotations

import pytest

from evalyn_sdk.cli.commands.export import (
    _build_html_export,
    _build_markdown_export,
    _flatten_export_rows,
    _metrics_summary_from_run_data,
)


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_run_data(
    *,
    run_id="run-001",
    dataset_name="test-dataset",
    num_items=3,
    metrics=None,
    summary=None,
):
    """Build realistic run_data dict for export testing."""
    if metrics is None:
        metrics = {"helpfulness": {"avg_score": 0.85, "pass_rate": 0.90},
                   "safety": {"avg_score": 0.95, "pass_rate": 1.0}}
    if summary is None:
        summary = {"metrics": metrics}

    results = []
    for i in range(num_items):
        item_id = f"item-{i:03d}"
        for metric_id in metrics:
            results.append({
                "item_id": item_id,
                "metrics": [
                    {
                        "metric_id": metric_id,
                        "score": metrics[metric_id]["avg_score"],
                        "passed": True,
                        "reason": "Looks good",
                    }
                ],
            })

    return {
        "id": run_id,
        "dataset_name": dataset_name,
        "started_at": "2026-03-14T10:00:00Z",
        "results": results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# _flatten_export_rows
# ---------------------------------------------------------------------------


class TestFlattenExportRows:
    def test_flattens_results_to_metric_rows(self):
        run_data = _make_run_data(num_items=2)
        rows = _flatten_export_rows(run_data)
        # 2 items x 2 metrics = 4 rows
        assert len(rows) == 4

    def test_row_keys(self):
        run_data = _make_run_data(num_items=1)
        rows = _flatten_export_rows(run_data)
        assert len(rows) > 0
        for row in rows:
            assert "item_id" in row
            assert "metric_id" in row
            assert "score" in row
            assert "passed" in row

    def test_empty_results(self):
        run_data = {"results": []}
        rows = _flatten_export_rows(run_data)
        assert rows == []

    def test_missing_results_key(self):
        rows = _flatten_export_rows({})
        assert rows == []


# ---------------------------------------------------------------------------
# _metrics_summary_from_run_data
# ---------------------------------------------------------------------------


class TestMetricsSummaryFromRunData:
    def test_new_format_with_metrics_key(self):
        run_data = {
            "summary": {
                "metrics": {"helpfulness": {"avg_score": 0.9, "pass_rate": 0.8}}
            }
        }
        result = _metrics_summary_from_run_data(run_data)
        assert "helpfulness" in result
        assert result["helpfulness"]["avg_score"] == 0.9

    def test_old_format_flat_summary(self):
        run_data = {
            "summary": {"helpfulness": {"avg_score": 0.7, "pass_rate": 0.6}}
        }
        result = _metrics_summary_from_run_data(run_data)
        assert "helpfulness" in result

    def test_empty_summary(self):
        result = _metrics_summary_from_run_data({"summary": {}})
        assert result == {}

    def test_missing_summary(self):
        result = _metrics_summary_from_run_data({})
        assert result == {}


# ---------------------------------------------------------------------------
# _build_markdown_export
# ---------------------------------------------------------------------------


class TestBuildMarkdownExport:
    def test_contains_header(self):
        md = _build_markdown_export(_make_run_data())
        assert "# Evaluation Report" in md

    def test_contains_run_id(self):
        md = _build_markdown_export(_make_run_data(run_id="abc-123"))
        assert "abc-123" in md

    def test_contains_dataset_name(self):
        md = _build_markdown_export(_make_run_data(dataset_name="my-eval"))
        assert "my-eval" in md

    def test_contains_metric_table(self):
        md = _build_markdown_export(_make_run_data())
        assert "| Metric |" in md
        assert "| helpfulness |" in md

    def test_shows_pass_rate_as_percentage(self):
        metrics = {"accuracy": {"avg_score": 0.75, "pass_rate": 0.80}}
        md = _build_markdown_export(_make_run_data(metrics=metrics))
        assert "80%" in md

    def test_shows_avg_score_formatted(self):
        metrics = {"accuracy": {"avg_score": 0.756, "pass_rate": 0.80}}
        md = _build_markdown_export(_make_run_data(metrics=metrics))
        assert "0.76" in md

    def test_results_count(self):
        md = _build_markdown_export(_make_run_data(num_items=5))
        # 5 items x 2 metrics = 10 results
        assert "10 items" in md

    def test_empty_summary_still_renders(self):
        run_data = _make_run_data()
        run_data["summary"] = {}
        md = _build_markdown_export(run_data)
        assert "# Evaluation Report" in md


# ---------------------------------------------------------------------------
# _build_html_export
# ---------------------------------------------------------------------------


class TestBuildHtmlExport:
    def test_valid_html_structure(self):
        html = _build_html_export(_make_run_data())
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_title(self):
        html = _build_html_export(_make_run_data(run_id="test-run-id"))
        assert "test-run-id" in html

    def test_contains_metric_rows(self):
        html = _build_html_export(_make_run_data())
        assert "helpfulness" in html
        assert "safety" in html

    def test_color_coding_high_pass_rate(self):
        metrics = {"good_metric": {"avg_score": 0.95, "pass_rate": 0.95}}
        html = _build_html_export(_make_run_data(metrics=metrics))
        assert "#4CAF50" in html  # green for >= 80%

    def test_color_coding_medium_pass_rate(self):
        metrics = {"mid_metric": {"avg_score": 0.70, "pass_rate": 0.70}}
        html = _build_html_export(_make_run_data(metrics=metrics))
        assert "#FF9800" in html  # orange for 60-80%

    def test_color_coding_low_pass_rate(self):
        metrics = {"bad_metric": {"avg_score": 0.40, "pass_rate": 0.40}}
        html = _build_html_export(_make_run_data(metrics=metrics))
        assert "#f44336" in html  # red for < 60%

    def test_contains_css_styles(self):
        html = _build_html_export(_make_run_data())
        assert "<style>" in html

    def test_contains_item_count(self):
        html = _build_html_export(_make_run_data(num_items=7))
        # results list has 7*2=14 entries but Items count is results length
        assert "14" in html or "7" in html

    def test_empty_metrics_still_renders(self):
        run_data = _make_run_data()
        run_data["summary"] = {}
        html = _build_html_export(run_data)
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Trace command utility functions
# ---------------------------------------------------------------------------

from evalyn_sdk.cli.commands.traces import (
    _format_value,
    _truncate,
)


class TestFormatValue:
    def test_string_value(self):
        assert _format_value("hello") == "hello"

    def test_dict_value_json_formatted(self):
        result = _format_value({"key": "val"})
        assert "key" in result
        assert "val" in result

    def test_list_value(self):
        result = _format_value([1, 2, 3])
        assert "1" in result

    def test_none_value(self):
        result = _format_value(None)
        assert "null" in result or "None" in result

    def test_long_string_truncated(self):
        long = "x" * 500
        result = _format_value(long, max_len=100)
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")

    def test_short_string_not_truncated(self):
        result = _format_value("short", max_len=300)
        assert result == "short"


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_string_truncated(self):
        long = "a" * 200
        result = _truncate(long, 50)
        assert len(result) <= 53  # 50 + "..."
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        s = "a" * 50
        assert _truncate(s, 50) == s

    def test_none_returns_empty(self):
        assert _truncate(None, 50) == ""

    def test_non_string_converted(self):
        result = _truncate(12345, 10)
        assert isinstance(result, str)
