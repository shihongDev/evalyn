"""Tests for analysis/report_templates module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.report_templates import (
    ReportSection,
    ReportTemplate,
    RenderedReport,
    SUMMARY_TEMPLATE,
    DETAILED_TEMPLATE,
    get_template,
    list_templates,
    render_section,
    render_report,
    create_custom_template,
    export_html,
)

import pytest


# ---------------------------------------------------------------------------
# get_template / list_templates
# ---------------------------------------------------------------------------


class TestTemplateLookup:
    def test_get_summary(self):
        t = get_template("summary")
        assert t.name == "summary"
        assert len(t.sections) == 3

    def test_get_detailed(self):
        t = get_template("detailed")
        assert t.name == "detailed"
        assert len(t.sections) == 5

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")

    def test_list_templates_has_two(self):
        templates = list_templates()
        assert len(templates) == 2
        names = {t.name for t in templates}
        assert "summary" in names
        assert "detailed" in names


# ---------------------------------------------------------------------------
# render_section
# ---------------------------------------------------------------------------


class TestRenderSection:
    def test_text_section(self):
        s = ReportSection("Intro", "text", content="Hello world")
        html = render_section(s)
        assert "<div" in html
        assert "Hello world" in html
        assert "<h2>Intro</h2>" in html

    def test_table_section(self):
        s = ReportSection("Data", "table")
        data = {
            "headers": ["Name", "Score"],
            "rows": [["item1", "0.9"], ["item2", "0.7"]],
        }
        html = render_section(s, data)
        assert "<table>" in html
        assert "<th>Name</th>" in html
        assert "<td>item1</td>" in html
        assert "<td>0.9</td>" in html

    def test_table_section_no_headers(self):
        s = ReportSection("Data", "table")
        data = {"rows": [["a", "b"]]}
        html = render_section(s, data)
        assert "<thead>" not in html
        assert "<td>a</td>" in html

    def test_chart_section(self):
        s = ReportSection("Chart", "chart", content="chart-data")
        html = render_section(s)
        assert 'class="chart-container"' in html
        assert "chart-data" in html

    def test_metric_section(self):
        s = ReportSection("Accuracy", "metric")
        html = render_section(s, {"value": "95%"})
        assert 'class="metric-value"' in html
        assert "95%" in html

    def test_text_section_with_data_content(self):
        s = ReportSection("Note", "text")
        html = render_section(s, {"content": "From data"})
        assert "From data" in html

    def test_section_escapes_html(self):
        s = ReportSection("XSS", "text", content="<script>alert(1)</script>")
        html = render_section(s)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_unknown_content_type_fallback(self):
        s = ReportSection("Other", "unknown", content="fallback")
        html = render_section(s)
        assert "fallback" in html
        assert "<div" in html


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


class TestRenderReport:
    def test_render_full(self):
        report = render_report(SUMMARY_TEMPLATE)
        assert report.template_name == "summary"
        assert report.sections_rendered == 3
        assert "<h2>Overview</h2>" in report.html
        assert "<h2>Metrics</h2>" in report.html

    def test_render_with_data(self):
        data = {
            "Overview": {"content": "Run completed"},
            "Metrics": {"headers": ["Metric", "Score"], "rows": [["acc", "0.9"]]},
        }
        report = render_report(SUMMARY_TEMPLATE, data)
        assert "Run completed" in report.html
        assert "<td>acc</td>" in report.html

    def test_render_empty_template(self):
        t = ReportTemplate(name="empty")
        report = render_report(t)
        assert report.sections_rendered == 0
        assert report.html == ""

    def test_render_with_header_footer(self):
        t = ReportTemplate(
            name="hf",
            header="<nav>Header</nav>",
            footer="<footer>Footer</footer>",
            sections=[ReportSection("Body", "text", content="Main")],
        )
        report = render_report(t)
        assert report.html.startswith("<nav>Header</nav>")
        assert report.html.endswith("<footer>Footer</footer>")
        assert "Main" in report.html


# ---------------------------------------------------------------------------
# create_custom_template
# ---------------------------------------------------------------------------


class TestCreateCustomTemplate:
    def test_create_basic(self):
        t = create_custom_template("custom", [
            {"name": "Intro", "content_type": "text"},
            {"name": "Data", "content_type": "table"},
        ])
        assert t.name == "custom"
        assert len(t.sections) == 2
        assert t.sections[0].content_type == "text"
        assert t.sections[1].content_type == "table"

    def test_create_with_css(self):
        t = create_custom_template("styled", [], css="body { color: red; }")
        assert t.css == "body { color: red; }"

    def test_create_empty(self):
        t = create_custom_template("blank", [])
        assert t.sections == []


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------


class TestExportHtml:
    def test_contains_title(self):
        report = RenderedReport("test", "<p>Content</p>", 1)
        doc = export_html(report, title="My Report")
        assert "<title>My Report</title>" in doc
        assert "<h1>My Report</h1>" in doc
        assert "<!DOCTYPE html>" in doc
        assert "<body>" in doc

    def test_default_title(self):
        report = RenderedReport("test", "<p>Body</p>", 1)
        doc = export_html(report)
        assert "<title>Evalyn Report</title>" in doc

    def test_escapes_title(self):
        report = RenderedReport("test", "", 0)
        doc = export_html(report, title="<script>bad</script>")
        assert "<script>bad</script>" not in doc
        assert "&lt;script&gt;" in doc

    def test_contains_report_html(self):
        report = RenderedReport("test", "<div>My section</div>", 1)
        doc = export_html(report)
        assert "<div>My section</div>" in doc


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_report_section_roundtrip(self):
        s = ReportSection("Intro", "text", content="Hello", data={"key": "val"})
        d = s.as_dict()
        s2 = ReportSection.from_dict(d)
        assert s2.name == "Intro"
        assert s2.content_type == "text"
        assert s2.content == "Hello"
        assert s2.data == {"key": "val"}

    def test_report_template_roundtrip(self):
        t = ReportTemplate(
            name="t1",
            description="desc",
            sections=[ReportSection("A", "text")],
            css=".x{}",
            header="<h>",
            footer="</h>",
        )
        d = t.as_dict()
        t2 = ReportTemplate.from_dict(d)
        assert t2.name == "t1"
        assert t2.description == "desc"
        assert len(t2.sections) == 1
        assert t2.css == ".x{}"
        assert t2.header == "<h>"
        assert t2.footer == "</h>"

    def test_rendered_report_as_dict(self):
        r = RenderedReport("summary", "<p>hi</p>", 3)
        d = r.as_dict()
        assert d["template_name"] == "summary"
        assert d["html"] == "<p>hi</p>"
        assert d["sections_rendered"] == 3

    def test_report_template_from_dict_defaults(self):
        d = {"name": "minimal"}
        t = ReportTemplate.from_dict(d)
        assert t.description == ""
        assert t.sections == []
        assert t.css == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sections_render(self):
        t = ReportTemplate(name="empty", sections=[])
        report = render_report(t)
        assert report.sections_rendered == 0

    def test_section_name_escaping(self):
        s = ReportSection("<b>Bold</b>", "text")
        html = render_section(s)
        assert "&lt;b&gt;Bold&lt;/b&gt;" in html

    def test_table_with_single_value_rows(self):
        s = ReportSection("Simple", "table")
        html = render_section(s, {"rows": ["a", "b"]})
        assert "<td>a</td>" in html
        assert "<td>b</td>" in html
