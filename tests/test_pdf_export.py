"""Tests for PDF HTML export module."""
from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.pdf_export import (
    PDFConfig,
    PDFSection,
    PDFExportResult,
    build_pdf_html,
    create_summary_section,
    create_metrics_section,
    create_chart_section,
    export_pdf_html,
    estimate_page_count,
)


# ---------------------------------------------------------------------------
# PDFConfig
# ---------------------------------------------------------------------------


class TestPDFConfig:
    def test_defaults(self):
        cfg = PDFConfig()
        assert cfg.title == "Evalyn Report"
        assert cfg.author == ""
        assert cfg.page_size == "A4"
        assert cfg.orientation == "portrait"
        assert cfg.include_charts is True
        assert cfg.include_raw_data is False

    def test_as_dict(self):
        cfg = PDFConfig(title="My Report", author="Alice")
        d = cfg.as_dict()
        assert d["title"] == "My Report"
        assert d["author"] == "Alice"

    def test_from_dict(self):
        d = {"title": "Custom", "page_size": "letter", "orientation": "landscape"}
        cfg = PDFConfig.from_dict(d)
        assert cfg.title == "Custom"
        assert cfg.page_size == "letter"
        assert cfg.orientation == "landscape"
        # Defaults for missing keys
        assert cfg.author == ""
        assert cfg.include_charts is True

    def test_from_dict_roundtrip(self):
        cfg = PDFConfig(title="RT", author="Bob", page_size="letter")
        restored = PDFConfig.from_dict(cfg.as_dict())
        assert restored.title == cfg.title
        assert restored.author == cfg.author
        assert restored.page_size == cfg.page_size


# ---------------------------------------------------------------------------
# PDFSection
# ---------------------------------------------------------------------------


class TestPDFSection:
    def test_as_dict(self):
        sec = PDFSection(title="Intro", content_html="<p>Hello</p>", page_break_after=True)
        d = sec.as_dict()
        assert d["title"] == "Intro"
        assert d["content_html"] == "<p>Hello</p>"
        assert d["page_break_after"] is True

    def test_defaults(self):
        sec = PDFSection(title="T")
        assert sec.content_html == ""
        assert sec.page_break_after is False


# ---------------------------------------------------------------------------
# PDFExportResult
# ---------------------------------------------------------------------------


class TestPDFExportResult:
    def test_as_dict(self):
        r = PDFExportResult(html_content="<h1>X</h1>", sections=3, estimated_pages=2, file_path="/tmp/r.html")
        d = r.as_dict()
        assert d["sections"] == 3
        assert d["estimated_pages"] == 2
        assert d["file_path"] == "/tmp/r.html"

    def test_format_text_with_path(self):
        r = PDFExportResult(html_content="", sections=5, estimated_pages=3, file_path="/out/report.html")
        text = r.format_text()
        assert "5 section(s)" in text
        assert "Estimated pages: 3" in text
        assert "/out/report.html" in text

    def test_format_text_without_path(self):
        r = PDFExportResult(html_content="", sections=1, estimated_pages=1)
        text = r.format_text()
        assert "File:" not in text


# ---------------------------------------------------------------------------
# build_pdf_html
# ---------------------------------------------------------------------------


class TestBuildPdfHtml:
    def test_contains_title(self):
        html = build_pdf_html([], PDFConfig(title="Test Title"))
        assert "Test Title" in html

    def test_has_print_css(self):
        html = build_pdf_html([], PDFConfig())
        assert "@media print" in html
        assert "@page" in html

    def test_landscape_orientation(self):
        html = build_pdf_html([], PDFConfig(orientation="landscape"))
        assert "landscape" in html

    def test_letter_page_size(self):
        html = build_pdf_html([], PDFConfig(page_size="letter"))
        assert "8.5in" in html

    def test_sections_rendered(self):
        sections = [PDFSection(title="Alpha", content_html="<p>content a</p>")]
        html = build_pdf_html(sections, PDFConfig())
        assert "Alpha" in html
        assert "content a" in html

    def test_page_break_class(self):
        sections = [PDFSection(title="Break", page_break_after=True)]
        html = build_pdf_html(sections, PDFConfig())
        assert "page-break" in html

    def test_author_shown(self):
        html = build_pdf_html([], PDFConfig(author="Charlie"))
        assert "Charlie" in html

    def test_no_author_when_empty(self):
        html = build_pdf_html([], PDFConfig(author=""))
        assert "Author:" not in html

    def test_html_escaping_in_title(self):
        html = build_pdf_html([], PDFConfig(title="<script>alert(1)</script>"))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# create_summary_section
# ---------------------------------------------------------------------------


class TestCreateSummarySection:
    def test_basic_summary(self):
        data = {"run_id": "abc", "dataset_name": "ds", "total_items": 10}
        sec = create_summary_section(data)
        assert sec.title == "Summary"
        assert "abc" in sec.content_html
        assert "ds" in sec.content_html

    def test_missing_keys(self):
        sec = create_summary_section({})
        assert sec.title == "Summary"
        assert "<table>" in sec.content_html


# ---------------------------------------------------------------------------
# create_metrics_section
# ---------------------------------------------------------------------------


class TestCreateMetricsSection:
    def test_table_structure(self):
        sec = create_metrics_section({"accuracy": 0.95, "f1": 0.88})
        assert "<table>" in sec.content_html
        assert "accuracy" in sec.content_html
        assert "0.9500" in sec.content_html
        assert "0.8800" in sec.content_html

    def test_empty_metrics(self):
        sec = create_metrics_section({})
        assert "<table>" in sec.content_html
        assert sec.title == "Metrics"


# ---------------------------------------------------------------------------
# create_chart_section
# ---------------------------------------------------------------------------


class TestCreateChartSection:
    def test_chart_placeholder(self):
        sec = create_chart_section("Score Dist", {"labels": ["A", "B"], "values": [1, 2]})
        assert "Score Dist" in sec.content_html
        assert "chart-placeholder" in sec.content_html
        assert "A: 1" in sec.content_html

    def test_empty_chart_data(self):
        sec = create_chart_section("Empty", {})
        assert "No data" in sec.content_html


# ---------------------------------------------------------------------------
# estimate_page_count
# ---------------------------------------------------------------------------


class TestEstimatePageCount:
    def test_empty_sections(self):
        assert estimate_page_count([]) == 1

    def test_small_content(self):
        sections = [PDFSection(title="T", content_html="<p>short</p>")]
        assert estimate_page_count(sections) == 1

    def test_large_content(self):
        big = PDFSection(title="Big", content_html="x" * 9000)
        pages = estimate_page_count([big])
        assert pages >= 3


# ---------------------------------------------------------------------------
# export_pdf_html
# ---------------------------------------------------------------------------


class TestExportPdfHtml:
    def test_writes_file(self, tmp_path):
        out = tmp_path / "report.html"
        sections = [PDFSection(title="S1", content_html="<p>hello</p>")]
        result = export_pdf_html(sections, str(out))
        assert out.exists()
        assert out.read_text(encoding="utf-8") == result.html_content

    def test_result_metadata(self, tmp_path):
        out = tmp_path / "r.html"
        sections = [PDFSection(title="A"), PDFSection(title="B")]
        result = export_pdf_html(sections, str(out))
        assert result.sections == 2
        assert result.estimated_pages >= 1
        assert result.file_path == str(out)

    def test_default_config_used(self, tmp_path):
        out = tmp_path / "def.html"
        result = export_pdf_html([PDFSection(title="T")], str(out))
        assert "Evalyn Report" in result.html_content

    def test_custom_config(self, tmp_path):
        out = tmp_path / "custom.html"
        cfg = PDFConfig(title="Custom Export", page_size="letter")
        result = export_pdf_html([PDFSection(title="X")], str(out), config=cfg)
        assert "Custom Export" in result.html_content
        assert "8.5in" in result.html_content

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "report.html"
        export_pdf_html([PDFSection(title="nested")], str(out))
        assert out.exists()
