"""
PDF-ready HTML report export for evaluation dashboards.

Generates print-optimized HTML with CSS @media print rules that can be
converted to PDF via the browser's "Print to PDF" function. No external
binary dependencies (wkhtmltopdf etc.) are required.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PDFConfig:
    """Configuration for PDF export."""
    title: str = "Evalyn Report"
    author: str = ""
    page_size: str = "A4"  # "A4" or "letter"
    orientation: str = "portrait"  # "portrait" or "landscape"
    include_charts: bool = True
    include_raw_data: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "page_size": self.page_size,
            "orientation": self.orientation,
            "include_charts": self.include_charts,
            "include_raw_data": self.include_raw_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PDFConfig:
        return cls(
            title=data.get("title", "Evalyn Report"),
            author=data.get("author", ""),
            page_size=data.get("page_size", "A4"),
            orientation=data.get("orientation", "portrait"),
            include_charts=data.get("include_charts", True),
            include_raw_data=data.get("include_raw_data", False),
        )


@dataclass
class PDFSection:
    """A single section in the PDF report."""
    title: str
    content_html: str = ""
    page_break_after: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content_html": self.content_html,
            "page_break_after": self.page_break_after,
        }


@dataclass
class PDFExportResult:
    """Result of a PDF export operation."""
    html_content: str
    sections: int = 0
    estimated_pages: int = 0
    file_path: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "html_content": self.html_content,
            "sections": self.sections,
            "estimated_pages": self.estimated_pages,
            "file_path": self.file_path,
        }

    def format_text(self) -> str:
        parts = [f"PDF Export: {self.sections} section(s)"]
        parts.append(f"Estimated pages: {self.estimated_pages}")
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_SIZES = {
    "A4": {"width": "210mm", "height": "297mm"},
    "letter": {"width": "8.5in", "height": "11in"},
}

# Approximate characters per page for estimation
_CHARS_PER_PAGE = 3000


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_pdf_html(sections: List[PDFSection], config: PDFConfig) -> str:
    """Build print-optimized HTML with CSS @media print rules.

    Args:
        sections: Ordered list of report sections.
        config: PDF export configuration.

    Returns:
        Complete HTML string ready for "Print to PDF".
    """
    size = _PAGE_SIZES.get(config.page_size, _PAGE_SIZES["A4"])
    escaped_title = _html.escape(config.title)
    escaped_author = _html.escape(config.author)

    css = f"""
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1A1A1A;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ font-size: 24px; margin-bottom: 4px; }}
        h2 {{ font-size: 18px; margin-top: 24px; border-bottom: 1px solid #E8E0D8; padding-bottom: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        .section {{ margin-bottom: 24px; }}
        .page-break {{ page-break-after: always; }}
        .meta {{ color: #555; font-size: 13px; margin-bottom: 16px; }}
        .chart-placeholder {{ border: 1px dashed #ccc; padding: 20px; text-align: center; color: #888; margin: 12px 0; }}

        @media print {{
            body {{ padding: 0; }}
            .page-break {{ page-break-after: always; }}
            @page {{
                size: {size["width"]} {size["height"]}{" landscape" if config.orientation == "landscape" else ""};
                margin: 15mm;
            }}
        }}
    </style>
    """

    header_parts = [f"<h1>{escaped_title}</h1>"]
    if escaped_author:
        header_parts.append(f'<div class="meta">Author: {escaped_author}</div>')

    section_parts: List[str] = []
    for sec in sections:
        cls = "section page-break" if sec.page_break_after else "section"
        escaped_sec_title = _html.escape(sec.title)
        section_parts.append(
            f'<div class="{cls}">'
            f"<h2>{escaped_sec_title}</h2>"
            f"{sec.content_html}"
            f"</div>"
        )

    body = "\n".join(header_parts) + "\n" + "\n".join(section_parts)

    return (
        "<!DOCTYPE html>\n"
        f"<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
        f"<title>{escaped_title}</title>\n{css}\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>"
    )


def create_summary_section(data: Dict[str, Any]) -> PDFSection:
    """Build a summary section from run data.

    Args:
        data: Dictionary with run metadata (run_id, dataset_name, created_at, etc.).

    Returns:
        A PDFSection containing an HTML summary table.
    """
    rows: List[str] = []
    for key in ("run_id", "dataset_name", "created_at", "total_items", "total_metrics"):
        val = data.get(key, "")
        rows.append(
            f"<tr><td><strong>{_html.escape(str(key))}</strong></td>"
            f"<td>{_html.escape(str(val))}</td></tr>"
        )

    html_content = f"<table><tbody>{''.join(rows)}</tbody></table>"
    return PDFSection(title="Summary", content_html=html_content)


def create_metrics_section(metrics: Dict[str, float]) -> PDFSection:
    """Build a per-metric table section.

    Args:
        metrics: Mapping of metric names to scores.

    Returns:
        A PDFSection with a metrics table.
    """
    rows = [
        f"<tr><td>{_html.escape(name)}</td><td>{score:.4f}</td></tr>"
        for name, score in metrics.items()
    ]
    html_content = (
        "<table>"
        "<thead><tr><th>Metric</th><th>Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return PDFSection(title="Metrics", content_html=html_content)


def create_chart_section(title: str, chart_data: Dict[str, Any]) -> PDFSection:
    """Build a chart placeholder section.

    Since PDF export relies on browser printing, charts are represented as
    placeholders describing the data.

    Args:
        title: Chart title.
        chart_data: Data that would populate the chart.

    Returns:
        A PDFSection with a chart placeholder.
    """
    labels = chart_data.get("labels", [])
    values = chart_data.get("values", [])
    items: List[str] = []
    for label, value in zip(labels, values):
        items.append(f"{_html.escape(str(label))}: {_html.escape(str(value))}")

    summary = ", ".join(items) if items else "No data"
    html_content = (
        f'<div class="chart-placeholder">'
        f"[Chart: {_html.escape(title)}] - {summary}"
        f"</div>"
    )
    return PDFSection(title=title, content_html=html_content)


def estimate_page_count(sections: List[PDFSection]) -> int:
    """Rough page estimate based on content length.

    Args:
        sections: Sections to estimate.

    Returns:
        Estimated number of pages (minimum 1).
    """
    total_chars = sum(len(s.content_html) + len(s.title) for s in sections)
    pages = max(1, (total_chars + _CHARS_PER_PAGE - 1) // _CHARS_PER_PAGE)
    return pages


def export_pdf_html(
    sections: List[PDFSection],
    file_path: str,
    config: Optional[PDFConfig] = None,
) -> PDFExportResult:
    """Write print-optimized HTML file.

    Args:
        sections: Report sections.
        file_path: Destination path for the HTML file.
        config: Optional PDF configuration (defaults apply if None).

    Returns:
        PDFExportResult with metadata about the export.
    """
    cfg = config or PDFConfig()
    html_content = build_pdf_html(sections, cfg)
    estimated = estimate_page_count(sections)

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")

    return PDFExportResult(
        html_content=html_content,
        sections=len(sections),
        estimated_pages=estimated,
        file_path=str(path),
    )
