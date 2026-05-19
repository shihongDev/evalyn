"""Custom report templates for HTML evaluation reports.

Provides dataclasses for defining report layouts and functions
for rendering them to HTML.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """A single section within a report template."""

    name: str
    content_type: str = "text"  # "text" / "chart" / "table" / "metric"
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content_type": self.content_type,
            "content": self.content,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReportSection:
        return cls(
            name=data["name"],
            content_type=data.get("content_type", "text"),
            content=data.get("content", ""),
            data=data.get("data", {}),
        )


@dataclass
class ReportTemplate:
    """A reusable report layout consisting of ordered sections."""

    name: str
    description: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    css: str = ""
    header: str = ""
    footer: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sections": [s.as_dict() for s in self.sections],
            "css": self.css,
            "header": self.header,
            "footer": self.footer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReportTemplate:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            sections=[ReportSection.from_dict(s) for s in data.get("sections", [])],
            css=data.get("css", ""),
            header=data.get("header", ""),
            footer=data.get("footer", ""),
        )


@dataclass
class RenderedReport:
    """The output of rendering a template with data."""

    template_name: str
    html: str
    sections_rendered: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "html": self.html,
            "sections_rendered": self.sections_rendered,
        }


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATE = ReportTemplate(
    name="summary",
    description="Compact evaluation summary",
    sections=[
        ReportSection("Overview", "text"),
        ReportSection("Metrics", "table"),
        ReportSection("Recommendations", "text"),
    ],
)

DETAILED_TEMPLATE = ReportTemplate(
    name="detailed",
    description="Detailed evaluation report with charts",
    sections=[
        ReportSection("Overview", "text"),
        ReportSection("Metrics", "table"),
        ReportSection("Score Distribution", "chart"),
        ReportSection("Failed Items", "table"),
        ReportSection("Recommendations", "text"),
    ],
)

_BUILTIN_TEMPLATES: Dict[str, ReportTemplate] = {
    "summary": SUMMARY_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_template(name: str) -> ReportTemplate:
    """Look up a built-in template by name. Raises ValueError if not found."""
    if name not in _BUILTIN_TEMPLATES:
        raise ValueError(
            f"Unknown template {name!r}. Available: {list(_BUILTIN_TEMPLATES.keys())}"
        )
    return _BUILTIN_TEMPLATES[name]


def list_templates() -> List[ReportTemplate]:
    """Return all built-in templates."""
    return list(_BUILTIN_TEMPLATES.values())


def render_section(section: ReportSection, data: Optional[Dict[str, Any]] = None) -> str:
    """Render a single section to an HTML fragment.

    Content types:
    - text: wrapped in a <div>
    - table: rendered as an HTML <table>
    - chart: wrapped in a <div class="chart">
    - metric: wrapped in a <div class="metric">
    """
    if data is None:
        data = {}
    safe_name = _html.escape(section.name)
    content = section.content or data.get("content", "")
    safe_content = _html.escape(str(content)) if content else ""

    if section.content_type == "text":
        return f'<div class="section text"><h2>{safe_name}</h2><p>{safe_content}</p></div>'

    if section.content_type == "table":
        rows = data.get("rows", [])
        headers = data.get("headers", [])
        parts: List[str] = [f'<div class="section table"><h2>{safe_name}</h2><table>']
        if headers:
            parts.append("<thead><tr>")
            for h in headers:
                parts.append(f"<th>{_html.escape(str(h))}</th>")
            parts.append("</tr></thead>")
        parts.append("<tbody>")
        for row in rows:
            parts.append("<tr>")
            cells = row if isinstance(row, (list, tuple)) else [row]
            for cell in cells:
                parts.append(f"<td>{_html.escape(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        return "".join(parts)

    if section.content_type == "chart":
        return f'<div class="section chart"><h2>{safe_name}</h2><div class="chart-container">{safe_content}</div></div>'

    if section.content_type == "metric":
        value = data.get("value", content)
        safe_value = _html.escape(str(value))
        return f'<div class="section metric"><h2>{safe_name}</h2><div class="metric-value">{safe_value}</div></div>'

    # Fallback
    return f'<div class="section"><h2>{safe_name}</h2><p>{safe_content}</p></div>'


def render_report(
    template: ReportTemplate, data: Optional[Dict[str, Any]] = None
) -> RenderedReport:
    """Render a full report template to HTML.

    The data dict is keyed by section name. Each section receives the
    sub-dict matching its name, falling back to the top-level data.
    """
    if data is None:
        data = {}
    parts: List[str] = []

    if template.header:
        parts.append(template.header)

    rendered_count = 0
    for section in template.sections:
        section_data = data.get(section.name, data)
        parts.append(render_section(section, section_data))
        rendered_count += 1

    if template.footer:
        parts.append(template.footer)

    return RenderedReport(
        template_name=template.name,
        html="\n".join(parts),
        sections_rendered=rendered_count,
    )


def create_custom_template(
    name: str,
    sections: List[Dict[str, Any]],
    css: str = "",
) -> ReportTemplate:
    """Create a custom template from section dicts."""
    return ReportTemplate(
        name=name,
        sections=[ReportSection.from_dict(s) for s in sections],
        css=css,
    )


def export_html(report: RenderedReport, title: str = "Evalyn Report") -> str:
    """Wrap a rendered report in a full HTML document with head/body/title."""
    safe_title = _html.escape(title)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        f"<title>{safe_title}</title>\n"
        "<meta charset=\"utf-8\">\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{safe_title}</h1>\n"
        f"{report.html}\n"
        "</body>\n"
        "</html>"
    )
