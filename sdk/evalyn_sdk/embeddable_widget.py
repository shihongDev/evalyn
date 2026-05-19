"""Embeddable HTML widget for evaluation dashboards.

Pure Python, no external dependencies. Generates self-contained HTML
with inline CSS, SVG charts, and PostMessage-based communication.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WidgetConfig:
    """Configuration for widget rendering."""

    width: str = "100%"
    height: str = "400px"
    charts: list[str] = field(default_factory=lambda: ["bar"])
    theme: str = "light"
    show_header: bool = True
    show_footer: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "charts": list(self.charts),
            "theme": self.theme,
            "show_header": self.show_header,
            "show_footer": self.show_footer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WidgetConfig:
        return cls(
            width=data.get("width", "100%"),
            height=data.get("height", "400px"),
            charts=data.get("charts", ["bar"]),
            theme=data.get("theme", "light"),
            show_header=data.get("show_header", True),
            show_footer=data.get("show_footer", True),
        )


@dataclass
class WidgetData:
    """Data payload for widget rendering."""

    title: str
    metrics: dict[str, float]
    labels: list[str] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    summary_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "metrics": dict(self.metrics),
            "labels": list(self.labels),
            "series": [dict(s) for s in self.series],
            "summary_text": self.summary_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WidgetData:
        return cls(
            title=data["title"],
            metrics=data.get("metrics", {}),
            labels=data.get("labels", []),
            series=data.get("series", []),
            summary_text=data.get("summary_text", ""),
        )


DEFAULT_CONFIG = WidgetConfig()


def generate_score_bar_svg(
    metrics: dict[str, float], width: int = 400, height: int = 200
) -> str:
    """Generate an SVG bar chart for the given metrics.

    Each metric gets an equal-width bar. Bar heights are proportional
    to the maximum value in the metrics dict (or 1.0 if all zero).
    """
    if not metrics:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'

    n = len(metrics)
    max_val = max(abs(v) for v in metrics.values()) if metrics else 1.0
    if max_val == 0:
        max_val = 1.0

    bar_width = max(1, (width - 20) // n - 4)
    chart_top = 20
    chart_bottom = height - 30
    chart_height = chart_bottom - chart_top

    bars: list[str] = []
    for i, (name, value) in enumerate(metrics.items()):
        bar_h = (abs(value) / max_val) * chart_height
        x = 10 + i * (bar_width + 4)
        y = chart_bottom - bar_h

        escaped_name = html.escape(str(name))
        escaped_value = html.escape(f"{value:.2f}")

        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
            f'fill="#4a90d9" rx="2" />'
        )
        # Label below bar
        label_x = x + bar_width // 2
        bars.append(
            f'<text x="{label_x}" y="{chart_bottom + 14}" '
            f'text-anchor="middle" font-size="10" fill="#666">'
            f"{escaped_name}</text>"
        )
        # Value above bar
        bars.append(
            f'<text x="{label_x}" y="{y - 4}" '
            f'text-anchor="middle" font-size="10" fill="#333">'
            f"{escaped_value}</text>"
        )

    bars_str = "\n".join(bars)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f"{bars_str}\n"
        f"</svg>"
    )


def generate_postmessage_script() -> str:
    """Generate JavaScript for parent-page communication via PostMessage API."""
    # Target origin is '*' intentionally: this widget is designed to be
    # embedded by arbitrary host pages. The filter handler only toggles
    # element visibility so the attack surface is minimal.
    return (
        "<script>\n"
        "(function() {\n"
        "  var ALLOWED_TYPES = {'evalyn:filter': true};\n"
        "  window.parent.postMessage({type: 'evalyn:ready'}, '*');\n"
        "  window.addEventListener('message', function(event) {\n"
        "    if (!event.data || !ALLOWED_TYPES[event.data.type]) return;\n"
        "    if (event.data.type === 'evalyn:filter') {\n"
        "      var filter = event.data.filter || {};\n"
        "      document.querySelectorAll('[data-metric]').forEach(function(el) {\n"
        "        if (filter.metric && el.dataset.metric !== filter.metric) {\n"
        "          el.style.display = 'none';\n"
        "        } else {\n"
        "          el.style.display = '';\n"
        "        }\n"
        "      });\n"
        "    }\n"
        "  });\n"
        "})();\n"
        "</script>"
    )


def _css_for_theme(theme: str) -> str:
    """Return inline CSS for the given theme."""
    if theme == "dark":
        return (
            "body { margin: 0; padding: 8px; font-family: sans-serif; "
            "background: #1e1e1e; color: #e0e0e0; }\n"
            ".stats-bar { display: flex; gap: 16px; flex-wrap: wrap; "
            "padding: 8px 0; border-bottom: 1px solid #444; }\n"
            ".stat-item { font-size: 14px; }\n"
            ".stat-label { color: #999; }\n"
            ".stat-value { font-weight: bold; color: #e0e0e0; }\n"
            ".chart-area { padding: 8px 0; }\n"
            ".header { font-size: 16px; font-weight: bold; padding-bottom: 4px; }\n"
            ".footer { font-size: 11px; color: #777; padding-top: 8px; "
            "border-top: 1px solid #444; }\n"
            ".summary { font-size: 13px; color: #bbb; padding: 4px 0; }\n"
        )
    return (
        "body { margin: 0; padding: 8px; font-family: sans-serif; "
        "background: #fff; color: #333; }\n"
        ".stats-bar { display: flex; gap: 16px; flex-wrap: wrap; "
        "padding: 8px 0; border-bottom: 1px solid #eee; }\n"
        ".stat-item { font-size: 14px; }\n"
        ".stat-label { color: #888; }\n"
        ".stat-value { font-weight: bold; color: #333; }\n"
        ".chart-area { padding: 8px 0; }\n"
        ".header { font-size: 16px; font-weight: bold; padding-bottom: 4px; }\n"
        ".footer { font-size: 11px; color: #999; padding-top: 8px; "
        "border-top: 1px solid #eee; }\n"
        ".summary { font-size: 13px; color: #666; padding: 4px 0; }\n"
    )


def generate_widget_html(
    data: WidgetData, config: WidgetConfig | None = None
) -> str:
    """Generate a minimal self-contained HTML widget.

    The output is a complete HTML document suitable for embedding in an iframe.
    All CSS is inlined, no external dependencies. Dynamic content is escaped.
    """
    cfg = config or DEFAULT_CONFIG

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    parts.append(f"<title>{html.escape(data.title)}</title>")
    parts.append(f"<style>\n{_css_for_theme(cfg.theme)}</style>")
    parts.append("</head>")
    parts.append(
        f'<body style="width:{html.escape(cfg.width)};'
        f'max-height:{html.escape(cfg.height)};overflow:auto;">'
    )

    # Header
    if cfg.show_header:
        parts.append(f'<div class="header">{html.escape(data.title)}</div>')

    # Stats bar
    if data.metrics:
        parts.append('<div class="stats-bar">')
        for name, value in data.metrics.items():
            escaped_name = html.escape(str(name))
            escaped_value = html.escape(f"{value:.2f}")
            parts.append(
                f'<div class="stat-item" data-metric="{escaped_name}">'
                f'<span class="stat-label">{escaped_name}:</span> '
                f'<span class="stat-value">{escaped_value}</span>'
                f"</div>"
            )
        parts.append("</div>")

    # Chart
    if "bar" in cfg.charts and data.metrics:
        parts.append('<div class="chart-area">')
        parts.append(generate_score_bar_svg(data.metrics))
        parts.append("</div>")

    # Summary
    if data.summary_text:
        parts.append(
            f'<div class="summary">{html.escape(data.summary_text)}</div>'
        )

    # Footer
    if cfg.show_footer:
        parts.append(
            '<div class="footer">Powered by Evalyn</div>'
        )

    # PostMessage script
    parts.append(generate_postmessage_script())

    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)
