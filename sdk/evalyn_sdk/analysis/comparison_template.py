"""
Configurable comparison layouts for different audiences.

Provides built-in templates (executive, technical, qa) and supports
custom templates with audience-specific field formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ComparisonField:
    """A single field in a comparison template."""

    name: str
    label: str = ""
    format_fn: str = "default"  # "default" / "percentage" / "currency" / "duration"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "format_fn": self.format_fn,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComparisonField:
        return cls(
            name=data["name"],
            label=data.get("label", ""),
            format_fn=data.get("format_fn", "default"),
        )


@dataclass
class ComparisonTemplate:
    """Layout template for rendering run comparisons."""

    name: str
    description: str = ""
    fields: List[ComparisonField] = field(default_factory=list)
    show_delta: bool = True
    show_chart: bool = False
    audience: str = "technical"  # "technical" / "executive" / "qa"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fields": [f.as_dict() for f in self.fields],
            "show_delta": self.show_delta,
            "show_chart": self.show_chart,
            "audience": self.audience,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComparisonTemplate:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            fields=[ComparisonField.from_dict(f) for f in data.get("fields", [])],
            show_delta=data.get("show_delta", True),
            show_chart=data.get("show_chart", False),
            audience=data.get("audience", "technical"),
        )


@dataclass
class ComparisonOutput:
    """Rendered comparison result."""

    template_name: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "rows": list(self.rows),
            "summary": self.summary,
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append(f"Comparison: {self.template_name}")
        lines.append("-" * 40)
        for row in self.rows:
            parts = []
            for key, value in row.items():
                parts.append(f"{key}: {value}")
            lines.append("  ".join(parts))
        if self.summary:
            lines.append("")
            lines.append(self.summary)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------

EXECUTIVE_TEMPLATE = ComparisonTemplate(
    "executive",
    "High-level summary for leadership",
    [
        ComparisonField("pass_rate", "Pass Rate", "percentage"),
        ComparisonField("cost_usd", "Cost", "currency"),
    ],
    audience="executive",
)

TECHNICAL_TEMPLATE = ComparisonTemplate(
    "technical",
    "Detailed per-metric breakdown",
    [
        ComparisonField("pass_rate", "Pass Rate", "percentage"),
        ComparisonField("avg_score", "Avg Score"),
        ComparisonField("p95_latency", "P95 Latency", "duration"),
        ComparisonField("cost_usd", "Cost", "currency"),
    ],
    audience="technical",
)

QA_TEMPLATE = ComparisonTemplate(
    "qa",
    "QA-focused with regressions highlighted",
    [
        ComparisonField("pass_rate", "Pass Rate", "percentage"),
        ComparisonField("regressions", "Regressions"),
        ComparisonField("new_failures", "New Failures"),
    ],
    audience="qa",
)

_BUILTIN_TEMPLATES: Dict[str, ComparisonTemplate] = {
    "executive": EXECUTIVE_TEMPLATE,
    "technical": TECHNICAL_TEMPLATE,
    "qa": QA_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_comparison_template(name: str) -> ComparisonTemplate:
    """Look up a built-in template by name."""
    if name not in _BUILTIN_TEMPLATES:
        raise KeyError(f"Unknown template: {name!r}")
    return _BUILTIN_TEMPLATES[name]


def list_comparison_templates() -> List[ComparisonTemplate]:
    """Return all built-in templates."""
    return list(_BUILTIN_TEMPLATES.values())


def format_value(value: Any, format_fn: str) -> str:
    """Format a value according to the given format function name."""
    if value is None:
        return "N/A"
    if format_fn == "percentage":
        return f"{float(value):.1f}%"
    if format_fn == "currency":
        return f"${float(value):.2f}"
    if format_fn == "duration":
        return f"{int(value)}ms"
    return str(value)


def render_comparison(
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    template: ComparisonTemplate,
) -> ComparisonOutput:
    """Render a comparison of two data dicts using the given template."""
    rows: List[Dict[str, Any]] = []
    for f in template.fields:
        val_a = data_a.get(f.name)
        val_b = data_b.get(f.name)
        label = f.label or f.name
        row: Dict[str, Any] = {
            "field": label,
            "a": format_value(val_a, f.format_fn),
            "b": format_value(val_b, f.format_fn),
        }
        if template.show_delta and val_a is not None and val_b is not None:
            try:
                row["delta"] = float(val_b) - float(val_a)
            except (TypeError, ValueError):
                row["delta"] = None
        rows.append(row)

    summary_parts: List[str] = []
    for row in rows:
        delta = row.get("delta")
        if delta is not None and delta != 0:
            direction = "up" if delta > 0 else "down"
            summary_parts.append(f"{row['field']}: {direction} by {abs(delta):.2f}")
    summary = "; ".join(summary_parts) if summary_parts else "No changes detected"

    return ComparisonOutput(
        template_name=template.name,
        rows=rows,
        summary=summary,
    )


def create_custom_template(
    name: str,
    fields: List[Dict[str, Any]],
    audience: str = "technical",
) -> ComparisonTemplate:
    """Factory for creating a custom comparison template."""
    parsed_fields = [ComparisonField.from_dict(f) for f in fields]
    return ComparisonTemplate(
        name=name,
        fields=parsed_fields,
        audience=audience,
    )
