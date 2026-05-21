"""Auto-generated browsable catalog of metrics.

Pure Python, no external dependencies. Builds a structured catalog from
the subjective metric registry and makes it available in multiple formats
(markdown, HTML, plain text).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CatalogEntry:
    """Single metric entry in the catalog."""

    metric_id: str
    metric_type: str
    description: str
    category: str
    scope: str
    rubric_levels: int
    bundle_memberships: list[str] = field(default_factory=list)
    recommended_use: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type,
            "description": self.description,
            "category": self.category,
            "scope": self.scope,
            "rubric_levels": self.rubric_levels,
            "bundle_memberships": list(self.bundle_memberships),
            "recommended_use": self.recommended_use,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        return cls(
            metric_id=data["metric_id"],
            metric_type=data["metric_type"],
            description=data.get("description", ""),
            category=data.get("category", ""),
            scope=data.get("scope", ""),
            rubric_levels=data.get("rubric_levels", 0),
            bundle_memberships=data.get("bundle_memberships", []),
            recommended_use=data.get("recommended_use", ""),
        )


@dataclass
class MetricCatalog:
    """Browsable catalog of all available metrics."""

    entries: list[CatalogEntry] = field(default_factory=list)
    categories: dict[str, str] = field(default_factory=dict)
    total_count: int = 0
    generated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "categories": dict(self.categories),
            "total_count": self.total_count,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricCatalog:
        return cls(
            entries=[CatalogEntry.from_dict(e) for e in data.get("entries", [])],
            categories=data.get("categories", {}),
            total_count=data.get("total_count", 0),
            generated_at=data.get("generated_at", ""),
        )


def _infer_recommended_use(metric: dict[str, Any]) -> str:
    """Infer a recommended use string from metric properties."""
    scope = metric.get("scope", "overall")
    category = metric.get("category", "")
    scope_hints = {
        "llm_call": "Per-call evaluation",
        "trace": "Full trace evaluation",
        "overall": "End-to-end evaluation",
    }
    hint = scope_hints.get(scope, "General evaluation")
    if category:
        return f"{hint} - {category}"
    return hint


def _count_rubric_levels(metric: dict[str, Any]) -> int:
    """Count rubric levels from metric config."""
    config = metric.get("config", {})
    rubric = config.get("rubric", metric.get("rubric"))
    if isinstance(rubric, dict):
        return len(rubric)
    if isinstance(rubric, list):
        return len(rubric)
    return 0


def build_catalog(
    registry: list[dict[str, Any]],
    categories: dict[str, str],
    bundles: dict[str, list[str]] | None = None,
) -> MetricCatalog:
    """Build catalog from registry data.

    Args:
        registry: List of metric dicts (SUBJECTIVE_REGISTRY format).
        categories: Category ID to description mapping.
        bundles: Optional bundle name to metric ID list mapping.

    Returns:
        Populated MetricCatalog.
    """
    bundle_map: dict[str, list[str]] = {}
    if bundles:
        for bundle_name, metric_ids in bundles.items():
            for mid in metric_ids:
                bundle_map.setdefault(mid, []).append(bundle_name)

    entries = []
    for metric in registry:
        mid = metric.get("id", "")
        entry = CatalogEntry(
            metric_id=mid,
            metric_type=metric.get("type", ""),
            description=metric.get("description", ""),
            category=metric.get("category", ""),
            scope=metric.get("scope", ""),
            rubric_levels=_count_rubric_levels(metric),
            bundle_memberships=sorted(bundle_map.get(mid, [])),
            recommended_use=_infer_recommended_use(metric),
        )
        entries.append(entry)

    return MetricCatalog(
        entries=entries,
        categories=dict(categories),
        total_count=len(entries),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def filter_catalog(
    catalog: MetricCatalog,
    category: str | None = None,
    scope: str | None = None,
    search: str | None = None,
) -> MetricCatalog:
    """Filter catalog by category, scope, or text search in id/description.

    Args:
        catalog: Source catalog.
        category: Filter by category (exact match).
        scope: Filter by scope (exact match).
        search: Case-insensitive substring search in metric_id and description.

    Returns:
        New MetricCatalog with matching entries only.
    """
    filtered = list(catalog.entries)

    if category:
        filtered = [e for e in filtered if e.category == category]

    if scope:
        filtered = [e for e in filtered if e.scope == scope]

    if search:
        term = search.lower()
        filtered = [
            e for e in filtered if term in e.metric_id.lower() or term in e.description.lower()
        ]

    return MetricCatalog(
        entries=filtered,
        categories=catalog.categories,
        total_count=len(filtered),
        generated_at=catalog.generated_at,
    )


def format_catalog_markdown(catalog: MetricCatalog) -> str:
    """Format catalog as markdown with TOC grouped by category.

    Each entry shows description, scope, and rubric levels.
    """
    lines: list[str] = []
    lines.append("# Metric Catalog")
    lines.append("")
    lines.append(f"Total metrics: {catalog.total_count}")
    lines.append(f"Generated: {catalog.generated_at}")
    lines.append("")

    # Group entries by category
    by_cat: dict[str, list[CatalogEntry]] = {}
    for entry in catalog.entries:
        by_cat.setdefault(entry.category, []).append(entry)

    # TOC
    lines.append("## Table of Contents")
    lines.append("")
    for cat in sorted(by_cat.keys()):
        desc = catalog.categories.get(cat, "")
        label = f"{cat}" + (f" - {desc}" if desc else "")
        anchor = cat.lower().replace(" ", "-")
        lines.append(f"- [{label}](#{anchor})")
    lines.append("")

    # Sections
    for cat in sorted(by_cat.keys()):
        desc = catalog.categories.get(cat, "")
        lines.append(f"## {cat}")
        if desc:
            lines.append(f"*{desc}*")
        lines.append("")

        for entry in sorted(by_cat[cat], key=lambda e: e.metric_id):
            lines.append(f"### {entry.metric_id}")
            lines.append(f"- **Description**: {entry.description}")
            lines.append(f"- **Type**: {entry.metric_type}")
            lines.append(f"- **Scope**: {entry.scope}")
            lines.append(f"- **Rubric levels**: {entry.rubric_levels}")
            if entry.bundle_memberships:
                lines.append(f"- **Bundles**: {', '.join(entry.bundle_memberships)}")
            if entry.recommended_use:
                lines.append(f"- **Recommended use**: {entry.recommended_use}")
            lines.append("")

    return "\n".join(lines)


def format_catalog_html(catalog: MetricCatalog) -> str:
    """Format catalog as simple HTML page with a table.

    Uses inline CSS only (no external dependencies). All user content
    is escaped with html.escape.
    """
    h = html.escape

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Metric Catalog</title>")
    parts.append("<style>")
    parts.append("body { font-family: sans-serif; margin: 2em; }")
    parts.append("table { border-collapse: collapse; width: 100%; }")
    parts.append("th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }")
    parts.append("th { background: #f4f4f4; }")
    parts.append("tr:hover { background: #fafafa; }")
    parts.append(".filter-links a { margin-right: 1em; }")
    parts.append(".search-hint { color: #666; font-style: italic; margin-bottom: 1em; }")
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append(f"<h1>Metric Catalog ({h(str(catalog.total_count))} metrics)</h1>")
    parts.append(f"<p>Generated: {h(catalog.generated_at)}</p>")

    # Category filter links
    cats = sorted(set(e.category for e in catalog.entries))
    if cats:
        parts.append('<div class="filter-links">')
        parts.append("<strong>Filter by category:</strong> ")
        for cat in cats:
            parts.append(f'<a href="#{h(cat)}">{h(cat)}</a>')
        parts.append("</div>")

    # Search hint
    parts.append('<p class="search-hint">Use browser search (Ctrl+F) to find specific metrics.</p>')

    # Table
    parts.append("<table>")
    parts.append("<thead><tr>")
    parts.append("<th>ID</th><th>Type</th><th>Category</th>")
    parts.append("<th>Scope</th><th>Description</th><th>Rubric Levels</th>")
    parts.append("<th>Bundles</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")

    for entry in sorted(catalog.entries, key=lambda e: (e.category, e.metric_id)):
        parts.append(f'<tr id="{h(entry.metric_id)}">')
        parts.append(f"<td>{h(entry.metric_id)}</td>")
        parts.append(f"<td>{h(entry.metric_type)}</td>")
        parts.append(f'<td id="{h(entry.category)}">{h(entry.category)}</td>')
        parts.append(f"<td>{h(entry.scope)}</td>")
        parts.append(f"<td>{h(entry.description)}</td>")
        parts.append(f"<td>{entry.rubric_levels}</td>")
        bundles_str = ", ".join(entry.bundle_memberships)
        parts.append(f"<td>{h(bundles_str)}</td>")
        parts.append("</tr>")

    parts.append("</tbody>")
    parts.append("</table>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


def format_catalog_plain(catalog: MetricCatalog) -> str:
    """Format catalog as plain text tabular format."""
    lines: list[str] = []
    lines.append(f"Metric Catalog ({catalog.total_count} metrics)")
    lines.append(f"Generated: {catalog.generated_at}")
    lines.append("")

    # Column headers
    header = f"{'ID':<35} {'Type':<12} {'Category':<15} {'Scope':<10} {'Rubric':<7} Description"
    lines.append(header)
    lines.append("-" * len(header))

    for entry in sorted(catalog.entries, key=lambda e: (e.category, e.metric_id)):
        line = (
            f"{entry.metric_id:<35} "
            f"{entry.metric_type:<12} "
            f"{entry.category:<15} "
            f"{entry.scope:<10} "
            f"{entry.rubric_levels:<7} "
            f"{entry.description}"
        )
        lines.append(line)

    return "\n".join(lines)


def get_category_summary(catalog: MetricCatalog) -> dict[str, int]:
    """Count metrics per category.

    Returns:
        Dict mapping category name to metric count.
    """
    counts: dict[str, int] = {}
    for entry in catalog.entries:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return counts
