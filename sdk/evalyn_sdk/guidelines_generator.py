"""Auto-generate annotation guidelines from metric definitions."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GuidelineSection:
    """A single section within an annotation guideline."""

    title: str
    content: str
    examples: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "examples": [dict(e) for e in self.examples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuidelineSection:
        return cls(
            title=data["title"],
            content=data["content"],
            examples=data.get("examples", []),
        )


@dataclass
class AnnotationGuideline:
    """Complete annotation guideline for a single metric."""

    metric_id: str
    metric_description: str
    sections: list[GuidelineSection] = field(default_factory=list)
    scale_description: str = ""
    edge_cases: list[str] = field(default_factory=list)
    generated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_description": self.metric_description,
            "sections": [s.as_dict() for s in self.sections],
            "scale_description": self.scale_description,
            "edge_cases": list(self.edge_cases),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationGuideline:
        return cls(
            metric_id=data["metric_id"],
            metric_description=data["metric_description"],
            sections=[
                GuidelineSection.from_dict(s) for s in data.get("sections", [])
            ],
            scale_description=data.get("scale_description", ""),
            edge_cases=data.get("edge_cases", []),
            generated_at=data.get("generated_at", ""),
        )


# ---------------------------------------------------------------------------
# Edge case extraction
# ---------------------------------------------------------------------------

_EDGE_CASE_KEYWORDS = [
    "borderline",
    "ambiguous",
    "may",
    "sometimes",
    "unless",
]


def extract_edge_cases(rubric: dict[str, str]) -> list[str]:
    """Identify potential edge cases from rubric text.

    Scans rubric descriptions for keywords that signal ambiguity:
    borderline, ambiguous, may, sometimes, unless.
    Returns de-duplicated list of sentences containing those keywords.
    """
    cases: list[str] = []
    seen: set[str] = set()

    for score, description in rubric.items():
        # Split into sentences (rough heuristic)
        sentences = re.split(r"(?<=[.!?])\s+", description)
        for sentence in sentences:
            lower = sentence.lower()
            for keyword in _EDGE_CASE_KEYWORDS:
                # Match keyword as a whole word
                if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
                    text = f"Score {score}: {sentence.strip()}"
                    if text not in seen:
                        seen.add(text)
                        cases.append(text)
                    break  # one match per sentence is enough

    return cases


# ---------------------------------------------------------------------------
# Guideline generation
# ---------------------------------------------------------------------------


def _build_overview_section(
    metric_id: str, description: str
) -> GuidelineSection:
    return GuidelineSection(
        title="Overview",
        content=(
            f"This guideline covers the '{metric_id}' metric. "
            f"{description}"
        ),
    )


def _build_scale_section(rubric: dict[str, str]) -> GuidelineSection:
    lines: list[str] = []
    for score in sorted(rubric.keys()):
        lines.append(f"- {score}: {rubric[score]}")
    return GuidelineSection(
        title="Scoring Scale",
        content="\n".join(lines),
    )


def _build_pass_fail_section(rubric: dict[str, str]) -> GuidelineSection:
    """Derive pass/fail criteria from the rubric.

    Convention: scores that parse as numbers >= half the max are 'pass'.
    If scores are not numeric, the first half (sorted) are fail and the rest pass.
    """
    keys = sorted(rubric.keys())
    try:
        numeric = {k: float(k) for k in keys}
        max_val = max(numeric.values()) if numeric else 1
        passing = [k for k, v in numeric.items() if v >= max_val / 2]
        failing = [k for k, v in numeric.items() if v < max_val / 2]
    except ValueError:
        mid = len(keys) // 2
        failing = keys[:mid] if mid > 0 else []
        passing = keys[mid:]

    pass_desc = ", ".join(passing) if passing else "none"
    fail_desc = ", ".join(failing) if failing else "none"
    content = (
        f"Passing scores: {pass_desc}\n"
        f"Failing scores: {fail_desc}"
    )
    return GuidelineSection(title="Pass/Fail Criteria", content=content)


def _build_examples_section(
    examples: list[dict[str, str]] | None,
) -> GuidelineSection:
    if not examples:
        return GuidelineSection(
            title="Examples",
            content="No examples provided.",
            examples=[],
        )
    return GuidelineSection(
        title="Examples",
        content=f"{len(examples)} example(s) provided below.",
        examples=list(examples),
    )


def _build_edge_cases_section(edge_cases: list[str]) -> GuidelineSection:
    if not edge_cases:
        return GuidelineSection(
            title="Edge Cases",
            content="No edge cases identified.",
        )
    lines = [f"- {c}" for c in edge_cases]
    return GuidelineSection(
        title="Edge Cases",
        content="\n".join(lines),
    )


def generate_guideline(
    metric_id: str,
    description: str,
    rubric: dict[str, str],
    examples: list[dict[str, str]] | None = None,
) -> AnnotationGuideline:
    """Convert a metric rubric into annotator-friendly guidelines.

    Produces sections: Overview, Scoring Scale, Pass/Fail Criteria,
    Examples, Edge Cases.
    """
    edge_cases = extract_edge_cases(rubric)

    sections = [
        _build_overview_section(metric_id, description),
        _build_scale_section(rubric),
        _build_pass_fail_section(rubric),
        _build_examples_section(examples),
        _build_edge_cases_section(edge_cases),
    ]

    scale_lines = [f"{s}: {rubric[s]}" for s in sorted(rubric.keys())]
    scale_description = "; ".join(scale_lines)

    return AnnotationGuideline(
        metric_id=metric_id,
        metric_description=description,
        sections=sections,
        scale_description=scale_description,
        edge_cases=edge_cases,
        generated_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_batch_guidelines(
    metrics: list[dict[str, Any]],
) -> list[AnnotationGuideline]:
    """Generate guidelines for multiple metrics.

    Each dict must have: metric_id (or id), description, rubric.
    Optional: examples.
    """
    results: list[AnnotationGuideline] = []
    for m in metrics:
        g = generate_guideline(
            metric_id=m.get("metric_id") or m["id"],
            description=m["description"],
            rubric=m["rubric"],
            examples=m.get("examples"),
        )
        results.append(g)
    return results


# ---------------------------------------------------------------------------
# Formatting: Markdown
# ---------------------------------------------------------------------------


def format_guideline_markdown(guideline: AnnotationGuideline) -> str:
    """Render a guideline as Markdown."""
    parts: list[str] = []
    parts.append(f"# {guideline.metric_id}")
    parts.append("")
    parts.append(f"*{guideline.metric_description}*")
    parts.append("")

    for section in guideline.sections:
        parts.append(f"## {section.title}")
        parts.append("")
        parts.append(section.content)
        parts.append("")

        if section.examples:
            for i, ex in enumerate(section.examples, 1):
                parts.append(f"### Example {i}")
                parts.append("")
                for key in ("input", "output", "label", "reasoning"):
                    if key in ex:
                        parts.append(f"- **{key}**: {ex[key]}")
                parts.append("")

    if guideline.edge_cases:
        parts.append("## Edge Cases Summary")
        parts.append("")
        for ec in guideline.edge_cases:
            parts.append(f"- {ec}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Formatting: HTML
# ---------------------------------------------------------------------------


def format_guideline_html(guideline: AnnotationGuideline) -> str:
    """Render a guideline as self-contained HTML with inline CSS.

    All dynamic content is escaped via html.escape.
    """
    esc = html.escape
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append("<meta charset=\"utf-8\">")
    parts.append(f"<title>{esc(guideline.metric_id)}</title>")
    parts.append("<style>")
    parts.append("body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }")
    parts.append("h1 { color: #333; } h2 { color: #555; border-bottom: 1px solid #ddd; }")
    parts.append(".example { background: #f9f9f9; padding: 10px; margin: 8px 0; border-left: 3px solid #4a90d9; }")
    parts.append(".edge-case { color: #c0392b; }")
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")

    parts.append(f"<h1>{esc(guideline.metric_id)}</h1>")
    parts.append(f"<p><em>{esc(guideline.metric_description)}</em></p>")

    for section in guideline.sections:
        parts.append(f"<h2>{esc(section.title)}</h2>")
        # Render content - preserve newlines
        content_html = esc(section.content).replace("\n", "<br>")
        parts.append(f"<p>{content_html}</p>")

        if section.examples:
            for i, ex in enumerate(section.examples, 1):
                parts.append("<div class=\"example\">")
                parts.append(f"<strong>Example {i}</strong><br>")
                for key in ("input", "output", "label", "reasoning"):
                    if key in ex:
                        parts.append(
                            f"<strong>{esc(key)}</strong>: {esc(str(ex[key]))}<br>"
                        )
                parts.append("</div>")

    if guideline.edge_cases:
        parts.append("<h2>Edge Cases Summary</h2>")
        parts.append("<ul>")
        for ec in guideline.edge_cases:
            parts.append(f"<li class=\"edge-case\">{esc(ec)}</li>")
        parts.append("</ul>")

    parts.append(f"<footer><small>Generated at {esc(guideline.generated_at)}</small></footer>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def format_guidelines_index(
    guidelines: list[AnnotationGuideline],
) -> str:
    """Generate a Markdown index page linking to each guideline."""
    parts: list[str] = []
    parts.append("# Annotation Guidelines Index")
    parts.append("")

    if not guidelines:
        parts.append("No guidelines generated.")
        parts.append("")
        return "\n".join(parts)

    parts.append("| # | Metric | Description | Sections | Edge Cases |")
    parts.append("|---|--------|-------------|----------|------------|")

    for i, g in enumerate(guidelines, 1):
        parts.append(
            f"| {i} | {g.metric_id} | {g.metric_description} "
            f"| {len(g.sections)} | {len(g.edge_cases)} |"
        )

    parts.append("")
    return "\n".join(parts)
