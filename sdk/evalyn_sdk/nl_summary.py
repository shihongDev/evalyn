"""
Natural language summary generation for evaluation analysis.

Provides heuristic summarization of metric results and LLM prompt
building for richer summaries. No external dependencies - pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""

    style: str = "executive"  # "executive", "technical", "brief"
    include_recommendations: bool = True
    max_metrics: int = 10

    def as_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "include_recommendations": self.include_recommendations,
            "max_metrics": self.max_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummaryConfig":
        return cls(
            style=data.get("style", "executive"),
            include_recommendations=data.get("include_recommendations", True),
            max_metrics=data.get("max_metrics", 10),
        )


@dataclass
class NLSummary:
    """Natural language summary of evaluation results."""

    title: str
    paragraphs: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "paragraphs": list(self.paragraphs),
            "recommendations": list(self.recommendations),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NLSummary":
        return cls(
            title=data.get("title", ""),
            paragraphs=data.get("paragraphs", []),
            recommendations=data.get("recommendations", []),
            generated_at=data.get("generated_at", ""),
        )


# ---------------------------------------------------------------------------
# Heuristic Summary
# ---------------------------------------------------------------------------

_PASS_RATE_LABELS = [
    (0.95, "excellent"),
    (0.85, "good"),
    (0.70, "acceptable"),
    (0.50, "concerning"),
    (0.0, "poor"),
]


def _pass_rate_label(rate: float) -> str:
    for threshold, label in _PASS_RATE_LABELS:
        if rate >= threshold:
            return label
    return "poor"


def _score_descriptor(score: float) -> str:
    if score >= 0.9:
        return "strong"
    if score >= 0.7:
        return "adequate"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "weak"
    return "very weak"


def build_heuristic_summary(
    metrics: Dict[str, float],
    pass_rate: float,
    total_items: int,
    config: Optional[SummaryConfig] = None,
) -> NLSummary:
    """Generate a heuristic summary from metric scores.

    Uses templates and thresholds to produce a plain English summary
    without any LLM calls. Includes pass rate overview, top/bottom
    metrics, and recommendations for low-scoring metrics.
    """
    cfg = config or SummaryConfig()
    paragraphs: List[str] = []
    recs: List[str] = []

    # Limit metrics
    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
    limited = sorted_metrics[: cfg.max_metrics]

    # Overall pass rate
    label = _pass_rate_label(pass_rate)
    paragraphs.append(
        f"The evaluation covered {total_items} items with an overall "
        f"pass rate of {pass_rate:.0%}, which is {label}."
    )

    if limited:
        # Top metrics
        top = limited[:3]
        if cfg.style == "brief":
            top_str = ", ".join(f"{m} ({s:.2f})" for m, s in top)
            paragraphs.append(f"Top metrics: {top_str}.")
        else:
            top_lines = [
                f"{m} scored {s:.2f} ({_score_descriptor(s)})" for m, s in top
            ]
            paragraphs.append("Strongest metrics: " + "; ".join(top_lines) + ".")

        # Bottom metrics
        bottom = [kv for kv in reversed(limited) if kv[1] < 0.7][:3]
        if bottom:
            if cfg.style == "brief":
                bot_str = ", ".join(f"{m} ({s:.2f})" for m, s in bottom)
                paragraphs.append(f"Metrics needing attention: {bot_str}.")
            else:
                bot_lines = [
                    f"{m} scored {s:.2f} ({_score_descriptor(s)})" for m, s in bottom
                ]
                paragraphs.append(
                    "Metrics requiring attention: " + "; ".join(bot_lines) + "."
                )

        # Technical style: add score distribution info
        if cfg.style == "technical" and len(limited) >= 2:
            scores = [s for _, s in limited]
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            paragraphs.append(
                f"Score range: {min_s:.2f} to {max_s:.2f}, "
                f"mean {avg:.2f} across {len(limited)} metrics."
            )

    # Recommendations
    if cfg.include_recommendations:
        recs = generate_recommendations(metrics)

    title = f"Evaluation Summary - {pass_rate:.0%} Pass Rate"

    return NLSummary(
        title=title,
        paragraphs=paragraphs,
        recommendations=recs,
    )


# ---------------------------------------------------------------------------
# LLM Prompt Builder
# ---------------------------------------------------------------------------


def build_summary_prompt(
    metrics: Dict[str, float],
    pass_rate: float,
    total_items: int,
    config: Optional[SummaryConfig] = None,
) -> str:
    """Construct an LLM prompt for natural language analysis summary.

    Returns a prompt string that can be sent to any LLM to generate
    a richer, more nuanced summary than the heuristic version.
    """
    cfg = config or SummaryConfig()

    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
    limited = sorted_metrics[: cfg.max_metrics]

    lines = [
        "You are an evaluation analyst. Summarize the following evaluation results "
        f"in a {cfg.style} style.",
        "",
        f"Total items evaluated: {total_items}",
        f"Overall pass rate: {pass_rate:.1%}",
        "",
        "Metric scores:",
    ]

    for name, score in limited:
        lines.append(f"  - {name}: {score:.3f}")

    lines.append("")

    if cfg.style == "executive":
        lines.append(
            "Write 2-3 paragraphs suitable for a non-technical audience. "
            "Focus on key takeaways and business impact."
        )
    elif cfg.style == "technical":
        lines.append(
            "Write 2-4 paragraphs with technical detail. "
            "Include score distributions, outliers, and statistical observations."
        )
    else:
        lines.append(
            "Write a brief 1-2 sentence summary of the key findings."
        )

    if cfg.include_recommendations:
        lines.append("")
        lines.append(
            "End with a bulleted list of recommendations prefixed with '- '."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------


def parse_llm_summary(response: str) -> NLSummary:
    """Parse an LLM response into an NLSummary.

    Splits text into paragraphs (blank-line separated) and extracts
    recommendations from bullet points (lines starting with '- ').
    """
    paragraphs: List[str] = []
    recommendations: List[str] = []

    current_paragraph: List[str] = []

    for line in response.strip().splitlines():
        stripped = line.strip()

        # Bullet point: treat as recommendation
        if stripped.startswith("- "):
            text = stripped[2:].strip()
            if text:
                recommendations.append(text)
            continue

        # Empty line: end current paragraph
        if not stripped:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        current_paragraph.append(stripped)

    # Flush remaining paragraph
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    title = "LLM Analysis Summary"
    if paragraphs:
        # Use first sentence of first paragraph as title if short enough
        first = paragraphs[0]
        first_sentence = first.split(".")[0]
        if len(first_sentence) <= 80:
            title = first_sentence

    return NLSummary(
        title=title,
        paragraphs=paragraphs,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Metric Insights
# ---------------------------------------------------------------------------


def generate_metric_insights(metrics: Dict[str, float]) -> List[str]:
    """Generate heuristic insights from metric scores.

    Returns human-readable insight strings describing each metric's
    performance level.
    """
    insights: List[str] = []

    for name, score in sorted(metrics.items(), key=lambda kv: kv[1], reverse=True):
        if score >= 0.9:
            insights.append(f"{name} is strong ({score:.2f})")
        elif score >= 0.7:
            insights.append(f"{name} is adequate ({score:.2f})")
        elif score >= 0.5:
            insights.append(f"{name} is moderate ({score:.2f})")
        elif score >= 0.3:
            insights.append(f"{name} needs attention ({score:.2f})")
        else:
            insights.append(f"{name} is critically low ({score:.2f})")

    return insights


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def generate_recommendations(
    metrics: Dict[str, float],
    threshold: float = 0.7,
) -> List[str]:
    """Suggest improvements for metrics below threshold.

    Returns a list of actionable recommendation strings for any
    metric scoring below the given threshold.
    """
    recs: List[str] = []

    for name, score in sorted(metrics.items(), key=lambda kv: kv[1]):
        if score >= threshold:
            continue

        if score < 0.3:
            recs.append(
                f"Urgently investigate {name} (score: {score:.2f}) - "
                f"consider reviewing the rubric or prompt."
            )
        elif score < 0.5:
            recs.append(
                f"Review and improve {name} (score: {score:.2f}) - "
                f"identify common failure patterns."
            )
        else:
            recs.append(
                f"Consider tuning {name} (score: {score:.2f}) - "
                f"small adjustments may push it above {threshold:.0%}."
            )

    return recs


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_nl_summary(summary: NLSummary) -> str:
    """Format an NLSummary as a human-readable string."""
    lines: List[str] = []

    lines.append(summary.title)
    lines.append("=" * len(summary.title))
    lines.append("")

    for para in summary.paragraphs:
        lines.append(para)
        lines.append("")

    if summary.recommendations:
        lines.append("Recommendations:")
        for rec in summary.recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

    return "\n".join(lines)
