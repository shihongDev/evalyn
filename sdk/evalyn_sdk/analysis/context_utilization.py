"""Context window utilization alerts.

Warn when spans approach context limits for their respective models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span

MODEL_CONTEXT_LIMITS = {
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-3.5": 200000,
}

DEFAULT_CONTEXT_LIMIT = 128000


@dataclass
class ContextAlert:
    """Alert for a span approaching its model's context limit."""

    span_id: str
    model: str
    tokens_used: int
    context_limit: int
    utilization_pct: float
    severity: str  # "warning" or "critical"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "context_limit": self.context_limit,
            "utilization_pct": round(self.utilization_pct, 2),
            "severity": self.severity,
        }


@dataclass
class ContextReport:
    """Aggregated context utilization report across spans."""

    alerts: List[ContextAlert] = field(default_factory=list)
    max_utilization_pct: float = 0.0
    mean_utilization_pct: float = 0.0
    models_at_risk: List[str] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alerts": [a.as_dict() for a in self.alerts],
            "max_utilization_pct": round(self.max_utilization_pct, 2),
            "mean_utilization_pct": round(self.mean_utilization_pct, 2),
            "models_at_risk": list(self.models_at_risk),
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextReport:
        alerts = []
        for a in data.get("alerts", []):
            alerts.append(
                ContextAlert(
                    span_id=a["span_id"],
                    model=a["model"],
                    tokens_used=a["tokens_used"],
                    context_limit=a["context_limit"],
                    utilization_pct=a.get("utilization_pct", 0.0),
                    severity=a.get("severity", "warning"),
                )
            )
        return cls(
            alerts=alerts,
            max_utilization_pct=data.get("max_utilization_pct", 0.0),
            mean_utilization_pct=data.get("mean_utilization_pct", 0.0),
            models_at_risk=data.get("models_at_risk", []),
            recommendation=data.get("recommendation", ""),
        )

    def format_text(self) -> str:
        lines = [
            "Context Window Utilization",
            f"  Max utilization:  {self.max_utilization_pct:.1f}%",
            f"  Mean utilization: {self.mean_utilization_pct:.1f}%",
        ]
        if self.models_at_risk:
            lines.append(f"  Models at risk:   {', '.join(self.models_at_risk)}")
        if self.recommendation:
            lines.append(f"  Recommendation:   {self.recommendation}")
        if self.alerts:
            lines.append("")
            lines.append("  Alerts:")
            for a in self.alerts:
                lines.append(
                    f"    [{a.severity.upper()}] span={a.span_id} model={a.model} "
                    f"{a.tokens_used}/{a.context_limit} ({a.utilization_pct:.1f}%)"
                )
        return "\n".join(lines)


def get_context_limit(model: str) -> int:
    """Lookup model context limit.

    Returns DEFAULT_CONTEXT_LIMIT for unknown models.
    """
    return MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)


def check_context_utilization(
    span: Span, threshold_pct: float = 80.0
) -> Optional[ContextAlert]:
    """Check if a span's token usage approaches its model's context limit.

    Args:
        span: Span with "model" and "total_tokens" attributes.
        threshold_pct: Utilization percentage above which to alert.

    Returns:
        ContextAlert if utilization exceeds threshold, else None.
    """
    model = span.attributes.get("model", "")
    tokens_used = span.attributes.get("total_tokens", 0)
    if tokens_used <= 0:
        return None

    limit = get_context_limit(model)
    utilization = (tokens_used / limit) * 100.0

    if utilization <= threshold_pct:
        return None

    severity = "critical" if utilization >= 95.0 else "warning"
    return ContextAlert(
        span_id=span.id,
        model=model,
        tokens_used=tokens_used,
        context_limit=limit,
        utilization_pct=utilization,
        severity=severity,
    )


def analyze_context_utilization(
    spans: List[Span], threshold_pct: float = 80.0
) -> ContextReport:
    """Analyze context utilization across all spans.

    Checks each span for context limit proximity, aggregates alerts,
    and recommends model upgrades when mean utilization is high.

    Args:
        spans: List of Span objects.
        threshold_pct: Utilization percentage above which to alert.

    Returns:
        ContextReport with alerts and summary statistics.
    """
    alerts: List[ContextAlert] = []
    utilizations: List[float] = []

    for span in spans:
        model = span.attributes.get("model", "")
        tokens_used = span.attributes.get("total_tokens", 0)
        if tokens_used <= 0:
            continue

        limit = get_context_limit(model)
        utilization = (tokens_used / limit) * 100.0
        utilizations.append(utilization)

        alert = check_context_utilization(span, threshold_pct)
        if alert is not None:
            alerts.append(alert)

    max_util = max(utilizations) if utilizations else 0.0
    mean_util = (sum(utilizations) / len(utilizations)) if utilizations else 0.0

    models_at_risk = sorted(set(a.model for a in alerts))
    recommendation = ""
    if mean_util > 70.0:
        recommendation = (
            "Mean context utilization exceeds 70% - consider upgrading to a "
            "model with a larger context window."
        )

    return ContextReport(
        alerts=alerts,
        max_utilization_pct=max_util,
        mean_utilization_pct=mean_util,
        models_at_risk=models_at_risk,
        recommendation=recommendation,
    )
