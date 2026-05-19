"""
Judge routing: route evaluation to different judge models per metric.

Provides a pure-Python abstraction for assigning specialized judge models
to metrics based on explicit overrides, category-based rules, or defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class JudgeModel:
    """A judge model with its capabilities and quality tier."""

    model_id: str
    provider: str
    quality_tier: str  # "fast" / "standard" / "premium"
    specializations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "quality_tier": self.quality_tier,
            "specializations": list(self.specializations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeModel:
        return cls(
            model_id=data["model_id"],
            provider=data["provider"],
            quality_tier=data["quality_tier"],
            specializations=list(data.get("specializations", [])),
        )


@dataclass
class JudgeRoute:
    """Assignment of a metric to a specific judge model."""

    metric_id: str
    assigned_model: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "assigned_model": self.assigned_model,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeRoute:
        return cls(
            metric_id=data["metric_id"],
            assigned_model=data["assigned_model"],
            reason=data.get("reason", ""),
        )


@dataclass
class JudgeRoutingConfig:
    """Configuration for routing metrics to judge models."""

    default_model: str = "gemini-2.0-flash"
    routes: dict[str, str] = field(default_factory=dict)  # metric_id -> model_id
    category_routes: dict[str, str] = field(default_factory=dict)  # category -> model_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_model": self.default_model,
            "routes": dict(self.routes),
            "category_routes": dict(self.category_routes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeRoutingConfig:
        return cls(
            default_model=data.get("default_model", "gemini-2.0-flash"),
            routes=dict(data.get("routes", {})),
            category_routes=dict(data.get("category_routes", {})),
        )


@dataclass
class JudgeRoutingResult:
    """Result of routing all metrics to judge models."""

    routes: list[JudgeRoute] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    total_metrics: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "routes": [r.as_dict() for r in self.routes],
            "models_used": list(self.models_used),
            "total_metrics": self.total_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeRoutingResult:
        return cls(
            routes=[JudgeRoute.from_dict(r) for r in data.get("routes", [])],
            models_used=list(data.get("models_used", [])),
            total_metrics=data.get("total_metrics", 0),
        )


# ---------------------------------------------------------------------------
# Built-in Judges
# ---------------------------------------------------------------------------

BUILTIN_JUDGES: list[JudgeModel] = [
    JudgeModel(
        model_id="gemini-2.0-flash",
        provider="google",
        quality_tier="fast",
        specializations=["general"],
    ),
    JudgeModel(
        model_id="gemini-2.0-pro",
        provider="google",
        quality_tier="premium",
        specializations=["general"],
    ),
    JudgeModel(
        model_id="claude-sonnet",
        provider="anthropic",
        quality_tier="standard",
        specializations=["safety", "quality"],
    ),
    JudgeModel(
        model_id="gpt-4o",
        provider="openai",
        quality_tier="standard",
        specializations=["general"],
    ),
]


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------


def route_metric(
    metric_id: str,
    metric_category: str,
    config: JudgeRoutingConfig,
) -> JudgeRoute:
    """Route a single metric to a judge model.

    Priority: specific route > category route > default model.
    """
    # 1. Check specific metric override
    if metric_id in config.routes:
        return JudgeRoute(
            metric_id=metric_id,
            assigned_model=config.routes[metric_id],
            reason=f"explicit route for metric '{metric_id}'",
        )

    # 2. Check category route
    if metric_category in config.category_routes:
        return JudgeRoute(
            metric_id=metric_id,
            assigned_model=config.category_routes[metric_category],
            reason=f"category route for '{metric_category}'",
        )

    # 3. Fall back to default
    return JudgeRoute(
        metric_id=metric_id,
        assigned_model=config.default_model,
        reason="default model",
    )


def route_all_metrics(
    metrics: list[dict[str, str]],
    config: JudgeRoutingConfig,
) -> JudgeRoutingResult:
    """Route all metrics to judge models.

    Each metric dict must have 'id' and 'category' keys.
    """
    routes: list[JudgeRoute] = []
    models_seen: dict[str, None] = {}  # ordered set

    for metric in metrics:
        route = route_metric(metric["id"], metric["category"], config)
        routes.append(route)
        models_seen[route.assigned_model] = None

    return JudgeRoutingResult(
        routes=routes,
        models_used=list(models_seen.keys()),
        total_metrics=len(routes),
    )


def estimate_routing_cost(
    result: JudgeRoutingResult,
    cost_per_model: dict[str, float],
) -> float:
    """Estimate total cost based on model assignments.

    Models not in cost_per_model are treated as zero cost.
    """
    total = 0.0
    for route in result.routes:
        total += cost_per_model.get(route.assigned_model, 0.0)
    return total


def format_routing_plan(result: JudgeRoutingResult) -> str:
    """Format a routing result as a human-readable plan."""
    lines: list[str] = []
    lines.append(f"Routing Plan ({result.total_metrics} metrics)")
    lines.append(f"Models: {', '.join(result.models_used)}")
    lines.append("")
    for route in result.routes:
        reason_suffix = f" ({route.reason})" if route.reason else ""
        lines.append(f"  {route.metric_id} -> {route.assigned_model}{reason_suffix}")
    return "\n".join(lines)
