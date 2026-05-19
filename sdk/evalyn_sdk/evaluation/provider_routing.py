"""Metric-specific provider routing.

Routes different metrics to different judge providers based on
configurable rules with pattern matching and priority ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingRule:
    """A rule mapping metric patterns to a provider/model."""

    metric_pattern: str
    provider: str
    model: str = ""
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_pattern": self.metric_pattern,
            "provider": self.provider,
            "model": self.model,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingRule:
        return cls(
            metric_pattern=data["metric_pattern"],
            provider=data["provider"],
            model=data.get("model", ""),
            priority=data.get("priority", 0),
        )


@dataclass
class RoutingDecision:
    """The result of routing a single metric."""

    metric_id: str
    provider: str
    model: str
    rule_name: str = "default"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "provider": self.provider,
            "model": self.model,
            "rule_name": self.rule_name,
        }


@dataclass
class RoutingConfig:
    """Configuration for metric routing."""

    rules: list[RoutingRule] = field(default_factory=list)
    default_provider: str = "gemini"
    default_model: str = "gemini-2.5-flash"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rules": [r.as_dict() for r in self.rules],
            "default_provider": self.default_provider,
            "default_model": self.default_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingConfig:
        rules = [RoutingRule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            rules=rules,
            default_provider=data.get("default_provider", "gemini"),
            default_model=data.get("default_model", "gemini-2.5-flash"),
        )


@dataclass
class RoutingReport:
    """Summary of routing decisions across all metrics."""

    decisions: list[RoutingDecision] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    estimated_cost_savings_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": [d.as_dict() for d in self.decisions],
            "providers_used": list(self.providers_used),
            "estimated_cost_savings_pct": self.estimated_cost_savings_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingReport:
        decisions = [
            RoutingDecision(
                metric_id=d["metric_id"],
                provider=d["provider"],
                model=d["model"],
                rule_name=d.get("rule_name", "default"),
            )
            for d in data.get("decisions", [])
        ]
        return cls(
            decisions=decisions,
            providers_used=data.get("providers_used", []),
            estimated_cost_savings_pct=data.get("estimated_cost_savings_pct", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Routing Report")
        lines.append("-" * 40)
        for d in self.decisions:
            model_str = f" ({d.model})" if d.model else ""
            lines.append(f"  {d.metric_id} -> {d.provider}{model_str} [{d.rule_name}]")
        if self.providers_used:
            lines.append("")
            lines.append(f"Providers used: {', '.join(self.providers_used)}")
        if self.estimated_cost_savings_pct > 0:
            lines.append(
                f"Estimated cost savings: {self.estimated_cost_savings_pct:.1f}%"
            )
        return "\n".join(lines)


def match_metric_to_rule(
    metric_id: str, rules: list[RoutingRule]
) -> RoutingRule | None:
    """Find the best matching rule for a metric.

    Matching order: exact match first, then substring match.
    If multiple rules match at the same level, highest priority wins.
    Returns None if no rule matches.
    """
    exact_matches: list[RoutingRule] = []
    substring_matches: list[RoutingRule] = []

    for rule in rules:
        if rule.metric_pattern == metric_id:
            exact_matches.append(rule)
        elif rule.metric_pattern in metric_id:
            substring_matches.append(rule)

    if exact_matches:
        return max(exact_matches, key=lambda r: r.priority)
    if substring_matches:
        return max(substring_matches, key=lambda r: r.priority)
    return None


def route_metric(metric_id: str, config: RoutingConfig) -> RoutingDecision:
    """Route a single metric to a provider/model based on config rules."""
    rule = match_metric_to_rule(metric_id, config.rules)
    if rule:
        return RoutingDecision(
            metric_id=metric_id,
            provider=rule.provider,
            model=rule.model or config.default_model,
            rule_name=rule.metric_pattern,
        )
    return RoutingDecision(
        metric_id=metric_id,
        provider=config.default_provider,
        model=config.default_model,
        rule_name="default",
    )


def route_all_metrics(
    metric_ids: list[str], config: RoutingConfig
) -> RoutingReport:
    """Route all metrics and produce a summary report."""
    decisions: list[RoutingDecision] = []
    provider_set: set[str] = set()

    for mid in metric_ids:
        decision = route_metric(mid, config)
        decisions.append(decision)
        provider_set.add(decision.provider)

    providers_used = sorted(provider_set)

    # Estimate savings: if more than one provider is used, assume routing
    # to cheaper providers saves proportionally to non-default usage
    savings = 0.0
    if len(decisions) > 0 and len(providers_used) > 1:
        non_default = sum(
            1 for d in decisions if d.provider != config.default_provider
        )
        savings = round((non_default / len(decisions)) * 100.0, 1)

    return RoutingReport(
        decisions=decisions,
        providers_used=providers_used,
        estimated_cost_savings_pct=savings,
    )


def create_safety_routing() -> RoutingConfig:
    """Preset config: route safety metrics to gemini, quality to default."""
    return RoutingConfig(
        rules=[
            RoutingRule(
                metric_pattern="safety",
                provider="gemini",
                model="gemini-2.5-flash",
                priority=10,
            ),
            RoutingRule(
                metric_pattern="toxicity",
                provider="gemini",
                model="gemini-2.5-flash",
                priority=10,
            ),
            RoutingRule(
                metric_pattern="harm",
                provider="gemini",
                model="gemini-2.5-flash",
                priority=10,
            ),
        ],
        default_provider="openai",
        default_model="gpt-4o-mini",
    )


def create_cost_optimized_routing() -> RoutingConfig:
    """Preset config: route simple metrics to cheap models, complex to expensive."""
    return RoutingConfig(
        rules=[
            RoutingRule(
                metric_pattern="length",
                provider="gemini",
                model="gemini-2.0-flash-lite",
                priority=5,
            ),
            RoutingRule(
                metric_pattern="format",
                provider="gemini",
                model="gemini-2.0-flash-lite",
                priority=5,
            ),
            RoutingRule(
                metric_pattern="coherence",
                provider="openai",
                model="gpt-4o",
                priority=10,
            ),
            RoutingRule(
                metric_pattern="reasoning",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                priority=10,
            ),
        ],
        default_provider="gemini",
        default_model="gemini-2.5-flash",
    )
