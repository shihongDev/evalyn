"""
Multi-model comparison: compare same prompts across different LLM providers.

Provides dataclasses and pure functions for aggregating per-model stats,
pairwise comparisons, ranking, and cost-efficiency analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Aggregated evaluation stats for a single model."""

    model: str
    provider: str = ""
    avg_score: float = 0.0
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    item_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "avg_score": self.avg_score,
            "pass_rate": self.pass_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost_usd": self.avg_cost_usd,
            "item_count": self.item_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResult:
        return cls(
            model=data["model"],
            provider=data.get("provider", ""),
            avg_score=data.get("avg_score", 0.0),
            pass_rate=data.get("pass_rate", 0.0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            avg_cost_usd=data.get("avg_cost_usd", 0.0),
            item_count=data.get("item_count", 0),
        )


@dataclass
class ModelComparison:
    """Pairwise comparison between two models."""

    model_a: str
    model_b: str
    score_delta: float = 0.0
    cost_delta: float = 0.0
    latency_delta: float = 0.0
    winner: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "score_delta": self.score_delta,
            "cost_delta": self.cost_delta,
            "latency_delta": self.latency_delta,
            "winner": self.winner,
        }

    def format_text(self) -> str:
        lines = [f"{self.model_a} vs {self.model_b}"]
        lines.append(f"  Score delta: {self.score_delta:+.4f}")
        lines.append(f"  Cost delta:  ${self.cost_delta:+.6f}")
        lines.append(f"  Latency delta: {self.latency_delta:+.1f}ms")
        lines.append(f"  Winner: {self.winner}")
        return "\n".join(lines)


@dataclass
class MultiModelReport:
    """Full multi-model comparison report."""

    models: list[ModelResult] = field(default_factory=list)
    comparisons: list[ModelComparison] = field(default_factory=list)
    best_quality: str = ""
    best_cost: str = ""
    best_latency: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [m.as_dict() for m in self.models],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "best_quality": self.best_quality,
            "best_cost": self.best_cost,
            "best_latency": self.best_latency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiModelReport:
        return cls(
            models=[ModelResult.from_dict(m) for m in data.get("models", [])],
            comparisons=[],  # comparisons are derived, not stored
            best_quality=data.get("best_quality", ""),
            best_cost=data.get("best_cost", ""),
            best_latency=data.get("best_latency", ""),
        )

    def format_text(self) -> str:
        lines = ["MULTI-MODEL COMPARISON"]
        lines.append("-" * 40)

        if self.models:
            lines.append("Models:")
            for m in self.models:
                lines.append(
                    f"  {m.model}: score={m.avg_score:.4f} "
                    f"cost=${m.avg_cost_usd:.6f} "
                    f"latency={m.avg_latency_ms:.1f}ms "
                    f"(n={m.item_count})"
                )

        if self.comparisons:
            lines.append("")
            lines.append("Comparisons:")
            for c in self.comparisons:
                lines.append(f"  {c.model_a} vs {c.model_b} -> {c.winner}")

        lines.append("")
        if self.best_quality:
            lines.append(f"Best quality: {self.best_quality}")
        if self.best_cost:
            lines.append(f"Best cost: {self.best_cost}")
        if self.best_latency:
            lines.append(f"Best latency: {self.best_latency}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_model_result(
    model: str,
    scores: list[float],
    latencies: list[float] | None = None,
    costs: list[float] | None = None,
    provider: str = "",
) -> ModelResult:
    """Aggregate per-model stats from raw score/latency/cost lists."""
    if latencies is None:
        latencies = []
    if costs is None:
        costs = []
    n = len(scores)
    if n == 0:
        return ModelResult(model=model, provider=provider, item_count=0)

    avg_score = sum(scores) / n
    passed = sum(1 for s in scores if s >= 0.5)
    pass_rate = passed / n

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0

    return ModelResult(
        model=model,
        provider=provider,
        avg_score=avg_score,
        pass_rate=pass_rate,
        avg_latency_ms=avg_latency,
        avg_cost_usd=avg_cost,
        item_count=n,
    )


def compare_two_models(a: ModelResult, b: ModelResult) -> ModelComparison:
    """Pairwise comparison. Winner has higher score; ties broken by lower cost."""
    score_delta = a.avg_score - b.avg_score
    cost_delta = a.avg_cost_usd - b.avg_cost_usd
    latency_delta = a.avg_latency_ms - b.avg_latency_ms

    if score_delta > 0:
        winner = a.model
    elif score_delta < 0:
        winner = b.model
    else:
        # Tie-break by lower cost
        if cost_delta < 0:
            winner = a.model
        elif cost_delta > 0:
            winner = b.model
        else:
            winner = a.model  # default to a if identical

    return ModelComparison(
        model_a=a.model,
        model_b=b.model,
        score_delta=score_delta,
        cost_delta=cost_delta,
        latency_delta=latency_delta,
        winner=winner,
    )


def build_multimodel_report(results: list[ModelResult]) -> MultiModelReport:
    """All pairwise comparisons and find best per dimension."""
    if not results:
        return MultiModelReport()

    comparisons: list[ModelComparison] = []
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            comparisons.append(compare_two_models(a, b))

    best_quality = max(results, key=lambda r: r.avg_score).model
    best_cost = min(results, key=lambda r: r.avg_cost_usd).model
    best_latency = min(results, key=lambda r: r.avg_latency_ms).model

    return MultiModelReport(
        models=list(results),
        comparisons=comparisons,
        best_quality=best_quality,
        best_cost=best_cost,
        best_latency=best_latency,
    )


def rank_models(results: list[ModelResult], by: str = "score") -> list[ModelResult]:
    """Sort models by score (desc), cost (asc), or latency (asc)."""
    if by == "score":
        return sorted(results, key=lambda r: r.avg_score, reverse=True)
    elif by == "cost":
        return sorted(results, key=lambda r: r.avg_cost_usd)
    elif by == "latency":
        return sorted(results, key=lambda r: r.avg_latency_ms)
    return list(results)


def compute_cost_efficiency(results: list[ModelResult]) -> dict[str, float]:
    """Score per dollar for each model. Returns 0.0 if cost is zero."""
    out: dict[str, float] = {}
    for r in results:
        if r.avg_cost_usd > 0:
            out[r.model] = r.avg_score / r.avg_cost_usd
        else:
            out[r.model] = 0.0
    return out
