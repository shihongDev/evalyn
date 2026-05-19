"""Metric benchmarking: measure cost and latency per metric.

Analyze evaluation run data to produce per-metric timing and cost
breakdown. Identifies the slowest and most expensive metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricBenchmark:
    """Benchmark data for a single metric."""

    metric_id: str
    metric_type: str  # "objective" or "subjective"
    item_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def avg_cost_per_item(self) -> float:
        return self.total_cost_usd / self.item_count if self.item_count > 0 else 0.0

    @property
    def avg_tokens_per_item(self) -> int:
        total = self.total_input_tokens + self.total_output_tokens
        return total // self.item_count if self.item_count > 0 else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type,
            "item_count": self.item_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_item": round(self.avg_cost_per_item, 6),
            "avg_tokens_per_item": self.avg_tokens_per_item,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report for all metrics in a run."""

    metrics: list[MetricBenchmark] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def most_expensive(self) -> MetricBenchmark | None:
        """Metric with highest total cost."""
        return max(self.metrics, key=lambda m: m.total_cost_usd) if self.metrics else None

    @property
    def cheapest(self) -> MetricBenchmark | None:
        """Metric with lowest total cost."""
        return min(self.metrics, key=lambda m: m.total_cost_usd) if self.metrics else None

    @property
    def objective_count(self) -> int:
        return sum(1 for m in self.metrics if m.metric_type == "objective")

    @property
    def subjective_count(self) -> int:
        return sum(1 for m in self.metrics if m.metric_type == "subjective")

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "objective_count": self.objective_count,
            "subjective_count": self.subjective_count,
            "metrics": [m.as_dict() for m in self.metrics],
            "most_expensive": self.most_expensive.metric_id if self.most_expensive else None,
        }

    def format_text(self) -> str:
        lines = [
            "Metric Benchmark Report",
            "=" * 50,
            f"Total cost:     ${self.total_cost_usd:.4f}",
            f"Total tokens:   {self.total_tokens:,}",
            f"Metrics:        {len(self.metrics)} ({self.objective_count} objective, {self.subjective_count} subjective)",
            "",
            f"{'Metric':<30} {'Type':<12} {'Items':>6} {'Cost':>10} {'Tokens/Item':>12}",
            "-" * 72,
        ]
        for m in sorted(self.metrics, key=lambda x: x.total_cost_usd, reverse=True):
            cost_str = f"${m.total_cost_usd:.4f}" if m.total_cost_usd > 0 else "free"
            lines.append(
                f"{m.metric_id:<30} {m.metric_type:<12} {m.item_count:>6} {cost_str:>10} {m.avg_tokens_per_item:>12}"
            )
        return "\n".join(lines)


def benchmark_metrics(
    metric_results: list,
    metric_specs: list | None = None,
) -> BenchmarkReport:
    """Compute per-metric benchmarks from evaluation results.

    Args:
        metric_results: List of MetricResult objects from an EvalRun.
        metric_specs: Optional list of MetricSpec for type information.

    Returns:
        BenchmarkReport with per-metric cost and token breakdowns.
    """
    # Build type lookup from specs
    type_map: dict[str, str] = {}
    if metric_specs:
        for spec in metric_specs:
            type_map[spec.id] = spec.type

    # Aggregate by metric
    aggregated: dict[str, MetricBenchmark] = {}

    for result in metric_results:
        mid = result.metric_id
        if mid not in aggregated:
            aggregated[mid] = MetricBenchmark(
                metric_id=mid,
                metric_type=type_map.get(mid, "unknown"),
            )

        bench = aggregated[mid]
        bench.item_count += 1
        bench.total_input_tokens += result.input_tokens or 0
        bench.total_output_tokens += result.output_tokens or 0

    # Compute costs using pricing table
    from ..trace.instrumentation.providers._shared import calculate_cost

    for bench in aggregated.values():
        if bench.total_input_tokens > 0 or bench.total_output_tokens > 0:
            # Try to get model from first result with this metric
            model = "gemini-2.5-flash-lite"  # default
            for r in metric_results:
                if r.metric_id == bench.metric_id and r.model:
                    model = r.model
                    break
            bench.total_cost_usd = calculate_cost(
                model, bench.total_input_tokens, bench.total_output_tokens
            )

    benchmarks = sorted(aggregated.values(), key=lambda m: m.total_cost_usd, reverse=True)
    total_cost = sum(m.total_cost_usd for m in benchmarks)
    total_tokens = sum(m.total_input_tokens + m.total_output_tokens for m in benchmarks)

    return BenchmarkReport(
        metrics=benchmarks,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
    )
