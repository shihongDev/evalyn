"""Evaluation dry-run: estimate cost and time before executing.

Provides cost and time estimates for an evaluation run without actually
calling any LLM APIs. Useful for budgeting and planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Average tokens per judge call by metric type (estimated from typical prompts)
_AVG_TOKENS_PER_JUDGE_CALL = {
    "input": 800,   # system prompt + rubric + input/output content
    "output": 150,  # judge response (JSON with score, passed, reason)
}

# Average time per evaluation (ms) based on metric type
_AVG_TIME_PER_EVAL_MS = {
    "objective": 5,       # local computation, very fast
    "subjective": 2000,   # LLM API call, typical latency
}


@dataclass
class DryRunEstimate:
    """Estimated cost and time for an evaluation run."""

    total_items: int
    total_metrics: int
    objective_metrics: int
    subjective_metrics: int
    total_evaluations: int  # items * metrics

    # Cost estimate
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float

    # Time estimate
    estimated_time_seconds: float
    estimated_time_with_workers: float  # adjusted for parallelism

    # Per-metric breakdown
    metric_estimates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_metrics": self.total_metrics,
            "objective_metrics": self.objective_metrics,
            "subjective_metrics": self.subjective_metrics,
            "total_evaluations": self.total_evaluations,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "estimated_time_seconds": round(self.estimated_time_seconds, 1),
            "estimated_time_with_workers": round(self.estimated_time_with_workers, 1),
            "metric_estimates": self.metric_estimates,
        }

    def format_text(self) -> str:
        """Format estimate as human-readable text."""
        lines = [
            "Evaluation Dry-Run Estimate",
            "=" * 40,
            f"Items:              {self.total_items}",
            f"Metrics:            {self.total_metrics} ({self.objective_metrics} objective, {self.subjective_metrics} subjective)",
            f"Total evaluations:  {self.total_evaluations}",
            "",
            "Cost estimate:",
            f"  Input tokens:     ~{self.estimated_input_tokens:,}",
            f"  Output tokens:    ~{self.estimated_output_tokens:,}",
            f"  Estimated cost:   ${self.estimated_cost_usd:.4f}",
            "",
            "Time estimate:",
            f"  Sequential:       ~{self.estimated_time_seconds:.0f}s",
            f"  With parallelism: ~{self.estimated_time_with_workers:.0f}s",
        ]

        if self.metric_estimates:
            lines.append("")
            lines.append("Per-metric breakdown:")
            for m in self.metric_estimates:
                cost_str = f"${m['estimated_cost_usd']:.4f}" if m["estimated_cost_usd"] > 0 else "free"
                lines.append(f"  {m['metric_id']:30} {m['type']:10} {cost_str}")

        return "\n".join(lines)


def estimate_eval_cost(
    item_count: int,
    metrics: list[Any],
    provider: str = "gemini",
    model: str | None = None,
    max_workers: int = 1,
    confidence_method: str = "none",
    confidence_samples: int = 3,
) -> DryRunEstimate:
    """Estimate cost and time for an evaluation run.

    Args:
        item_count: Number of dataset items.
        metrics: List of Metric instances.
        provider: LLM provider for subjective metrics.
        model: Specific model name (for cost lookup).
        max_workers: Number of parallel workers.
        confidence_method: Confidence estimation method.
        confidence_samples: Number of samples for consistency confidence.

    Returns:
        DryRunEstimate with cost and time projections.
    """
    from ..defaults import DEFAULT_MODELS_BY_PROVIDER
    from ..trace.instrumentation.providers._shared import (
        _match_model_costs,
    )

    if model is None:
        model = DEFAULT_MODELS_BY_PROVIDER.get(provider, "gemini-2.5-flash-lite")

    costs = _match_model_costs(model.lower())
    input_cost_per_1m = costs["input"] if costs else 1.0
    output_cost_per_1m = costs["output"] if costs else 1.0

    objective_count = 0
    subjective_count = 0
    metric_estimates = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time_ms = 0

    # Confidence multiplier: consistency sampling multiplies judge calls
    confidence_multiplier = 1
    if confidence_method == "consistency":
        confidence_multiplier = confidence_samples

    for metric in metrics:
        is_subjective = metric.spec.type == "subjective"

        if is_subjective:
            subjective_count += 1
            calls_per_item = confidence_multiplier
            input_tokens = _AVG_TOKENS_PER_JUDGE_CALL["input"] * calls_per_item * item_count
            output_tokens = _AVG_TOKENS_PER_JUDGE_CALL["output"] * calls_per_item * item_count
            cost = (input_tokens / 1_000_000) * input_cost_per_1m + (output_tokens / 1_000_000) * output_cost_per_1m
            time_ms = _AVG_TIME_PER_EVAL_MS["subjective"] * calls_per_item * item_count
        else:
            objective_count += 1
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            time_ms = _AVG_TIME_PER_EVAL_MS["objective"] * item_count

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_time_ms += time_ms

        metric_estimates.append({
            "metric_id": metric.spec.id,
            "type": metric.spec.type,
            "evaluations": item_count,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
        })

    total_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m + (total_output_tokens / 1_000_000) * output_cost_per_1m

    # Time with parallelism: subjective metrics benefit from workers
    sequential_time = total_time_ms / 1000
    parallel_time = sequential_time
    if max_workers > 1 and subjective_count > 0:
        # Objective metrics are fast and sequential overhead is negligible
        obj_time = _AVG_TIME_PER_EVAL_MS["objective"] * objective_count * item_count / 1000
        subj_time = (total_time_ms / 1000) - obj_time
        parallel_time = obj_time + (subj_time / min(max_workers, subjective_count * item_count))

    return DryRunEstimate(
        total_items=item_count,
        total_metrics=len(metrics),
        objective_metrics=objective_count,
        subjective_metrics=subjective_count,
        total_evaluations=item_count * len(metrics),
        estimated_input_tokens=total_input_tokens,
        estimated_output_tokens=total_output_tokens,
        estimated_cost_usd=total_cost,
        estimated_time_seconds=sequential_time,
        estimated_time_with_workers=parallel_time,
        metric_estimates=metric_estimates,
    )
