"""Metric prompt token count estimation.

Estimates the number of tokens each metric will consume before running
evaluation. Useful for cost planning and budget estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Average chars per token (rough estimate, varies by model/language)
CHARS_PER_TOKEN = 4


@dataclass
class MetricTokenEstimate:
    """Token estimate for a single metric."""

    metric_id: str
    metric_type: str
    prompt_tokens: int = 0       # estimated prompt/system tokens
    input_tokens_per_item: int = 0  # estimated per-item input tokens
    output_tokens_per_item: int = 0  # estimated per-item output tokens
    total_per_item: int = 0

    @property
    def is_free(self) -> bool:
        return self.metric_type == "objective"

    def estimate_for_dataset(self, item_count: int) -> int:
        """Estimate total tokens for a dataset."""
        if self.is_free:
            return 0
        return (self.prompt_tokens + self.input_tokens_per_item + self.output_tokens_per_item) * item_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "type": self.metric_type,
            "prompt_tokens": self.prompt_tokens,
            "input_tokens_per_item": self.input_tokens_per_item,
            "output_tokens_per_item": self.output_tokens_per_item,
            "total_per_item": self.total_per_item,
            "is_free": self.is_free,
        }


def estimate_metric_tokens(metric_spec) -> MetricTokenEstimate:
    """Estimate token usage for a metric.

    Args:
        metric_spec: MetricSpec object.

    Returns:
        MetricTokenEstimate with per-item and prompt token counts.
    """
    if metric_spec.type == "objective":
        return MetricTokenEstimate(
            metric_id=metric_spec.id,
            metric_type="objective",
        )

    config = metric_spec.config or {}
    prompt = config.get("prompt", "")
    rubric = config.get("rubric", [])

    # Prompt tokens: system prompt + rubric
    prompt_chars = len(prompt) + sum(len(r) for r in rubric)
    prompt_tokens = max(50, prompt_chars // CHARS_PER_TOKEN)  # minimum 50

    # Per-item: average input content + formatting overhead
    input_tokens = 200  # typical input/output content
    output_tokens = 40  # judge response (JSON with score, passed, reason)

    total = prompt_tokens + input_tokens + output_tokens

    return MetricTokenEstimate(
        metric_id=metric_spec.id,
        metric_type="subjective",
        prompt_tokens=prompt_tokens,
        input_tokens_per_item=input_tokens,
        output_tokens_per_item=output_tokens,
        total_per_item=total,
    )


def estimate_all_tokens(
    metric_specs: list,
    item_count: int = 0,
) -> dict[str, Any]:
    """Estimate total token usage across all metrics.

    Args:
        metric_specs: List of MetricSpec objects.
        item_count: Number of items (for total estimate).

    Returns:
        Dict with per-metric estimates and totals.
    """
    estimates = [estimate_metric_tokens(spec) for spec in metric_specs]

    total_per_item = sum(e.total_per_item for e in estimates)
    total_for_dataset = sum(e.estimate_for_dataset(item_count) for e in estimates)
    objective_count = sum(1 for e in estimates if e.is_free)
    subjective_count = len(estimates) - objective_count

    return {
        "metrics": [e.as_dict() for e in estimates],
        "total_per_item": total_per_item,
        "total_for_dataset": total_for_dataset,
        "item_count": item_count,
        "objective_count": objective_count,
        "subjective_count": subjective_count,
    }
