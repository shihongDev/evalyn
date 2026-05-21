"""Metric rubric preview: show the exact judge prompt before evaluation.

Displays the full prompt that will be sent to the LLM judge for each
subjective metric, with template variables filled in with sample data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPreview:
    """Preview of a metric's evaluation configuration."""

    metric_id: str
    metric_type: str
    description: str
    prompt: str | None = None  # full judge prompt (subjective only)
    rubric: list[str] | None = None  # rubric criteria
    threshold: float | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "metric_id": self.metric_id,
            "type": self.metric_type,
            "description": self.description,
        }
        if self.prompt:
            d["prompt"] = self.prompt
        if self.rubric:
            d["rubric"] = self.rubric
        if self.threshold is not None:
            d["threshold"] = self.threshold
        return d

    def format_text(self) -> str:
        lines = [
            f"Metric: {self.metric_id} ({self.metric_type})",
            f"  Description: {self.description}",
        ]
        if self.threshold is not None:
            lines.append(f"  Threshold:   {self.threshold}")
        if self.rubric:
            lines.append(f"  Rubric ({len(self.rubric)} criteria):")
            for i, r in enumerate(self.rubric, 1):
                lines.append(f"    {i}. {r}")
        if self.prompt:
            lines.append("  Prompt preview:")
            # Show first 200 chars
            preview = self.prompt[:200]
            if len(self.prompt) > 200:
                preview += "..."
            lines.append(f"    {preview}")
        return "\n".join(lines)


def preview_metric(metric_spec) -> MetricPreview:
    """Generate a preview of a metric's configuration.

    Args:
        metric_spec: MetricSpec object.

    Returns:
        MetricPreview with prompt, rubric, and config details.
    """
    config = metric_spec.config or {}
    return MetricPreview(
        metric_id=metric_spec.id,
        metric_type=metric_spec.type,
        description=metric_spec.description,
        prompt=config.get("prompt"),
        rubric=config.get("rubric"),
        threshold=config.get("threshold"),
        config=config,
    )


def preview_all_metrics(metric_specs: list) -> list[MetricPreview]:
    """Generate previews for all metrics.

    Args:
        metric_specs: List of MetricSpec objects.

    Returns:
        List of MetricPreview objects.
    """
    return [preview_metric(spec) for spec in metric_specs]


def preview_from_template(template_id: str) -> MetricPreview | None:
    """Preview a metric from its template ID.

    Looks up the template in the subjective registry and returns
    its prompt and rubric.

    Args:
        template_id: Template ID (e.g., "helpfulness_accuracy").

    Returns:
        MetricPreview if template found, None otherwise.
    """
    try:
        from .subjective import SUBJECTIVE_REGISTRY

        for tpl in SUBJECTIVE_REGISTRY:
            if tpl["id"] == template_id:
                config = tpl.get("config", {})
                return MetricPreview(
                    metric_id=tpl["id"],
                    metric_type="subjective",
                    description=tpl.get("description", ""),
                    prompt=tpl.get("prompt"),
                    rubric=config.get("rubric"),
                    threshold=config.get("threshold"),
                    config=config,
                )
    except ImportError:
        pass
    return None


def estimate_prompt_tokens(preview: MetricPreview) -> int:
    """Estimate token count for the metric's judge prompt.

    Uses a rough 4 chars per token estimate.

    Args:
        preview: MetricPreview to estimate.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    if preview.prompt:
        total_chars += len(preview.prompt)
    if preview.rubric:
        total_chars += sum(len(r) for r in preview.rubric)
    # Add overhead for system prompt structure
    total_chars += 200  # JSON formatting, instructions
    return total_chars // 4  # rough estimate
