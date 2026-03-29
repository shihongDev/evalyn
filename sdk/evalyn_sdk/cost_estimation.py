"""Simulation cost estimation: estimate token costs before running.

Provides cost models for common LLM providers, token approximation,
per-mode cost breakdowns, and multi-model comparison tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Pricing model for a single LLM."""

    model_name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    currency: str = "USD"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CostModel:
        return cls(
            model_name=data["model_name"],
            input_cost_per_1k=data["input_cost_per_1k"],
            output_cost_per_1k=data["output_cost_per_1k"],
            currency=data.get("currency", "USD"),
        )


@dataclass
class CostEstimate:
    """Estimated cost for a simulation run."""

    total_tokens: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    breakdown: List[Dict[str, Any]]
    currency: str = "USD"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "breakdown": list(self.breakdown),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CostEstimate:
        return cls(
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            breakdown=data.get("breakdown", []),
            currency=data.get("currency", "USD"),
        )


@dataclass
class CostReport:
    """Full cost report bundling estimate, model, and run metadata."""

    estimate: CostEstimate
    model: CostModel
    item_count: int
    is_dry_run: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate.as_dict(),
            "model": self.model.as_dict(),
            "item_count": self.item_count,
            "is_dry_run": self.is_dry_run,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CostReport:
        return cls(
            estimate=CostEstimate.from_dict(data["estimate"]),
            model=CostModel.from_dict(data["model"]),
            item_count=data.get("item_count", 0),
            is_dry_run=data.get("is_dry_run", True),
        )


# ---------------------------------------------------------------------------
# Default Cost Models
# ---------------------------------------------------------------------------

DEFAULT_COST_MODELS: Dict[str, CostModel] = {
    "gemini-2.0-flash": CostModel(
        model_name="gemini-2.0-flash",
        input_cost_per_1k=0.075,
        output_cost_per_1k=0.30,
    ),
    "gpt-4o": CostModel(
        model_name="gpt-4o",
        input_cost_per_1k=2.50,
        output_cost_per_1k=10.00,
    ),
    "gpt-4o-mini": CostModel(
        model_name="gpt-4o-mini",
        input_cost_per_1k=0.15,
        output_cost_per_1k=0.60,
    ),
    "claude-sonnet": CostModel(
        model_name="claude-sonnet",
        input_cost_per_1k=3.00,
        output_cost_per_1k=15.00,
    ),
}


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Approximate token count: ceil(len(text) / 4)."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_simulation_cost(
    item_count: int,
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 300,
    model: Optional[CostModel] = None,
    modes: Optional[Dict[str, int]] = None,
) -> CostEstimate:
    """Estimate total cost for a simulation run.

    Parameters
    ----------
    item_count:
        Number of items to simulate.
    avg_input_tokens:
        Average input tokens per item.
    avg_output_tokens:
        Average output tokens per item.
    model:
        Cost model to use. Defaults to gemini-2.0-flash.
    modes:
        Optional dict of mode_name to item_count for per-mode breakdown.
        If provided, item_count is ignored and the sum of mode counts is used.
    """
    if model is None:
        model = DEFAULT_COST_MODELS["gemini-2.0-flash"]

    breakdown: List[Dict[str, Any]] = []

    if modes:
        total_input = 0
        total_output = 0
        total_cost = 0.0
        for mode_name, mode_count in modes.items():
            m_input = mode_count * avg_input_tokens
            m_output = mode_count * avg_output_tokens
            m_cost = (
                (m_input / 1000) * model.input_cost_per_1k
                + (m_output / 1000) * model.output_cost_per_1k
            )
            total_input += m_input
            total_output += m_output
            total_cost += m_cost
            breakdown.append({
                "mode": mode_name,
                "items": mode_count,
                "input_tokens": m_input,
                "output_tokens": m_output,
                "cost": round(m_cost, 6),
            })
    else:
        total_input = item_count * avg_input_tokens
        total_output = item_count * avg_output_tokens
        total_cost = (
            (total_input / 1000) * model.input_cost_per_1k
            + (total_output / 1000) * model.output_cost_per_1k
        )

    return CostEstimate(
        total_tokens=total_input + total_output,
        total_cost=round(total_cost, 6),
        input_tokens=total_input,
        output_tokens=total_output,
        breakdown=breakdown,
        currency=model.currency,
    )


def build_cost_report(
    item_count: int,
    model_name: str = "gemini-2.0-flash",
    modes: Optional[Dict[str, int]] = None,
) -> CostReport:
    """Build a full cost report for a simulation.

    Uses DEFAULT_COST_MODELS to look up pricing. Falls back to
    gemini-2.0-flash if model_name is not recognized.
    """
    model = DEFAULT_COST_MODELS.get(model_name)
    if model is None:
        model = DEFAULT_COST_MODELS["gemini-2.0-flash"]

    estimate = estimate_simulation_cost(
        item_count=item_count, model=model, modes=modes,
    )
    effective_count = sum(modes.values()) if modes else item_count
    return CostReport(
        estimate=estimate,
        model=model,
        item_count=effective_count,
        is_dry_run=True,
    )


def format_cost_report(report: CostReport) -> str:
    """Format a cost report as a human-readable string."""
    lines: List[str] = []
    est = report.estimate
    lines.append("Cost Estimation Report")
    lines.append("=" * 40)
    lines.append(f"Model: {report.model.model_name}")
    lines.append(f"Items: {report.item_count}")
    lines.append(f"Dry run: {report.is_dry_run}")
    lines.append("")
    lines.append(f"Input tokens:  {est.input_tokens:,}")
    lines.append(f"Output tokens: {est.output_tokens:,}")
    lines.append(f"Total tokens:  {est.total_tokens:,}")
    lines.append(f"Total cost:    {est.currency} {est.total_cost:.4f}")

    if est.breakdown:
        lines.append("")
        lines.append("Per-mode breakdown:")
        for entry in est.breakdown:
            lines.append(
                f"  {entry['mode']}: {entry['items']} items, "
                f"{entry['input_tokens']} in / {entry['output_tokens']} out, "
                f"{est.currency} {entry['cost']:.4f}"
            )

    return "\n".join(lines)


def compare_model_costs(
    item_count: int,
    models: Optional[List[str]] = None,
) -> str:
    """Compare costs across models in a table format.

    If models is None, all DEFAULT_COST_MODELS are compared.
    """
    if models is None:
        model_names = sorted(DEFAULT_COST_MODELS.keys())
    else:
        model_names = list(models)

    lines: List[str] = []
    header = f"{'Model':<20} {'Input/1K':>10} {'Output/1K':>10} {'Est. Cost':>12}"
    lines.append("Model Cost Comparison")
    lines.append("=" * 56)
    lines.append(header)
    lines.append("-" * 56)

    for name in model_names:
        m = DEFAULT_COST_MODELS.get(name)
        if m is None:
            continue
        est = estimate_simulation_cost(item_count=item_count, model=m)
        lines.append(
            f"{name:<20} {m.input_cost_per_1k:>10.4f} "
            f"{m.output_cost_per_1k:>10.4f} {est.total_cost:>12.4f}"
        )

    lines.append("-" * 56)
    lines.append(f"Items: {item_count}")
    return "\n".join(lines)
