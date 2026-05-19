"""Model pricing lookup and cost estimation.

Provides a built-in pricing table for common LLM models and functions
to estimate evaluation costs from token counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
        "cached_per_1k": 0.0000375,
    },
    "gemini-2.5-pro": {
        "input_per_1k": 0.00125,
        "output_per_1k": 0.005,
        "cached_per_1k": 0.000315,
    },
    "gpt-4o": {
        "input_per_1k": 0.0025,
        "output_per_1k": 0.01,
    },
    "gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
    },
    "claude-sonnet-4-20250514": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
    "claude-haiku-3.5": {
        "input_per_1k": 0.0008,
        "output_per_1k": 0.004,
    },
}


@dataclass
class ModelPrice:
    """Pricing information for a single model."""

    model: str
    input_per_1k_tokens: float
    output_per_1k_tokens: float
    cached_per_1k_tokens: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_per_1k_tokens": self.input_per_1k_tokens,
            "output_per_1k_tokens": self.output_per_1k_tokens,
            "cached_per_1k_tokens": self.cached_per_1k_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPrice:
        return cls(
            model=data["model"],
            input_per_1k_tokens=data["input_per_1k_tokens"],
            output_per_1k_tokens=data["output_per_1k_tokens"],
            cached_per_1k_tokens=data.get("cached_per_1k_tokens", 0.0),
        )


@dataclass
class CostEstimate:
    """Estimated cost for a model invocation or evaluation run."""

    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }

    def format_text(self) -> str:
        lines = [
            f"Cost Estimate - {self.model}",
            f"  Input tokens:  {self.input_tokens:,}",
            f"  Output tokens: {self.output_tokens:,}",
        ]
        if self.cached_tokens > 0:
            lines.append(f"  Cached tokens: {self.cached_tokens:,}")
        lines.append(f"  Est. cost:     ${self.estimated_cost_usd:.6f}")
        return "\n".join(lines)


@dataclass
class PricingTable:
    """Collection of model prices for cost estimation."""

    prices: dict[str, ModelPrice] = field(default_factory=dict)

    def get_price(self, model: str) -> ModelPrice | None:
        """Look up pricing for a model. Returns None if not found."""
        return self.prices.get(model)

    def add_price(self, model: str, price: ModelPrice) -> None:
        """Add or update pricing for a model."""
        self.prices[model] = price

    def list_models(self) -> list[str]:
        """Return sorted list of model names in the table."""
        return sorted(self.prices.keys())

    def as_dict(self) -> dict[str, Any]:
        return {
            "prices": {k: v.as_dict() for k, v in self.prices.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PricingTable:
        prices = {
            k: ModelPrice.from_dict(v)
            for k, v in data.get("prices", {}).items()
        }
        return cls(prices=prices)


def get_default_pricing() -> PricingTable:
    """Return a PricingTable populated with built-in model prices."""
    table = PricingTable()
    for model, info in MODEL_PRICING.items():
        table.add_price(
            model,
            ModelPrice(
                model=model,
                input_per_1k_tokens=info["input_per_1k"],
                output_per_1k_tokens=info["output_per_1k"],
                cached_per_1k_tokens=info.get("cached_per_1k", 0.0),
            ),
        )
    return table


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    pricing: PricingTable | None = None,
) -> CostEstimate:
    """Compute cost from token counts.

    If the model is not found in the pricing table, estimated_cost_usd is 0.
    """
    table = pricing or get_default_pricing()
    price = table.get_price(model)

    cost = 0.0
    if price is not None:
        cost += (input_tokens / 1000.0) * price.input_per_1k_tokens
        cost += (output_tokens / 1000.0) * price.output_per_1k_tokens
        cost += (cached_tokens / 1000.0) * price.cached_per_1k_tokens

    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        estimated_cost_usd=cost,
    )


def estimate_run_cost(
    model: str,
    num_items: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
    pricing: PricingTable | None = None,
) -> CostEstimate:
    """Estimate total cost for a full evaluation run."""
    total_input = num_items * avg_input_tokens
    total_output = num_items * avg_output_tokens
    return estimate_cost(
        model=model,
        input_tokens=total_input,
        output_tokens=total_output,
        pricing=pricing,
    )


def compare_model_costs(
    models: list[str],
    input_tokens: int,
    output_tokens: int,
) -> list[CostEstimate]:
    """Compare costs across models, sorted cheapest first."""
    table = get_default_pricing()
    estimates = [
        estimate_cost(model, input_tokens, output_tokens, pricing=table)
        for model in models
    ]
    estimates.sort(key=lambda e: e.estimated_cost_usd)
    return estimates
