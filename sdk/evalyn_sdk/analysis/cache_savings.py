"""Prompt cache savings report.

Analyze how much prompt caching saved per run by aggregating
cache_creation_tokens, cache_read_tokens, and input_tokens from spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import Span


@dataclass
class CacheSavings:
    """Aggregated cache savings across all spans in a run."""

    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_input_tokens: int = 0
    actual_cost_usd: float = 0.0
    hypothetical_cost_usd: float = 0.0
    savings_usd: float = 0.0
    savings_pct: float = 0.0
    cache_hit_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_input_tokens": self.total_input_tokens,
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "hypothetical_cost_usd": round(self.hypothetical_cost_usd, 6),
            "savings_usd": round(self.savings_usd, 6),
            "savings_pct": round(self.savings_pct, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheSavings:
        return cls(
            total_cache_creation_tokens=data.get("total_cache_creation_tokens", 0),
            total_cache_read_tokens=data.get("total_cache_read_tokens", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            actual_cost_usd=data.get("actual_cost_usd", 0.0),
            hypothetical_cost_usd=data.get("hypothetical_cost_usd", 0.0),
            savings_usd=data.get("savings_usd", 0.0),
            savings_pct=data.get("savings_pct", 0.0),
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Prompt Cache Savings",
            f"  Cache creation tokens: {self.total_cache_creation_tokens}",
            f"  Cache read tokens:     {self.total_cache_read_tokens}",
            f"  Total input tokens:    {self.total_input_tokens}",
            f"  Actual cost:           ${self.actual_cost_usd:.4f}",
            f"  Hypothetical cost:     ${self.hypothetical_cost_usd:.4f}",
            f"  Savings:               ${self.savings_usd:.4f} ({self.savings_pct:.1f}%)",
            f"  Cache hit rate:        {self.cache_hit_rate:.2%}",
        ]
        return "\n".join(lines)


@dataclass
class CacheRecommendation:
    """Recommendation to cache a repeated prompt pattern."""

    prompt_pattern: str
    repetition_count: int
    cacheable: bool
    estimated_savings_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_pattern": self.prompt_pattern,
            "repetition_count": self.repetition_count,
            "cacheable": self.cacheable,
            "estimated_savings_pct": round(self.estimated_savings_pct, 2),
        }


def compute_cache_savings(spans: list[Span]) -> CacheSavings:
    """Aggregate cache savings from span attributes.

    Reads "cache_creation_tokens", "cache_read_tokens", "input_tokens",
    and "cost" from each span's attributes. Computes hypothetical cost
    as actual_cost + estimated savings from cache reads.

    Args:
        spans: List of Span objects with cache-related attributes.

    Returns:
        CacheSavings with aggregated totals.
    """
    total_cache_creation = 0
    total_cache_read = 0
    total_input = 0
    total_cost = 0.0

    for span in spans:
        attrs = span.attributes
        total_cache_creation += attrs.get("cache_creation_tokens", 0)
        total_cache_read += attrs.get("cache_read_tokens", 0)
        total_input += attrs.get("input_tokens", 0)
        total_cost += attrs.get("cost", 0.0)

    # Estimate average cost per token from spans that have both cost and input tokens
    avg_cost_per_token = 0.0
    cost_spans_tokens = 0
    cost_spans_cost = 0.0
    for span in spans:
        attrs = span.attributes
        inp = attrs.get("input_tokens", 0)
        c = attrs.get("cost", 0.0)
        if inp > 0 and c > 0:
            cost_spans_tokens += inp
            cost_spans_cost += c
    if cost_spans_tokens > 0:
        avg_cost_per_token = cost_spans_cost / cost_spans_tokens

    # Hypothetical cost: what it would have cost without caching
    # Cache reads would have been full-price input tokens
    savings_estimate = total_cache_read * avg_cost_per_token
    hypothetical = total_cost + savings_estimate
    savings_pct = (savings_estimate / hypothetical * 100.0) if hypothetical > 0 else 0.0

    # Cache hit rate: fraction of cacheable tokens that were cache reads
    cacheable_total = total_cache_creation + total_cache_read
    hit_rate = (total_cache_read / cacheable_total) if cacheable_total > 0 else 0.0

    return CacheSavings(
        total_cache_creation_tokens=total_cache_creation,
        total_cache_read_tokens=total_cache_read,
        total_input_tokens=total_input,
        actual_cost_usd=total_cost,
        hypothetical_cost_usd=hypothetical,
        savings_usd=savings_estimate,
        savings_pct=savings_pct,
        cache_hit_rate=hit_rate,
    )


def analyze_prompt_patterns(spans: list[Span]) -> list[CacheRecommendation]:
    """Group spans by first 100 chars of input text and recommend caching.

    High-repetition groups (repetition_count >= 3) are marked as cacheable.

    Args:
        spans: List of Span objects.

    Returns:
        List of CacheRecommendation sorted by repetition count descending.
    """
    groups: dict[str, int] = {}
    for span in spans:
        text = str(span.attributes.get("input_text", ""))
        if not text:
            continue
        pattern = text[:100]
        groups[pattern] = groups.get(pattern, 0) + 1

    recs = []
    for pattern, count in groups.items():
        cacheable = count >= 3
        # Estimate: if cacheable, subsequent calls save ~90% of that pattern's cost
        est_savings = 90.0 * (1 - 1 / count) if cacheable else 0.0
        recs.append(
            CacheRecommendation(
                prompt_pattern=pattern,
                repetition_count=count,
                cacheable=cacheable,
                estimated_savings_pct=round(est_savings, 2),
            )
        )

    recs.sort(key=lambda r: r.repetition_count, reverse=True)
    return recs


def build_cache_report(spans: list[Span]) -> dict[str, Any]:
    """Full cache savings report combining savings and recommendations.

    Args:
        spans: List of Span objects.

    Returns:
        Dict with "savings" and "recommendations" keys.
    """
    savings = compute_cache_savings(spans)
    recommendations = analyze_prompt_patterns(spans)
    return {
        "savings": savings.as_dict(),
        "recommendations": [r.as_dict() for r in recommendations],
    }
