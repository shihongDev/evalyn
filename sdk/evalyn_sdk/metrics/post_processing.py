"""Metric output post-processing: pluggable transforms on raw judge output.

Apply score transformations after metric evaluation but before final
result recording. Built-in processors: clamp, threshold, invert, round.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class PostProcessor:
    """A named score post-processor."""

    name: str
    transform: Callable[[float], float]
    description: str = ""


# Built-in post-processors
def score_clamp(min_val: float = 0.0, max_val: float = 1.0) -> PostProcessor:
    """Clamp score to [min_val, max_val] range."""
    return PostProcessor(
        name=f"clamp({min_val},{max_val})",
        transform=lambda s: max(min_val, min(max_val, s)),
        description=f"Clamp score to [{min_val}, {max_val}]",
    )


def binary_threshold(threshold: float = 0.5) -> PostProcessor:
    """Convert continuous score to binary (0.0 or 1.0) at threshold."""
    return PostProcessor(
        name=f"binary_threshold({threshold})",
        transform=lambda s: 1.0 if s >= threshold else 0.0,
        description=f"Binary: 1.0 if score >= {threshold}, else 0.0",
    )


def invert_score() -> PostProcessor:
    """Invert score: 1.0 - score."""
    return PostProcessor(
        name="invert",
        transform=lambda s: 1.0 - s,
        description="Invert: 1.0 - score",
    )


def score_round(decimals: int = 2) -> PostProcessor:
    """Round score to N decimal places."""
    return PostProcessor(
        name=f"round({decimals})",
        transform=lambda s: round(s, decimals),
        description=f"Round to {decimals} decimal places",
    )


def scale(factor: float) -> PostProcessor:
    """Multiply score by a factor."""
    return PostProcessor(
        name=f"scale({factor})",
        transform=lambda s: s * factor,
        description=f"Multiply score by {factor}",
    )


class PostProcessorChain:
    """Chain of post-processors applied in order.

    Usage:
        chain = PostProcessorChain([
            score_clamp(0, 1),
            score_round(2),
        ])
        processed_score = chain.apply(0.8567)  # -> 0.86
    """

    def __init__(self, processors: list[PostProcessor] | None = None):
        self.processors = processors or []

    def add(self, processor: PostProcessor) -> PostProcessorChain:
        """Add a processor to the chain. Returns self for fluent API."""
        self.processors.append(processor)
        return self

    def apply(self, score: float) -> float:
        """Apply all processors in order.

        Args:
            score: Raw score value.

        Returns:
            Processed score.
        """
        result = score
        for proc in self.processors:
            try:
                result = proc.transform(result)
            except Exception:
                pass  # skip failing processors
        return result

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.processors]

    def as_dict(self) -> dict[str, Any]:
        return {
            "processors": [{"name": p.name, "description": p.description} for p in self.processors],
        }


# Registry of built-in processors by name
BUILTIN_PROCESSORS: dict[str, Callable] = {
    "clamp": score_clamp,
    "binary_threshold": binary_threshold,
    "invert": invert_score,
    "round": score_round,
    "scale": scale,
}


def build_chain_from_config(config: list[dict[str, Any]]) -> PostProcessorChain:
    """Build a post-processor chain from YAML/dict config.

    Config format:
        [
            {"name": "clamp", "min": 0, "max": 1},
            {"name": "round", "decimals": 2},
        ]

    Args:
        config: List of processor definitions.

    Returns:
        PostProcessorChain ready to use.
    """
    chain = PostProcessorChain()

    for entry in config:
        name = entry.get("name", "")
        factory = BUILTIN_PROCESSORS.get(name)
        if factory is None:
            continue

        # Pass config params to factory (excluding "name")
        params = {k: v for k, v in entry.items() if k != "name"}
        try:
            processor = factory(**params) if params else factory()
            chain.add(processor)
        except TypeError:
            pass  # skip misconfigured processors

    return chain
