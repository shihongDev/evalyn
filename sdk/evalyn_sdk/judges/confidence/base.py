"""Base class for confidence estimation methods.

All confidence methods inherit from ConfidenceEstimator and implement
the estimate() method to calculate confidence from LLM outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfidenceResult:
    """Result from confidence estimation.

    Attributes:
        score: Confidence score in range [0, 1]. Higher = more confident.
        method: Name of the method used.
        details: Additional method-specific details.
    """

    score: float
    method: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp score to [0, 1]
        self.score = max(0.0, min(1.0, self.score))


class ConfidenceEstimator(ABC):
    """Base class for confidence estimation methods.

    Usage:
        estimator = LogprobsConfidence()
        result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
        print(f"Confidence: {result.score:.2f}")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this confidence method."""

    @abstractmethod
    def estimate(self, **kwargs) -> ConfidenceResult:
        """Estimate confidence from provided data.

        Each subclass requires specific keyword arguments:
        - LogprobsConfidence: logprobs (list of floats), top_logprobs (optional)
        - DeepConfConfidence: logprobs (list of floats)
        - SelfConsistencyConfidence: samples (list of result dicts) OR generate_fn (callable)
        - VerbalizedConfidence: response (str) OR parsed_response (dict)
        - PerplexityConfidence: logprobs (list of floats)

        Returns:
            ConfidenceResult with score in [0, 1] and method-specific details.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_confidence_estimator(method: str, **kwargs) -> ConfidenceEstimator:
    """Factory function to get a confidence estimator by name.

    Args:
        method: Method name - "logprobs", "deepconf", "consistency", "entropy", etc.
        **kwargs: Method-specific configuration

    Returns:
        Configured ConfidenceEstimator instance

    Raises:
        ValueError: If method is unknown
    """
    from .consistency import SelfConsistencyConfidence
    from .logprobs import (
        DeepConfConfidence,
        EntropyConfidence,
        LogprobsConfidence,
        PerplexityConfidence,
    )
    from .verbalized import VerbalizedConfidence

    methods = {
        "logprobs": LogprobsConfidence,
        "deepconf": DeepConfConfidence,
        "deepconf_bottom10": lambda **kw: DeepConfConfidence(strategy="bottom10", **kw),
        "deepconf_tail": lambda **kw: DeepConfConfidence(strategy="tail", **kw),
        "deepconf_average": lambda **kw: DeepConfConfidence(strategy="average", **kw),
        "perplexity": PerplexityConfidence,
        "entropy": EntropyConfidence,
        "consistency": SelfConsistencyConfidence,
        "self_consistency": SelfConsistencyConfidence,
        "verbalized": VerbalizedConfidence,
    }

    if method not in methods:
        available = ", ".join(methods.keys())
        raise ValueError(f"Unknown confidence method '{method}'. Available: {available}")

    return methods[method](**kwargs)


__all__ = ["ConfidenceResult", "ConfidenceEstimator", "get_confidence_estimator"]
