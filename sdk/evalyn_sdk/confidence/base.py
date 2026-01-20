"""Base class for confidence estimation methods.

All confidence methods inherit from ConfidenceEstimator and implement
the estimate() method to calculate confidence from LLM outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    details: Dict[str, Any] = None

    def __post_init__(self):
        # Clamp score to [0, 1]
        self.score = max(0.0, min(1.0, self.score))
        if self.details is None:
            self.details = {}


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
        pass

    @abstractmethod
    def estimate(self, **kwargs) -> ConfidenceResult:
        """Estimate confidence from provided data.

        Args:
            **kwargs: Method-specific arguments (logprobs, samples, etc.)

        Returns:
            ConfidenceResult with score and details.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_confidence_estimator(method: str, **kwargs) -> ConfidenceEstimator:
    """Factory function to get a confidence estimator by name.

    Args:
        method: Method name - "logprobs", "consistency", "entropy", "verbalized"
        **kwargs: Method-specific configuration

    Returns:
        Configured ConfidenceEstimator instance

    Raises:
        ValueError: If method is unknown
    """
    from .logprobs import LogprobsConfidence, PerplexityConfidence
    from .consistency import SelfConsistencyConfidence
    from .verbalized import VerbalizedConfidence

    methods = {
        "logprobs": LogprobsConfidence,
        "perplexity": PerplexityConfidence,
        "consistency": SelfConsistencyConfidence,
        "self_consistency": SelfConsistencyConfidence,
        "verbalized": VerbalizedConfidence,
    }

    if method not in methods:
        available = ", ".join(methods.keys())
        raise ValueError(f"Unknown confidence method '{method}'. Available: {available}")

    return methods[method](**kwargs)


__all__ = ["ConfidenceResult", "ConfidenceEstimator", "get_confidence_estimator"]
