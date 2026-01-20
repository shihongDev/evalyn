"""Confidence estimation methods for LLM outputs.

This module provides multiple methods for estimating confidence:
- LogprobsConfidence: Most accurate, uses token log probabilities
- SelfConsistencyConfidence: Measures agreement across multiple samples
- MajorityVoteConfidence: Weighted voting-based confidence
- PerplexityConfidence: Perplexity-based confidence
- EntropyConfidence: Entropy-based confidence from top-k logprobs
- VerbalizedConfidence: Extract self-reported confidence (unreliable)

Usage:
    from evalyn_sdk.confidence import LogprobsConfidence, get_confidence_estimator

    # Direct instantiation
    estimator = LogprobsConfidence()
    result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
    print(f"Confidence: {result.score:.2f}")

    # Factory function
    estimator = get_confidence_estimator("logprobs", aggregation="mean")
    result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
"""

from .base import ConfidenceEstimator, ConfidenceResult, get_confidence_estimator
from .logprobs import LogprobsConfidence, PerplexityConfidence, EntropyConfidence
from .consistency import SelfConsistencyConfidence, MajorityVoteConfidence
from .verbalized import VerbalizedConfidence

__all__ = [
    # Base
    "ConfidenceEstimator",
    "ConfidenceResult",
    "get_confidence_estimator",
    # Logprobs-based (recommended)
    "LogprobsConfidence",
    "PerplexityConfidence",
    "EntropyConfidence",
    # Consistency-based
    "SelfConsistencyConfidence",
    "MajorityVoteConfidence",
    # Verbalized (unreliable)
    "VerbalizedConfidence",
]
