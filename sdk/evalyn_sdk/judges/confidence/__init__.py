"""Confidence estimation methods for LLM outputs.

This module provides multiple methods for estimating confidence:
- LogprobsConfidence: Most accurate, uses token log probabilities
- DeepConfConfidence: Meta AI's DeepConf with bottom-10%/tail strategies
- SelfConsistencyConfidence: Measures agreement across multiple samples
- MajorityVoteConfidence: Weighted voting-based confidence
- PerplexityConfidence: Perplexity-based confidence
- EntropyConfidence: Entropy-based confidence from top-k logprobs
- VerbalizedConfidence: Extract self-reported confidence (unreliable)

Usage:
    from evalyn_sdk.confidence import LogprobsConfidence, DeepConfConfidence

    # Standard logprobs
    estimator = LogprobsConfidence()
    result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
    print(f"Confidence: {result.score:.2f}")

    # DeepConf (better error detection)
    estimator = DeepConfConfidence(strategy="bottom10")
    result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2, -1.5, -0.4])
"""

from .base import ConfidenceEstimator, ConfidenceResult, get_confidence_estimator
from .logprobs import (
    LogprobsConfidence,
    PerplexityConfidence,
    EntropyConfidence,
    DeepConfConfidence,
)
from .consistency import SelfConsistencyConfidence, MajorityVoteConfidence
from .verbalized import VerbalizedConfidence

__all__ = [
    # Base
    "ConfidenceEstimator",
    "ConfidenceResult",
    "get_confidence_estimator",
    # Logprobs-based (recommended)
    "LogprobsConfidence",
    "DeepConfConfidence",
    "PerplexityConfidence",
    "EntropyConfidence",
    # Consistency-based
    "SelfConsistencyConfidence",
    "MajorityVoteConfidence",
    # Verbalized (unreliable)
    "VerbalizedConfidence",
]
