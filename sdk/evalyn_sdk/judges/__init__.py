"""Judges package for output quality assessment.

This package contains:
- LLMJudge: LLM-based judge for subjective evaluation
- EchoJudge: Debug judge for testing
- confidence/: Confidence estimation methods (logprobs, deepconf, consistency)
"""

from .llm_judge import LLMJudge, EchoJudge
from .confidence import (
    ConfidenceEstimator,
    ConfidenceResult,
    get_confidence_estimator,
    LogprobsConfidence,
    DeepConfConfidence,
    PerplexityConfidence,
    EntropyConfidence,
    SelfConsistencyConfidence,
    MajorityVoteConfidence,
    VerbalizedConfidence,
)

__all__ = [
    # LLM Judges
    "LLMJudge",
    "EchoJudge",
    # Confidence base
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
