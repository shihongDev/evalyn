"""Judges package for output quality assessment.

This package contains:
- LLMJudge: LLM-based judge for subjective evaluation
- EchoJudge: Debug judge for testing
- confidence/: Confidence estimation methods (logprobs, deepconf, consistency)
"""

from .confidence import (
    ConfidenceEstimator,
    ConfidenceResult,
    DeepConfConfidence,
    EntropyConfidence,
    LogprobsConfidence,
    MajorityVoteConfidence,
    PerplexityConfidence,
    SelfConsistencyConfidence,
    VerbalizedConfidence,
    get_confidence_estimator,
)
from .llm_judge import EchoJudge, LLMJudge

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
