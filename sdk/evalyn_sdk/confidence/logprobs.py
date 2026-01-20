"""Logprobs-based confidence estimation methods.

These methods calculate confidence from token-level log probabilities,
which research shows is the most accurate approach for LLM confidence.

References:
- "Token-level generation probabilities (logprobs) are by far the most
  accurate technique for estimating LLM confidence." - Refuel AI
- VATBox/llm-confidence: exp(sum(logprobs)) for confidence
"""

from __future__ import annotations

import math
from typing import List, Optional

from .base import ConfidenceEstimator, ConfidenceResult


class LogprobsConfidence(ConfidenceEstimator):
    """Calculate confidence from token log probabilities.

    Uses the formula: confidence = exp(mean(logprobs))

    This converts average log probability to a probability in [0, 1].
    Higher logprobs (closer to 0) = higher confidence.

    Args:
        aggregation: How to aggregate token logprobs - "mean", "min", "sum"
        normalize: Whether to apply sigmoid normalization for better calibration

    Usage:
        estimator = LogprobsConfidence()
        result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
        # result.score ~ 0.82
    """

    def __init__(self, aggregation: str = "mean", normalize: bool = False):
        self.aggregation = aggregation
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "logprobs"

    def estimate(
        self,
        logprobs: List[float],
        top_logprobs: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> ConfidenceResult:
        """Estimate confidence from logprobs.

        Args:
            logprobs: List of log probabilities for each token
            top_logprobs: Optional list of top-k logprobs per token (for entropy)

        Returns:
            ConfidenceResult with confidence score
        """
        if not logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_logprobs"})

        # Filter out None values
        valid_logprobs = [lp for lp in logprobs if lp is not None]
        if not valid_logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "all_none"})

        # Aggregate logprobs
        if self.aggregation == "min":
            agg_logprob = min(valid_logprobs)
        elif self.aggregation == "sum":
            agg_logprob = sum(valid_logprobs)
        else:  # mean
            agg_logprob = sum(valid_logprobs) / len(valid_logprobs)

        # Convert to probability
        if self.normalize:
            # Sigmoid normalization for better calibration
            # Maps logprob -1 -> ~0.5, -0.1 -> ~0.7, -2 -> ~0.27
            confidence = 1 / (1 + math.exp(-agg_logprob - 0.5))
        else:
            # Direct probability: exp(logprob)
            # For mean aggregation, this is geometric mean of probabilities
            confidence = math.exp(agg_logprob)

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "aggregation": self.aggregation,
                "agg_logprob": agg_logprob,
                "num_tokens": len(valid_logprobs),
                "min_logprob": min(valid_logprobs),
                "max_logprob": max(valid_logprobs),
            },
        )


class PerplexityConfidence(ConfidenceEstimator):
    """Calculate confidence from perplexity (inverse of logprobs).

    Perplexity = exp(-mean(logprobs))
    Lower perplexity = higher confidence

    This method normalizes perplexity to a 0-1 confidence score.

    Args:
        max_perplexity: Perplexity value that maps to confidence 0
                       (default: 10, meaning perplexity >= 10 -> confidence ~ 0)

    Usage:
        estimator = PerplexityConfidence()
        result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2])
    """

    def __init__(self, max_perplexity: float = 10.0):
        self.max_perplexity = max_perplexity

    @property
    def name(self) -> str:
        return "perplexity"

    def estimate(self, logprobs: List[float], **kwargs) -> ConfidenceResult:
        """Estimate confidence from perplexity.

        Args:
            logprobs: List of log probabilities for each token

        Returns:
            ConfidenceResult with confidence score
        """
        if not logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_logprobs"})

        valid_logprobs = [lp for lp in logprobs if lp is not None]
        if not valid_logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "all_none"})

        # Calculate perplexity
        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        perplexity = math.exp(-avg_logprob)

        # Normalize to confidence: low perplexity = high confidence
        # perplexity 1 -> confidence 1, perplexity max_perplexity -> confidence ~0
        confidence = max(0.0, 1.0 - (perplexity - 1) / (self.max_perplexity - 1))

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "perplexity": perplexity,
                "avg_logprob": avg_logprob,
                "num_tokens": len(valid_logprobs),
            },
        )


class EntropyConfidence(ConfidenceEstimator):
    """Calculate confidence from token-level entropy.

    Uses Shannon entropy of top-k token probabilities.
    Lower entropy = more concentrated probability = higher confidence.

    Args:
        normalize: Whether to normalize entropy to [0, 1] based on max entropy

    Usage:
        estimator = EntropyConfidence()
        result = estimator.estimate(top_logprobs=[[-0.1, -2.3, -3.0], ...])
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "entropy"

    def estimate(
        self,
        top_logprobs: List[List[float]],
        logprobs: Optional[List[float]] = None,
        **kwargs,
    ) -> ConfidenceResult:
        """Estimate confidence from entropy of top-k logprobs.

        Args:
            top_logprobs: List of top-k logprobs for each token position
            logprobs: Optional single logprobs (falls back to LogprobsConfidence)

        Returns:
            ConfidenceResult with confidence score
        """
        if not top_logprobs:
            # Fall back to simple logprobs if top_logprobs not available
            if logprobs:
                return LogprobsConfidence().estimate(logprobs=logprobs)
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_data"})

        entropies = []
        for token_logprobs in top_logprobs:
            if not token_logprobs:
                continue
            # Convert logprobs to probabilities
            probs = [math.exp(lp) for lp in token_logprobs if lp is not None]
            if not probs:
                continue
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            # Calculate entropy: -sum(p * log(p))
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            entropies.append(entropy)

        if not entropies:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_valid_tokens"})

        avg_entropy = sum(entropies) / len(entropies)

        if self.normalize:
            # Max entropy for k choices is log(k)
            # Assuming top-5, max entropy ~ 1.6
            k = len(top_logprobs[0]) if top_logprobs else 5
            max_entropy = math.log(k)
            confidence = 1.0 - min(avg_entropy / max_entropy, 1.0)
        else:
            # Lower entropy = higher confidence (inverse relationship)
            confidence = math.exp(-avg_entropy)

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "avg_entropy": avg_entropy,
                "num_tokens": len(entropies),
                "min_entropy": min(entropies),
                "max_entropy": max(entropies),
            },
        )


class DeepConfConfidence(ConfidenceEstimator):
    """Meta AI's DeepConf confidence estimation.

    DeepConf uses logprobs with specialized aggregation strategies that
    better distinguish correct from incorrect reasoning traces.

    Key insight: Bottom-10% and tail confidence better detect errors than
    simple average confidence.

    Args:
        strategy: Aggregation strategy:
            - "average": Mean confidence over all tokens
            - "bottom10": Mean of lowest 10% confidences (default, best for error detection)
            - "tail": Confidence over final N tokens
        tail_tokens: Number of final tokens for tail strategy (default: 256)

    References:
        - "Deep Think with Confidence" (arXiv:2508.15260)
        - https://jiaweizzhao.github.io/deepconf/

    Usage:
        estimator = DeepConfConfidence(strategy="bottom10")
        result = estimator.estimate(logprobs=[-0.1, -0.3, -0.2, -1.5, -0.4])
    """

    def __init__(self, strategy: str = "bottom10", tail_tokens: int = 256):
        if strategy not in ("average", "bottom10", "tail"):
            raise ValueError(f"Invalid strategy: {strategy}. Use 'average', 'bottom10', or 'tail'")
        self.strategy = strategy
        self.tail_tokens = tail_tokens

    @property
    def name(self) -> str:
        return f"deepconf_{self.strategy}"

    def estimate(self, logprobs: List[float], **kwargs) -> ConfidenceResult:
        """Estimate confidence using DeepConf strategy.

        Args:
            logprobs: List of log probabilities for each token

        Returns:
            ConfidenceResult with confidence score
        """
        if not logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_logprobs"})

        valid_logprobs = [lp for lp in logprobs if lp is not None]
        if not valid_logprobs:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "all_none"})

        # Convert logprobs to token-level confidence scores
        token_confidences = [math.exp(lp) for lp in valid_logprobs]

        # Apply strategy
        if self.strategy == "bottom10":
            # Bottom-10% Group Confidence: mean of lowest 10% confidences
            # Better at detecting incorrect reasoning traces
            sorted_conf = sorted(token_confidences)
            bottom_count = max(1, len(sorted_conf) // 10)
            selected = sorted_conf[:bottom_count]
            confidence = sum(selected) / len(selected)
            details = {
                "bottom_count": bottom_count,
                "bottom_confidences": selected[:5],  # First 5 for debugging
            }

        elif self.strategy == "tail":
            # Tail Confidence: confidence over final N tokens
            # Useful for reasoning tasks where conclusion quality matters
            tail = token_confidences[-self.tail_tokens:]
            confidence = sum(tail) / len(tail)
            details = {
                "tail_tokens_used": len(tail),
                "tail_tokens_requested": self.tail_tokens,
            }

        else:  # average
            # Average Trace Confidence: mean confidence over all tokens
            confidence = sum(token_confidences) / len(token_confidences)
            details = {}

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "strategy": self.strategy,
                "num_tokens": len(valid_logprobs),
                "avg_confidence": sum(token_confidences) / len(token_confidences),
                "min_confidence": min(token_confidences),
                "max_confidence": max(token_confidences),
                **details,
            },
        )


__all__ = ["LogprobsConfidence", "PerplexityConfidence", "EntropyConfidence", "DeepConfConfidence"]
