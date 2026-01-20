"""Self-consistency based confidence estimation.

These methods run multiple samples and measure agreement as confidence.
Higher agreement = higher confidence.

References:
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- "Replacing Judges with Juries" (Verga et al., 2024)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, List, Optional

from .base import ConfidenceEstimator, ConfidenceResult


class SelfConsistencyConfidence(ConfidenceEstimator):
    """Calculate confidence from agreement across multiple samples.

    Runs the same prompt N times and measures how often responses agree.
    Higher agreement = higher confidence.

    Args:
        n_samples: Number of samples to generate (default: 5)
        temperature: Sampling temperature for diversity (default: 0.7)
        answer_extractor: Optional function to extract comparable answer from response

    Usage:
        estimator = SelfConsistencyConfidence(n_samples=5)
        result = estimator.estimate(
            generate_fn=lambda: llm.generate(prompt),
            answer_extractor=lambda r: r.get("passed")
        )
    """

    def __init__(
        self,
        n_samples: int = 5,
        temperature: float = 0.7,
        answer_extractor: Optional[Callable[[Any], Any]] = None,
    ):
        self.n_samples = n_samples
        self.temperature = temperature
        self.answer_extractor = answer_extractor or (lambda x: x)

    @property
    def name(self) -> str:
        return "self_consistency"

    def estimate(
        self,
        samples: Optional[List[Any]] = None,
        generate_fn: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> ConfidenceResult:
        """Estimate confidence from sample agreement.

        Args:
            samples: Pre-generated list of responses to compare
            generate_fn: Function to generate a single sample (called n_samples times)

        Returns:
            ConfidenceResult with agreement-based confidence
        """
        # Get samples either from provided list or by generating
        if samples is not None:
            all_samples = samples
        elif generate_fn is not None:
            all_samples = [generate_fn() for _ in range(self.n_samples)]
        else:
            return ConfidenceResult(
                score=0.5,
                method=self.name,
                details={"reason": "no_samples_or_generator"},
            )

        if not all_samples:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "empty_samples"})

        # Extract comparable answers
        try:
            answers = [self.answer_extractor(s) for s in all_samples]
        except Exception as e:
            return ConfidenceResult(
                score=0.5,
                method=self.name,
                details={"reason": f"extraction_error: {e}"},
            )

        # Convert to hashable for counting
        hashable_answers = []
        for ans in answers:
            if isinstance(ans, dict):
                hashable_answers.append(str(sorted(ans.items())))
            elif isinstance(ans, list):
                hashable_answers.append(str(ans))
            else:
                hashable_answers.append(ans)

        # Count agreement
        counter = Counter(hashable_answers)
        most_common_answer, most_common_count = counter.most_common(1)[0]

        # Confidence = fraction agreeing with majority
        confidence = most_common_count / len(answers)

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "n_samples": len(answers),
                "majority_count": most_common_count,
                "unique_answers": len(counter),
                "majority_answer": most_common_answer,
                "distribution": dict(counter),
            },
        )


class MajorityVoteConfidence(ConfidenceEstimator):
    """Calculate confidence from weighted majority voting.

    Similar to SelfConsistencyConfidence but supports weighted votes.

    Args:
        min_agreement: Minimum agreement ratio to consider confident (default: 0.6)

    Usage:
        estimator = MajorityVoteConfidence()
        result = estimator.estimate(
            votes=[True, True, False, True, True],
            weights=[1.0, 0.8, 0.9, 1.0, 0.7]
        )
    """

    def __init__(self, min_agreement: float = 0.6):
        self.min_agreement = min_agreement

    @property
    def name(self) -> str:
        return "majority_vote"

    def estimate(
        self,
        votes: List[Any],
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> ConfidenceResult:
        """Estimate confidence from weighted votes.

        Args:
            votes: List of vote values
            weights: Optional weights for each vote (default: uniform)

        Returns:
            ConfidenceResult with weighted agreement confidence
        """
        if not votes:
            return ConfidenceResult(score=0.5, method=self.name, details={"reason": "no_votes"})

        if weights is None:
            weights = [1.0] * len(votes)

        if len(weights) != len(votes):
            return ConfidenceResult(
                score=0.5,
                method=self.name,
                details={"reason": "weights_length_mismatch"},
            )

        # Calculate weighted counts
        weighted_counts = {}
        for vote, weight in zip(votes, weights):
            key = str(vote) if isinstance(vote, (dict, list)) else vote
            weighted_counts[key] = weighted_counts.get(key, 0) + weight

        # Find majority
        total_weight = sum(weights)
        majority_answer = max(weighted_counts.keys(), key=lambda k: weighted_counts[k])
        majority_weight = weighted_counts[majority_answer]

        # Confidence = weighted agreement ratio
        confidence = majority_weight / total_weight if total_weight > 0 else 0.5

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "majority_answer": majority_answer,
                "majority_weight": majority_weight,
                "total_weight": total_weight,
                "unique_answers": len(weighted_counts),
                "weighted_distribution": weighted_counts,
            },
        )


__all__ = ["SelfConsistencyConfidence", "MajorityVoteConfidence"]
