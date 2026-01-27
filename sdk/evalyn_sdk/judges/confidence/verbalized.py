"""Verbalized confidence estimation.

WARNING: Research shows verbalized confidence is unreliable!
LLMs tend to hallucinate confidence scores and are often overconfident.

Use logprobs or self-consistency methods instead for accurate confidence.

References:
- "Explicitly prompting the LLM to output a confidence score, while popular,
  is highly unreliable. This technique had the lowest accuracy and the
  highest standard deviation across datasets." - Refuel AI
- "Verbalized confidence produces overconfident estimates." - arXiv 2512.22245
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

from .base import ConfidenceEstimator, ConfidenceResult


class VerbalizedConfidence(ConfidenceEstimator):
    """Extract confidence from LLM's self-reported confidence score.

    WARNING: This method is unreliable! LLMs hallucinate confidence scores.
    Prefer LogprobsConfidence or SelfConsistencyConfidence instead.

    Extracts confidence from responses like:
    - "Confidence: 85%"
    - "I am 90% confident"
    - {"confidence": 0.85}

    Args:
        warn: Whether to emit a warning about unreliability (default: True)
        default_score: Score to return if no confidence found (default: 0.5)

    Usage:
        estimator = VerbalizedConfidence()
        result = estimator.estimate(response="I am 85% confident this is correct")
        # result.score = 0.85
    """

    def __init__(self, warn: bool = True, default_score: float = 0.5):
        self.warn = warn
        self.default_score = default_score
        if warn:
            warnings.warn(
                "VerbalizedConfidence is unreliable! LLMs hallucinate confidence scores. "
                "Use LogprobsConfidence or SelfConsistencyConfidence for accurate results.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def name(self) -> str:
        return "verbalized"

    def estimate(
        self,
        response: Optional[str] = None,
        parsed_response: Optional[dict] = None,
        **kwargs,
    ) -> ConfidenceResult:
        """Extract verbalized confidence from response.

        Args:
            response: Raw text response from LLM
            parsed_response: Parsed JSON response (may contain "confidence" key)

        Returns:
            ConfidenceResult with extracted confidence
        """
        confidence = None
        source = None

        # Try parsed response first
        if parsed_response and isinstance(parsed_response, dict):
            for key in ("confidence", "conf", "certainty", "probability"):
                if key in parsed_response:
                    val = parsed_response[key]
                    if isinstance(val, (int, float)):
                        # Normalize if percentage
                        confidence = val / 100 if val > 1 else val
                        source = f"parsed_{key}"
                        break

        # Try extracting from text
        if confidence is None and response:
            confidence, source = self._extract_from_text(response)

        if confidence is None:
            return ConfidenceResult(
                score=self.default_score,
                method=self.name,
                details={"reason": "not_found", "warning": "verbalized_unreliable"},
            )

        return ConfidenceResult(
            score=confidence,
            method=self.name,
            details={
                "source": source,
                "warning": "verbalized_confidence_is_unreliable",
            },
        )

    def _extract_from_text(self, text: str) -> tuple:
        """Extract confidence score from text patterns."""
        patterns = [
            # "Confidence: 85%" or "Confidence: 0.85"
            r"[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)\s*%?",
            # "85% confident" or "90% certain"
            r"(\d+(?:\.\d+)?)\s*%?\s*(?:confident|certain|sure)",
            # "I am confident (0.85)" or "confidence (85%)"
            r"(?:confident|confidence|certainty)[^\d]*(\d+(?:\.\d+)?)\s*%?",
            # "probability: 0.9"
            r"[Pp]robability[:\s]+(\d+(?:\.\d+)?)",
            # Generic percentage at end: "... 85%"
            r"(\d+(?:\.\d+)?)\s*%\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                # Normalize if percentage (> 1)
                confidence = value / 100 if value > 1 else value
                return confidence, f"pattern:{pattern[:30]}"

        return None, None


__all__ = ["VerbalizedConfidence"]
