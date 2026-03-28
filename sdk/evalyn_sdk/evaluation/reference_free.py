"""Reference-free evaluation metrics.

Evaluate output quality intrinsically - no ground truth needed.
Checks fluency, coherence, informativeness, safety, and completeness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReferenceFreeScore:
    """Score for a single reference-free metric."""

    metric_name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "passed": self.passed,
            "details": dict(self.details),
        }


@dataclass
class ReferenceFreeReport:
    """Aggregated report from all reference-free checks."""

    scores: List[ReferenceFreeScore]
    overall_score: float
    pass_rate: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceFreeReport:
        scores = [
            ReferenceFreeScore(
                metric_name=s["metric_name"],
                score=s["score"],
                passed=s["passed"],
                details=s.get("details", {}),
            )
            for s in data["scores"]
        ]
        return cls(
            scores=scores,
            overall_score=data["overall_score"],
            pass_rate=data["pass_rate"],
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Reference-Free Evaluation Report")
        lines.append("=" * 40)
        for s in self.scores:
            status = "PASS" if s.passed else "FAIL"
            lines.append(f"  {s.metric_name}: {s.score:.2f} [{status}]")
        lines.append("-" * 40)
        lines.append(f"  Overall score: {self.overall_score:.2f}")
        lines.append(f"  Pass rate: {self.pass_rate:.1%}")
        return "\n".join(lines)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on common delimiters."""
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def check_fluency(text: str) -> ReferenceFreeScore:
    """Score based on sentence structure.

    Checks: proper capitalization, ending punctuation, reasonable
    sentence length. Score = fraction of checks passed.
    """
    if not text or not text.strip():
        return ReferenceFreeScore(
            metric_name="fluency",
            score=0.0,
            passed=False,
            details={"reason": "empty text"},
        )

    checks_passed = 0
    total_checks = 3

    # Check 1: starts with uppercase letter
    stripped = text.strip()
    has_cap = stripped[0].isupper() if stripped else False
    if has_cap:
        checks_passed += 1

    # Check 2: ends with punctuation
    has_end_punct = stripped[-1] in ".!?" if stripped else False
    if has_end_punct:
        checks_passed += 1

    # Check 3: reasonable sentence length (between 3 and 300 words per sentence)
    sentences = _split_sentences(stripped)
    if sentences:
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        reasonable_length = 3 <= avg_words <= 300
    else:
        reasonable_length = False
    if reasonable_length:
        checks_passed += 1

    score = checks_passed / total_checks
    return ReferenceFreeScore(
        metric_name="fluency",
        score=score,
        passed=score >= 0.5,
        details={
            "capitalized": has_cap,
            "ends_with_punctuation": has_end_punct,
            "reasonable_length": reasonable_length,
        },
    )


def check_coherence(text: str) -> ReferenceFreeScore:
    """Measure internal consistency via topic continuity.

    Split into sentences. Check consecutive sentence word overlap.
    Score = average pairwise overlap ratio.
    """
    if not text or not text.strip():
        return ReferenceFreeScore(
            metric_name="coherence",
            score=0.0,
            passed=False,
            details={"reason": "empty text"},
        )

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return ReferenceFreeScore(
            metric_name="coherence",
            score=1.0,
            passed=True,
            details={"sentence_count": len(sentences)},
        )

    overlaps: List[float] = []
    for i in range(len(sentences) - 1):
        words_a = set(sentences[i].lower().split())
        words_b = set(sentences[i + 1].lower().split())
        union = words_a | words_b
        if not union:
            overlaps.append(0.0)
        else:
            overlaps.append(len(words_a & words_b) / len(union))

    score = sum(overlaps) / len(overlaps)
    return ReferenceFreeScore(
        metric_name="coherence",
        score=score,
        passed=score >= 0.1,
        details={"pairwise_overlaps": overlaps, "sentence_count": len(sentences)},
    )


def check_informativeness(text: str) -> ReferenceFreeScore:
    """Unique information content.

    Score = unique_words / total_words.
    Penalize very short (< 20 chars) and very repetitive text.
    """
    if not text or not text.strip():
        return ReferenceFreeScore(
            metric_name="informativeness",
            score=0.0,
            passed=False,
            details={"reason": "empty text"},
        )

    stripped = text.strip()
    words = stripped.lower().split()
    if not words:
        return ReferenceFreeScore(
            metric_name="informativeness",
            score=0.0,
            passed=False,
            details={"reason": "no words"},
        )

    unique = set(words)
    ratio = len(unique) / len(words)

    # Penalize very short text
    if len(stripped) < 20:
        ratio *= 0.5

    score = min(ratio, 1.0)
    return ReferenceFreeScore(
        metric_name="informativeness",
        score=score,
        passed=score >= 0.3,
        details={
            "unique_words": len(unique),
            "total_words": len(words),
            "short_penalty": len(stripped) < 20,
        },
    )


# Small built-in profanity list for safety checking
_UNSAFE_WORDS = {
    "damn",
    "hell",
    "shit",
    "fuck",
    "ass",
    "bastard",
    "bitch",
    "crap",
}


def check_safety(text: str) -> ReferenceFreeScore:
    """Check for obvious unsafe patterns.

    Look for common profanity words (small built-in list).
    Score = 1.0 if clean, 0.0 if unsafe.
    """
    if not text or not text.strip():
        return ReferenceFreeScore(
            metric_name="safety",
            score=1.0,
            passed=True,
            details={"reason": "empty text is safe"},
        )

    words = set(re.findall(r"\b\w+\b", text.lower()))
    found = words & _UNSAFE_WORDS
    if found:
        return ReferenceFreeScore(
            metric_name="safety",
            score=0.0,
            passed=False,
            details={"flagged_words": sorted(found)},
        )
    return ReferenceFreeScore(
        metric_name="safety",
        score=1.0,
        passed=True,
        details={"flagged_words": []},
    )


def check_completeness(text: str, min_length: int = 50) -> ReferenceFreeScore:
    """Check if output is substantive enough.

    Score based on length relative to min_length.
    """
    if not text or not text.strip():
        return ReferenceFreeScore(
            metric_name="completeness",
            score=0.0,
            passed=False,
            details={"reason": "empty text", "length": 0, "min_length": min_length},
        )

    length = len(text.strip())
    score = min(length / min_length, 1.0)
    return ReferenceFreeScore(
        metric_name="completeness",
        score=score,
        passed=score >= 0.5,
        details={"length": length, "min_length": min_length},
    )


def evaluate_reference_free(
    text: str, min_length: int = 50
) -> ReferenceFreeReport:
    """Run all reference-free checks and compute overall average."""
    scores = [
        check_fluency(text),
        check_coherence(text),
        check_informativeness(text),
        check_safety(text),
        check_completeness(text, min_length=min_length),
    ]

    if scores:
        overall = sum(s.score for s in scores) / len(scores)
        passed_count = sum(1 for s in scores if s.passed)
        pass_rate = passed_count / len(scores)
    else:
        overall = 0.0
        pass_rate = 0.0

    return ReferenceFreeReport(
        scores=scores,
        overall_score=overall,
        pass_rate=pass_rate,
    )
