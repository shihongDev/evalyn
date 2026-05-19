"""Reasoning evaluation.

Evaluate chain-of-thought quality: faithfulness, consistency,
evidence usage, and conclusion validity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens from text."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Compute word overlap ratio between two strings.

    Returns |intersection| / |union|, or 0.0 if both empty.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    text: str
    step_type: str = "reasoning"
    referenced_evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "step_type": self.step_type,
            "referenced_evidence": list(self.referenced_evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningStep:
        return cls(
            text=data["text"],
            step_type=data.get("step_type", "reasoning"),
            referenced_evidence=list(data.get("referenced_evidence", [])),
        )


@dataclass
class ReasoningEvalResult:
    """Result of evaluating a reasoning chain."""

    faithfulness: float
    logical_consistency: float
    evidence_usage: float
    conclusion_validity: float
    overall: float
    step_count: int
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "logical_consistency": self.logical_consistency,
            "evidence_usage": self.evidence_usage,
            "conclusion_validity": self.conclusion_validity,
            "overall": self.overall,
            "step_count": self.step_count,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningEvalResult:
        return cls(
            faithfulness=data["faithfulness"],
            logical_consistency=data["logical_consistency"],
            evidence_usage=data["evidence_usage"],
            conclusion_validity=data["conclusion_validity"],
            overall=data["overall"],
            step_count=data["step_count"],
            details=dict(data.get("details", {})),
        )

    def format_text(self) -> str:
        lines = [
            "Reasoning Evaluation Result",
            f"  faithfulness:         {self.faithfulness:.3f}",
            f"  logical_consistency:  {self.logical_consistency:.3f}",
            f"  evidence_usage:       {self.evidence_usage:.3f}",
            f"  conclusion_validity:  {self.conclusion_validity:.3f}",
            f"  overall:              {self.overall:.3f}",
            f"  step_count:           {self.step_count}",
        ]
        return "\n".join(lines)


def evaluate_faithfulness(steps: list[ReasoningStep], source_text: str) -> float:
    """Fraction of reasoning steps that reference content from source_text.

    A step is faithful if word overlap with source > 0.1.
    """
    reasoning_steps = [s for s in steps if s.step_type == "reasoning"]
    if not reasoning_steps:
        return 1.0
    faithful = 0
    for step in reasoning_steps:
        if _word_overlap(step.text, source_text) > 0.1:
            faithful += 1
    return faithful / len(reasoning_steps)


def evaluate_logical_consistency(steps: list[ReasoningStep]) -> float:
    """Check that later reasoning steps build on earlier ones.

    Uses word overlap between consecutive reasoning steps.
    Returns average overlap ratio.
    """
    reasoning_steps = [s for s in steps if s.step_type == "reasoning"]
    if len(reasoning_steps) < 2:
        return 1.0
    overlaps: list[float] = []
    for i in range(1, len(reasoning_steps)):
        overlap = _word_overlap(reasoning_steps[i - 1].text, reasoning_steps[i].text)
        overlaps.append(overlap)
    return sum(overlaps) / len(overlaps)


def evaluate_evidence_usage(steps: list[ReasoningStep]) -> float:
    """Fraction of steps that have non-empty referenced_evidence."""
    if not steps:
        return 1.0
    with_evidence = sum(1 for s in steps if s.referenced_evidence)
    return with_evidence / len(steps)


def evaluate_conclusion_validity(steps: list[ReasoningStep]) -> float:
    """Check that conclusion steps reference earlier reasoning.

    Score = word overlap between conclusion text and all prior reasoning text.
    """
    conclusions = [s for s in steps if s.step_type == "conclusion"]
    reasoning_steps = [s for s in steps if s.step_type == "reasoning"]
    if not conclusions:
        return 1.0
    if not reasoning_steps:
        return 0.0
    all_reasoning_text = " ".join(s.text for s in reasoning_steps)
    scores: list[float] = []
    for c in conclusions:
        scores.append(_word_overlap(c.text, all_reasoning_text))
    return sum(scores) / len(scores)


def evaluate_reasoning(
    steps: list[ReasoningStep], source_text: str = ""
) -> ReasoningEvalResult:
    """Full reasoning evaluation.

    Overall = average of all four metrics.
    """
    faithfulness = evaluate_faithfulness(steps, source_text)
    consistency = evaluate_logical_consistency(steps)
    evidence = evaluate_evidence_usage(steps)
    conclusion = evaluate_conclusion_validity(steps)
    overall = (faithfulness + consistency + evidence + conclusion) / 4.0
    return ReasoningEvalResult(
        faithfulness=faithfulness,
        logical_consistency=consistency,
        evidence_usage=evidence,
        conclusion_validity=conclusion,
        overall=overall,
        step_count=len(steps),
        details={
            "source_text_length": len(source_text),
            "reasoning_step_count": sum(
                1 for s in steps if s.step_type == "reasoning"
            ),
            "conclusion_step_count": sum(
                1 for s in steps if s.step_type == "conclusion"
            ),
        },
    )


def extract_reasoning_steps(text: str) -> list[ReasoningStep]:
    """Split text into reasoning steps.

    Splits on numbered patterns (1., 2., etc.) or keywords:
    therefore, because, however, in conclusion.
    """
    if not text.strip():
        return []

    # Try numbered pattern first
    numbered = re.split(r"\d+\.\s+", text.strip())
    # First element is usually empty or preamble
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 2:
        steps: list[ReasoningStep] = []
        for chunk in numbered:
            step_type = "reasoning"
            lower = chunk.lower()
            if lower.startswith("in conclusion") or lower.startswith("therefore"):
                step_type = "conclusion"
            elif lower.startswith("because") or lower.startswith("given"):
                step_type = "premise"
            steps.append(ReasoningStep(text=chunk, step_type=step_type))
        return steps

    # Fall back to keyword splitting
    pattern = r"(?i)\b(therefore|because|however|in conclusion)\b"
    parts = re.split(pattern, text.strip())
    if len(parts) <= 1:
        return [ReasoningStep(text=text.strip())]

    steps = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if not chunk:
            i += 1
            continue
        # Check if this part is a keyword
        if re.match(pattern, chunk, re.IGNORECASE):
            # Merge keyword with the following text
            if i + 1 < len(parts):
                merged = (chunk + " " + parts[i + 1]).strip()
                i += 2
            else:
                merged = chunk
                i += 1
            lower = merged.lower()
            if lower.startswith("in conclusion") or lower.startswith("therefore"):
                step_type = "conclusion"
            elif lower.startswith("because"):
                step_type = "premise"
            else:
                step_type = "reasoning"
            if merged:
                steps.append(ReasoningStep(text=merged, step_type=step_type))
        else:
            steps.append(ReasoningStep(text=chunk, step_type="reasoning"))
            i += 1

    return [s for s in steps if s.text.strip()]
