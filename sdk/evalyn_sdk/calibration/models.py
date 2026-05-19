"""Data classes for calibration module.

Contains shared data structures used across calibration components:
- AlignmentMetrics: Confusion matrix and derived metrics
- DisagreementCase/Analysis: Tracking judge-human disagreements
- PromptOptimizationResult: Results from prompt optimization
- ValidationResult: Results from prompt validation
- TokenAccumulator: Token usage tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.api_client import GenerateResult


@dataclass
class TokenAccumulator:
    """Accumulates token usage across multiple LLM calls for cost tracking."""

    _per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def _ensure_model(self, model_key: str) -> None:
        """Ensure a model entry exists in the per-model dict."""
        if model_key not in self._per_model:
            self._per_model[model_key] = {"input_tokens": 0, "output_tokens": 0}

    def add(self, result: GenerateResult) -> None:
        """Add tokens from a GenerateResult."""
        model_key = result.model if result.model else "_unknown"
        self._ensure_model(model_key)
        self._per_model[model_key]["input_tokens"] += result.input_tokens
        self._per_model[model_key]["output_tokens"] += result.output_tokens

    def add_usage(
        self, input_tokens: int, output_tokens: int, model: str | None = None
    ) -> None:
        """Add tokens directly from counts (e.g., from LLMJudge.score() result)."""
        model_key = model if model else "_unknown"
        self._ensure_model(model_key)
        self._per_model[model_key]["input_tokens"] += input_tokens
        self._per_model[model_key]["output_tokens"] += output_tokens

    @property
    def input_tokens(self) -> int:
        """Total input tokens across all models (backwards compatible)."""
        return sum(v["input_tokens"] for v in self._per_model.values())

    @property
    def output_tokens(self) -> int:
        """Total output tokens across all models (backwards compatible)."""
        return sum(v["output_tokens"] for v in self._per_model.values())

    @property
    def models(self) -> set:
        """Set of model names used (backwards compatible)."""
        return {k for k in self._per_model if k != "_unknown"}

    def as_usage_summary(self) -> dict:
        """Convert to usage_summary dict compatible with print_token_usage_summary."""
        from ..trace.instrumentation.providers._shared import (
            calculate_cost,
            is_model_pricing_known,
        )

        total_tokens = self.input_tokens + self.output_tokens
        has_unknown_pricing = any(not is_model_pricing_known(m) for m in self.models)

        # Calculate total cost using actual per-model token counts
        total_cost = 0.0
        has_real_models = False
        for model_key, counts in self._per_model.items():
            if model_key == "_unknown":
                continue
            has_real_models = True
            total_cost += calculate_cost(
                model_key,
                counts["input_tokens"],
                counts["output_tokens"],
            )

        if not has_real_models and total_tokens > 0:
            # Fallback: use default rate when no model info available
            total_cost = total_tokens / 1_000_000 * 1.0

        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": total_tokens,
            "models_used": sorted(self.models),
            "total_cost_usd": total_cost,
            "has_unknown_pricing": has_unknown_pricing,
        }


@dataclass
class AlignmentMetrics:
    """
    Alignment metrics between LLM judge and human annotations.
    Uses classification terminology: judge prediction vs human ground truth.
    """

    # Confusion matrix counts
    true_positive: int = 0  # Both judge and human say PASS
    true_negative: int = 0  # Both judge and human say FAIL
    false_positive: int = 0  # Judge says PASS, human says FAIL
    false_negative: int = 0  # Judge says FAIL, human says PASS

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        """Overall agreement rate."""
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        """When judge says PASS, how often is human also PASS."""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of human PASS cases, how many did judge catch."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Of human FAIL cases, how many did judge correctly identify."""
        denom = self.true_negative + self.false_positive
        return self.true_negative / denom if denom > 0 else 0.0

    @property
    def cohens_kappa(self) -> float:
        """
        Cohen's kappa: agreement beyond chance.
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        """
        if self.total == 0:
            return 0.0

        # Observed agreement
        p_o = self.accuracy

        # Expected agreement (by chance)
        judge_pass_rate = (self.true_positive + self.false_positive) / self.total
        human_pass_rate = (self.true_positive + self.false_negative) / self.total
        p_e = (judge_pass_rate * human_pass_rate) + (
            (1 - judge_pass_rate) * (1 - human_pass_rate)
        )

        if p_e >= 1.0:
            return 1.0 if p_o >= 1.0 else 0.0

        return (p_o - p_e) / (1 - p_e)

    def record(self, predicted: bool, actual: bool) -> None:
        """Record a single prediction vs actual comparison."""
        if predicted and actual:
            self.true_positive += 1
        elif not predicted and not actual:
            self.true_negative += 1
        elif predicted and not actual:
            self.false_positive += 1
        else:
            self.false_negative += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "confusion_matrix": {
                "true_positive": self.true_positive,
                "true_negative": self.true_negative,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            },
            "total_samples": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
        }


@dataclass
class DisagreementCase:
    """A single case where judge and human disagree."""

    call_id: str
    input_text: str
    output_text: str
    judge_passed: bool
    judge_reason: str
    human_passed: bool
    human_notes: str
    disagreement_type: str  # "false_positive" or "false_negative"

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "input": self.input_text[:500],  # Truncate for readability
            "output": self.output_text[:500],
            "judge_passed": self.judge_passed,
            "judge_reason": self.judge_reason,
            "human_passed": self.human_passed,
            "human_notes": self.human_notes,
            "type": self.disagreement_type,
        }


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement patterns."""

    false_positives: list[DisagreementCase] = field(default_factory=list)
    false_negatives: list[DisagreementCase] = field(default_factory=list)

    @property
    def total_disagreements(self) -> int:
        return len(self.false_positives) + len(self.false_negatives)

    def get_pattern_summary(self) -> dict[str, Any]:
        """Summarize common patterns in disagreements."""
        return {
            "false_positive_count": len(self.false_positives),
            "false_negative_count": len(self.false_negatives),
            "false_positive_examples": [d.as_dict() for d in self.false_positives[:5]],
            "false_negative_examples": [d.as_dict() for d in self.false_negatives[:5]],
        }


@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization (LLM or GEPA)."""

    original_rubric: list[str]
    improved_rubric: list[str]
    improvement_reasoning: str
    suggested_additions: list[str]
    suggested_removals: list[str]
    estimated_improvement: str  # "low", "medium", "high"
    # New fields for storing the full prompt
    original_preamble: str = ""
    optimized_preamble: str = ""
    full_prompt: str = ""  # Complete prompt ready to use (preamble + rubric + format)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "original_rubric": self.original_rubric,
            "improved_rubric": self.improved_rubric,
            "improvement_reasoning": self.improvement_reasoning,
            "suggested_additions": self.suggested_additions,
            "suggested_removals": self.suggested_removals,
            "estimated_improvement": self.estimated_improvement,
        }
        # Include preamble fields if they have content
        if self.original_preamble:
            result["original_preamble"] = self.original_preamble
        if self.optimized_preamble:
            result["optimized_preamble"] = self.optimized_preamble
        if self.full_prompt:
            result["full_prompt"] = self.full_prompt
        return result


@dataclass
class ValidationResult:
    """Result of validating optimized prompt against validation set."""

    original_f1: float
    optimized_f1: float
    original_accuracy: float
    optimized_accuracy: float
    improvement_delta: float
    is_better: bool
    confidence: str  # "high", "medium", "low"
    recommendation: str  # "use_optimized", "keep_original", "uncertain"
    validation_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_f1": self.original_f1,
            "optimized_f1": self.optimized_f1,
            "original_accuracy": self.original_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "improvement_delta": self.improvement_delta,
            "is_better": self.is_better,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "validation_samples": self.validation_samples,
        }


__all__ = [
    "TokenAccumulator",
    "AlignmentMetrics",
    "DisagreementCase",
    "DisagreementAnalysis",
    "PromptOptimizationResult",
    "ValidationResult",
]
