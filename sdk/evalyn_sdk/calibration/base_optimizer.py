"""BaseOptimizer: Abstract base class for calibration optimizers.

Provides shared functionality for all prompt optimizers:
- Train/val splitting
- Preamble scoring (F1 via LLM judge)
- Result building

Subclasses must implement the `optimize` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..defaults import DEFAULT_EVAL_MODEL
from ..utils.api_client import GeminiClient
from .models import (
    AlignmentMetrics,
    PromptOptimizationResult,
    TokenAccumulator,
)
from .utils import (
    build_dataset_from_annotations,
    build_full_prompt,
    parse_judge_response,
)


class BaseOptimizer(ABC):
    """Abstract base class for preamble optimizers.

    Provides common helpers so subclasses only need to implement `optimize`.
    """

    def __init__(self, config: Any, api_key: Optional[str] = None):
        """Initialize base optimizer.

        Args:
            config: Optimizer-specific configuration dataclass.
            api_key: Optional API key for Gemini (default: from env).
        """
        self.config = config
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def split_train_val(
        self,
        metric_results: list,
        annotations: list,
        dataset_items: Optional[list],
        train_ratio: float = 0.7,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split annotated data into train and validation sets.

        Delegates to the shared utility ``build_dataset_from_annotations``.

        Returns:
            Tuple of (trainset, valset).
        """
        return build_dataset_from_annotations(
            metric_results, annotations, dataset_items, train_ratio
        )

    def score_preamble(
        self,
        preamble: str,
        rubric: List[str],
        examples: List[Dict[str, Any]],
        accumulator: Optional[TokenAccumulator] = None,
    ) -> float:
        """Score a preamble by running it as a judge on examples.

        Uses the scorer client to evaluate each example and computes F1.

        Args:
            preamble: The preamble to evaluate.
            rubric: Fixed rubric criteria.
            examples: Examples with 'input', 'output', 'expected' keys.
            accumulator: Optional token tracker.

        Returns:
            F1 score (0.0 - 1.0).
        """
        full_prompt = build_full_prompt(preamble, rubric)
        metrics = AlignmentMetrics()

        for ex in examples:
            eval_prompt = f"""{full_prompt}

## Input to evaluate
{ex.get("input", "")[:1000]}

## Output to evaluate
{ex.get("output", "")[:1000]}

Provide your verdict:"""

            try:
                result = self.scorer_client.generate_with_usage(eval_prompt)
                if accumulator:
                    accumulator.add(result)
                judge_pass = parse_judge_response(result.text)
                human_pass = ex.get("expected") == "PASS"
                metrics.record(judge_pass, human_pass)
            except Exception:
                pass

        return metrics.f1

    def build_result(
        self,
        original_preamble: str,
        optimized_preamble: str,
        rubric: List[str],
        reasoning: str,
        estimated_improvement: str,
    ) -> PromptOptimizationResult:
        """Build a standard PromptOptimizationResult.

        Args:
            original_preamble: The preamble before optimization.
            optimized_preamble: The improved preamble.
            rubric: Fixed rubric (unchanged).
            reasoning: Explanation of what changed and why.
            estimated_improvement: One of "none", "low", "medium", "high", "unknown".

        Returns:
            PromptOptimizationResult ready for the calibration record.
        """
        full_prompt = build_full_prompt(optimized_preamble, rubric)
        return PromptOptimizationResult(
            original_rubric=rubric,
            improved_rubric=rubric,
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=original_preamble,
            optimized_preamble=optimized_preamble,
            full_prompt=full_prompt,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def scorer_client(self) -> GeminiClient:
        """Return the scorer LLM client used by ``score_preamble``."""
        ...

    @abstractmethod
    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: List[str],
        current_preamble: str,
        metric_results: list,
        annotations: list,
        disagreements: Any = None,
        dataset_items: Optional[list] = None,
        accumulator: Optional[TokenAccumulator] = None,
        **kwargs: Any,
    ) -> PromptOptimizationResult:
        """Run the optimization loop and return an optimized preamble.

        Subclasses must implement this method.
        """
        ...


__all__ = ["BaseOptimizer"]
