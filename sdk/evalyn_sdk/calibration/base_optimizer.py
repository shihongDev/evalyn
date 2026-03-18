"""Base class for new preamble optimizers.

Provides shared utilities: train/val split, candidate scoring, result building.
Existing optimizers (Basic, APE, OPRO, GEPANative, GEPA) do NOT need to
subclass this - they continue working as-is via the factory adapter.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .models import AlignmentMetrics, PromptOptimizationResult, TokenAccumulator
from .utils import build_dataset_from_annotations, build_full_prompt, parse_judge_response


class BaseOptimizer:
    """Common foundation for new preamble optimizers."""

    def __init__(self, config: Any, api_key: str | None = None):
        self.config = config
        self._api_key = api_key

    def split_train_val(
        self,
        metric_results,
        annotations,
        dataset_items,
        train_ratio: float = 0.7,
    ) -> tuple[list, list]:
        """Split annotated examples into train/val sets.

        Delegates to build_dataset_from_annotations from utils.py.
        Returns (train_examples, val_examples) where each example is a dict
        with keys: id, input, output, expected ("PASS"/"FAIL"), call_id.
        """
        return build_dataset_from_annotations(
            metric_results, annotations, dataset_items, train_split=train_ratio
        )

    def score_preamble(
        self,
        preamble: str,
        rubric: List[str],
        examples: list,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> float:
        """Score a candidate preamble on labeled examples. Returns F1.

        Uses GeminiClient.generate_with_usage (same pattern as APE._score_candidate
        and OPRO._evaluate_prompt). Builds prompt from preamble+rubric, sends
        each example through, parses pass/fail, computes alignment.

        Note: build_dataset_from_annotations returns "PASS"/"FAIL" strings
        for the 'expected' field, not booleans.
        """
        from ..utils.api_client import GeminiClient

        full_prompt = build_full_prompt(preamble, rubric)
        scorer_model = getattr(self.config, "scorer_model", None)
        client = GeminiClient(model=scorer_model, api_key=self._api_key)

        metrics = AlignmentMetrics()
        for ex in examples:
            try:
                eval_input = f"INPUT: {ex.get('input', '')}\nOUTPUT: {ex.get('output', '')}"
                result = client.generate_with_usage(full_prompt + "\n\n" + eval_input)
                predicted = parse_judge_response(result.text)
                actual = ex.get("expected") == "PASS"
                metrics.record(predicted, actual)
                if accumulator:
                    accumulator.add(result)
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
        """Construct a standard PromptOptimizationResult."""
        return PromptOptimizationResult(
            original_rubric=list(rubric),
            improved_rubric=list(rubric),  # rubric always stays fixed
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=original_preamble,
            optimized_preamble=optimized_preamble,
            full_prompt=build_full_prompt(optimized_preamble, rubric),
        )

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: List[str],
        current_preamble: str,
        metric_results: list,
        annotations: list,
        disagreements: Any = None,
        dataset_items: list | None = None,
        accumulator: TokenAccumulator | None = None,
        **kwargs,
    ) -> PromptOptimizationResult:
        raise NotImplementedError
