"""Base class for new preamble optimizers.

Provides shared utilities: train/val split, candidate scoring, result building.
Existing optimizers (Basic, APE, OPRO, GEPANative, GEPA) do NOT need to
subclass this - they continue working as-is via the factory adapter.
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .models import AlignmentMetrics, PromptOptimizationResult, TokenAccumulator
from .utils import build_dataset_from_annotations, build_full_prompt, parse_judge_response


class BaseOptimizer(ABC):
    """Common foundation for new preamble optimizers."""

    def __init__(self, config: Any, api_key: str | None = None):
        self.config = config
        self._api_key = api_key
        self._base_scorer_client = None

    def _get_scorer_client(self):
        """Lazily create the scorer client for preamble evaluation."""
        if self._base_scorer_client is None:
            from ..utils.api_client import GeminiClient

            scorer_model = getattr(self.config, "scorer_model", None)
            timeout = getattr(self.config, "timeout", 120)
            self._base_scorer_client = GeminiClient(
                model=scorer_model, temperature=0.0, api_key=self._api_key, timeout=timeout,
            )
        return self._base_scorer_client

    def _generate_with_retry(self, client, prompt: str, max_retries: int = 2,
                              accumulator=None) -> Optional["GenerateResult"]:
        """Call client.generate_with_usage with retry and exponential backoff.

        Args:
            client: An LLM client with a generate_with_usage method.
            prompt: The prompt string to send.
            max_retries: Number of retries after the initial attempt (default 2).
            accumulator: Optional TokenAccumulator to track token usage on success.

        Returns:
            GenerateResult on success, or None if all attempts fail.
        """
        for attempt in range(1 + max_retries):
            try:
                result = client.generate_with_usage(prompt)
                if accumulator:
                    accumulator.add(result)
                return result
            except Exception as exc:
                if attempt < max_retries:
                    backoff = 4 ** attempt  # 1s after first failure, 4s after second
                    logger.warning(
                        "LLM API call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, 1 + max_retries, exc, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "LLM API call failed after %d attempts: %s",
                        1 + max_retries, exc,
                    )
                    return None

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
        max_samples: Optional[int] = None,
    ) -> float:
        """Score a candidate preamble on labeled examples. Returns F1.

        Uses GeminiClient.generate_with_usage (same pattern as APE._score_candidate
        and OPRO._evaluate_prompt). Builds prompt from preamble+rubric, sends
        each example through, parses pass/fail, computes alignment.

        Args:
            preamble: The preamble to evaluate.
            rubric: Fixed rubric criteria.
            examples: Labeled examples to score against.
            accumulator: Optional TokenAccumulator to track token usage.
            max_samples: When set, randomly sample that many examples before
                scoring. Useful for large datasets.

        Note: build_dataset_from_annotations returns "PASS"/"FAIL" strings
        for the 'expected' field, not booleans.
        """
        full_prompt = build_full_prompt(preamble, rubric)
        client = self._get_scorer_client()

        sample = examples
        if max_samples is not None and len(examples) > max_samples:
            sample = random.sample(examples, max_samples)

        metrics = AlignmentMetrics()
        for ex in sample:
            eval_input = f"INPUT: {ex.get('input', '')}\nOUTPUT: {ex.get('output', '')}"
            result = self._generate_with_retry(
                client, full_prompt + "\n\n" + eval_input, accumulator=accumulator,
            )
            if result is None:
                continue
            predicted = parse_judge_response(result.text)
            actual = ex.get("expected") == "PASS"
            metrics.record(predicted, actual)
        return metrics.f1

    def _collect_failures(
        self,
        preamble: str,
        rubric: List[str],
        examples: List[Dict[str, Any]],
        max_failures: int = 5,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> List[Dict[str, Any]]:
        """Score examples and return ones where prediction != human label.

        Args:
            preamble: The preamble to evaluate.
            rubric: Fixed rubric criteria.
            examples: Labeled examples to check.
            max_failures: Maximum number of failure dicts to return.
            accumulator: Optional TokenAccumulator to track token usage.

        Returns:
            List of failure dicts with keys: input, output, judge_said, human_said.
        """
        full_prompt = build_full_prompt(preamble, rubric)
        client = self._get_scorer_client()

        failures: List[Dict[str, Any]] = []
        for ex in examples:
            eval_input = f"INPUT: {str(ex.get('input', ''))[:500]}\nOUTPUT: {str(ex.get('output', ''))[:500]}"
            result = self._generate_with_retry(
                client, full_prompt + "\n\n" + eval_input, accumulator=accumulator,
            )
            if result is None:
                continue
            predicted = parse_judge_response(result.text)
            actual = ex.get("expected") == "PASS"
            if predicted != actual:
                failures.append({
                    "input": ex.get("input", "")[:500],
                    "output": ex.get("output", "")[:500],
                    "judge_said": "PASS" if predicted else "FAIL",
                    "human_said": "PASS" if actual else "FAIL",
                })
                if len(failures) >= max_failures:
                    break
        return failures

    @staticmethod
    def classify_improvement(delta: float) -> str:
        """Classify an improvement delta into a human-readable level.

        Args:
            delta: The F1 improvement (best - seed).

        Returns:
            One of "high", "medium", "low", or "none".
        """
        if delta > 0.05:
            return "high"
        elif delta > 0.02:
            return "medium"
        elif delta > 0:
            return "low"
        else:
            return "none"

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
        dataset_items: list | None = None,
        accumulator: TokenAccumulator | None = None,
        **kwargs,
    ) -> PromptOptimizationResult:
        """Optimize the preamble. Subclasses must implement this."""
        ...
