"""TextGrad: Automatic Differentiation via Text.

Implements a TextGrad-inspired optimizer for preamble calibration.
Iteratively refines the judge prompt through critique-and-revise cycles:
1. Score current preamble on training examples
2. Collect failure cases (prediction != human label)
3. Ask LLM to critique the preamble based on failures
4. Ask LLM to revise the preamble based on critique
5. Keep the best preamble seen (greedy-best tracking)
6. Early stopping when no improvement for N iterations

Reference: https://arxiv.org/abs/2406.07496

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from tqdm import tqdm

from ..defaults import DEFAULT_EVAL_MODEL
from ..utils.api_client import GeminiClient
from .base_optimizer import BaseOptimizer
from .models import (
    PromptOptimizationResult,
    TokenAccumulator,
)


@dataclass
class TextGradConfig:
    """Configuration for TextGrad optimizer."""

    max_iterations: int = 8
    improvement_threshold: float = 0.01
    num_failure_examples: int = 5
    early_stop_patience: int = 3
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL


CRITIQUE_TEMPLATE = """You are an expert at analyzing LLM evaluation prompts.

Below is a preamble used to instruct an LLM judge. The judge uses this preamble together with a fixed rubric to decide PASS/FAIL on outputs.

## PREAMBLE
{preamble}

## FAILURE CASES
The preamble led to incorrect judgments on the following examples.
For each case the judge's verdict disagreed with the human ground-truth label.

{failures_text}

Given this preamble and these failures, what specific weaknesses caused these errors?
Be concrete: refer to missing instructions, ambiguous wording, or blind spots.
Return ONLY the critique as plain text (no JSON)."""


REVISE_TEMPLATE = """You are an expert at improving LLM evaluation prompts.

Below is a preamble used to instruct an LLM judge, followed by a critique of its weaknesses.

## CURRENT PREAMBLE
{preamble}

## CRITIQUE
{critique}

Revise this preamble to address the following critique. Keep the same general structure but fix the identified weaknesses.

Guidelines:
- Keep it concise but complete (100-300 words)
- Clear role definition
- Address the specific failure patterns highlighted in the critique
- Do NOT include any rubric criteria (those are appended separately)

Return ONLY the revised preamble text (no JSON, no explanation)."""


class TextGradOptimizer(BaseOptimizer):
    """TextGrad-inspired iterative preamble optimizer.

    Uses critique-and-revise cycles to refine the judge preamble.
    Each iteration:
    1. Collects failure cases where judge disagrees with human labels
    2. Asks LLM to critique the current preamble based on those failures
    3. Asks LLM to revise the preamble based on the critique
    4. Scores the revised preamble and keeps it if it improves

    Greedy-best tracking ensures we never regress.
    Early stopping triggers when no improvement for ``early_stop_patience`` iterations.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.
    """

    def __init__(
        self,
        config: TextGradConfig | None = None,
        api_key: str | None = None,
    ):
        """Initialize TextGrad optimizer.

        Args:
            config: TextGrad configuration (uses defaults if not provided).
            api_key: Optional API key for Gemini (default: from env).
        """
        super().__init__(config=config or TextGradConfig(), api_key=api_key)
        self._task_client: GeminiClient | None = None
        self._scorer_client: GeminiClient | None = None

    # ------------------------------------------------------------------
    # Lazy client properties
    # ------------------------------------------------------------------

    @property
    def task_client(self) -> GeminiClient:
        """Lazy-initialized task LLM client (for critique and revision)."""
        if self._task_client is None:
            self._task_client = GeminiClient(
                model=self.config.task_model,
                temperature=0.7,
                api_key=self._api_key,
                timeout=120,
            )
        return self._task_client

    @property
    def scorer_client(self) -> GeminiClient:
        """Lazy-initialized scorer LLM client (for evaluating prompts)."""
        if self._scorer_client is None:
            self._scorer_client = GeminiClient(
                model=self.config.scorer_model,
                temperature=0.0,
                api_key=self._api_key,
                timeout=120,
            )
        return self._scorer_client

    # ------------------------------------------------------------------
    # Core algorithm steps
    # ------------------------------------------------------------------

    # _collect_failures is inherited from BaseOptimizer

    def _critique(
        self,
        preamble: str,
        failures: list[dict[str, Any]],
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Ask LLM to critique the preamble based on failure cases.

        Args:
            preamble: Current preamble text.
            failures: List of failure dicts with input/output/judge_said/human_said.
            accumulator: Optional token tracker.

        Returns:
            Critique text identifying specific weaknesses.
        """
        failures_text = ""
        for i, f in enumerate(failures, 1):
            failures_text += (
                f"Case {i}:\n"
                f"  Input: {f['input'][:300]}\n"
                f"  Output: {f['output'][:300]}\n"
                f"  Judge said: {f['judge_said']}, Human said: {f['human_said']}\n\n"
            )

        prompt = CRITIQUE_TEMPLATE.format(
            preamble=preamble,
            failures_text=failures_text,
        )

        try:
            result = self.task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            return result.text.strip()
        except Exception:
            logger.warning("Failed to generate critique, using fallback", exc_info=True)
            return "Unable to generate critique."

    def _revise(
        self,
        preamble: str,
        critique: str,
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Ask LLM to revise the preamble based on the critique.

        Args:
            preamble: Current preamble text.
            critique: Critique identifying weaknesses.
            accumulator: Optional token tracker.

        Returns:
            Revised preamble text.
        """
        prompt = REVISE_TEMPLATE.format(
            preamble=preamble,
            critique=critique,
        )

        try:
            result = self.task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            return result.text.strip()
        except Exception:
            logger.warning("Failed to revise preamble, keeping current version", exc_info=True)
            return preamble  # Fall back to current preamble on error

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: list[str],
        current_preamble: str = "",
        metric_results: list,
        annotations: list,
        disagreements: Any = None,
        dataset_items: list | None = None,
        accumulator: TokenAccumulator | None = None,
        **kwargs: Any,
    ) -> PromptOptimizationResult:
        """Run TextGrad optimization loop.

        Algorithm:
        1. Split data into train/val sets
        2. Score seed preamble
        3. For each iteration:
           a. Collect failure cases from training set
           b. Critique the current preamble
           c. Revise the preamble
           d. Score the revised preamble
           e. Keep if better (greedy-best)
           f. Early stop if no improvement for N iterations
        4. Validate best preamble on held-out data

        Args:
            metric_id: The metric being calibrated.
            current_rubric: Fixed rubric criteria (NOT optimized).
            current_preamble: Current preamble text (the part to optimize).
            metric_results: Evaluation results from the judge.
            annotations: Human annotations.
            disagreements: Unused (accepted for interface compatibility).
            dataset_items: Optional dataset items for context.
            accumulator: Optional TokenAccumulator to track token usage.

        Returns:
            PromptOptimizationResult with optimized preamble.
        """
        # Build datasets
        trainset, valset = self.split_train_val(
            metric_results, annotations, dataset_items
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=(
                    "Not enough data for TextGrad optimization "
                    "(need at least 5 annotated examples)"
                ),
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Initialize with seed prompt
        if current_preamble:
            seed_preamble = current_preamble.strip()
        else:
            seed_preamble = (
                f"You are an expert evaluator for the metric: {metric_id}. "
                "Carefully analyze the output quality and provide an honest assessment."
            )

        # Score seed preamble on training set
        best_preamble = seed_preamble
        best_score = self.score_preamble(
            best_preamble, current_rubric, trainset, accumulator
        )
        seed_score = best_score

        current_preamble_text = seed_preamble
        no_improvement_count = 0

        # Optimization loop
        pbar = tqdm(
            range(1, self.config.max_iterations + 1),
            desc="TextGrad",
            unit="iter",
        )
        for iteration in pbar:
            # Step 1: Collect failure cases
            failures = self._collect_failures(
                current_preamble_text, current_rubric, trainset,
                max_failures=self.config.num_failure_examples, accumulator=accumulator,
            )

            if not failures:
                # Perfect score on training set - no failures to learn from
                break

            # Step 2: Critique
            critique = self._critique(
                current_preamble_text, failures, accumulator
            )

            # Step 3: Revise
            revised_preamble = self._revise(
                current_preamble_text, critique, accumulator
            )

            # Step 4: Score revised preamble
            revised_score = self.score_preamble(
                revised_preamble, current_rubric, trainset, accumulator
            )

            # Step 5: Greedy-best tracking
            if revised_score > best_score + self.config.improvement_threshold:
                best_preamble = revised_preamble
                best_score = revised_score
                current_preamble_text = revised_preamble
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # Still use revised preamble as current to allow exploration
                current_preamble_text = revised_preamble

            pbar.set_postfix({"best_f1": f"{best_score:.1%}"})

            # Step 6: Early stopping
            if no_improvement_count >= self.config.early_stop_patience:
                break

        # Validate on held-out data
        val_score = self.score_preamble(
            best_preamble, current_rubric, valset, accumulator
        )
        seed_val_score = self.score_preamble(
            seed_preamble, current_rubric, valset, accumulator
        )

        # Determine improvement level
        improvement_delta = val_score - seed_val_score
        estimated_improvement = self.classify_improvement(improvement_delta)

        reasoning = (
            f"TextGrad optimization completed. "
            f"Train F1: {best_score:.3f} (seed: {seed_score:.3f}). "
            f"Val F1: {val_score:.3f} (seed: {seed_val_score:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return self.build_result(
            original_preamble=current_preamble or seed_preamble,
            optimized_preamble=best_preamble,
            rubric=current_rubric,
            reasoning=reasoning,
            estimated_improvement=estimated_improvement,
        )


__all__ = ["TextGradConfig", "TextGradOptimizer"]
