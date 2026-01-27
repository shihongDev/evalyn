"""BasicOptimizer: Single-shot LLM-based preamble optimizer.

The simplest calibration approach - uses one LLM call to analyze
disagreement patterns and suggest improvements.
"""

from __future__ import annotations

import json
from typing import List, Optional

from ..defaults import DEFAULT_EVAL_MODEL
from ..utils.api_client import GeminiClient
from .models import (
    AlignmentMetrics,
    DisagreementAnalysis,
    PromptOptimizationResult,
    TokenAccumulator,
)
from .utils import build_full_prompt


class BasicOptimizer:
    """
    Single-shot LLM-based preamble optimizer (simplest approach).

    Uses one LLM call to analyze disagreement patterns and suggest improvements.
    No iteration, no search - just pattern analysis and a single suggested preamble.

    IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
    The rubric (evaluation criteria) is kept FIXED as defined by humans.
    """

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client: Optional[GeminiClient] = None

    @property
    def client(self) -> GeminiClient:
        """Lazy-initialized Gemini API client."""
        if self._client is None:
            self._client = GeminiClient(
                model=self.model,
                temperature=0.7,  # Higher for diverse preamble generation
                api_key=self._api_key,
                timeout=90,
            )
        return self._client

    def optimize(
        self,
        metric_id: str,
        current_rubric: List[str],
        disagreements: DisagreementAnalysis,
        alignment_metrics: AlignmentMetrics,
        current_preamble: str = "",
        accumulator: Optional[TokenAccumulator] = None,
    ) -> PromptOptimizationResult:
        """
        Analyze disagreements and suggest preamble improvements.

        IMPORTANT: Only the preamble is optimized. The rubric stays fixed.

        Args:
            metric_id: The metric being calibrated
            current_rubric: Fixed rubric criteria (NOT optimized)
            disagreements: Analysis of false positives/negatives
            alignment_metrics: Computed alignment metrics
            current_preamble: Current preamble text (the part to optimize)
            accumulator: Optional TokenAccumulator to track token usage
        """
        # Use default preamble if none provided
        if not current_preamble:
            current_preamble = f"You are an expert evaluator for the metric: {metric_id}. Carefully analyze the output quality and provide an honest assessment."

        # Build the optimization prompt
        prompt = self._build_optimization_prompt(
            metric_id,
            current_rubric,
            current_preamble,
            disagreements,
            alignment_metrics,
        )

        try:
            result = self.client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            return self._parse_optimization_response(
                result.text, current_rubric, current_preamble
            )
        except Exception as e:
            # Return a fallback result on error
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=f"Optimization failed: {e}",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

    def _build_optimization_prompt(
        self,
        metric_id: str,
        current_rubric: List[str],
        current_preamble: str,
        disagreements: DisagreementAnalysis,
        alignment_metrics: AlignmentMetrics,
    ) -> str:
        """Build prompt for preamble optimization (rubric stays fixed)."""

        # Format current rubric
        rubric_text = (
            "\n".join([f"- {r}" for r in current_rubric])
            if current_rubric
            else "(no rubric defined)"
        )

        # Format false positives (judge too lenient)
        fp_examples = ""
        for i, case in enumerate(disagreements.false_positives[:3], 1):
            fp_examples += f"""
Example {i} (Judge said PASS, Human said FAIL):
  Input: {case.input_text[:300]}...
  Output: {case.output_text[:300]}...
  Judge reason: {case.judge_reason}
  Human notes: {case.human_notes}
"""

        # Format false negatives (judge too strict)
        fn_examples = ""
        for i, case in enumerate(disagreements.false_negatives[:3], 1):
            fn_examples += f"""
Example {i} (Judge said FAIL, Human said PASS):
  Input: {case.input_text[:300]}...
  Output: {case.output_text[:300]}...
  Judge reason: {case.judge_reason}
  Human notes: {case.human_notes}
"""

        prompt = f"""You are an expert at improving LLM evaluation prompts.

## METRIC BEING CALIBRATED
Metric ID: {metric_id}

## CURRENT PREAMBLE (the part you will improve)
{current_preamble}

## FIXED RUBRIC (do NOT modify - for context only)
{rubric_text}

## ALIGNMENT STATISTICS
- Accuracy: {alignment_metrics.accuracy:.1%}
- Precision: {alignment_metrics.precision:.1%} (when judge says PASS, human agrees {alignment_metrics.precision:.1%} of the time)
- Recall: {alignment_metrics.recall:.1%} (judge catches {alignment_metrics.recall:.1%} of human-approved outputs)
- F1 Score: {alignment_metrics.f1:.1%}
- Cohen's Kappa: {alignment_metrics.cohens_kappa:.3f}

## FALSE POSITIVES ({len(disagreements.false_positives)} cases)
Judge was too LENIENT - said PASS when human said FAIL:
{fp_examples if fp_examples else "(none)"}

## FALSE NEGATIVES ({len(disagreements.false_negatives)} cases)
Judge was too STRICT - said FAIL when human said PASS:
{fn_examples if fn_examples else "(none)"}

## YOUR TASK
Analyze the disagreement patterns and write an improved PREAMBLE (not the rubric).
The preamble is the system prompt/instructions that frame how the judge should evaluate.

Return a JSON object with:
{{
  "optimized_preamble": "Your improved preamble text (100-300 words)",
  "improvement_reasoning": "Explanation of what patterns you found and why these changes help",
  "estimated_improvement": "low|medium|high"
}}

Guidelines for good preambles:
- Clear role definition (e.g., "You are an expert evaluator...")
- Specific instructions on how to interpret the rubric
- Guidance on common edge cases based on the disagreement patterns
- Emphasis on what matters most for this metric
- If many false positives: emphasize being more STRICT in evaluation
- If many false negatives: emphasize being more LENIENT or nuanced
- Keep it concise but complete (100-300 words ideal)

IMPORTANT: Do NOT suggest changes to the rubric. The rubric is fixed.

Return ONLY the JSON object, no other text."""

        return prompt

    def _parse_optimization_response(
        self,
        response_text: str,
        original_rubric: List[str],
        original_preamble: str,
    ) -> PromptOptimizationResult:
        """Parse the LLM response into a structured result."""

        # Try to extract JSON from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            # Find JSON object
            start = text.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = text[start : i + 1]
                            parsed = json.loads(json_str)

                            optimized_preamble = parsed.get(
                                "optimized_preamble", original_preamble
                            )
                            full_prompt = build_full_prompt(
                                optimized_preamble, original_rubric
                            )

                            return PromptOptimizationResult(
                                original_rubric=original_rubric,
                                improved_rubric=original_rubric,  # Rubric stays fixed
                                improvement_reasoning=parsed.get(
                                    "improvement_reasoning", ""
                                ),
                                suggested_additions=[],
                                suggested_removals=[],
                                estimated_improvement=parsed.get(
                                    "estimated_improvement", "unknown"
                                ),
                                original_preamble=original_preamble,
                                optimized_preamble=optimized_preamble,
                                full_prompt=full_prompt,
                            )
        except Exception:
            pass

        # Fallback if parsing fails
        return PromptOptimizationResult(
            original_rubric=original_rubric,
            improved_rubric=original_rubric,
            improvement_reasoning=f"Could not parse LLM response: {response_text[:200]}",
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement="unknown",
            original_preamble=original_preamble,
        )


__all__ = ["BasicOptimizer"]
