"""MIPROv2: Multi-stage Instruction and Demonstration Joint Optimizer.

Implements an evalyn-adapted version of "Optimizing Instructions and Demonstrations Jointly"
(MIPROv2). The optimizer uses a three-stage pipeline:

1. Stage 1 - Instruction Generation: Generate N diverse preamble candidates from
   disagreement patterns and the current preamble.
2. Stage 2 - Demo Bootstrap: Select K diverse demonstrations from correct evaluations,
   bucketed by label (PASS/FAIL) and input length (short/long).
3. Stage 3 - Joint Selection: Score each instruction alone, take top 3, then greedily
   add demos one at a time, keeping each demo only if F1 improves.

The final preamble is the best instruction concatenated with the selected demo text.

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

from ...defaults import DEFAULT_EVAL_MODEL
from ...models import Annotation, DatasetItem, MetricResult
from ...utils.api_client import GeminiClient
from ..base_optimizer import BaseOptimizer
from ..models import (
    DisagreementAnalysis,
    PromptOptimizationResult,
    TokenAccumulator,
)
from ..utils import (
    build_dataset_from_annotations,
    build_full_prompt,
    parse_candidates_response,
)


@dataclass
class MIPROv2Config:
    """Configuration for MIPROv2 optimizer."""

    num_instructions: int = 6
    num_demos: int = 3
    eval_samples: int = 10
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL


INSTRUCTION_GENERATION_TEMPLATE = """You are an expert at improving LLM evaluation prompts.

Current judge preamble:
{current_preamble}

FIXED RUBRIC (for context - do not modify):
{rubric_text}

DISAGREEMENT SUMMARY:
- False positives ({fp_count} cases - judge too lenient): {fp_summary}
- False negatives ({fn_count} cases - judge too strict): {fn_summary}

Generate {num_instructions} diverse improved versions of the PREAMBLE (not the rubric).
Each version should:
1. Address the specific failure patterns above
2. Be a complete, standalone preamble (100-300 words)
3. NOT include the rubric (appended separately)
4. Use a distinct strategy (e.g., stricter tone, edge-case guidance, role framing)

Output format - return ONLY a JSON array of preamble strings:
["preamble 1", "preamble 2", ...]"""


class MIPROv2Optimizer(BaseOptimizer):
    """
    MIPROv2: Multi-stage Instruction and Demonstration Joint Optimizer.

    Uses a three-stage pipeline to jointly optimize instructions and demonstrations
    for LLM judge calibration.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.

    Algorithm:
    1. INSTRUCT: Generate N diverse preamble candidates from disagreement patterns
    2. DEMO: Bootstrap K diverse demonstrations from correct evaluations
    3. SELECT: Score instructions, take top 3, greedily add demos
    """

    def __init__(
        self,
        config: MIPROv2Config | None = None,
        api_key: str | None = None,
    ):
        """Initialize MIPROv2 optimizer.

        Args:
            config: MIPROv2 configuration (uses defaults if not provided)
            api_key: Optional API key for Gemini (default: from env)
        """
        super().__init__(config=config or MIPROv2Config(), api_key=api_key)
        self._task_client_instance: GeminiClient | None = None

    @property
    def _task_client(self) -> GeminiClient:
        """Lazy-initialized task LLM client (for generating instructions)."""
        if self._task_client_instance is None:
            self._task_client_instance = GeminiClient(
                model=self.config.task_model,
                temperature=0.8,
                api_key=self._api_key,
                timeout=120,
            )
        return self._task_client_instance

    # ------------------------------------------------------------------
    # Stage 1: Instruction Generation
    # ------------------------------------------------------------------

    def _generate_instructions(
        self,
        current_preamble: str,
        disagreements: DisagreementAnalysis | None,
        rubric: list[str],
        accumulator: TokenAccumulator | None = None,
    ) -> list[str]:
        """Generate diverse preamble candidates from disagreement patterns.

        Args:
            current_preamble: Current preamble to improve
            disagreements: Analysis of false positives/negatives
            rubric: Fixed rubric criteria
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            List of candidate preamble strings
        """
        rubric_text = "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"

        fp_count = 0
        fn_count = 0
        fp_summary = "(none)"
        fn_summary = "(none)"

        if disagreements:
            fp_count = len(disagreements.false_positives)
            fn_count = len(disagreements.false_negatives)
            if disagreements.false_positives:
                fp_parts = []
                for case in disagreements.false_positives[:3]:
                    fp_parts.append(
                        f"output={case.output_text[:100]}... "
                        f"judge_reason={case.judge_reason[:80]}..."
                    )
                fp_summary = "; ".join(fp_parts)
            if disagreements.false_negatives:
                fn_parts = []
                for case in disagreements.false_negatives[:3]:
                    fn_parts.append(
                        f"output={case.output_text[:100]}... "
                        f"judge_reason={case.judge_reason[:80]}..."
                    )
                fn_summary = "; ".join(fn_parts)

        prompt = INSTRUCTION_GENERATION_TEMPLATE.format(
            current_preamble=current_preamble,
            rubric_text=rubric_text,
            fp_count=fp_count,
            fp_summary=fp_summary,
            fn_count=fn_count,
            fn_summary=fn_summary,
            num_instructions=self.config.num_instructions,
        )

        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            return parse_candidates_response(result.text)
        except Exception:
            logger.warning("Failed to generate instruction candidates", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Stage 2: Demo Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_demos(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[str]:
        """Select diverse demonstrations from correct evaluations.

        Diversity heuristic: bucket by (label=PASS/FAIL) x (input length < median / >= median),
        then sample evenly across buckets.

        Args:
            train_examples: Training examples with input, output, expected fields

        Returns:
            List of formatted demo strings
        """
        # Only use examples where the evaluation was correct
        # (In the training set context, all examples have human labels -
        # we use them as "correct" demonstrations)
        correct_examples = [ex for ex in train_examples if ex.get("expected")]
        if not correct_examples:
            return []

        # Compute median input length for bucketing
        input_lengths = [len(ex.get("input", "")) for ex in correct_examples]
        med_length = median(input_lengths) if input_lengths else 0

        # Create 4 buckets: (PASS/FAIL) x (short/long)
        buckets: dict[str, list[dict[str, Any]]] = {
            "PASS_short": [],
            "PASS_long": [],
            "FAIL_short": [],
            "FAIL_long": [],
        }

        for ex in correct_examples:
            label = ex.get("expected", "FAIL")
            length_key = "short" if len(ex.get("input", "")) < med_length else "long"
            bucket_key = f"{label}_{length_key}"
            if bucket_key in buckets:
                buckets[bucket_key].append(ex)

        # Sample evenly across non-empty buckets
        non_empty_buckets = {k: v for k, v in buckets.items() if v}
        if not non_empty_buckets:
            return []

        selected: list[dict[str, Any]] = []
        remaining = self.config.num_demos
        bucket_keys = list(non_empty_buckets.keys())
        random.shuffle(bucket_keys)

        # Round-robin across buckets
        idx = 0
        while remaining > 0 and any(non_empty_buckets.get(k) for k in bucket_keys):
            key = bucket_keys[idx % len(bucket_keys)]
            bucket = non_empty_buckets.get(key, [])
            if bucket:
                chosen = random.choice(bucket)
                selected.append(chosen)
                bucket.remove(chosen)
                remaining -= 1
            idx += 1

        # Format each demo
        demos = []
        for ex in selected:
            label = ex.get("expected", "FAIL")
            reason = ex.get("reason", "correct evaluation")
            demo = (
                f"Example: INPUT: {ex.get('input', '')[:300]} "
                f"OUTPUT: {ex.get('output', '')[:300]} "
                f"-> VERDICT: {label.lower()} REASON: {reason}"
            )
            demos.append(demo)

        return demos

    # ------------------------------------------------------------------
    # Stage 3: Joint Selection
    # ------------------------------------------------------------------

    def _select_best_instructions(
        self,
        instructions: list[str],
        rubric: list[str],
        train_examples: list[dict[str, Any]],
        accumulator: TokenAccumulator | None = None,
    ) -> list[str]:
        """Score each instruction on the training set and return the top 3.

        Args:
            instructions: Candidate preamble strings
            rubric: Fixed rubric criteria
            train_examples: Training examples for scoring
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            Top 3 instructions sorted by F1 score (descending)
        """
        scored: list[tuple[str, float]] = []
        for instruction in instructions:
            f1 = self.score_preamble(
                instruction,
                rubric,
                train_examples,
                accumulator,
                max_samples=self.config.eval_samples,
            )
            scored.append((instruction, f1))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:3]]

    def _greedy_demo_selection(
        self,
        instruction: str,
        demos: list[str],
        rubric: list[str],
        train_examples: list[dict[str, Any]],
        accumulator: TokenAccumulator | None = None,
    ) -> list[str]:
        """Greedily add demos to the instruction, keeping only those that improve F1.

        Args:
            instruction: The base instruction preamble
            demos: Available demo strings to try adding
            rubric: Fixed rubric criteria
            train_examples: Training examples for scoring
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            List of selected demo strings
        """
        base_preamble = instruction
        current_f1 = self.score_preamble(
            base_preamble, rubric, train_examples, accumulator, max_samples=self.config.eval_samples
        )

        selected_demos: list[str] = []
        remaining_demos = list(demos)

        for _ in range(len(demos)):
            if not remaining_demos:
                break

            best_demo = None
            best_f1 = current_f1

            for demo in remaining_demos:
                candidate_preamble = self._embed_demos(instruction, selected_demos + [demo])
                f1 = self.score_preamble(
                    candidate_preamble,
                    rubric,
                    train_examples,
                    accumulator,
                    max_samples=self.config.eval_samples,
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_demo = demo

            if best_demo is not None:
                selected_demos.append(best_demo)
                remaining_demos.remove(best_demo)
                current_f1 = best_f1
            else:
                # No demo improved F1, stop
                break

        return selected_demos

    def _embed_demos(self, instruction: str, selected_demos: list[str]) -> str:
        """Concatenate instruction with selected demo text.

        Args:
            instruction: The base instruction preamble
            selected_demos: List of formatted demo strings

        Returns:
            Combined preamble with instruction and demos
        """
        if not selected_demos:
            return instruction

        demo_text = "\n".join(selected_demos)
        return instruction + "\n\nHere are examples of correct evaluations:\n" + demo_text

    # ------------------------------------------------------------------
    # Main optimize entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: list[str],
        metric_results: list[MetricResult],
        annotations: list[Annotation],
        disagreements: DisagreementAnalysis | None = None,
        dataset_items: list[DatasetItem] | None = None,
        current_preamble: str = "",
        accumulator: TokenAccumulator | None = None,
        **kwargs: Any,
    ) -> PromptOptimizationResult:
        """Run MIPROv2 optimization.

        The optimization process:
        1. Generate N diverse instruction candidates from disagreements
        2. Bootstrap K diverse demonstrations from correct evaluations
        3. Score instructions alone, take top 3
        4. Greedily add demos to each top instruction
        5. Return the best instruction+demo combination

        Args:
            metric_id: The metric being calibrated
            current_rubric: Fixed rubric criteria (NOT optimized)
            metric_results: Evaluation results from the judge
            annotations: Human annotations
            disagreements: Analysis of false positives/negatives
            dataset_items: Optional dataset items for context
            current_preamble: Current preamble text (the part to optimize)
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            PromptOptimizationResult with optimized preamble
        """
        # Build datasets
        trainset, valset = build_dataset_from_annotations(
            metric_results, annotations, dataset_items, 0.7
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=(
                    "Not enough data for MIPROv2 optimization (need at least 5 annotated examples)"
                ),
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        if current_preamble:
            seed_preamble = current_preamble.strip()
        else:
            seed_preamble = (
                f"You are an expert evaluator for the metric: {metric_id}. "
                "Carefully analyze the output quality and provide an honest assessment."
            )

        # Stage 1: Generate diverse instruction candidates
        instructions = self._generate_instructions(
            seed_preamble, disagreements, current_rubric, accumulator
        )

        if not instructions:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="MIPROv2 failed to generate instruction candidates",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Include seed as a candidate for fair comparison
        instructions.insert(0, seed_preamble)

        # Stage 2: Bootstrap diverse demonstrations
        demos = self._bootstrap_demos(trainset)

        # Stage 3: Joint selection
        # Score instructions alone, take top 3
        top_instructions = self._select_best_instructions(
            instructions, current_rubric, trainset, accumulator
        )

        # For each top instruction, greedily add demos
        best_preamble = top_instructions[0] if top_instructions else seed_preamble
        best_f1 = 0.0

        for instruction in top_instructions:
            if demos:
                selected_demos = self._greedy_demo_selection(
                    instruction, demos, current_rubric, trainset, accumulator
                )
                candidate = self._embed_demos(instruction, selected_demos)
            else:
                candidate = instruction

            f1 = self.score_preamble(
                candidate, current_rubric, valset, accumulator, max_samples=self.config.eval_samples
            )
            if f1 > best_f1:
                best_f1 = f1
                best_preamble = candidate

        # Evaluate seed on validation set for comparison
        seed_f1 = self.score_preamble(
            seed_preamble, current_rubric, valset, accumulator, max_samples=self.config.eval_samples
        )

        # Determine improvement
        improvement_delta = best_f1 - seed_f1
        estimated_improvement = self.classify_improvement(improvement_delta)

        full_optimized_prompt = build_full_prompt(best_preamble, current_rubric)

        reasoning = (
            f"MIPROv2 optimization completed. "
            f"Generated {len(instructions) - 1} instruction candidates. "
            f"Bootstrapped {len(demos)} demos. "
            f"Top 3 instructions evaluated with greedy demo selection. "
            f"Best val F1: {best_f1:.3f} (seed: {seed_f1:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return PromptOptimizationResult(
            original_rubric=current_rubric,
            improved_rubric=current_rubric,
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=current_preamble,
            optimized_preamble=best_preamble,
            full_prompt=full_optimized_prompt,
        )


__all__ = ["MIPROv2Config", "MIPROv2Optimizer"]
