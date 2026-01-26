"""APE: Automatic Prompt Engineer.

Implements a seed-based APE algorithm for prompt optimization.
Uses the current prompt as a template and generates variations based on disagreement patterns.
Candidate selection uses UCB (Upper Confidence Bound) for efficient evaluation.

Key concepts:
- Seed-based generation: Uses current prompt + disagreements to generate candidates
- UCB selection: Multi-armed bandit approach to efficiently evaluate candidates
- Minimal API calls: UCB reduces evaluation cost compared to exhaustive search

Reference: https://arxiv.org/abs/2211.01910

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..models import Annotation, DatasetItem, MetricResult
from ..utils.api_client import GeminiClient, GenerateResult
from .calibration import (
    AlignmentMetrics,
    DisagreementAnalysis,
    PromptOptimizationResult,
    TokenAccumulator,
    build_full_prompt,
)


@dataclass
class APEConfig:
    """Configuration for APE optimizer."""

    num_candidates: int = 10  # Number of candidate prompts to generate
    eval_rounds: int = 5  # UCB evaluation rounds
    eval_samples_per_round: int = 5  # Samples per candidate per round
    generator_model: str = "gemini-2.5-flash"  # Model for generating candidates
    scorer_model: str = "gemini-2.5-flash-lite"  # Model for scoring (judge)
    exploration_weight: float = 1.0  # UCB exploration parameter
    train_split: float = 0.7  # Train/val split ratio
    temperature: float = 0.8  # Higher for diverse candidate generation
    timeout: int = 120  # API timeout in seconds


PROPOSAL_TEMPLATE = """You are an expert at improving LLM evaluation prompts.

Current judge preamble:
{current_preamble}

This preamble (when combined with the rubric below) has alignment issues with human labels:

FIXED RUBRIC (for context - do not modify):
{rubric_text}

FALSE POSITIVES ({fp_count} cases - judge is too LENIENT):
{false_positive_examples}

FALSE NEGATIVES ({fn_count} cases - judge is too STRICT):
{false_negative_examples}

Generate {num_candidates} improved versions of the PREAMBLE (not the rubric).
Each version should:
1. Address the specific failure patterns shown above
2. Be a complete, standalone preamble (100-300 words)
3. NOT include the rubric (that will be appended separately)
4. Focus on improving evaluation quality for this specific metric

Output format - return ONLY a JSON array of preamble strings:
["preamble 1", "preamble 2", ...]"""


class APEOptimizer:
    """
    APE: Automatic Prompt Engineer with UCB selection.

    Uses seed-based candidate generation from the current prompt,
    with UCB-based evaluation for efficient candidate selection.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.

    Algorithm:
    1. PROPOSE: Generate N candidate prompts from current prompt + disagreements
    2. SCORE: Evaluate candidates on validation set using UCB sampling
    3. SELECT: Return best-scoring prompt
    """

    def __init__(
        self,
        config: Optional[APEConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize APE optimizer.

        Args:
            config: APE configuration (uses defaults if not provided)
            api_key: Optional API key for Gemini (default: from env)
        """
        self.config = config or APEConfig()
        self._api_key = api_key
        self._generator_client: Optional[GeminiClient] = None
        self._scorer_client: Optional[GeminiClient] = None

    @property
    def generator_client(self) -> GeminiClient:
        """Lazy-initialized generator LLM client (for generating candidates)."""
        if self._generator_client is None:
            self._generator_client = GeminiClient(
                model=self.config.generator_model,
                temperature=self.config.temperature,
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._generator_client

    @property
    def scorer_client(self) -> GeminiClient:
        """Lazy-initialized scorer LLM client (for evaluating prompts)."""
        if self._scorer_client is None:
            self._scorer_client = GeminiClient(
                model=self.config.scorer_model,
                temperature=0.0,  # Deterministic for scoring
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._scorer_client

    def _build_dataset_from_annotations(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert calibration data to train/val sets.
        Each example contains: input, output, expected (human label).
        """
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        items_by_call: Dict[str, DatasetItem] = {}
        if dataset_items:
            for item in dataset_items:
                call_id = item.metadata.get("call_id", item.id)
                items_by_call[call_id] = item

        examples = []
        for res in metric_results:
            ann = ann_by_call.get(res.call_id)
            if not ann:
                continue

            item = items_by_call.get(res.call_id)
            input_text = ""
            output_text = ""
            if item:
                input_text = json.dumps(item.input, default=str) if item.input else ""
                output_text = str(item.output) if item.output else ""

            examples.append(
                {
                    "input": input_text,
                    "output": output_text,
                    "expected": "PASS" if ann.label else "FAIL",
                    "call_id": res.call_id,
                }
            )

        # Shuffle and split into train/val
        random.shuffle(examples)
        split_idx = int(len(examples) * self.config.train_split)
        trainset = examples[:split_idx]
        valset = examples[split_idx:]

        return trainset, valset

    def _propose_candidates(
        self,
        current_preamble: str,
        rubric: List[str],
        disagreements: DisagreementAnalysis,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> List[str]:
        """
        Generate candidate prompts based on current prompt and disagreement patterns.

        Args:
            current_preamble: Current preamble to improve
            rubric: Fixed rubric criteria
            disagreements: Analysis of false positives/negatives
            accumulator: Optional TokenAccumulator to track token usage
        """
        # Format rubric for context
        rubric_text = (
            "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"
        )

        # Format false positive examples
        fp_examples = ""
        for i, case in enumerate(disagreements.false_positives[:3], 1):
            fp_examples += f"""
Example {i}:
  Output: {case.output_text[:200]}...
  Judge reason: {case.judge_reason[:150]}...
  Human notes: {case.human_notes[:100]}...
"""
        if not fp_examples:
            fp_examples = "(none)"

        # Format false negative examples
        fn_examples = ""
        for i, case in enumerate(disagreements.false_negatives[:3], 1):
            fn_examples += f"""
Example {i}:
  Output: {case.output_text[:200]}...
  Judge reason: {case.judge_reason[:150]}...
  Human notes: {case.human_notes[:100]}...
"""
        if not fn_examples:
            fn_examples = "(none)"

        # Build proposal prompt
        prompt = PROPOSAL_TEMPLATE.format(
            current_preamble=current_preamble,
            rubric_text=rubric_text,
            fp_count=len(disagreements.false_positives),
            false_positive_examples=fp_examples,
            fn_count=len(disagreements.false_negatives),
            false_negative_examples=fn_examples,
            num_candidates=self.config.num_candidates,
        )

        try:
            result = self.generator_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            return self._parse_candidates_response(result.text)
        except Exception:
            return []

    def _parse_candidates_response(self, response: str) -> List[str]:
        """Parse the generator's response to extract candidate preambles."""
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        # Find JSON array
        try:
            start = text.find("[")
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "[":
                        depth += 1
                    elif text[i] == "]":
                        depth -= 1
                        if depth == 0:
                            json_str = text[start : i + 1]
                            candidates = json.loads(json_str)
                            # Ensure all items are strings
                            return [str(c) for c in candidates if c]
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def _score_candidate(
        self,
        preamble: str,
        rubric: List[str],
        examples: List[Dict[str, Any]],
        accumulator: Optional[TokenAccumulator] = None,
    ) -> float:
        """
        Score a candidate prompt by running it as a judge on examples.

        Args:
            preamble: The preamble to evaluate
            rubric: Fixed rubric criteria
            examples: Examples to evaluate on
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            F1 score
        """
        full_prompt = build_full_prompt(preamble, rubric)
        metrics = AlignmentMetrics()

        for ex in examples:
            # Build the evaluation prompt for this example
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
                judge_pass = self._parse_judge_response(result.text)
                human_pass = ex.get("expected") == "PASS"
                metrics.record(judge_pass, human_pass)
            except Exception:
                # On error, skip this example
                pass

        return metrics.f1

    def _parse_judge_response(self, response: str) -> bool:
        """Parse judge response to extract pass/fail verdict."""
        text = response.strip().lower()

        # Try to find JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return bool(data.get("passed", False))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for keywords
        if "true" in text or '"passed": true' in text:
            return True
        if "false" in text or '"passed": false' in text:
            return False

        # Default to fail if unclear
        return False

    def _ucb_select(
        self,
        candidates: List[str],
        rubric: List[str],
        val_examples: List[Dict[str, Any]],
        accumulator: Optional[TokenAccumulator] = None,
    ) -> Tuple[str, float]:
        """
        Select the best candidate using UCB (Upper Confidence Bound) algorithm.

        UCB balances exploration (trying less-evaluated candidates) with
        exploitation (focusing on high-scoring candidates).

        Args:
            candidates: Candidate preambles to evaluate
            rubric: Fixed rubric criteria
            val_examples: Validation examples
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            Tuple of (best_preamble, mean_score)
        """
        if not candidates:
            return "", 0.0

        # Track scores for each candidate
        scores: Dict[int, List[float]] = {i: [] for i in range(len(candidates))}

        for round_num in range(self.config.eval_rounds):
            # Calculate UCB scores for each candidate
            ucb_scores = []
            total_evaluations = sum(len(s) for s in scores.values())

            for i in range(len(candidates)):
                if not scores[i]:
                    # Unexplored candidate gets infinity (explore first)
                    ucb = float("inf")
                else:
                    mean = sum(scores[i]) / len(scores[i])
                    n = len(scores[i])
                    # UCB1 formula: mean + c * sqrt(ln(total) / n)
                    if total_evaluations > 0:
                        exploration = self.config.exploration_weight * math.sqrt(
                            math.log(total_evaluations + 1) / n
                        )
                    else:
                        exploration = self.config.exploration_weight
                    ucb = mean + exploration
                ucb_scores.append(ucb)

            # Select candidate with highest UCB score
            selected_idx = ucb_scores.index(max(ucb_scores))

            # Sample examples for evaluation
            sample_size = min(self.config.eval_samples_per_round, len(val_examples))
            if sample_size == 0:
                continue
            sample = random.sample(val_examples, sample_size)

            # Evaluate selected candidate
            score = self._score_candidate(
                candidates[selected_idx], rubric, sample, accumulator
            )
            scores[selected_idx].append(score)

        # Return candidate with highest mean score
        mean_scores = [(i, sum(s) / len(s) if s else 0.0) for i, s in scores.items()]
        best_idx, best_score = max(mean_scores, key=lambda x: x[1])

        return candidates[best_idx], best_score

    def optimize(
        self,
        metric_id: str,
        current_rubric: List[str],
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        disagreements: DisagreementAnalysis,
        dataset_items: Optional[List[DatasetItem]] = None,
        current_preamble: str = "",
        accumulator: Optional[TokenAccumulator] = None,
    ) -> PromptOptimizationResult:
        """
        Run APE optimization.

        The optimization process:
        1. PROPOSE: Generate N candidate prompts from current prompt + disagreements
        2. SCORE: Evaluate candidates on validation set using UCB sampling
        3. SELECT: Return best-scoring prompt

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
        trainset, valset = self._build_dataset_from_annotations(
            metric_results, annotations, dataset_items
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="Not enough data for APE optimization (need at least 5 annotated examples)",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Initialize with seed prompt if provided, otherwise use default
        if current_preamble:
            seed_preamble = current_preamble.strip()
        else:
            seed_preamble = f"You are an expert evaluator for the metric: {metric_id}. Carefully analyze the output quality and provide an honest assessment."

        # Step 1: PROPOSE - Generate candidates based on disagreements
        candidates = self._propose_candidates(
            seed_preamble, current_rubric, disagreements, accumulator
        )

        if not candidates:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="APE failed to generate candidate prompts",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Include original as a candidate for fair comparison
        candidates.insert(0, seed_preamble)

        # Step 2 & 3: SCORE + SELECT using UCB
        best_preamble, best_score = self._ucb_select(
            candidates, current_rubric, valset, accumulator
        )

        # Evaluate seed on same examples for comparison
        seed_score = self._score_candidate(
            seed_preamble, current_rubric, valset, accumulator
        )

        # Determine improvement
        improvement_delta = best_score - seed_score
        if improvement_delta > 0.05:
            estimated_improvement = "high"
        elif improvement_delta > 0.02:
            estimated_improvement = "medium"
        elif improvement_delta > 0:
            estimated_improvement = "low"
        else:
            estimated_improvement = "none"

        # Build full optimized prompt
        full_optimized_prompt = build_full_prompt(best_preamble, current_rubric)

        # Build reasoning
        reasoning = (
            f"APE optimization completed. "
            f"Generated {len(candidates) - 1} candidates. "
            f"UCB rounds: {self.config.eval_rounds}. "
            f"Best F1: {best_score:.3f} (seed: {seed_score:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return PromptOptimizationResult(
            original_rubric=current_rubric,
            improved_rubric=current_rubric,  # Rubric stays the same
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=current_preamble,
            optimized_preamble=best_preamble,
            full_prompt=full_optimized_prompt,
        )


__all__ = ["APEConfig", "APEOptimizer"]
