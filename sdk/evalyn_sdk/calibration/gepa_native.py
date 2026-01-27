"""GEPA Native: Native implementation of GEPA with token tracking.

GEPA (Generalized Evolutionary Prompt Algorithm) uses Pareto-based selection
and reflection-guided mutation for prompt optimization.

Key concepts:
- Pareto frontier: Maintain multiple candidates, each best on different subsets
- Reflection: Analyze failures with an LLM to guide targeted improvements
- Mini-batch evaluation: Efficient budget use via batched scoring

Reference: https://arxiv.org/abs/2507.19457

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..defaults import DEFAULT_GENERATOR_MODEL
from ..models import Annotation, DatasetItem, MetricResult
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


@dataclass
class ParetoCandidate:
    """A candidate in the Pareto frontier."""

    preamble: str
    scores: Dict[str, float] = field(default_factory=dict)  # example_id -> score
    total_f1: float = 0.0
    generation: int = 0

    def get_score(self, example_id: str) -> float:
        return self.scores.get(example_id, 0.0)


@dataclass
class GEPANativeConfig:
    """Configuration for native GEPA optimizer."""

    task_model: str = DEFAULT_GENERATOR_MODEL  # Model for evaluation (judge)
    reflection_model: str = DEFAULT_GENERATOR_MODEL  # Model for generating mutations
    max_metric_calls: int = 150  # Budget for optimization
    num_initial_candidates: int = 5  # Initial population size
    mini_batch_size: int = 5  # Samples per feedback cycle
    pareto_set_size: int = 10  # Max Pareto set samples for validation
    exploit_prob: float = 0.9  # Probability of selecting best vs random
    train_split: float = 0.7  # Train/val split ratio
    temperature: float = 0.7  # Temperature for candidate generation
    timeout: int = 120  # API timeout in seconds


REFLECTION_TEMPLATE = """You are an expert at improving LLM evaluation prompts.

## CURRENT PROMPT
{current_preamble}

## FIXED RUBRIC (for context - do not modify)
{rubric_text}

## FAILURES TO ADDRESS
The current prompt made incorrect judgments on these examples:

{failure_examples}

## YOUR TASK
Analyze why the current prompt failed on these examples, then propose ONE improved version.

Guidelines:
- Fix the specific failure patterns shown above
- Keep the core evaluation intent
- Be concise but complete (100-300 words)
- Do NOT include the rubric (it will be appended separately)

Return ONLY the improved preamble text, no explanation or formatting."""


INITIAL_GENERATION_TEMPLATE = """You are an expert at creating LLM evaluation prompts.

## TASK
Create a judge prompt for evaluating outputs on this metric: {metric_id}

## SEED PROMPT
{seed_preamble}

## RUBRIC (for context)
{rubric_text}

## YOUR TASK
Generate a DIFFERENT but effective preamble that evaluates the same criteria.
Create a unique variation that might catch different failure modes.

Guidelines:
- Clear role definition
- Specific instructions on how to evaluate
- Guidance on edge cases
- 100-300 words

Return ONLY the preamble text, no explanation or formatting."""


class GEPANativeOptimizer:
    """
    Native GEPA implementation with full token tracking.

    Uses Pareto-based selection and reflection-guided mutation
    to optimize judge prompts without external dependencies.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.
    """

    def __init__(
        self,
        config: Optional[GEPANativeConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize GEPA-Native optimizer.

        Args:
            config: GEPA configuration (uses defaults if not provided)
            api_key: Optional API key for Gemini (default: from env)
        """
        self.config = config or GEPANativeConfig()
        self._api_key = api_key
        self._task_client: Optional[GeminiClient] = None
        self._reflection_client: Optional[GeminiClient] = None

    @property
    def task_client(self) -> GeminiClient:
        """Lazy-initialized task LLM client (for evaluation)."""
        if self._task_client is None:
            self._task_client = GeminiClient(
                model=self.config.task_model,
                temperature=0.0,  # Deterministic for scoring
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._task_client

    @property
    def reflection_client(self) -> GeminiClient:
        """Lazy-initialized reflection LLM client (for mutations)."""
        if self._reflection_client is None:
            self._reflection_client = GeminiClient(
                model=self.config.reflection_model,
                temperature=self.config.temperature,
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._reflection_client

    def _evaluate_candidate_on_examples(
        self,
        candidate: ParetoCandidate,
        rubric: List[str],
        examples: List[Dict[str, Any]],
        accumulator: Optional[TokenAccumulator] = None,
    ) -> Tuple[AlignmentMetrics, List[Dict[str, Any]]]:
        """
        Evaluate a candidate on examples, tracking per-example scores.

        Returns:
            Tuple of (metrics, failures) where failures are examples the judge got wrong.
        """
        full_prompt = build_full_prompt(candidate.preamble, rubric)
        metrics = AlignmentMetrics()
        failures: List[Dict[str, Any]] = []

        for ex in examples:
            eval_prompt = f"""{full_prompt}

## Input to evaluate
{ex.get("input", "")[:1000]}

## Output to evaluate
{ex.get("output", "")[:1000]}

Provide your verdict:"""

            try:
                result = self.task_client.generate_with_usage(eval_prompt)
                if accumulator:
                    accumulator.add(result)

                judge_pass = parse_judge_response(result.text)
                human_pass = ex.get("expected") == "PASS"
                metrics.record(judge_pass, human_pass)

                # Track per-example score (1 if correct, 0 if wrong)
                correct = judge_pass == human_pass
                candidate.scores[ex["id"]] = 1.0 if correct else 0.0

                # Collect failures for reflection
                if not correct:
                    failures.append(
                        {
                            "id": ex["id"],
                            "input": ex.get("input", "")[:300],
                            "output": ex.get("output", "")[:300],
                            "expected": ex.get("expected"),
                            "judge_said": "PASS" if judge_pass else "FAIL",
                            "reason": result.text[:200] if result.text else "",
                        }
                    )
            except Exception:
                candidate.scores[ex["id"]] = 0.0

        candidate.total_f1 = metrics.f1
        return metrics, failures

    def _generate_initial_candidates(
        self,
        seed_preamble: str,
        metric_id: str,
        rubric: List[str],
        accumulator: Optional[TokenAccumulator] = None,
    ) -> List[ParetoCandidate]:
        """Generate initial diverse candidates from seed.

        Args:
            seed_preamble: Starting preamble to generate variations from
            metric_id: The metric being optimized
            rubric: Fixed rubric criteria (for context)
            accumulator: Optional TokenAccumulator to track token usage
        """
        rubric_text = (
            "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"
        )

        candidates = [ParetoCandidate(preamble=seed_preamble, generation=0)]

        # Generate N-1 variations
        for _ in range(self.config.num_initial_candidates - 1):
            prompt = INITIAL_GENERATION_TEMPLATE.format(
                metric_id=metric_id,
                seed_preamble=seed_preamble,
                rubric_text=rubric_text,
            )

            try:
                result = self.reflection_client.generate_with_usage(prompt)
                if accumulator:
                    accumulator.add(result)

                new_preamble = result.text.strip()
                if new_preamble and len(new_preamble) > 20:
                    candidates.append(
                        ParetoCandidate(preamble=new_preamble, generation=0)
                    )
            except Exception:
                pass

        return candidates

    def _reflect_and_mutate(
        self,
        candidate: ParetoCandidate,
        rubric: List[str],
        failures: List[Dict[str, Any]],
        generation: int,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> Optional[ParetoCandidate]:
        """Generate a mutation by reflecting on failures.

        Args:
            candidate: The candidate to mutate
            rubric: Fixed rubric criteria (for context)
            failures: List of failure cases to address
            generation: Current generation number
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            New mutated candidate, or None if mutation failed
        """
        if not failures:
            return None

        rubric_text = (
            "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"
        )

        # Format failure examples
        failure_text = ""
        for i, fail in enumerate(failures[:5], 1):
            failure_text += f"""
Example {i}:
  Input: {fail["input"][:200]}...
  Output: {fail["output"][:200]}...
  Human said: {fail["expected"]}
  Judge said: {fail["judge_said"]}
  Judge reasoning: {fail["reason"][:150]}...
"""

        prompt = REFLECTION_TEMPLATE.format(
            current_preamble=candidate.preamble,
            rubric_text=rubric_text,
            failure_examples=failure_text,
        )

        try:
            result = self.reflection_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)

            new_preamble = result.text.strip()
            if new_preamble and len(new_preamble) > 20:
                # Remove markdown if present
                if new_preamble.startswith("```"):
                    lines = new_preamble.split("\n")
                    new_preamble = "\n".join(
                        lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                    ).strip()

                return ParetoCandidate(preamble=new_preamble, generation=generation)
        except Exception:
            pass

        return None

    def _select_pareto_candidate(
        self,
        frontier: List[ParetoCandidate],
        pareto_examples: List[Dict[str, Any]],
    ) -> ParetoCandidate:
        """
        Select a candidate using Pareto-weighted probability.

        Candidates that achieve "best" status on more examples get higher weight.
        """
        if len(frontier) == 1:
            return frontier[0]

        # Calculate per-example best scores
        example_ids = [ex["id"] for ex in pareto_examples]
        best_scores: Dict[str, float] = {}
        for ex_id in example_ids:
            best_scores[ex_id] = max(c.get_score(ex_id) for c in frontier)

        # Count how often each candidate achieves best score
        weights: List[float] = []
        for candidate in frontier:
            count = sum(
                1
                for ex_id in example_ids
                if candidate.get_score(ex_id) >= best_scores[ex_id] - 0.01
            )
            weights.append(max(count, 0.1))  # Minimum weight to avoid zero

        # Normalize
        total = sum(weights)
        probs = [w / total for w in weights]

        # Select with exploitation bias
        if random.random() < self.config.exploit_prob:
            # Exploit: select proportional to weight
            return random.choices(frontier, weights=probs, k=1)[0]
        else:
            # Explore: uniform random
            return random.choice(frontier)

    def _is_dominated(
        self,
        candidate: ParetoCandidate,
        others: List[ParetoCandidate],
        example_ids: List[str],
    ) -> bool:
        """Check if candidate is dominated by any other in the frontier."""
        for other in others:
            if other is candidate:
                continue

            # Check if 'other' dominates 'candidate'
            all_geq = True
            some_greater = False
            for ex_id in example_ids:
                c_score = candidate.get_score(ex_id)
                o_score = other.get_score(ex_id)
                if o_score < c_score:
                    all_geq = False
                    break
                if o_score > c_score:
                    some_greater = True

            if all_geq and some_greater:
                return True

        return False

    def _update_pareto_frontier(
        self,
        frontier: List[ParetoCandidate],
        new_candidate: ParetoCandidate,
        pareto_examples: List[Dict[str, Any]],
        max_size: int = 10,
    ) -> List[ParetoCandidate]:
        """Add candidate to frontier if non-dominated, prune dominated ones."""
        example_ids = [ex["id"] for ex in pareto_examples]

        # Check if new candidate is dominated
        if self._is_dominated(new_candidate, frontier, example_ids):
            return frontier

        # Add new candidate and remove any it dominates
        updated = [new_candidate]
        for existing in frontier:
            if not self._is_dominated(existing, [new_candidate], example_ids):
                updated.append(existing)

        # Prune to max size by F1 score
        if len(updated) > max_size:
            updated.sort(key=lambda c: c.total_f1, reverse=True)
            updated = updated[:max_size]

        return updated

    def optimize(
        self,
        metric_id: str,
        current_rubric: List[str],
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
        current_preamble: str = "",
        accumulator: Optional[TokenAccumulator] = None,
    ) -> PromptOptimizationResult:
        """
        Run GEPA optimization loop.

        The optimization process:
        1. Generate initial diverse candidates
        2. Evaluate all on Pareto set (subset of examples)
        3. Iterative loop:
           a. Select candidate via Pareto dominance weighting
           b. Evaluate on mini-batch, collect failures
           c. Reflect on failures to generate mutation
           d. Validate mutation on Pareto set
           e. Add to frontier if non-dominated
        4. Return best prompt by overall F1

        Args:
            metric_id: The metric being calibrated
            current_rubric: Fixed rubric criteria (NOT optimized)
            metric_results: Evaluation results from the judge
            annotations: Human annotations
            dataset_items: Optional dataset items for context
            current_preamble: Current preamble text (the part to optimize)
            accumulator: Optional TokenAccumulator to track token usage

        Returns:
            PromptOptimizationResult with optimized preamble
        """
        if accumulator is None:
            accumulator = TokenAccumulator()

        # Build datasets
        trainset, valset = build_dataset_from_annotations(
            metric_results, annotations, dataset_items, self.config.train_split
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="Not enough data for GEPA optimization (need at least 5 annotated examples)",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Initialize seed preamble
        if current_preamble:
            seed_preamble = current_preamble.strip()
        else:
            seed_preamble = f"You are an expert evaluator for the metric: {metric_id}. Carefully analyze the output quality and provide an honest assessment."

        # Select Pareto set (subset of training for efficient validation)
        pareto_size = min(self.config.pareto_set_size, len(trainset))
        pareto_examples = random.sample(trainset, pareto_size)

        # Step 1: Generate initial candidates
        frontier = self._generate_initial_candidates(
            seed_preamble, metric_id, current_rubric, accumulator
        )

        # Track budget (approximate: each eval is one call)
        budget = self.config.max_metric_calls
        budget -= len(frontier) - 1  # Initial generation calls

        # Step 2: Evaluate all initial candidates on Pareto set
        for candidate in frontier:
            if budget <= 0:
                break
            metrics, _ = self._evaluate_candidate_on_examples(
                candidate, current_rubric, pareto_examples, accumulator
            )
            budget -= len(pareto_examples)

        # Track best overall
        best_candidate = max(frontier, key=lambda c: c.total_f1)
        generation = 0

        # Step 3: Main optimization loop
        # Reserve budget for one full iteration (mini-batch eval + pareto validation)
        initial_budget = budget
        threshold = self.config.mini_batch_size + self.config.pareto_set_size
        with tqdm(total=initial_budget - threshold, desc="GEPA", unit="call") as pbar:
            while budget > threshold:
                generation += 1
                budget_before = budget

                # Select candidate via Pareto weighting
                selected = self._select_pareto_candidate(frontier, pareto_examples)

                # Evaluate on mini-batch from remaining training data
                remaining = [ex for ex in trainset if ex not in pareto_examples]
                if len(remaining) < self.config.mini_batch_size:
                    remaining = trainset

                mini_batch = random.sample(
                    remaining, min(self.config.mini_batch_size, len(remaining))
                )

                _, failures = self._evaluate_candidate_on_examples(
                    selected, current_rubric, mini_batch, accumulator
                )
                budget -= len(mini_batch)

                # Reflect on failures to generate mutation (if any failures)
                mutation = None
                if failures:
                    mutation = self._reflect_and_mutate(
                        selected, current_rubric, failures, generation, accumulator
                    )
                    budget -= 1  # Reflection call

                # Validate mutation on Pareto set and update frontier
                if mutation is not None:
                    metrics, _ = self._evaluate_candidate_on_examples(
                        mutation, current_rubric, pareto_examples, accumulator
                    )
                    budget -= len(pareto_examples)

                    frontier = self._update_pareto_frontier(
                        frontier, mutation, pareto_examples, max_size=10
                    )

                    current_best = max(frontier, key=lambda c: c.total_f1)
                    if current_best.total_f1 > best_candidate.total_f1:
                        best_candidate = current_best

                pbar.update(budget_before - budget)
                pbar.set_postfix({"gen": generation, "frontier": len(frontier)})

        # Step 4: Final validation on full validation set
        val_metrics, _ = self._evaluate_candidate_on_examples(
            best_candidate, current_rubric, valset, accumulator
        )

        # Evaluate seed for comparison
        seed_candidate = ParetoCandidate(preamble=seed_preamble)
        seed_metrics, _ = self._evaluate_candidate_on_examples(
            seed_candidate, current_rubric, valset, accumulator
        )

        # Build result
        improvement_delta = val_metrics.f1 - seed_metrics.f1
        if improvement_delta > 0.05:
            estimated_improvement = "high"
        elif improvement_delta > 0.02:
            estimated_improvement = "medium"
        elif improvement_delta > 0:
            estimated_improvement = "low"
        else:
            estimated_improvement = "none"

        full_optimized_prompt = build_full_prompt(
            best_candidate.preamble, current_rubric
        )

        reasoning = (
            f"GEPA-Native optimization completed. "
            f"Generations: {generation}. "
            f"Frontier size: {len(frontier)}. "
            f"Val F1: {val_metrics.f1:.3f} (seed: {seed_metrics.f1:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return PromptOptimizationResult(
            original_rubric=current_rubric,
            improved_rubric=current_rubric,  # Rubric stays fixed
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=current_preamble,
            optimized_preamble=best_candidate.preamble,
            full_prompt=full_optimized_prompt,
        )


__all__ = ["GEPANativeConfig", "GEPANativeOptimizer", "ParetoCandidate"]
