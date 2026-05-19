"""PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution.

Co-evolves preambles alongside "mutation prompts" (meta-instructions for
how to improve preambles). Each generation: mutate preambles using their
mutation prompts, score, select top-K, evolve mutation prompts.

Key concepts:
- BreederUnit: A (preamble, mutation_prompt) pair that co-evolves
- Mutation prompts: Meta-strategies that guide how preambles are improved
- Self-referential: Mutation prompts themselves evolve based on performance

Reference: https://arxiv.org/abs/2309.16797

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
from ..models import Annotation, DatasetItem, MetricResult
from ..utils.api_client import GeminiClient
from .base_optimizer import BaseOptimizer
from .models import (
    PromptOptimizationResult,
    TokenAccumulator,
)
from .utils import (
    build_full_prompt,
    parse_candidates_response,
)


@dataclass
class BreederUnit:
    """A co-evolving (preamble, mutation_prompt) pair."""

    preamble: str
    mutation_prompt: str
    f1_score: float = 0.0


@dataclass
class PromptBreederConfig:
    """Configuration for PromptBreeder optimizer."""

    population_size: int = 6
    generations: int = 5
    num_initial_mutation_prompts: int = 4
    early_stop_patience: int = 2
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL
    train_split: float = 0.7
    temperature: float = 0.8
    timeout: int = 120


PREAMBLE_VARIANT_TEMPLATE = """You are an expert at creating LLM evaluation prompts.

Here is an existing judge preamble:
{seed_preamble}

Create a meaningfully different variant of this preamble that:
1. Preserves the core evaluation intent
2. Uses a different framing, tone, or emphasis
3. Is a complete, standalone preamble (100-300 words)
4. Does NOT include any rubric criteria (those are appended separately)

Return ONLY the new preamble text, no explanation."""


MUTATION_STRATEGIES_TEMPLATE = """You are an expert at prompt engineering.

Your task is to generate {count} diverse strategies for improving LLM judge preambles.
Each strategy should be a short instruction (1-2 sentences) describing a specific
approach to making a judge preamble more effective.

Examples of good strategies:
- "Add explicit examples of edge cases the judge should watch for."
- "Rewrite using more precise language and remove ambiguous terms."
- "Emphasize the most common failure mode and add guardrails against it."

Generate {count} DIFFERENT strategies. Return ONLY a JSON array of strategy strings:
["strategy 1", "strategy 2", ...]"""


MUTATION_TEMPLATE = """You are improving an LLM judge's preamble.

Strategy: {mutation_prompt}

Current preamble:
{preamble}

Here are challenging examples that require careful evaluation:
{failures}

Apply the strategy to produce an improved preamble.
- Keep it 100-300 words
- Do NOT include rubric criteria (those are appended separately)
- Return ONLY the improved preamble text, no explanation."""


EVOLVE_MUTATION_TEMPLATE = """This mutation strategy was used to improve an LLM judge preamble:
"{mutation_prompt}"

It produced a preamble scoring F1={current_f1:.3f}.
The previous best scored F1={prev_best_f1:.3f}.

How should this strategy be refined to produce even better preambles?
Return ONLY the improved strategy (1-2 sentences), no explanation."""


class PromptBreederOptimizer(BaseOptimizer):
    """
    PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution.

    Co-evolves preambles alongside mutation prompts. Each BreederUnit pairs
    a preamble with a mutation strategy. Evolution proceeds by:
    1. Mutating preambles using their paired mutation strategies
    2. Scoring mutated preambles on training data
    3. Selecting top-K units
    4. Evolving the mutation strategies themselves

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.
    """

    def __init__(
        self,
        config: PromptBreederConfig | None = None,
        api_key: str | None = None,
    ):
        super().__init__(config=config or PromptBreederConfig(), api_key=api_key)
        self._task_client_instance: GeminiClient | None = None

    @property
    def _task_client(self) -> GeminiClient:
        """Lazy-initialized task LLM client (for generating/mutating prompts)."""
        if self._task_client_instance is None:
            self._task_client_instance = GeminiClient(
                model=self.config.task_model,
                temperature=self.config.temperature,
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._task_client_instance

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _init_population(
        self,
        current_preamble: str,
        metric_id: str,
        accumulator: TokenAccumulator | None = None,
    ) -> list[BreederUnit]:
        """Initialize population of BreederUnit pairs.

        Generates preamble variants and mutation strategies, then pairs them.
        Unit[0] always uses the original preamble.
        """
        pop_size = self.config.population_size
        num_mut = self.config.num_initial_mutation_prompts

        # -- Generate preamble variants --
        preambles: list[str] = [current_preamble]
        for _ in range(pop_size - 1):
            prompt = PREAMBLE_VARIANT_TEMPLATE.format(seed_preamble=current_preamble)
            try:
                result = self._task_client.generate_with_usage(prompt)
                if accumulator:
                    accumulator.add(result)
                text = result.text.strip()
                if text:
                    preambles.append(text)
            except Exception:
                logger.warning("Failed to generate preamble variant, using seed", exc_info=True)
                preambles.append(current_preamble)

        # Pad if we got fewer than pop_size
        while len(preambles) < pop_size:
            preambles.append(current_preamble)

        # -- Generate mutation strategies --
        mutation_prompts: list[str] = []
        strategies_prompt = MUTATION_STRATEGIES_TEMPLATE.format(count=num_mut)
        try:
            result = self._task_client.generate_with_usage(strategies_prompt)
            if accumulator:
                accumulator.add(result)
            mutation_prompts = parse_candidates_response(result.text)
        except Exception:
            logger.warning("Failed to generate mutation strategies, using defaults", exc_info=True)

        # Fallback defaults
        defaults = [
            "Add explicit edge-case handling and boundary conditions.",
            "Use more precise and unambiguous language throughout.",
            "Emphasize the most critical evaluation criteria first.",
            "Add concrete positive and negative examples to guide judgment.",
        ]
        while len(mutation_prompts) < num_mut:
            mutation_prompts.append(defaults[len(mutation_prompts) % len(defaults)])

        # -- Pair them --
        units: list[BreederUnit] = []
        for i in range(pop_size):
            units.append(
                BreederUnit(
                    preamble=preambles[i],
                    mutation_prompt=mutation_prompts[i % len(mutation_prompts)],
                )
            )

        return units

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def _format_failures(
        self, examples: list[dict[str, Any]], max_show: int = 3
    ) -> str:
        """Format failure examples for inclusion in prompts."""
        if not examples:
            return "(none)"
        lines = []
        for i, ex in enumerate(examples[:max_show], 1):
            lines.append(
                f"Example {i}: input={ex.get('input', '')[:200]}... "
                f"output={ex.get('output', '')[:200]}... "
                f"expected={ex.get('expected', '?')}"
            )
        return "\n".join(lines)

    def _apply_mutation(
        self,
        unit: BreederUnit,
        failures: str,
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Mutate a preamble using its paired mutation strategy.

        Returns the new preamble string (or the original on failure).
        """
        prompt = MUTATION_TEMPLATE.format(
            mutation_prompt=unit.mutation_prompt,
            preamble=unit.preamble,
            failures=failures,
        )
        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            text = result.text.strip()
            return text if text else unit.preamble
        except Exception:
            logger.warning("Failed to apply mutation, using original preamble", exc_info=True)
            return unit.preamble

    def _evolve_mutation_prompt(
        self,
        unit: BreederUnit,
        prev_best_f1: float,
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Evolve a mutation strategy based on how well it performed.

        Returns the refined mutation prompt string (or the original on failure).
        """
        prompt = EVOLVE_MUTATION_TEMPLATE.format(
            mutation_prompt=unit.mutation_prompt,
            current_f1=unit.f1_score,
            prev_best_f1=prev_best_f1,
        )
        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            text = result.text.strip()
            return text if text else unit.mutation_prompt
        except Exception:
            logger.warning("Failed to evolve mutation strategy, keeping current", exc_info=True)
            return unit.mutation_prompt

    # ------------------------------------------------------------------
    # Main optimize loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: list[str],
        metric_results: list[MetricResult],
        annotations: list[Annotation],
        dataset_items: list[DatasetItem] | None = None,
        current_preamble: str = "",
        accumulator: TokenAccumulator | None = None,
        **kwargs: Any,
    ) -> PromptOptimizationResult:
        """
        Run PromptBreeder optimization.

        Algorithm:
        1. Split data into train/val
        2. Initialize population of BreederUnit pairs
        3. Score initial population
        4. For each generation:
           a. Mutate all preambles using their mutation prompts
           b. Score all mutated preambles
           c. Select top-K units
           d. Evolve mutation prompts
           e. Check early stopping
        5. Validate best on val set and return result
        """
        # -- Data preparation --
        trainset, valset = self.split_train_val(
            metric_results,
            annotations,
            dataset_items,
            train_ratio=self.config.train_split,
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=(
                    "Not enough data for PromptBreeder optimization "
                    "(need at least 5 annotated examples)"
                ),
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        seed_preamble = current_preamble.strip() if current_preamble else (
            f"You are an expert evaluator for the metric: {metric_id}. "
            "Carefully analyze the output quality and provide an honest assessment."
        )

        # -- Initialize population --
        population = self._init_population(seed_preamble, metric_id, accumulator)

        # -- Score initial population --
        for unit in population:
            unit.f1_score = self.score_preamble(
                unit.preamble, current_rubric, trainset, accumulator
            )

        best_f1 = max(u.f1_score for u in population)
        seed_train_f1 = population[0].f1_score  # population[0] is always the seed
        no_improve_count = 0

        # -- Generational loop --
        gen = 0
        pbar = tqdm(
            range(1, self.config.generations + 1),
            desc="PromptBreeder",
            unit="gen",
        )
        for _gen in pbar:
            prev_best_f1 = best_f1

            # Collect failure examples for mutation context
            failures = self._format_failures(
                [ex for ex in trainset if ex.get("expected") == "FAIL"]
            )

            # a. Mutate all preambles
            for unit in population:
                new_preamble = self._apply_mutation(unit, failures, accumulator)
                unit.preamble = new_preamble

            # b. Score all mutated preambles
            for unit in population:
                unit.f1_score = self.score_preamble(
                    unit.preamble, current_rubric, trainset, accumulator
                )

            # c. Select top-K (keep population_size units, sorted by score)
            population.sort(key=lambda u: u.f1_score, reverse=True)
            population = population[: self.config.population_size]

            # d. Evolve mutation prompts
            for unit in population:
                unit.mutation_prompt = self._evolve_mutation_prompt(
                    unit, prev_best_f1, accumulator
                )

            # e. Check early stopping
            gen_best_f1 = population[0].f1_score
            if gen_best_f1 > best_f1:
                best_f1 = gen_best_f1
                no_improve_count = 0
            else:
                no_improve_count += 1

            pbar.set_postfix({"best_f1": f"{best_f1:.1%}"})

            if no_improve_count >= self.config.early_stop_patience:
                break

        # -- Validate best on val set --
        best_unit = population[0]
        val_f1 = self.score_preamble(
            best_unit.preamble, current_rubric, valset, accumulator
        )
        seed_val_f1 = self.score_preamble(
            seed_preamble, current_rubric, valset, accumulator
        )

        # -- Build result --
        improvement_delta = val_f1 - seed_val_f1
        estimated_improvement = self.classify_improvement(improvement_delta)

        full_optimized_prompt = build_full_prompt(best_unit.preamble, current_rubric)

        reasoning = (
            f"PromptBreeder optimization completed. "
            f"Generations: {gen}/{self.config.generations}. "
            f"Population: {self.config.population_size}. "
            f"Train F1: {best_unit.f1_score:.3f} (seed: {seed_train_f1:.3f}). "
            f"Val F1: {val_f1:.3f} (seed: {seed_val_f1:.3f}). "
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
            optimized_preamble=best_unit.preamble,
            full_prompt=full_optimized_prompt,
        )


__all__ = ["BreederUnit", "PromptBreederConfig", "PromptBreederOptimizer"]
