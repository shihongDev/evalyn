"""EvoPrompt: Evolutionary Prompt Optimization.

Adapts evolutionary algorithms to prompt optimization for evalyn calibration.
Each generation: tournament selection -> crossover -> mutation -> score -> elitist selection.
Early stopping when best F1 is unchanged for N generations.

Reference: https://arxiv.org/abs/2309.08532

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

from tqdm import tqdm

from ..defaults import DEFAULT_EVAL_MODEL
from ..utils.api_client import GeminiClient
from .base_optimizer import BaseOptimizer
from .models import (
    DisagreementAnalysis,
    PromptOptimizationResult,
    TokenAccumulator,
)


@dataclass
class EvoPromptConfig:
    """Configuration for EvoPrompt optimizer."""

    population_size: int = 8
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 2
    early_stop_patience: int = 2
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL


# -- Prompt templates --

INIT_POPULATION_TEMPLATE = """You are an expert at writing LLM evaluation prompts.

Current judge preamble:
{current_preamble}

FIXED RUBRIC (for context - do not modify):
{rubric_text}

Disagreement patterns between the judge and human annotators:
{disagreement_summary}

Generate {count} diverse improved preambles that address these disagreements.
Each preamble should:
1. Be a complete standalone system prompt (100-300 words)
2. NOT include the rubric (it will be appended separately)
3. Take a different approach to fixing the failure patterns

Return ONLY a JSON array of preamble strings:
["preamble 1", "preamble 2", ...]"""

CROSSOVER_TEMPLATE = """You are an expert at writing LLM evaluation prompts.

Combine the strengths of these two evaluation preambles into one improved version.

Parent A:
{parent1}

Parent B:
{parent2}

Create a single preamble that merges the best aspects of both parents.
The result should be 100-300 words and be a complete standalone preamble.
Do NOT include the rubric.

Return ONLY the new preamble text, nothing else."""

MUTATE_TEMPLATE = """You are an expert at improving LLM evaluation prompts.

Current preamble:
{preamble}

This preamble made errors on these examples:
{failures}

Improve the preamble to handle these cases better.
Keep the overall structure but refine the instructions.
The result should be 100-300 words and be a complete standalone preamble.
Do NOT include the rubric.

Return ONLY the improved preamble text, nothing else."""


class EvoPromptOptimizer(BaseOptimizer):
    """
    Evolutionary prompt optimizer using tournament selection, crossover, and mutation.

    Algorithm per generation:
    1. Tournament selection to pick parents
    2. Probabilistic crossover to produce offspring
    3. Probabilistic mutation on offspring
    4. Score offspring on training examples
    5. Merge parents + offspring, keep top-K (elitist selection)

    IMPORTANT: Only the preamble is optimized. The rubric stays fixed.
    """

    def __init__(
        self,
        config: EvoPromptConfig | None = None,
        api_key: str | None = None,
    ):
        cfg = config or EvoPromptConfig()
        super().__init__(config=cfg, api_key=api_key)
        self._task_client_instance: GeminiClient | None = None
        self._scorer_client_instance: GeminiClient | None = None

    @property
    def _task_client(self) -> GeminiClient:
        """Lazy-initialized LLM client for population init, crossover, mutation."""
        if self._task_client_instance is None:
            self._task_client_instance = GeminiClient(
                model=self.config.task_model,
                temperature=0.8,
                api_key=self._api_key,
            )
        return self._task_client_instance

    @property
    def scorer_client(self) -> GeminiClient:
        """Lazy-initialized scorer client for failure collection."""
        if self._scorer_client_instance is None:
            self._scorer_client_instance = GeminiClient(
                model=self.config.scorer_model,
                temperature=0.0,
                api_key=self._api_key,
            )
        return self._scorer_client_instance

    # -- Evolutionary operators --

    def _init_population(
        self,
        current_preamble: str,
        disagreements: DisagreementAnalysis | None,
        rubric: list[str],
        accumulator: TokenAccumulator | None = None,
    ) -> list[str]:
        """Generate initial population of preambles.

        Returns population_size preambles: the current one plus LLM-generated variants.
        """
        count = self.config.population_size - 1
        if count <= 0:
            return [current_preamble]

        rubric_text = (
            "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"
        )

        disagreement_summary = "(no disagreement data available)"
        if disagreements and disagreements.total_disagreements > 0:
            parts = []
            if disagreements.false_positives:
                parts.append(
                    f"False positives ({len(disagreements.false_positives)} cases - judge too lenient):"
                )
                for case in disagreements.false_positives[:3]:
                    parts.append(f"  - Output: {case.output_text[:150]}...")
            if disagreements.false_negatives:
                parts.append(
                    f"False negatives ({len(disagreements.false_negatives)} cases - judge too strict):"
                )
                for case in disagreements.false_negatives[:3]:
                    parts.append(f"  - Output: {case.output_text[:150]}...")
            disagreement_summary = "\n".join(parts)

        prompt = INIT_POPULATION_TEMPLATE.format(
            current_preamble=current_preamble,
            rubric_text=rubric_text,
            disagreement_summary=disagreement_summary,
            count=count,
        )

        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            from .utils import parse_candidates_response

            variants = parse_candidates_response(result.text)
        except Exception:
            logger.warning("Failed to generate preamble variants, using seed only", exc_info=True)
            variants = []

        # Always include current preamble; pad with it if generation fell short
        population = [current_preamble] + variants
        while len(population) < self.config.population_size:
            population.append(current_preamble)
        return population[: self.config.population_size]

    def _tournament_select(
        self,
        scored_population: list[tuple[str, float]],
        tournament_size: int,
    ) -> str:
        """Select one parent via tournament selection.

        Args:
            scored_population: list of (preamble, f1_score) tuples
            tournament_size: number of contestants per tournament

        Returns:
            The winning preamble string.
        """
        k = min(tournament_size, len(scored_population))
        contestants = random.sample(scored_population, k)
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]

    def _crossover(
        self,
        parent1: str,
        parent2: str,
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Combine two parents into one offspring via LLM."""
        prompt = CROSSOVER_TEMPLATE.format(parent1=parent1, parent2=parent2)
        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            text = result.text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                text = text.strip()
            return text if text else parent1
        except Exception:
            logger.warning("Crossover failed, returning parent1 unchanged", exc_info=True)
            return parent1

    def _mutate(
        self,
        preamble: str,
        failures: list[dict],
        accumulator: TokenAccumulator | None = None,
    ) -> str:
        """Mutate a preamble using failure examples as guidance."""
        if not failures:
            failure_text = "(no specific failure examples available)"
        else:
            parts = []
            for i, ex in enumerate(failures[:3], 1):
                parts.append(
                    f"Example {i}: input={ex.get('input', '')[:100]}... "
                    f"output={ex.get('output', '')[:100]}... "
                    f"expected={ex.get('expected', '?')}"
                )
            failure_text = "\n".join(parts)

        prompt = MUTATE_TEMPLATE.format(preamble=preamble, failures=failure_text)
        try:
            result = self._task_client.generate_with_usage(prompt)
            if accumulator:
                accumulator.add(result)
            text = result.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                text = text.strip()
            return text if text else preamble
        except Exception:
            logger.warning("Mutation failed, using original preamble", exc_info=True)
            return preamble

    # _collect_failures is inherited from BaseOptimizer

    # -- Main optimization loop --

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: list[str],
        current_preamble: str,
        metric_results: list,
        annotations: list,
        disagreements: Any = None,
        dataset_items: list | None = None,
        accumulator: TokenAccumulator | None = None,
        **kwargs,
    ) -> PromptOptimizationResult:
        """Run evolutionary prompt optimization.

        Steps:
        1. Split data into train/val
        2. Initialize population (current preamble + LLM variants)
        3. For each generation:
           a. Score population on training set
           b. Create offspring via tournament -> crossover -> mutation
           c. Score offspring
           d. Elitist selection: keep top-K from parents + offspring
           e. Early stop if best F1 unchanged for patience generations
        4. Validate best on val set and return result
        """
        # Split data
        trainset, valset = self.split_train_val(
            metric_results, annotations, dataset_items
        )

        if len(trainset) < 3 or len(valset) < 2:
            return self.build_result(
                original_preamble=current_preamble,
                optimized_preamble=current_preamble,
                rubric=current_rubric,
                reasoning="Not enough data for EvoPrompt optimization (need at least 5 annotated examples)",
                estimated_improvement="unknown",
            )

        seed = current_preamble.strip() if current_preamble else (
            f"You are an expert evaluator for the metric: {metric_id}. "
            "Carefully analyze the output quality and provide an honest assessment."
        )

        # Step 2: initialize population
        population = self._init_population(
            seed, disagreements, current_rubric, accumulator
        )

        # Score initial population
        scored: list[tuple[str, float]] = []
        for preamble in population:
            f1 = self.score_preamble(preamble, current_rubric, trainset, accumulator)
            scored.append((preamble, f1))

        best_f1 = max(s[1] for s in scored)
        no_improve_count = 0
        generations_run = 0

        # Collect actual failure cases for mutation guidance
        best_preamble = max(scored, key=lambda s: s[1])[0]
        failure_examples = self._collect_failures(
            best_preamble, current_rubric, trainset, accumulator=accumulator
        )

        pbar = tqdm(range(self.config.generations), desc="EvoPrompt", unit="gen")
        for gen in pbar:
            generations_run = gen + 1
            offspring: list[str] = []

            # Create offspring equal to population size
            for _ in range(self.config.population_size):
                parent1 = self._tournament_select(scored, self.config.tournament_size)
                parent2 = self._tournament_select(scored, self.config.tournament_size)

                # Crossover
                if random.random() < self.config.crossover_rate:
                    child = self._crossover(parent1, parent2, accumulator)
                else:
                    child = random.choice([parent1, parent2])

                # Mutation
                if random.random() < self.config.mutation_rate:
                    child = self._mutate(child, failure_examples, accumulator)

                offspring.append(child)

            # Score offspring
            scored_offspring: list[tuple[str, float]] = []
            for child in offspring:
                f1 = self.score_preamble(child, current_rubric, trainset, accumulator)
                scored_offspring.append((child, f1))

            # Elitist selection: merge parents + offspring, keep top-K
            merged = scored + scored_offspring
            merged.sort(key=lambda x: x[1], reverse=True)
            scored = merged[: self.config.population_size]

            gen_best = scored[0][1]
            pbar.set_postfix({"best_f1": f"{gen_best:.1%}"})

            # Early stopping
            if gen_best > best_f1:
                best_f1 = gen_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.early_stop_patience:
                break

        # Best preamble from final population
        best_preamble = scored[0][0]

        # Validate on held-out set
        val_f1 = self.score_preamble(best_preamble, current_rubric, valset, accumulator)
        seed_val_f1 = self.score_preamble(seed, current_rubric, valset, accumulator)

        improvement_delta = val_f1 - seed_val_f1
        est = self.classify_improvement(improvement_delta)

        reasoning = (
            f"EvoPrompt optimization completed. "
            f"Population: {self.config.population_size}, "
            f"generations run: {generations_run}/{self.config.generations}. "
            f"Train best F1: {best_f1:.3f}. "
            f"Val F1: {val_f1:.3f} (seed: {seed_val_f1:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return self.build_result(
            original_preamble=current_preamble,
            optimized_preamble=best_preamble,
            rubric=current_rubric,
            reasoning=reasoning,
            estimated_improvement=est,
        )


__all__ = ["EvoPromptConfig", "EvoPromptOptimizer"]
