"""
Confidence-Aware Prompt Optimization (CAPO) algorithm.

Pure Python evolutionary optimizer for prompt engineering. No external
dependencies, no LLM calls - this is the optimization framework that
manages population, selection, mutation, and convergence.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PromptCandidate:
    """A single candidate prompt in the optimization population."""

    prompt_text: str
    score: float = 0.0
    confidence: float = 0.0
    generation: int = 0
    mutations_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "score": self.score,
            "confidence": self.confidence,
            "generation": self.generation,
            "mutations_applied": list(self.mutations_applied),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptCandidate:
        return cls(
            prompt_text=data["prompt_text"],
            score=data.get("score", 0.0),
            confidence=data.get("confidence", 0.0),
            generation=data.get("generation", 0),
            mutations_applied=list(data.get("mutations_applied", [])),
        )


@dataclass
class CAPOConfig:
    """Configuration for the CAPO optimization run."""

    population_size: int = 10
    max_generations: int = 20
    confidence_threshold: float = 0.8
    mutation_rate: float = 0.3
    elite_count: int = 2
    tournament_size: int = 3
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "confidence_threshold": self.confidence_threshold,
            "mutation_rate": self.mutation_rate,
            "elite_count": self.elite_count,
            "tournament_size": self.tournament_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CAPOConfig:
        return cls(
            population_size=data.get("population_size", 10),
            max_generations=data.get("max_generations", 20),
            confidence_threshold=data.get("confidence_threshold", 0.8),
            mutation_rate=data.get("mutation_rate", 0.3),
            elite_count=data.get("elite_count", 2),
            tournament_size=data.get("tournament_size", 3),
            seed=data.get("seed"),
        )


@dataclass
class CAPOState:
    """Mutable state of an optimization run."""

    generation: int
    population: list[PromptCandidate]
    best_candidate: PromptCandidate | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "population": [c.as_dict() for c in self.population],
            "best_candidate": self.best_candidate.as_dict() if self.best_candidate else None,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CAPOState:
        population = [PromptCandidate.from_dict(c) for c in data.get("population", [])]
        best_raw = data.get("best_candidate")
        best = PromptCandidate.from_dict(best_raw) if best_raw else None
        return cls(
            generation=data.get("generation", 0),
            population=population,
            best_candidate=best,
            history=list(data.get("history", [])),
        )


@dataclass
class MutationOperator:
    """Describes a single mutation strategy."""

    name: str
    description: str
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MutationOperator:
        return cls(
            name=data["name"],
            description=data["description"],
            weight=data.get("weight", 1.0),
        )


# ---------------------------------------------------------------------------
# Built-in Mutation Operators
# ---------------------------------------------------------------------------

MUTATION_OPERATORS: list[MutationOperator] = [
    MutationOperator(
        name="rephrase",
        description="Rephrase the prompt while preserving meaning",
        weight=1.0,
    ),
    MutationOperator(
        name="elaborate",
        description="Add detail and specificity to the prompt",
        weight=1.0,
    ),
    MutationOperator(
        name="simplify",
        description="Remove redundancy and simplify language",
        weight=1.0,
    ),
    MutationOperator(
        name="add_example",
        description="Append an illustrative example to the prompt",
        weight=0.8,
    ),
    MutationOperator(
        name="remove_section",
        description="Remove the least important sentence",
        weight=0.6,
    ),
    MutationOperator(
        name="reorder",
        description="Reorder sentences for better logical flow",
        weight=0.7,
    ),
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def initialize_population(seed_prompt: str, config: CAPOConfig) -> CAPOState:
    """Create an initial population from a seed prompt.

    Generates variations by appending suffixes, prepending instructions,
    and making minor textual tweaks.
    """
    rng = random.Random(config.seed)
    population: list[PromptCandidate] = []

    # The seed itself is always candidate 0
    population.append(PromptCandidate(prompt_text=seed_prompt, generation=0))

    suffixes = [
        " Be specific.",
        " Think step by step.",
        " Be concise.",
        " Provide examples.",
        " Explain your reasoning.",
    ]
    prefixes = [
        "You are an expert. ",
        "As a helpful assistant, ",
        "Given the following task, ",
        "Carefully consider: ",
        "Please respond accurately. ",
    ]

    attempts = 0
    max_attempts = config.population_size * 10
    while len(population) < config.population_size and attempts < max_attempts:
        attempts += 1
        strategy = rng.choice(["suffix", "prefix", "both", "identity"])
        text = seed_prompt
        if strategy == "suffix":
            text = seed_prompt + rng.choice(suffixes)
        elif strategy == "prefix":
            text = rng.choice(prefixes) + seed_prompt
        elif strategy == "both":
            text = rng.choice(prefixes) + seed_prompt + rng.choice(suffixes)
        # identity keeps the original (slight diversity via later mutation)

        # Avoid exact duplicates
        if any(c.prompt_text == text for c in population):
            continue
        population.append(PromptCandidate(prompt_text=text, generation=0))

    return CAPOState(generation=0, population=population)


def select_parent(
    population: list[PromptCandidate],
    tournament_size: int,
    rng: random.Random,
) -> PromptCandidate:
    """Tournament selection weighted by confidence-adjusted score.

    Picks *tournament_size* random candidates and returns the one with the
    highest ``score * confidence`` product.
    """
    size = min(tournament_size, len(population))
    contestants = rng.sample(population, size)
    return max(contestants, key=lambda c: c.score * max(c.confidence, 0.01))


def apply_mutation(
    candidate: PromptCandidate,
    operator: MutationOperator,
    rng: random.Random,
) -> PromptCandidate:
    """Apply a text mutation to produce a new candidate.

    Uses simple string transforms so the module stays dependency-free.
    """
    text = candidate.prompt_text

    if operator.name == "rephrase":
        # Swap two adjacent sentences
        sentences = _split_sentences(text)
        if len(sentences) >= 2:
            idx = rng.randint(0, len(sentences) - 2)
            sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
        text = " ".join(sentences)

    elif operator.name == "elaborate":
        text = text.rstrip()
        additions = [
            " Please provide a detailed response.",
            " Include all relevant details.",
            " Be thorough in your answer.",
        ]
        text += rng.choice(additions)

    elif operator.name == "simplify":
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            # Remove a random sentence
            idx = rng.randint(0, len(sentences) - 1)
            sentences.pop(idx)
        text = " ".join(sentences)

    elif operator.name == "add_example":
        text = text.rstrip() + " For example, consider a typical case."

    elif operator.name == "remove_section":
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            sentences.pop(-1)
        text = " ".join(sentences)

    elif operator.name == "reorder":
        sentences = _split_sentences(text)
        if len(sentences) >= 2:
            rng.shuffle(sentences)
        text = " ".join(sentences)

    return PromptCandidate(
        prompt_text=text,
        score=0.0,
        confidence=0.0,
        generation=candidate.generation + 1,
        mutations_applied=list(candidate.mutations_applied) + [operator.name],
    )


def compute_confidence_score(scores: list[float]) -> float:
    """Compute a confidence value from a list of evaluation scores.

    Low variance yields high confidence (close to 1.0). Returns 0.0 for
    empty or single-element lists.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)
    # Map std to confidence: std=0 -> conf=1.0, large std -> conf near 0
    confidence = 1.0 / (1.0 + std)
    return round(confidence, 4)


def should_stop(state: CAPOState, config: CAPOConfig) -> bool:
    """Check whether the optimization should terminate.

    Stops when:
    - max generations reached, or
    - best candidate's confidence exceeds the threshold and at least
      3 generations have passed (convergence).
    """
    if state.generation >= config.max_generations:
        return True
    if (
        state.best_candidate
        and state.best_candidate.confidence >= config.confidence_threshold
        and state.generation >= 3
    ):
        return True
    return False


def advance_generation(
    state: CAPOState,
    score_fn: Callable[[str], float],
    config: CAPOConfig,
) -> CAPOState:
    """Run one generation: evaluate, select, mutate, produce next population."""
    rng = random.Random((config.seed or 0) + state.generation)

    # --- Evaluate current population ---
    for candidate in state.population:
        if candidate.score == 0.0 and candidate.confidence == 0.0:
            raw_scores = [score_fn(candidate.prompt_text) for _ in range(3)]
            candidate.score = sum(raw_scores) / len(raw_scores)
            candidate.confidence = compute_confidence_score(raw_scores)

    # --- Sort by confidence-adjusted score ---
    ranked = sorted(
        state.population,
        key=lambda c: c.score * max(c.confidence, 0.01),
        reverse=True,
    )

    # --- Elitism: carry top candidates forward ---
    elite_count = min(config.elite_count, len(ranked))
    next_population: list[PromptCandidate] = list(ranked[:elite_count])

    # --- Fill remaining slots via tournament + mutation ---
    op_weights = [op.weight for op in MUTATION_OPERATORS]
    while len(next_population) < config.population_size:
        parent = select_parent(ranked, config.tournament_size, rng)
        if rng.random() < config.mutation_rate:
            operator = rng.choices(MUTATION_OPERATORS, weights=op_weights, k=1)[0]
            child = apply_mutation(parent, operator, rng)
        else:
            # Clone parent into next generation without mutation
            child = PromptCandidate(
                prompt_text=parent.prompt_text,
                score=0.0,
                confidence=0.0,
                generation=state.generation + 1,
                mutations_applied=list(parent.mutations_applied),
            )
        next_population.append(child)

    # --- Update best candidate ---
    best = ranked[0]
    if state.best_candidate is None or (
        best.score * max(best.confidence, 0.01)
        > state.best_candidate.score * max(state.best_candidate.confidence, 0.01)
    ):
        current_best = best
    else:
        current_best = state.best_candidate

    # --- Record generation stats ---
    scores = [c.score for c in ranked]
    gen_stats: dict[str, Any] = {
        "generation": state.generation,
        "best_score": ranked[0].score,
        "best_confidence": ranked[0].confidence,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "population_size": len(ranked),
    }

    return CAPOState(
        generation=state.generation + 1,
        population=next_population,
        best_candidate=current_best,
        history=list(state.history) + [gen_stats],
    )


def run_optimization(
    seed_prompt: str,
    score_fn: Callable[[str], float],
    config: CAPOConfig | None = None,
) -> CAPOState:
    """Run the full CAPO optimization loop.

    Returns the final state including the best candidate and history.
    """
    if config is None:
        config = CAPOConfig()

    state = initialize_population(seed_prompt, config)

    while not should_stop(state, config):
        state = advance_generation(state, score_fn, config)

    # Final evaluation pass if population was never scored
    for candidate in state.population:
        if candidate.score == 0.0 and candidate.confidence == 0.0:
            raw_scores = [score_fn(candidate.prompt_text) for _ in range(3)]
            candidate.score = sum(raw_scores) / len(raw_scores)
            candidate.confidence = compute_confidence_score(raw_scores)

    return state


def format_optimization_report(state: CAPOState) -> str:
    """Produce a human-readable summary of the optimization run."""
    lines: list[str] = []
    lines.append("CAPO Optimization Report")
    lines.append("=" * 40)
    lines.append(f"Generations completed: {state.generation}")
    lines.append(f"Population size: {len(state.population)}")

    if state.best_candidate:
        bc = state.best_candidate
        lines.append("")
        lines.append("Best Candidate")
        lines.append("-" * 40)
        lines.append(f"  Score: {bc.score:.4f}")
        lines.append(f"  Confidence: {bc.confidence:.4f}")
        lines.append(f"  Generation: {bc.generation}")
        lines.append(f"  Mutations: {', '.join(bc.mutations_applied) or 'none'}")
        preview = bc.prompt_text[:120]
        if len(bc.prompt_text) > 120:
            preview += "..."
        lines.append(f"  Prompt: {preview}")

    if state.history:
        lines.append("")
        lines.append("Generation History")
        lines.append("-" * 40)
        for entry in state.history:
            lines.append(
                f"  Gen {entry['generation']}: "
                f"best={entry['best_score']:.4f} "
                f"avg={entry['avg_score']:.4f} "
                f"conf={entry['best_confidence']:.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on period-space boundaries."""
    parts = text.replace(". ", ".\n").split("\n")
    return [p.strip() for p in parts if p.strip()]
