"""
Evol-Instruct data evolution: evolve evaluation items through iterative
complexity increases using depth and breadth strategies. Pure Python, no
external dependencies.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EvolutionStrategy:
    """A named evolution strategy with a set of transforms."""

    strategy_id: str  # "depth" or "breadth"
    description: str
    transforms: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "transforms": list(self.transforms),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvolutionStrategy:
        return cls(
            strategy_id=data["strategy_id"],
            description=data["description"],
            transforms=list(data["transforms"]),
        )


@dataclass
class EvolvedItem:
    """A single evolved item with provenance."""

    original_text: str
    evolved_text: str
    strategy: str
    generation: int
    quality_score: float = 0.0
    accepted: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "evolved_text": self.evolved_text,
            "strategy": self.strategy,
            "generation": self.generation,
            "quality_score": self.quality_score,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvolvedItem:
        return cls(
            original_text=data["original_text"],
            evolved_text=data["evolved_text"],
            strategy=data["strategy"],
            generation=data["generation"],
            quality_score=data.get("quality_score", 0.0),
            accepted=data.get("accepted", True),
        )


@dataclass
class EvolutionConfig:
    """Configuration for batch evolution."""

    max_generations: int = 3
    quality_threshold: float = 0.3
    strategies: List[str] = field(default_factory=lambda: ["depth", "breadth"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_generations": self.max_generations,
            "quality_threshold": self.quality_threshold,
            "strategies": list(self.strategies),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvolutionConfig:
        return cls(
            max_generations=data.get("max_generations", 3),
            quality_threshold=data.get("quality_threshold", 0.3),
            strategies=data.get("strategies", ["depth", "breadth"]),
        )


@dataclass
class EvolutionReport:
    """Summary of a batch evolution run."""

    items: List[EvolvedItem]
    total_evolved: int
    accepted_count: int
    rejected_count: int
    avg_quality: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "total_evolved": self.total_evolved,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "avg_quality": self.avg_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvolutionReport:
        return cls(
            items=[EvolvedItem.from_dict(d) for d in data["items"]],
            total_evolved=data["total_evolved"],
            accepted_count=data["accepted_count"],
            rejected_count=data["rejected_count"],
            avg_quality=data["avg_quality"],
        )


# ---------------------------------------------------------------------------
# Transform Catalogs
# ---------------------------------------------------------------------------

DEPTH_TRANSFORMS: List[str] = [
    "add_constraint",
    "add_reasoning_step",
    "add_edge_case",
    "increase_specificity",
    "add_conditional",
]

BREADTH_TRANSFORMS: List[str] = [
    "topic_variation",
    "domain_transfer",
    "rephrase",
    "change_format",
    "add_context",
]

# Built-in strategy objects
DEPTH_STRATEGY = EvolutionStrategy(
    strategy_id="depth",
    description="Increase complexity by adding constraints and reasoning steps",
    transforms=list(DEPTH_TRANSFORMS),
)

BREADTH_STRATEGY = EvolutionStrategy(
    strategy_id="breadth",
    description="Increase diversity by rephrasing, changing domain, or format",
    transforms=list(BREADTH_TRANSFORMS),
)

_STRATEGIES: Dict[str, EvolutionStrategy] = {
    "depth": DEPTH_STRATEGY,
    "breadth": BREADTH_STRATEGY,
}


# ---------------------------------------------------------------------------
# Depth Transforms
# ---------------------------------------------------------------------------

_CONSTRAINTS = [
    "the solution must use no more than O(n) time complexity",
    "the input may contain Unicode characters",
    "the answer must be case-insensitive",
    "empty inputs should return a meaningful default",
    "the result must be deterministic",
]

_EDGE_CASES = [
    "the input is empty",
    "all values are identical",
    "the input contains negative numbers",
    "the input exceeds 10,000 elements",
    "special characters are present",
]

_REASONING_PREFIXES = [
    "First, identify the core requirement, then",
    "Begin by analyzing the constraints, then",
    "Start by breaking the problem into sub-problems, then",
    "First reason about the boundary conditions, then",
    "Consider the trade-offs involved, then",
]

_SPECIFICITY_TAGS = [
    "Provide a step-by-step solution.",
    "Include time and space complexity analysis.",
    "Explain your reasoning at each step.",
    "State any assumptions explicitly.",
    "Give an example input and expected output.",
]

_CONDITIONALS = [
    "If the input is sorted, optimize accordingly; otherwise, handle the general case.",
    "If the dataset fits in memory, use an in-memory approach; otherwise, use streaming.",
    "If precision matters more than speed, prefer exact methods; otherwise, use approximations.",
    "If the user specifies a language, respond in that language; otherwise, default to English.",
    "If the input contains duplicates, explain how they are handled.",
]


def apply_depth_transform(text: str, transform: str, rng: random.Random) -> str:
    """Apply an in-depth complexity-increasing transform to the text."""
    if transform == "add_constraint":
        constraint = rng.choice(_CONSTRAINTS)
        return f"{text} Additionally, ensure that {constraint}."
    if transform == "add_reasoning_step":
        prefix = rng.choice(_REASONING_PREFIXES)
        return f"{prefix} {text.rstrip('.')}"
    if transform == "add_edge_case":
        edge = rng.choice(_EDGE_CASES)
        return f"{text} Consider the edge case where {edge}."
    if transform == "increase_specificity":
        tag = rng.choice(_SPECIFICITY_TAGS)
        return f"{text} {tag}"
    if transform == "add_conditional":
        cond = rng.choice(_CONDITIONALS)
        return f"{text} {cond}"
    return text


# ---------------------------------------------------------------------------
# Breadth Transforms
# ---------------------------------------------------------------------------

_DOMAIN_SWAPS = [
    ("software", "biology"),
    ("data", "music"),
    ("algorithm", "recipe"),
    ("system", "ecosystem"),
    ("function", "process"),
    ("test", "experiment"),
    ("user", "participant"),
    ("error", "anomaly"),
]

_REPHRASE_TEMPLATES = [
    "Restate the following in different terms: {text}",
    "In other words: {text}",
    "Put differently: {text}",
    "An alternative way to phrase this: {text}",
]

_FORMAT_TEMPLATES = [
    "Present your answer as a numbered list: {text}",
    "Respond in a Q&A format: {text}",
    "Structure your answer with clear headings: {text}",
    "Provide your answer as a concise summary followed by details: {text}",
]

_CONTEXT_ADDITIONS = [
    "You are a senior engineer reviewing code.",
    "Assume the reader has no prior background.",
    "This is part of a timed assessment.",
    "The audience is a non-technical stakeholder.",
    "Consider this in the context of a production environment.",
]


def apply_breadth_transform(text: str, transform: str, rng: random.Random) -> str:
    """Apply a breadth-expanding transform to the text."""
    if transform == "topic_variation":
        # Shuffle sentence order (if multiple sentences)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 1:
            shuffled = list(sentences)
            rng.shuffle(shuffled)
            return " ".join(shuffled)
        return text
    if transform == "domain_transfer":
        result = text
        pair = rng.choice(_DOMAIN_SWAPS)
        result = result.replace(pair[0], pair[1])
        return result
    if transform == "rephrase":
        template = rng.choice(_REPHRASE_TEMPLATES)
        return template.format(text=text)
    if transform == "change_format":
        template = rng.choice(_FORMAT_TEMPLATES)
        return template.format(text=text)
    if transform == "add_context":
        ctx = rng.choice(_CONTEXT_ADDITIONS)
        return f"{ctx} {text}"
    return text


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------


def score_evolution_quality(original: str, evolved: str) -> float:
    """
    Heuristic quality score for an evolution.

    Components (each 0-1, averaged):
    - Length increase ratio (capped at 2x)
    - Vocabulary expansion (new unique words / original unique words, capped)
    - Structural complexity (sentence count increase, capped)
    """
    if not original or not evolved:
        return 0.0

    # Length ratio: how much longer the evolved text is (capped at 2x)
    len_ratio = min(len(evolved) / max(len(original), 1), 3.0) - 1.0
    len_score = min(max(len_ratio / 2.0, 0.0), 1.0)

    # Vocabulary expansion
    orig_words = set(original.lower().split())
    evolved_words = set(evolved.lower().split())
    new_words = evolved_words - orig_words
    vocab_score = min(len(new_words) / max(len(orig_words), 1), 1.0)

    # Structural complexity: sentence count increase
    orig_sentences = len([s for s in re.split(r"[.!?]+", original) if s.strip()])
    evolved_sentences = len([s for s in re.split(r"[.!?]+", evolved) if s.strip()])
    sent_increase = evolved_sentences - orig_sentences
    struct_score = min(max(sent_increase / 3.0, 0.0), 1.0)

    return (len_score + vocab_score + struct_score) / 3.0


# ---------------------------------------------------------------------------
# Evolution Functions
# ---------------------------------------------------------------------------


def evolve_item(
    text: str,
    strategy: str = "depth",
    generation: int = 0,
    rng: random.Random | None = None,
) -> EvolvedItem:
    """Evolve a single text item using the given strategy."""
    if rng is None:
        rng = random.Random()

    strat = _STRATEGIES.get(strategy)
    if strat is None:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    transform = rng.choice(strat.transforms)

    if strategy == "depth":
        evolved_text = apply_depth_transform(text, transform, rng)
    else:
        evolved_text = apply_breadth_transform(text, transform, rng)

    quality = score_evolution_quality(text, evolved_text)

    return EvolvedItem(
        original_text=text,
        evolved_text=evolved_text,
        strategy=strategy,
        generation=generation,
        quality_score=quality,
        accepted=True,
    )


def evolve_batch(
    items: list[str],
    config: EvolutionConfig,
    rng: random.Random | None = None,
) -> EvolutionReport:
    """
    Evolve all items through multiple generations, filtering by quality.

    Each generation feeds the evolved text of accepted items from the previous
    generation back through the pipeline. Items below the quality threshold
    are marked rejected but still tracked.
    """
    if rng is None:
        rng = random.Random()

    all_evolved: list[EvolvedItem] = []
    accepted_count = 0
    rejected_count = 0

    for text in items:
        current_text = text
        for gen in range(config.max_generations):
            strategy = rng.choice(config.strategies)
            item = evolve_item(current_text, strategy=strategy, generation=gen, rng=rng)
            item.original_text = text  # always track original

            if item.quality_score < config.quality_threshold:
                item.accepted = False
                rejected_count += 1
                all_evolved.append(item)
                break  # stop evolving this item
            else:
                accepted_count += 1
                all_evolved.append(item)
                current_text = item.evolved_text

    total = len(all_evolved)
    qualities = [e.quality_score for e in all_evolved]
    avg_q = sum(qualities) / max(len(qualities), 1)

    return EvolutionReport(
        items=all_evolved,
        total_evolved=total,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        avg_quality=avg_q,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_evolution_report(report: EvolutionReport) -> str:
    """Human-readable summary of an evolution report."""
    lines = [
        "Evolution Report",
        "=" * 40,
        f"Total evolved:  {report.total_evolved}",
        f"Accepted:       {report.accepted_count}",
        f"Rejected:       {report.rejected_count}",
        f"Avg quality:    {report.avg_quality:.3f}",
        "",
    ]
    for i, item in enumerate(report.items):
        status = "ACCEPTED" if item.accepted else "REJECTED"
        lines.append(
            f"  [{i + 1}] gen={item.generation} strategy={item.strategy} "
            f"quality={item.quality_score:.3f} {status}"
        )
        preview = item.evolved_text[:80]
        if len(item.evolved_text) > 80:
            preview += "..."
        lines.append(f"      {preview}")
    return "\n".join(lines)
