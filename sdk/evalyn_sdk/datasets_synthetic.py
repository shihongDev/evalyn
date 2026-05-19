"""
Synthetic data generation for evaluation datasets.

Generate configurable synthetic items with category/difficulty distribution,
adversarial variants, and dataset validation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = [
    "What is {topic}?",
    "Define {topic}.",
    "Summarize {topic} briefly.",
    "Give a one-sentence description of {topic}.",
]

_MEDIUM_TEMPLATES = [
    "Explain how {topic} works and give an example.",
    "Compare {topic} with its closest alternative.",
    "Describe the key benefits and drawbacks of {topic}.",
    "Outline the main concepts behind {topic}.",
]

_HARD_TEMPLATES = [
    "Analyze the trade-offs of {topic} in a production system. "
    "Provide concrete metrics, edge cases, and a recommended mitigation strategy.",
    "Given a scenario where {topic} fails under high concurrency, "
    "describe the root cause, diagnostic steps, and a fix with code.",
    "Design a system that leverages {topic} at scale. "
    "Include architecture diagram description, failure modes, and capacity plan.",
    "Critically evaluate three competing approaches to {topic}. "
    "Rank them by latency, cost, and reliability with justification.",
]

_DIFFICULTY_TEMPLATES: Dict[str, List[str]] = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
}

_TOPICS = [
    "caching",
    "load balancing",
    "database indexing",
    "rate limiting",
    "error handling",
    "input validation",
    "authentication",
    "logging",
    "serialization",
    "concurrency",
    "encryption",
    "pagination",
    "retry logic",
    "circuit breakers",
    "feature flags",
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset generation."""

    num_items: int = 10
    categories: List[str] = field(default_factory=lambda: ["general"])
    difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_items": self.num_items,
            "categories": list(self.categories),
            "difficulty_distribution": dict(self.difficulty_distribution),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticConfig:
        return cls(
            num_items=data.get("num_items", 10),
            categories=data.get("categories", ["general"]),
            difficulty_distribution=data.get(
                "difficulty_distribution", {"easy": 0.3, "medium": 0.5, "hard": 0.2}
            ),
            seed=data.get("seed"),
        )


@dataclass
class SyntheticItem:
    """A single synthetic evaluation item."""

    id: str
    input_text: str
    expected_output: str = ""
    category: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "category": self.category,
            "difficulty": self.difficulty,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticItem:
        return cls(
            id=data["id"],
            input_text=data["input_text"],
            expected_output=data.get("expected_output", ""),
            category=data.get("category", "general"),
            difficulty=data.get("difficulty", "medium"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SyntheticDataset:
    """A collection of synthetic evaluation items with config metadata."""

    items: List[SyntheticItem] = field(default_factory=list)
    config: Optional[SyntheticConfig] = None

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts

    @property
    def by_difficulty(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.difficulty] = counts.get(item.difficulty, 0) + 1
        return counts

    def format_text(self) -> str:
        lines = [f"SyntheticDataset: {self.count} items"]
        for cat, n in sorted(self.by_category.items()):
            lines.append(f"  {cat}: {n}")
        for diff, n in sorted(self.by_difficulty.items()):
            lines.append(f"  [{diff}]: {n}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "config": self.config.as_dict() if self.config else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticDataset:
        items = [SyntheticItem.from_dict(d) for d in data.get("items", [])]
        config_data = data.get("config")
        config = SyntheticConfig.from_dict(config_data) if config_data else None
        return cls(items=items, config=config)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _make_rng(seed: Optional[int] = None) -> random.Random:
    """Create a seeded Random instance."""
    return random.Random(seed)


def _hash_id(item_id: str) -> int:
    """Deterministic int from item_id for template selection."""
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16)


def generate_synthetic_item(
    item_id: str,
    category: str = "general",
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> SyntheticItem:
    """Generate a single synthetic evaluation item.

    Easy items have short, simple inputs. Hard items have complex multi-part inputs.
    Template selection is deterministic when seed is provided.
    """
    rng = _make_rng(seed)
    templates = _DIFFICULTY_TEMPLATES.get(difficulty, _MEDIUM_TEMPLATES)
    topic_idx = (_hash_id(item_id) + (seed or 0)) % len(_TOPICS)
    topic = _TOPICS[topic_idx]
    template = templates[rng.randint(0, len(templates) - 1)]
    input_text = template.format(topic=topic)
    expected_hash = hashlib.sha256(
        f"{item_id}:{category}:{difficulty}".encode()
    ).hexdigest()[:12]
    expected_output = f"expected_{expected_hash}"
    return SyntheticItem(
        id=item_id,
        input_text=input_text,
        expected_output=expected_output,
        category=category,
        difficulty=difficulty,
        metadata={"synthetic": True, "topic": topic},
    )


def generate_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Generate a full dataset based on config.

    Distributes items across categories and difficulties according to
    the configured distribution.
    """
    rng = _make_rng(config.seed)
    items: List[SyntheticItem] = []

    if config.num_items == 0:
        return SyntheticDataset(items=items, config=config)

    # Build difficulty pool based on distribution
    difficulties: List[str] = []
    for diff, weight in config.difficulty_distribution.items():
        count = max(1, round(weight * config.num_items))
        difficulties.extend([diff] * count)
    # Trim or pad to exact num_items
    rng.shuffle(difficulties)
    while len(difficulties) < config.num_items:
        difficulties.append(
            rng.choice(list(config.difficulty_distribution.keys()))
        )
    difficulties = difficulties[: config.num_items]

    for i in range(config.num_items):
        category = config.categories[i % len(config.categories)]
        difficulty = difficulties[i]
        item_id = f"synth_{i:04d}"
        item_seed = (config.seed + i) if config.seed is not None else None
        items.append(
            generate_synthetic_item(
                item_id=item_id,
                category=category,
                difficulty=difficulty,
                seed=item_seed,
            )
        )

    return SyntheticDataset(items=items, config=config)


def generate_adversarial_items(
    base_items: List[SyntheticItem],
    num_adversarial: int = 5,
    seed: Optional[int] = None,
) -> List[SyntheticItem]:
    """Create adversarial variants of base items.

    Produces items with typos, edge cases, and boundary inputs.
    """
    if not base_items:
        return []

    rng = _make_rng(seed)
    adversarial: List[SyntheticItem] = []

    transforms = [
        ("typo", _add_typos),
        ("empty_suffix", _add_empty_suffix),
        ("boundary", _add_boundary_chars),
        ("repeat", _repeat_input),
        ("truncate", _truncate_input),
    ]

    for i in range(num_adversarial):
        base = base_items[i % len(base_items)]
        transform_name, transform_fn = transforms[i % len(transforms)]
        new_input = transform_fn(base.input_text, rng)
        adversarial.append(
            SyntheticItem(
                id=f"{base.id}_adv_{i:03d}",
                input_text=new_input,
                expected_output=base.expected_output,
                category=base.category,
                difficulty=base.difficulty,
                metadata={
                    "synthetic": True,
                    "adversarial": True,
                    "transform": transform_name,
                    "source_id": base.id,
                },
            )
        )

    return adversarial


def validate_synthetic_dataset(
    dataset: SyntheticDataset,
) -> Tuple[bool, List[str]]:
    """Validate a synthetic dataset.

    Checks:
    - No duplicate IDs
    - All items have input_text
    - Category distribution matches config (if config present)

    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # Check duplicate IDs
    seen_ids: set[str] = set()
    for item in dataset.items:
        if item.id in seen_ids:
            errors.append(f"Duplicate ID: {item.id}")
        seen_ids.add(item.id)

    # Check input_text present
    for item in dataset.items:
        if not item.input_text or not item.input_text.strip():
            errors.append(f"Item {item.id} has empty input_text")

    # Check category distribution matches config
    if dataset.config and dataset.items:
        actual_cats = set(item.category for item in dataset.items)
        expected_cats = set(dataset.config.categories)
        extra = actual_cats - expected_cats
        if extra:
            errors.append(f"Unexpected categories: {sorted(extra)}")

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Adversarial transforms (private)
# ---------------------------------------------------------------------------


def _add_typos(text: str, rng: random.Random) -> str:
    if len(text) < 3:
        return text
    chars = list(text)
    pos = rng.randint(1, len(chars) - 2)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _add_empty_suffix(text: str, rng: random.Random) -> str:
    padding = " " * rng.randint(3, 10)
    return text + padding


def _add_boundary_chars(text: str, rng: random.Random) -> str:
    boundary = rng.choice(["<script>alert(1)</script>", "\x00\x01\x02", "' OR 1=1 --"])
    return text + " " + boundary


def _repeat_input(text: str, rng: random.Random) -> str:
    repeats = rng.randint(2, 5)
    return (text + " ") * repeats


def _truncate_input(text: str, rng: random.Random) -> str:
    if len(text) <= 3:
        return text
    cut = rng.randint(1, max(1, len(text) // 2))
    return text[:cut]
