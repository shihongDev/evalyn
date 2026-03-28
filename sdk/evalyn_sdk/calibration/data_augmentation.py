"""Calibration data augmentation.

Augment calibration examples by paraphrasing to improve optimizer
generalization. Provides synonym replacement, sentence shuffling,
and truncation strategies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Synonym Table
# ---------------------------------------------------------------------------

_SYNONYMS: Dict[str, str] = {
    "good": "excellent",
    "bad": "poor",
    "great": "outstanding",
    "important": "crucial",
    "help": "assist",
    "use": "utilize",
    "make": "create",
    "get": "obtain",
    "show": "demonstrate",
    "give": "provide",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AugmentedExample:
    """A single augmented example."""

    original_id: str
    augmented_text: str
    augmentation_method: str
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_id": self.original_id,
            "augmented_text": self.augmented_text,
            "augmentation_method": self.augmentation_method,
            "label": self.label,
        }


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    methods: List[str] = field(default_factory=lambda: ["synonym", "shuffle", "truncate"])
    augmentations_per_example: int = 2
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "methods": list(self.methods),
            "augmentations_per_example": self.augmentations_per_example,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AugmentationConfig:
        return cls(
            methods=data.get("methods", ["synonym", "shuffle", "truncate"]),
            augmentations_per_example=data.get("augmentations_per_example", 2),
            seed=data.get("seed", None),
        )


@dataclass
class AugmentationResult:
    """Summary of an augmentation batch run."""

    original_count: int
    augmented_count: int
    methods_used: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "augmented_count": self.augmented_count,
            "methods_used": list(self.methods_used),
        }

    def format_text(self) -> str:
        lines = [
            "Augmentation Result",
            f"  Originals:  {self.original_count}",
            f"  Augmented:  {self.augmented_count}",
            f"  Methods:    {', '.join(self.methods_used)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Augmentation Functions
# ---------------------------------------------------------------------------


def augment_synonym(text: str, seed: Optional[int] = None) -> str:
    """Replace common words with synonyms from the built-in table.

    Matches are case-insensitive but preserve original casing pattern
    (all lower or title case).
    """
    if not text:
        return text

    rng = random.Random(seed)
    words = text.split()
    result: List[str] = []

    for word in words:
        stripped = word.strip(".,!?;:'\"()[]")
        lower = stripped.lower()
        if lower in _SYNONYMS:
            replacement = _SYNONYMS[lower]
            # Preserve casing
            if stripped[0].isupper():
                replacement = replacement.capitalize()
            # Re-attach punctuation
            prefix = word[: word.index(stripped[0])] if stripped and stripped[0] in word else ""
            suffix = word[word.rindex(stripped[-1]) + 1 :] if stripped else ""
            result.append(prefix + replacement + suffix)
        else:
            result.append(word)

    return " ".join(result)


def augment_shuffle(text: str, seed: Optional[int] = None) -> str:
    """Shuffle sentences in the text.

    Splits on '. ', shuffles, and rejoins.
    """
    if not text:
        return text

    rng = random.Random(seed)
    sentences = text.split(". ")
    if len(sentences) <= 1:
        return text

    rng.shuffle(sentences)
    return ". ".join(sentences)


def augment_truncate(text: str, ratio: float = 0.8) -> str:
    """Keep the first ratio fraction of the text by character count."""
    if not text:
        return text

    keep = max(1, int(len(text) * ratio))
    return text[:keep]


def augment_example(text: str, method: str, seed: Optional[int] = None) -> str:
    """Route to the appropriate augmentation method."""
    if method == "synonym":
        return augment_synonym(text, seed=seed)
    elif method == "shuffle":
        return augment_shuffle(text, seed=seed)
    elif method == "truncate":
        return augment_truncate(text)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")


def augment_batch(
    examples: List[Tuple[str, str, str]],
    config: AugmentationConfig,
) -> Tuple[List[AugmentedExample], AugmentationResult]:
    """Augment a batch of examples.

    Each input tuple is (id, text, label). For each example, apply
    up to augmentations_per_example methods from the config, cycling
    through the configured methods list.
    """
    rng = random.Random(config.seed)
    augmented: List[AugmentedExample] = []
    methods_used: set[str] = set()

    for ex_id, text, label in examples:
        for i in range(config.augmentations_per_example):
            method = config.methods[i % len(config.methods)]
            # Derive a per-example seed for reproducibility
            ex_seed = rng.randint(0, 2**31) if config.seed is not None else None
            aug_text = augment_example(text, method, seed=ex_seed)
            augmented.append(
                AugmentedExample(
                    original_id=ex_id,
                    augmented_text=aug_text,
                    augmentation_method=method,
                    label=label,
                )
            )
            methods_used.add(method)

    result = AugmentationResult(
        original_count=len(examples),
        augmented_count=len(augmented),
        methods_used=sorted(methods_used),
    )
    return augmented, result
