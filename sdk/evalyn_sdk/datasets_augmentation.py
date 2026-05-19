"""
Data augmentation: automatically expand datasets with variations.

Provides pure functions to paraphrase, inject noise, change casing,
and truncate inputs - producing AugmentedItem records for each variation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AugmentationMethod:
    """Descriptor for a single augmentation technique."""

    name: str
    description: str = ""
    preserves_label: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "preserves_label": self.preserves_label,
        }


@dataclass
class AugmentedItem:
    """One augmented variant of an original dataset item."""

    original_id: str
    augmented_id: str
    input_text: str
    output_text: str = ""
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_id": self.original_id,
            "augmented_id": self.augmented_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "method": self.method,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentedItem:
        return cls(
            original_id=data["original_id"],
            augmented_id=data["augmented_id"],
            input_text=data["input_text"],
            output_text=data.get("output_text", ""),
            method=data.get("method", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AugmentationReport:
    """Summary statistics for an augmentation run."""

    original_count: int = 0
    augmented_count: int = 0
    methods_used: list[str] = field(default_factory=list)
    expansion_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_count": self.original_count,
            "augmented_count": self.augmented_count,
            "methods_used": self.methods_used,
            "expansion_ratio": self.expansion_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentationReport:
        return cls(
            original_count=data.get("original_count", 0),
            augmented_count=data.get("augmented_count", 0),
            methods_used=data.get("methods_used", []),
            expansion_ratio=data.get("expansion_ratio", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Augmentation Report",
            f"  Original items:  {self.original_count}",
            f"  Augmented items: {self.augmented_count}",
            f"  Expansion ratio: {self.expansion_ratio:.2f}x",
            f"  Methods used:    {', '.join(self.methods_used) if self.methods_used else 'none'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synonym map (small, deterministic)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, str] = {
    "quick": "fast",
    "fast": "quick",
    "big": "large",
    "large": "big",
    "small": "tiny",
    "tiny": "small",
    "good": "great",
    "great": "good",
    "bad": "poor",
    "poor": "bad",
    "happy": "glad",
    "glad": "happy",
    "easy": "simple",
    "simple": "easy",
    "hard": "difficult",
    "difficult": "hard",
    "start": "begin",
    "begin": "start",
    "end": "finish",
    "finish": "end",
}


# ---------------------------------------------------------------------------
# Augmentation Functions
# ---------------------------------------------------------------------------


def paraphrase_input(text: str, seed: int | None = None) -> str:
    """Simple paraphrase by reordering clauses and swapping synonyms."""
    if not text:
        return text

    rng = random.Random(seed)

    # Split on common clause separators
    for sep in [". ", ", ", "; "]:
        parts = text.split(sep)
        if len(parts) > 1:
            rng.shuffle(parts)
            text = sep.join(parts)
            break

    # Swap synonyms (word-level)
    words = text.split()
    result = []
    for w in words:
        lower = w.lower()
        if lower in _SYNONYMS and rng.random() < 0.5:
            replacement = _SYNONYMS[lower]
            # Preserve original casing of first char
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result.append(replacement)
        else:
            result.append(w)

    return " ".join(result)


def add_noise(text: str, noise_rate: float = 0.05, seed: int | None = None) -> str:
    """Add random typos at noise_rate frequency."""
    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)
    noise_ops = ["swap", "delete", "duplicate"]

    for i in range(len(chars)):
        if rng.random() < noise_rate:
            op = rng.choice(noise_ops)
            if op == "swap" and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "delete":
                chars[i] = ""
            elif op == "duplicate":
                chars[i] = chars[i] * 2

    return "".join(chars)


def change_case_style(text: str) -> str:
    """Toggle between sentence case and title case."""
    if not text:
        return text

    # Detect: if most words start uppercase, convert to sentence case; else title case
    words = text.split()
    if not words:
        return text

    upper_starts = sum(1 for w in words if w[0].isupper())
    if upper_starts > len(words) // 2:
        # Convert to sentence case: lowercase everything, then capitalize first char
        result = text.lower()
        return result[0].upper() + result[1:] if result else result
    else:
        return text.title()


def truncate_input(text: str, ratio: float = 0.8) -> str:
    """Keep first ratio% of text (by characters)."""
    if not text:
        return text
    keep = max(1, int(len(text) * ratio))
    return text[:keep]


def augment_item(
    item: dict[str, Any],
    methods: list[str] | None = None,
    seed: int | None = None,
) -> list[AugmentedItem]:
    """Apply augmentation methods to one item.

    Item dict should have "id", "input", and optionally "output".
    """
    if methods is None:
        methods = ["paraphrase"]

    item_id = item.get("id", "unknown")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    method_map = {
        "paraphrase": lambda t, s: paraphrase_input(t, seed=s),
        "noise": lambda t, s: add_noise(t, seed=s),
        "case": lambda t, _s: change_case_style(t),
        "truncate": lambda t, _s: truncate_input(t),
    }

    results: list[AugmentedItem] = []
    for idx, method_name in enumerate(methods):
        fn = method_map.get(method_name)
        if fn is None:
            continue
        aug_seed = seed + idx if seed is not None else None
        augmented_text = fn(input_text, aug_seed)
        results.append(
            AugmentedItem(
                original_id=item_id,
                augmented_id=f"{item_id}_aug_{method_name}_{idx}",
                input_text=augmented_text,
                output_text=output_text,
                method=method_name,
                metadata={"seed": aug_seed},
            )
        )

    return results


def augment_dataset(
    items: list[dict[str, Any]],
    methods: list[str] | None = None,
    augmentations_per_item: int = 2,
    seed: int | None = None,
) -> tuple[list[AugmentedItem], AugmentationReport]:
    """Augment a full dataset, returning augmented items and a report."""
    if methods is None:
        methods = ["paraphrase", "noise"]

    all_augmented: list[AugmentedItem] = []
    methods_actually_used: set[str] = set()

    for item_idx, item in enumerate(items):
        item_seed = seed + item_idx * 100 if seed is not None else None
        # Cycle through methods to reach augmentations_per_item
        chosen_methods = [methods[i % len(methods)] for i in range(augmentations_per_item)]
        augmented = augment_item(item, methods=chosen_methods, seed=item_seed)
        all_augmented.extend(augmented)
        methods_actually_used.update(m for m in chosen_methods if m in _METHOD_REGISTRY)

    original_count = len(items)
    augmented_count = len(all_augmented)
    expansion_ratio = augmented_count / original_count if original_count > 0 else 0.0

    report = AugmentationReport(
        original_count=original_count,
        augmented_count=augmented_count,
        methods_used=sorted(methods_actually_used),
        expansion_ratio=expansion_ratio,
    )

    return all_augmented, report


# ---------------------------------------------------------------------------
# Method Registry
# ---------------------------------------------------------------------------

_METHOD_REGISTRY: dict[str, AugmentationMethod] = {
    "paraphrase": AugmentationMethod(
        name="paraphrase",
        description="Reorder clauses and swap synonyms",
        preserves_label=True,
    ),
    "noise": AugmentationMethod(
        name="noise",
        description="Inject random typos at a configurable rate",
        preserves_label=True,
    ),
    "case": AugmentationMethod(
        name="case",
        description="Toggle between sentence case and title case",
        preserves_label=True,
    ),
    "truncate": AugmentationMethod(
        name="truncate",
        description="Keep the first N% of the input text",
        preserves_label=False,
    ),
}


def list_augmentation_methods() -> list[AugmentationMethod]:
    """Return all available augmentation methods."""
    return list(_METHOD_REGISTRY.values())
