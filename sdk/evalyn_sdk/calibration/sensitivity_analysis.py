"""Calibration sensitivity analysis.

Measure alignment sensitivity to small prompt perturbations -
word swaps, whitespace, case changes, and synonym substitution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Built-in synonym dictionary
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    "good": ["great", "fine", "nice"],
    "bad": ["poor", "terrible", "awful"],
    "fast": ["quick", "rapid", "speedy"],
    "slow": ["sluggish", "gradual", "unhurried"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minor"],
    "happy": ["glad", "pleased", "joyful"],
    "sad": ["unhappy", "gloomy", "sorrowful"],
    "easy": ["simple", "straightforward", "effortless"],
    "hard": ["difficult", "tough", "challenging"],
    "important": ["significant", "crucial", "vital"],
    "help": ["assist", "aid", "support"],
    "show": ["display", "present", "demonstrate"],
    "make": ["create", "build", "produce"],
    "use": ["utilize", "employ", "apply"],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Perturbation:
    """A single text perturbation."""

    original: str
    perturbed: str
    method: str = "word_swap"
    change_description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "method": self.method,
            "change_description": self.change_description,
        }


@dataclass
class SensitivityResult:
    """Result of scoring a perturbation."""

    perturbation: Perturbation
    original_score: float
    perturbed_score: float
    delta: float
    is_sensitive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "perturbation": self.perturbation.as_dict(),
            "original_score": self.original_score,
            "perturbed_score": self.perturbed_score,
            "delta": self.delta,
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class SensitivityReport:
    """Aggregated sensitivity analysis report."""

    results: list[SensitivityResult] = field(default_factory=list)
    avg_sensitivity: float = 0.0
    max_sensitivity: float = 0.0
    sensitive_count: int = 0
    robust_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "avg_sensitivity": self.avg_sensitivity,
            "max_sensitivity": self.max_sensitivity,
            "sensitive_count": self.sensitive_count,
            "robust_count": self.robust_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensitivityReport:
        results = []
        for r in data.get("results", []):
            p_data = r["perturbation"]
            perturbation = Perturbation(
                original=p_data["original"],
                perturbed=p_data["perturbed"],
                method=p_data.get("method", "word_swap"),
                change_description=p_data.get("change_description", ""),
            )
            results.append(
                SensitivityResult(
                    perturbation=perturbation,
                    original_score=r["original_score"],
                    perturbed_score=r["perturbed_score"],
                    delta=r["delta"],
                    is_sensitive=r.get("is_sensitive", False),
                )
            )
        return cls(
            results=results,
            avg_sensitivity=data.get("avg_sensitivity", 0.0),
            max_sensitivity=data.get("max_sensitivity", 0.0),
            sensitive_count=data.get("sensitive_count", 0),
            robust_count=data.get("robust_count", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Sensitivity Report: {len(self.results)} perturbations",
            f"  Avg sensitivity: {self.avg_sensitivity:.4f}",
            f"  Max sensitivity: {self.max_sensitivity:.4f}",
            f"  Sensitive: {self.sensitive_count}, Robust: {self.robust_count}",
        ]
        for r in self.results:
            marker = "SENSITIVE" if r.is_sensitive else "robust"
            lines.append(
                f"  [{r.perturbation.method}] delta={r.delta:+.4f} ({marker})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perturbation Functions
# ---------------------------------------------------------------------------


def perturb_word_swap(text: str, seed: int | None = None) -> Perturbation:
    """Swap two random adjacent words."""
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 2:
        return Perturbation(
            original=text,
            perturbed=text,
            method="word_swap",
            change_description="text too short to swap",
        )
    idx = rng.randint(0, len(words) - 2)
    new_words = list(words)
    new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
    perturbed = " ".join(new_words)
    return Perturbation(
        original=text,
        perturbed=perturbed,
        method="word_swap",
        change_description=f"swapped words at positions {idx} and {idx + 1}",
    )


def perturb_add_whitespace(text: str) -> Perturbation:
    """Add extra whitespace between random words."""
    words = text.split()
    if len(words) < 2:
        return Perturbation(
            original=text,
            perturbed=text,
            method="add_whitespace",
            change_description="text too short for whitespace insertion",
        )
    # Add double space between each pair of words
    perturbed = "  ".join(words)
    return Perturbation(
        original=text,
        perturbed=perturbed,
        method="add_whitespace",
        change_description="added extra whitespace between words",
    )


def perturb_case_change(text: str, seed: int | None = None) -> Perturbation:
    """Randomly change case of one word."""
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return Perturbation(
            original=text,
            perturbed=text,
            method="case_change",
            change_description="empty text",
        )
    idx = rng.randint(0, len(words) - 1)
    new_words = list(words)
    word = new_words[idx]
    if word == word.upper():
        new_words[idx] = word.lower()
    else:
        new_words[idx] = word.upper()
    perturbed = " ".join(new_words)
    return Perturbation(
        original=text,
        perturbed=perturbed,
        method="case_change",
        change_description=f"changed case of word at position {idx}",
    )


def perturb_synonym(text: str, seed: int | None = None) -> Perturbation:
    """Replace one word with a synonym from built-in dictionary."""
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return Perturbation(
            original=text,
            perturbed=text,
            method="synonym",
            change_description="empty text",
        )

    # Find words that have synonyms
    candidates = []
    for i, w in enumerate(words):
        if w.lower() in _SYNONYMS:
            candidates.append(i)

    if not candidates:
        return Perturbation(
            original=text,
            perturbed=text,
            method="synonym",
            change_description="no synonyms available",
        )

    idx = rng.choice(candidates)
    original_word = words[idx]
    synonyms = _SYNONYMS[original_word.lower()]
    replacement = rng.choice(synonyms)

    # Preserve original case style
    if original_word[0].isupper():
        replacement = replacement.capitalize()
    if original_word.isupper():
        replacement = replacement.upper()

    new_words = list(words)
    new_words[idx] = replacement
    perturbed = " ".join(new_words)
    return Perturbation(
        original=text,
        perturbed=perturbed,
        method="synonym",
        change_description=f"replaced '{original_word}' with '{replacement}'",
    )


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def generate_perturbations(
    text: str, num_perturbations: int = 5, seed: int | None = None
) -> list[Perturbation]:
    """Generate multiple perturbations using different methods."""
    rng = random.Random(seed)
    methods = [perturb_word_swap, perturb_add_whitespace, perturb_case_change, perturb_synonym]
    perturbations: list[Perturbation] = []

    for i in range(num_perturbations):
        method_seed = rng.randint(0, 2**31 - 1)
        fn = methods[i % len(methods)]
        if fn is perturb_add_whitespace:
            p = fn(text)
        else:
            p = fn(text, seed=method_seed)
        perturbations.append(p)

    return perturbations


def analyze_sensitivity(
    results: list[SensitivityResult], threshold: float = 0.05
) -> SensitivityReport:
    """Compute avg/max sensitivity and classify results.

    A result is sensitive if abs(delta) > threshold.
    """
    if not results:
        return SensitivityReport()

    classified: list[SensitivityResult] = []
    sensitive_count = 0
    robust_count = 0

    for r in results:
        is_sensitive = abs(r.delta) > threshold
        classified.append(
            SensitivityResult(
                perturbation=r.perturbation,
                original_score=r.original_score,
                perturbed_score=r.perturbed_score,
                delta=r.delta,
                is_sensitive=is_sensitive,
            )
        )
        if is_sensitive:
            sensitive_count += 1
        else:
            robust_count += 1

    abs_deltas = [abs(r.delta) for r in results]
    avg_sensitivity = sum(abs_deltas) / len(abs_deltas)
    max_sensitivity = max(abs_deltas)

    return SensitivityReport(
        results=classified,
        avg_sensitivity=avg_sensitivity,
        max_sensitivity=max_sensitivity,
        sensitive_count=sensitive_count,
        robust_count=robust_count,
    )
