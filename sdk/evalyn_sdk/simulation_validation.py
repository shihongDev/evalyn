"""
Simulation validation: verify generated items match expected distributions.

Compares seed and generated text collections via length distribution,
vocabulary overlap, and duplicate detection. Pure Python - no external deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DistributionStats:
    """Summary statistics for a numeric distribution."""

    mean: float
    std_dev: float
    min_val: float
    max_val: float
    median: float
    count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "median": self.median,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributionStats:
        return cls(
            mean=data["mean"],
            std_dev=data["std_dev"],
            min_val=data["min_val"],
            max_val=data["max_val"],
            median=data["median"],
            count=data["count"],
        )


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    score: float
    details: str
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        return cls(
            check_name=data["check_name"],
            passed=data["passed"],
            score=data["score"],
            details=data["details"],
            threshold=data["threshold"],
        )


@dataclass
class ValidationReport:
    """Aggregated report of all validation checks."""

    results: list[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    total_checks: int = 0
    passed_checks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "overall_passed": self.overall_passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        results = [ValidationResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            overall_passed=data.get("overall_passed", True),
            total_checks=data.get("total_checks", 0),
            passed_checks=data.get("passed_checks", 0),
        )


# ---------------------------------------------------------------------------
# Distribution Helpers
# ---------------------------------------------------------------------------


def _compute_mean(values: list[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_std_dev(values: list[float], mean: float) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _compute_median(values: list[float]) -> float:
    """Compute median of sorted values."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def compute_length_distribution(texts: list[str]) -> DistributionStats:
    """Compute character-length statistics for a list of texts."""
    if not texts:
        return DistributionStats(
            mean=0.0,
            std_dev=0.0,
            min_val=0.0,
            max_val=0.0,
            median=0.0,
            count=0,
        )
    lengths = [float(len(t)) for t in texts]
    mean = _compute_mean(lengths)
    return DistributionStats(
        mean=mean,
        std_dev=_compute_std_dev(lengths, mean),
        min_val=min(lengths),
        max_val=max(lengths),
        median=_compute_median(lengths),
        count=len(lengths),
    )


def compare_distributions(
    seed_stats: DistributionStats,
    gen_stats: DistributionStats,
    tolerance: float = 0.5,
) -> ValidationResult:
    """Compare two distributions by mean and std_dev within a tolerance ratio.

    The score is the maximum relative difference across mean and std_dev.
    Passes if score <= tolerance.
    """
    if seed_stats.mean == 0 and gen_stats.mean == 0:
        return ValidationResult(
            check_name="distribution_comparison",
            passed=True,
            score=0.0,
            details="Both distributions have zero mean",
            threshold=tolerance,
        )

    # Relative difference for mean
    denom_mean = max(abs(seed_stats.mean), 1.0)
    mean_diff = abs(seed_stats.mean - gen_stats.mean) / denom_mean

    # Relative difference for std_dev
    denom_std = max(abs(seed_stats.std_dev), 1.0)
    std_diff = abs(seed_stats.std_dev - gen_stats.std_dev) / denom_std

    score = max(mean_diff, std_diff)
    passed = score <= tolerance

    return ValidationResult(
        check_name="distribution_comparison",
        passed=passed,
        score=score,
        details=(f"mean diff ratio: {mean_diff:.4f}, std_dev diff ratio: {std_diff:.4f}"),
        threshold=tolerance,
    )


def _word_set(text: str) -> set:
    """Extract lowercased word set from text."""
    return set(text.lower().split())


def compute_vocabulary_overlap(seed_texts: list[str], gen_texts: list[str]) -> float:
    """Compute Jaccard coefficient of word sets between seed and generated texts."""
    seed_words: set = set()
    for t in seed_texts:
        seed_words.update(_word_set(t))
    gen_words: set = set()
    for t in gen_texts:
        gen_words.update(_word_set(t))

    if not seed_words and not gen_words:
        return 1.0

    intersection = seed_words & gen_words
    union = seed_words | gen_words
    if not union:
        return 1.0
    return len(intersection) / len(union)


def check_vocabulary_overlap(
    seed_texts: list[str],
    gen_texts: list[str],
    min_overlap: float = 0.1,
) -> ValidationResult:
    """Check that vocabulary overlap meets a minimum threshold."""
    score = compute_vocabulary_overlap(seed_texts, gen_texts)
    passed = score >= min_overlap
    return ValidationResult(
        check_name="vocabulary_overlap",
        passed=passed,
        score=score,
        details=f"Jaccard overlap: {score:.4f}, minimum required: {min_overlap:.4f}",
        threshold=min_overlap,
    )


def find_duplicates(texts: list[str]) -> list[tuple[int, int]]:
    """Return pairs of indices that are exact duplicates."""
    pairs: list[tuple[int, int]] = []
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        if t in seen:
            pairs.append((seen[t], i))
        else:
            seen[t] = i
    return pairs


def find_near_duplicates(texts: list[str], threshold: float = 0.9) -> list[tuple[int, int, float]]:
    """Return pairs with Jaccard word similarity >= threshold."""
    results: list[tuple[int, int, float]] = []
    word_sets = [_word_set(t) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = word_sets[i], word_sets[j]
            union = a | b
            if not union:
                sim = 1.0
            else:
                sim = len(a & b) / len(union)
            if sim >= threshold:
                results.append((i, j, sim))
    return results


def check_deduplication(
    gen_texts: list[str],
    seed_texts: list[str],
    max_dup_rate: float = 0.1,
) -> ValidationResult:
    """Check duplicate rate between generated and seed texts.

    Duplicate rate = (number of gen texts that appear in seed) / len(gen_texts).
    """
    if not gen_texts:
        return ValidationResult(
            check_name="deduplication",
            passed=True,
            score=0.0,
            details="No generated texts to check",
            threshold=max_dup_rate,
        )
    seed_set = set(seed_texts)
    dup_count = sum(1 for t in gen_texts if t in seed_set)
    rate = dup_count / len(gen_texts)
    passed = rate <= max_dup_rate
    return ValidationResult(
        check_name="deduplication",
        passed=passed,
        score=rate,
        details=f"{dup_count}/{len(gen_texts)} generated texts found in seed ({rate:.4f})",
        threshold=max_dup_rate,
    )


def validate_simulation(seed_texts: list[str], gen_texts: list[str]) -> ValidationReport:
    """Run all validation checks on seed vs generated texts."""
    results: list[ValidationResult] = []

    # 1. Distribution comparison
    seed_stats = compute_length_distribution(seed_texts)
    gen_stats = compute_length_distribution(gen_texts)
    results.append(compare_distributions(seed_stats, gen_stats))

    # 2. Vocabulary overlap
    results.append(check_vocabulary_overlap(seed_texts, gen_texts))

    # 3. Deduplication
    results.append(check_deduplication(gen_texts, seed_texts))

    passed_count = sum(1 for r in results if r.passed)
    return ValidationReport(
        results=results,
        overall_passed=all(r.passed for r in results),
        total_checks=len(results),
        passed_checks=passed_count,
    )


def format_validation_report(report: ValidationReport) -> str:
    """Format a human-readable validation report."""
    lines: list[str] = []
    lines.append("Simulation Validation Report")
    lines.append("=" * 40)
    status = "PASSED" if report.overall_passed else "FAILED"
    lines.append(f"Overall: {status} ({report.passed_checks}/{report.total_checks} checks passed)")
    lines.append("")

    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"[{mark}] {r.check_name}")
        lines.append(f"  Score: {r.score:.4f} (threshold: {r.threshold:.4f})")
        lines.append(f"  {r.details}")
        lines.append("")

    return "\n".join(lines)
