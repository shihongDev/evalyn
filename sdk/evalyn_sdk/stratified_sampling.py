"""Score-stratified sampling: ensure representation across the full score range.

Creates equal-width bins from 0 to 1, assigns items to bins by score,
then samples from each bin to build a balanced subset.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreBin:
    """A single bin in the score histogram."""

    bin_id: str
    low: float
    high: float
    item_ids: list[str] = field(default_factory=list)
    count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bin_id": self.bin_id,
            "low": self.low,
            "high": self.high,
            "item_ids": self.item_ids,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreBin:
        return cls(
            bin_id=data["bin_id"],
            low=data["low"],
            high=data["high"],
            item_ids=data.get("item_ids", []),
            count=data.get("count", 0),
        )


@dataclass
class StratifiedConfig:
    """Configuration for stratified sampling."""

    n_bins: int = 5
    sample_per_bin: int = 10
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "sample_per_bin": self.sample_per_bin,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StratifiedConfig:
        return cls(
            n_bins=data.get("n_bins", 5),
            sample_per_bin=data.get("sample_per_bin", 10),
            seed=data.get("seed"),
        )


@dataclass
class StratifiedResult:
    """Result of a stratified sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    bins: list[ScoreBin] = field(default_factory=list)
    total_pool: int = 0
    total_selected: int = 0
    bin_balance: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "bins": [b.as_dict() for b in self.bins],
            "total_pool": self.total_pool,
            "total_selected": self.total_selected,
            "bin_balance": self.bin_balance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StratifiedResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            bins=[ScoreBin.from_dict(b) for b in data.get("bins", [])],
            total_pool=data.get("total_pool", 0),
            total_selected=data.get("total_selected", 0),
            bin_balance=data.get("bin_balance", 0.0),
        )


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def create_score_bins(scores: dict[str, float], n_bins: int = 5) -> list[ScoreBin]:
    """Create equal-width bins spanning [0, 1] and assign items.

    Each bin covers a range of width 1/n_bins. The last bin includes 1.0.
    Items exactly on a boundary go into the lower bin, except items at 1.0
    which go into the last bin.
    """
    if n_bins <= 0:
        return []

    width = 1.0 / n_bins
    bins: list[ScoreBin] = []
    for i in range(n_bins):
        low = i * width
        high = (i + 1) * width
        bins.append(
            ScoreBin(
                bin_id=f"bin_{i}",
                low=low,
                high=high,
                item_ids=[],
                count=0,
            )
        )

    for item_id in sorted(scores):
        score = scores[item_id]
        # Determine which bin this score falls into
        if score >= 1.0:
            idx = n_bins - 1
        elif score <= 0.0:
            idx = 0
        else:
            idx = int(score / width)
            # Clamp for floating-point edge cases
            if idx >= n_bins:
                idx = n_bins - 1
        bins[idx].item_ids.append(item_id)
        bins[idx].count += 1

    return bins


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_from_bins(
    bins: list[ScoreBin],
    sample_per_bin: int = 10,
    seed: int | None = None,
) -> list[str]:
    """Sample min(sample_per_bin, available) items from each bin.

    Returns the sampled IDs in bin order.
    """
    rng = random.Random(seed)
    selected: list[str] = []
    for b in bins:
        take = min(sample_per_bin, len(b.item_ids))
        if take <= 0:
            continue
        if take >= len(b.item_ids):
            selected.extend(sorted(b.item_ids))
        else:
            selected.extend(rng.sample(b.item_ids, take))
    return selected


# ---------------------------------------------------------------------------
# Balance metric
# ---------------------------------------------------------------------------


def compute_bin_balance(bins: list[ScoreBin], sampled_counts: list[int]) -> float:
    """Compute how uniform the sampled counts are across bins.

    Returns 1 - coefficient_of_variation of sampled counts.
    A value of 1.0 means perfectly uniform; lower values indicate imbalance.
    Returns 1.0 for empty inputs or when all counts are zero.
    Clamps to 0.0 if coefficient of variation exceeds 1.0.
    """
    if not sampled_counts:
        return 1.0

    n = len(sampled_counts)
    total = sum(sampled_counts)
    if total == 0:
        return 1.0

    mean = total / n
    if mean == 0.0:
        return 1.0

    variance = sum((c - mean) ** 2 for c in sampled_counts) / n
    std = math.sqrt(variance)
    cv = std / mean

    return max(0.0, 1.0 - cv)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_stratified_sampling(scores: dict[str, float], config: StratifiedConfig) -> StratifiedResult:
    """Full stratified sampling pipeline.

    1. Create score bins
    2. Sample from each bin
    3. Compute bin balance
    """
    if not scores:
        return StratifiedResult(
            selected_ids=[],
            bins=[],
            total_pool=0,
            total_selected=0,
            bin_balance=1.0,
        )

    bins = create_score_bins(scores, n_bins=config.n_bins)
    selected = sample_from_bins(bins, sample_per_bin=config.sample_per_bin, seed=config.seed)

    # Compute sampled counts per bin
    sampled_counts: list[int] = []
    for b in bins:
        take = min(config.sample_per_bin, len(b.item_ids))
        sampled_counts.append(take)

    balance = compute_bin_balance(bins, sampled_counts)

    return StratifiedResult(
        selected_ids=selected,
        bins=bins,
        total_pool=len(scores),
        total_selected=len(selected),
        bin_balance=balance,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_stratified_report(result: StratifiedResult) -> str:
    """Format a human-readable stratified sampling report with per-bin breakdown."""
    lines: list[str] = []
    lines.append("Stratified Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Total selected: {result.total_selected}")
    lines.append(f"Bin balance: {result.bin_balance:.4f}")
    lines.append(f"Number of bins: {len(result.bins)}")
    lines.append("")
    lines.append("Per-bin breakdown:")
    lines.append(f"  {'Bin':<10} {'Range':<15} {'Items':>8} {'Sampled':>8}")
    lines.append(f"  {'-' * 10} {'-' * 15} {'-' * 8} {'-' * 8}")
    selected_set = set(result.selected_ids)
    for b in result.bins:
        sampled = sum(1 for item_id in b.item_ids if item_id in selected_set)
        range_str = f"[{b.low:.2f}, {b.high:.2f})"
        lines.append(f"  {b.bin_id:<10} {range_str:<15} {b.count:>8} {sampled:>8}")
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
