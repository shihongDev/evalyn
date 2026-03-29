"""Coreset sampling to find a minimal representative subset.

Uses word-set features as a proxy for item similarity. Greedy
farthest-point construction preserves distributional coverage.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CoresetConfig:
    """Configuration for coreset sampling."""

    target_size: int = 50
    max_error: float = 0.1
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_size": self.target_size,
            "max_error": self.max_error,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoresetConfig:
        return cls(
            target_size=data.get("target_size", 50),
            max_error=data.get("max_error", 0.1),
            seed=data.get("seed"),
        )


@dataclass
class CoresetItem:
    """A single item selected for the coreset."""

    item_id: str
    weight: float
    representative_of: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "weight": self.weight,
            "representative_of": self.representative_of,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoresetItem:
        return cls(
            item_id=data["item_id"],
            weight=data["weight"],
            representative_of=data["representative_of"],
        )


@dataclass
class CoresetResult:
    """Result of a coreset sampling run."""

    selected: List[CoresetItem] = field(default_factory=list)
    total_pool: int = 0
    compression_ratio: float = 0.0
    max_approximation_error: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected": [s.as_dict() for s in self.selected],
            "total_pool": self.total_pool,
            "compression_ratio": self.compression_ratio,
            "max_approximation_error": self.max_approximation_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoresetResult:
        return cls(
            selected=[CoresetItem.from_dict(s) for s in data.get("selected", [])],
            total_pool=data.get("total_pool", 0),
            compression_ratio=data.get("compression_ratio", 0.0),
            max_approximation_error=data.get("max_approximation_error", 0.0),
        )


# ---------------------------------------------------------------------------
# Feature extraction and distance
# ---------------------------------------------------------------------------


def compute_item_features(text: str) -> set[str]:
    """Extract lowercased word set from text, keeping words of length >= 2."""
    return {w for w in text.lower().split() if len(w) >= 2}


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    """Jaccard distance between two sets. Returns 1.0 if both are empty."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def greedy_coreset(
    items: dict[str, str],
    target_size: int,
    seed: int | None = None,
) -> CoresetResult:
    """Greedy farthest-point coreset construction.

    1. Pre-compute word-set features for every item.
    2. Pick the most "central" item (highest total Jaccard similarity
       to all others) as the first coreset member.
    3. Iteratively add the item that is farthest from the current
       coreset (max min-distance to any coreset member).
    4. Assign each non-coreset item to its nearest coreset member
       and derive weights proportional to representation count.
    """
    if not items:
        return CoresetResult(
            selected=[],
            total_pool=0,
            compression_ratio=0.0,
            max_approximation_error=0.0,
        )

    ids = list(items.keys())
    n = len(ids)
    actual_target = min(target_size, n)

    # Pre-compute features
    features: Dict[str, set[str]] = {
        item_id: compute_item_features(text) for item_id, text in items.items()
    }

    # Pre-compute pairwise distances (symmetric, so store upper triangle)
    dist_cache: Dict[tuple[str, str], float] = {}

    def get_dist(a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in dist_cache:
            dist_cache[key] = _jaccard_distance(features[a], features[b])
        return dist_cache[key]

    # Step 1: find the most central item (highest total similarity = lowest total distance)
    best_id = ids[0]
    best_total_dist = float("inf")
    for cid in ids:
        total = sum(get_dist(cid, oid) for oid in ids if oid != cid)
        if total < best_total_dist:
            best_total_dist = total
            best_id = cid

    coreset_ids: List[str] = [best_id]

    # min_dist_to_coreset[id] = distance to nearest coreset member
    min_dist: Dict[str, float] = {}
    for item_id in ids:
        if item_id == best_id:
            continue
        min_dist[item_id] = get_dist(item_id, best_id)

    # Step 2: greedy farthest-point addition
    rng = random.Random(seed)
    while len(coreset_ids) < actual_target:
        # Find item with max min-distance to coreset
        candidates = [
            item_id for item_id in min_dist if item_id not in set(coreset_ids)
        ]
        if not candidates:
            break
        max_dist = max(min_dist[c] for c in candidates)
        # Break ties randomly
        tied = [c for c in candidates if min_dist[c] == max_dist]
        chosen = rng.choice(tied) if len(tied) > 1 else tied[0]
        coreset_ids.append(chosen)
        # Update min distances
        for item_id in candidates:
            if item_id == chosen:
                continue
            d = get_dist(item_id, chosen)
            if d < min_dist[item_id]:
                min_dist[item_id] = d

    # Step 3: assign each item to nearest coreset member, compute weights
    coreset_set = set(coreset_ids)
    representation: Dict[str, int] = {cid: 0 for cid in coreset_ids}
    for item_id in ids:
        nearest = coreset_ids[0]
        nearest_dist = get_dist(item_id, coreset_ids[0])
        for cid in coreset_ids[1:]:
            d = get_dist(item_id, cid)
            if d < nearest_dist:
                nearest = cid
                nearest_dist = d
        representation[nearest] += 1

    total_pool = n
    selected: List[CoresetItem] = []
    for cid in coreset_ids:
        rep = representation[cid]
        weight = rep / total_pool if total_pool > 0 else 0.0
        selected.append(CoresetItem(item_id=cid, weight=weight, representative_of=rep))

    approx_error = compute_approximation_error(coreset_ids, items)
    compression = len(coreset_ids) / total_pool if total_pool > 0 else 0.0

    return CoresetResult(
        selected=selected,
        total_pool=total_pool,
        compression_ratio=compression,
        max_approximation_error=approx_error,
    )


def compute_approximation_error(
    coreset_ids: list[str], all_items: dict[str, str]
) -> float:
    """Max Jaccard distance from any item to its nearest coreset member.

    Returns 0.0 if coreset_ids is empty or all_items is empty.
    """
    if not coreset_ids or not all_items:
        return 0.0

    features: Dict[str, set[str]] = {
        item_id: compute_item_features(text) for item_id, text in all_items.items()
    }
    coreset_features = {cid: features[cid] for cid in coreset_ids if cid in features}

    max_error = 0.0
    for item_id, feat in features.items():
        min_d = min(
            _jaccard_distance(feat, cf) for cf in coreset_features.values()
        )
        if min_d > max_error:
            max_error = min_d
    return max_error


def validate_coreset(
    result: CoresetResult, items: dict[str, str], max_error: float = 0.1
) -> bool:
    """True if the coreset's approximation error is within the bound."""
    coreset_ids = [s.item_id for s in result.selected]
    error = compute_approximation_error(coreset_ids, items)
    return error <= max_error


def run_coreset_sampling(
    items: dict[str, str], config: CoresetConfig
) -> CoresetResult:
    """Full coreset sampling pipeline: build coreset from config."""
    return greedy_coreset(
        items=items,
        target_size=config.target_size,
        seed=config.seed,
    )


def format_coreset_report(result: CoresetResult) -> str:
    """Human-readable coreset report."""
    lines = [
        "Coreset Report",
        f"  Total pool:          {result.total_pool}",
        f"  Selected:            {len(result.selected)}",
        f"  Compression ratio:   {result.compression_ratio:.4f}",
        f"  Max approx. error:   {result.max_approximation_error:.4f}",
        "",
        "  Selected items:",
    ]
    for item in result.selected:
        lines.append(
            f"    {item.item_id}: weight={item.weight:.4f}, "
            f"represents={item.representative_of}"
        )
    return "\n".join(lines)
