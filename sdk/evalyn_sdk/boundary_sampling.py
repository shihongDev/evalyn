"""Cluster boundary sampling: select items near cluster decision boundaries.

Uses word-set Jaccard distance as a lightweight proxy for semantic distance.
Items that sit between two clusters (ambiguous membership) are prioritized
for annotation. Pure Python, no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundaryScore:
    """Boundary score for a single item."""

    item_id: str
    boundary_distance: float  # 0 = on boundary, 1 = at centroid
    cluster_id: int
    is_boundary: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "boundary_distance": self.boundary_distance,
            "cluster_id": self.cluster_id,
            "is_boundary": self.is_boundary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryScore:
        return cls(
            item_id=data["item_id"],
            boundary_distance=data["boundary_distance"],
            cluster_id=data["cluster_id"],
            is_boundary=data["is_boundary"],
        )


@dataclass
class BoundaryConfig:
    """Configuration for boundary sampling."""

    n_clusters: int = 3
    sample_size: int = 50
    boundary_threshold: float = 0.3  # items below this are "boundary"
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "sample_size": self.sample_size,
            "boundary_threshold": self.boundary_threshold,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryConfig:
        return cls(
            n_clusters=data.get("n_clusters", 3),
            sample_size=data.get("sample_size", 50),
            boundary_threshold=data.get("boundary_threshold", 0.3),
            seed=data.get("seed"),
        )


@dataclass
class BoundaryResult:
    """Result of a boundary sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    scores: list[BoundaryScore] = field(default_factory=list)
    boundary_count: int = 0
    centroid_count: int = 0
    total_pool: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "scores": [s.as_dict() for s in self.scores],
            "boundary_count": self.boundary_count,
            "centroid_count": self.centroid_count,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            scores=[
                BoundaryScore.from_dict(s) for s in data.get("scores", [])
            ],
            boundary_count=data.get("boundary_count", 0),
            centroid_count=data.get("centroid_count", 0),
            total_pool=data.get("total_pool", 0),
        )


# ---------------------------------------------------------------------------
# Distance computations
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercased words."""
    return set(text.lower().split())


def compute_centroid_distance(
    item_words: set[str], centroid_words: set[str]
) -> float:
    """1 - Jaccard similarity between item and centroid word sets.

    Returns 1.0 when there is no overlap, 0.0 for identical sets.
    Returns 1.0 if both sets are empty.
    """
    if not item_words and not centroid_words:
        return 1.0
    union = item_words | centroid_words
    if not union:
        return 1.0
    intersection = item_words & centroid_words
    return 1.0 - len(intersection) / len(union)


def compute_boundary_distance(
    item_words: set[str], centroids: list[set[str]]
) -> float:
    """Distance to nearest centroid minus distance to second-nearest.

    Small values mean the item sits between two clusters (boundary).
    Returns 0.0 for fewer than 2 centroids.
    """
    if len(centroids) < 2:
        return 0.0

    distances = [compute_centroid_distance(item_words, c) for c in centroids]
    distances.sort()

    # Boundary distance = difference between nearest and second-nearest
    return distances[1] - distances[0]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _compute_centroid(texts: list[str]) -> set[str]:
    """Compute centroid as the set of words appearing in more than half the texts."""
    if not texts:
        return set()

    word_counts: dict[str, int] = {}
    for text in texts:
        for word in _tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1

    threshold = len(texts) / 2.0
    return {w for w, c in word_counts.items() if c > threshold}


def assign_clusters(
    items: dict[str, str],
    n_clusters: int,
) -> tuple[list[set[str]], dict[str, int]]:
    """Simple k-means-like clustering using word-set Jaccard distance.

    1. Evenly split items into initial clusters
    2. Compute centroids as high-frequency words per cluster
    3. Reassign each item to nearest centroid
    4. Recompute centroids after reassignment

    Returns (centroids, item_id -> cluster_id mapping).
    """
    if not items:
        return [], {}

    sorted_ids = sorted(items.keys())
    n = len(sorted_ids)
    k = min(n_clusters, n)
    if k <= 0:
        return [], {}

    # Step 1: initial even split
    assignment: dict[str, int] = {}
    for i, item_id in enumerate(sorted_ids):
        assignment[item_id] = i % k

    # Step 2: compute initial centroids
    centroids: list[set[str]] = []
    for c in range(k):
        cluster_texts = [
            items[iid] for iid in sorted_ids if assignment[iid] == c
        ]
        centroids.append(_compute_centroid(cluster_texts))

    # Step 3: reassign by nearest centroid
    for item_id in sorted_ids:
        words = _tokenize(items[item_id])
        distances = [compute_centroid_distance(words, cent) for cent in centroids]
        assignment[item_id] = distances.index(min(distances))

    # Step 4: recompute centroids after reassignment
    new_centroids: list[set[str]] = []
    for c in range(k):
        cluster_texts = [
            items[iid] for iid in sorted_ids if assignment[iid] == c
        ]
        new_centroids.append(_compute_centroid(cluster_texts))

    return new_centroids, assignment


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_boundary_items(
    items: dict[str, str],
    config: BoundaryConfig,
) -> list[BoundaryScore]:
    """Score each item by its proximity to a cluster boundary.

    Low boundary_distance means the item is between two clusters.
    """
    if not items:
        return []

    centroids, assignment = assign_clusters(items, config.n_clusters)

    scores: list[BoundaryScore] = []
    for item_id in sorted(items.keys()):
        words = _tokenize(items[item_id])
        bd = compute_boundary_distance(words, centroids)
        is_boundary = bd < config.boundary_threshold
        cluster_id = assignment.get(item_id, 0)
        scores.append(
            BoundaryScore(
                item_id=item_id,
                boundary_distance=bd,
                cluster_id=cluster_id,
                is_boundary=is_boundary,
            )
        )

    return scores


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _select_boundary_items(
    scores: list[BoundaryScore],
    sample_size: int,
    seed: int | None = None,
) -> list[str]:
    """Select items with lowest boundary distance (most ambiguous).

    Sort by boundary_distance ascending, with seeded tiebreaking.
    """
    if not scores:
        return []

    rng = random.Random(seed)
    keyed = []
    for s in scores:
        tiebreaker = rng.random()
        keyed.append((s.boundary_distance, tiebreaker, s.item_id))

    # Sort ascending by boundary distance, then by tiebreaker
    keyed.sort(key=lambda x: (x[0], x[1]))

    k = min(sample_size, len(keyed))
    return [item_id for _, _, item_id in keyed[:k]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_boundary_sampling(
    items: dict[str, str],
    config: BoundaryConfig,
) -> BoundaryResult:
    """Full boundary sampling pipeline.

    1. Cluster items by word-set similarity
    2. Score each item by boundary proximity
    3. Select items closest to boundaries
    """
    if not items:
        return BoundaryResult(
            selected_ids=[],
            scores=[],
            boundary_count=0,
            centroid_count=0,
            total_pool=0,
        )

    scores = score_boundary_items(items, config)

    selected = _select_boundary_items(scores, config.sample_size, config.seed)

    boundary_count = sum(1 for s in scores if s.is_boundary)
    centroid_count = sum(1 for s in scores if not s.is_boundary)

    return BoundaryResult(
        selected_ids=selected,
        scores=scores,
        boundary_count=boundary_count,
        centroid_count=centroid_count,
        total_pool=len(items),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_boundary_report(result: BoundaryResult) -> str:
    """Format a human-readable boundary sampling report."""
    lines: list[str] = []
    lines.append("Boundary Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected count: {len(result.selected_ids)}")
    lines.append(f"Boundary items: {result.boundary_count}")
    lines.append(f"Centroid items: {result.centroid_count}")
    lines.append("")
    lines.append("Closest to boundary:")
    top = sorted(result.scores, key=lambda s: s.boundary_distance)[:10]
    for s in top:
        label = "boundary" if s.is_boundary else "centroid"
        lines.append(
            f"  {s.item_id}: dist={s.boundary_distance:.3f} "
            f"cluster={s.cluster_id} [{label}]"
        )
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
