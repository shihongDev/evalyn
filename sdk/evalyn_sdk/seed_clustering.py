"""
Seed clustering: group seeds before simulation to ensure diverse coverage.

Pure Python, no external dependencies. Uses word-set Jaccard similarity
as a lightweight proxy for embedding-based clustering.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "be",
    "this", "that", "not", "has", "had", "have", "been", "will", "can",
    "do", "does", "did", "if", "so", "no", "up", "out", "its", "all",
    "my", "we", "he", "she", "they", "you", "me", "us", "him", "her",
    "who", "what", "when", "how", "which", "where", "there", "here",
    "then", "than", "also", "just", "about", "into", "over", "after",
    "more", "some", "any", "each", "very", "too", "own", "same",
})


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on whitespace, filter short words and stop words."""
    return {
        w for w in text.lower().split()
        if len(w) >= 3 and w not in _STOP_WORDS
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SeedCluster:
    """A cluster of seed items grouped by word-set similarity."""

    cluster_id: int
    seed_ids: List[str] = field(default_factory=list)
    centroid_keywords: List[str] = field(default_factory=list)
    size: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "seed_ids": list(self.seed_ids),
            "centroid_keywords": list(self.centroid_keywords),
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SeedCluster:
        return cls(
            cluster_id=data["cluster_id"],
            seed_ids=data["seed_ids"],
            centroid_keywords=data["centroid_keywords"],
            size=data["size"],
        )


@dataclass
class ClusteringResult:
    """Result of seed clustering."""

    clusters: List[SeedCluster] = field(default_factory=list)
    total_seeds: int = 0
    n_clusters: int = 0
    silhouette_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clusters": [c.as_dict() for c in self.clusters],
            "total_seeds": self.total_seeds,
            "n_clusters": self.n_clusters,
            "silhouette_score": self.silhouette_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClusteringResult:
        return cls(
            clusters=[SeedCluster.from_dict(c) for c in data["clusters"]],
            total_seeds=data["total_seeds"],
            n_clusters=data["n_clusters"],
            silhouette_score=data["silhouette_score"],
        )


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def _compute_centroid(word_sets: list[set[str]], top_k: int = 10) -> set[str]:
    """Compute centroid as the top-k most frequent words across word sets."""
    counter: Counter[str] = Counter()
    for ws in word_sets:
        counter.update(ws)
    return {w for w, _ in counter.most_common(top_k)}


def cluster_seeds(
    seeds: dict[str, str],
    n_clusters: int = 5,
    max_iterations: int = 10,
) -> ClusteringResult:
    """Cluster seed texts by word-set similarity using a k-means-like algorithm.

    Steps:
    1. Initialize centroids by picking n evenly-spaced seeds.
    2. Assign each seed to nearest centroid by Jaccard similarity.
    3. Update centroids as union of top-k frequent words.
    4. Repeat for max_iterations or until stable.

    Args:
        seeds: mapping seed_id -> seed text
        n_clusters: desired number of clusters
        max_iterations: convergence limit

    Returns:
        ClusteringResult with cluster assignments.
    """
    if not seeds:
        return ClusteringResult(
            clusters=[],
            total_seeds=0,
            n_clusters=0,
            silhouette_score=0.0,
        )

    seed_ids = list(seeds.keys())
    n = len(seed_ids)

    # Clamp cluster count
    actual_k = min(n_clusters, n)
    if actual_k <= 0:
        actual_k = 1

    # Precompute word sets
    word_sets: dict[str, set[str]] = {
        sid: _tokenize(seeds[sid]) for sid in seed_ids
    }

    # Step 1: Initialize centroids by picking evenly-spaced seeds
    step = max(1, n // actual_k)
    centroids: list[set[str]] = []
    for i in range(actual_k):
        idx = min(i * step, n - 1)
        centroids.append(set(word_sets[seed_ids[idx]]))

    assignments: list[int] = [0] * n

    for _iteration in range(max_iterations):
        # Step 2: Assign each seed to nearest centroid
        new_assignments: list[int] = []
        for i, sid in enumerate(seed_ids):
            ws = word_sets[sid]
            best_cluster = 0
            best_sim = -1.0
            for c_idx, centroid in enumerate(centroids):
                sim = _jaccard_similarity(ws, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c_idx
            new_assignments.append(best_cluster)

        # Check for convergence
        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Step 3: Update centroids
        for c_idx in range(actual_k):
            members = [
                word_sets[seed_ids[i]]
                for i in range(n)
                if assignments[i] == c_idx
            ]
            if members:
                centroids[c_idx] = _compute_centroid(members)

    # Build cluster objects
    cluster_members: dict[int, list[str]] = {i: [] for i in range(actual_k)}
    for i, sid in enumerate(seed_ids):
        cluster_members[assignments[i]].append(sid)

    clusters: list[SeedCluster] = []
    for c_idx in range(actual_k):
        members = cluster_members[c_idx]
        if not members:
            continue
        clusters.append(SeedCluster(
            cluster_id=c_idx,
            seed_ids=members,
            centroid_keywords=sorted(centroids[c_idx])[:10],
            size=len(members),
        ))

    # Compute silhouette score
    sil = _compute_silhouette(seed_ids, word_sets, assignments, actual_k)

    return ClusteringResult(
        clusters=clusters,
        total_seeds=n,
        n_clusters=len(clusters),
        silhouette_score=sil,
    )


def _compute_silhouette(
    seed_ids: list[str],
    word_sets: dict[str, set[str]],
    assignments: list[int],
    k: int,
) -> float:
    """Compute simplified silhouette score based on Jaccard distance."""
    n = len(seed_ids)
    if n <= 1 or k <= 1:
        return 0.0

    silhouettes: list[float] = []

    for i in range(n):
        ws_i = word_sets[seed_ids[i]]
        c_i = assignments[i]

        # a(i): average distance to other points in same cluster
        same_dists: list[float] = []
        for j in range(n):
            if j != i and assignments[j] == c_i:
                same_dists.append(1.0 - _jaccard_similarity(ws_i, word_sets[seed_ids[j]]))

        if not same_dists:
            silhouettes.append(0.0)
            continue

        a_i = sum(same_dists) / len(same_dists)

        # b(i): minimum average distance to any other cluster
        b_i = float("inf")
        for c_idx in range(k):
            if c_idx == c_i:
                continue
            other_dists: list[float] = []
            for j in range(n):
                if assignments[j] == c_idx:
                    other_dists.append(1.0 - _jaccard_similarity(ws_i, word_sets[seed_ids[j]]))
            if other_dists:
                avg_dist = sum(other_dists) / len(other_dists)
                if avg_dist < b_i:
                    b_i = avg_dist

        if b_i == float("inf"):
            silhouettes.append(0.0)
            continue

        max_ab = max(a_i, b_i)
        if max_ab == 0:
            silhouettes.append(0.0)
        else:
            silhouettes.append((b_i - a_i) / max_ab)

    if not silhouettes:
        return 0.0
    return sum(silhouettes) / len(silhouettes)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_from_clusters(
    result: ClusteringResult,
    total_samples: int,
    seeds: dict[str, str],
) -> list[str]:
    """Proportional sampling: allocate samples to clusters by cluster size.

    Args:
        result: clustering result
        total_samples: total number of seed IDs to return
        seeds: original seed dict (used for seed IDs)

    Returns:
        List of seed_ids sampled proportionally from each cluster.
    """
    if not result.clusters or total_samples <= 0:
        return []

    total = result.total_seeds
    if total == 0:
        return []

    sampled: list[str] = []

    # Allocate proportional counts, ensuring at least 1 per cluster if possible
    remaining = total_samples
    allocations: list[int] = []
    for cluster in result.clusters:
        proportion = cluster.size / total
        count = max(1, round(proportion * total_samples))
        allocations.append(count)

    # Adjust if total allocations exceed total_samples
    alloc_sum = sum(allocations)
    if alloc_sum > total_samples:
        # Scale down - remove from largest allocations first
        excess = alloc_sum - total_samples
        sorted_indices = sorted(
            range(len(allocations)),
            key=lambda i: allocations[i],
            reverse=True,
        )
        for idx in sorted_indices:
            if excess <= 0:
                break
            reduce = min(allocations[idx] - 1, excess)
            allocations[idx] -= reduce
            excess -= reduce

    for cluster, count in zip(result.clusters, allocations):
        available = list(cluster.seed_ids)
        take = min(count, len(available))
        sampled.extend(available[:take])

    # Trim to exact count
    return sampled[:total_samples]


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def compute_cluster_diversity(
    result: ClusteringResult,
    seeds: dict[str, str],
) -> float:
    """Average inter-cluster Jaccard distance.

    Measures how different clusters are from each other. Higher is more diverse.
    Returns 0.0 if fewer than 2 clusters.
    """
    if len(result.clusters) < 2:
        return 0.0

    # Compute word set per cluster (union of all seed word sets)
    cluster_word_sets: list[set[str]] = []
    for cluster in result.clusters:
        combined: set[str] = set()
        for sid in cluster.seed_ids:
            if sid in seeds:
                combined |= _tokenize(seeds[sid])
        cluster_word_sets.append(combined)

    total_distance = 0.0
    pair_count = 0
    n = len(cluster_word_sets)

    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard_similarity(cluster_word_sets[i], cluster_word_sets[j])
            total_distance += 1.0 - sim
            pair_count += 1

    if pair_count == 0:
        return 0.0
    return total_distance / pair_count


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def find_outlier_seeds(
    result: ClusteringResult,
    seeds: dict[str, str],
    threshold: float = 0.1,
) -> list[str]:
    """Find seeds with very low similarity to their cluster centroid.

    A seed is an outlier if its Jaccard similarity to its cluster's centroid
    keywords is below the threshold.

    Args:
        result: clustering result
        seeds: original seed dict
        threshold: maximum similarity to be considered an outlier

    Returns:
        List of seed_ids that are outliers.
    """
    outliers: list[str] = []

    for cluster in result.clusters:
        centroid_set = set(cluster.centroid_keywords)
        for sid in cluster.seed_ids:
            if sid not in seeds:
                continue
            ws = _tokenize(seeds[sid])
            sim = _jaccard_similarity(ws, centroid_set)
            if sim < threshold:
                outliers.append(sid)

    return outliers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_clustering_report(result: ClusteringResult) -> str:
    """Format a human-readable clustering report."""
    lines: list[str] = []
    lines.append("SEED CLUSTERING REPORT")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Total seeds: {result.total_seeds}")
    lines.append(f"Clusters: {result.n_clusters}")
    lines.append(f"Silhouette score: {result.silhouette_score:.3f}")
    lines.append("")

    for cluster in result.clusters:
        lines.append(f"Cluster {cluster.cluster_id} ({cluster.size} seeds)")
        if cluster.centroid_keywords:
            lines.append(f"  Keywords: {', '.join(cluster.centroid_keywords)}")
        if cluster.seed_ids:
            preview = cluster.seed_ids[:5]
            lines.append(f"  Seeds: {', '.join(preview)}")
            if len(cluster.seed_ids) > 5:
                lines.append(f"  ... and {len(cluster.seed_ids) - 5} more")
        lines.append("")

    return "\n".join(lines)
