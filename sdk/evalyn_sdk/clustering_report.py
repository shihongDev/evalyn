"""
Dataset item clustering report: word-set Jaccard k-means clustering.

Groups text items into natural clusters using word-set features,
extracts keywords, finds representatives, and generates human-readable reports.
Pure Python with no external dependencies.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Stopwords (minimal set for clustering)
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "these", "those",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ClusterDescription:
    """Description of a single cluster."""

    cluster_id: int
    size: int
    top_keywords: List[str] = field(default_factory=list)
    description: str = ""
    representative_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "top_keywords": list(self.top_keywords),
            "description": self.description,
            "representative_ids": list(self.representative_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClusterDescription:
        return cls(
            cluster_id=data.get("cluster_id", 0),
            size=data.get("size", 0),
            top_keywords=data.get("top_keywords", []),
            description=data.get("description", ""),
            representative_ids=data.get("representative_ids", []),
        )


@dataclass
class ClusteringReportConfig:
    """Configuration for clustering report generation."""

    n_clusters: int = 5
    top_keywords: int = 10
    representatives_per_cluster: int = 3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "top_keywords": self.top_keywords,
            "representatives_per_cluster": self.representatives_per_cluster,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClusteringReportConfig:
        return cls(
            n_clusters=data.get("n_clusters", 5),
            top_keywords=data.get("top_keywords", 10),
            representatives_per_cluster=data.get("representatives_per_cluster", 3),
        )


@dataclass
class ClusteringReport:
    """Complete clustering report."""

    clusters: List[ClusterDescription] = field(default_factory=list)
    total_items: int = 0
    n_clusters: int = 0
    silhouette_hint: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clusters": [c.as_dict() for c in self.clusters],
            "total_items": self.total_items,
            "n_clusters": self.n_clusters,
            "silhouette_hint": self.silhouette_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClusteringReport:
        return cls(
            clusters=[ClusterDescription.from_dict(c) for c in data.get("clusters", [])],
            total_items=data.get("total_items", 0),
            n_clusters=data.get("n_clusters", 0),
            silhouette_hint=data.get("silhouette_hint", 0.0),
        )


# ---------------------------------------------------------------------------
# Word-set helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase non-stopword words."""
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """Jaccard distance (1 - similarity)."""
    return 1.0 - _jaccard(a, b)


def _centroid_words(word_sets: List[Set[str]]) -> Set[str]:
    """Compute centroid as the set of words appearing in >50% of items."""
    if not word_sets:
        return set()
    counts: Counter = Counter()
    for ws in word_sets:
        for w in ws:
            counts[w] += 1
    threshold = len(word_sets) / 2.0
    return {w for w, c in counts.items() if c > threshold}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_items(items: Dict[str, str], n_clusters: int = 5) -> Dict[str, int]:
    """K-means-like clustering using word-set Jaccard distance.

    Args:
        items: Mapping of item_id to text content.
        n_clusters: Number of clusters to create.

    Returns:
        Mapping of item_id to cluster_id (0-indexed).
    """
    if not items:
        return {}

    ids = list(items.keys())
    word_sets = [_tokenize(items[id_]) for id_ in ids]
    n = len(ids)

    # Clamp n_clusters
    k = min(n_clusters, n)
    if k <= 0:
        k = 1

    # Initialize centroids by picking k spread-out items
    rng = random.Random(42)
    initial_indices = rng.sample(range(n), k) if n >= k else list(range(n))
    centroids = [word_sets[i].copy() for i in initial_indices]

    assignments = [0] * n
    max_iterations = 20

    for _ in range(max_iterations):
        # Assign each item to nearest centroid
        new_assignments = [0] * n
        for i in range(n):
            best_cluster = 0
            best_dist = float("inf")
            for c in range(k):
                dist = _jaccard_distance(word_sets[i], centroids[c])
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c
            new_assignments[i] = best_cluster

        # Check convergence
        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Recompute centroids
        for c in range(k):
            cluster_sets = [word_sets[i] for i in range(n) if assignments[i] == c]
            if cluster_sets:
                centroids[c] = _centroid_words(cluster_sets)

    return {ids[i]: assignments[i] for i in range(n)}


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def extract_cluster_keywords(
    items: Dict[str, str],
    assignments: Dict[str, int],
    cluster_id: int,
    top_n: int = 10,
) -> List[str]:
    """Extract the most frequent words in a cluster.

    Args:
        items: Mapping of item_id to text content.
        assignments: Mapping of item_id to cluster_id.
        cluster_id: Which cluster to extract keywords for.
        top_n: Number of top keywords to return.

    Returns:
        List of keywords sorted by frequency (descending).
    """
    word_counts: Counter = Counter()
    for item_id, cid in assignments.items():
        if cid != cluster_id:
            continue
        text = items.get(item_id, "")
        words = _tokenize(text)
        word_counts.update(words)

    return [word for word, _ in word_counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


def find_representatives(
    items: Dict[str, str],
    assignments: Dict[str, int],
    cluster_id: int,
    n: int = 3,
) -> List[str]:
    """Find items closest to the cluster centroid.

    Args:
        items: Mapping of item_id to text content.
        assignments: Mapping of item_id to cluster_id.
        cluster_id: Which cluster to find representatives for.
        n: Number of representatives to return.

    Returns:
        List of item_ids closest to centroid.
    """
    cluster_ids = [iid for iid, cid in assignments.items() if cid == cluster_id]
    if not cluster_ids:
        return []

    cluster_sets = {iid: _tokenize(items.get(iid, "")) for iid in cluster_ids}
    centroid = _centroid_words(list(cluster_sets.values()))

    scored = []
    for iid in cluster_ids:
        dist = _jaccard_distance(cluster_sets[iid], centroid)
        scored.append((dist, iid))

    scored.sort(key=lambda x: x[0])
    return [iid for _, iid in scored[:n]]


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------


def generate_cluster_description(keywords: List[str]) -> str:
    """Generate a template-based description from keywords.

    Returns a string like: "Items about kw1, kw2, and kw3"
    """
    if not keywords:
        return "Items with no distinguishing keywords"
    if len(keywords) == 1:
        return f"Items about {keywords[0]}"
    if len(keywords) == 2:
        return f"Items about {keywords[0]} and {keywords[1]}"
    top = keywords[:3]
    return f"Items about {top[0]}, {top[1]}, and {top[2]}"


# ---------------------------------------------------------------------------
# Silhouette hint
# ---------------------------------------------------------------------------


def _compute_silhouette_hint(
    items: Dict[str, str],
    assignments: Dict[str, int],
) -> float:
    """Compute a simplified silhouette score hint.

    Uses Jaccard distance between items and their cluster vs other clusters.
    Returns average silhouette in [-1, 1]. Higher is better.
    """
    if not items or not assignments:
        return 0.0

    ids = list(assignments.keys())
    word_sets = {iid: _tokenize(items.get(iid, "")) for iid in ids}
    cluster_ids_set = set(assignments.values())

    if len(cluster_ids_set) < 2:
        return 0.0

    silhouettes: List[float] = []
    for iid in ids:
        own_cluster = assignments[iid]
        own_members = [j for j in ids if j != iid and assignments[j] == own_cluster]

        if not own_members:
            silhouettes.append(0.0)
            continue

        # a(i): mean distance to own cluster
        a_i = sum(_jaccard_distance(word_sets[iid], word_sets[j]) for j in own_members) / len(own_members)

        # b(i): min mean distance to another cluster
        b_i = float("inf")
        for cid in cluster_ids_set:
            if cid == own_cluster:
                continue
            other_members = [j for j in ids if assignments[j] == cid]
            if not other_members:
                continue
            mean_dist = sum(_jaccard_distance(word_sets[iid], word_sets[j]) for j in other_members) / len(other_members)
            if mean_dist < b_i:
                b_i = mean_dist

        if b_i == float("inf"):
            silhouettes.append(0.0)
            continue

        denom = max(a_i, b_i)
        if denom == 0:
            silhouettes.append(0.0)
        else:
            silhouettes.append((b_i - a_i) / denom)

    if not silhouettes:
        return 0.0
    return sum(silhouettes) / len(silhouettes)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_clustering_report(
    items: Dict[str, str],
    config: ClusteringReportConfig | None = None,
) -> ClusteringReport:
    """Build a complete clustering report.

    Args:
        items: Mapping of item_id to text content.
        config: Optional configuration. Uses defaults if None.

    Returns:
        ClusteringReport with cluster descriptions.
    """
    if config is None:
        config = ClusteringReportConfig()

    if not items:
        return ClusteringReport(
            clusters=[],
            total_items=0,
            n_clusters=0,
            silhouette_hint=0.0,
        )

    assignments = cluster_items(items, n_clusters=config.n_clusters)
    actual_clusters = set(assignments.values())

    cluster_descriptions: List[ClusterDescription] = []
    for cid in sorted(actual_clusters):
        keywords = extract_cluster_keywords(items, assignments, cid, top_n=config.top_keywords)
        reps = find_representatives(items, assignments, cid, n=config.representatives_per_cluster)
        desc = generate_cluster_description(keywords)
        size = sum(1 for v in assignments.values() if v == cid)

        cluster_descriptions.append(ClusterDescription(
            cluster_id=cid,
            size=size,
            top_keywords=keywords,
            description=desc,
            representative_ids=reps,
        ))

    silhouette = _compute_silhouette_hint(items, assignments)

    return ClusteringReport(
        clusters=cluster_descriptions,
        total_items=len(items),
        n_clusters=len(actual_clusters),
        silhouette_hint=round(silhouette, 4),
    )


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


def format_clustering_report(report: ClusteringReport) -> str:
    """Format a ClusteringReport for human-readable display."""
    lines: List[str] = []
    lines.append("Clustering Report")
    lines.append("=" * 40)
    lines.append(f"Total items: {report.total_items}")
    lines.append(f"Clusters: {report.n_clusters}")
    lines.append(f"Silhouette hint: {report.silhouette_hint:.4f}")
    lines.append("")

    for cluster in report.clusters:
        lines.append(f"Cluster {cluster.cluster_id} ({cluster.size} items)")
        lines.append("-" * 30)
        lines.append(f"  Description: {cluster.description}")
        if cluster.top_keywords:
            lines.append(f"  Keywords: {', '.join(cluster.top_keywords)}")
        if cluster.representative_ids:
            lines.append(f"  Representatives: {', '.join(cluster.representative_ids)}")
        lines.append("")

    return "\n".join(lines)
