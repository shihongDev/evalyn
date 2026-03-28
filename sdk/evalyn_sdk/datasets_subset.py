"""Dataset subset extraction via clustering.

Groups dataset items by keywords, input length, or random assignment,
then allows sampling representative subsets from each cluster.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "above", "below", "between",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "else", "up", "out",
})


@dataclass
class ClusterInfo:
    """Metadata about a single cluster."""

    cluster_id: int = 0
    size: int = 0
    centroid_item_id: str = ""
    keywords: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "centroid_item_id": self.centroid_item_id,
            "keywords": self.keywords,
        }


@dataclass
class SubsetConfig:
    """Configuration for subset extraction."""

    num_clusters: int = 5
    method: str = "keyword"  # "keyword", "length", or "random"
    items_per_cluster: int = 0  # 0 means all
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_clusters": self.num_clusters,
            "method": self.method,
            "items_per_cluster": self.items_per_cluster,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubsetConfig:
        return cls(
            num_clusters=data.get("num_clusters", 5),
            method=data.get("method", "keyword"),
            items_per_cluster=data.get("items_per_cluster", 0),
            seed=data.get("seed"),
        )


@dataclass
class SubsetResult:
    """Result of subset extraction with cluster assignments."""

    subsets: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    clusters: List[ClusterInfo] = field(default_factory=list)
    total_items: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subsets": {
                str(k): v for k, v in self.subsets.items()
            },
            "clusters": [c.as_dict() for c in self.clusters],
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubsetResult:
        subsets = {}
        for k, v in data.get("subsets", {}).items():
            subsets[int(k)] = v
        return cls(
            subsets=subsets,
            clusters=[
                ClusterInfo(
                    cluster_id=c.get("cluster_id", 0),
                    size=c.get("size", 0),
                    centroid_item_id=c.get("centroid_item_id", ""),
                    keywords=c.get("keywords", []),
                )
                for c in data.get("clusters", [])
            ],
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Subset extraction: {len(self.clusters)} clusters, {self.total_items} total items",
        ]
        for cluster in self.clusters:
            kw = ", ".join(cluster.keywords[:5]) if cluster.keywords else "none"
            lines.append(
                f"  Cluster {cluster.cluster_id}: {cluster.size} items - keywords: {kw}"
            )
        return "\n".join(lines)


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens, filtering stop words and short tokens."""
    words = text.lower().split()
    # Strip punctuation from edges
    cleaned = []
    for w in words:
        w = w.strip(".,;:!?\"'()[]{}/-")
        if len(w) > 2 and w not in _STOP_WORDS:
            cleaned.append(w)
    return cleaned


def compute_cluster_keywords(
    items: List[Dict[str, Any]], input_field: str = "input", top_n: int = 5
) -> List[str]:
    """Extract the top keywords from a list of items."""
    counter: Counter = Counter()
    for item in items:
        text = str(item.get(input_field, ""))
        tokens = _tokenize(text)
        counter.update(tokens)
    return [word for word, _ in counter.most_common(top_n)]


def cluster_by_keywords(
    items: List[Dict[str, Any]], num_clusters: int = 5, input_field: str = "input"
) -> SubsetResult:
    """Cluster items by shared keywords.

    Extracts the most common words across all items and assigns each item
    to the cluster whose keyword appears most frequently in its text.
    """
    if not items:
        return SubsetResult(total_items=0)

    num_clusters = min(num_clusters, len(items))

    # Find top keywords across all items
    global_counter: Counter = Counter()
    for item in items:
        text = str(item.get(input_field, ""))
        tokens = _tokenize(text)
        # Count unique tokens per item to avoid length bias
        global_counter.update(set(tokens))

    top_keywords = [word for word, _ in global_counter.most_common(num_clusters)]

    # If we have fewer unique keywords than clusters, pad with generic labels
    while len(top_keywords) < num_clusters:
        top_keywords.append(f"cluster_{len(top_keywords)}")

    # Assign each item to the cluster whose keyword appears most in its text
    subsets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_clusters)}

    for item in items:
        text = str(item.get(input_field, "")).lower()
        best_cluster = 0
        best_count = -1
        for ci, kw in enumerate(top_keywords):
            count = text.count(kw)
            if count > best_count:
                best_count = count
                best_cluster = ci
        subsets[best_cluster].append(item)

    clusters = []
    for ci in range(num_clusters):
        cluster_items = subsets[ci]
        centroid_id = ""
        if cluster_items:
            centroid_id = str(cluster_items[0].get("id", ""))
        kw_list = compute_cluster_keywords(cluster_items, input_field=input_field)
        clusters.append(
            ClusterInfo(
                cluster_id=ci,
                size=len(cluster_items),
                centroid_item_id=centroid_id,
                keywords=[top_keywords[ci]] + [k for k in kw_list if k != top_keywords[ci]][:4],
            )
        )

    return SubsetResult(subsets=subsets, clusters=clusters, total_items=len(items))


def cluster_by_length(
    items: List[Dict[str, Any]], num_clusters: int = 5, input_field: str = "input"
) -> SubsetResult:
    """Cluster items by input text length into equal-width buckets."""
    if not items:
        return SubsetResult(total_items=0)

    num_clusters = min(num_clusters, len(items))

    lengths = [len(str(item.get(input_field, ""))) for item in items]
    min_len = min(lengths)
    max_len = max(lengths)

    # Avoid division by zero when all items have same length
    bucket_width = (max_len - min_len + 1) / num_clusters if max_len > min_len else 1

    subsets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_clusters)}

    for item, length in zip(items, lengths):
        if max_len == min_len:
            bucket = 0
        else:
            bucket = int((length - min_len) / bucket_width)
            bucket = min(bucket, num_clusters - 1)
        subsets[bucket].append(item)

    clusters = []
    for ci in range(num_clusters):
        cluster_items = subsets[ci]
        centroid_id = ""
        if cluster_items:
            centroid_id = str(cluster_items[0].get("id", ""))
        low = int(min_len + ci * bucket_width)
        high = int(min_len + (ci + 1) * bucket_width - 1)
        clusters.append(
            ClusterInfo(
                cluster_id=ci,
                size=len(cluster_items),
                centroid_item_id=centroid_id,
                keywords=[f"len:{low}-{high}"],
            )
        )

    return SubsetResult(subsets=subsets, clusters=clusters, total_items=len(items))


def extract_subset(
    items: List[Dict[str, Any]], config: SubsetConfig
) -> SubsetResult:
    """Route to the appropriate clustering method based on config."""
    if config.seed is not None:
        random.seed(config.seed)

    if config.method == "keyword":
        result = cluster_by_keywords(items, num_clusters=config.num_clusters)
    elif config.method == "length":
        result = cluster_by_length(items, num_clusters=config.num_clusters)
    elif config.method == "random":
        result = _cluster_random(items, num_clusters=config.num_clusters, seed=config.seed)
    else:
        result = cluster_by_keywords(items, num_clusters=config.num_clusters)

    if config.items_per_cluster > 0:
        sampled_items = sample_from_clusters(result, config.items_per_cluster, seed=config.seed)
        # Rebuild subsets with sampled items
        new_subsets: Dict[int, List[Dict[str, Any]]] = {}
        for ci, cluster_items in result.subsets.items():
            sampled_ids = {str(s.get("id", "")) for s in sampled_items}
            new_subsets[ci] = [item for item in cluster_items if str(item.get("id", "")) in sampled_ids]
        result.subsets = new_subsets
        for cluster in result.clusters:
            cluster.size = len(result.subsets.get(cluster.cluster_id, []))

    return result


def _cluster_random(
    items: List[Dict[str, Any]], num_clusters: int = 5, seed: Optional[int] = None
) -> SubsetResult:
    """Assign items to clusters randomly."""
    if not items:
        return SubsetResult(total_items=0)

    num_clusters = min(num_clusters, len(items))
    rng = random.Random(seed)

    subsets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_clusters)}
    for item in items:
        ci = rng.randint(0, num_clusters - 1)
        subsets[ci].append(item)

    clusters = []
    for ci in range(num_clusters):
        cluster_items = subsets[ci]
        centroid_id = ""
        if cluster_items:
            centroid_id = str(cluster_items[0].get("id", ""))
        clusters.append(
            ClusterInfo(
                cluster_id=ci,
                size=len(cluster_items),
                centroid_item_id=centroid_id,
                keywords=["random"],
            )
        )

    return SubsetResult(subsets=subsets, clusters=clusters, total_items=len(items))


def sample_from_clusters(
    result: SubsetResult, items_per_cluster: int = 3, seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Sample items from each cluster for a representative subset."""
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []

    for ci in sorted(result.subsets.keys()):
        cluster_items = result.subsets[ci]
        if not cluster_items:
            continue
        n = min(items_per_cluster, len(cluster_items))
        chosen = rng.sample(cluster_items, n)
        sampled.extend(chosen)

    return sampled
