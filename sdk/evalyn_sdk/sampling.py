"""Sampling modes for dataset construction.

Pure functions: List[DatasetItem] -> List[DatasetItem].
Embedding-dependent modes (diverse, clustered, deduplicate) lazily import
sentence-transformers and fall back to random sampling if unavailable.
"""

from __future__ import annotations

import json
import math
import random as _random

from .models import DatasetItem

SAMPLING_MODES = ("all", "random", "diverse", "stratified", "clustered")


def input_to_text(item: DatasetItem) -> str:
    """Serialize a DatasetItem's input dict to a stable text string."""
    return json.dumps(item.input, sort_keys=True, default=str)


def _get_embeddings(texts: list[str]) -> numpy.ndarray:
    """Compute embeddings using SentenceTransformer. Raises ImportError if unavailable."""
    from sentence_transformers import SentenceTransformer

    print("Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=False)


def _cosine_similarity_matrix(embeddings: numpy.ndarray) -> numpy.ndarray:
    """Compute pairwise cosine similarity matrix."""
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def deduplicate(
    items: list[DatasetItem],
    threshold: float = 0.95,
) -> list[DatasetItem]:
    """Remove near-duplicate inputs based on cosine similarity.

    Greedy: iterate in order, drop any item whose similarity to an already-kept
    item exceeds the threshold. Falls back to identity if deps unavailable.
    """
    if len(items) <= 1:
        return list(items)

    try:
        texts = [input_to_text(it) for it in items]
        embeddings = _get_embeddings(texts)
        sim = _cosine_similarity_matrix(embeddings)
    except ImportError:
        print("Warning: sentence-transformers not installed, skipping deduplication")
        return list(items)

    keep_indices: list[int] = []
    for i in range(len(items)):
        if not any(sim[i, j] > threshold for j in keep_indices):
            keep_indices.append(i)

    removed = len(items) - len(keep_indices)
    if removed > 0:
        print(f"Deduplicated: removed {removed} near-duplicate items")
    return [items[i] for i in keep_indices]


def sample_random(
    items: list[DatasetItem],
    limit: int,
    seed: int = 42,
) -> list[DatasetItem]:
    """Uniform random sample."""
    if len(items) <= limit:
        return list(items)
    rng = _random.Random(seed)
    return rng.sample(items, limit)


def sample_diverse(
    items: list[DatasetItem],
    limit: int,
    seed: int = 42,
) -> list[DatasetItem]:
    """Farthest-point sampling on input embeddings.

    Picks a random seed, then iteratively adds the point with the maximum
    minimum distance to all already-selected points. O(n * limit).
    Falls back to random sampling if deps unavailable.
    """
    if len(items) <= limit:
        return list(items)

    try:
        import numpy as np

        texts = [input_to_text(it) for it in items]
        embeddings = _get_embeddings(texts)
    except ImportError:
        print("Warning: sentence-transformers not installed, falling back to random sampling")
        return sample_random(items, limit)

    n = len(items)
    # Precompute pairwise distance matrix (1 - cosine_sim)
    sim = _cosine_similarity_matrix(embeddings)
    dist = 1.0 - sim

    # Start with a random seed
    rng = _random.Random(seed)
    seed_idx = rng.randrange(n)
    selected: list[int] = [seed_idx]

    # min_dist[i] = min distance from i to any selected point
    min_dist = dist[seed_idx].copy()

    for _ in range(limit - 1):
        # Pick the point farthest from all selected
        # Mask already-selected by setting their min_dist to -1
        for s in selected:
            min_dist[s] = -1.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        # Update min distances
        np.minimum(min_dist, dist[next_idx], out=min_dist)

    return [items[i] for i in selected]


def _duration_bucket(duration_ms) -> str:
    """Classify a duration value into a speed bucket."""
    if duration_ms is None:
        return "unknown"
    try:
        d = float(duration_ms)
    except (TypeError, ValueError):
        return "unknown"
    if d < 100:
        return "fast"
    if d < 1000:
        return "medium"
    if d < 5000:
        return "slow"
    return "very_slow"


def _stratification_key(item: DatasetItem) -> str:
    """Group key: function name + duration bucket + input key signature."""
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    func = meta.get("function", "unknown")
    bucket = _duration_bucket(meta.get("duration_ms"))
    input_keys = sorted(item.input.keys()) if isinstance(item.input, dict) else []
    return f"{func}|{bucket}|{','.join(input_keys)}"


def sample_stratified(
    items: list[DatasetItem],
    limit: int,
    seed: int = 42,
) -> list[DatasetItem]:
    """Proportional stratified sampling by function + duration bucket + input keys."""
    if len(items) <= limit:
        return list(items)

    rng = _random.Random(seed)

    # Group items
    groups: dict[str, list[DatasetItem]] = {}
    for item in items:
        key = _stratification_key(item)
        groups.setdefault(key, []).append(item)

    # Proportional allocation
    total = len(items)
    result: list[DatasetItem] = []
    remaining = limit

    sorted_keys = sorted(groups.keys())
    allocations: list[tuple[str, int]] = []
    for key in sorted_keys:
        group = groups[key]
        alloc = max(1, math.floor(len(group) / total * limit))
        alloc = min(alloc, len(group), remaining)
        allocations.append((key, alloc))
        remaining -= alloc
    if remaining < 0:
        remaining = 0

    # Distribute remaining slots round-robin to largest groups
    if remaining > 0:
        by_size = sorted(
            allocations, key=lambda x: len(groups[x[0]]), reverse=True
        )
        new_allocs = dict(allocations)
        for i in range(remaining):
            key = by_size[i % len(by_size)][0]
            if new_allocs[key] < len(groups[key]):
                new_allocs[key] += 1
        allocations = [(k, new_allocs[k]) for k, _ in allocations]

    for key, alloc in allocations:
        group = groups[key]
        if alloc >= len(group):
            result.extend(group)
        else:
            result.extend(rng.sample(group, alloc))

    return result[:limit]


def sample_clustered(
    items: list[DatasetItem],
    limit: int,
    n_clusters: int = 10,
) -> list[DatasetItem]:
    """KMeans on embeddings, equal sample per cluster.

    Falls back to random sampling if deps unavailable.
    """
    if len(items) <= limit:
        return list(items)

    try:
        import numpy as np
        from sklearn.cluster import KMeans

        texts = [input_to_text(it) for it in items]
        embeddings = _get_embeddings(texts)
    except ImportError:
        print("Warning: sentence-transformers/sklearn not installed, falling back to random sampling")
        return sample_random(items, limit)

    actual_clusters = min(n_clusters, len(items))
    km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Group by cluster
    cluster_items: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        cluster_items.setdefault(int(label), []).append(i)

    # Equal allocation per cluster
    per_cluster = max(1, limit // actual_clusters)
    rng = _random.Random(42)

    result_indices: list[int] = []
    for cluster_id in sorted(cluster_items.keys()):
        indices = cluster_items[cluster_id]
        take = min(per_cluster, len(indices))
        result_indices.extend(rng.sample(indices, take))

    # If we have room left, fill from remaining items
    selected_set = set(result_indices)
    if len(result_indices) < limit:
        remaining = [i for i in range(len(items)) if i not in selected_set]
        rng.shuffle(remaining)
        result_indices.extend(remaining[: limit - len(result_indices)])

    return [items[i] for i in result_indices[:limit]]


def apply_sampling(
    items: list[DatasetItem],
    mode: str = "all",
    limit: int = 500,
    *,
    dedup: bool = False,
    dedup_threshold: float = 0.95,
    n_clusters: int = 10,
    seed: int = 42,
) -> list[DatasetItem]:
    """Orchestrator: apply deduplication then sampling mode.

    Args:
        items: input dataset items (already filtered by project/version/etc)
        mode: one of 'all', 'random', 'diverse', 'stratified', 'clustered'
        limit: max items to return
        dedup: whether to deduplicate before sampling
        dedup_threshold: cosine similarity threshold for dedup
        n_clusters: number of clusters for 'clustered' mode
        seed: random seed for reproducibility
    """
    if mode not in SAMPLING_MODES:
        print(f"Warning: unknown sampling mode '{mode}', using 'all'")
        mode = "all"

    if dedup:
        items = deduplicate(items, threshold=dedup_threshold)

    if mode == "random":
        return sample_random(items, limit, seed=seed)
    if mode == "diverse":
        return sample_diverse(items, limit, seed=seed)
    if mode == "stratified":
        return sample_stratified(items, limit, seed=seed)
    if mode == "clustered":
        return sample_clustered(items, limit, n_clusters=n_clusters)
    return items[:limit]
