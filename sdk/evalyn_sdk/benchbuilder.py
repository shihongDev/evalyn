"""
BenchBuilder auto-curation: automatically curate evaluation prompts
from production traces.

Pure Python, no external dependencies. Uses word-set features as
lightweight proxy for embedding-based analysis.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Stop words (shared lightweight set)
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

_QUESTION_WORDS = frozenset({
    "why", "how", "what", "explain", "describe", "compare", "analyze",
    "evaluate", "discuss", "elaborate", "contrast", "justify", "prove",
    "derive", "synthesize", "critique",
})


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, return word list."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 2]


def _word_set(text: str) -> set[str]:
    """Unique content words (no stop words)."""
    return {w for w in _tokenize(text) if w not in _STOP_WORDS and len(w) >= 3}


def _sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class TraceScore:
    """Quality and difficulty scores for a single trace."""

    trace_id: str
    quality_score: float
    difficulty_score: float
    topic_cluster: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "quality_score": self.quality_score,
            "difficulty_score": self.difficulty_score,
            "topic_cluster": self.topic_cluster,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceScore:
        return cls(
            trace_id=data["trace_id"],
            quality_score=data["quality_score"],
            difficulty_score=data["difficulty_score"],
            topic_cluster=data.get("topic_cluster", -1),
        )


@dataclass
class CurationConfig:
    """Configuration for benchmark curation."""

    target_size: int = 100
    min_quality: float = 0.3
    diversity_weight: float = 0.5
    seed: int | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_size": self.target_size,
            "min_quality": self.min_quality,
            "diversity_weight": self.diversity_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurationConfig:
        return cls(
            target_size=data.get("target_size", 100),
            min_quality=data.get("min_quality", 0.3),
            diversity_weight=data.get("diversity_weight", 0.5),
            seed=data.get("seed"),
        )


@dataclass
class CurationResult:
    """Result of benchmark curation."""

    selected_ids: List[str] = field(default_factory=list)
    scores: List[TraceScore] = field(default_factory=list)
    total_traces: int = 0
    avg_quality: float = 0.0
    avg_difficulty: float = 0.0
    n_clusters_represented: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "scores": [s.as_dict() for s in self.scores],
            "total_traces": self.total_traces,
            "avg_quality": self.avg_quality,
            "avg_difficulty": self.avg_difficulty,
            "n_clusters_represented": self.n_clusters_represented,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurationResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            scores=[TraceScore.from_dict(s) for s in data.get("scores", [])],
            total_traces=data.get("total_traces", 0),
            avg_quality=data.get("avg_quality", 0.0),
            avg_difficulty=data.get("avg_difficulty", 0.0),
            n_clusters_represented=data.get("n_clusters_represented", 0),
        )


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------


def score_trace_quality(text: str) -> float:
    """Heuristic quality score for a trace text.

    Combines:
    - Length reasonableness (not too short, not too long)
    - Vocabulary diversity (unique words / total words)
    - Sentence structure (has multiple sentences)

    Args:
        text: the trace text.

    Returns:
        Quality score in [0, 1].
    """
    if not text or not text.strip():
        return 0.0

    words = _tokenize(text)
    n_words = len(words)

    # Length score: peak around 50-200 words, penalize extremes.
    if n_words < 5:
        length_score = n_words / 5.0
    elif n_words <= 200:
        length_score = 1.0
    elif n_words <= 500:
        length_score = 1.0 - (n_words - 200) / 600.0
    else:
        length_score = 0.5

    length_score = max(0.0, min(1.0, length_score))

    # Vocabulary diversity: unique / total.
    if n_words == 0:
        diversity_score = 0.0
    else:
        unique = len(set(words))
        diversity_score = min(1.0, unique / n_words)

    # Sentence structure: having 2+ sentences is good.
    sents = _sentences(text)
    n_sents = len(sents)
    if n_sents == 0:
        structure_score = 0.0
    elif n_sents == 1:
        structure_score = 0.5
    elif n_sents <= 10:
        structure_score = 1.0
    else:
        structure_score = 0.8

    # Weighted combination.
    score = 0.4 * length_score + 0.35 * diversity_score + 0.25 * structure_score
    return max(0.0, min(1.0, score))


def score_trace_difficulty(text: str) -> float:
    """Heuristic difficulty score for a trace text.

    Combines:
    - Word count (longer prompts tend to be harder)
    - Average word length (longer words suggest technical content)
    - Question complexity (presence of analytical question words)

    Args:
        text: the trace text.

    Returns:
        Difficulty score in [0, 1].
    """
    if not text or not text.strip():
        return 0.0

    words = _tokenize(text)
    n_words = len(words)

    # Word count contribution: more words = harder.
    if n_words <= 10:
        count_score = 0.1
    elif n_words <= 50:
        count_score = 0.3
    elif n_words <= 150:
        count_score = 0.5
    elif n_words <= 300:
        count_score = 0.7
    else:
        count_score = 0.9

    # Average word length: longer words = more technical.
    if n_words == 0:
        length_score = 0.0
    else:
        avg_len = sum(len(w) for w in words) / n_words
        # Map: avg_len 3 -> 0.1, avg_len 6 -> 0.5, avg_len 9+ -> 1.0
        length_score = max(0.0, min(1.0, (avg_len - 3.0) / 6.0))

    # Question complexity: presence of analytical / higher-order words.
    lower_text = text.lower()
    complexity_hits = sum(1 for w in _QUESTION_WORDS if w in lower_text)
    complexity_score = min(1.0, complexity_hits / 3.0)

    score = 0.35 * count_score + 0.35 * length_score + 0.3 * complexity_score
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Clustering (word-set k-means)
# ---------------------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def _centroid_words(word_sets: list[set[str]], top_k: int = 10) -> set[str]:
    """Compute centroid as top-k most frequent words."""
    counter: Counter[str] = Counter()
    for ws in word_sets:
        counter.update(ws)
    return {w for w, _ in counter.most_common(top_k)}


def cluster_traces(
    texts: dict[str, str],
    n_clusters: int = 5,
) -> dict[str, int]:
    """Assign traces to topic clusters using word-set k-means.

    Args:
        texts: mapping trace_id -> trace text.
        n_clusters: desired number of clusters.

    Returns:
        Mapping trace_id -> cluster index.
    """
    if not texts:
        return {}

    trace_ids = list(texts.keys())
    n = len(trace_ids)
    actual_k = max(1, min(n_clusters, n))

    # Precompute word sets.
    word_sets: dict[str, set[str]] = {
        tid: _word_set(texts[tid]) for tid in trace_ids
    }

    # Initialize centroids: evenly spaced.
    step = max(1, n // actual_k)
    centroids: list[set[str]] = []
    for i in range(actual_k):
        idx = min(i * step, n - 1)
        centroids.append(set(word_sets[trace_ids[idx]]))

    assignments: dict[str, int] = {}
    max_iterations = 10

    for _ in range(max_iterations):
        new_assignments: dict[str, int] = {}
        for tid in trace_ids:
            ws = word_sets[tid]
            best_cluster = 0
            best_sim = -1.0
            for ci, centroid in enumerate(centroids):
                sim = _jaccard(ws, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = ci
            new_assignments[tid] = best_cluster

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update centroids.
        for ci in range(actual_k):
            cluster_sets = [
                word_sets[tid]
                for tid in trace_ids
                if assignments.get(tid) == ci
            ]
            if cluster_sets:
                centroids[ci] = _centroid_words(cluster_sets)

    return assignments


# ---------------------------------------------------------------------------
# Curation Pipeline
# ---------------------------------------------------------------------------


def curate_benchmark(
    traces: dict[str, str],
    config: CurationConfig | None = None,
) -> CurationResult:
    """Full curation pipeline: score, cluster, select diverse high-quality traces.

    Steps:
    1. Score quality and difficulty for each trace.
    2. Cluster traces by topic.
    3. Filter by min_quality.
    4. Select proportionally from each cluster, preferring higher quality
       with diversity weighting.

    Args:
        traces: mapping trace_id -> trace text.
        config: curation configuration (uses defaults if None).

    Returns:
        CurationResult with selected trace IDs and metadata.
    """
    if config is None:
        config = CurationConfig()

    if not traces:
        return CurationResult(
            selected_ids=[],
            scores=[],
            total_traces=0,
            avg_quality=0.0,
            avg_difficulty=0.0,
            n_clusters_represented=0,
        )

    rng = random.Random(config.seed)

    # Step 1: Score all traces.
    all_scores: dict[str, TraceScore] = {}
    for tid, text in traces.items():
        q = score_trace_quality(text)
        d = score_trace_difficulty(text)
        all_scores[tid] = TraceScore(
            trace_id=tid,
            quality_score=q,
            difficulty_score=d,
        )

    # Step 2: Cluster.
    cluster_map = cluster_traces(traces, n_clusters=5)
    for tid, cluster_id in cluster_map.items():
        all_scores[tid].topic_cluster = cluster_id

    # Step 3: Filter by min_quality.
    candidates = {
        tid: sc
        for tid, sc in all_scores.items()
        if sc.quality_score >= config.min_quality
    }

    if not candidates:
        return CurationResult(
            selected_ids=[],
            scores=list(all_scores.values()),
            total_traces=len(traces),
            avg_quality=0.0,
            avg_difficulty=0.0,
            n_clusters_represented=0,
        )

    target = min(config.target_size, len(candidates))

    # Step 4: Group candidates by cluster.
    clusters: dict[int, list[str]] = {}
    for tid, sc in candidates.items():
        c = sc.topic_cluster
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(tid)

    # Sort within each cluster by quality (descending), with random tiebreak.
    for c in clusters:
        clusters[c].sort(
            key=lambda tid: (-candidates[tid].quality_score, rng.random()),
        )

    # Proportional selection across clusters.
    n_clusters = len(clusters)
    cluster_ids = sorted(clusters.keys())

    # Base allocation per cluster.
    base_per_cluster = max(1, target // n_clusters)
    selected_ids: list[str] = []
    selected_set: set[str] = set()

    # Round 1: take base_per_cluster from each cluster.
    for c in cluster_ids:
        pool = clusters[c]
        take = min(base_per_cluster, len(pool))
        for tid in pool[:take]:
            if len(selected_ids) >= target:
                break
            if tid not in selected_set:
                selected_ids.append(tid)
                selected_set.add(tid)

    # Round 2: fill remaining from all clusters by quality.
    if len(selected_ids) < target:
        remaining = [
            tid for tid in candidates if tid not in selected_set
        ]
        remaining.sort(
            key=lambda tid: (-candidates[tid].quality_score, rng.random()),
        )
        for tid in remaining:
            if len(selected_ids) >= target:
                break
            selected_ids.append(tid)
            selected_set.add(tid)

    # Compute summary stats.
    selected_scores = [all_scores[tid] for tid in selected_ids]
    avg_q = (
        sum(s.quality_score for s in selected_scores) / len(selected_scores)
        if selected_scores
        else 0.0
    )
    avg_d = (
        sum(s.difficulty_score for s in selected_scores) / len(selected_scores)
        if selected_scores
        else 0.0
    )
    clusters_represented = len({
        all_scores[tid].topic_cluster for tid in selected_ids
    })

    return CurationResult(
        selected_ids=selected_ids,
        scores=list(all_scores.values()),
        total_traces=len(traces),
        avg_quality=avg_q,
        avg_difficulty=avg_d,
        n_clusters_represented=clusters_represented,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_curation_report(result: CurationResult) -> str:
    """Format a human-readable curation report.

    Args:
        result: curation result.

    Returns:
        Multi-line report string.
    """
    lines: list[str] = []
    lines.append("BenchBuilder Curation Report")
    lines.append("=" * 40)
    lines.append(f"Total traces evaluated: {result.total_traces}")
    lines.append(f"Selected for benchmark: {len(result.selected_ids)}")
    lines.append(f"Average quality: {result.avg_quality:.3f}")
    lines.append(f"Average difficulty: {result.avg_difficulty:.3f}")
    lines.append(f"Clusters represented: {result.n_clusters_represented}")
    lines.append("")

    if result.selected_ids:
        lines.append("Selected Trace IDs:")
        lines.append("-" * 40)
        for i, tid in enumerate(result.selected_ids, 1):
            # Find score for this trace.
            score_map = {s.trace_id: s for s in result.scores}
            sc = score_map.get(tid)
            if sc:
                lines.append(
                    f"  {i:3d}. {tid}"
                    f"  quality={sc.quality_score:.2f}"
                    f"  difficulty={sc.difficulty_score:.2f}"
                    f"  cluster={sc.topic_cluster}"
                )
            else:
                lines.append(f"  {i:3d}. {tid}")

    return "\n".join(lines)
