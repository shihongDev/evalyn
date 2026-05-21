"""
Simulation coverage report: compare coverage of simulated vs production data.

Pure Python, no external dependencies. Uses word-set-based analysis as a
lightweight proxy for semantic embedding coverage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "as",
        "was",
        "are",
        "be",
        "this",
        "that",
        "not",
        "has",
        "had",
        "have",
        "been",
        "will",
        "can",
        "do",
        "does",
        "did",
        "if",
        "so",
        "no",
        "up",
        "out",
        "its",
        "all",
        "my",
        "we",
        "he",
        "she",
        "they",
        "you",
        "me",
        "us",
        "him",
        "her",
        "who",
        "what",
        "when",
        "how",
        "which",
        "where",
        "there",
        "here",
        "then",
        "than",
        "also",
        "just",
        "about",
        "into",
        "over",
        "after",
        "more",
        "some",
        "any",
        "each",
        "very",
        "too",
        "own",
        "same",
    }
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CoverageRegion:
    """A topic region identified in production data with coverage gap info."""

    region_id: str
    keywords: list[str]
    simulated_count: int
    production_count: int
    gap: float  # production_count / max(1, simulated_count) - higher means bigger gap

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "keywords": list(self.keywords),
            "simulated_count": self.simulated_count,
            "production_count": self.production_count,
            "gap": self.gap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageRegion:
        return cls(
            region_id=data["region_id"],
            keywords=data["keywords"],
            simulated_count=data["simulated_count"],
            production_count=data["production_count"],
            gap=data["gap"],
        )


@dataclass
class CoverageResult:
    """Full coverage analysis result."""

    overlap_score: float
    total_sim_items: int
    total_prod_items: int
    uncovered_regions: list[CoverageRegion]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overlap_score": self.overlap_score,
            "total_sim_items": self.total_sim_items,
            "total_prod_items": self.total_prod_items,
            "uncovered_regions": [r.as_dict() for r in self.uncovered_regions],
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageResult:
        return cls(
            overlap_score=data["overlap_score"],
            total_sim_items=data["total_sim_items"],
            total_prod_items=data["total_prod_items"],
            uncovered_regions=[CoverageRegion.from_dict(r) for r in data["uncovered_regions"]],
            recommendations=data["recommendations"],
        )


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, split, filter short words and stop words."""
    return [w for w in text.lower().split() if len(w) >= 3 and w not in _STOP_WORDS]


def _vocabulary(texts: list[str]) -> set[str]:
    """Build vocabulary set from a list of texts."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(_tokenize(text))
    return vocab


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def extract_keywords(texts: list[str], top_n: int = 50) -> list[str]:
    """Extract the most frequent words from texts.

    Words are lowercased, filtered to minimum length 3, and stop words
    are removed. Returns up to top_n keywords sorted by frequency (descending).
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    return [word for word, _ in counter.most_common(top_n)]


def compute_vocabulary_coverage(
    sim_texts: list[str],
    prod_texts: list[str],
) -> float:
    """Fraction of production vocabulary covered by simulation vocabulary.

    Returns 0.0 if production vocabulary is empty.
    """
    sim_vocab = _vocabulary(sim_texts)
    prod_vocab = _vocabulary(prod_texts)

    if not prod_vocab:
        return 0.0

    covered = prod_vocab & sim_vocab
    return len(covered) / len(prod_vocab)


def identify_coverage_gaps(
    sim_texts: list[str],
    prod_texts: list[str],
    n_regions: int = 5,
) -> list[CoverageRegion]:
    """Find topic regions in production not well-covered by simulation.

    Identifies production keywords missing from simulation vocabulary,
    then clusters them into n_regions groups using a simple round-robin
    co-occurrence approach.

    Each region's gap = production_count / max(1, simulated_count).
    """
    sim_vocab = _vocabulary(sim_texts)
    prod_vocab = _vocabulary(prod_texts)

    # Find production words not covered by simulation
    uncovered_words = prod_vocab - sim_vocab
    if not uncovered_words:
        return []

    # Count how often each uncovered word appears in production
    prod_counter: Counter[str] = Counter()
    for text in prod_texts:
        tokens = _tokenize(text)
        for t in tokens:
            if t in uncovered_words:
                prod_counter[t] += 1

    # Count how often each uncovered word appears in simulation (should be 0
    # for truly uncovered words, but some partial matches possible)
    sim_counter: Counter[str] = Counter()
    for text in sim_texts:
        tokens = _tokenize(text)
        for t in tokens:
            if t in uncovered_words:
                sim_counter[t] += 1

    # Sort uncovered words by production frequency (most frequent first)
    sorted_words = sorted(uncovered_words, key=lambda w: prod_counter[w], reverse=True)

    # Cluster into n_regions groups by round-robin assignment
    # This ensures each cluster gets a mix of high and low frequency words
    actual_n = min(n_regions, len(sorted_words))
    if actual_n == 0:
        return []

    clusters: list[list[str]] = [[] for _ in range(actual_n)]
    for i, word in enumerate(sorted_words):
        clusters[i % actual_n].append(word)

    regions: list[CoverageRegion] = []
    for idx, cluster_words in enumerate(clusters):
        prod_count = sum(prod_counter[w] for w in cluster_words)
        sim_count = sum(sim_counter[w] for w in cluster_words)
        gap = prod_count / max(1, sim_count)

        regions.append(
            CoverageRegion(
                region_id=f"region_{idx}",
                keywords=cluster_words,
                simulated_count=sim_count,
                production_count=prod_count,
                gap=gap,
            )
        )

    # Sort by gap descending (largest gaps first)
    regions.sort(key=lambda r: r.gap, reverse=True)
    return regions


def build_coverage_report(
    sim_texts: list[str],
    prod_texts: list[str],
) -> CoverageResult:
    """Build a full coverage report with recommendations.

    Combines vocabulary coverage analysis, gap identification, and
    targeted recommendations.
    """
    overlap = compute_vocabulary_coverage(sim_texts, prod_texts)
    gaps = identify_coverage_gaps(sim_texts, prod_texts)
    recommendations = _build_recommendations(overlap, gaps)

    return CoverageResult(
        overlap_score=overlap,
        total_sim_items=len(sim_texts),
        total_prod_items=len(prod_texts),
        uncovered_regions=gaps,
        recommendations=recommendations,
    )


def suggest_simulation_targets(gaps: list[CoverageRegion]) -> list[str]:
    """Suggest topics to simulate based on gap regions.

    Returns one suggestion per gap region, ordered by gap size (largest first).
    """
    sorted_gaps = sorted(gaps, key=lambda g: g.gap, reverse=True)
    suggestions: list[str] = []
    for region in sorted_gaps:
        top_keywords = region.keywords[:5]
        keyword_str = ", ".join(top_keywords)
        suggestions.append(f"Add simulations covering: {keyword_str} (gap score: {region.gap:.1f})")
    return suggestions


def format_coverage_report(result: CoverageResult) -> str:
    """Format a human-readable coverage report."""
    lines: list[str] = []
    lines.append("SIMULATION COVERAGE REPORT")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Vocabulary overlap: {result.overlap_score:.1%}")
    lines.append(f"Simulation items:   {result.total_sim_items}")
    lines.append(f"Production items:   {result.total_prod_items}")
    lines.append("")

    if result.uncovered_regions:
        lines.append(f"Uncovered Regions ({len(result.uncovered_regions)}):")
        for region in result.uncovered_regions:
            kw_str = ", ".join(region.keywords[:5])
            lines.append(
                f"  {region.region_id}: [{kw_str}] "
                f"gap={region.gap:.1f}, "
                f"prod={region.production_count}, sim={region.simulated_count}"
            )
        lines.append("")

    if result.recommendations:
        lines.append("Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _build_recommendations(
    overlap: float,
    gaps: list[CoverageRegion],
) -> list[str]:
    """Generate recommendations based on coverage analysis."""
    recs: list[str] = []

    if overlap < 0.3:
        recs.append(
            "Critical: simulation vocabulary covers less than 30% of production. "
            "Consider a broad expansion of simulation topics."
        )
    elif overlap < 0.6:
        recs.append(
            "Warning: simulation vocabulary covers less than 60% of production. "
            "Targeted expansion recommended."
        )
    elif overlap < 0.9:
        recs.append("Moderate coverage. Fill specific gaps to reach 90%+ overlap.")
    else:
        recs.append("Good coverage (90%+). Focus on edge cases and rare topics.")

    for region in gaps[:3]:
        top_kw = ", ".join(region.keywords[:3])
        recs.append(f"Gap in region '{region.region_id}': add simulations for [{top_kw}].")

    return recs
