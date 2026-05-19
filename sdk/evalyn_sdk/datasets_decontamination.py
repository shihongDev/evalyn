"""
Dataset decontamination: detect items that overlap with known benchmarks.

Provides n-gram based similarity checking to identify dataset items that
may be contaminated by benchmark data, plus a full decontamination pipeline
that returns clean items and a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ContaminationMatch:
    """A single contamination detection between an item and a benchmark."""

    item_id: str
    benchmark_name: str
    similarity: float
    matched_text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "benchmark_name": self.benchmark_name,
            "similarity": self.similarity,
            "matched_text": self.matched_text,
        }


@dataclass
class DecontaminationConfig:
    """Configuration for contamination detection."""

    min_similarity: float = 0.8
    ngram_size: int = 5
    check_input: bool = True
    check_output: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_similarity": self.min_similarity,
            "ngram_size": self.ngram_size,
            "check_input": self.check_input,
            "check_output": self.check_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecontaminationConfig:
        return cls(
            min_similarity=data.get("min_similarity", 0.8),
            ngram_size=data.get("ngram_size", 5),
            check_input=data.get("check_input", True),
            check_output=data.get("check_output", True),
        )


@dataclass
class DecontaminationReport:
    """Summary of a decontamination run."""

    matches: List[ContaminationMatch] = field(default_factory=list)
    clean_count: int = 0
    contaminated_count: int = 0
    contamination_rate: float = 0.0
    benchmarks_checked: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "matches": [m.as_dict() for m in self.matches],
            "clean_count": self.clean_count,
            "contaminated_count": self.contaminated_count,
            "contamination_rate": self.contamination_rate,
            "benchmarks_checked": list(self.benchmarks_checked),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecontaminationReport:
        return cls(
            matches=[
                ContaminationMatch(**m) for m in data.get("matches", [])
            ],
            clean_count=data.get("clean_count", 0),
            contaminated_count=data.get("contaminated_count", 0),
            contamination_rate=data.get("contamination_rate", 0.0),
            benchmarks_checked=data.get("benchmarks_checked", []),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        total = self.clean_count + self.contaminated_count
        lines.append(f"Decontamination Report: {total} items checked")
        lines.append(
            f"  Clean: {self.clean_count}, "
            f"Contaminated: {self.contaminated_count}, "
            f"Rate: {self.contamination_rate:.1%}"
        )
        if self.benchmarks_checked:
            lines.append(
                f"  Benchmarks: {', '.join(self.benchmarks_checked)}"
            )
        if self.matches:
            lines.append(f"  Matches ({len(self.matches)}):")
            for m in self.matches:
                lines.append(
                    f"    - {m.item_id}: {m.benchmark_name} "
                    f"(similarity={m.similarity:.2f})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def compute_ngrams(text: str, n: int = 5) -> set:
    """Extract character n-grams from text.

    Returns an empty set for text shorter than n characters.
    """
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def compute_ngram_similarity(text_a: str, text_b: str, n: int = 5) -> float:
    """Compute Jaccard similarity between character n-gram sets.

    Returns 0.0 if either text produces no n-grams.
    Returns 1.0 for identical texts (that are long enough for n-grams).
    """
    ngrams_a = compute_ngrams(text_a, n)
    ngrams_b = compute_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


def check_contamination(
    item_text: str,
    benchmark_texts: List[str],
    benchmark_name: str,
    config: DecontaminationConfig,
) -> List[ContaminationMatch]:
    """Check one item's text against a list of benchmark texts.

    Returns a ContaminationMatch for each benchmark text whose similarity
    meets or exceeds config.min_similarity.
    """
    matches: List[ContaminationMatch] = []
    for bt in benchmark_texts:
        sim = compute_ngram_similarity(item_text, bt, config.ngram_size)
        if sim >= config.min_similarity:
            matches.append(
                ContaminationMatch(
                    item_id="",  # caller fills this in
                    benchmark_name=benchmark_name,
                    similarity=sim,
                    matched_text=bt[:200],
                )
            )
    return matches


def decontaminate_dataset(
    items: List[Dict[str, Any]],
    benchmarks: Dict[str, List[str]],
    config: Optional[DecontaminationConfig] = None,
) -> Tuple[List[Dict[str, Any]], DecontaminationReport]:
    """Check all items against benchmarks and return clean items with a report.

    Each item should have an "input" field and optionally an "output" field.
    An item is flagged as contaminated if any of its checked fields match
    any benchmark text above the similarity threshold.

    Args:
        items: list of dicts, each with at least an "input" key.
        benchmarks: mapping of benchmark name to list of benchmark texts.
        config: optional config; defaults are used if None.

    Returns:
        (clean_items, report) where clean_items excludes contaminated items.
    """
    if config is None:
        config = DecontaminationConfig()

    all_matches: List[ContaminationMatch] = []
    contaminated_indices: set = set()

    for i, item in enumerate(items):
        item_id = item.get("id", str(i))
        texts_to_check: List[str] = []
        if config.check_input and "input" in item:
            texts_to_check.append(str(item["input"]))
        if config.check_output and "output" in item:
            texts_to_check.append(str(item["output"]))

        for bench_name, bench_texts in benchmarks.items():
            for text in texts_to_check:
                matches = check_contamination(
                    text, bench_texts, bench_name, config
                )
                for m in matches:
                    m.item_id = str(item_id)
                    all_matches.append(m)
                    contaminated_indices.add(i)

    clean_items = [
        item for i, item in enumerate(items) if i not in contaminated_indices
    ]
    total = len(items)
    contaminated_count = len(contaminated_indices)
    clean_count = total - contaminated_count

    report = DecontaminationReport(
        matches=all_matches,
        clean_count=clean_count,
        contaminated_count=contaminated_count,
        contamination_rate=(
            contaminated_count / total if total > 0 else 0.0
        ),
        benchmarks_checked=sorted(benchmarks.keys()),
    )
    return clean_items, report


def get_known_benchmark_names() -> List[str]:
    """Return a list of well-known benchmark names."""
    return [
        "ARC",
        "BBH",
        "DROP",
        "GSM8K",
        "HellaSwag",
        "HumanEval",
        "MATH",
        "MBPP",
        "MMLU",
        "TruthfulQA",
        "WinoGrande",
    ]
