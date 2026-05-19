"""Dataset item semantic search using word-set similarity.

Find items by natural language query using Jaccard similarity on
word-set tokenizations. Pure Python, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class SearchConfig:
    """Configuration for semantic search."""

    max_results: int = 10
    min_similarity: float = 0.1
    boost_exact_match: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_results": self.max_results,
            "min_similarity": self.min_similarity,
            "boost_exact_match": self.boost_exact_match,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchConfig:
        return cls(
            max_results=data.get("max_results", 10),
            min_similarity=data.get("min_similarity", 0.1),
            boost_exact_match=data.get("boost_exact_match", True),
        )


@dataclass
class SearchHit:
    """A single search result."""

    item_id: str
    text: str
    similarity: float
    matched_terms: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "similarity": self.similarity,
            "matched_terms": list(self.matched_terms),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchHit:
        return cls(
            item_id=data["item_id"],
            text=data["text"],
            similarity=data["similarity"],
            matched_terms=data.get("matched_terms", []),
        )


@dataclass
class SearchResult:
    """Aggregated search results."""

    hits: List[SearchHit] = field(default_factory=list)
    query: str = ""
    total_searched: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hits": [h.as_dict() for h in self.hits],
            "query": self.query,
            "total_searched": self.total_searched,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        return cls(
            hits=[SearchHit.from_dict(h) for h in data.get("hits", [])],
            query=data.get("query", ""),
            total_searched=data.get("total_searched", 0),
        )


# ---------------------------------------------------------------------------
# Tokenization and similarity
# ---------------------------------------------------------------------------


def tokenize_query(query: str) -> set[str]:
    """Lowercase, split, and filter short words (< 2 chars)."""
    words = _WORD_RE.findall(query.lower())
    return {w for w in words if len(w) >= 2}


def compute_similarity(query_tokens: set[str], item_tokens: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not query_tokens or not item_tokens:
        return 0.0
    intersection = query_tokens & item_tokens
    union = query_tokens | item_tokens
    return len(intersection) / len(union)


def find_matched_terms(query_tokens: set[str], item_tokens: set[str]) -> List[str]:
    """Return sorted list of tokens appearing in both sets."""
    return sorted(query_tokens & item_tokens)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_dataset(
    query: str,
    items: Dict[str, str],
    config: SearchConfig | None = None,
) -> SearchResult:
    """Search all items by query, rank by Jaccard similarity.

    If boost_exact_match is enabled, items whose text contains the
    exact query substring (case-insensitive) get a +0.2 similarity bonus.
    """
    if config is None:
        config = SearchConfig()

    query_tokens = tokenize_query(query)
    query_lower = query.lower()
    hits: List[SearchHit] = []

    for item_id, text in items.items():
        item_tokens = tokenize_query(text)
        sim = compute_similarity(query_tokens, item_tokens)

        if config.boost_exact_match and query_lower and query_lower in text.lower():
            sim += 0.2

        if sim < config.min_similarity:
            continue

        matched = find_matched_terms(query_tokens, item_tokens)
        hits.append(SearchHit(
            item_id=item_id,
            text=text,
            similarity=sim,
            matched_terms=matched,
        ))

    hits.sort(key=lambda h: h.similarity, reverse=True)
    hits = hits[: config.max_results]

    return SearchResult(hits=hits, query=query, total_searched=len(items))


def search_with_filters(
    query: str,
    items: Dict[str, str],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_field: str = "",
    filter_value: str = "",
    config: SearchConfig | None = None,
) -> SearchResult:
    """Search with optional metadata filtering.

    If filter_field and filter_value are provided, only items whose
    metadata contains the matching field value are searched.
    """
    if metadata and filter_field and filter_value:
        filtered_items = {
            k: v
            for k, v in items.items()
            if str(metadata.get(k, {}).get(filter_field, "")) == filter_value
        }
    else:
        filtered_items = dict(items)

    return search_dataset(query, filtered_items, config)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_search_results(result: SearchResult) -> str:
    """Format search results as a human-readable string."""
    if not result.hits:
        return f'No results for "{result.query}" (searched {result.total_searched} items).'

    lines: List[str] = [
        f'Results for "{result.query}" ({len(result.hits)} of {result.total_searched} items):'
    ]
    for i, hit in enumerate(result.hits, 1):
        preview = hit.text[:80]
        if len(hit.text) > 80:
            preview += "..."
        lines.append(
            f"  {i}. [{hit.item_id}] (sim={hit.similarity:.3f}) {preview}"
        )
        if hit.matched_terms:
            lines.append(f"     matched: {', '.join(hit.matched_terms)}")

    return "\n".join(lines)
