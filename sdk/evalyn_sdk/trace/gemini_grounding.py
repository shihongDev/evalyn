"""Gemini grounding metadata capture - capture search grounding results from Gemini responses."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class GroundingSource:
    """A single grounding source from a Gemini response."""

    uri: str = ""
    title: str = ""
    snippet: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "title": self.title,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundingSource:
        return cls(
            uri=data.get("uri", ""),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
        )


@dataclass
class GroundingMetadata:
    """Aggregated grounding metadata from a Gemini response."""

    search_queries: List[str] = field(default_factory=list)
    sources: List[GroundingSource] = field(default_factory=list)
    grounding_score: float = 0.0
    is_grounded: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "search_queries": list(self.search_queries),
            "sources": [s.as_dict() for s in self.sources],
            "grounding_score": self.grounding_score,
            "is_grounded": self.is_grounded,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundingMetadata:
        return cls(
            search_queries=data.get("search_queries", []),
            sources=[GroundingSource.from_dict(s) for s in data.get("sources", [])],
            grounding_score=data.get("grounding_score", 0.0),
            is_grounded=data.get("is_grounded", False),
        )

    def format_text(self) -> str:
        lines = ["Grounding Metadata:"]
        lines.append(f"  Grounded: {self.is_grounded}")
        lines.append(f"  Score: {self.grounding_score}")
        if self.search_queries:
            lines.append(f"  Queries: {', '.join(self.search_queries)}")
        if self.sources:
            lines.append(f"  Sources ({len(self.sources)}):")
            for s in self.sources:
                title_part = s.title or s.uri or "(no title)"
                lines.append(f"    - {title_part}")
        else:
            lines.append("  No sources")
        return "\n".join(lines)


def extract_grounding(response_data: Dict[str, Any]) -> GroundingMetadata:
    """Extract grounding metadata from a Gemini response dict."""
    candidates = response_data.get("candidates", [{}])
    if not candidates:
        return GroundingMetadata()
    first_candidate = candidates[0]
    raw = first_candidate.get("groundingMetadata", {})
    if not raw:
        return GroundingMetadata()

    search_queries = raw.get("searchQueries", [])
    raw_sources = raw.get("groundingSources", [])
    sources = []
    for rs in raw_sources:
        sources.append(GroundingSource(
            uri=rs.get("uri", ""),
            title=rs.get("title", ""),
            snippet=rs.get("snippet", ""),
        ))

    grounding_score = raw.get("groundingScore", 0.0)
    is_grounded = bool(sources) or grounding_score > 0.0

    return GroundingMetadata(
        search_queries=search_queries,
        sources=sources,
        grounding_score=grounding_score,
        is_grounded=is_grounded,
    )


def inject_grounding_into_span(span: Span, metadata: GroundingMetadata) -> Span:
    """Add grounding metadata to span attributes under 'gemini.grounding'. Returns a new span."""
    new_span = copy.deepcopy(span)
    new_span.attributes["gemini.grounding"] = metadata.as_dict()
    return new_span


def extract_grounding_from_span(span: Span) -> Optional[GroundingMetadata]:
    """Extract grounding metadata from span attributes."""
    raw = span.attributes.get("gemini.grounding")
    if raw is None:
        return None
    return GroundingMetadata.from_dict(raw)


def has_grounding(span: Span) -> bool:
    """Quick check if span has grounding data."""
    return "gemini.grounding" in span.attributes


def compute_grounding_stats(spans: List[Span]) -> Dict[str, Any]:
    """Aggregate grounding stats across spans.

    Returns: total grounded spans, avg grounding score, total sources, common search queries.
    """
    grounded_count = 0
    total_score = 0.0
    total_sources = 0
    query_counter: Counter = Counter()

    for s in spans:
        meta = extract_grounding_from_span(s)
        if meta is None:
            continue
        if meta.is_grounded:
            grounded_count += 1
        total_score += meta.grounding_score
        total_sources += len(meta.sources)
        for q in meta.search_queries:
            query_counter[q] += 1

    scored_count = sum(
        1 for s in spans if extract_grounding_from_span(s) is not None
    )
    avg_score = total_score / scored_count if scored_count > 0 else 0.0

    return {
        "total_grounded_spans": grounded_count,
        "avg_grounding_score": avg_score,
        "total_sources": total_sources,
        "common_search_queries": query_counter.most_common(10),
    }
