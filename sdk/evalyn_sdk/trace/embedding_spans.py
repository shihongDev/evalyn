from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import Span


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EmbeddingSpanData:
    """Data payload for an embedding operation span."""

    model: str
    input_texts: list[str] = field(default_factory=list)
    dimensions: int = 0
    token_count: int = 0
    batch_size: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_texts": list(self.input_texts),
            "dimensions": self.dimensions,
            "token_count": self.token_count,
            "batch_size": self.batch_size,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSpanData:
        return cls(
            model=data["model"],
            input_texts=data.get("input_texts", []),
            dimensions=data.get("dimensions", 0),
            token_count=data.get("token_count", 0),
            batch_size=data.get("batch_size", 0),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class RerankerSpanData:
    """Data payload for a reranker operation span."""

    model: str
    query: str = ""
    documents: list[str] = field(default_factory=list)
    top_k: int = 0
    token_count: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "query": self.query,
            "documents": list(self.documents),
            "top_k": self.top_k,
            "token_count": self.token_count,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RerankerSpanData:
        return cls(
            model=data["model"],
            query=data.get("query", ""),
            documents=data.get("documents", []),
            top_k=data.get("top_k", 0),
            token_count=data.get("token_count", 0),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class EmbeddingStats:
    """Aggregate statistics across embedding spans."""

    total_embeddings: int
    total_tokens: int
    avg_dimensions: float
    models_used: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_embeddings": self.total_embeddings,
            "total_tokens": self.total_tokens,
            "avg_dimensions": self.avg_dimensions,
            "models_used": list(self.models_used),
        }

    def format_text(self) -> str:
        """Human-readable summary of embedding stats."""
        lines = [
            f"Total embeddings: {self.total_embeddings}",
            f"Total tokens: {self.total_tokens}",
            f"Avg dimensions: {self.avg_dimensions:.1f}",
            f"Models used: {', '.join(self.models_used) if self.models_used else 'none'}",
        ]
        return "\n".join(lines)


def create_embedding_span(
    span_id: str,
    model: str,
    input_texts: list[str],
    dimensions: int = 0,
    token_count: int = 0,
) -> Span:
    """Create a Span with span_type='embedding' and EmbeddingSpanData in attributes."""
    data = EmbeddingSpanData(
        model=model,
        input_texts=input_texts,
        dimensions=dimensions,
        token_count=token_count,
        batch_size=len(input_texts),
    )
    return Span(
        id=span_id,
        name=f"embedding:{model}",
        span_type="embedding",
        parent_id=None,
        start_time=_now_utc(),
        attributes={"embedding_data": data.as_dict()},
    )


def create_reranker_span(
    span_id: str,
    model: str,
    query: str,
    documents: list[str],
    top_k: int = 0,
) -> Span:
    """Create a Span with span_type='reranker' and RerankerSpanData in attributes."""
    data = RerankerSpanData(
        model=model,
        query=query,
        documents=documents,
        top_k=top_k,
    )
    return Span(
        id=span_id,
        name=f"reranker:{model}",
        span_type="reranker",
        parent_id=None,
        start_time=_now_utc(),
        attributes={"reranker_data": data.as_dict()},
    )


def extract_embedding_data(span: Span) -> EmbeddingSpanData | None:
    """Extract EmbeddingSpanData from span attributes. Returns None if not an embedding span."""
    if span.span_type != "embedding":
        return None
    raw = span.attributes.get("embedding_data")
    if not raw or not isinstance(raw, dict):
        return None
    return EmbeddingSpanData.from_dict(raw)


def extract_reranker_data(span: Span) -> RerankerSpanData | None:
    """Extract RerankerSpanData from span attributes. Returns None if not a reranker span."""
    if span.span_type != "reranker":
        return None
    raw = span.attributes.get("reranker_data")
    if not raw or not isinstance(raw, dict):
        return None
    return RerankerSpanData.from_dict(raw)


def compute_embedding_stats(spans: list[Span]) -> EmbeddingStats:
    """Aggregate stats from embedding spans in the given list."""
    total_embeddings = 0
    total_tokens = 0
    dimensions_sum = 0
    dimensions_count = 0
    models: list[str] = []

    for s in spans:
        data = extract_embedding_data(s)
        if data is None:
            continue
        total_embeddings += len(data.input_texts)
        total_tokens += data.token_count
        if data.dimensions > 0:
            dimensions_sum += data.dimensions
            dimensions_count += 1
        if data.model and data.model not in models:
            models.append(data.model)

    avg_dim = dimensions_sum / dimensions_count if dimensions_count > 0 else 0.0

    return EmbeddingStats(
        total_embeddings=total_embeddings,
        total_tokens=total_tokens,
        avg_dimensions=avg_dim,
        models_used=models,
    )
