from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..models import Span


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RetrievalContext:
    """Context for a retrieval operation in a RAG pipeline."""

    query: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    top_k: int = 0
    similarity_scores: list[float] = field(default_factory=list)
    retrieval_method: str = "vector"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "documents": list(self.documents),
            "top_k": self.top_k,
            "similarity_scores": list(self.similarity_scores),
            "retrieval_method": self.retrieval_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalContext:
        return cls(
            query=data["query"],
            documents=data.get("documents", []),
            top_k=data.get("top_k", 0),
            similarity_scores=data.get("similarity_scores", []),
            retrieval_method=data.get("retrieval_method", "vector"),
        )


@dataclass
class MemoryOperation:
    """A single memory read/write/delete/search operation."""

    operation_type: str  # "read", "write", "delete", "search"
    key: str = ""
    value_summary: str = ""
    timestamp_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "key": self.key,
            "value_summary": self.value_summary,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryOperation:
        return cls(
            operation_type=data["operation_type"],
            key=data.get("key", ""),
            value_summary=data.get("value_summary", ""),
            timestamp_ms=data.get("timestamp_ms", 0.0),
        )


@dataclass
class RAGSpanData:
    """Data payload for a RAG-related span."""

    retrieval_context: RetrievalContext | None = None
    memory_ops: list[MemoryOperation] = field(default_factory=list)
    augmented_prompt_length: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_context": self.retrieval_context.as_dict()
            if self.retrieval_context
            else None,
            "memory_ops": [op.as_dict() for op in self.memory_ops],
            "augmented_prompt_length": self.augmented_prompt_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGSpanData:
        rc_raw = data.get("retrieval_context")
        return cls(
            retrieval_context=RetrievalContext.from_dict(rc_raw)
            if rc_raw
            else None,
            memory_ops=[
                MemoryOperation.from_dict(op)
                for op in data.get("memory_ops", [])
            ],
            augmented_prompt_length=data.get("augmented_prompt_length", 0),
        )


@dataclass
class RAGStats:
    """Aggregate statistics across RAG spans."""

    total_retrievals: int = 0
    total_memory_ops: int = 0
    avg_top_k: float = 0.0
    avg_similarity: float = 0.0
    retrieval_methods: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_retrievals": self.total_retrievals,
            "total_memory_ops": self.total_memory_ops,
            "avg_top_k": self.avg_top_k,
            "avg_similarity": self.avg_similarity,
            "retrieval_methods": list(self.retrieval_methods),
        }

    def format_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Total retrievals: {self.total_retrievals}",
            f"Total memory ops: {self.total_memory_ops}",
            f"Avg top_k: {self.avg_top_k:.1f}",
            f"Avg similarity: {self.avg_similarity:.3f}",
            f"Retrieval methods: {', '.join(self.retrieval_methods) if self.retrieval_methods else 'none'}",
        ]
        return "\n".join(lines)


def create_retrieval_span(
    span_id: str,
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 0,
    scores: list[float] | None = None,
) -> Span:
    """Create a retrieval span with RAGSpanData."""
    if scores is None:
        scores = []
    rc = RetrievalContext(
        query=query,
        documents=documents,
        top_k=top_k,
        similarity_scores=list(scores),
    )
    data = RAGSpanData(retrieval_context=rc)
    return Span(
        id=span_id,
        name=f"retrieval:{query[:30]}",
        span_type="retrieval",
        parent_id=None,
        start_time=_now_utc(),
        attributes={"rag_data": data.as_dict()},
    )


def create_memory_span(
    span_id: str,
    operations: list[MemoryOperation],
) -> Span:
    """Create a memory operation span."""
    data = RAGSpanData(memory_ops=operations)
    return Span(
        id=span_id,
        name="memory_ops",
        span_type="custom",
        parent_id=None,
        start_time=_now_utc(),
        attributes={"rag_data": data.as_dict()},
    )


def extract_rag_data(span: Span) -> RAGSpanData | None:
    """Extract RAGSpanData from span attributes. Returns None if not present."""
    raw = span.attributes.get("rag_data")
    if not raw or not isinstance(raw, dict):
        return None
    return RAGSpanData.from_dict(raw)


def compute_rag_stats(spans: list[Span]) -> RAGStats:
    """Aggregate retrieval stats across spans."""
    total_retrievals = 0
    total_memory_ops = 0
    top_k_sum = 0
    top_k_count = 0
    sim_sum = 0.0
    sim_count = 0
    methods: list[str] = []

    for s in spans:
        data = extract_rag_data(s)
        if data is None:
            continue
        if data.retrieval_context is not None:
            total_retrievals += 1
            rc = data.retrieval_context
            if rc.top_k > 0:
                top_k_sum += rc.top_k
                top_k_count += 1
            for score in rc.similarity_scores:
                sim_sum += score
                sim_count += 1
            if rc.retrieval_method and rc.retrieval_method not in methods:
                methods.append(rc.retrieval_method)
        total_memory_ops += len(data.memory_ops)

    return RAGStats(
        total_retrievals=total_retrievals,
        total_memory_ops=total_memory_ops,
        avg_top_k=top_k_sum / top_k_count if top_k_count > 0 else 0.0,
        avg_similarity=sim_sum / sim_count if sim_count > 0 else 0.0,
        retrieval_methods=methods,
    )


def analyze_retrieval_quality(spans: list[Span]) -> dict[str, Any]:
    """Analyze retrieval effectiveness across spans.

    Returns dict with:
    - avg_similarity_score: mean similarity across all retrieval scores
    - zero_result_rate: fraction of retrievals returning no documents
    - top_k_distribution: mapping of top_k value to count
    """
    all_scores: list[float] = []
    zero_results = 0
    total_retrievals = 0
    top_k_dist: dict[int, int] = {}

    for s in spans:
        data = extract_rag_data(s)
        if data is None or data.retrieval_context is None:
            continue
        rc = data.retrieval_context
        total_retrievals += 1
        all_scores.extend(rc.similarity_scores)
        if len(rc.documents) == 0:
            zero_results += 1
        k = rc.top_k
        top_k_dist[k] = top_k_dist.get(k, 0) + 1

    avg_sim = sum(all_scores) / len(all_scores) if all_scores else 0.0
    zero_rate = zero_results / total_retrievals if total_retrievals > 0 else 0.0

    return {
        "avg_similarity_score": avg_sim,
        "zero_result_rate": zero_rate,
        "top_k_distribution": top_k_dist,
    }
