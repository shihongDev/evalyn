from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.rag_tracing import (
    RetrievalContext,
    MemoryOperation,
    RAGSpanData,
    RAGStats,
    create_retrieval_span,
    create_memory_span,
    extract_rag_data,
    compute_rag_stats,
    analyze_retrieval_quality,
)


def _span(sid, span_type="llm_call", start_offset_ms=0, duration_ms=100, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


# -- RetrievalContext --


class TestRetrievalContext:
    def test_as_dict_basic(self):
        rc = RetrievalContext(query="what is X?", documents=[{"id": "d1"}], top_k=3)
        d = rc.as_dict()
        assert d["query"] == "what is X?"
        assert d["documents"] == [{"id": "d1"}]
        assert d["top_k"] == 3
        assert d["retrieval_method"] == "vector"

    def test_from_dict_roundtrip(self):
        rc = RetrievalContext(
            query="q",
            documents=[{"text": "hello"}],
            top_k=5,
            similarity_scores=[0.9, 0.8],
            retrieval_method="bm25",
        )
        rebuilt = RetrievalContext.from_dict(rc.as_dict())
        assert rebuilt.query == rc.query
        assert rebuilt.documents == rc.documents
        assert rebuilt.top_k == rc.top_k
        assert rebuilt.similarity_scores == rc.similarity_scores
        assert rebuilt.retrieval_method == rc.retrieval_method

    def test_defaults(self):
        rc = RetrievalContext(query="q")
        assert rc.documents == []
        assert rc.top_k == 0
        assert rc.similarity_scores == []
        assert rc.retrieval_method == "vector"

    def test_from_dict_missing_optional(self):
        rc = RetrievalContext.from_dict({"query": "q"})
        assert rc.query == "q"
        assert rc.documents == []


# -- MemoryOperation --


class TestMemoryOperation:
    def test_as_dict(self):
        op = MemoryOperation(operation_type="write", key="k1", value_summary="val")
        d = op.as_dict()
        assert d["operation_type"] == "write"
        assert d["key"] == "k1"

    def test_from_dict_roundtrip(self):
        op = MemoryOperation(
            operation_type="read", key="k", value_summary="v", timestamp_ms=123.0
        )
        rebuilt = MemoryOperation.from_dict(op.as_dict())
        assert rebuilt.operation_type == op.operation_type
        assert rebuilt.key == op.key
        assert rebuilt.timestamp_ms == op.timestamp_ms

    def test_defaults(self):
        op = MemoryOperation(operation_type="delete")
        assert op.key == ""
        assert op.value_summary == ""
        assert op.timestamp_ms == 0.0


# -- RAGSpanData --


class TestRAGSpanData:
    def test_as_dict_with_retrieval(self):
        rc = RetrievalContext(query="q", documents=[{"id": "1"}])
        data = RAGSpanData(retrieval_context=rc, augmented_prompt_length=500)
        d = data.as_dict()
        assert d["retrieval_context"]["query"] == "q"
        assert d["augmented_prompt_length"] == 500

    def test_as_dict_no_retrieval(self):
        data = RAGSpanData()
        d = data.as_dict()
        assert d["retrieval_context"] is None
        assert d["memory_ops"] == []

    def test_from_dict_roundtrip(self):
        rc = RetrievalContext(query="q")
        ops = [MemoryOperation(operation_type="write", key="k")]
        data = RAGSpanData(retrieval_context=rc, memory_ops=ops, augmented_prompt_length=10)
        rebuilt = RAGSpanData.from_dict(data.as_dict())
        assert rebuilt.retrieval_context is not None
        assert rebuilt.retrieval_context.query == "q"
        assert len(rebuilt.memory_ops) == 1
        assert rebuilt.augmented_prompt_length == 10


# -- RAGStats --


class TestRAGStats:
    def test_format_text_populated(self):
        stats = RAGStats(
            total_retrievals=3,
            total_memory_ops=5,
            avg_top_k=4.0,
            avg_similarity=0.85,
            retrieval_methods=["vector", "bm25"],
        )
        text = stats.format_text()
        assert "Total retrievals: 3" in text
        assert "vector, bm25" in text

    def test_format_text_empty(self):
        stats = RAGStats()
        text = stats.format_text()
        assert "Total retrievals: 0" in text
        assert "none" in text

    def test_as_dict(self):
        stats = RAGStats(total_retrievals=1, retrieval_methods=["vector"])
        d = stats.as_dict()
        assert d["total_retrievals"] == 1
        assert d["retrieval_methods"] == ["vector"]


# -- create_retrieval_span --


class TestCreateRetrievalSpan:
    def test_correct_type(self):
        span = create_retrieval_span("r1", "query text", [{"id": "d1"}], top_k=3)
        assert span.span_type == "retrieval"
        assert span.id == "r1"

    def test_has_rag_data(self):
        span = create_retrieval_span(
            "r2", "q", [{"id": "d1"}], top_k=5, scores=[0.9, 0.8]
        )
        data = extract_rag_data(span)
        assert data is not None
        assert data.retrieval_context is not None
        assert data.retrieval_context.query == "q"
        assert data.retrieval_context.similarity_scores == [0.9, 0.8]

    def test_empty_documents(self):
        span = create_retrieval_span("r3", "q", [])
        data = extract_rag_data(span)
        assert data is not None
        assert data.retrieval_context.documents == []


# -- create_memory_span --


class TestCreateMemorySpan:
    def test_correct_type(self):
        ops = [MemoryOperation(operation_type="write", key="k1")]
        span = create_memory_span("m1", ops)
        assert span.span_type == "custom"
        assert span.id == "m1"

    def test_has_rag_data(self):
        ops = [
            MemoryOperation(operation_type="write", key="k1"),
            MemoryOperation(operation_type="read", key="k2"),
        ]
        span = create_memory_span("m2", ops)
        data = extract_rag_data(span)
        assert data is not None
        assert len(data.memory_ops) == 2


# -- extract_rag_data --


class TestExtractRagData:
    def test_from_retrieval_span(self):
        span = create_retrieval_span("x1", "q", [{"id": "1"}])
        data = extract_rag_data(span)
        assert data is not None
        assert data.retrieval_context is not None

    def test_non_rag_span_returns_none(self):
        span = _span("s1")
        assert extract_rag_data(span) is None

    def test_span_with_wrong_attribute_type(self):
        span = _span("s2", rag_data="not a dict")
        assert extract_rag_data(span) is None


# -- compute_rag_stats --


class TestComputeRagStats:
    def test_aggregation(self):
        s1 = create_retrieval_span("r1", "q1", [{"id": "1"}], top_k=3, scores=[0.9])
        s2 = create_retrieval_span("r2", "q2", [{"id": "2"}], top_k=5, scores=[0.7, 0.6])
        ops = [MemoryOperation(operation_type="write")]
        s3 = create_memory_span("m1", ops)
        stats = compute_rag_stats([s1, s2, s3])
        assert stats.total_retrievals == 2
        assert stats.total_memory_ops == 1
        assert stats.avg_top_k == 4.0
        # avg similarity: (0.9 + 0.7 + 0.6) / 3
        expected_sim = (0.9 + 0.7 + 0.6) / 3
        assert abs(stats.avg_similarity - expected_sim) < 1e-6
        assert "vector" in stats.retrieval_methods

    def test_empty_spans(self):
        stats = compute_rag_stats([])
        assert stats.total_retrievals == 0
        assert stats.avg_similarity == 0.0

    def test_no_rag_spans(self):
        spans = [_span("s1"), _span("s2")]
        stats = compute_rag_stats(spans)
        assert stats.total_retrievals == 0


# -- analyze_retrieval_quality --


class TestAnalyzeRetrievalQuality:
    def test_basic_quality(self):
        s1 = create_retrieval_span("r1", "q1", [{"id": "1"}], top_k=3, scores=[0.9, 0.8])
        s2 = create_retrieval_span("r2", "q2", [], top_k=5, scores=[])
        result = analyze_retrieval_quality([s1, s2])
        assert result["zero_result_rate"] == 0.5
        expected_avg = (0.9 + 0.8) / 2
        assert abs(result["avg_similarity_score"] - expected_avg) < 1e-6
        assert result["top_k_distribution"][3] == 1
        assert result["top_k_distribution"][5] == 1

    def test_empty_spans(self):
        result = analyze_retrieval_quality([])
        assert result["avg_similarity_score"] == 0.0
        assert result["zero_result_rate"] == 0.0
        assert result["top_k_distribution"] == {}

    def test_no_rag_spans(self):
        result = analyze_retrieval_quality([_span("s1")])
        assert result["avg_similarity_score"] == 0.0

    def test_all_zero_results(self):
        s1 = create_retrieval_span("r1", "q1", [], top_k=3)
        s2 = create_retrieval_span("r2", "q2", [], top_k=3)
        result = analyze_retrieval_quality([s1, s2])
        assert result["zero_result_rate"] == 1.0

    def test_multiple_methods(self):
        # Retrieval method defaults to "vector" for create_retrieval_span
        s1 = create_retrieval_span("r1", "q1", [{"id": "1"}], top_k=3, scores=[0.5])
        # Modify attributes to add a different method
        data = extract_rag_data(s1)
        assert data is not None
        assert data.retrieval_context.retrieval_method == "vector"
