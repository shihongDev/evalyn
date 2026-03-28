from __future__ import annotations

from evalyn_sdk.trace.embedding_spans import (
    EmbeddingSpanData,
    RerankerSpanData,
    EmbeddingStats,
    create_embedding_span,
    create_reranker_span,
    extract_embedding_data,
    extract_reranker_data,
    compute_embedding_stats,
)
from evalyn_sdk.models import Span
from datetime import datetime, timezone


def _make_now():
    return datetime.now(timezone.utc)


# -- create_embedding_span --

def test_create_embedding_span_type():
    span = create_embedding_span("s1", "text-embedding-3-small", ["hello"])
    assert span.span_type == "embedding"


def test_create_embedding_span_id():
    span = create_embedding_span("s1", "text-embedding-3-small", ["hello"])
    assert span.id == "s1"


def test_create_embedding_span_name():
    span = create_embedding_span("s1", "text-embedding-3-small", ["hello"])
    assert span.name == "embedding:text-embedding-3-small"


def test_create_embedding_span_attributes():
    span = create_embedding_span("s1", "model-x", ["a", "b"], dimensions=768, token_count=50)
    data = span.attributes.get("embedding_data")
    assert data is not None
    assert data["model"] == "model-x"
    assert data["dimensions"] == 768
    assert data["token_count"] == 50
    assert data["batch_size"] == 2


# -- create_reranker_span --

def test_create_reranker_span_type():
    span = create_reranker_span("r1", "reranker-v1", "query", ["doc1"])
    assert span.span_type == "reranker"


def test_create_reranker_span_id():
    span = create_reranker_span("r1", "reranker-v1", "query", ["doc1"])
    assert span.id == "r1"


def test_create_reranker_span_name():
    span = create_reranker_span("r1", "reranker-v1", "query", ["doc1"])
    assert span.name == "reranker:reranker-v1"


def test_create_reranker_span_attributes():
    span = create_reranker_span("r1", "reranker-v1", "find this", ["doc1", "doc2"], top_k=5)
    data = span.attributes.get("reranker_data")
    assert data is not None
    assert data["query"] == "find this"
    assert data["top_k"] == 5
    assert len(data["documents"]) == 2


# -- extract_embedding_data --

def test_extract_embedding_data_from_embedding_span():
    span = create_embedding_span("s1", "model-a", ["text"], dimensions=512, token_count=10)
    data = extract_embedding_data(span)
    assert data is not None
    assert data.model == "model-a"
    assert data.dimensions == 512


def test_extract_embedding_data_wrong_type():
    span = create_reranker_span("r1", "model-b", "q", ["d"])
    data = extract_embedding_data(span)
    assert data is None


def test_extract_embedding_data_missing_attributes():
    span = Span(id="x", name="test", span_type="embedding", parent_id=None, start_time=_make_now())
    data = extract_embedding_data(span)
    assert data is None


# -- extract_reranker_data --

def test_extract_reranker_data_from_reranker_span():
    span = create_reranker_span("r1", "model-r", "query text", ["doc"], top_k=3)
    data = extract_reranker_data(span)
    assert data is not None
    assert data.model == "model-r"
    assert data.query == "query text"
    assert data.top_k == 3


def test_extract_reranker_data_wrong_type():
    span = create_embedding_span("s1", "model-e", ["text"])
    data = extract_reranker_data(span)
    assert data is None


def test_extract_reranker_data_missing_attributes():
    span = Span(id="x", name="test", span_type="reranker", parent_id=None, start_time=_make_now())
    data = extract_reranker_data(span)
    assert data is None


# -- compute_embedding_stats --

def test_compute_embedding_stats_basic():
    spans = [
        create_embedding_span("s1", "model-a", ["t1", "t2"], dimensions=768, token_count=20),
        create_embedding_span("s2", "model-a", ["t3"], dimensions=768, token_count=10),
    ]
    stats = compute_embedding_stats(spans)
    assert stats.total_embeddings == 3
    assert stats.total_tokens == 30
    assert stats.avg_dimensions == 768.0
    assert stats.models_used == ["model-a"]


def test_compute_embedding_stats_multiple_models():
    spans = [
        create_embedding_span("s1", "model-a", ["t1"], dimensions=512, token_count=5),
        create_embedding_span("s2", "model-b", ["t2"], dimensions=1024, token_count=8),
    ]
    stats = compute_embedding_stats(spans)
    assert stats.total_embeddings == 2
    assert stats.total_tokens == 13
    assert stats.avg_dimensions == (512 + 1024) / 2
    assert "model-a" in stats.models_used
    assert "model-b" in stats.models_used


def test_compute_embedding_stats_empty():
    stats = compute_embedding_stats([])
    assert stats.total_embeddings == 0
    assert stats.total_tokens == 0
    assert stats.avg_dimensions == 0.0
    assert stats.models_used == []


def test_compute_embedding_stats_skips_non_embedding():
    spans = [
        create_reranker_span("r1", "reranker-v1", "q", ["d"]),
        create_embedding_span("s1", "model-a", ["t1"], dimensions=768, token_count=10),
    ]
    stats = compute_embedding_stats(spans)
    assert stats.total_embeddings == 1
    assert stats.total_tokens == 10


# -- EmbeddingSpanData as_dict / from_dict --

def test_embedding_span_data_as_dict():
    data = EmbeddingSpanData(
        model="model-x",
        input_texts=["a", "b"],
        dimensions=512,
        token_count=20,
        batch_size=2,
        duration_ms=15.5,
    )
    d = data.as_dict()
    assert d["model"] == "model-x"
    assert d["dimensions"] == 512
    assert d["batch_size"] == 2
    assert d["duration_ms"] == 15.5


def test_embedding_span_data_from_dict():
    d = {"model": "model-y", "input_texts": ["c"], "dimensions": 1024}
    data = EmbeddingSpanData.from_dict(d)
    assert data.model == "model-y"
    assert data.dimensions == 1024
    assert data.token_count == 0  # default


def test_embedding_span_data_roundtrip():
    original = EmbeddingSpanData(
        model="model-z", input_texts=["x", "y"], dimensions=256,
        token_count=100, batch_size=2, duration_ms=42.0,
    )
    restored = EmbeddingSpanData.from_dict(original.as_dict())
    assert restored.model == original.model
    assert restored.input_texts == original.input_texts
    assert restored.dimensions == original.dimensions
    assert restored.duration_ms == original.duration_ms


# -- RerankerSpanData as_dict / from_dict --

def test_reranker_span_data_as_dict():
    data = RerankerSpanData(
        model="reranker-1", query="search query", documents=["d1", "d2"],
        top_k=3, token_count=50, duration_ms=20.0,
    )
    d = data.as_dict()
    assert d["model"] == "reranker-1"
    assert d["query"] == "search query"
    assert d["top_k"] == 3


def test_reranker_span_data_from_dict():
    d = {"model": "reranker-2", "query": "q", "documents": ["d"]}
    data = RerankerSpanData.from_dict(d)
    assert data.model == "reranker-2"
    assert data.top_k == 0  # default


def test_reranker_span_data_roundtrip():
    original = RerankerSpanData(
        model="reranker-3", query="find me", documents=["a", "b", "c"],
        top_k=5, token_count=80, duration_ms=33.3,
    )
    restored = RerankerSpanData.from_dict(original.as_dict())
    assert restored.model == original.model
    assert restored.query == original.query
    assert restored.documents == original.documents
    assert restored.top_k == original.top_k


# -- EmbeddingStats format_text --

def test_embedding_stats_format_text():
    stats = EmbeddingStats(
        total_embeddings=5, total_tokens=100, avg_dimensions=768.0,
        models_used=["model-a", "model-b"],
    )
    text = stats.format_text()
    assert "Total embeddings: 5" in text
    assert "Total tokens: 100" in text
    assert "768.0" in text
    assert "model-a" in text
    assert "model-b" in text


def test_embedding_stats_format_text_no_models():
    stats = EmbeddingStats(
        total_embeddings=0, total_tokens=0, avg_dimensions=0.0, models_used=[],
    )
    text = stats.format_text()
    assert "none" in text


# -- EmbeddingStats as_dict --

def test_embedding_stats_as_dict():
    stats = EmbeddingStats(
        total_embeddings=3, total_tokens=60, avg_dimensions=512.0,
        models_used=["m1"],
    )
    d = stats.as_dict()
    assert d["total_embeddings"] == 3
    assert d["models_used"] == ["m1"]
