"""Tests for gemini_grounding module."""

from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.gemini_grounding import (
    GroundingMetadata,
    GroundingSource,
    compute_grounding_stats,
    extract_grounding,
    extract_grounding_from_span,
    has_grounding,
    inject_grounding_into_span,
)


def _span(sid="s1", **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- GroundingSource as_dict / from_dict ---


def test_grounding_source_as_dict():
    s = GroundingSource(uri="https://example.com", title="Example", snippet="text")
    d = s.as_dict()
    assert d == {"uri": "https://example.com", "title": "Example", "snippet": "text"}


def test_grounding_source_from_dict():
    d = {"uri": "https://a.com", "title": "A", "snippet": "snip"}
    s = GroundingSource.from_dict(d)
    assert s.uri == "https://a.com"
    assert s.title == "A"
    assert s.snippet == "snip"


def test_grounding_source_roundtrip():
    original = GroundingSource(uri="https://b.com", title="B", snippet="bb")
    restored = GroundingSource.from_dict(original.as_dict())
    assert restored.uri == original.uri
    assert restored.title == original.title
    assert restored.snippet == original.snippet


def test_grounding_source_defaults():
    s = GroundingSource()
    assert s.uri == ""
    assert s.title == ""
    assert s.snippet == ""


def test_grounding_source_from_dict_defaults():
    s = GroundingSource.from_dict({})
    assert s.uri == ""
    assert s.title == ""
    assert s.snippet == ""


# --- GroundingMetadata as_dict / from_dict ---


def test_grounding_metadata_as_dict():
    meta = GroundingMetadata(
        search_queries=["test query"],
        sources=[GroundingSource(uri="https://x.com", title="X")],
        grounding_score=0.85,
        is_grounded=True,
    )
    d = meta.as_dict()
    assert d["search_queries"] == ["test query"]
    assert len(d["sources"]) == 1
    assert d["grounding_score"] == 0.85
    assert d["is_grounded"] is True


def test_grounding_metadata_from_dict():
    d = {
        "search_queries": ["q1", "q2"],
        "sources": [{"uri": "https://y.com", "title": "Y", "snippet": "s"}],
        "grounding_score": 0.9,
        "is_grounded": True,
    }
    meta = GroundingMetadata.from_dict(d)
    assert meta.search_queries == ["q1", "q2"]
    assert len(meta.sources) == 1
    assert meta.grounding_score == 0.9
    assert meta.is_grounded is True


def test_grounding_metadata_roundtrip():
    original = GroundingMetadata(
        search_queries=["q"],
        sources=[GroundingSource("https://z.com", "Z", "zz")],
        grounding_score=0.5,
        is_grounded=True,
    )
    restored = GroundingMetadata.from_dict(original.as_dict())
    assert restored.search_queries == original.search_queries
    assert len(restored.sources) == len(original.sources)
    assert restored.grounding_score == original.grounding_score
    assert restored.is_grounded == original.is_grounded


def test_grounding_metadata_defaults():
    meta = GroundingMetadata()
    assert meta.search_queries == []
    assert meta.sources == []
    assert meta.grounding_score == 0.0
    assert meta.is_grounded is False


def test_grounding_metadata_format_text_with_sources():
    meta = GroundingMetadata(
        search_queries=["climate change"],
        sources=[
            GroundingSource(uri="https://a.com", title="Article A"),
            GroundingSource(uri="https://b.com", title="Article B"),
        ],
        grounding_score=0.75,
        is_grounded=True,
    )
    text = meta.format_text()
    assert "Grounding Metadata:" in text
    assert "Grounded: True" in text
    assert "Score: 0.75" in text
    assert "climate change" in text
    assert "Article A" in text
    assert "Sources (2):" in text


def test_grounding_metadata_format_text_empty():
    meta = GroundingMetadata()
    text = meta.format_text()
    assert "Grounded: False" in text
    assert "No sources" in text


# --- extract_grounding ---


def test_extract_grounding_valid():
    response = {
        "candidates": [
            {
                "groundingMetadata": {
                    "searchQueries": ["what is AI"],
                    "groundingSources": [
                        {"uri": "https://wiki.com", "title": "AI Wiki", "snippet": "AI is..."}
                    ],
                    "groundingScore": 0.92,
                }
            }
        ]
    }
    meta = extract_grounding(response)
    assert meta.is_grounded is True
    assert meta.search_queries == ["what is AI"]
    assert len(meta.sources) == 1
    assert meta.sources[0].uri == "https://wiki.com"
    assert meta.grounding_score == 0.92


def test_extract_grounding_no_metadata():
    response = {"candidates": [{"content": "hello"}]}
    meta = extract_grounding(response)
    assert meta.is_grounded is False
    assert meta.sources == []
    assert meta.search_queries == []


def test_extract_grounding_empty_response():
    meta = extract_grounding({})
    assert meta.is_grounded is False


def test_extract_grounding_empty_candidates():
    meta = extract_grounding({"candidates": []})
    assert meta.is_grounded is False


def test_extract_grounding_score_only():
    response = {
        "candidates": [
            {
                "groundingMetadata": {
                    "groundingScore": 0.6,
                }
            }
        ]
    }
    meta = extract_grounding(response)
    assert meta.is_grounded is True
    assert meta.grounding_score == 0.6
    assert meta.sources == []


# --- inject / extract span ---


def test_inject_grounding_into_span():
    s = _span()
    meta = GroundingMetadata(
        search_queries=["q"],
        sources=[GroundingSource("https://a.com", "A", "aa")],
        grounding_score=0.8,
        is_grounded=True,
    )
    new_s = inject_grounding_into_span(s, meta)
    assert "gemini.grounding" in new_s.attributes
    assert new_s.attributes["gemini.grounding"]["grounding_score"] == 0.8


def test_inject_preserves_original_span():
    s = _span(existing="value")
    meta = GroundingMetadata(is_grounded=True)
    new_s = inject_grounding_into_span(s, meta)
    assert "gemini.grounding" not in s.attributes
    assert "gemini.grounding" in new_s.attributes
    assert new_s.attributes.get("existing") == "value"


def test_extract_grounding_from_span_roundtrip():
    s = _span()
    meta = GroundingMetadata(
        search_queries=["q1"],
        sources=[GroundingSource("https://a.com", "A", "snip")],
        grounding_score=0.7,
        is_grounded=True,
    )
    new_s = inject_grounding_into_span(s, meta)
    extracted = extract_grounding_from_span(new_s)
    assert extracted is not None
    assert extracted.search_queries == ["q1"]
    assert len(extracted.sources) == 1
    assert extracted.grounding_score == 0.7
    assert extracted.is_grounded is True


def test_extract_grounding_from_span_none():
    s = _span()
    assert extract_grounding_from_span(s) is None


# --- has_grounding ---


def test_has_grounding_true():
    s = _span()
    meta = GroundingMetadata(is_grounded=True)
    new_s = inject_grounding_into_span(s, meta)
    assert has_grounding(new_s) is True


def test_has_grounding_false():
    s = _span()
    assert has_grounding(s) is False


# --- compute_grounding_stats ---


def test_compute_grounding_stats_aggregation():
    s1 = inject_grounding_into_span(
        _span("s1"),
        GroundingMetadata(
            search_queries=["q1", "q2"],
            sources=[GroundingSource("https://a.com")],
            grounding_score=0.8,
            is_grounded=True,
        ),
    )
    s2 = inject_grounding_into_span(
        _span("s2"),
        GroundingMetadata(
            search_queries=["q1"],
            sources=[GroundingSource("https://b.com"), GroundingSource("https://c.com")],
            grounding_score=0.6,
            is_grounded=True,
        ),
    )
    s3 = _span("s3")  # no grounding

    stats = compute_grounding_stats([s1, s2, s3])
    assert stats["total_grounded_spans"] == 2
    assert stats["total_sources"] == 3
    assert abs(stats["avg_grounding_score"] - 0.7) < 0.01
    # q1 appears in both spans
    query_counts = dict(stats["common_search_queries"])
    assert query_counts["q1"] == 2
    assert query_counts["q2"] == 1


def test_compute_grounding_stats_empty():
    stats = compute_grounding_stats([])
    assert stats["total_grounded_spans"] == 0
    assert stats["avg_grounding_score"] == 0.0
    assert stats["total_sources"] == 0
    assert stats["common_search_queries"] == []


def test_compute_grounding_stats_no_grounded():
    s = inject_grounding_into_span(
        _span(),
        GroundingMetadata(grounding_score=0.0, is_grounded=False),
    )
    stats = compute_grounding_stats([s])
    assert stats["total_grounded_spans"] == 0
    assert stats["avg_grounding_score"] == 0.0
