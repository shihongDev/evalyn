"""Tests for the FTS5 full-text search module."""

from __future__ import annotations

import os
import tempfile

import pytest

from evalyn_sdk.fts_search import (
    SearchIndex,
    SearchResult,
    build_index_from_traces,
    format_search_results,
    search_and_filter,
)


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_create_minimal(self):
        r = SearchResult(item_id="1", content="hello", score=1.0)
        assert r.item_id == "1"
        assert r.content == "hello"
        assert r.score == 1.0
        assert r.snippet == ""
        assert r.source_field == ""

    def test_create_full(self):
        r = SearchResult(
            item_id="x",
            content="body",
            score=0.5,
            snippet="<b>body</b>",
            source_field="input",
        )
        assert r.snippet == "<b>body</b>"
        assert r.source_field == "input"

    def test_as_dict(self):
        r = SearchResult("1", "c", 2.0, "s", "input")
        d = r.as_dict()
        assert d == {
            "item_id": "1",
            "content": "c",
            "score": 2.0,
            "snippet": "s",
            "source_field": "input",
        }

    def test_from_dict_minimal(self):
        r = SearchResult.from_dict({"item_id": "a", "content": "b"})
        assert r.item_id == "a"
        assert r.score == 0.0
        assert r.snippet == ""

    def test_from_dict_full(self):
        d = {
            "item_id": "z",
            "content": "text",
            "score": 3.14,
            "snippet": "snip",
            "source_field": "output",
        }
        r = SearchResult.from_dict(d)
        assert r.score == 3.14
        assert r.source_field == "output"

    def test_roundtrip(self):
        original = SearchResult("id1", "content text", 1.5, "snip", "input")
        restored = SearchResult.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.content == original.content
        assert restored.score == original.score
        assert restored.snippet == original.snippet
        assert restored.source_field == original.source_field


# ---------------------------------------------------------------------------
# SearchIndex - basic operations
# ---------------------------------------------------------------------------


class TestSearchIndexBasic:
    def test_empty_index_count(self):
        idx = SearchIndex()
        assert idx.count() == 0
        idx.close()

    def test_add_document_and_count(self):
        idx = SearchIndex()
        idx.add_document("1", "hello world")
        assert idx.count() == 1
        idx.close()

    def test_add_documents_batch(self):
        idx = SearchIndex()
        docs = [
            ("1", "apple pie", "input"),
            ("2", "banana split", "input"),
            ("3", "cherry tart", "output"),
        ]
        idx.add_documents(docs)
        assert idx.count() == 3
        idx.close()

    def test_clear(self):
        idx = SearchIndex()
        idx.add_document("1", "something")
        idx.add_document("2", "else")
        assert idx.count() == 2
        idx.clear()
        assert idx.count() == 0
        idx.close()

    def test_close_and_reopen_file_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = SearchIndex(db_path)
            idx.add_document("1", "persistent data")
            idx.close()

            idx2 = SearchIndex(db_path)
            assert idx2.count() == 1
            results = idx2.search("persistent")
            assert len(results) == 1
            idx2.close()
        finally:
            os.unlink(db_path)

    def test_add_document_default_source_field(self):
        idx = SearchIndex()
        idx.add_document("1", "test content")
        results = idx.search("test")
        assert results[0].source_field == "input"
        idx.close()

    def test_add_document_custom_source_field(self):
        idx = SearchIndex()
        idx.add_document("1", "test content", source_field="output")
        results = idx.search("test")
        assert results[0].source_field == "output"
        idx.close()


# ---------------------------------------------------------------------------
# SearchIndex - search
# ---------------------------------------------------------------------------


class TestSearchIndexSearch:
    @pytest.fixture()
    def populated_index(self):
        idx = SearchIndex()
        docs = [
            ("1", "The quick brown fox jumps over the lazy dog", "input"),
            ("2", "Machine learning models require large datasets", "input"),
            ("3", "Python is a popular programming language", "input"),
            ("4", "The fox ran quickly through the forest", "output"),
            ("5", "Deep learning is a subset of machine learning", "output"),
        ]
        idx.add_documents(docs)
        yield idx
        idx.close()

    def test_search_basic(self, populated_index):
        results = populated_index.search("fox")
        assert len(results) >= 1
        ids = [r.item_id for r in results]
        assert "1" in ids

    def test_search_returns_search_results(self, populated_index):
        results = populated_index.search("python")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

    def test_search_scores_positive(self, populated_index):
        results = populated_index.search("machine learning")
        for r in results:
            assert r.score > 0

    def test_search_limit(self, populated_index):
        results = populated_index.search("the", limit=2)
        assert len(results) <= 2

    def test_search_no_match(self, populated_index):
        results = populated_index.search("xyznonexistent")
        assert len(results) == 0

    def test_search_empty_query(self, populated_index):
        results = populated_index.search("")
        assert len(results) == 0

    def test_search_whitespace_query(self, populated_index):
        results = populated_index.search("   ")
        assert len(results) == 0

    def test_search_multiple_terms(self, populated_index):
        results = populated_index.search("quick brown")
        assert len(results) >= 1
        assert results[0].item_id == "1"

    def test_search_ranking_relevance(self, populated_index):
        # "machine learning" appears explicitly in items 2 and 5
        results = populated_index.search("machine learning")
        ids = [r.item_id for r in results]
        assert "2" in ids
        assert "5" in ids

    def test_search_preserves_content(self, populated_index):
        results = populated_index.search("python")
        assert "Python" in results[0].content or "python" in results[0].content.lower()


# ---------------------------------------------------------------------------
# SearchIndex - search_with_highlight
# ---------------------------------------------------------------------------


class TestSearchWithHighlight:
    @pytest.fixture()
    def index(self):
        idx = SearchIndex()
        idx.add_document("1", "The quick brown fox jumps over the lazy dog", "input")
        idx.add_document("2", "Python is great for data science", "input")
        yield idx
        idx.close()

    def test_highlight_returns_snippets(self, index):
        results = index.search_with_highlight("fox")
        assert len(results) >= 1
        assert results[0].snippet != ""

    def test_highlight_contains_markers(self, index):
        results = index.search_with_highlight("python")
        assert len(results) >= 1
        snippet = results[0].snippet
        assert "[[B]]" in snippet and "[[/B]]" in snippet

    def test_highlight_empty_query(self, index):
        results = index.search_with_highlight("")
        assert len(results) == 0

    def test_highlight_scores_positive(self, index):
        results = index.search_with_highlight("data")
        for r in results:
            assert r.score > 0

    def test_highlight_limit(self, index):
        results = index.search_with_highlight("the", limit=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# build_index_from_traces
# ---------------------------------------------------------------------------


class TestBuildIndexFromTraces:
    def test_basic_traces(self):
        traces = [
            {"id": "t1", "input": "What is AI?", "output": "AI is artificial intelligence."},
            {"id": "t2", "input": "Hello", "output": "Hi there"},
        ]
        idx = SearchIndex()
        count = build_index_from_traces(traces, idx)
        assert count == 4  # 2 traces x 2 fields
        assert idx.count() == 4
        idx.close()

    def test_missing_output(self):
        traces = [{"id": "t1", "input": "Question"}]
        idx = SearchIndex()
        count = build_index_from_traces(traces, idx)
        assert count == 1
        idx.close()

    def test_missing_input(self):
        traces = [{"id": "t1", "output": "Answer"}]
        idx = SearchIndex()
        count = build_index_from_traces(traces, idx)
        assert count == 1
        idx.close()

    def test_empty_traces(self):
        idx = SearchIndex()
        count = build_index_from_traces([], idx)
        assert count == 0
        assert idx.count() == 0
        idx.close()

    def test_custom_fields(self):
        traces = [{"id": "t1", "prompt": "Hello", "response": "World"}]
        idx = SearchIndex()
        count = build_index_from_traces(
            traces, idx, input_field="prompt", output_field="response"
        )
        assert count == 2
        results = idx.search("Hello")
        assert len(results) >= 1
        assert results[0].source_field == "prompt"
        idx.close()

    def test_item_id_fallback(self):
        traces = [{"item_id": "fallback1", "input": "data"}]
        idx = SearchIndex()
        build_index_from_traces(traces, idx)
        results = idx.search("data")
        assert results[0].item_id == "fallback1"
        idx.close()

    def test_searchable_after_build(self):
        traces = [
            {"id": "t1", "input": "quantum computing basics"},
            {"id": "t2", "input": "classical music theory"},
        ]
        idx = SearchIndex()
        build_index_from_traces(traces, idx)
        results = idx.search("quantum")
        assert len(results) == 1
        assert results[0].item_id == "t1"
        idx.close()


# ---------------------------------------------------------------------------
# format_search_results
# ---------------------------------------------------------------------------


class TestFormatSearchResults:
    def test_empty_results(self):
        assert format_search_results([]) == "No results found."

    def test_single_result(self):
        results = [SearchResult("1", "hello world", 1.5, "", "input")]
        text = format_search_results(results)
        assert "1." in text
        assert "[1]" in text
        assert "1.5000" in text
        assert "input" in text
        assert "hello world" in text

    def test_snippet_preferred_over_content(self):
        results = [SearchResult("1", "full content", 1.0, "short snippet", "input")]
        text = format_search_results(results)
        assert "short snippet" in text

    def test_long_content_truncated(self):
        long_content = "x" * 500
        results = [SearchResult("1", long_content, 1.0, "", "input")]
        text = format_search_results(results)
        assert "..." in text

    def test_multiple_results_numbered(self):
        results = [
            SearchResult("a", "first", 2.0, "", "input"),
            SearchResult("b", "second", 1.0, "", "output"),
        ]
        text = format_search_results(results)
        assert "1." in text
        assert "2." in text


# ---------------------------------------------------------------------------
# search_and_filter
# ---------------------------------------------------------------------------


class TestSearchAndFilter:
    @pytest.fixture()
    def index(self):
        idx = SearchIndex()
        docs = [
            ("1", "machine learning model training", "input"),
            ("2", "model evaluation metrics", "output"),
            ("3", "training data preprocessing", "input"),
        ]
        idx.add_documents(docs)
        yield idx
        idx.close()

    def test_filter_by_source_field(self, index):
        results = search_and_filter(index, "model", source_field="input")
        assert all(r.source_field == "input" for r in results)

    def test_filter_output_only(self, index):
        results = search_and_filter(index, "model", source_field="output")
        assert all(r.source_field == "output" for r in results)

    def test_no_filter(self, index):
        results = search_and_filter(index, "model", source_field=None)
        assert len(results) >= 2

    def test_filter_no_match(self, index):
        results = search_and_filter(index, "training", source_field="output")
        assert len(results) == 0

    def test_filter_respects_limit(self, index):
        results = search_and_filter(index, "training", source_field="input", limit=1)
        assert len(results) <= 1

    def test_filter_empty_query(self, index):
        results = search_and_filter(index, "", source_field="input")
        assert len(results) == 0
