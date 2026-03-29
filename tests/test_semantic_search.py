"""Tests for semantic_search module."""

from __future__ import annotations

import pytest

from evalyn_sdk.semantic_search import (
    SearchConfig,
    SearchHit,
    SearchResult,
    compute_similarity,
    find_matched_terms,
    format_search_results,
    search_dataset,
    search_with_filters,
    tokenize_query,
)


# ---------------------------------------------------------------------------
# SearchConfig tests
# ---------------------------------------------------------------------------


class TestSearchConfig:
    def test_defaults(self):
        c = SearchConfig()
        assert c.max_results == 10
        assert c.min_similarity == 0.1
        assert c.boost_exact_match is True

    def test_as_dict(self):
        c = SearchConfig(max_results=5, min_similarity=0.3, boost_exact_match=False)
        d = c.as_dict()
        assert d == {
            "max_results": 5,
            "min_similarity": 0.3,
            "boost_exact_match": False,
        }

    def test_from_dict(self):
        c = SearchConfig.from_dict({"max_results": 3})
        assert c.max_results == 3
        assert c.min_similarity == 0.1  # default

    def test_roundtrip(self):
        c = SearchConfig(max_results=7, min_similarity=0.5, boost_exact_match=False)
        assert SearchConfig.from_dict(c.as_dict()) == c


# ---------------------------------------------------------------------------
# SearchHit tests
# ---------------------------------------------------------------------------


class TestSearchHit:
    def test_basic(self):
        h = SearchHit(item_id="1", text="hello", similarity=0.5)
        assert h.matched_terms == []

    def test_as_dict(self):
        h = SearchHit(
            item_id="1",
            text="t",
            similarity=0.8,
            matched_terms=["foo", "bar"],
        )
        d = h.as_dict()
        assert d["matched_terms"] == ["foo", "bar"]

    def test_from_dict(self):
        h = SearchHit.from_dict(
            {"item_id": "x", "text": "y", "similarity": 0.3}
        )
        assert h.matched_terms == []

    def test_roundtrip(self):
        h = SearchHit(
            item_id="id1", text="txt", similarity=0.9, matched_terms=["a"]
        )
        assert SearchHit.from_dict(h.as_dict()) == h


# ---------------------------------------------------------------------------
# SearchResult tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult()
        assert r.hits == []
        assert r.query == ""
        assert r.total_searched == 0

    def test_as_dict(self):
        r = SearchResult(
            hits=[SearchHit(item_id="1", text="t", similarity=0.5)],
            query="q",
            total_searched=10,
        )
        d = r.as_dict()
        assert len(d["hits"]) == 1
        assert d["total_searched"] == 10

    def test_from_dict_empty(self):
        r = SearchResult.from_dict({})
        assert r.hits == []

    def test_roundtrip(self):
        r = SearchResult(
            hits=[SearchHit(item_id="a", text="b", similarity=0.4, matched_terms=["b"])],
            query="test",
            total_searched=5,
        )
        r2 = SearchResult.from_dict(r.as_dict())
        assert len(r2.hits) == 1
        assert r2.query == "test"


# ---------------------------------------------------------------------------
# tokenize_query tests
# ---------------------------------------------------------------------------


class TestTokenizeQuery:
    def test_basic(self):
        tokens = tokenize_query("Hello World")
        assert tokens == {"hello", "world"}

    def test_filters_short(self):
        tokens = tokenize_query("I am a dog")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "dog" in tokens

    def test_empty(self):
        assert tokenize_query("") == set()

    def test_special_chars(self):
        tokens = tokenize_query("hello! world?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_numbers(self):
        tokens = tokenize_query("version 42 released")
        assert "42" in tokens
        assert "version" in tokens

    def test_lowercase(self):
        tokens = tokenize_query("UPPER lower MiXeD")
        assert "upper" in tokens
        assert "lower" in tokens
        assert "mixed" in tokens

    def test_single_char_filtered(self):
        tokens = tokenize_query("x y z abc")
        assert "x" not in tokens
        assert "abc" in tokens


# ---------------------------------------------------------------------------
# compute_similarity tests
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def test_identical(self):
        s = compute_similarity({"a", "b"}, {"a", "b"})
        assert s == 1.0

    def test_disjoint(self):
        s = compute_similarity({"a", "b"}, {"c", "d"})
        assert s == 0.0

    def test_partial_overlap(self):
        s = compute_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection={b,c}, union={a,b,c,d} -> 2/4=0.5
        assert s == pytest.approx(0.5)

    def test_empty_query(self):
        assert compute_similarity(set(), {"a"}) == 0.0

    def test_empty_item(self):
        assert compute_similarity({"a"}, set()) == 0.0

    def test_both_empty(self):
        assert compute_similarity(set(), set()) == 0.0

    def test_subset(self):
        s = compute_similarity({"a"}, {"a", "b"})
        # 1/2 = 0.5
        assert s == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# find_matched_terms tests
# ---------------------------------------------------------------------------


class TestFindMatchedTerms:
    def test_intersection(self):
        result = find_matched_terms({"a", "b", "c"}, {"b", "c", "d"})
        assert result == ["b", "c"]

    def test_no_overlap(self):
        result = find_matched_terms({"a"}, {"b"})
        assert result == []

    def test_sorted(self):
        result = find_matched_terms({"z", "a", "m"}, {"m", "z"})
        assert result == ["m", "z"]


# ---------------------------------------------------------------------------
# search_dataset tests
# ---------------------------------------------------------------------------


class TestSearchDataset:
    def _items(self):
        return {
            "doc1": "machine learning is a type of artificial intelligence",
            "doc2": "the cat sat on the mat",
            "doc3": "deep learning and neural networks for AI",
            "doc4": "cooking recipes for Italian food",
        }

    def test_basic_search(self):
        result = search_dataset("machine learning", self._items())
        assert result.total_searched == 4
        assert len(result.hits) >= 1
        assert result.hits[0].item_id == "doc1"

    def test_query_stored(self):
        result = search_dataset("test query", self._items())
        assert result.query == "test query"

    def test_min_similarity_filter(self):
        config = SearchConfig(min_similarity=0.99)
        result = search_dataset("totally unrelated query xyz", self._items(), config)
        assert len(result.hits) == 0

    def test_max_results(self):
        config = SearchConfig(max_results=1)
        result = search_dataset("learning", self._items(), config)
        assert len(result.hits) <= 1

    def test_exact_match_boost(self):
        items = {
            "a": "machine learning rocks",
            "b": "machine and learning separate",
        }
        config = SearchConfig(boost_exact_match=True)
        result = search_dataset("machine learning", items, config)
        # "a" contains the exact substring "machine learning"
        assert result.hits[0].item_id == "a"

    def test_no_boost(self):
        items = {
            "a": "machine learning rocks",
            "b": "machine learning rocks too",
        }
        config = SearchConfig(boost_exact_match=False)
        result = search_dataset("machine learning", items, config)
        # Both have exact match but boost is off - result should still work
        assert len(result.hits) >= 1

    def test_empty_items(self):
        result = search_dataset("query", {})
        assert result.hits == []
        assert result.total_searched == 0

    def test_empty_query(self):
        result = search_dataset("", self._items())
        assert result.hits == []

    def test_matched_terms_populated(self):
        result = search_dataset("machine learning", self._items())
        hit = result.hits[0]
        assert "machine" in hit.matched_terms
        assert "learning" in hit.matched_terms

    def test_results_sorted_by_similarity(self):
        result = search_dataset("learning artificial intelligence", self._items())
        sims = [h.similarity for h in result.hits]
        assert sims == sorted(sims, reverse=True)


# ---------------------------------------------------------------------------
# search_with_filters tests
# ---------------------------------------------------------------------------


class TestSearchWithFilters:
    def _items(self):
        return {
            "doc1": "machine learning tutorial",
            "doc2": "cooking recipe for pasta",
            "doc3": "deep learning guide",
        }

    def _metadata(self):
        return {
            "doc1": {"category": "tech", "level": "beginner"},
            "doc2": {"category": "food", "level": "intermediate"},
            "doc3": {"category": "tech", "level": "advanced"},
        }

    def test_no_filter(self):
        result = search_with_filters("learning", self._items())
        assert result.total_searched == 3

    def test_filter_by_category(self):
        result = search_with_filters(
            "learning",
            self._items(),
            metadata=self._metadata(),
            filter_field="category",
            filter_value="tech",
        )
        assert result.total_searched == 2
        ids = {h.item_id for h in result.hits}
        assert "doc2" not in ids

    def test_filter_no_match(self):
        result = search_with_filters(
            "learning",
            self._items(),
            metadata=self._metadata(),
            filter_field="category",
            filter_value="sports",
        )
        assert result.total_searched == 0
        assert result.hits == []

    def test_filter_field_empty_means_no_filter(self):
        result = search_with_filters(
            "learning",
            self._items(),
            metadata=self._metadata(),
            filter_field="",
            filter_value="tech",
        )
        assert result.total_searched == 3

    def test_filter_value_empty_means_no_filter(self):
        result = search_with_filters(
            "learning",
            self._items(),
            metadata=self._metadata(),
            filter_field="category",
            filter_value="",
        )
        assert result.total_searched == 3

    def test_with_config(self):
        config = SearchConfig(max_results=1)
        result = search_with_filters(
            "learning",
            self._items(),
            metadata=self._metadata(),
            filter_field="category",
            filter_value="tech",
            config=config,
        )
        assert len(result.hits) <= 1


# ---------------------------------------------------------------------------
# format_search_results tests
# ---------------------------------------------------------------------------


class TestFormatSearchResults:
    def test_no_results(self):
        r = SearchResult(query="test", total_searched=5)
        out = format_search_results(r)
        assert "No results" in out
        assert "test" in out
        assert "5" in out

    def test_with_results(self):
        r = SearchResult(
            hits=[
                SearchHit(
                    item_id="d1",
                    text="hello world",
                    similarity=0.75,
                    matched_terms=["hello"],
                )
            ],
            query="hello",
            total_searched=10,
        )
        out = format_search_results(r)
        assert "hello" in out
        assert "d1" in out
        assert "0.750" in out
        assert "matched:" in out

    def test_long_text_truncated(self):
        long_text = "word " * 50  # 250 chars
        r = SearchResult(
            hits=[
                SearchHit(item_id="x", text=long_text, similarity=0.5)
            ],
            query="word",
            total_searched=1,
        )
        out = format_search_results(r)
        assert "..." in out

    def test_multiple_hits(self):
        r = SearchResult(
            hits=[
                SearchHit(item_id="a", text="first", similarity=0.9),
                SearchHit(item_id="b", text="second", similarity=0.5),
            ],
            query="q",
            total_searched=3,
        )
        out = format_search_results(r)
        assert "1." in out
        assert "2." in out
        assert "2 of 3" in out
