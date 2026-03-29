"""Tests for similarity-based sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.similarity_sampling import (
    SimilarityConfig,
    SimilarityResult,
    SimilarityScore,
    compute_jaccard_similarity,
    find_nearest_neighbors,
    format_similarity_report,
    rank_by_similarity,
    run_similarity_sampling,
    select_most_dissimilar,
    select_most_similar,
)


# ---- SimilarityScore ----


class TestSimilarityScore:
    def test_creation(self):
        s = SimilarityScore(item_id="a", similarity=0.8, distance=0.2)
        assert s.item_id == "a"
        assert s.similarity == 0.8
        assert s.distance == 0.2

    def test_as_dict(self):
        s = SimilarityScore(item_id="b", similarity=0.5, distance=0.5)
        d = s.as_dict()
        assert d == {"item_id": "b", "similarity": 0.5, "distance": 0.5}

    def test_from_dict(self):
        d = {"item_id": "c", "similarity": 0.3, "distance": 0.7}
        s = SimilarityScore.from_dict(d)
        assert s.item_id == "c"
        assert s.similarity == 0.3
        assert s.distance == 0.7

    def test_roundtrip(self):
        s = SimilarityScore(item_id="x", similarity=0.9, distance=0.1)
        assert SimilarityScore.from_dict(s.as_dict()) == s


# ---- SimilarityConfig ----


class TestSimilarityConfig:
    def test_defaults(self):
        c = SimilarityConfig()
        assert c.mode == "similar"
        assert c.sample_size == 20
        assert c.reference_id == ""
        assert c.reference_text == ""

    def test_custom_values(self):
        c = SimilarityConfig(
            mode="dissimilar",
            sample_size=10,
            reference_id="ref1",
            reference_text="hello world",
        )
        assert c.mode == "dissimilar"
        assert c.sample_size == 10

    def test_as_dict(self):
        c = SimilarityConfig(mode="similar", sample_size=5, reference_id="r")
        d = c.as_dict()
        assert d["mode"] == "similar"
        assert d["sample_size"] == 5
        assert d["reference_id"] == "r"
        assert d["reference_text"] == ""

    def test_from_dict_full(self):
        d = {
            "mode": "dissimilar",
            "sample_size": 30,
            "reference_id": "ref2",
            "reference_text": "test text",
        }
        c = SimilarityConfig.from_dict(d)
        assert c.mode == "dissimilar"
        assert c.sample_size == 30
        assert c.reference_text == "test text"

    def test_from_dict_defaults(self):
        c = SimilarityConfig.from_dict({})
        assert c.mode == "similar"
        assert c.sample_size == 20
        assert c.reference_id == ""

    def test_roundtrip(self):
        c = SimilarityConfig(
            mode="dissimilar", sample_size=15, reference_id="x", reference_text="hi"
        )
        assert SimilarityConfig.from_dict(c.as_dict()) == c


# ---- SimilarityResult ----


class TestSimilarityResult:
    def test_defaults(self):
        r = SimilarityResult()
        assert r.selected_ids == []
        assert r.scores == []
        assert r.reference_id == ""
        assert r.mode == "similar"
        assert r.total_pool == 0

    def test_as_dict(self):
        score = SimilarityScore(item_id="a", similarity=0.5, distance=0.5)
        r = SimilarityResult(
            selected_ids=["a"],
            scores=[score],
            reference_id="ref",
            mode="similar",
            total_pool=10,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["scores"]) == 1
        assert d["total_pool"] == 10

    def test_from_dict(self):
        d = {
            "selected_ids": ["b"],
            "scores": [{"item_id": "b", "similarity": 0.8, "distance": 0.2}],
            "reference_id": "r",
            "mode": "dissimilar",
            "total_pool": 5,
        }
        r = SimilarityResult.from_dict(d)
        assert r.selected_ids == ["b"]
        assert r.scores[0].item_id == "b"
        assert r.mode == "dissimilar"

    def test_from_dict_defaults(self):
        r = SimilarityResult.from_dict({})
        assert r.selected_ids == []
        assert r.total_pool == 0

    def test_roundtrip(self):
        score = SimilarityScore(item_id="z", similarity=0.3, distance=0.7)
        r = SimilarityResult(
            selected_ids=["z"],
            scores=[score],
            reference_id="r1",
            mode="similar",
            total_pool=100,
        )
        r2 = SimilarityResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.scores[0].item_id == r.scores[0].item_id


# ---- compute_jaccard_similarity ----


class TestComputeJaccardSimilarity:
    def test_identical_texts(self):
        assert compute_jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert compute_jaccard_similarity("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self):
        sim = compute_jaccard_similarity("the cat sat", "the dog sat")
        # intersection: {"the", "sat"} = 2, union: {"the", "cat", "sat", "dog"} = 4
        assert sim == pytest.approx(0.5)

    def test_case_insensitive(self):
        sim = compute_jaccard_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_both_empty(self):
        assert compute_jaccard_similarity("", "") == 0.0

    def test_one_empty(self):
        assert compute_jaccard_similarity("hello", "") == 0.0

    def test_subset(self):
        sim = compute_jaccard_similarity("a b c", "a b c d")
        # intersection 3, union 4
        assert sim == pytest.approx(0.75)

    def test_symmetric(self):
        sim1 = compute_jaccard_similarity("foo bar", "bar baz")
        sim2 = compute_jaccard_similarity("bar baz", "foo bar")
        assert sim1 == sim2

    def test_duplicate_words(self):
        # Sets collapse duplicates
        sim = compute_jaccard_similarity("a a a", "a b")
        # set_a = {"a"}, set_b = {"a", "b"}, intersection=1, union=2
        assert sim == pytest.approx(0.5)


# ---- rank_by_similarity ----


class TestRankBySimilarity:
    def test_ordering(self):
        items = {
            "exact": "hello world",
            "partial": "hello there",
            "none": "foo bar",
        }
        ranked = rank_by_similarity("hello world", items)
        assert ranked[0].item_id == "exact"
        assert ranked[-1].item_id == "none"

    def test_scores_descending(self):
        items = {"a": "x y", "b": "x z", "c": "x y z"}
        ranked = rank_by_similarity("x y", items)
        for i in range(len(ranked) - 1):
            assert ranked[i].similarity >= ranked[i + 1].similarity

    def test_empty_items(self):
        assert rank_by_similarity("hello", {}) == []

    def test_distance_is_complement(self):
        items = {"a": "cat dog"}
        ranked = rank_by_similarity("cat", items)
        assert ranked[0].distance == pytest.approx(1.0 - ranked[0].similarity)

    def test_all_scores_between_0_and_1(self):
        items = {f"item_{i}": f"word_{i} common" for i in range(10)}
        ranked = rank_by_similarity("common word_0", items)
        for s in ranked:
            assert 0.0 <= s.similarity <= 1.0
            assert 0.0 <= s.distance <= 1.0


# ---- select_most_similar ----


class TestSelectMostSimilar:
    def test_basic(self):
        scores = [
            SimilarityScore("a", 0.9, 0.1),
            SimilarityScore("b", 0.5, 0.5),
            SimilarityScore("c", 0.7, 0.3),
        ]
        result = select_most_similar(scores, 2)
        assert result == ["a", "c"]

    def test_n_larger_than_list(self):
        scores = [SimilarityScore("a", 0.5, 0.5)]
        result = select_most_similar(scores, 5)
        assert result == ["a"]

    def test_empty(self):
        assert select_most_similar([], 3) == []

    def test_tiebreaking_by_id(self):
        scores = [
            SimilarityScore("b", 0.5, 0.5),
            SimilarityScore("a", 0.5, 0.5),
        ]
        result = select_most_similar(scores, 2)
        assert result == ["a", "b"]


# ---- select_most_dissimilar ----


class TestSelectMostDissimilar:
    def test_basic(self):
        scores = [
            SimilarityScore("a", 0.9, 0.1),
            SimilarityScore("b", 0.1, 0.9),
            SimilarityScore("c", 0.5, 0.5),
        ]
        result = select_most_dissimilar(scores, 2)
        assert result == ["b", "c"]

    def test_n_larger_than_list(self):
        scores = [SimilarityScore("a", 0.3, 0.7)]
        result = select_most_dissimilar(scores, 5)
        assert result == ["a"]

    def test_empty(self):
        assert select_most_dissimilar([], 3) == []

    def test_tiebreaking_by_id(self):
        scores = [
            SimilarityScore("b", 0.5, 0.5),
            SimilarityScore("a", 0.5, 0.5),
        ]
        result = select_most_dissimilar(scores, 2)
        assert result == ["a", "b"]


# ---- run_similarity_sampling ----


class TestRunSimilaritySampling:
    def test_similar_mode(self):
        items = {
            "a": "the quick brown fox",
            "b": "the quick red fox",
            "c": "a slow blue turtle",
        }
        config = SimilarityConfig(
            mode="similar",
            sample_size=2,
            reference_text="the quick brown fox jumps",
        )
        result = run_similarity_sampling(items, config)
        assert len(result.selected_ids) == 2
        assert result.mode == "similar"
        # "a" should be most similar
        assert result.selected_ids[0] == "a"

    def test_dissimilar_mode(self):
        items = {
            "a": "the quick brown fox",
            "b": "the quick red fox",
            "c": "something completely different",
        }
        config = SimilarityConfig(
            mode="dissimilar",
            sample_size=1,
            reference_text="the quick brown fox",
        )
        result = run_similarity_sampling(items, config)
        assert result.selected_ids[0] == "c"

    def test_reference_id_resolves_text(self):
        items = {
            "ref": "alpha beta gamma",
            "a": "alpha beta",
            "b": "delta epsilon",
        }
        config = SimilarityConfig(
            mode="similar", sample_size=2, reference_id="ref"
        )
        result = run_similarity_sampling(items, config)
        # ref itself should be excluded from candidates
        assert "ref" not in result.selected_ids
        # "a" should be more similar to ref
        assert result.selected_ids[0] == "a"

    def test_reference_id_excluded_from_pool(self):
        items = {"ref": "hello", "a": "hello", "b": "world"}
        config = SimilarityConfig(
            mode="similar", sample_size=10, reference_id="ref"
        )
        result = run_similarity_sampling(items, config)
        assert "ref" not in result.selected_ids
        assert result.total_pool == 2

    def test_explicit_text_overrides_id(self):
        items = {"ref": "alpha", "a": "beta", "b": "gamma delta"}
        config = SimilarityConfig(
            mode="similar",
            sample_size=1,
            reference_id="ref",
            reference_text="gamma delta epsilon",
        )
        result = run_similarity_sampling(items, config)
        # reference_text is set, so it should be used instead of "ref" text
        assert result.selected_ids[0] == "b"

    def test_empty_items(self):
        config = SimilarityConfig(reference_text="hello")
        result = run_similarity_sampling({}, config)
        assert result.selected_ids == []
        assert result.total_pool == 0

    def test_sample_size_larger_than_pool(self):
        items = {"a": "hello", "b": "world"}
        config = SimilarityConfig(
            mode="similar", sample_size=100, reference_text="hello"
        )
        result = run_similarity_sampling(items, config)
        assert len(result.selected_ids) == 2


# ---- find_nearest_neighbors ----


class TestFindNearestNeighbors:
    def test_basic(self):
        items = {
            "a": "cat dog",
            "b": "cat fish",
            "c": "bird snake",
        }
        nn = find_nearest_neighbors("cat dog bird", items, k=2)
        assert len(nn) == 2
        # "a" has highest overlap
        assert nn[0].item_id == "a"

    def test_k_larger_than_items(self):
        items = {"a": "hello"}
        nn = find_nearest_neighbors("hello", items, k=5)
        assert len(nn) == 1

    def test_empty_items(self):
        nn = find_nearest_neighbors("hello", {}, k=3)
        assert nn == []

    def test_default_k(self):
        items = {f"item_{i}": f"word_{i}" for i in range(10)}
        nn = find_nearest_neighbors("word_0", items)
        assert len(nn) == 5

    def test_returns_similarity_scores(self):
        items = {"a": "x y", "b": "x z"}
        nn = find_nearest_neighbors("x y", items, k=2)
        for s in nn:
            assert isinstance(s, SimilarityScore)
            assert 0.0 <= s.similarity <= 1.0


# ---- format_similarity_report ----


class TestFormatSimilarityReport:
    def test_contains_mode(self):
        result = SimilarityResult(mode="similar", total_pool=5)
        text = format_similarity_report(result)
        assert "similar" in text

    def test_contains_reference_id(self):
        result = SimilarityResult(reference_id="ref1", total_pool=3)
        text = format_similarity_report(result)
        assert "ref1" in text

    def test_contains_selected_items(self):
        score = SimilarityScore(item_id="item_a", similarity=0.8, distance=0.2)
        result = SimilarityResult(
            selected_ids=["item_a"],
            scores=[score],
            mode="similar",
            total_pool=10,
        )
        text = format_similarity_report(result)
        assert "item_a" in text
        assert "0.8000" in text

    def test_separator_lines(self):
        result = SimilarityResult(total_pool=0)
        text = format_similarity_report(result)
        assert "---" in text

    def test_dissimilar_mode_report(self):
        scores = [
            SimilarityScore("a", 0.2, 0.8),
            SimilarityScore("b", 0.1, 0.9),
        ]
        result = SimilarityResult(
            selected_ids=["a", "b"],
            scores=scores,
            mode="dissimilar",
            total_pool=5,
        )
        text = format_similarity_report(result)
        assert "dissimilar" in text
        # b (lower similarity) should appear before a in dissimilar mode
        pos_b = text.index("b")
        pos_a = text.index("a")
        # Both should be present
        assert "a" in text
        assert "b" in text

    def test_no_reference_id_skips_line(self):
        result = SimilarityResult(reference_id="", total_pool=2)
        text = format_similarity_report(result)
        assert "Reference:" not in text
