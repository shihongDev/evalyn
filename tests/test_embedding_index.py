"""Tests for the dataset embedding index module."""

from __future__ import annotations

import json
import os
import tempfile

from evalyn_sdk.embedding_index import (
    EmbeddingEntry,
    EmbeddingIndex,
    build_index_from_dataset,
    compute_feature_hash,
    extract_features,
    format_index_stats,
)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_basic(self):
        features = extract_features("hello world")
        assert features == ["hello", "world"]

    def test_lowercased(self):
        features = extract_features("Hello World")
        assert features == ["hello", "world"]

    def test_min_length_2(self):
        features = extract_features("I am a big dog")
        assert "i" not in features
        assert "a" not in features
        assert "am" in features
        assert "big" in features
        assert "dog" in features

    def test_sorted(self):
        features = extract_features("zebra apple mango")
        assert features == ["apple", "mango", "zebra"]

    def test_deduplication(self):
        features = extract_features("the the the cat cat")
        assert features == ["cat", "the"]

    def test_empty_string(self):
        features = extract_features("")
        assert features == []

    def test_punctuation_stripped(self):
        features = extract_features("hello, world! how are you?")
        assert "hello" in features
        assert "world" in features
        # Single-char "," etc. should not appear
        assert "," not in features

    def test_numbers_included(self):
        features = extract_features("version 42 is great")
        assert "42" in features
        assert "version" in features

    def test_single_char_words_excluded(self):
        features = extract_features("x y z ab cd")
        assert "x" not in features
        assert "ab" in features


# ---------------------------------------------------------------------------
# compute_feature_hash
# ---------------------------------------------------------------------------


class TestComputeFeatureHash:
    def test_deterministic(self):
        features = ["alpha", "beta"]
        h1 = compute_feature_hash(features)
        h2 = compute_feature_hash(features)
        assert h1 == h2

    def test_different_features_different_hash(self):
        h1 = compute_feature_hash(["alpha", "beta"])
        h2 = compute_feature_hash(["gamma", "delta"])
        assert h1 != h2

    def test_order_matters(self):
        h1 = compute_feature_hash(["alpha", "beta"])
        h2 = compute_feature_hash(["beta", "alpha"])
        assert h1 != h2

    def test_sha256_length(self):
        h = compute_feature_hash(["test"])
        assert len(h) == 64

    def test_empty_features(self):
        h = compute_feature_hash([])
        assert len(h) == 64


# ---------------------------------------------------------------------------
# EmbeddingEntry
# ---------------------------------------------------------------------------


class TestEmbeddingEntry:
    def test_creation(self):
        e = EmbeddingEntry(item_id="i1", features=["cat", "dog"], feature_hash="abc")
        assert e.item_id == "i1"
        assert e.features == ["cat", "dog"]
        assert e.feature_hash == "abc"

    def test_as_dict(self):
        e = EmbeddingEntry(item_id="i2", features=["x"], feature_hash="h")
        d = e.as_dict()
        assert d["item_id"] == "i2"
        assert d["features"] == ["x"]
        assert d["feature_hash"] == "h"

    def test_from_dict(self):
        d = {"item_id": "i3", "features": ["a", "b"], "feature_hash": "hh"}
        e = EmbeddingEntry.from_dict(d)
        assert e.item_id == "i3"
        assert e.features == ["a", "b"]

    def test_from_dict_defaults(self):
        e = EmbeddingEntry.from_dict({"item_id": "i4"})
        assert e.features == []
        assert e.feature_hash == ""

    def test_roundtrip(self):
        e = EmbeddingEntry(item_id="rt", features=["m", "n"], feature_hash="xyz")
        restored = EmbeddingEntry.from_dict(e.as_dict())
        assert restored.item_id == e.item_id
        assert restored.features == e.features
        assert restored.feature_hash == e.feature_hash


# ---------------------------------------------------------------------------
# EmbeddingIndex - add / get / size
# ---------------------------------------------------------------------------


class TestEmbeddingIndexBasic:
    def test_empty_index(self):
        idx = EmbeddingIndex()
        assert idx.size() == 0

    def test_add_and_get(self):
        idx = EmbeddingIndex()
        idx.add("doc1", "the quick brown fox")
        entry = idx.get("doc1")
        assert entry is not None
        assert entry.item_id == "doc1"
        assert "quick" in entry.features
        assert "brown" in entry.features

    def test_get_missing(self):
        idx = EmbeddingIndex()
        assert idx.get("nope") is None

    def test_size_after_adds(self):
        idx = EmbeddingIndex()
        idx.add("a", "text a")
        idx.add("b", "text b")
        assert idx.size() == 2

    def test_overwrite_existing(self):
        idx = EmbeddingIndex()
        idx.add("a", "first version")
        idx.add("a", "second version different words")
        assert idx.size() == 1
        entry = idx.get("a")
        assert "second" in entry.features

    def test_add_batch(self):
        idx = EmbeddingIndex()
        count = idx.add_batch({"x": "hello world", "y": "foo bar"})
        assert count == 2
        assert idx.size() == 2
        assert idx.get("x") is not None
        assert idx.get("y") is not None

    def test_add_batch_empty(self):
        idx = EmbeddingIndex()
        count = idx.add_batch({})
        assert count == 0
        assert idx.size() == 0


# ---------------------------------------------------------------------------
# EmbeddingIndex - nearest_neighbors
# ---------------------------------------------------------------------------


class TestNearestNeighbors:
    def test_exact_match(self):
        idx = EmbeddingIndex()
        idx.add("a", "quick brown fox")
        idx.add("b", "lazy green turtle")
        results = idx.nearest_neighbors("quick brown fox", k=1)
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][1] == 1.0

    def test_partial_overlap(self):
        idx = EmbeddingIndex()
        idx.add("a", "quick brown fox")
        idx.add("b", "quick brown dog")
        results = idx.nearest_neighbors("quick brown fox", k=2)
        assert results[0][0] == "a"
        assert results[0][1] > results[1][1]

    def test_k_limits_results(self):
        idx = EmbeddingIndex()
        for i in range(10):
            idx.add(f"doc{i}", f"word{i} common text")
        results = idx.nearest_neighbors("common text", k=3)
        assert len(results) == 3

    def test_empty_index(self):
        idx = EmbeddingIndex()
        results = idx.nearest_neighbors("anything", k=5)
        assert results == []

    def test_sorted_descending(self):
        idx = EmbeddingIndex()
        idx.add("a", "alpha beta gamma")
        idx.add("b", "alpha beta")
        idx.add("c", "alpha")
        results = idx.nearest_neighbors("alpha beta gamma", k=3)
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_no_overlap(self):
        idx = EmbeddingIndex()
        idx.add("a", "cat dog bird")
        results = idx.nearest_neighbors("xyz qqq rrr", k=1)
        assert len(results) == 1
        assert results[0][1] == 0.0


# ---------------------------------------------------------------------------
# EmbeddingIndex - find_duplicates
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    def test_exact_duplicates(self):
        idx = EmbeddingIndex()
        idx.add("a", "hello world foo")
        idx.add("b", "hello world foo")
        dupes = idx.find_duplicates(threshold=0.95)
        assert len(dupes) == 1
        assert dupes[0][2] == 1.0

    def test_no_duplicates(self):
        idx = EmbeddingIndex()
        idx.add("a", "alpha beta gamma")
        idx.add("b", "delta epsilon zeta")
        dupes = idx.find_duplicates(threshold=0.95)
        assert dupes == []

    def test_threshold_respected(self):
        idx = EmbeddingIndex()
        idx.add("a", "word1 word2 word3 word4 word5")
        idx.add("b", "word1 word2 word3 word4 word6")
        # Jaccard = 4/6 ~ 0.667
        assert idx.find_duplicates(threshold=0.9) == []
        low_thresh = idx.find_duplicates(threshold=0.5)
        assert len(low_thresh) == 1

    def test_empty_index(self):
        idx = EmbeddingIndex()
        assert idx.find_duplicates() == []

    def test_single_item(self):
        idx = EmbeddingIndex()
        idx.add("only", "some text here")
        assert idx.find_duplicates() == []

    def test_sorted_by_similarity(self):
        idx = EmbeddingIndex()
        idx.add("a", "word1 word2 word3")
        idx.add("b", "word1 word2 word3")  # sim = 1.0
        idx.add("c", "word1 word2 word4")  # sim with a = 2/4 = 0.5
        dupes = idx.find_duplicates(threshold=0.4)
        if len(dupes) > 1:
            sims = [d[2] for d in dupes]
            assert sims == sorted(sims, reverse=True)


# ---------------------------------------------------------------------------
# EmbeddingIndex - save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_from_file(self):
        idx = EmbeddingIndex()
        idx.add("d1", "hello world")
        idx.add("d2", "foo bar baz")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            idx.save(path)
            loaded = EmbeddingIndex.load_from_file(path)
            assert loaded.size() == 2
            assert loaded.get("d1") is not None
            assert loaded.get("d2") is not None
            assert loaded.get("d1").features == idx.get("d1").features
        finally:
            os.unlink(path)

    def test_save_creates_valid_json(self):
        idx = EmbeddingIndex()
        idx.add("x", "test data")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            idx.save(path)
            with open(path) as fh:
                data = json.load(fh)
            assert "entries" in data
            assert len(data["entries"]) == 1
        finally:
            os.unlink(path)

    def test_instance_load(self):
        idx = EmbeddingIndex()
        idx.add("d1", "hello world")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            idx.save(path)
            idx2 = EmbeddingIndex()
            idx2.load(path)
            assert idx2.size() == 1
            assert idx2.get("d1") is not None
        finally:
            os.unlink(path)

    def test_load_preserves_feature_hash(self):
        idx = EmbeddingIndex()
        idx.add("item", "some important text here")
        original_hash = idx.get("item").feature_hash

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            idx.save(path)
            loaded = EmbeddingIndex.load_from_file(path)
            assert loaded.get("item").feature_hash == original_hash
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# build_index_from_dataset
# ---------------------------------------------------------------------------


class TestBuildIndexFromDataset:
    def test_basic(self):
        items = {"a": "hello world", "b": "foo bar"}
        idx = build_index_from_dataset(items)
        assert idx.size() == 2
        assert idx.get("a") is not None

    def test_empty(self):
        idx = build_index_from_dataset({})
        assert idx.size() == 0

    def test_single_item(self):
        idx = build_index_from_dataset({"only": "single document here"})
        assert idx.size() == 1


# ---------------------------------------------------------------------------
# format_index_stats
# ---------------------------------------------------------------------------


class TestFormatIndexStats:
    def test_empty_index(self):
        idx = EmbeddingIndex()
        result = format_index_stats(idx)
        assert "0 entries" in result

    def test_nonempty_index(self):
        idx = EmbeddingIndex()
        idx.add("a", "hello world")
        idx.add("b", "foo bar baz qux quux")
        result = format_index_stats(idx)
        assert "Entries:" in result
        assert "2" in result
        assert "Avg features" in result

    def test_stats_values(self):
        idx = EmbeddingIndex()
        idx.add("a", "ab cd")  # 2 features
        idx.add("b", "ef gh ij kl")  # 4 features
        result = format_index_stats(idx)
        assert "Min features: 2" in result
        assert "Max features: 4" in result
        assert "Avg features: 3.0" in result

    def test_single_entry_stats(self):
        idx = EmbeddingIndex()
        idx.add("x", "one two three")
        result = format_index_stats(idx)
        assert "1" in result
        assert "Min features" in result
