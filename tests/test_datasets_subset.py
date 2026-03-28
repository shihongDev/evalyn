"""Tests for evalyn_sdk.datasets_subset.

Run with: uv run pytest tests/test_datasets_subset.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_subset import (
    ClusterInfo,
    SubsetConfig,
    SubsetResult,
    cluster_by_keywords,
    cluster_by_length,
    extract_subset,
    sample_from_clusters,
    compute_cluster_keywords,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(item_id: str, text: str = "") -> dict:
    return {"id": item_id, "input": text or f"input for item {item_id}"}


def _items_with_topics() -> list:
    """Items with clear keyword groupings."""
    return [
        {"id": "1", "input": "python programming language tutorial"},
        {"id": "2", "input": "python code review best practices"},
        {"id": "3", "input": "python function decorator pattern"},
        {"id": "4", "input": "javascript framework comparison react"},
        {"id": "5", "input": "javascript async await promises"},
        {"id": "6", "input": "javascript dom manipulation tutorial"},
        {"id": "7", "input": "database query optimization indexes"},
        {"id": "8", "input": "database schema migration tool"},
        {"id": "9", "input": "database replication consistency"},
        {"id": "10", "input": "machine learning model training"},
    ]


def _items_varying_length() -> list:
    """Items with clearly different lengths."""
    return [
        {"id": "short1", "input": "hi"},
        {"id": "short2", "input": "ok"},
        {"id": "med1", "input": "a medium length input string here"},
        {"id": "med2", "input": "another medium length input text"},
        {"id": "long1", "input": "this is a much longer input string that contains many words and should end up in a different bucket"},
        {"id": "long2", "input": "another very long input string with lots of content to make it clearly longer than the medium ones"},
    ]


# ---------------------------------------------------------------------------
# ClusterInfo
# ---------------------------------------------------------------------------

class TestClusterInfo:
    def test_as_dict(self):
        info = ClusterInfo(cluster_id=0, size=5, centroid_item_id="c1", keywords=["python"])
        d = info.as_dict()
        assert d["cluster_id"] == 0
        assert d["size"] == 5
        assert d["keywords"] == ["python"]

    def test_defaults(self):
        info = ClusterInfo()
        assert info.cluster_id == 0
        assert info.size == 0
        assert info.centroid_item_id == ""
        assert info.keywords == []


# ---------------------------------------------------------------------------
# SubsetConfig
# ---------------------------------------------------------------------------

class TestSubsetConfig:
    def test_as_dict(self):
        config = SubsetConfig(num_clusters=3, method="length", seed=42)
        d = config.as_dict()
        assert d["num_clusters"] == 3
        assert d["method"] == "length"
        assert d["seed"] == 42

    def test_from_dict(self):
        data = {"num_clusters": 10, "method": "random", "items_per_cluster": 2, "seed": 7}
        config = SubsetConfig.from_dict(data)
        assert config.num_clusters == 10
        assert config.method == "random"
        assert config.items_per_cluster == 2
        assert config.seed == 7

    def test_roundtrip(self):
        original = SubsetConfig(num_clusters=4, method="keyword", items_per_cluster=5, seed=99)
        restored = SubsetConfig.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()

    def test_defaults(self):
        config = SubsetConfig()
        assert config.num_clusters == 5
        assert config.method == "keyword"
        assert config.items_per_cluster == 0
        assert config.seed is None

    def test_from_dict_defaults(self):
        config = SubsetConfig.from_dict({})
        assert config.num_clusters == 5
        assert config.method == "keyword"


# ---------------------------------------------------------------------------
# SubsetResult
# ---------------------------------------------------------------------------

class TestSubsetResult:
    def test_as_dict(self):
        result = SubsetResult(
            subsets={0: [{"id": "a"}], 1: [{"id": "b"}]},
            clusters=[
                ClusterInfo(cluster_id=0, size=1, keywords=["python"]),
                ClusterInfo(cluster_id=1, size=1, keywords=["java"]),
            ],
            total_items=2,
        )
        d = result.as_dict()
        assert d["total_items"] == 2
        # Keys become strings in as_dict
        assert "0" in d["subsets"]

    def test_from_dict(self):
        data = {
            "subsets": {"0": [{"id": "x"}], "1": [{"id": "y"}]},
            "clusters": [
                {"cluster_id": 0, "size": 1, "keywords": ["k1"]},
            ],
            "total_items": 2,
        }
        result = SubsetResult.from_dict(data)
        assert result.total_items == 2
        assert 0 in result.subsets
        assert 1 in result.subsets

    def test_roundtrip(self):
        original = SubsetResult(
            subsets={0: [{"id": "a"}]},
            clusters=[ClusterInfo(cluster_id=0, size=1)],
            total_items=1,
        )
        restored = SubsetResult.from_dict(original.as_dict())
        assert restored.total_items == original.total_items
        assert len(restored.subsets) == len(original.subsets)

    def test_format_text(self):
        result = SubsetResult(
            subsets={0: [{"id": "a"}, {"id": "b"}]},
            clusters=[ClusterInfo(cluster_id=0, size=2, keywords=["python", "code"])],
            total_items=2,
        )
        text = result.format_text()
        assert "1 clusters" in text
        assert "2 total items" in text
        assert "python" in text

    def test_format_text_empty(self):
        result = SubsetResult()
        text = result.format_text()
        assert "0 clusters" in text
        assert "0 total items" in text


# ---------------------------------------------------------------------------
# cluster_by_keywords
# ---------------------------------------------------------------------------

class TestClusterByKeywords:
    def test_groups_similar_items(self):
        items = _items_with_topics()
        result = cluster_by_keywords(items, num_clusters=3)
        assert result.total_items == 10
        assert len(result.clusters) == 3
        # All items should be assigned to some cluster
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 10

    def test_cluster_info_populated(self):
        items = _items_with_topics()
        result = cluster_by_keywords(items, num_clusters=3)
        for cluster in result.clusters:
            assert len(cluster.keywords) > 0

    def test_single_item(self):
        items = [_item("only", "solo item")]
        result = cluster_by_keywords(items, num_clusters=3)
        assert result.total_items == 1
        # num_clusters capped to number of items
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 1

    def test_empty_items(self):
        result = cluster_by_keywords([])
        assert result.total_items == 0
        assert result.clusters == []

    def test_fewer_items_than_clusters(self):
        items = [_item("a", "hello world"), _item("b", "goodbye world")]
        result = cluster_by_keywords(items, num_clusters=10)
        assert result.total_items == 2
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 2


# ---------------------------------------------------------------------------
# cluster_by_length
# ---------------------------------------------------------------------------

class TestClusterByLength:
    def test_creates_buckets(self):
        items = _items_varying_length()
        result = cluster_by_length(items, num_clusters=3)
        assert result.total_items == 6
        assert len(result.clusters) == 3
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 6

    def test_length_keywords_contain_range(self):
        items = _items_varying_length()
        result = cluster_by_length(items, num_clusters=2)
        for cluster in result.clusters:
            assert any("len:" in kw for kw in cluster.keywords)

    def test_empty_items(self):
        result = cluster_by_length([])
        assert result.total_items == 0

    def test_single_item(self):
        items = [_item("one", "hello")]
        result = cluster_by_length(items, num_clusters=3)
        assert result.total_items == 1
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 1

    def test_same_length_items(self):
        items = [{"id": str(i), "input": "same"} for i in range(5)]
        result = cluster_by_length(items, num_clusters=3)
        assert result.total_items == 5
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 5


# ---------------------------------------------------------------------------
# extract_subset
# ---------------------------------------------------------------------------

class TestExtractSubset:
    def test_routes_to_keyword(self):
        items = _items_with_topics()
        config = SubsetConfig(num_clusters=3, method="keyword")
        result = extract_subset(items, config)
        assert result.total_items == 10
        assert len(result.clusters) == 3

    def test_routes_to_length(self):
        items = _items_varying_length()
        config = SubsetConfig(num_clusters=2, method="length")
        result = extract_subset(items, config)
        assert result.total_items == 6
        assert len(result.clusters) == 2

    def test_routes_to_random(self):
        items = [_item(str(i)) for i in range(20)]
        config = SubsetConfig(num_clusters=4, method="random", seed=42)
        result = extract_subset(items, config)
        assert result.total_items == 20
        total_assigned = sum(len(v) for v in result.subsets.values())
        assert total_assigned == 20

    def test_unknown_method_defaults_to_keyword(self):
        items = _items_with_topics()
        config = SubsetConfig(num_clusters=2, method="nonexistent")
        result = extract_subset(items, config)
        assert result.total_items == 10


# ---------------------------------------------------------------------------
# sample_from_clusters
# ---------------------------------------------------------------------------

class TestSampleFromClusters:
    def test_correct_count(self):
        items = [_item(str(i)) for i in range(30)]
        result = cluster_by_keywords(items, num_clusters=3)
        sampled = sample_from_clusters(result, items_per_cluster=2)
        # At most 2 per cluster, at most 6 total
        assert len(sampled) <= 6

    def test_deterministic_with_seed(self):
        items = [_item(str(i), f"word{i % 3} text{i}") for i in range(30)]
        result = cluster_by_keywords(items, num_clusters=3)
        s1 = sample_from_clusters(result, items_per_cluster=2, seed=42)
        s2 = sample_from_clusters(result, items_per_cluster=2, seed=42)
        assert [x["id"] for x in s1] == [x["id"] for x in s2]

    def test_different_seeds_may_differ(self):
        items = [_item(str(i), f"word{i % 5} text{i}") for i in range(50)]
        result = cluster_by_keywords(items, num_clusters=5)
        s1 = sample_from_clusters(result, items_per_cluster=2, seed=1)
        s2 = sample_from_clusters(result, items_per_cluster=2, seed=999)
        # With enough items they should differ (not guaranteed, but very likely)
        ids1 = [x["id"] for x in s1]
        ids2 = [x["id"] for x in s2]
        # At minimum, both produce results
        assert len(ids1) > 0
        assert len(ids2) > 0

    def test_empty_clusters(self):
        result = SubsetResult(subsets={0: [], 1: []}, clusters=[], total_items=0)
        sampled = sample_from_clusters(result, items_per_cluster=3)
        assert sampled == []

    def test_fewer_items_than_requested(self):
        result = SubsetResult(
            subsets={0: [{"id": "a"}]},
            clusters=[ClusterInfo(cluster_id=0, size=1)],
            total_items=1,
        )
        sampled = sample_from_clusters(result, items_per_cluster=10)
        assert len(sampled) == 1


# ---------------------------------------------------------------------------
# compute_cluster_keywords
# ---------------------------------------------------------------------------

class TestComputeClusterKeywords:
    def test_top_words(self):
        items = [
            {"input": "python programming"},
            {"input": "python tutorial"},
            {"input": "python code review"},
        ]
        kw = compute_cluster_keywords(items, top_n=2)
        assert "python" in kw
        assert len(kw) == 2

    def test_empty_items(self):
        kw = compute_cluster_keywords([], top_n=5)
        assert kw == []

    def test_custom_input_field(self):
        items = [
            {"text": "alpha beta gamma"},
            {"text": "alpha delta"},
        ]
        kw = compute_cluster_keywords(items, input_field="text", top_n=3)
        assert "alpha" in kw

    def test_stops_at_top_n(self):
        items = [{"input": "one two three four five six seven"}]
        kw = compute_cluster_keywords(items, top_n=3)
        assert len(kw) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_items_with_no_input_field(self):
        items = [{"id": "1"}, {"id": "2"}]
        result = cluster_by_keywords(items, num_clusters=2)
        # Should not crash; items get empty string as input
        assert result.total_items == 2

    def test_items_with_numeric_input(self):
        items = [{"id": "1", "input": 12345}]
        result = cluster_by_keywords(items, num_clusters=1)
        assert result.total_items == 1

    def test_subset_config_with_items_per_cluster(self):
        items = [_item(str(i), f"topic{i % 3} content") for i in range(30)]
        config = SubsetConfig(num_clusters=3, method="keyword", items_per_cluster=2)
        result = extract_subset(items, config)
        # Each non-empty cluster should have at most 2 items
        for ci, cluster_items in result.subsets.items():
            assert len(cluster_items) <= 2
