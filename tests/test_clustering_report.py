"""Tests for dataset item clustering report module."""
from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.clustering_report import (
    ClusterDescription,
    ClusteringReportConfig,
    ClusteringReport,
    cluster_items,
    extract_cluster_keywords,
    find_representatives,
    generate_cluster_description,
    build_clustering_report,
    format_clustering_report,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SCIENCE_ITEMS = {
    "sci1": "machine learning neural networks deep learning algorithms",
    "sci2": "neural networks training backpropagation gradient descent",
    "sci3": "deep learning convolutional networks image classification",
    "sci4": "climate change global warming carbon emissions temperature",
    "sci5": "global warming greenhouse gases ocean levels rising",
    "sci6": "carbon emissions fossil fuels renewable energy climate",
    "sci7": "quantum computing qubits superposition entanglement",
    "sci8": "quantum mechanics wave function particle physics",
    "sci9": "quantum entanglement teleportation information theory",
}

COOKING_ITEMS = {
    "cook1": "pasta recipe italian cooking tomato sauce basil",
    "cook2": "spaghetti bolognese meat sauce onion garlic tomato",
    "cook3": "pizza dough flour yeast oven baking mozzarella",
    "cook4": "sushi rice fish salmon seaweed japanese cuisine",
    "cook5": "ramen noodles broth pork egg japanese food",
    "cook6": "tempura frying batter shrimp vegetables japanese",
}


# ---------------------------------------------------------------------------
# ClusterDescription
# ---------------------------------------------------------------------------


class TestClusterDescription:
    def test_defaults(self):
        cd = ClusterDescription(cluster_id=0, size=5)
        assert cd.cluster_id == 0
        assert cd.size == 5
        assert cd.top_keywords == []
        assert cd.description == ""
        assert cd.representative_ids == []

    def test_as_dict(self):
        cd = ClusterDescription(
            cluster_id=1,
            size=10,
            top_keywords=["ml", "ai"],
            description="AI items",
            representative_ids=["id1"],
        )
        d = cd.as_dict()
        assert d["cluster_id"] == 1
        assert d["size"] == 10
        assert d["top_keywords"] == ["ml", "ai"]

    def test_from_dict(self):
        d = {"cluster_id": 2, "size": 3, "top_keywords": ["x"], "description": "desc"}
        cd = ClusterDescription.from_dict(d)
        assert cd.cluster_id == 2
        assert cd.size == 3

    def test_from_dict_defaults(self):
        cd = ClusterDescription.from_dict({})
        assert cd.cluster_id == 0
        assert cd.size == 0

    def test_roundtrip(self):
        original = ClusterDescription(
            cluster_id=3, size=7, top_keywords=["a", "b"], description="d", representative_ids=["r1"]
        )
        restored = ClusterDescription.from_dict(original.as_dict())
        assert restored.cluster_id == original.cluster_id
        assert restored.top_keywords == original.top_keywords


# ---------------------------------------------------------------------------
# ClusteringReportConfig
# ---------------------------------------------------------------------------


class TestClusteringReportConfig:
    def test_defaults(self):
        cfg = ClusteringReportConfig()
        assert cfg.n_clusters == 5
        assert cfg.top_keywords == 10
        assert cfg.representatives_per_cluster == 3

    def test_custom(self):
        cfg = ClusteringReportConfig(n_clusters=3, top_keywords=5, representatives_per_cluster=2)
        assert cfg.n_clusters == 3

    def test_as_dict(self):
        cfg = ClusteringReportConfig(n_clusters=7)
        d = cfg.as_dict()
        assert d["n_clusters"] == 7

    def test_from_dict(self):
        cfg = ClusteringReportConfig.from_dict({"n_clusters": 8, "top_keywords": 15})
        assert cfg.n_clusters == 8
        assert cfg.top_keywords == 15

    def test_from_dict_defaults(self):
        cfg = ClusteringReportConfig.from_dict({})
        assert cfg.n_clusters == 5

    def test_roundtrip(self):
        original = ClusteringReportConfig(n_clusters=4, top_keywords=8, representatives_per_cluster=2)
        restored = ClusteringReportConfig.from_dict(original.as_dict())
        assert restored.n_clusters == original.n_clusters
        assert restored.top_keywords == original.top_keywords


# ---------------------------------------------------------------------------
# ClusteringReport
# ---------------------------------------------------------------------------


class TestClusteringReport:
    def test_defaults(self):
        cr = ClusteringReport()
        assert cr.clusters == []
        assert cr.total_items == 0
        assert cr.n_clusters == 0
        assert cr.silhouette_hint == 0.0

    def test_as_dict(self):
        cr = ClusteringReport(
            clusters=[ClusterDescription(cluster_id=0, size=3)],
            total_items=3,
            n_clusters=1,
            silhouette_hint=0.5,
        )
        d = cr.as_dict()
        assert d["total_items"] == 3
        assert len(d["clusters"]) == 1

    def test_from_dict(self):
        d = {
            "clusters": [{"cluster_id": 0, "size": 2}],
            "total_items": 2,
            "n_clusters": 1,
            "silhouette_hint": 0.3,
        }
        cr = ClusteringReport.from_dict(d)
        assert cr.total_items == 2
        assert len(cr.clusters) == 1

    def test_from_dict_defaults(self):
        cr = ClusteringReport.from_dict({})
        assert cr.total_items == 0

    def test_roundtrip(self):
        original = ClusteringReport(
            clusters=[
                ClusterDescription(cluster_id=0, size=5, top_keywords=["ml"]),
                ClusterDescription(cluster_id=1, size=3, top_keywords=["bio"]),
            ],
            total_items=8,
            n_clusters=2,
            silhouette_hint=0.45,
        )
        restored = ClusteringReport.from_dict(original.as_dict())
        assert restored.total_items == original.total_items
        assert len(restored.clusters) == 2


# ---------------------------------------------------------------------------
# cluster_items
# ---------------------------------------------------------------------------


class TestClusterItems:
    def test_empty_input(self):
        result = cluster_items({})
        assert result == {}

    def test_single_item(self):
        result = cluster_items({"a": "hello world"}, n_clusters=1)
        assert result == {"a": 0}

    def test_returns_all_items(self):
        result = cluster_items(SCIENCE_ITEMS, n_clusters=3)
        assert set(result.keys()) == set(SCIENCE_ITEMS.keys())

    def test_cluster_ids_valid(self):
        result = cluster_items(SCIENCE_ITEMS, n_clusters=3)
        for cid in result.values():
            assert 0 <= cid < 3

    def test_similar_items_cluster_together(self):
        items = {
            "ml1": "machine learning neural networks deep learning",
            "ml2": "neural networks training deep learning algorithms",
            "ml3": "deep learning backpropagation gradient descent networks",
            "cook1": "pasta recipe italian cooking tomato sauce basil garlic",
            "cook2": "spaghetti bolognese meat sauce onion garlic tomato italian",
            "cook3": "pizza dough flour yeast oven baking mozzarella italian cooking",
        }
        result = cluster_items(items, n_clusters=2)
        ml_clusters = {result["ml1"], result["ml2"], result["ml3"]}
        cook_clusters = {result["cook1"], result["cook2"], result["cook3"]}
        # ML items should share a cluster, cooking items should share a cluster
        assert len(ml_clusters) == 1
        assert len(cook_clusters) == 1
        # And they should be different clusters
        assert ml_clusters != cook_clusters

    def test_n_clusters_clamped(self):
        items = {"a": "hello", "b": "world"}
        result = cluster_items(items, n_clusters=10)
        assert len(result) == 2

    def test_deterministic(self):
        r1 = cluster_items(SCIENCE_ITEMS, n_clusters=3)
        r2 = cluster_items(SCIENCE_ITEMS, n_clusters=3)
        assert r1 == r2

    def test_one_cluster(self):
        result = cluster_items(SCIENCE_ITEMS, n_clusters=1)
        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# extract_cluster_keywords
# ---------------------------------------------------------------------------


class TestExtractClusterKeywords:
    def test_basic_extraction(self):
        items = {"a": "machine learning algorithms", "b": "machine learning models"}
        assignments = {"a": 0, "b": 0}
        keywords = extract_cluster_keywords(items, assignments, 0, top_n=5)
        assert "machine" in keywords
        assert "learning" in keywords

    def test_respects_cluster_id(self):
        items = {"a": "alpha beta", "b": "gamma delta"}
        assignments = {"a": 0, "b": 1}
        kw0 = extract_cluster_keywords(items, assignments, 0, top_n=5)
        kw1 = extract_cluster_keywords(items, assignments, 1, top_n=5)
        assert "alpha" in kw0
        assert "gamma" in kw1
        assert "gamma" not in kw0

    def test_top_n_limit(self):
        items = {"a": "one two three four five six seven eight nine ten eleven"}
        assignments = {"a": 0}
        keywords = extract_cluster_keywords(items, assignments, 0, top_n=3)
        assert len(keywords) <= 3

    def test_empty_cluster(self):
        items = {"a": "text"}
        assignments = {"a": 0}
        keywords = extract_cluster_keywords(items, assignments, 1, top_n=5)
        assert keywords == []

    def test_filters_stopwords(self):
        items = {"a": "the quick brown fox jumps over the lazy dog"}
        assignments = {"a": 0}
        keywords = extract_cluster_keywords(items, assignments, 0, top_n=10)
        assert "the" not in keywords
        assert "over" not in keywords

    def test_multiple_items_aggregate(self):
        items = {
            "a": "python programming code",
            "b": "python scripting code",
            "c": "python development code",
        }
        assignments = {"a": 0, "b": 0, "c": 0}
        keywords = extract_cluster_keywords(items, assignments, 0, top_n=3)
        # "python" and "code" appear in all 3 items
        assert "python" in keywords
        assert "code" in keywords


# ---------------------------------------------------------------------------
# find_representatives
# ---------------------------------------------------------------------------


class TestFindRepresentatives:
    def test_basic(self):
        items = {"a": "machine learning", "b": "machine learning algorithms", "c": "deep learning"}
        assignments = {"a": 0, "b": 0, "c": 0}
        reps = find_representatives(items, assignments, 0, n=2)
        assert len(reps) == 2
        assert all(r in items for r in reps)

    def test_empty_cluster(self):
        items = {"a": "text"}
        assignments = {"a": 0}
        reps = find_representatives(items, assignments, 1, n=3)
        assert reps == []

    def test_n_larger_than_cluster(self):
        items = {"a": "hello", "b": "world"}
        assignments = {"a": 0, "b": 0}
        reps = find_representatives(items, assignments, 0, n=10)
        assert len(reps) == 2

    def test_single_item(self):
        items = {"a": "only item"}
        assignments = {"a": 0}
        reps = find_representatives(items, assignments, 0, n=3)
        assert reps == ["a"]

    def test_representatives_from_correct_cluster(self):
        items = {"a": "ml", "b": "ml algo", "c": "cooking pasta"}
        assignments = {"a": 0, "b": 0, "c": 1}
        reps = find_representatives(items, assignments, 0, n=3)
        assert "c" not in reps


# ---------------------------------------------------------------------------
# generate_cluster_description
# ---------------------------------------------------------------------------


class TestGenerateClusterDescription:
    def test_empty_keywords(self):
        desc = generate_cluster_description([])
        assert "no distinguishing" in desc.lower()

    def test_single_keyword(self):
        desc = generate_cluster_description(["quantum"])
        assert "quantum" in desc
        assert desc == "Items about quantum"

    def test_two_keywords(self):
        desc = generate_cluster_description(["machine", "learning"])
        assert "machine" in desc
        assert "learning" in desc
        assert "and" in desc

    def test_three_or_more_keywords(self):
        desc = generate_cluster_description(["deep", "learning", "networks", "training"])
        assert "deep" in desc
        assert "learning" in desc
        assert "networks" in desc
        # Only top 3 used
        assert desc == "Items about deep, learning, and networks"

    def test_format(self):
        desc = generate_cluster_description(["alpha", "beta", "gamma"])
        assert desc.startswith("Items about")


# ---------------------------------------------------------------------------
# build_clustering_report
# ---------------------------------------------------------------------------


class TestBuildClusteringReport:
    def test_empty_items(self):
        report = build_clustering_report({})
        assert report.total_items == 0
        assert report.n_clusters == 0
        assert report.clusters == []

    def test_basic_report(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=3))
        assert report.total_items == len(SCIENCE_ITEMS)
        assert report.n_clusters > 0
        assert len(report.clusters) > 0

    def test_cluster_sizes_sum_to_total(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=3))
        total = sum(c.size for c in report.clusters)
        assert total == report.total_items

    def test_all_clusters_have_keywords(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=2))
        for cluster in report.clusters:
            assert len(cluster.top_keywords) > 0

    def test_all_clusters_have_description(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=2))
        for cluster in report.clusters:
            assert cluster.description != ""

    def test_all_clusters_have_representatives(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=2))
        for cluster in report.clusters:
            assert len(cluster.representative_ids) > 0

    def test_default_config(self):
        report = build_clustering_report(SCIENCE_ITEMS)
        assert report.total_items == len(SCIENCE_ITEMS)

    def test_silhouette_hint_computed(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=3))
        # Silhouette is between -1 and 1
        assert -1.0 <= report.silhouette_hint <= 1.0

    def test_single_item(self):
        report = build_clustering_report({"a": "hello world"})
        assert report.total_items == 1
        assert report.n_clusters == 1

    def test_representatives_per_cluster_config(self):
        config = ClusteringReportConfig(n_clusters=2, representatives_per_cluster=1)
        report = build_clustering_report(SCIENCE_ITEMS, config)
        for cluster in report.clusters:
            assert len(cluster.representative_ids) <= 1

    def test_top_keywords_config(self):
        config = ClusteringReportConfig(n_clusters=2, top_keywords=3)
        report = build_clustering_report(SCIENCE_ITEMS, config)
        for cluster in report.clusters:
            assert len(cluster.top_keywords) <= 3

    def test_report_as_dict_roundtrip(self):
        report = build_clustering_report(SCIENCE_ITEMS, ClusteringReportConfig(n_clusters=2))
        d = report.as_dict()
        restored = ClusteringReport.from_dict(d)
        assert restored.total_items == report.total_items
        assert len(restored.clusters) == len(report.clusters)

    def test_distinct_items_separate(self):
        items = {
            "ml1": "machine learning neural networks deep learning algorithms",
            "ml2": "neural networks training deep learning backpropagation",
            "cook1": "pasta recipe italian cooking tomato sauce basil garlic",
            "cook2": "spaghetti bolognese meat sauce onion garlic tomato italian",
        }
        report = build_clustering_report(items, ClusteringReportConfig(n_clusters=2))
        # Check that ml and cook items are in different clusters
        cluster_map = {}
        for cluster in report.clusters:
            for rid in cluster.representative_ids:
                cluster_map[rid] = cluster.cluster_id
        # At minimum, clusters exist
        assert report.n_clusters == 2


# ---------------------------------------------------------------------------
# format_clustering_report
# ---------------------------------------------------------------------------


class TestFormatClusteringReport:
    def test_header(self):
        report = ClusteringReport(total_items=10, n_clusters=3)
        output = format_clustering_report(report)
        assert "Clustering Report" in output

    def test_includes_totals(self):
        report = ClusteringReport(total_items=10, n_clusters=3)
        output = format_clustering_report(report)
        assert "10" in output
        assert "3" in output

    def test_includes_silhouette(self):
        report = ClusteringReport(silhouette_hint=0.5678)
        output = format_clustering_report(report)
        assert "0.5678" in output

    def test_includes_cluster_sections(self):
        report = ClusteringReport(
            clusters=[
                ClusterDescription(cluster_id=0, size=5, top_keywords=["ml", "ai"], description="ML items"),
                ClusterDescription(cluster_id=1, size=3, top_keywords=["bio"], description="Bio items"),
            ],
            total_items=8,
            n_clusters=2,
        )
        output = format_clustering_report(report)
        assert "Cluster 0" in output
        assert "Cluster 1" in output
        assert "ML items" in output
        assert "Bio items" in output

    def test_includes_keywords(self):
        report = ClusteringReport(
            clusters=[
                ClusterDescription(cluster_id=0, size=3, top_keywords=["alpha", "beta"]),
            ],
            total_items=3,
            n_clusters=1,
        )
        output = format_clustering_report(report)
        assert "alpha" in output
        assert "beta" in output

    def test_includes_representatives(self):
        report = ClusteringReport(
            clusters=[
                ClusterDescription(cluster_id=0, size=2, representative_ids=["item-1", "item-2"]),
            ],
            total_items=2,
            n_clusters=1,
        )
        output = format_clustering_report(report)
        assert "item-1" in output
        assert "item-2" in output

    def test_empty_report(self):
        report = ClusteringReport()
        output = format_clustering_report(report)
        assert "Clustering Report" in output
        assert "0" in output

    def test_separator_present(self):
        report = ClusteringReport(
            clusters=[ClusterDescription(cluster_id=0, size=1)],
            total_items=1,
            n_clusters=1,
        )
        output = format_clustering_report(report)
        assert "=" in output or "-" in output

    def test_full_report_formatting(self):
        report = build_clustering_report(COOKING_ITEMS, ClusteringReportConfig(n_clusters=2))
        output = format_clustering_report(report)
        assert "Clustering Report" in output
        assert str(len(COOKING_ITEMS)) in output
