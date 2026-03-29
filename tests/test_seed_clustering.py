"""Tests for seed clustering module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.seed_clustering import (
    SeedCluster,
    ClusteringResult,
    cluster_seeds,
    sample_from_clusters,
    compute_cluster_diversity,
    find_outlier_seeds,
    format_clustering_report,
    _tokenize,
    _jaccard_similarity,
    _compute_centroid,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("hello world python code")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "code" in tokens

    def test_removes_stop_words(self):
        tokens = _tokenize("the quick brown fox and the lazy dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_removes_short_words(self):
        tokens = _tokenize("go to do it as is be")
        # All words are 2 chars or fewer (or stop words)
        assert len(tokens) == 0

    def test_lowercases(self):
        tokens = _tokenize("Python CODE Hello")
        assert "python" in tokens
        assert "code" in tokens
        assert "hello" in tokens

    def test_empty_string(self):
        assert _tokenize("") == set()


# ---------------------------------------------------------------------------
# _jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4
        assert abs(sim - 0.5) < 1e-9

    def test_both_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert _jaccard_similarity({"a"}, set()) == 0.0

    def test_subset(self):
        sim = _jaccard_similarity({"a", "b"}, {"a", "b", "c"})
        # intersection=2, union=3
        assert abs(sim - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# _compute_centroid
# ---------------------------------------------------------------------------


class TestComputeCentroid:
    def test_single_set(self):
        result = _compute_centroid([{"alpha", "beta", "gamma"}])
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

    def test_frequency_ordering(self):
        sets = [{"cat", "dog"}, {"cat", "fish"}, {"cat", "bird"}]
        result = _compute_centroid(sets, top_k=1)
        assert "cat" in result  # cat appears 3 times

    def test_top_k_limits(self):
        sets = [{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"}]
        result = _compute_centroid(sets, top_k=3)
        assert len(result) == 3

    def test_empty_list(self):
        result = _compute_centroid([])
        assert result == set()


# ---------------------------------------------------------------------------
# SeedCluster dataclass
# ---------------------------------------------------------------------------


class TestSeedCluster:
    def test_construction(self):
        c = SeedCluster(cluster_id=0, seed_ids=["a", "b"], centroid_keywords=["foo"], size=2)
        assert c.cluster_id == 0
        assert c.seed_ids == ["a", "b"]
        assert c.size == 2

    def test_as_dict(self):
        c = SeedCluster(cluster_id=1, seed_ids=["x"], centroid_keywords=["bar"], size=1)
        d = c.as_dict()
        assert d["cluster_id"] == 1
        assert d["seed_ids"] == ["x"]
        assert d["centroid_keywords"] == ["bar"]
        assert d["size"] == 1

    def test_from_dict(self):
        d = {"cluster_id": 2, "seed_ids": ["a", "b"], "centroid_keywords": ["c"], "size": 2}
        c = SeedCluster.from_dict(d)
        assert c.cluster_id == 2
        assert c.seed_ids == ["a", "b"]

    def test_roundtrip(self):
        original = SeedCluster(cluster_id=5, seed_ids=["s1", "s2", "s3"], centroid_keywords=["kw1", "kw2"], size=3)
        restored = SeedCluster.from_dict(original.as_dict())
        assert restored.cluster_id == original.cluster_id
        assert restored.seed_ids == original.seed_ids
        assert restored.centroid_keywords == original.centroid_keywords
        assert restored.size == original.size

    def test_defaults(self):
        c = SeedCluster(cluster_id=0)
        assert c.seed_ids == []
        assert c.centroid_keywords == []
        assert c.size == 0


# ---------------------------------------------------------------------------
# ClusteringResult dataclass
# ---------------------------------------------------------------------------


class TestClusteringResult:
    def test_construction(self):
        cr = ClusteringResult(clusters=[], total_seeds=0, n_clusters=0)
        assert cr.total_seeds == 0
        assert cr.silhouette_score == 0.0

    def test_as_dict(self):
        c = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=["k"], size=1)
        cr = ClusteringResult(clusters=[c], total_seeds=1, n_clusters=1, silhouette_score=0.5)
        d = cr.as_dict()
        assert d["total_seeds"] == 1
        assert d["n_clusters"] == 1
        assert d["silhouette_score"] == 0.5
        assert len(d["clusters"]) == 1

    def test_from_dict(self):
        d = {
            "clusters": [{"cluster_id": 0, "seed_ids": ["a"], "centroid_keywords": [], "size": 1}],
            "total_seeds": 1,
            "n_clusters": 1,
            "silhouette_score": 0.3,
        }
        cr = ClusteringResult.from_dict(d)
        assert cr.total_seeds == 1
        assert len(cr.clusters) == 1
        assert cr.clusters[0].seed_ids == ["a"]

    def test_roundtrip(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a", "b"], centroid_keywords=["x"], size=2)
        c2 = SeedCluster(cluster_id=1, seed_ids=["c"], centroid_keywords=["y"], size=1)
        original = ClusteringResult(clusters=[c1, c2], total_seeds=3, n_clusters=2, silhouette_score=0.7)
        restored = ClusteringResult.from_dict(original.as_dict())
        assert restored.total_seeds == original.total_seeds
        assert restored.n_clusters == original.n_clusters
        assert len(restored.clusters) == 2


# ---------------------------------------------------------------------------
# cluster_seeds
# ---------------------------------------------------------------------------


class TestClusterSeeds:
    def test_empty_seeds(self):
        result = cluster_seeds({})
        assert result.total_seeds == 0
        assert result.n_clusters == 0
        assert result.clusters == []

    def test_single_seed(self):
        result = cluster_seeds({"s1": "hello world python"}, n_clusters=1)
        assert result.total_seeds == 1
        assert result.n_clusters == 1
        assert result.clusters[0].seed_ids == ["s1"]

    def test_clamps_clusters_to_seed_count(self):
        seeds = {"s1": "alpha beta", "s2": "gamma delta"}
        result = cluster_seeds(seeds, n_clusters=10)
        # Cannot have more clusters than seeds
        assert result.n_clusters <= 2

    def test_basic_clustering(self):
        seeds = {
            "a1": "machine learning neural network deep learning",
            "a2": "neural network training gradient descent",
            "a3": "deep learning convolutional network",
            "b1": "database query optimization sql index",
            "b2": "sql database schema migration",
            "b3": "query performance index tuning",
        }
        result = cluster_seeds(seeds, n_clusters=2)
        assert result.total_seeds == 6
        assert result.n_clusters == 2
        # All seeds should be assigned
        all_assigned = []
        for c in result.clusters:
            all_assigned.extend(c.seed_ids)
        assert sorted(all_assigned) == sorted(seeds.keys())

    def test_cluster_sizes_sum_to_total(self):
        seeds = {f"s{i}": f"word_{i} topic_{i % 3}" for i in range(20)}
        result = cluster_seeds(seeds, n_clusters=4)
        total_size = sum(c.size for c in result.clusters)
        assert total_size == result.total_seeds

    def test_every_seed_assigned_once(self):
        seeds = {f"s{i}": f"text number {i} unique content" for i in range(10)}
        result = cluster_seeds(seeds, n_clusters=3)
        all_ids = []
        for c in result.clusters:
            all_ids.extend(c.seed_ids)
        assert len(all_ids) == len(set(all_ids))
        assert sorted(all_ids) == sorted(seeds.keys())

    def test_centroid_keywords_populated(self):
        seeds = {
            "s1": "python programming language syntax",
            "s2": "python development scripting automation",
        }
        result = cluster_seeds(seeds, n_clusters=1)
        assert len(result.clusters[0].centroid_keywords) > 0

    def test_silhouette_score_range(self):
        seeds = {
            "a1": "cat dog animal pet",
            "a2": "cat kitten feline pet",
            "b1": "car engine motor vehicle",
            "b2": "car truck automobile drive",
        }
        result = cluster_seeds(seeds, n_clusters=2)
        assert -1.0 <= result.silhouette_score <= 1.0

    def test_n_clusters_one(self):
        seeds = {"s1": "alpha", "s2": "beta", "s3": "gamma"}
        result = cluster_seeds(seeds, n_clusters=1)
        assert result.n_clusters == 1
        assert result.clusters[0].size == 3


# ---------------------------------------------------------------------------
# sample_from_clusters
# ---------------------------------------------------------------------------


class TestSampleFromClusters:
    def _make_result(self) -> tuple:
        seeds = {
            "a1": "topic alpha first",
            "a2": "topic alpha second",
            "a3": "topic alpha third",
            "b1": "topic beta first",
            "b2": "topic beta second",
        }
        c1 = SeedCluster(cluster_id=0, seed_ids=["a1", "a2", "a3"], centroid_keywords=["alpha"], size=3)
        c2 = SeedCluster(cluster_id=1, seed_ids=["b1", "b2"], centroid_keywords=["beta"], size=2)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=5, n_clusters=2)
        return result, seeds

    def test_proportional_allocation(self):
        result, seeds = self._make_result()
        sampled = sample_from_clusters(result, 3, seeds)
        assert len(sampled) == 3

    def test_returns_valid_seed_ids(self):
        result, seeds = self._make_result()
        sampled = sample_from_clusters(result, 4, seeds)
        for sid in sampled:
            assert sid in seeds

    def test_empty_result(self):
        result = ClusteringResult(clusters=[], total_seeds=0, n_clusters=0)
        sampled = sample_from_clusters(result, 5, {})
        assert sampled == []

    def test_zero_samples(self):
        result, seeds = self._make_result()
        sampled = sample_from_clusters(result, 0, seeds)
        assert sampled == []

    def test_more_samples_than_seeds(self):
        result, seeds = self._make_result()
        sampled = sample_from_clusters(result, 100, seeds)
        # Cannot return more than total available
        assert len(sampled) <= 5

    def test_single_cluster(self):
        c = SeedCluster(cluster_id=0, seed_ids=["x", "y", "z"], centroid_keywords=[], size=3)
        result = ClusteringResult(clusters=[c], total_seeds=3, n_clusters=1)
        sampled = sample_from_clusters(result, 2, {"x": "", "y": "", "z": ""})
        assert len(sampled) == 2


# ---------------------------------------------------------------------------
# compute_cluster_diversity
# ---------------------------------------------------------------------------


class TestComputeClusterDiversity:
    def test_single_cluster(self):
        c = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=[], size=1)
        result = ClusteringResult(clusters=[c], total_seeds=1, n_clusters=1)
        assert compute_cluster_diversity(result, {"a": "hello"}) == 0.0

    def test_identical_clusters(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=[], size=1)
        c2 = SeedCluster(cluster_id=1, seed_ids=["b"], centroid_keywords=[], size=1)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=2, n_clusters=2)
        seeds = {"a": "hello world python", "b": "hello world python"}
        div = compute_cluster_diversity(result, seeds)
        assert div == 0.0

    def test_disjoint_clusters(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=[], size=1)
        c2 = SeedCluster(cluster_id=1, seed_ids=["b"], centroid_keywords=[], size=1)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=2, n_clusters=2)
        seeds = {"a": "alpha beta gamma", "b": "delta epsilon zeta"}
        div = compute_cluster_diversity(result, seeds)
        assert div == 1.0

    def test_partial_overlap(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=[], size=1)
        c2 = SeedCluster(cluster_id=1, seed_ids=["b"], centroid_keywords=[], size=1)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=2, n_clusters=2)
        seeds = {"a": "alpha beta gamma", "b": "alpha delta epsilon"}
        div = compute_cluster_diversity(result, seeds)
        assert 0.0 < div < 1.0

    def test_empty_clusters(self):
        result = ClusteringResult(clusters=[], total_seeds=0, n_clusters=0)
        assert compute_cluster_diversity(result, {}) == 0.0

    def test_multiple_seeds_per_cluster(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a", "b"], centroid_keywords=[], size=2)
        c2 = SeedCluster(cluster_id=1, seed_ids=["c", "d"], centroid_keywords=[], size=2)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=4, n_clusters=2)
        seeds = {
            "a": "machine learning training",
            "b": "neural network learning",
            "c": "database query index",
            "d": "sql schema migration",
        }
        div = compute_cluster_diversity(result, seeds)
        assert 0.0 < div <= 1.0


# ---------------------------------------------------------------------------
# find_outlier_seeds
# ---------------------------------------------------------------------------


class TestFindOutlierSeeds:
    def test_no_outliers(self):
        c = SeedCluster(
            cluster_id=0,
            seed_ids=["a"],
            centroid_keywords=["python", "programming", "language"],
            size=1,
        )
        result = ClusteringResult(clusters=[c], total_seeds=1, n_clusters=1)
        seeds = {"a": "python programming language scripting"}
        outliers = find_outlier_seeds(result, seeds, threshold=0.1)
        assert outliers == []

    def test_outlier_detected(self):
        c = SeedCluster(
            cluster_id=0,
            seed_ids=["a", "b"],
            centroid_keywords=["python", "programming", "language"],
            size=2,
        )
        result = ClusteringResult(clusters=[c], total_seeds=2, n_clusters=1)
        seeds = {
            "a": "python programming language",
            "b": "completely unrelated random gibberish xyz",
        }
        outliers = find_outlier_seeds(result, seeds, threshold=0.1)
        assert "b" in outliers

    def test_threshold_zero(self):
        # threshold=0 means only seeds with literally zero similarity are outliers
        c = SeedCluster(
            cluster_id=0,
            seed_ids=["a"],
            centroid_keywords=["alpha"],
            size=1,
        )
        result = ClusteringResult(clusters=[c], total_seeds=1, n_clusters=1)
        seeds = {"a": "beta gamma delta"}
        outliers = find_outlier_seeds(result, seeds, threshold=0.0)
        assert outliers == []  # 0.0 is not < 0.0

    def test_high_threshold_all_outliers(self):
        c = SeedCluster(
            cluster_id=0,
            seed_ids=["a", "b"],
            centroid_keywords=["specific"],
            size=2,
        )
        result = ClusteringResult(clusters=[c], total_seeds=2, n_clusters=1)
        seeds = {"a": "alpha beta gamma", "b": "delta epsilon zeta"}
        outliers = find_outlier_seeds(result, seeds, threshold=1.0)
        assert len(outliers) == 2

    def test_empty_result(self):
        result = ClusteringResult(clusters=[], total_seeds=0, n_clusters=0)
        assert find_outlier_seeds(result, {}) == []

    def test_missing_seed_in_dict(self):
        c = SeedCluster(
            cluster_id=0,
            seed_ids=["a", "missing"],
            centroid_keywords=["keyword"],
            size=2,
        )
        result = ClusteringResult(clusters=[c], total_seeds=2, n_clusters=1)
        seeds = {"a": "keyword stuff things"}
        # Should not crash, just skip missing seeds
        outliers = find_outlier_seeds(result, seeds, threshold=0.1)
        assert isinstance(outliers, list)


# ---------------------------------------------------------------------------
# format_clustering_report
# ---------------------------------------------------------------------------


class TestFormatClusteringReport:
    def test_basic_report(self):
        c = SeedCluster(cluster_id=0, seed_ids=["s1", "s2"], centroid_keywords=["kw1"], size=2)
        result = ClusteringResult(clusters=[c], total_seeds=2, n_clusters=1, silhouette_score=0.5)
        report = format_clustering_report(result)
        assert "SEED CLUSTERING REPORT" in report
        assert "Total seeds: 2" in report
        assert "Clusters: 1" in report
        assert "0.500" in report

    def test_empty_report(self):
        result = ClusteringResult(clusters=[], total_seeds=0, n_clusters=0)
        report = format_clustering_report(result)
        assert "Total seeds: 0" in report

    def test_multiple_clusters(self):
        c1 = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=["x"], size=1)
        c2 = SeedCluster(cluster_id=1, seed_ids=["b"], centroid_keywords=["y"], size=1)
        result = ClusteringResult(clusters=[c1, c2], total_seeds=2, n_clusters=2)
        report = format_clustering_report(result)
        assert "Cluster 0" in report
        assert "Cluster 1" in report

    def test_large_cluster_truncated(self):
        ids = [f"s{i}" for i in range(10)]
        c = SeedCluster(cluster_id=0, seed_ids=ids, centroid_keywords=[], size=10)
        result = ClusteringResult(clusters=[c], total_seeds=10, n_clusters=1)
        report = format_clustering_report(result)
        assert "and 5 more" in report

    def test_keywords_in_report(self):
        c = SeedCluster(cluster_id=0, seed_ids=["a"], centroid_keywords=["foo", "bar"], size=1)
        result = ClusteringResult(clusters=[c], total_seeds=1, n_clusters=1)
        report = format_clustering_report(result)
        assert "foo" in report
        assert "bar" in report


# ---------------------------------------------------------------------------
# Integration: cluster_seeds -> sample_from_clusters
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_cluster_then_sample(self):
        seeds = {f"s{i}": f"topic_{i % 3} content word_{i}" for i in range(15)}
        result = cluster_seeds(seeds, n_clusters=3)
        sampled = sample_from_clusters(result, 6, seeds)
        assert 1 <= len(sampled) <= 6
        for sid in sampled:
            assert sid in seeds

    def test_cluster_then_diversity(self):
        seeds = {
            "a1": "python programming code syntax",
            "a2": "python scripting automation",
            "b1": "database sql query optimization",
            "b2": "schema migration index tuning",
        }
        result = cluster_seeds(seeds, n_clusters=2)
        div = compute_cluster_diversity(result, seeds)
        assert 0.0 <= div <= 1.0

    def test_cluster_then_outliers(self):
        seeds = {
            "s1": "common shared topic area",
            "s2": "common shared topic area again",
            "s3": "common shared repeated topic",
            "outlier": "zzz qqq xxx yyy completely different",
        }
        result = cluster_seeds(seeds, n_clusters=2)
        outliers = find_outlier_seeds(result, seeds, threshold=0.15)
        # The outlier may or may not be detected depending on cluster assignment,
        # but the function should not crash
        assert isinstance(outliers, list)
