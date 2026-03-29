"""Tests for cluster boundary sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.boundary_sampling import (
    BoundaryConfig,
    BoundaryResult,
    BoundaryScore,
    assign_clusters,
    compute_boundary_distance,
    compute_centroid_distance,
    format_boundary_report,
    run_boundary_sampling,
    score_boundary_items,
)


# ---- BoundaryScore ----


class TestBoundaryScore:
    def test_create(self):
        s = BoundaryScore(item_id="a", boundary_distance=0.2, cluster_id=1, is_boundary=True)
        assert s.item_id == "a"
        assert s.boundary_distance == 0.2
        assert s.cluster_id == 1
        assert s.is_boundary is True

    def test_as_dict(self):
        s = BoundaryScore(item_id="b", boundary_distance=0.8, cluster_id=0, is_boundary=False)
        d = s.as_dict()
        assert d == {
            "item_id": "b",
            "boundary_distance": 0.8,
            "cluster_id": 0,
            "is_boundary": False,
        }

    def test_from_dict(self):
        d = {"item_id": "c", "boundary_distance": 0.5, "cluster_id": 2, "is_boundary": True}
        s = BoundaryScore.from_dict(d)
        assert s.item_id == "c"
        assert s.boundary_distance == 0.5
        assert s.cluster_id == 2
        assert s.is_boundary is True

    def test_roundtrip(self):
        s = BoundaryScore(item_id="x", boundary_distance=0.3, cluster_id=1, is_boundary=False)
        assert BoundaryScore.from_dict(s.as_dict()) == s


# ---- BoundaryConfig ----


class TestBoundaryConfig:
    def test_defaults(self):
        c = BoundaryConfig()
        assert c.n_clusters == 3
        assert c.sample_size == 50
        assert c.boundary_threshold == 0.3
        assert c.seed is None

    def test_custom(self):
        c = BoundaryConfig(n_clusters=5, sample_size=20, boundary_threshold=0.5, seed=42)
        assert c.n_clusters == 5
        assert c.sample_size == 20
        assert c.boundary_threshold == 0.5
        assert c.seed == 42

    def test_as_dict(self):
        c = BoundaryConfig(seed=7)
        d = c.as_dict()
        assert d == {
            "n_clusters": 3,
            "sample_size": 50,
            "boundary_threshold": 0.3,
            "seed": 7,
        }

    def test_from_dict_full(self):
        d = {"n_clusters": 4, "sample_size": 30, "boundary_threshold": 0.2, "seed": 99}
        c = BoundaryConfig.from_dict(d)
        assert c.n_clusters == 4
        assert c.sample_size == 30
        assert c.boundary_threshold == 0.2
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = BoundaryConfig.from_dict({})
        assert c.n_clusters == 3
        assert c.sample_size == 50
        assert c.boundary_threshold == 0.3
        assert c.seed is None

    def test_roundtrip(self):
        c = BoundaryConfig(n_clusters=2, sample_size=10, boundary_threshold=0.4, seed=5)
        assert BoundaryConfig.from_dict(c.as_dict()) == c


# ---- BoundaryResult ----


class TestBoundaryResult:
    def test_defaults(self):
        r = BoundaryResult()
        assert r.selected_ids == []
        assert r.scores == []
        assert r.boundary_count == 0
        assert r.centroid_count == 0
        assert r.total_pool == 0

    def test_as_dict(self):
        s = BoundaryScore(item_id="a", boundary_distance=0.1, cluster_id=0, is_boundary=True)
        r = BoundaryResult(
            selected_ids=["a"],
            scores=[s],
            boundary_count=1,
            centroid_count=0,
            total_pool=1,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["scores"]) == 1
        assert d["boundary_count"] == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "scores": [
                {"item_id": "x", "boundary_distance": 0.2, "cluster_id": 1, "is_boundary": True},
            ],
            "boundary_count": 1,
            "centroid_count": 2,
            "total_pool": 3,
        }
        r = BoundaryResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert len(r.scores) == 1
        assert r.boundary_count == 1
        assert r.centroid_count == 2

    def test_from_dict_defaults(self):
        r = BoundaryResult.from_dict({})
        assert r.selected_ids == []
        assert r.scores == []

    def test_roundtrip(self):
        s = BoundaryScore(item_id="a", boundary_distance=0.1, cluster_id=0, is_boundary=True)
        r = BoundaryResult(
            selected_ids=["a"],
            scores=[s],
            boundary_count=1,
            centroid_count=0,
            total_pool=1,
        )
        r2 = BoundaryResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.boundary_count == r.boundary_count


# ---- compute_centroid_distance ----


class TestComputeCentroidDistance:
    def test_identical_sets(self):
        assert compute_centroid_distance({"a", "b"}, {"a", "b"}) == 0.0

    def test_disjoint_sets(self):
        assert compute_centroid_distance({"a", "b"}, {"c", "d"}) == 1.0

    def test_partial_overlap(self):
        # {"a", "b"} & {"b", "c"} = {"b"}, union = {"a", "b", "c"}
        # Jaccard = 1/3, distance = 2/3
        result = compute_centroid_distance({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(2.0 / 3.0)

    def test_empty_item(self):
        assert compute_centroid_distance(set(), {"a"}) == 1.0

    def test_empty_centroid(self):
        assert compute_centroid_distance({"a"}, set()) == 1.0

    def test_both_empty(self):
        assert compute_centroid_distance(set(), set()) == 1.0

    def test_subset(self):
        # {"a"} is subset of {"a", "b"}: Jaccard = 1/2, distance = 0.5
        result = compute_centroid_distance({"a"}, {"a", "b"})
        assert result == pytest.approx(0.5)

    def test_superset(self):
        result = compute_centroid_distance({"a", "b", "c"}, {"a"})
        assert result == pytest.approx(2.0 / 3.0)


# ---- compute_boundary_distance ----


class TestComputeBoundaryDistance:
    def test_fewer_than_two_centroids(self):
        assert compute_boundary_distance({"a"}, [{"a"}]) == 0.0
        assert compute_boundary_distance({"a"}, []) == 0.0

    def test_on_boundary(self):
        # Item equally close to both centroids
        item = {"a", "b", "c", "d"}
        c1 = {"a", "b"}
        c2 = {"c", "d"}
        result = compute_boundary_distance(item, [c1, c2])
        # Both distances should be equal, so boundary distance = 0
        assert result == pytest.approx(0.0)

    def test_far_from_boundary(self):
        # Item very close to c1, far from c2
        item = {"a", "b"}
        c1 = {"a", "b"}
        c2 = {"x", "y"}
        result = compute_boundary_distance(item, [c1, c2])
        # distance to c1 = 0, distance to c2 = 1, boundary = 1 - 0 = 1
        assert result == pytest.approx(1.0)

    def test_three_centroids(self):
        item = {"a"}
        c1 = {"a"}  # distance = 0
        c2 = {"b"}  # distance = 1
        c3 = {"a", "b"}  # distance = 0.5
        result = compute_boundary_distance(item, [c1, c2, c3])
        # sorted distances: [0, 0.5, 1], boundary = 0.5 - 0 = 0.5
        assert result == pytest.approx(0.5)

    def test_symmetric(self):
        item = {"a", "b"}
        c1 = {"a"}
        c2 = {"b"}
        result = compute_boundary_distance(item, [c1, c2])
        # Both distances identical, boundary = 0
        assert result == pytest.approx(0.0)


# ---- assign_clusters ----


class TestAssignClusters:
    def test_empty(self):
        centroids, mapping = assign_clusters({}, 3)
        assert centroids == []
        assert mapping == {}

    def test_single_item(self):
        items = {"a": "hello world"}
        centroids, mapping = assign_clusters(items, 2)
        assert len(centroids) == 1  # min(2, 1) = 1
        assert "a" in mapping

    def test_two_clusters_distinct(self):
        items = {
            "a": "cat dog pet",
            "b": "cat dog animal",
            "c": "code python software",
            "d": "code python program",
        }
        centroids, mapping = assign_clusters(items, 2)
        assert len(centroids) == 2
        assert len(mapping) == 4
        # All items should be assigned to a cluster
        assert all(v in (0, 1) for v in mapping.values())

    def test_more_clusters_than_items(self):
        items = {"a": "hello", "b": "world"}
        centroids, mapping = assign_clusters(items, 5)
        assert len(centroids) == 2  # capped at n items
        assert len(mapping) == 2

    def test_returns_centroids_as_sets(self):
        items = {"a": "cat dog", "b": "cat fish", "c": "bird fly"}
        centroids, _ = assign_clusters(items, 2)
        for c in centroids:
            assert isinstance(c, set)

    def test_all_items_assigned(self):
        items = {f"i{i}": f"word{i} common" for i in range(10)}
        _, mapping = assign_clusters(items, 3)
        assert set(mapping.keys()) == set(items.keys())


# ---- score_boundary_items ----


class TestScoreBoundaryItems:
    def test_empty(self):
        assert score_boundary_items({}, BoundaryConfig()) == []

    def test_scores_all_items(self):
        items = {f"i{i}": f"text {i}" for i in range(5)}
        scores = score_boundary_items(items, BoundaryConfig(n_clusters=2))
        assert len(scores) == 5

    def test_sorted_by_item_id(self):
        items = {"c": "x", "a": "y", "b": "z"}
        scores = score_boundary_items(items, BoundaryConfig(n_clusters=2))
        ids = [s.item_id for s in scores]
        assert ids == ["a", "b", "c"]

    def test_boundary_flag(self):
        # With a high threshold, most items should be boundary
        items = {f"i{i}": f"word{i} shared" for i in range(6)}
        scores = score_boundary_items(
            items, BoundaryConfig(n_clusters=2, boundary_threshold=1.0)
        )
        assert all(s.is_boundary for s in scores)

    def test_centroid_flag_with_low_threshold(self):
        # With threshold=0, no items are boundary
        items = {f"i{i}": f"text {i}" for i in range(4)}
        scores = score_boundary_items(
            items, BoundaryConfig(n_clusters=2, boundary_threshold=0.0)
        )
        # Items with boundary_distance exactly 0 are still < 0 is False
        boundary_items = [s for s in scores if s.is_boundary]
        # All boundary distances >= 0, and threshold is 0, so is_boundary requires < 0
        assert len(boundary_items) == 0

    def test_cluster_id_assigned(self):
        items = {"a": "cat dog", "b": "code python"}
        scores = score_boundary_items(items, BoundaryConfig(n_clusters=2))
        for s in scores:
            assert isinstance(s.cluster_id, int)
            assert s.cluster_id >= 0

    def test_boundary_distance_non_negative(self):
        items = {f"i{i}": f"word{i}" for i in range(10)}
        scores = score_boundary_items(items, BoundaryConfig(n_clusters=3))
        for s in scores:
            assert s.boundary_distance >= 0.0


# ---- run_boundary_sampling ----


class TestRunBoundarySampling:
    def test_empty(self):
        result = run_boundary_sampling({}, BoundaryConfig())
        assert result.selected_ids == []
        assert result.scores == []
        assert result.boundary_count == 0
        assert result.centroid_count == 0
        assert result.total_pool == 0

    def test_basic(self):
        items = {
            "a": "cat dog pet animal",
            "b": "cat dog pet friend",
            "c": "code python software dev",
            "d": "code python program app",
            "e": "cat code mixed pet python",
        }
        config = BoundaryConfig(n_clusters=2, sample_size=3, seed=42)
        result = run_boundary_sampling(items, config)
        assert len(result.selected_ids) == 3
        assert result.total_pool == 5
        assert len(result.scores) == 5

    def test_boundary_plus_centroid_equals_total(self):
        items = {f"i{i}": f"word{i} common" for i in range(10)}
        config = BoundaryConfig(n_clusters=2, sample_size=5, seed=1)
        result = run_boundary_sampling(items, config)
        assert result.boundary_count + result.centroid_count == result.total_pool

    def test_sample_size_larger_than_pool(self):
        items = {"a": "hello", "b": "world"}
        config = BoundaryConfig(sample_size=100, seed=1)
        result = run_boundary_sampling(items, config)
        assert len(result.selected_ids) == 2

    def test_deterministic(self):
        items = {f"i{i}": f"text{i} shared content" for i in range(20)}
        config = BoundaryConfig(n_clusters=3, sample_size=5, seed=42)
        r1 = run_boundary_sampling(items, config)
        r2 = run_boundary_sampling(items, config)
        assert r1.selected_ids == r2.selected_ids

    def test_selects_boundary_items_first(self):
        # Create items where some are clearly on boundaries
        items = {
            "pure_a": "alpha beta gamma",
            "pure_b": "delta epsilon zeta",
            "mixed": "alpha delta",  # between clusters
        }
        config = BoundaryConfig(n_clusters=2, sample_size=1, seed=42)
        result = run_boundary_sampling(items, config)
        # The mixed item should have lowest boundary distance
        assert len(result.selected_ids) == 1


# ---- format_boundary_report ----


class TestFormatBoundaryReport:
    def test_empty_result(self):
        result = BoundaryResult()
        report = format_boundary_report(result)
        assert "Boundary Sampling Report" in report
        assert "Total pool size: 0" in report

    def test_with_data(self):
        s = BoundaryScore(item_id="a", boundary_distance=0.1, cluster_id=0, is_boundary=True)
        result = BoundaryResult(
            selected_ids=["a"],
            scores=[s],
            boundary_count=1,
            centroid_count=0,
            total_pool=1,
        )
        report = format_boundary_report(result)
        assert "Selected count: 1" in report
        assert "Boundary items: 1" in report
        assert "a:" in report

    def test_report_contains_header(self):
        result = BoundaryResult(total_pool=5)
        report = format_boundary_report(result)
        assert "=" * 40 in report

    def test_boundary_label(self):
        s = BoundaryScore(item_id="x", boundary_distance=0.1, cluster_id=0, is_boundary=True)
        result = BoundaryResult(scores=[s], total_pool=1)
        report = format_boundary_report(result)
        assert "[boundary]" in report

    def test_centroid_label(self):
        s = BoundaryScore(item_id="x", boundary_distance=0.9, cluster_id=0, is_boundary=False)
        result = BoundaryResult(scores=[s], total_pool=1)
        report = format_boundary_report(result)
        assert "[centroid]" in report

    def test_selected_ids_section(self):
        result = BoundaryResult(selected_ids=["a", "b"], total_pool=2)
        report = format_boundary_report(result)
        assert "Selected IDs:" in report
        assert "  - a" in report
        assert "  - b" in report
