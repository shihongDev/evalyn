"""Tests for adversarial sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.adversarial_sampling import (
    AdversarialConfig,
    AdversarialResult,
    AdversarialWeight,
    compute_boundary_weights,
    compute_cohort_weights,
    compute_failure_weights,
    format_adversarial_report,
    merge_weights,
    run_adversarial_sampling,
)


# ---- AdversarialWeight ----


class TestAdversarialWeight:
    def test_construction(self):
        w = AdversarialWeight(item_id="a", weight=2.5, reason="failure")
        assert w.item_id == "a"
        assert w.weight == 2.5
        assert w.reason == "failure"

    def test_as_dict(self):
        w = AdversarialWeight(item_id="x", weight=1.0, reason="pass")
        d = w.as_dict()
        assert d == {"item_id": "x", "weight": 1.0, "reason": "pass"}

    def test_from_dict(self):
        d = {"item_id": "z", "weight": 3.0, "reason": "boundary"}
        w = AdversarialWeight.from_dict(d)
        assert w.item_id == "z"
        assert w.weight == 3.0
        assert w.reason == "boundary"

    def test_roundtrip(self):
        w = AdversarialWeight(item_id="q", weight=4.5, reason="test reason")
        assert AdversarialWeight.from_dict(w.as_dict()) == w


# ---- AdversarialConfig ----


class TestAdversarialConfig:
    def test_defaults(self):
        c = AdversarialConfig()
        assert c.failure_boost == 3.0
        assert c.boundary_boost == 2.0
        assert c.boundary_width == 0.1
        assert c.sample_size == 50
        assert c.seed is None

    def test_custom_values(self):
        c = AdversarialConfig(
            failure_boost=5.0,
            boundary_boost=4.0,
            boundary_width=0.2,
            sample_size=100,
            seed=42,
        )
        assert c.failure_boost == 5.0
        assert c.sample_size == 100

    def test_as_dict(self):
        c = AdversarialConfig(seed=7)
        d = c.as_dict()
        assert d["seed"] == 7
        assert d["failure_boost"] == 3.0

    def test_from_dict_full(self):
        d = {
            "failure_boost": 10.0,
            "boundary_boost": 5.0,
            "boundary_width": 0.05,
            "sample_size": 25,
            "seed": 99,
        }
        c = AdversarialConfig.from_dict(d)
        assert c.failure_boost == 10.0
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = AdversarialConfig.from_dict({})
        assert c.failure_boost == 3.0
        assert c.seed is None

    def test_roundtrip(self):
        c = AdversarialConfig(failure_boost=2.0, seed=11)
        assert AdversarialConfig.from_dict(c.as_dict()) == c


# ---- AdversarialResult ----


class TestAdversarialResult:
    def test_defaults(self):
        r = AdversarialResult()
        assert r.selected_ids == []
        assert r.weights == []
        assert r.failure_count == 0
        assert r.boundary_count == 0
        assert r.total_pool == 0

    def test_as_dict(self):
        r = AdversarialResult(
            selected_ids=["a", "b"],
            weights=[AdversarialWeight("a", 2.0, "fail")],
            failure_count=1,
            boundary_count=0,
            total_pool=5,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert len(d["weights"]) == 1
        assert d["failure_count"] == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "weights": [{"item_id": "x", "weight": 3.0, "reason": "r"}],
            "failure_count": 1,
            "boundary_count": 2,
            "total_pool": 10,
        }
        r = AdversarialResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert r.weights[0].item_id == "x"

    def test_from_dict_empty(self):
        r = AdversarialResult.from_dict({})
        assert r.selected_ids == []
        assert r.total_pool == 0


# ---- compute_failure_weights ----


class TestComputeFailureWeights:
    def test_all_failures(self):
        scores = {"a": 0.1, "b": 0.2, "c": 0.3}
        ws = compute_failure_weights(scores, threshold=0.5, boost=3.0)
        assert len(ws) == 3
        for w in ws:
            assert w.weight == 3.0
            assert "failure" in w.reason

    def test_all_passes(self):
        scores = {"a": 0.6, "b": 0.7, "c": 0.9}
        ws = compute_failure_weights(scores, threshold=0.5, boost=3.0)
        assert all(w.weight == 1.0 for w in ws)

    def test_mixed(self):
        scores = {"a": 0.3, "b": 0.7}
        ws = compute_failure_weights(scores, threshold=0.5, boost=5.0)
        by_id = {w.item_id: w for w in ws}
        assert by_id["a"].weight == 5.0
        assert by_id["b"].weight == 1.0

    def test_exact_threshold_passes(self):
        scores = {"a": 0.5}
        ws = compute_failure_weights(scores, threshold=0.5)
        assert ws[0].weight == 1.0

    def test_empty_scores(self):
        ws = compute_failure_weights({})
        assert ws == []

    def test_custom_threshold(self):
        scores = {"a": 0.6}
        ws = compute_failure_weights(scores, threshold=0.8, boost=2.0)
        assert ws[0].weight == 2.0

    def test_sorted_output(self):
        scores = {"c": 0.1, "a": 0.2, "b": 0.3}
        ws = compute_failure_weights(scores)
        ids = [w.item_id for w in ws]
        assert ids == ["a", "b", "c"]


# ---- compute_boundary_weights ----


class TestComputeBoundaryWeights:
    def test_in_boundary(self):
        scores = {"a": 0.45, "b": 0.55}
        ws = compute_boundary_weights(scores, threshold=0.5, width=0.1, boost=2.0)
        assert all(w.weight == 2.0 for w in ws)

    def test_outside_boundary(self):
        scores = {"a": 0.1, "b": 0.9}
        ws = compute_boundary_weights(scores, threshold=0.5, width=0.1, boost=2.0)
        assert all(w.weight == 1.0 for w in ws)

    def test_on_edge_included(self):
        scores = {"a": 0.4, "b": 0.6}
        ws = compute_boundary_weights(scores, threshold=0.5, width=0.1)
        assert all(w.weight > 1.0 for w in ws)

    def test_empty_scores(self):
        ws = compute_boundary_weights({})
        assert ws == []

    def test_narrow_width(self):
        scores = {"a": 0.49, "b": 0.51, "c": 0.45}
        ws = compute_boundary_weights(scores, threshold=0.5, width=0.02)
        by_id = {w.item_id: w for w in ws}
        assert by_id["a"].weight > 1.0
        assert by_id["b"].weight > 1.0
        assert by_id["c"].weight == 1.0

    def test_reason_contains_boundary(self):
        scores = {"a": 0.5}
        ws = compute_boundary_weights(scores, threshold=0.5, width=0.1)
        assert "boundary" in ws[0].reason


# ---- compute_cohort_weights ----


class TestComputeCohortWeights:
    def test_worst_cohort_boosted(self):
        scores = {"a": 0.2, "b": 0.3, "c": 0.8, "d": 0.9}
        cohorts = {"low": ["a", "b"], "high": ["c", "d"]}
        ws = compute_cohort_weights(scores, cohorts)
        by_id = {w.item_id: w for w in ws}
        assert by_id["a"].weight == 2.0
        assert by_id["b"].weight == 2.0
        assert by_id["c"].weight == 1.0

    def test_no_cohorts(self):
        scores = {"a": 0.5}
        ws = compute_cohort_weights(scores, {})
        assert ws[0].weight == 1.0

    def test_empty_scores(self):
        ws = compute_cohort_weights({}, {"g": ["x"]})
        assert ws == []

    def test_single_cohort(self):
        scores = {"a": 0.1, "b": 0.9}
        cohorts = {"only": ["a", "b"]}
        ws = compute_cohort_weights(scores, cohorts)
        # Single cohort is the worst by default, both get boosted
        assert all(w.weight == 2.0 for w in ws)

    def test_item_not_in_any_cohort(self):
        scores = {"a": 0.1, "b": 0.9, "c": 0.5}
        cohorts = {"g": ["a"]}
        ws = compute_cohort_weights(scores, cohorts)
        by_id = {w.item_id: w for w in ws}
        assert by_id["a"].weight == 2.0
        assert by_id["c"].weight == 1.0

    def test_cohort_members_not_in_scores(self):
        scores = {"a": 0.5}
        cohorts = {"g": ["x", "y"]}
        ws = compute_cohort_weights(scores, cohorts)
        # No scored members in cohort, fallback
        assert ws[0].weight == 1.0


# ---- merge_weights ----


class TestMergeWeights:
    def test_single_list(self):
        ws = [AdversarialWeight("a", 2.0, "r1")]
        merged = merge_weights([ws])
        assert len(merged) == 1
        assert merged[0].weight == 2.0

    def test_two_lists_same_ids(self):
        w1 = [AdversarialWeight("a", 2.0, "failure")]
        w2 = [AdversarialWeight("a", 3.0, "boundary")]
        merged = merge_weights([w1, w2])
        assert len(merged) == 1
        assert merged[0].weight == 5.0
        assert "failure" in merged[0].reason
        assert "boundary" in merged[0].reason

    def test_disjoint_lists(self):
        w1 = [AdversarialWeight("a", 1.0, "r1")]
        w2 = [AdversarialWeight("b", 2.0, "r2")]
        merged = merge_weights([w1, w2])
        assert len(merged) == 2

    def test_empty_lists(self):
        merged = merge_weights([])
        assert merged == []

    def test_empty_inner_list(self):
        merged = merge_weights([[]])
        assert merged == []

    def test_three_lists(self):
        w1 = [AdversarialWeight("a", 1.0, "r1")]
        w2 = [AdversarialWeight("a", 2.0, "r2")]
        w3 = [AdversarialWeight("a", 3.0, "r3")]
        merged = merge_weights([w1, w2, w3])
        assert merged[0].weight == 6.0

    def test_sorted_output(self):
        w1 = [AdversarialWeight("c", 1.0, "r"), AdversarialWeight("a", 1.0, "r")]
        merged = merge_weights([w1])
        assert [m.item_id for m in merged] == ["a", "c"]


# ---- run_adversarial_sampling ----


class TestRunAdversarialSampling:
    def test_empty_scores(self):
        r = run_adversarial_sampling({}, AdversarialConfig())
        assert r.selected_ids == []
        assert r.total_pool == 0

    def test_small_pool(self):
        scores = {"a": 0.1, "b": 0.9}
        config = AdversarialConfig(sample_size=10, seed=42)
        r = run_adversarial_sampling(scores, config)
        # Pool smaller than sample_size: all selected
        assert set(r.selected_ids) == {"a", "b"}
        assert r.total_pool == 2

    def test_failure_count(self):
        scores = {"a": 0.1, "b": 0.3, "c": 0.7}
        r = run_adversarial_sampling(scores, AdversarialConfig(seed=1))
        assert r.failure_count == 2

    def test_boundary_count(self):
        scores = {"a": 0.45, "b": 0.55, "c": 0.9}
        config = AdversarialConfig(boundary_width=0.1, seed=1)
        r = run_adversarial_sampling(scores, config)
        assert r.boundary_count == 2

    def test_deterministic_with_seed(self):
        scores = {f"item_{i}": i / 100.0 for i in range(100)}
        config = AdversarialConfig(sample_size=20, seed=42)
        r1 = run_adversarial_sampling(scores, config)
        r2 = run_adversarial_sampling(scores, config)
        assert r1.selected_ids == r2.selected_ids

    def test_different_seeds_different_results(self):
        scores = {f"item_{i}": i / 100.0 for i in range(100)}
        r1 = run_adversarial_sampling(scores, AdversarialConfig(sample_size=20, seed=1))
        r2 = run_adversarial_sampling(scores, AdversarialConfig(sample_size=20, seed=2))
        # Very unlikely to be identical
        assert r1.selected_ids != r2.selected_ids

    def test_with_cohorts(self):
        scores = {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}
        cohorts = {"weak": ["a", "b"], "strong": ["c", "d"]}
        config = AdversarialConfig(sample_size=4, seed=1)
        r = run_adversarial_sampling(scores, config, cohorts=cohorts)
        assert len(r.selected_ids) == 4
        assert r.total_pool == 4

    def test_weights_populated(self):
        scores = {"a": 0.1, "b": 0.9}
        r = run_adversarial_sampling(scores, AdversarialConfig(seed=1))
        assert len(r.weights) == 2
        by_id = {w.item_id: w for w in r.weights}
        # "a" is a failure and near boundary: should have higher weight
        assert by_id["a"].weight > by_id["b"].weight

    def test_sample_size_respected(self):
        scores = {f"i{i}": i / 200.0 for i in range(200)}
        config = AdversarialConfig(sample_size=30, seed=7)
        r = run_adversarial_sampling(scores, config)
        assert len(r.selected_ids) == 30

    def test_failures_boosted_in_selection(self):
        # Many passes, few failures: failures should appear more often
        scores = {f"pass_{i}": 0.9 for i in range(90)}
        scores.update({f"fail_{i}": 0.1 for i in range(10)})
        config = AdversarialConfig(sample_size=20, failure_boost=10.0, seed=42)
        r = run_adversarial_sampling(scores, config)
        fail_selected = [s for s in r.selected_ids if s.startswith("fail_")]
        # With high boost, most failures should be selected
        assert len(fail_selected) >= 5

    def test_no_duplicate_selections(self):
        scores = {f"i{i}": i / 50.0 for i in range(50)}
        config = AdversarialConfig(sample_size=25, seed=1)
        r = run_adversarial_sampling(scores, config)
        assert len(r.selected_ids) == len(set(r.selected_ids))


# ---- format_adversarial_report ----


class TestFormatAdversarialReport:
    def test_contains_header(self):
        r = AdversarialResult(total_pool=10, failure_count=3, boundary_count=2)
        report = format_adversarial_report(r)
        assert "Adversarial Sampling Report" in report

    def test_contains_stats(self):
        r = AdversarialResult(
            selected_ids=["a"],
            total_pool=5,
            failure_count=2,
            boundary_count=1,
        )
        report = format_adversarial_report(r)
        assert "Total pool size: 5" in report
        assert "Failure items: 2" in report
        assert "Boundary items: 1" in report

    def test_contains_selected_ids(self):
        r = AdversarialResult(selected_ids=["x", "y"])
        report = format_adversarial_report(r)
        assert "- x" in report
        assert "- y" in report

    def test_empty_result(self):
        r = AdversarialResult()
        report = format_adversarial_report(r)
        assert "Adversarial Sampling Report" in report
        assert "Total pool size: 0" in report

    def test_top_weights_shown(self):
        ws = [AdversarialWeight(f"i{i}", float(i), "r") for i in range(15)]
        r = AdversarialResult(weights=ws, total_pool=15)
        report = format_adversarial_report(r)
        # Top 10 should be shown
        assert "i14" in report
