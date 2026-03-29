"""Tests for importance sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.importance_sampling import (
    SampleWeight,
    SamplingConfig,
    SamplingResult,
    compute_confidence_weights,
    compute_custom_weights,
    compute_effective_sample_size,
    compute_inverse_pass_rate_weights,
    format_sampling_report,
    run_importance_sampling,
    weighted_sample,
)


# ---------------------------------------------------------------------------
# SampleWeight
# ---------------------------------------------------------------------------


class TestSampleWeight:
    def test_basic_creation(self):
        w = SampleWeight(item_id="a", weight=1.5, reason="test")
        assert w.item_id == "a"
        assert w.weight == 1.5
        assert w.reason == "test"

    def test_default_reason(self):
        w = SampleWeight(item_id="b", weight=2.0)
        assert w.reason == ""

    def test_as_dict(self):
        w = SampleWeight(item_id="x", weight=0.5, reason="low")
        d = w.as_dict()
        assert d == {"item_id": "x", "weight": 0.5, "reason": "low"}

    def test_from_dict(self):
        d = {"item_id": "y", "weight": 3.0, "reason": "high"}
        w = SampleWeight.from_dict(d)
        assert w.item_id == "y"
        assert w.weight == 3.0
        assert w.reason == "high"

    def test_from_dict_missing_reason(self):
        d = {"item_id": "z", "weight": 1.0}
        w = SampleWeight.from_dict(d)
        assert w.reason == ""

    def test_roundtrip(self):
        original = SampleWeight(item_id="rt", weight=7.7, reason="round")
        restored = SampleWeight.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.weight == original.weight
        assert restored.reason == original.reason


# ---------------------------------------------------------------------------
# SamplingConfig
# ---------------------------------------------------------------------------


class TestSamplingConfig:
    def test_defaults(self):
        c = SamplingConfig(strategy="confidence")
        assert c.sample_size == 100
        assert c.min_weight == 0.01
        assert c.seed is None

    def test_custom_values(self):
        c = SamplingConfig(strategy="inverse_pass_rate", sample_size=50, seed=42)
        assert c.strategy == "inverse_pass_rate"
        assert c.sample_size == 50
        assert c.seed == 42

    def test_as_dict(self):
        c = SamplingConfig(strategy="custom", sample_size=10, min_weight=0.1, seed=7)
        d = c.as_dict()
        assert d["strategy"] == "custom"
        assert d["sample_size"] == 10
        assert d["seed"] == 7

    def test_from_dict(self):
        d = {"strategy": "confidence", "sample_size": 25}
        c = SamplingConfig.from_dict(d)
        assert c.strategy == "confidence"
        assert c.sample_size == 25
        assert c.min_weight == 0.01

    def test_roundtrip(self):
        original = SamplingConfig(strategy="inverse_pass_rate", seed=99)
        restored = SamplingConfig.from_dict(original.as_dict())
        assert restored.strategy == original.strategy
        assert restored.seed == original.seed


# ---------------------------------------------------------------------------
# SamplingResult
# ---------------------------------------------------------------------------


class TestSamplingResult:
    def test_defaults(self):
        r = SamplingResult()
        assert r.selected_ids == []
        assert r.weights_used == []
        assert r.total_pool == 0
        assert r.effective_sample_size == 0.0

    def test_as_dict(self):
        w = SampleWeight(item_id="a", weight=1.0)
        r = SamplingResult(
            selected_ids=["a"],
            weights_used=[w],
            total_pool=5,
            effective_sample_size=3.0,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["weights_used"]) == 1
        assert d["total_pool"] == 5

    def test_from_dict(self):
        d = {
            "selected_ids": ["x", "y"],
            "weights_used": [
                {"item_id": "x", "weight": 2.0, "reason": "r"},
            ],
            "total_pool": 10,
            "effective_sample_size": 5.5,
        }
        r = SamplingResult.from_dict(d)
        assert r.selected_ids == ["x", "y"]
        assert len(r.weights_used) == 1
        assert r.weights_used[0].item_id == "x"

    def test_roundtrip(self):
        w = SampleWeight(item_id="q", weight=4.0, reason="test")
        original = SamplingResult(
            selected_ids=["q"],
            weights_used=[w],
            total_pool=1,
            effective_sample_size=1.0,
        )
        restored = SamplingResult.from_dict(original.as_dict())
        assert restored.selected_ids == original.selected_ids
        assert restored.total_pool == original.total_pool


# ---------------------------------------------------------------------------
# compute_inverse_pass_rate_weights
# ---------------------------------------------------------------------------


class TestInversePassRateWeights:
    def test_low_score_high_weight(self):
        scores = {"a": 0.1, "b": 0.9}
        weights = compute_inverse_pass_rate_weights(scores)
        by_id = {w.item_id: w for w in weights}
        assert by_id["a"].weight > by_id["b"].weight

    def test_zero_score_clamped(self):
        scores = {"a": 0.0}
        weights = compute_inverse_pass_rate_weights(scores, min_weight=0.01)
        assert weights[0].weight == 1.0 / 0.01

    def test_threshold_labels(self):
        scores = {"a": 0.3, "b": 0.7}
        weights = compute_inverse_pass_rate_weights(scores, threshold=0.5)
        by_id = {w.item_id: w for w in weights}
        assert by_id["a"].reason == "failed"
        assert by_id["b"].reason == "passed"

    def test_all_passing(self):
        scores = {"a": 0.8, "b": 0.9, "c": 1.0}
        weights = compute_inverse_pass_rate_weights(scores, threshold=0.5)
        assert all(w.reason == "passed" for w in weights)

    def test_empty_scores(self):
        weights = compute_inverse_pass_rate_weights({})
        assert weights == []

    def test_custom_min_weight(self):
        scores = {"a": 0.001}
        w1 = compute_inverse_pass_rate_weights(scores, min_weight=0.01)
        w2 = compute_inverse_pass_rate_weights(scores, min_weight=0.1)
        # With higher min_weight, the clamped value is larger so weight is smaller
        assert w1[0].weight > w2[0].weight


# ---------------------------------------------------------------------------
# compute_confidence_weights
# ---------------------------------------------------------------------------


class TestConfidenceWeights:
    def test_low_confidence_high_weight(self):
        confs = {"a": 0.1, "b": 0.9}
        weights = compute_confidence_weights(confs)
        by_id = {w.item_id: w for w in weights}
        assert by_id["a"].weight > by_id["b"].weight

    def test_zero_confidence_clamped(self):
        confs = {"a": 0.0}
        weights = compute_confidence_weights(confs)
        assert weights[0].weight == 1.0 / 0.01

    def test_reason_labels(self):
        confs = {"a": 0.3, "b": 0.7}
        weights = compute_confidence_weights(confs)
        by_id = {w.item_id: w for w in weights}
        assert by_id["a"].reason == "low_confidence"
        assert by_id["b"].reason == "high_confidence"

    def test_empty_input(self):
        assert compute_confidence_weights({}) == []

    def test_boundary_confidence(self):
        confs = {"a": 0.5}
        weights = compute_confidence_weights(confs)
        assert weights[0].reason == "high_confidence"


# ---------------------------------------------------------------------------
# compute_custom_weights
# ---------------------------------------------------------------------------


class TestCustomWeights:
    def test_identity_function(self):
        items = {"a": 2.0, "b": 3.0}
        weights = compute_custom_weights(items, lambda x: x)
        by_id = {w.item_id: w for w in weights}
        assert by_id["a"].weight == 2.0
        assert by_id["b"].weight == 3.0

    def test_square_function(self):
        items = {"a": 4.0}
        weights = compute_custom_weights(items, lambda x: x * x)
        assert weights[0].weight == 16.0

    def test_reason_is_custom(self):
        items = {"a": 1.0}
        weights = compute_custom_weights(items, lambda x: x)
        assert weights[0].reason == "custom"

    def test_empty_input(self):
        assert compute_custom_weights({}, lambda x: x) == []


# ---------------------------------------------------------------------------
# weighted_sample
# ---------------------------------------------------------------------------


class TestWeightedSample:
    def test_deterministic_with_seed(self):
        weights = [
            SampleWeight(item_id=f"i{i}", weight=float(i + 1))
            for i in range(10)
        ]
        s1 = weighted_sample(weights, 5, seed=42)
        s2 = weighted_sample(weights, 5, seed=42)
        assert s1 == s2

    def test_different_seeds_differ(self):
        weights = [
            SampleWeight(item_id=f"i{i}", weight=float(i + 1))
            for i in range(20)
        ]
        s1 = weighted_sample(weights, 10, seed=1)
        s2 = weighted_sample(weights, 10, seed=2)
        # Very unlikely to be identical with different seeds
        assert s1 != s2

    def test_empty_weights(self):
        assert weighted_sample([], 5) == []

    def test_zero_sample_size(self):
        weights = [SampleWeight(item_id="a", weight=1.0)]
        assert weighted_sample(weights, 0) == []

    def test_sample_size_capped_to_pool(self):
        weights = [SampleWeight(item_id=f"i{i}", weight=1.0) for i in range(3)]
        result = weighted_sample(weights, 100, seed=42)
        assert len(result) <= 3

    def test_high_weight_item_selected(self):
        weights = [
            SampleWeight(item_id="rare", weight=0.001),
            SampleWeight(item_id="common", weight=1000.0),
        ]
        result = weighted_sample(weights, 1, seed=42)
        assert "common" in result

    def test_returns_unique_ids(self):
        weights = [SampleWeight(item_id=f"i{i}", weight=1.0) for i in range(5)]
        result = weighted_sample(weights, 5, seed=42)
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# compute_effective_sample_size
# ---------------------------------------------------------------------------


class TestEffectiveSampleSize:
    def test_equal_weights(self):
        # ESS should equal n when all weights are equal
        ess = compute_effective_sample_size([1.0, 1.0, 1.0, 1.0])
        assert abs(ess - 4.0) < 1e-9

    def test_single_dominant_weight(self):
        # One very large weight, rest small: ESS near 1
        ess = compute_effective_sample_size([1000.0, 0.001, 0.001, 0.001])
        assert ess < 2.0

    def test_empty_weights(self):
        assert compute_effective_sample_size([]) == 0.0

    def test_single_weight(self):
        assert abs(compute_effective_sample_size([5.0]) - 1.0) < 1e-9

    def test_all_zeros(self):
        assert compute_effective_sample_size([0.0, 0.0]) == 0.0

    def test_ess_less_than_n(self):
        weights = [10.0, 1.0, 1.0, 1.0]
        ess = compute_effective_sample_size(weights)
        assert ess < len(weights)
        assert ess > 1.0


# ---------------------------------------------------------------------------
# run_importance_sampling
# ---------------------------------------------------------------------------


class TestRunImportanceSampling:
    def test_inverse_pass_rate_strategy(self):
        pool = {f"item_{i}": float(i) / 10 for i in range(1, 11)}
        config = SamplingConfig(
            strategy="inverse_pass_rate", sample_size=5, seed=42
        )
        result = run_importance_sampling(pool, config)
        assert len(result.selected_ids) <= 5
        assert result.total_pool == 10
        assert result.effective_sample_size > 0

    def test_confidence_strategy(self):
        pool = {f"item_{i}": float(i) / 10 for i in range(1, 11)}
        config = SamplingConfig(strategy="confidence", sample_size=5, seed=42)
        result = run_importance_sampling(pool, config)
        assert len(result.selected_ids) <= 5

    def test_custom_strategy(self):
        pool = {"a": 1.0, "b": 2.0, "c": 3.0}
        config = SamplingConfig(strategy="custom", sample_size=2, seed=42)
        result = run_importance_sampling(pool, config)
        assert len(result.selected_ids) <= 2

    def test_unknown_strategy_raises(self):
        pool = {"a": 1.0}
        config = SamplingConfig(strategy="unknown")
        with pytest.raises(ValueError, match="(?i)unknown"):
            run_importance_sampling(pool, config)

    def test_deterministic_with_seed(self):
        pool = {f"item_{i}": float(i) / 20 for i in range(20)}
        config = SamplingConfig(
            strategy="inverse_pass_rate", sample_size=10, seed=123
        )
        r1 = run_importance_sampling(pool, config)
        r2 = run_importance_sampling(pool, config)
        assert r1.selected_ids == r2.selected_ids

    def test_weights_populated(self):
        pool = {"a": 0.5, "b": 0.5}
        config = SamplingConfig(strategy="confidence", sample_size=2, seed=1)
        result = run_importance_sampling(pool, config)
        assert len(result.weights_used) == 2

    def test_ess_populated(self):
        pool = {f"i{i}": 0.1 * (i + 1) for i in range(10)}
        config = SamplingConfig(
            strategy="inverse_pass_rate", sample_size=5, seed=42
        )
        result = run_importance_sampling(pool, config)
        assert result.effective_sample_size > 0


# ---------------------------------------------------------------------------
# format_sampling_report
# ---------------------------------------------------------------------------


class TestFormatSamplingReport:
    def test_contains_header(self):
        result = SamplingResult(total_pool=10, effective_sample_size=5.0)
        report = format_sampling_report(result)
        assert "Importance Sampling Report" in report

    def test_contains_pool_size(self):
        result = SamplingResult(total_pool=42, effective_sample_size=10.0)
        report = format_sampling_report(result)
        assert "42" in report

    def test_contains_selected_ids(self):
        result = SamplingResult(
            selected_ids=["item_abc"],
            total_pool=1,
            effective_sample_size=1.0,
        )
        report = format_sampling_report(result)
        assert "item_abc" in report

    def test_shows_top_weights(self):
        weights = [
            SampleWeight(item_id="heavy", weight=100.0, reason="top"),
            SampleWeight(item_id="light", weight=0.1, reason="bottom"),
        ]
        result = SamplingResult(
            selected_ids=["heavy"],
            weights_used=weights,
            total_pool=2,
            effective_sample_size=1.5,
        )
        report = format_sampling_report(result)
        assert "heavy" in report
        assert "Top weighted items" in report

    def test_empty_result(self):
        result = SamplingResult()
        report = format_sampling_report(result)
        assert "Total pool size: 0" in report
