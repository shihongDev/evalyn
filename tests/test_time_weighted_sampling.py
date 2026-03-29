"""Tests for time-weighted sampling module."""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.time_weighted_sampling import (
    TimeWeight,
    TimeWeightConfig,
    TimeWeightResult,
    compute_age_days,
    compute_exponential_weight,
    compute_time_weights,
    ensure_minimum_representation,
    format_time_weight_report,
    run_time_weighted_sampling,
    sample_by_time,
)


# ---------------------------------------------------------------------------
# TimeWeight
# ---------------------------------------------------------------------------


class TestTimeWeight:
    def test_basic_creation(self):
        w = TimeWeight(item_id="a", timestamp="2026-01-01T00:00:00+00:00", age_days=5.0, weight=0.8)
        assert w.item_id == "a"
        assert w.age_days == 5.0
        assert w.weight == 0.8

    def test_as_dict(self):
        w = TimeWeight(item_id="x", timestamp="2026-01-01T00:00:00+00:00", age_days=1.0, weight=0.5)
        d = w.as_dict()
        assert d == {
            "item_id": "x",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "age_days": 1.0,
            "weight": 0.5,
        }

    def test_from_dict(self):
        d = {
            "item_id": "y",
            "timestamp": "2026-03-01T12:00:00+00:00",
            "age_days": 3.5,
            "weight": 0.9,
        }
        w = TimeWeight.from_dict(d)
        assert w.item_id == "y"
        assert w.age_days == 3.5

    def test_roundtrip(self):
        original = TimeWeight(item_id="rt", timestamp="2026-02-15T00:00:00+00:00", age_days=10.0, weight=0.3)
        restored = TimeWeight.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.timestamp == original.timestamp
        assert restored.age_days == original.age_days
        assert restored.weight == original.weight


# ---------------------------------------------------------------------------
# TimeWeightConfig
# ---------------------------------------------------------------------------


class TestTimeWeightConfig:
    def test_defaults(self):
        c = TimeWeightConfig()
        assert c.half_life_days == 7.0
        assert c.min_weight == 0.01
        assert c.reference_time == ""

    def test_custom_values(self):
        c = TimeWeightConfig(half_life_days=14.0, min_weight=0.05, reference_time="2026-03-01T00:00:00+00:00")
        assert c.half_life_days == 14.0
        assert c.min_weight == 0.05

    def test_as_dict(self):
        c = TimeWeightConfig(half_life_days=3.0)
        d = c.as_dict()
        assert d["half_life_days"] == 3.0

    def test_from_dict(self):
        d = {"half_life_days": 10.0, "min_weight": 0.02, "reference_time": "2026-01-01T00:00:00+00:00"}
        c = TimeWeightConfig.from_dict(d)
        assert c.half_life_days == 10.0
        assert c.min_weight == 0.02

    def test_from_dict_defaults(self):
        c = TimeWeightConfig.from_dict({})
        assert c.half_life_days == 7.0
        assert c.min_weight == 0.01
        assert c.reference_time == ""

    def test_roundtrip(self):
        original = TimeWeightConfig(half_life_days=5.0, min_weight=0.1)
        restored = TimeWeightConfig.from_dict(original.as_dict())
        assert restored.half_life_days == original.half_life_days
        assert restored.min_weight == original.min_weight


# ---------------------------------------------------------------------------
# TimeWeightResult
# ---------------------------------------------------------------------------


class TestTimeWeightResult:
    def test_defaults(self):
        r = TimeWeightResult()
        assert r.selected_ids == []
        assert r.weights == []
        assert r.total_pool == 0
        assert r.effective_days_covered == 0.0

    def test_as_dict(self):
        w = TimeWeight(item_id="a", timestamp="2026-01-01T00:00:00+00:00", age_days=1.0, weight=0.5)
        r = TimeWeightResult(selected_ids=["a"], weights=[w], total_pool=1, effective_days_covered=0.0)
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["weights"]) == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x", "y"],
            "weights": [
                {"item_id": "x", "timestamp": "2026-01-01T00:00:00+00:00", "age_days": 1.0, "weight": 0.5},
            ],
            "total_pool": 5,
            "effective_days_covered": 10.0,
        }
        r = TimeWeightResult.from_dict(d)
        assert r.selected_ids == ["x", "y"]
        assert r.total_pool == 5

    def test_roundtrip(self):
        w = TimeWeight(item_id="a", timestamp="2026-01-01T00:00:00+00:00", age_days=2.0, weight=0.7)
        original = TimeWeightResult(selected_ids=["a"], weights=[w], total_pool=1, effective_days_covered=0.0)
        restored = TimeWeightResult.from_dict(original.as_dict())
        assert restored.selected_ids == original.selected_ids
        assert restored.total_pool == original.total_pool


# ---------------------------------------------------------------------------
# compute_age_days
# ---------------------------------------------------------------------------


class TestComputeAgeDays:
    def test_same_time(self):
        ts = "2026-03-01T00:00:00+00:00"
        assert compute_age_days(ts, ts) == 0.0

    def test_one_day(self):
        ts = "2026-03-01T00:00:00+00:00"
        ref = "2026-03-02T00:00:00+00:00"
        assert compute_age_days(ts, ref) == pytest.approx(1.0)

    def test_seven_days(self):
        ts = "2026-03-01T00:00:00+00:00"
        ref = "2026-03-08T00:00:00+00:00"
        assert compute_age_days(ts, ref) == pytest.approx(7.0)

    def test_fractional_day(self):
        ts = "2026-03-01T00:00:00+00:00"
        ref = "2026-03-01T12:00:00+00:00"
        assert compute_age_days(ts, ref) == pytest.approx(0.5)

    def test_future_timestamp_gives_negative(self):
        ts = "2026-03-10T00:00:00+00:00"
        ref = "2026-03-01T00:00:00+00:00"
        assert compute_age_days(ts, ref) < 0

    def test_naive_timestamps_treated_as_utc(self):
        ts = "2026-03-01T00:00:00"
        ref = "2026-03-02T00:00:00"
        assert compute_age_days(ts, ref) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_exponential_weight
# ---------------------------------------------------------------------------


class TestComputeExponentialWeight:
    def test_zero_age(self):
        assert compute_exponential_weight(0.0, 7.0) == pytest.approx(1.0)

    def test_one_half_life(self):
        assert compute_exponential_weight(7.0, 7.0) == pytest.approx(0.5)

    def test_two_half_lives(self):
        assert compute_exponential_weight(14.0, 7.0) == pytest.approx(0.25)

    def test_min_weight_floor(self):
        # Very old item should hit min_weight
        assert compute_exponential_weight(1000.0, 7.0, min_weight=0.01) == 0.01

    def test_custom_min_weight(self):
        w = compute_exponential_weight(1000.0, 7.0, min_weight=0.05)
        assert w == 0.05

    def test_zero_half_life(self):
        assert compute_exponential_weight(5.0, 0.0, min_weight=0.01) == 0.01

    def test_negative_age_gives_high_weight(self):
        w = compute_exponential_weight(-1.0, 7.0)
        assert w > 1.0


# ---------------------------------------------------------------------------
# compute_time_weights
# ---------------------------------------------------------------------------


class TestComputeTimeWeights:
    def test_empty_items(self):
        config = TimeWeightConfig(reference_time="2026-03-10T00:00:00+00:00")
        assert compute_time_weights({}, config) == []

    def test_single_item(self):
        config = TimeWeightConfig(
            half_life_days=7.0,
            reference_time="2026-03-08T00:00:00+00:00",
        )
        items = {"a": "2026-03-01T00:00:00+00:00"}
        result = compute_time_weights(items, config)
        assert len(result) == 1
        assert result[0].item_id == "a"
        assert result[0].age_days == pytest.approx(7.0)
        assert result[0].weight == pytest.approx(0.5)

    def test_recent_items_weighted_higher(self):
        config = TimeWeightConfig(
            half_life_days=7.0,
            reference_time="2026-03-15T00:00:00+00:00",
        )
        items = {
            "recent": "2026-03-14T00:00:00+00:00",
            "old": "2026-03-01T00:00:00+00:00",
        }
        result = compute_time_weights(items, config)
        by_id = {w.item_id: w for w in result}
        assert by_id["recent"].weight > by_id["old"].weight

    def test_min_weight_applied(self):
        config = TimeWeightConfig(
            half_life_days=1.0,
            min_weight=0.05,
            reference_time="2026-03-29T00:00:00+00:00",
        )
        items = {"ancient": "2025-01-01T00:00:00+00:00"}
        result = compute_time_weights(items, config)
        assert result[0].weight == 0.05


# ---------------------------------------------------------------------------
# sample_by_time
# ---------------------------------------------------------------------------


class TestSampleByTime:
    def test_empty_weights(self):
        assert sample_by_time([], 5) == []

    def test_zero_sample_size(self):
        w = [TimeWeight("a", "2026-01-01T00:00:00+00:00", 1.0, 1.0)]
        assert sample_by_time(w, 0) == []

    def test_sample_all(self):
        weights = [
            TimeWeight("a", "2026-01-01T00:00:00+00:00", 1.0, 1.0),
            TimeWeight("b", "2026-01-02T00:00:00+00:00", 0.5, 0.8),
        ]
        result = sample_by_time(weights, 10, seed=42)
        assert set(result) == {"a", "b"}

    def test_sample_size_respected(self):
        weights = [
            TimeWeight(str(i), "2026-01-01T00:00:00+00:00", float(i), 1.0)
            for i in range(20)
        ]
        result = sample_by_time(weights, 5, seed=42)
        assert len(result) == 5

    def test_no_duplicates(self):
        weights = [
            TimeWeight(str(i), "2026-01-01T00:00:00+00:00", float(i), 1.0)
            for i in range(10)
        ]
        result = sample_by_time(weights, 10, seed=42)
        assert len(result) == len(set(result))

    def test_deterministic_with_seed(self):
        weights = [
            TimeWeight(str(i), "2026-01-01T00:00:00+00:00", float(i), 1.0)
            for i in range(20)
        ]
        r1 = sample_by_time(weights, 5, seed=123)
        r2 = sample_by_time(weights, 5, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        weights = [
            TimeWeight(str(i), "2026-01-01T00:00:00+00:00", float(i), 1.0)
            for i in range(100)
        ]
        r1 = sample_by_time(weights, 10, seed=1)
        r2 = sample_by_time(weights, 10, seed=2)
        # Very unlikely to be the same with 100 items
        assert r1 != r2

    def test_high_weight_preferred(self):
        # One item has very high weight, others very low
        weights = [
            TimeWeight("high", "2026-01-01T00:00:00+00:00", 0.0, 1000.0),
            TimeWeight("low1", "2026-01-01T00:00:00+00:00", 10.0, 0.001),
            TimeWeight("low2", "2026-01-01T00:00:00+00:00", 20.0, 0.001),
        ]
        # With 1 sample, should almost always pick the high-weight item
        result = sample_by_time(weights, 1, seed=42)
        assert result == ["high"]


# ---------------------------------------------------------------------------
# ensure_minimum_representation
# ---------------------------------------------------------------------------


class TestEnsureMinimumRepresentation:
    def test_empty(self):
        assert ensure_minimum_representation([]) == []

    def test_no_boost_needed(self):
        weights = [
            TimeWeight("a", "2026-03-01T00:00:00+00:00", 1.0, 0.9),
            TimeWeight("b", "2026-03-02T00:00:00+00:00", 2.0, 0.8),
        ]
        result = ensure_minimum_representation(weights, min_per_period=1, period_days=7.0)
        # Both in same period (age 1-2 days, both in period 0), both have good weight
        assert len(result) == 2

    def test_boost_old_items(self):
        weights = [
            TimeWeight("recent", "2026-03-28T00:00:00+00:00", 1.0, 0.9),
            TimeWeight("old", "2026-02-01T00:00:00+00:00", 56.0, 0.001),
        ]
        result = ensure_minimum_representation(weights, min_per_period=1, period_days=7.0)
        by_id = {w.item_id: w for w in result}
        # Old item should have been boosted
        assert by_id["old"].weight > 0.001

    def test_preserves_item_ids(self):
        weights = [
            TimeWeight(f"item_{i}", "2026-01-01T00:00:00+00:00", float(i * 7), 0.5)
            for i in range(5)
        ]
        result = ensure_minimum_representation(weights)
        result_ids = {w.item_id for w in result}
        assert result_ids == {f"item_{i}" for i in range(5)}

    def test_same_length_output(self):
        weights = [
            TimeWeight("a", "2026-01-01T00:00:00+00:00", 1.0, 0.5),
            TimeWeight("b", "2026-01-01T00:00:00+00:00", 8.0, 0.1),
            TimeWeight("c", "2026-01-01T00:00:00+00:00", 15.0, 0.01),
        ]
        result = ensure_minimum_representation(weights)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# run_time_weighted_sampling
# ---------------------------------------------------------------------------


class TestRunTimeWeightedSampling:
    def test_basic_run(self):
        config = TimeWeightConfig(
            half_life_days=7.0,
            reference_time="2026-03-15T00:00:00+00:00",
        )
        items = {
            "a": "2026-03-14T00:00:00+00:00",
            "b": "2026-03-10T00:00:00+00:00",
            "c": "2026-03-01T00:00:00+00:00",
        }
        result = run_time_weighted_sampling(items, 2, config, seed=42)
        assert len(result.selected_ids) == 2
        assert result.total_pool == 3

    def test_sample_all(self):
        config = TimeWeightConfig(reference_time="2026-03-15T00:00:00+00:00")
        items = {"a": "2026-03-14T00:00:00+00:00"}
        result = run_time_weighted_sampling(items, 10, config, seed=42)
        assert result.selected_ids == ["a"]

    def test_empty_items(self):
        result = run_time_weighted_sampling({}, 5, seed=42)
        assert result.selected_ids == []
        assert result.total_pool == 0

    def test_default_config(self):
        items = {"a": "2026-03-28T00:00:00+00:00"}
        result = run_time_weighted_sampling(items, 1, seed=42)
        assert result.total_pool == 1

    def test_effective_days_covered(self):
        config = TimeWeightConfig(
            half_life_days=30.0,
            reference_time="2026-03-15T00:00:00+00:00",
        )
        items = {
            "recent": "2026-03-14T00:00:00+00:00",
            "old": "2026-02-15T00:00:00+00:00",
        }
        result = run_time_weighted_sampling(items, 2, config, seed=42)
        # Both selected, so effective_days_covered should be about 27 days
        assert result.effective_days_covered > 20

    def test_weights_populated(self):
        config = TimeWeightConfig(reference_time="2026-03-15T00:00:00+00:00")
        items = {"a": "2026-03-14T00:00:00+00:00", "b": "2026-03-13T00:00:00+00:00"}
        result = run_time_weighted_sampling(items, 1, config, seed=42)
        assert len(result.weights) == 2  # all items, not just selected


# ---------------------------------------------------------------------------
# format_time_weight_report
# ---------------------------------------------------------------------------


class TestFormatTimeWeightReport:
    def test_basic_report(self):
        w = TimeWeight(item_id="a", timestamp="2026-03-01T00:00:00+00:00", age_days=5.0, weight=0.7)
        result = TimeWeightResult(
            selected_ids=["a"],
            weights=[w],
            total_pool=1,
            effective_days_covered=0.0,
        )
        report = format_time_weight_report(result)
        assert "Time-Weighted Sampling Report" in report
        assert "Total pool: 1" in report
        assert "Selected: 1" in report

    def test_selected_marker(self):
        w = TimeWeight(item_id="a", timestamp="2026-03-01T00:00:00+00:00", age_days=5.0, weight=0.7)
        result = TimeWeightResult(selected_ids=["a"], weights=[w], total_pool=1)
        report = format_time_weight_report(result)
        assert " *" in report  # selected marker

    def test_unselected_no_marker(self):
        w = TimeWeight(item_id="a", timestamp="2026-03-01T00:00:00+00:00", age_days=5.0, weight=0.7)
        result = TimeWeightResult(selected_ids=[], weights=[w], total_pool=1)
        report = format_time_weight_report(result)
        lines = report.split("\n")
        weight_lines = [l for l in lines if "a:" in l]
        assert len(weight_lines) == 1
        assert not weight_lines[0].endswith(" *")

    def test_empty_result(self):
        result = TimeWeightResult()
        report = format_time_weight_report(result)
        assert "Total pool: 0" in report

    def test_multiple_items_sorted_by_age(self):
        weights = [
            TimeWeight("old", "2026-01-01T00:00:00+00:00", 60.0, 0.1),
            TimeWeight("new", "2026-03-01T00:00:00+00:00", 1.0, 0.9),
        ]
        result = TimeWeightResult(
            selected_ids=["new"], weights=weights, total_pool=2, effective_days_covered=59.0
        )
        report = format_time_weight_report(result)
        # "new" should appear before "old" (sorted by age ascending)
        new_pos = report.index("new:")
        old_pos = report.index("old:")
        assert new_pos < old_pos
