"""Tests for embedding drift sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.drift_sampling import (
    DriftConfig,
    DriftResult,
    DriftScore,
    compute_dataset_drift,
    compute_text_drift,
    format_drift_report,
    identify_new_items,
    identify_removed_items,
    run_drift_sampling,
    sample_by_drift,
)


# ---- DriftScore ----


class TestDriftScore:
    def test_defaults(self):
        s = DriftScore(item_id="a", drift_magnitude=0.5)
        assert s.item_id == "a"
        assert s.drift_magnitude == 0.5
        assert s.old_version == ""
        assert s.new_version == ""

    def test_custom_values(self):
        s = DriftScore(item_id="x", drift_magnitude=0.9, old_version="v1", new_version="v2")
        assert s.old_version == "v1"
        assert s.new_version == "v2"

    def test_as_dict(self):
        s = DriftScore(item_id="b", drift_magnitude=0.3, old_version="old", new_version="new")
        d = s.as_dict()
        assert d == {
            "item_id": "b",
            "drift_magnitude": 0.3,
            "old_version": "old",
            "new_version": "new",
        }

    def test_from_dict_full(self):
        d = {"item_id": "c", "drift_magnitude": 0.7, "old_version": "a", "new_version": "b"}
        s = DriftScore.from_dict(d)
        assert s.item_id == "c"
        assert s.drift_magnitude == 0.7
        assert s.old_version == "a"

    def test_from_dict_defaults(self):
        s = DriftScore.from_dict({"item_id": "z", "drift_magnitude": 0.1})
        assert s.old_version == ""
        assert s.new_version == ""

    def test_roundtrip(self):
        s = DriftScore(item_id="rt", drift_magnitude=0.42, old_version="v1", new_version="v2")
        assert DriftScore.from_dict(s.as_dict()) == s


# ---- DriftConfig ----


class TestDriftConfig:
    def test_defaults(self):
        c = DriftConfig()
        assert c.sample_size == 50
        assert c.min_drift == 0.1
        assert c.seed is None

    def test_custom_values(self):
        c = DriftConfig(sample_size=10, min_drift=0.5, seed=42)
        assert c.sample_size == 10
        assert c.min_drift == 0.5
        assert c.seed == 42

    def test_as_dict(self):
        c = DriftConfig(sample_size=20, min_drift=0.2, seed=7)
        d = c.as_dict()
        assert d == {"sample_size": 20, "min_drift": 0.2, "seed": 7}

    def test_from_dict_full(self):
        c = DriftConfig.from_dict({"sample_size": 30, "min_drift": 0.3, "seed": 99})
        assert c.sample_size == 30
        assert c.min_drift == 0.3
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = DriftConfig.from_dict({})
        assert c.sample_size == 50
        assert c.min_drift == 0.1
        assert c.seed is None

    def test_roundtrip(self):
        c = DriftConfig(sample_size=15, min_drift=0.05, seed=3)
        assert DriftConfig.from_dict(c.as_dict()) == c


# ---- DriftResult ----


class TestDriftResult:
    def test_defaults(self):
        r = DriftResult()
        assert r.selected_ids == []
        assert r.drift_scores == []
        assert r.mean_drift == 0.0
        assert r.max_drift == 0.0
        assert r.total_pool == 0

    def test_custom_values(self):
        scores = [DriftScore(item_id="a", drift_magnitude=0.5)]
        r = DriftResult(
            selected_ids=["a"],
            drift_scores=scores,
            mean_drift=0.5,
            max_drift=0.5,
            total_pool=1,
        )
        assert r.selected_ids == ["a"]
        assert len(r.drift_scores) == 1

    def test_as_dict(self):
        r = DriftResult(
            selected_ids=["x"],
            drift_scores=[DriftScore(item_id="x", drift_magnitude=0.8)],
            mean_drift=0.8,
            max_drift=0.8,
            total_pool=1,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["x"]
        assert d["mean_drift"] == 0.8
        assert len(d["drift_scores"]) == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["a", "b"],
            "drift_scores": [
                {"item_id": "a", "drift_magnitude": 0.6},
                {"item_id": "b", "drift_magnitude": 0.4},
            ],
            "mean_drift": 0.5,
            "max_drift": 0.6,
            "total_pool": 2,
        }
        r = DriftResult.from_dict(d)
        assert len(r.drift_scores) == 2
        assert r.drift_scores[0].item_id == "a"

    def test_from_dict_defaults(self):
        r = DriftResult.from_dict({})
        assert r.selected_ids == []
        assert r.mean_drift == 0.0

    def test_roundtrip(self):
        r = DriftResult(
            selected_ids=["q"],
            drift_scores=[DriftScore(item_id="q", drift_magnitude=0.33)],
            mean_drift=0.33,
            max_drift=0.33,
            total_pool=1,
        )
        r2 = DriftResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.mean_drift == r.mean_drift
        assert r2.drift_scores[0] == r.drift_scores[0]


# ---- compute_text_drift ----


class TestComputeTextDrift:
    def test_identical_texts(self):
        assert compute_text_drift("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert compute_text_drift("alpha beta", "gamma delta") == 1.0

    def test_partial_overlap(self):
        # "hello" shared, "world" vs "earth" different
        # intersection = {"hello"}, union = {"hello", "world", "earth"}
        # 1 - 1/3 = 2/3
        drift = compute_text_drift("hello world", "hello earth")
        assert abs(drift - 2.0 / 3.0) < 1e-9

    def test_both_empty(self):
        assert compute_text_drift("", "") == 0.0

    def test_one_empty(self):
        assert compute_text_drift("", "something") == 1.0
        assert compute_text_drift("something", "") == 1.0

    def test_case_insensitive(self):
        assert compute_text_drift("Hello World", "hello world") == 0.0

    def test_duplicate_words_no_effect(self):
        # Sets: {"the", "cat"} vs {"the", "cat"} - identical
        assert compute_text_drift("the the cat", "the cat cat") == 0.0

    def test_returns_float_between_0_and_1(self):
        drift = compute_text_drift("some text here", "some other stuff")
        assert 0.0 <= drift <= 1.0


# ---- compute_dataset_drift ----


class TestComputeDatasetDrift:
    def test_all_common(self):
        old = {"a": "hello world", "b": "foo bar"}
        new = {"a": "hello earth", "b": "foo bar"}
        scores = compute_dataset_drift(old, new)
        assert len(scores) == 2
        ids = [s.item_id for s in scores]
        assert "a" in ids
        assert "b" in ids

    def test_no_common(self):
        old = {"a": "x"}
        new = {"b": "y"}
        assert compute_dataset_drift(old, new) == []

    def test_empty_inputs(self):
        assert compute_dataset_drift({}, {}) == []
        assert compute_dataset_drift({"a": "x"}, {}) == []

    def test_stores_versions(self):
        old = {"item1": "old text"}
        new = {"item1": "new text"}
        scores = compute_dataset_drift(old, new)
        assert scores[0].old_version == "old text"
        assert scores[0].new_version == "new text"

    def test_sorted_by_id(self):
        old = {"c": "x", "a": "x", "b": "x"}
        new = {"c": "y", "a": "y", "b": "y"}
        scores = compute_dataset_drift(old, new)
        assert [s.item_id for s in scores] == ["a", "b", "c"]

    def test_identical_items_zero_drift(self):
        old = {"a": "same text"}
        new = {"a": "same text"}
        scores = compute_dataset_drift(old, new)
        assert scores[0].drift_magnitude == 0.0


# ---- identify_new_items / identify_removed_items ----


class TestIdentifyItems:
    def test_new_items(self):
        old = {"a": "x", "b": "y"}
        new = {"a": "x", "c": "z", "d": "w"}
        assert identify_new_items(old, new) == ["c", "d"]

    def test_no_new_items(self):
        old = {"a": "x", "b": "y"}
        new = {"a": "x"}
        assert identify_new_items(old, new) == []

    def test_removed_items(self):
        old = {"a": "x", "b": "y", "c": "z"}
        new = {"a": "x"}
        assert identify_removed_items(old, new) == ["b", "c"]

    def test_no_removed_items(self):
        old = {"a": "x"}
        new = {"a": "x", "b": "y"}
        assert identify_removed_items(old, new) == []

    def test_empty_old(self):
        assert identify_new_items({}, {"a": "x"}) == ["a"]
        assert identify_removed_items({}, {"a": "x"}) == []

    def test_empty_new(self):
        assert identify_new_items({"a": "x"}, {}) == []
        assert identify_removed_items({"a": "x"}, {}) == ["a"]


# ---- sample_by_drift ----


class TestSampleByDrift:
    def test_selects_top_by_magnitude(self):
        scores = [
            DriftScore(item_id="a", drift_magnitude=0.9),
            DriftScore(item_id="b", drift_magnitude=0.5),
            DriftScore(item_id="c", drift_magnitude=0.3),
            DriftScore(item_id="d", drift_magnitude=0.1),
        ]
        config = DriftConfig(sample_size=2, min_drift=0.1)
        result = sample_by_drift(scores, config)
        assert len(result) == 2
        assert "a" in result
        assert "b" in result

    def test_filters_below_min_drift(self):
        scores = [
            DriftScore(item_id="a", drift_magnitude=0.9),
            DriftScore(item_id="b", drift_magnitude=0.05),
        ]
        config = DriftConfig(sample_size=10, min_drift=0.1)
        result = sample_by_drift(scores, config)
        assert result == ["a"]

    def test_empty_scores(self):
        assert sample_by_drift([], DriftConfig()) == []

    def test_all_below_threshold(self):
        scores = [
            DriftScore(item_id="a", drift_magnitude=0.01),
            DriftScore(item_id="b", drift_magnitude=0.05),
        ]
        config = DriftConfig(min_drift=0.1)
        assert sample_by_drift(scores, config) == []

    def test_sample_size_limits(self):
        scores = [DriftScore(item_id=str(i), drift_magnitude=0.5) for i in range(100)]
        config = DriftConfig(sample_size=3, min_drift=0.0)
        result = sample_by_drift(scores, config)
        assert len(result) == 3

    def test_seed_reproducibility(self):
        scores = [DriftScore(item_id=str(i), drift_magnitude=0.5) for i in range(20)]
        config = DriftConfig(sample_size=5, min_drift=0.0, seed=42)
        r1 = sample_by_drift(scores, config)
        r2 = sample_by_drift(scores, config)
        assert r1 == r2

    def test_min_drift_exact_boundary(self):
        scores = [DriftScore(item_id="a", drift_magnitude=0.1)]
        config = DriftConfig(min_drift=0.1)
        # Exactly at threshold should be included
        assert sample_by_drift(scores, config) == ["a"]


# ---- run_drift_sampling ----


class TestRunDriftSampling:
    def test_basic_pipeline(self):
        old = {"a": "hello world", "b": "foo bar", "c": "test case"}
        new = {"a": "hello earth", "b": "foo bar", "c": "completely different words now"}
        config = DriftConfig(sample_size=10, min_drift=0.0)
        result = run_drift_sampling(old, new, config)
        assert isinstance(result, DriftResult)
        assert result.total_pool == 3
        assert len(result.drift_scores) == 3

    def test_mean_drift_computed(self):
        old = {"a": "x", "b": "y"}
        new = {"a": "x", "b": "z"}
        config = DriftConfig(sample_size=10, min_drift=0.0)
        result = run_drift_sampling(old, new, config)
        # "a" drift = 0.0, "b" drift = 1.0 -> mean = 0.5
        assert abs(result.mean_drift - 0.5) < 1e-9

    def test_max_drift_computed(self):
        old = {"a": "x", "b": "same"}
        new = {"a": "y", "b": "same"}
        config = DriftConfig(sample_size=10, min_drift=0.0)
        result = run_drift_sampling(old, new, config)
        assert result.max_drift == 1.0

    def test_empty_datasets(self):
        result = run_drift_sampling({}, {}, DriftConfig())
        assert result.total_pool == 0
        assert result.selected_ids == []
        assert result.mean_drift == 0.0

    def test_no_common_items(self):
        old = {"a": "x"}
        new = {"b": "y"}
        result = run_drift_sampling(old, new, DriftConfig())
        assert result.total_pool == 0
        assert result.selected_ids == []

    def test_respects_min_drift(self):
        old = {"a": "hello world", "b": "hello world"}
        new = {"a": "hello world", "b": "completely new stuff"}
        config = DriftConfig(sample_size=10, min_drift=0.5)
        result = run_drift_sampling(old, new, config)
        # "a" has 0 drift, "b" has high drift
        assert "a" not in result.selected_ids
        assert "b" in result.selected_ids

    def test_respects_sample_size(self):
        old = {str(i): f"word{i}" for i in range(50)}
        new = {str(i): f"different{i}" for i in range(50)}
        config = DriftConfig(sample_size=5, min_drift=0.0)
        result = run_drift_sampling(old, new, config)
        assert len(result.selected_ids) == 5


# ---- format_drift_report ----


class TestFormatDriftReport:
    def test_contains_header(self):
        result = DriftResult()
        report = format_drift_report(result)
        assert "Drift Sampling Report" in report

    def test_contains_stats(self):
        result = DriftResult(
            selected_ids=["a"],
            drift_scores=[DriftScore(item_id="a", drift_magnitude=0.75)],
            mean_drift=0.75,
            max_drift=0.75,
            total_pool=1,
        )
        report = format_drift_report(result)
        assert "0.7500" in report
        assert "Total common items: 1" in report
        assert "Selected samples: 1" in report

    def test_lists_selected_ids(self):
        result = DriftResult(selected_ids=["item1", "item2"])
        report = format_drift_report(result)
        assert "item1" in report
        assert "item2" in report

    def test_empty_result(self):
        result = DriftResult()
        report = format_drift_report(result)
        assert "Total common items: 0" in report
        assert "Selected samples: 0" in report
