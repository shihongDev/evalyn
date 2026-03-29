"""Tests for metadata-conditional sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.metadata_sampling import (
    MetadataConfig,
    MetadataResult,
    SamplingRate,
    apply_sampling_rates,
    compute_actual_rates,
    format_metadata_report,
    group_by_metadata,
    run_metadata_sampling,
)


# ---------------------------------------------------------------------------
# SamplingRate
# ---------------------------------------------------------------------------


class TestSamplingRate:
    def test_defaults(self):
        sr = SamplingRate(field_value="en", rate=0.8)
        assert sr.field_value == "en"
        assert sr.rate == 0.8
        assert sr.description == ""

    def test_custom_description(self):
        sr = SamplingRate(field_value="fr", rate=0.3, description="low rate")
        assert sr.description == "low rate"

    def test_as_dict(self):
        sr = SamplingRate(field_value="en", rate=0.9, description="high")
        d = sr.as_dict()
        assert d == {
            "field_value": "en",
            "rate": 0.9,
            "description": "high",
        }

    def test_from_dict_full(self):
        d = {"field_value": "ja", "rate": 0.5, "description": "medium"}
        sr = SamplingRate.from_dict(d)
        assert sr.field_value == "ja"
        assert sr.rate == 0.5
        assert sr.description == "medium"

    def test_from_dict_minimal(self):
        d = {"field_value": "de", "rate": 0.1}
        sr = SamplingRate.from_dict(d)
        assert sr.description == ""

    def test_roundtrip(self):
        sr = SamplingRate(field_value="zh", rate=0.7, description="test")
        assert SamplingRate.from_dict(sr.as_dict()) == sr


# ---------------------------------------------------------------------------
# MetadataConfig
# ---------------------------------------------------------------------------


class TestMetadataConfig:
    def test_defaults(self):
        c = MetadataConfig(field_name="language")
        assert c.field_name == "language"
        assert c.rates == []
        assert c.default_rate == 0.5
        assert c.seed is None

    def test_custom_values(self):
        rates = [SamplingRate(field_value="en", rate=0.9)]
        c = MetadataConfig(
            field_name="lang",
            rates=rates,
            default_rate=0.3,
            seed=42,
        )
        assert c.default_rate == 0.3
        assert c.seed == 42
        assert len(c.rates) == 1

    def test_as_dict(self):
        c = MetadataConfig(
            field_name="topic",
            rates=[SamplingRate(field_value="math", rate=1.0)],
            default_rate=0.2,
            seed=7,
        )
        d = c.as_dict()
        assert d["field_name"] == "topic"
        assert len(d["rates"]) == 1
        assert d["rates"][0]["field_value"] == "math"
        assert d["default_rate"] == 0.2
        assert d["seed"] == 7

    def test_from_dict_full(self):
        d = {
            "field_name": "lang",
            "rates": [{"field_value": "en", "rate": 0.8}],
            "default_rate": 0.4,
            "seed": 99,
        }
        c = MetadataConfig.from_dict(d)
        assert c.field_name == "lang"
        assert len(c.rates) == 1
        assert c.default_rate == 0.4
        assert c.seed == 99

    def test_from_dict_minimal(self):
        d = {"field_name": "cat"}
        c = MetadataConfig.from_dict(d)
        assert c.rates == []
        assert c.default_rate == 0.5
        assert c.seed is None

    def test_roundtrip(self):
        c = MetadataConfig(
            field_name="topic",
            rates=[SamplingRate(field_value="a", rate=0.1)],
            default_rate=0.6,
            seed=12,
        )
        assert MetadataConfig.from_dict(c.as_dict()).field_name == c.field_name


# ---------------------------------------------------------------------------
# MetadataResult
# ---------------------------------------------------------------------------


class TestMetadataResult:
    def test_defaults(self):
        r = MetadataResult()
        assert r.selected_ids == []
        assert r.per_value_counts == {}
        assert r.total_pool == 0
        assert r.total_selected == 0

    def test_as_dict(self):
        r = MetadataResult(
            selected_ids=["a", "b"],
            per_value_counts={"en": {"total": 5, "sampled": 2, "rate": 0.5}},
            total_pool=10,
            total_selected=2,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert d["total_pool"] == 10

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "per_value_counts": {"fr": {"total": 3, "sampled": 1, "rate": 0.3}},
            "total_pool": 5,
            "total_selected": 1,
        }
        r = MetadataResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert r.total_selected == 1

    def test_from_dict_empty(self):
        r = MetadataResult.from_dict({})
        assert r.selected_ids == []
        assert r.total_pool == 0

    def test_roundtrip(self):
        r = MetadataResult(
            selected_ids=["a"],
            per_value_counts={"v": {"total": 2, "sampled": 1, "rate": 0.5}},
            total_pool=2,
            total_selected=1,
        )
        r2 = MetadataResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.total_pool == r.total_pool


# ---------------------------------------------------------------------------
# group_by_metadata
# ---------------------------------------------------------------------------


class TestGroupByMetadata:
    def test_basic_grouping(self):
        items = [
            {"id": "1", "lang": "en"},
            {"id": "2", "lang": "fr"},
            {"id": "3", "lang": "en"},
        ]
        groups = group_by_metadata(items, "lang")
        assert set(groups["en"]) == {"1", "3"}
        assert groups["fr"] == ["2"]

    def test_missing_field_goes_to_unknown(self):
        items = [
            {"id": "1", "lang": "en"},
            {"id": "2"},
            {"id": "3", "lang": "en"},
        ]
        groups = group_by_metadata(items, "lang")
        assert "_unknown" in groups
        assert groups["_unknown"] == ["2"]

    def test_empty_items(self):
        groups = group_by_metadata([], "field")
        assert groups == {}

    def test_all_same_value(self):
        items = [{"id": str(i), "cat": "A"} for i in range(5)]
        groups = group_by_metadata(items, "cat")
        assert len(groups) == 1
        assert len(groups["A"]) == 5

    def test_numeric_values_converted_to_string(self):
        items = [
            {"id": "1", "score": 5},
            {"id": "2", "score": 10},
        ]
        groups = group_by_metadata(items, "score")
        assert "5" in groups
        assert "10" in groups

    def test_single_item(self):
        items = [{"id": "x", "topic": "math"}]
        groups = group_by_metadata(items, "topic")
        assert groups == {"math": ["x"]}


# ---------------------------------------------------------------------------
# apply_sampling_rates
# ---------------------------------------------------------------------------


class TestApplySamplingRates:
    def test_full_rate(self):
        groups = {"A": ["1", "2", "3"]}
        sampled = apply_sampling_rates(groups, {"A": 1.0}, seed=42)
        assert set(sampled["A"]) == {"1", "2", "3"}

    def test_zero_rate(self):
        groups = {"A": ["1", "2", "3"]}
        sampled = apply_sampling_rates(groups, {"A": 0.0}, seed=42)
        assert sampled["A"] == []

    def test_default_rate_used(self):
        groups = {"A": ["1", "2", "3", "4"]}
        sampled = apply_sampling_rates(groups, {}, default_rate=0.5, seed=42)
        assert 0 < len(sampled["A"]) <= 4

    def test_rate_clamped_above_one(self):
        groups = {"A": ["1", "2"]}
        sampled = apply_sampling_rates(groups, {"A": 1.5}, seed=0)
        assert set(sampled["A"]) == {"1", "2"}

    def test_rate_clamped_below_zero(self):
        groups = {"A": ["1", "2"]}
        sampled = apply_sampling_rates(groups, {"A": -0.5}, seed=0)
        assert sampled["A"] == []

    def test_deterministic_with_seed(self):
        groups = {"A": [str(i) for i in range(20)]}
        rates = {"A": 0.5}
        r1 = apply_sampling_rates(groups, rates, seed=123)
        r2 = apply_sampling_rates(groups, rates, seed=123)
        assert r1["A"] == r2["A"]

    def test_different_seeds_different_results(self):
        groups = {"A": [str(i) for i in range(100)]}
        rates = {"A": 0.5}
        r1 = apply_sampling_rates(groups, rates, seed=1)
        r2 = apply_sampling_rates(groups, rates, seed=2)
        # Very unlikely to be the same with 100 items at 50%
        assert set(r1["A"]) != set(r2["A"])

    def test_empty_group(self):
        groups = {"A": []}
        sampled = apply_sampling_rates(groups, {"A": 1.0}, seed=0)
        assert sampled["A"] == []

    def test_multiple_groups(self):
        groups = {"A": ["1", "2"], "B": ["3", "4", "5"]}
        rates = {"A": 1.0, "B": 0.0}
        sampled = apply_sampling_rates(groups, rates, seed=0)
        assert set(sampled["A"]) == {"1", "2"}
        assert sampled["B"] == []


# ---------------------------------------------------------------------------
# run_metadata_sampling
# ---------------------------------------------------------------------------


class TestRunMetadataSampling:
    def _make_items(self):
        return [
            {"id": "1", "lang": "en"},
            {"id": "2", "lang": "en"},
            {"id": "3", "lang": "fr"},
            {"id": "4", "lang": "fr"},
            {"id": "5", "lang": "de"},
        ]

    def test_basic_run(self):
        items = self._make_items()
        config = MetadataConfig(
            field_name="lang",
            rates=[
                SamplingRate(field_value="en", rate=1.0),
                SamplingRate(field_value="fr", rate=0.0),
            ],
            default_rate=1.0,
            seed=42,
        )
        result = run_metadata_sampling(items, config)
        assert "1" in result.selected_ids or "2" in result.selected_ids
        # en at 100%: both selected
        en_count = result.per_value_counts["en"]["sampled"]
        assert en_count == 2
        # fr at 0%: none selected
        fr_count = result.per_value_counts["fr"]["sampled"]
        assert fr_count == 0
        # de at default 100%
        de_count = result.per_value_counts["de"]["sampled"]
        assert de_count == 1

    def test_total_pool_matches(self):
        items = self._make_items()
        config = MetadataConfig(field_name="lang", seed=1)
        result = run_metadata_sampling(items, config)
        assert result.total_pool == 5

    def test_total_selected_consistent(self):
        items = self._make_items()
        config = MetadataConfig(field_name="lang", seed=1)
        result = run_metadata_sampling(items, config)
        assert result.total_selected == len(result.selected_ids)

    def test_per_value_counts_keys(self):
        items = self._make_items()
        config = MetadataConfig(field_name="lang", seed=1)
        result = run_metadata_sampling(items, config)
        assert set(result.per_value_counts.keys()) == {"en", "fr", "de"}

    def test_empty_items(self):
        config = MetadataConfig(field_name="lang", seed=1)
        result = run_metadata_sampling([], config)
        assert result.total_pool == 0
        assert result.total_selected == 0
        assert result.selected_ids == []

    def test_deterministic(self):
        items = [{"id": str(i), "cat": "A"} for i in range(50)]
        config = MetadataConfig(
            field_name="cat",
            rates=[SamplingRate(field_value="A", rate=0.5)],
            seed=42,
        )
        r1 = run_metadata_sampling(items, config)
        r2 = run_metadata_sampling(items, config)
        assert r1.selected_ids == r2.selected_ids


# ---------------------------------------------------------------------------
# compute_actual_rates
# ---------------------------------------------------------------------------


class TestComputeActualRates:
    def test_basic(self):
        result = MetadataResult(
            per_value_counts={
                "en": {"total": 10, "sampled": 5, "rate": 0.5},
                "fr": {"total": 4, "sampled": 2, "rate": 0.5},
            }
        )
        actual = compute_actual_rates(result)
        assert actual["en"] == pytest.approx(0.5)
        assert actual["fr"] == pytest.approx(0.5)

    def test_zero_total(self):
        result = MetadataResult(
            per_value_counts={
                "empty": {"total": 0, "sampled": 0, "rate": 1.0},
            }
        )
        actual = compute_actual_rates(result)
        assert actual["empty"] == 0.0

    def test_full_rate(self):
        result = MetadataResult(
            per_value_counts={
                "a": {"total": 3, "sampled": 3, "rate": 1.0},
            }
        )
        actual = compute_actual_rates(result)
        assert actual["a"] == pytest.approx(1.0)

    def test_empty_counts(self):
        result = MetadataResult(per_value_counts={})
        actual = compute_actual_rates(result)
        assert actual == {}


# ---------------------------------------------------------------------------
# format_metadata_report
# ---------------------------------------------------------------------------


class TestFormatMetadataReport:
    def test_report_contains_header(self):
        result = MetadataResult(
            selected_ids=["1"],
            per_value_counts={"en": {"total": 2, "sampled": 1, "rate": 0.5}},
            total_pool=2,
            total_selected=1,
        )
        report = format_metadata_report(result)
        assert "Metadata Sampling Report" in report
        assert "pool=2" in report
        assert "selected=1" in report

    def test_report_contains_values(self):
        result = MetadataResult(
            selected_ids=["1"],
            per_value_counts={
                "en": {"total": 5, "sampled": 3, "rate": 0.6},
                "fr": {"total": 3, "sampled": 1, "rate": 0.3},
            },
            total_pool=8,
            total_selected=4,
        )
        report = format_metadata_report(result)
        assert "en" in report
        assert "fr" in report

    def test_report_with_empty_result(self):
        result = MetadataResult()
        report = format_metadata_report(result)
        assert "Metadata Sampling Report" in report

    def test_report_values_sorted(self):
        result = MetadataResult(
            per_value_counts={
                "zebra": {"total": 1, "sampled": 1, "rate": 1.0},
                "alpha": {"total": 1, "sampled": 1, "rate": 1.0},
            },
            total_pool=2,
            total_selected=2,
        )
        report = format_metadata_report(result)
        alpha_pos = report.index("alpha")
        zebra_pos = report.index("zebra")
        assert alpha_pos < zebra_pos

    def test_report_is_string(self):
        result = MetadataResult(
            per_value_counts={"a": {"total": 1, "sampled": 1, "rate": 1.0}},
        )
        report = format_metadata_report(result)
        assert isinstance(report, str)
        assert len(report) > 0
