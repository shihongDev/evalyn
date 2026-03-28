"""Tests for evaluation profiles."""

import pytest
from evalyn_sdk.models import Metric, MetricSpec, MetricResult
from evalyn_sdk.evaluation.profiles import (
    EvalProfile,
    get_profile,
    list_profiles,
    BUILTIN_PROFILES,
    SMOKE_TEST,
    STANDARD,
    THOROUGH,
    COST_OPTIMIZED,
)


def _metric(mid: str, mtype: str = "objective") -> Metric:
    spec = MetricSpec(id=mid, name=mid, type=mtype)
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=mid, item_id=i.id, call_id=c.id,
        score=1.0, passed=True, details={},
    ))


class TestBuiltinProfiles:
    def test_smoke_test(self):
        assert SMOKE_TEST.name == "smoke-test"
        assert SMOKE_TEST.metric_types == ["objective"]
        assert SMOKE_TEST.max_workers == 4
        assert SMOKE_TEST.checkpoint_enabled is False

    def test_standard(self):
        assert STANDARD.name == "standard"
        assert "subjective" in STANDARD.metric_types
        assert STANDARD.max_workers == 2

    def test_thorough(self):
        assert THOROUGH.confidence_method == "consistency"
        assert THOROUGH.confidence_samples == 5

    def test_cost_optimized(self):
        assert COST_OPTIMIZED.provider == "ollama"
        assert COST_OPTIMIZED.batch_mode is True

    def test_four_builtin_profiles(self):
        assert len(BUILTIN_PROFILES) == 4


class TestGetProfile:
    def test_get_builtin(self):
        p = get_profile("standard")
        assert p.name == "standard"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_get_custom(self):
        customs = {"my-profile": {"max_workers": 8, "provider": "openai"}}
        p = get_profile("my-profile", custom_profiles=customs)
        assert p.name == "my-profile"
        assert p.max_workers == 8
        assert p.provider == "openai"

    def test_custom_overrides_builtin(self):
        customs = {"standard": {"max_workers": 16}}
        p = get_profile("standard", custom_profiles=customs)
        assert p.max_workers == 16  # custom overrides


class TestListProfiles:
    def test_list_builtin_only(self):
        profiles = list_profiles()
        assert len(profiles) == 4
        names = {p.name for p in profiles}
        assert "smoke-test" in names
        assert "standard" in names

    def test_list_with_custom(self):
        customs = {"ci-gate": {"max_workers": 1}}
        profiles = list_profiles(custom_profiles=customs)
        assert len(profiles) == 5
        names = {p.name for p in profiles}
        assert "ci-gate" in names


class TestFilterMetrics:
    def test_smoke_test_filters_subjective(self):
        metrics = [
            _metric("json_valid", "objective"),
            _metric("helpfulness", "subjective"),
            _metric("latency", "objective"),
        ]
        filtered = SMOKE_TEST.filter_metrics(metrics)
        assert len(filtered) == 2
        assert all(m.spec.type == "objective" for m in filtered)

    def test_standard_keeps_all(self):
        metrics = [
            _metric("json_valid", "objective"),
            _metric("helpfulness", "subjective"),
        ]
        filtered = STANDARD.filter_metrics(metrics)
        assert len(filtered) == 2

    def test_empty_metrics(self):
        assert SMOKE_TEST.filter_metrics([]) == []


class TestEvalProfileSerialization:
    def test_as_dict(self):
        d = STANDARD.as_dict()
        assert d["name"] == "standard"
        assert d["max_workers"] == 2
        assert "metric_types" in d

    def test_roundtrip(self):
        d = THOROUGH.as_dict()
        restored = EvalProfile.from_dict(d)
        assert restored.name == THOROUGH.name
        assert restored.confidence_method == THOROUGH.confidence_method
        assert restored.confidence_samples == THOROUGH.confidence_samples

    def test_from_dict_defaults(self):
        p = EvalProfile.from_dict({"name": "minimal"})
        assert p.max_workers == 1
        assert p.provider == "gemini"
        assert p.checkpoint_enabled is True
