"""Tests for the offline_eval module."""

from __future__ import annotations

import pytest

from evalyn_sdk.offline_eval import (
    METRIC_CAPABILITIES,
    OfflineCapability,
    OfflineCheckResult,
    OfflineConfig,
    check_offline_compatibility,
    estimate_offline_coverage,
    filter_offline_metrics,
    format_offline_check,
    suggest_offline_alternatives,
    validate_offline_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strict_config():
    return OfflineConfig(strict=True, allow_local_models=False)


@pytest.fixture
def lenient_config():
    return OfflineConfig(strict=False, allow_local_models=True)


@pytest.fixture
def local_model_config():
    return OfflineConfig(strict=True, allow_local_models=True, local_model_provider="ollama")


@pytest.fixture
def all_offline_ids():
    return ["exact_match", "regex_match", "json_valid", "bleu", "latency"]


@pytest.fixture
def mixed_ids():
    return ["exact_match", "helpfulness", "bleu", "safety"]


@pytest.fixture
def all_online_ids():
    return ["helpfulness", "coherence", "safety", "hallucination"]


# ---------------------------------------------------------------------------
# OfflineCapability
# ---------------------------------------------------------------------------


class TestOfflineCapability:
    def test_as_dict(self):
        cap = OfflineCapability("exact_match", True, [], "")
        d = cap.as_dict()
        assert d["metric_id"] == "exact_match"
        assert d["offline_supported"] is True
        assert d["requires"] == []

    def test_from_dict_roundtrip(self):
        cap = OfflineCapability("help", False, ["llm_api"], "bleu")
        restored = OfflineCapability.from_dict(cap.as_dict())
        assert restored.metric_id == cap.metric_id
        assert restored.offline_supported == cap.offline_supported
        assert restored.requires == cap.requires
        assert restored.fallback == cap.fallback

    def test_from_dict_defaults(self):
        cap = OfflineCapability.from_dict({
            "metric_id": "x",
            "offline_supported": True,
        })
        assert cap.requires == []
        assert cap.fallback == ""


# ---------------------------------------------------------------------------
# OfflineConfig
# ---------------------------------------------------------------------------


class TestOfflineConfig:
    def test_as_dict(self, strict_config):
        d = strict_config.as_dict()
        assert d["strict"] is True
        assert d["allow_local_models"] is False

    def test_from_dict_roundtrip(self, strict_config):
        restored = OfflineConfig.from_dict(strict_config.as_dict())
        assert restored.strict == strict_config.strict
        assert restored.allow_local_models == strict_config.allow_local_models

    def test_defaults(self):
        cfg = OfflineConfig()
        assert cfg.strict is True
        assert cfg.allow_local_models is True
        assert cfg.local_model_provider == "ollama"
        assert cfg.cached_artifacts_dir == ""

    def test_from_dict_defaults(self):
        cfg = OfflineConfig.from_dict({})
        assert cfg.strict is True
        assert cfg.allow_local_models is True


# ---------------------------------------------------------------------------
# OfflineCheckResult
# ---------------------------------------------------------------------------


class TestOfflineCheckResult:
    def test_as_dict(self):
        r = OfflineCheckResult(True, [], ["exact_match"], [])
        d = r.as_dict()
        assert d["all_offline"] is True
        assert d["offline_metrics"] == ["exact_match"]

    def test_from_dict_roundtrip(self):
        r = OfflineCheckResult(False, ["help"], ["bleu"], ["warn1"])
        restored = OfflineCheckResult.from_dict(r.as_dict())
        assert restored.all_offline == r.all_offline
        assert restored.online_metrics == r.online_metrics
        assert restored.offline_metrics == r.offline_metrics
        assert restored.warnings == r.warnings


# ---------------------------------------------------------------------------
# METRIC_CAPABILITIES
# ---------------------------------------------------------------------------


class TestMetricCapabilities:
    def test_has_at_least_20_entries(self):
        assert len(METRIC_CAPABILITIES) >= 20

    def test_offline_metrics_have_no_requires(self):
        for mid, cap in METRIC_CAPABILITIES.items():
            if cap.offline_supported:
                assert cap.requires == [], f"{mid} is offline but has requires"

    def test_online_metrics_have_requires(self):
        for mid, cap in METRIC_CAPABILITIES.items():
            if not cap.offline_supported:
                assert len(cap.requires) > 0, f"{mid} is online but has no requires"

    def test_known_offline_metrics(self):
        offline = [m for m, c in METRIC_CAPABILITIES.items() if c.offline_supported]
        assert "exact_match" in offline
        assert "bleu" in offline
        assert "json_valid" in offline
        assert "latency" in offline

    def test_known_online_metrics(self):
        online = [m for m, c in METRIC_CAPABILITIES.items() if not c.offline_supported]
        assert "helpfulness" in online
        assert "coherence" in online
        assert "semantic_similarity" in online

    def test_fact_check_requires_internet(self):
        cap = METRIC_CAPABILITIES["fact_check"]
        assert "internet" in cap.requires


# ---------------------------------------------------------------------------
# check_offline_compatibility
# ---------------------------------------------------------------------------


class TestCheckOfflineCompatibility:
    def test_all_offline_metrics(self, all_offline_ids, strict_config):
        result = check_offline_compatibility(all_offline_ids, strict_config)
        assert result.all_offline is True
        assert result.online_metrics == []
        assert set(result.offline_metrics) == set(all_offline_ids)

    def test_mixed_metrics_strict_no_local(self, mixed_ids, strict_config):
        result = check_offline_compatibility(mixed_ids, strict_config)
        assert result.all_offline is False
        assert "helpfulness" in result.online_metrics
        assert "exact_match" in result.offline_metrics

    def test_mixed_metrics_with_local_models(self, mixed_ids, local_model_config):
        result = check_offline_compatibility(mixed_ids, local_model_config)
        # helpfulness and safety only need llm_api, local models can handle it
        assert "helpfulness" in result.offline_metrics
        assert "safety" in result.offline_metrics

    def test_unknown_metric_strict(self, strict_config):
        result = check_offline_compatibility(["nonexistent_metric"], strict_config)
        assert "nonexistent_metric" in result.online_metrics
        assert any("Unknown" in w for w in result.warnings)

    def test_unknown_metric_lenient(self, lenient_config):
        result = check_offline_compatibility(["nonexistent_metric"], lenient_config)
        assert "nonexistent_metric" in result.offline_metrics
        assert any("Unknown" in w for w in result.warnings)

    def test_empty_metrics(self, strict_config):
        result = check_offline_compatibility([], strict_config)
        assert result.all_offline is True
        assert result.online_metrics == []
        assert result.offline_metrics == []

    def test_internet_required_always_online(self, local_model_config):
        result = check_offline_compatibility(["fact_check"], local_model_config)
        assert "fact_check" in result.online_metrics

    def test_warnings_for_local_model_usage(self, local_model_config):
        result = check_offline_compatibility(["helpfulness"], local_model_config)
        assert any("ollama" in w for w in result.warnings)

    def test_fallback_suggestion_in_warning(self, strict_config):
        result = check_offline_compatibility(["helpfulness"], strict_config)
        assert any("fallback" in w.lower() or "token_length" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# filter_offline_metrics
# ---------------------------------------------------------------------------


class TestFilterOfflineMetrics:
    def test_filters_to_offline_only(self, mixed_ids):
        result = filter_offline_metrics(mixed_ids)
        assert "exact_match" in result
        assert "bleu" in result
        assert "helpfulness" not in result
        assert "safety" not in result

    def test_all_offline_returns_all(self, all_offline_ids):
        result = filter_offline_metrics(all_offline_ids)
        assert set(result) == set(all_offline_ids)

    def test_all_online_returns_empty(self, all_online_ids):
        result = filter_offline_metrics(all_online_ids)
        assert result == []

    def test_empty_input(self):
        assert filter_offline_metrics([]) == []

    def test_unknown_metric_excluded(self):
        result = filter_offline_metrics(["unknown_metric", "exact_match"])
        assert result == ["exact_match"]

    def test_preserves_order(self):
        ids = ["bleu", "exact_match", "json_valid"]
        result = filter_offline_metrics(ids)
        assert result == ids


# ---------------------------------------------------------------------------
# suggest_offline_alternatives
# ---------------------------------------------------------------------------


class TestSuggestOfflineAlternatives:
    def test_helpfulness_alternatives(self):
        alts = suggest_offline_alternatives("helpfulness")
        assert len(alts) > 0
        # All alternatives should be offline-capable
        for a in alts:
            cap = METRIC_CAPABILITIES.get(a)
            assert cap is not None and cap.offline_supported, f"{a} not offline"

    def test_already_offline_no_alternatives(self):
        alts = suggest_offline_alternatives("exact_match")
        assert alts == []

    def test_coherence_alternatives(self):
        alts = suggest_offline_alternatives("coherence")
        assert "bleu" in alts

    def test_semantic_similarity_alternatives(self):
        alts = suggest_offline_alternatives("semantic_similarity")
        assert len(alts) > 0

    def test_fact_check_alternatives(self):
        alts = suggest_offline_alternatives("fact_check")
        assert len(alts) > 0

    def test_unknown_metric_empty_alternatives(self):
        alts = suggest_offline_alternatives("totally_unknown")
        assert alts == []


# ---------------------------------------------------------------------------
# validate_offline_run
# ---------------------------------------------------------------------------


class TestValidateOfflineRun:
    def test_all_offline_strict_passes(self, all_offline_ids, strict_config):
        ok, errors = validate_offline_run(all_offline_ids, strict_config)
        assert ok is True
        assert errors == []

    def test_mixed_strict_fails(self, mixed_ids, strict_config):
        ok, errors = validate_offline_run(mixed_ids, strict_config)
        assert ok is False
        assert len(errors) > 0

    def test_mixed_lenient_passes(self, mixed_ids, lenient_config):
        ok, errors = validate_offline_run(mixed_ids, lenient_config)
        assert ok is True

    def test_empty_metrics_fails(self, strict_config):
        ok, errors = validate_offline_run([], strict_config)
        assert ok is False
        assert any("No metrics" in e for e in errors)

    def test_error_messages_include_metric_id(self, strict_config):
        ok, errors = validate_offline_run(["helpfulness"], strict_config)
        assert ok is False
        assert any("helpfulness" in e for e in errors)

    def test_error_messages_include_requirements(self, strict_config):
        ok, errors = validate_offline_run(["helpfulness"], strict_config)
        assert any("llm_api" in e for e in errors)

    def test_local_models_allow_llm_metrics(self, local_model_config):
        ok, errors = validate_offline_run(["helpfulness", "exact_match"], local_model_config)
        assert ok is True

    def test_internet_metrics_fail_even_with_local_models(self, local_model_config):
        ok, errors = validate_offline_run(["fact_check"], local_model_config)
        assert ok is False


# ---------------------------------------------------------------------------
# format_offline_check
# ---------------------------------------------------------------------------


class TestFormatOfflineCheck:
    def test_all_offline_header(self, all_offline_ids, strict_config):
        result = check_offline_compatibility(all_offline_ids, strict_config)
        text = format_offline_check(result)
        assert "ALL METRICS OFFLINE-COMPATIBLE" in text

    def test_some_online_header(self, mixed_ids, strict_config):
        result = check_offline_compatibility(mixed_ids, strict_config)
        text = format_offline_check(result)
        assert "SOME METRICS REQUIRE ONLINE ACCESS" in text

    def test_lists_offline_metrics(self, all_offline_ids, strict_config):
        result = check_offline_compatibility(all_offline_ids, strict_config)
        text = format_offline_check(result)
        assert "exact_match" in text

    def test_lists_online_metrics(self, mixed_ids, strict_config):
        result = check_offline_compatibility(mixed_ids, strict_config)
        text = format_offline_check(result)
        assert "helpfulness" in text

    def test_shows_fallback(self, strict_config):
        result = check_offline_compatibility(["helpfulness"], strict_config)
        text = format_offline_check(result)
        assert "fallback" in text.lower() or "token_length" in text

    def test_shows_warnings(self, strict_config):
        result = check_offline_compatibility(["nonexistent"], strict_config)
        text = format_offline_check(result)
        assert "Warnings" in text


# ---------------------------------------------------------------------------
# estimate_offline_coverage
# ---------------------------------------------------------------------------


class TestEstimateOfflineCoverage:
    def test_all_offline(self, all_offline_ids):
        assert estimate_offline_coverage(all_offline_ids) == 1.0

    def test_all_online(self, all_online_ids):
        assert estimate_offline_coverage(all_online_ids) == 0.0

    def test_mixed(self, mixed_ids):
        # exact_match and bleu are offline, helpfulness and safety are not
        coverage = estimate_offline_coverage(mixed_ids)
        assert coverage == pytest.approx(0.5)

    def test_empty(self):
        assert estimate_offline_coverage([]) == 0.0

    def test_unknown_metrics_count_as_online(self):
        coverage = estimate_offline_coverage(["unknown1", "exact_match"])
        assert coverage == pytest.approx(0.5)

    def test_single_offline(self):
        assert estimate_offline_coverage(["exact_match"]) == 1.0

    def test_single_online(self):
        assert estimate_offline_coverage(["helpfulness"]) == 0.0
