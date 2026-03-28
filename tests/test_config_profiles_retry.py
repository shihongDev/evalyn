"""Tests for config profiles and retry budget."""

import os
import pytest
from unittest.mock import patch
from evalyn_sdk.config_profiles import (
    ConfigProfile,
    load_profiles,
    resolve_config_with_profile,
    get_active_profile_name,
    list_profiles,
)
from evalyn_sdk.evaluation.retry_budget import (
    RetryBudget,
    RetryBudgetConfig,
)


# =============================================================================
# Config profiles tests
# =============================================================================

class TestConfigProfile:
    def test_basic_creation(self):
        p = ConfigProfile(name="dev", overrides={"provider": "ollama"})
        assert p.name == "dev"
        assert p.get("provider") == "ollama"

    def test_get_default(self):
        p = ConfigProfile(name="dev")
        assert p.get("missing", "default") == "default"

    def test_as_dict_roundtrip(self):
        p = ConfigProfile(name="dev", overrides={"a": 1}, description="Development")
        d = p.as_dict()
        assert d["name"] == "dev"
        assert d["overrides"]["a"] == 1

    def test_from_dict(self):
        p = ConfigProfile.from_dict("prod", {"provider": "gemini", "max_workers": 2})
        assert p.name == "prod"
        assert p.get("provider") == "gemini"
        assert p.get("max_workers") == 2


class TestLoadProfiles:
    def test_load_from_config(self):
        config = {
            "profiles": {
                "dev": {"provider": "ollama", "max_workers": 1},
                "prod": {"provider": "gemini", "max_workers": 4},
            }
        }
        profiles = load_profiles(config)
        assert len(profiles) == 2
        assert "dev" in profiles
        assert "prod" in profiles
        assert profiles["dev"].get("provider") == "ollama"

    def test_empty_profiles(self):
        assert load_profiles({}) == {}
        assert load_profiles({"profiles": {}}) == {}

    def test_non_dict_values_skipped(self):
        config = {"profiles": {"bad": "not a dict", "good": {"x": 1}}}
        profiles = load_profiles(config)
        assert len(profiles) == 1
        assert "good" in profiles


class TestResolveConfig:
    def test_no_profile(self):
        config = {"provider": "gemini", "max_workers": 2}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVALYN_PROFILE", None)
            resolved = resolve_config_with_profile(config)
        assert resolved["provider"] == "gemini"
        assert "_active_profile" not in resolved

    def test_profile_overrides(self):
        config = {
            "provider": "gemini",
            "max_workers": 2,
            "profiles": {
                "dev": {"provider": "ollama", "max_workers": 1},
            },
        }
        resolved = resolve_config_with_profile(config, profile_name="dev")
        assert resolved["provider"] == "ollama"
        assert resolved["max_workers"] == 1
        assert resolved["_active_profile"] == "dev"

    def test_nested_merge(self):
        config = {
            "api_keys": {"gemini": "key1", "openai": "key2"},
            "profiles": {
                "dev": {"api_keys": {"gemini": "dev_key"}},
            },
        }
        resolved = resolve_config_with_profile(config, profile_name="dev")
        assert resolved["api_keys"]["gemini"] == "dev_key"
        assert resolved["api_keys"]["openai"] == "key2"  # preserved from base

    def test_unknown_profile_returns_base(self):
        config = {"provider": "gemini"}
        resolved = resolve_config_with_profile(config, profile_name="nonexistent")
        assert resolved["provider"] == "gemini"

    def test_env_var_profile(self):
        config = {
            "profiles": {"staging": {"provider": "openai"}},
            "provider": "gemini",
        }
        with patch.dict(os.environ, {"EVALYN_PROFILE": "staging"}):
            resolved = resolve_config_with_profile(config)
        assert resolved["provider"] == "openai"


class TestGetActiveProfile:
    def test_from_env(self):
        with patch.dict(os.environ, {"EVALYN_PROFILE": "prod"}):
            assert get_active_profile_name() == "prod"

    def test_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVALYN_PROFILE", None)
            assert get_active_profile_name() is None


class TestListProfiles:
    def test_list(self):
        config = {
            "profiles": {
                "dev": {"provider": "ollama"},
                "prod": {"provider": "gemini"},
            }
        }
        result = list_profiles(config)
        assert len(result) == 2
        names = [n for n, _ in result]
        assert "dev" in names
        assert "prod" in names

    def test_empty(self):
        assert list_profiles({}) == []


# =============================================================================
# Retry budget tests
# =============================================================================

class TestRetryBudgetConfig:
    def test_default_unlimited(self):
        cfg = RetryBudgetConfig()
        assert not cfg.is_configured()

    def test_configured(self):
        cfg = RetryBudgetConfig(max_retries_total=50)
        assert cfg.is_configured()


class TestRetryBudget:
    def test_unlimited_total_allows_many(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_per_item=1000))
        for i in range(100):
            assert budget.can_retry("item-1", f"metric-{i}")
            budget.record_retry("item-1", f"metric-{i}")

    def test_total_budget_enforced(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_total=3))
        for i in range(3):
            assert budget.can_retry(f"item-{i}", "m")
            budget.record_retry(f"item-{i}", "m")
        assert not budget.can_retry("item-99", "m")
        assert budget.budget_exhausted

    def test_per_item_limit(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_per_item=2))
        budget.record_retry("item-1", "m")
        budget.record_retry("item-1", "m")
        assert not budget.can_retry("item-1", "m")
        assert budget.can_retry("item-2", "m")  # different item OK

    def test_per_metric_limit(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_per_metric=2))
        budget.record_retry("i1", "metric-a")
        budget.record_retry("i2", "metric-a")
        assert not budget.can_retry("i3", "metric-a")
        assert budget.can_retry("i3", "metric-b")  # different metric OK

    def test_retries_remaining(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_total=10))
        budget.record_retry("i1", "m")
        budget.record_retry("i2", "m")
        assert budget.retries_remaining == 8

    def test_retries_remaining_unlimited(self):
        budget = RetryBudget()
        assert budget.retries_remaining is None

    def test_summary(self):
        budget = RetryBudget(RetryBudgetConfig(max_retries_total=100))
        budget.record_retry("i1", "m1")
        budget.record_retry("i1", "m2")
        budget.record_retry("i2", "m1")
        s = budget.summary()
        assert s["total_retries"] == 3
        assert s["items_retried"] == 2
        assert s["metrics_retried"] == 2
        assert s["max_retries_total"] == 100
        assert s["retries_remaining"] == 97
        assert len(s["top_retried_items"]) == 2

    def test_summary_no_limits(self):
        budget = RetryBudget()
        s = budget.summary()
        assert "max_retries_total" not in s

    def test_total_retries_property(self):
        budget = RetryBudget()
        assert budget.total_retries == 0
        budget.record_retry("i1", "m1")
        assert budget.total_retries == 1
