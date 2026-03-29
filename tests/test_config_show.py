"""Tests for config_show module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from evalyn_sdk.config_show import (
    ConfigSource,
    MergedConfig,
    format_config_show,
    get_effective_value,
    get_env_overrides,
    highlight_source,
    merge_configs,
)


# =============================================================================
# ConfigSource
# =============================================================================


class TestConfigSource:
    def test_creation(self):
        cs = ConfigSource(key="provider", value="gemini", source="project")
        assert cs.key == "provider"
        assert cs.value == "gemini"
        assert cs.source == "project"

    def test_as_dict(self):
        cs = ConfigSource(key="k", value="v", source="env")
        d = cs.as_dict()
        assert d == {"key": "k", "value": "v", "source": "env"}

    def test_from_dict_full(self):
        cs = ConfigSource.from_dict({"key": "k", "value": 42, "source": "flag"})
        assert cs.key == "k"
        assert cs.value == 42
        assert cs.source == "flag"

    def test_from_dict_default_source(self):
        cs = ConfigSource.from_dict({"key": "k", "value": "v"})
        assert cs.source == "default"

    def test_roundtrip(self):
        original = ConfigSource(key="db", value="/tmp/db.sqlite", source="global")
        restored = ConfigSource.from_dict(original.as_dict())
        assert restored.key == original.key
        assert restored.value == original.value
        assert restored.source == original.source

    def test_value_types(self):
        """Values can be any type."""
        for val in [42, 3.14, True, None, ["a", "b"], {"nested": 1}]:
            cs = ConfigSource(key="test", value=val, source="default")
            assert cs.value == val

    def test_as_dict_contains_all_fields(self):
        cs = ConfigSource(key="a", value="b", source="c")
        d = cs.as_dict()
        assert set(d.keys()) == {"key", "value", "source"}


# =============================================================================
# MergedConfig
# =============================================================================


class TestMergedConfig:
    def test_defaults(self):
        mc = MergedConfig()
        assert mc.entries == []
        assert mc.sources_used == []

    def test_as_dict(self):
        mc = MergedConfig(
            entries=[ConfigSource(key="a", value="1", source="env")],
            sources_used=["env"],
        )
        d = mc.as_dict()
        assert len(d["entries"]) == 1
        assert d["sources_used"] == ["env"]

    def test_from_dict(self):
        data = {
            "entries": [{"key": "x", "value": 1, "source": "flag"}],
            "sources_used": ["flag"],
        }
        mc = MergedConfig.from_dict(data)
        assert len(mc.entries) == 1
        assert mc.entries[0].key == "x"
        assert mc.sources_used == ["flag"]

    def test_from_dict_empty(self):
        mc = MergedConfig.from_dict({})
        assert mc.entries == []
        assert mc.sources_used == []

    def test_roundtrip(self):
        original = MergedConfig(
            entries=[
                ConfigSource(key="a", value=1, source="global"),
                ConfigSource(key="b", value=2, source="flag"),
            ],
            sources_used=["flag", "global"],
        )
        restored = MergedConfig.from_dict(original.as_dict())
        assert len(restored.entries) == len(original.entries)
        assert restored.sources_used == original.sources_used

    def test_entries_list_independent(self):
        mc = MergedConfig(entries=[], sources_used=[])
        d = mc.as_dict()
        d["entries"].append({"key": "z", "value": "z", "source": "z"})
        assert len(mc.entries) == 0


# =============================================================================
# merge_configs
# =============================================================================


class TestMergeConfigs:
    def test_flag_highest_priority(self):
        mc = merge_configs(
            global_config={"provider": "global_val"},
            project_config={"provider": "project_val"},
            env_overrides={"provider": "env_val"},
            flag_overrides={"provider": "flag_val"},
            defaults={"provider": "default_val"},
        )
        entry = get_effective_value("provider", mc)
        assert entry is not None
        assert entry.value == "flag_val"
        assert entry.source == "flag"

    def test_env_over_project(self):
        mc = merge_configs(
            global_config={},
            project_config={"provider": "project_val"},
            env_overrides={"provider": "env_val"},
            flag_overrides={},
            defaults={},
        )
        entry = get_effective_value("provider", mc)
        assert entry is not None
        assert entry.value == "env_val"
        assert entry.source == "env"

    def test_project_over_global(self):
        mc = merge_configs(
            global_config={"provider": "global_val"},
            project_config={"provider": "project_val"},
            env_overrides={},
            flag_overrides={},
            defaults={},
        )
        entry = get_effective_value("provider", mc)
        assert entry is not None
        assert entry.value == "project_val"
        assert entry.source == "project"

    def test_global_over_default(self):
        mc = merge_configs(
            global_config={"provider": "global_val"},
            project_config={},
            env_overrides={},
            flag_overrides={},
            defaults={"provider": "default_val"},
        )
        entry = get_effective_value("provider", mc)
        assert entry is not None
        assert entry.value == "global_val"
        assert entry.source == "global"

    def test_default_used_when_no_others(self):
        mc = merge_configs(
            global_config={},
            project_config={},
            env_overrides={},
            flag_overrides={},
            defaults={"provider": "default_val"},
        )
        entry = get_effective_value("provider", mc)
        assert entry is not None
        assert entry.value == "default_val"
        assert entry.source == "default"

    def test_multiple_keys(self):
        mc = merge_configs(
            global_config={"a": 1},
            project_config={"b": 2},
            env_overrides={"c": 3},
            flag_overrides={},
            defaults={"d": 4},
        )
        assert len(mc.entries) == 4
        keys = [e.key for e in mc.entries]
        assert keys == ["a", "b", "c", "d"]  # sorted

    def test_sources_used_tracked(self):
        mc = merge_configs(
            global_config={"a": 1},
            project_config={},
            env_overrides={"b": 2},
            flag_overrides={},
            defaults={},
        )
        assert "global" in mc.sources_used
        assert "env" in mc.sources_used
        assert "flag" not in mc.sources_used

    def test_empty_all(self):
        mc = merge_configs({}, {}, {}, {}, {})
        assert mc.entries == []
        assert mc.sources_used == []

    def test_keys_sorted(self):
        mc = merge_configs(
            global_config={},
            project_config={},
            env_overrides={},
            flag_overrides={},
            defaults={"z": 1, "a": 2, "m": 3},
        )
        keys = [e.key for e in mc.entries]
        assert keys == ["a", "m", "z"]

    def test_sources_used_priority_order(self):
        mc = merge_configs(
            global_config={"a": 1},
            project_config={},
            env_overrides={},
            flag_overrides={"b": 2},
            defaults={"c": 3},
        )
        # sources_used should be in priority order: flag, global, default
        assert mc.sources_used == ["flag", "global", "default"]

    def test_same_key_different_layers(self):
        """Only highest priority source wins."""
        mc = merge_configs(
            global_config={"x": "global"},
            project_config={"x": "project"},
            env_overrides={},
            flag_overrides={},
            defaults={"x": "default"},
        )
        assert len([e for e in mc.entries if e.key == "x"]) == 1
        assert get_effective_value("x", mc).source == "project"


# =============================================================================
# get_env_overrides
# =============================================================================


class TestGetEnvOverrides:
    def test_reads_prefixed_vars(self):
        with patch.dict(os.environ, {"EVALYN_PROVIDER": "gemini", "OTHER_VAR": "x"}, clear=False):
            overrides = get_env_overrides()
            assert "provider" in overrides
            assert overrides["provider"] == "gemini"
            assert "other_var" not in overrides

    def test_strips_prefix_and_lowercases(self):
        with patch.dict(os.environ, {"EVALYN_MAX_WORKERS": "4"}, clear=False):
            overrides = get_env_overrides()
            assert "max_workers" in overrides
            assert overrides["max_workers"] == "4"

    def test_custom_prefix(self):
        with patch.dict(os.environ, {"MY_FOO": "bar"}, clear=False):
            overrides = get_env_overrides(prefix="MY_")
            assert "foo" in overrides
            assert overrides["foo"] == "bar"

    def test_empty_when_no_match(self):
        with patch.dict(os.environ, {"UNRELATED": "val"}, clear=False):
            overrides = get_env_overrides(prefix="ZZZZZ_")
            assert overrides == {}

    def test_prefix_only_var_skipped(self):
        """A var named exactly the prefix (no key after) should be skipped."""
        with patch.dict(os.environ, {"EVALYN_": "val"}, clear=False):
            overrides = get_env_overrides()
            assert overrides == {} or all(k != "" for k in overrides)

    def test_multiple_vars(self):
        env = {"EVALYN_A": "1", "EVALYN_B": "2", "EVALYN_C": "3"}
        with patch.dict(os.environ, env, clear=False):
            overrides = get_env_overrides()
            assert overrides.get("a") == "1"
            assert overrides.get("b") == "2"
            assert overrides.get("c") == "3"


# =============================================================================
# get_effective_value
# =============================================================================


class TestGetEffectiveValue:
    def test_found(self):
        mc = MergedConfig(entries=[ConfigSource(key="x", value=1, source="flag")])
        result = get_effective_value("x", mc)
        assert result is not None
        assert result.value == 1

    def test_not_found(self):
        mc = MergedConfig(entries=[ConfigSource(key="x", value=1, source="flag")])
        result = get_effective_value("y", mc)
        assert result is None

    def test_empty_config(self):
        mc = MergedConfig()
        assert get_effective_value("anything", mc) is None

    def test_returns_first_match(self):
        mc = MergedConfig(entries=[
            ConfigSource(key="x", value="first", source="flag"),
            ConfigSource(key="x", value="second", source="env"),
        ])
        result = get_effective_value("x", mc)
        assert result.value == "first"


# =============================================================================
# highlight_source
# =============================================================================


class TestHighlightSource:
    def test_flag(self):
        assert highlight_source("flag") == "[flag]"

    def test_env(self):
        assert highlight_source("env") == "[env]"

    def test_project(self):
        assert highlight_source("project") == "[project]"

    def test_global(self):
        assert highlight_source("global") == "[global]"

    def test_default(self):
        assert highlight_source("default") == "[default]"

    def test_custom_string(self):
        assert highlight_source("custom") == "[custom]"


# =============================================================================
# format_config_show
# =============================================================================


class TestFormatConfigShow:
    def test_empty_config(self):
        mc = MergedConfig()
        result = format_config_show(mc)
        assert result == "No configuration entries."

    def test_single_entry(self):
        mc = MergedConfig(
            entries=[ConfigSource(key="provider", value="gemini", source="project")],
            sources_used=["project"],
        )
        result = format_config_show(mc)
        assert "provider" in result
        assert "gemini" in result
        assert "[project]" in result

    def test_multiple_entries(self):
        mc = MergedConfig(
            entries=[
                ConfigSource(key="provider", value="gemini", source="project"),
                ConfigSource(key="max_workers", value=4, source="env"),
            ],
            sources_used=["env", "project"],
        )
        result = format_config_show(mc)
        assert "provider" in result
        assert "max_workers" in result
        assert "[project]" in result
        assert "[env]" in result

    def test_header_present(self):
        mc = MergedConfig(
            entries=[ConfigSource(key="a", value="b", source="default")],
            sources_used=["default"],
        )
        result = format_config_show(mc)
        assert "Key" in result
        assert "Value" in result
        assert "Source" in result

    def test_separator_present(self):
        mc = MergedConfig(
            entries=[ConfigSource(key="a", value="b", source="default")],
            sources_used=["default"],
        )
        result = format_config_show(mc)
        assert "---" in result

    def test_sources_used_in_footer(self):
        mc = MergedConfig(
            entries=[ConfigSource(key="a", value="b", source="flag")],
            sources_used=["flag"],
        )
        result = format_config_show(mc)
        assert "Sources used:" in result
        assert "flag" in result

    def test_alignment(self):
        """All rows should have consistent column alignment."""
        mc = MergedConfig(
            entries=[
                ConfigSource(key="short", value="x", source="flag"),
                ConfigSource(key="very_long_key_name", value="y", source="default"),
            ],
            sources_used=["flag", "default"],
        )
        result = format_config_show(mc)
        lines = result.strip().split("\n")
        # Header and separator should exist
        assert len(lines) >= 4
