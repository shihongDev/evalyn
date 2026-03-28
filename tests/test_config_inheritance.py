"""Tests for config_inheritance module."""
from __future__ import annotations

import json
import os
import tempfile

from evalyn_sdk.config_inheritance import (
    ConfigLayer,
    MergedConfig,
    deep_merge,
    build_config,
    load_config_layer,
)


# --- deep_merge ---

class TestDeepMerge:
    def test_scalar_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_list_replacement(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_both_empty(self):
        result = deep_merge({}, {})
        assert result == {}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": 1}}
        override = {"a": "flat"}
        result = deep_merge(base, override)
        assert result == {"a": "flat"}

    def test_override_scalar_with_dict(self):
        base = {"a": "flat"}
        override = {"a": {"nested": 1}}
        result = deep_merge(base, override)
        assert result == {"a": {"nested": 1}}

    def test_does_not_mutate_base(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 99}}
        deep_merge(base, override)
        assert base == {"a": 1, "b": {"c": 2}}


# --- MergedConfig ---

class TestMergedConfig:
    def test_get_from_top_layer(self):
        layers = [
            ConfigLayer("base", {"key": "base_val"}),
            ConfigLayer("project", {"key": "project_val"}),
        ]
        cfg = MergedConfig(layers=layers)
        assert cfg.get("key") == "project_val"

    def test_get_fallback_to_lower_layer(self):
        layers = [
            ConfigLayer("base", {"only_in_base": 42}),
            ConfigLayer("project", {"other": "x"}),
        ]
        cfg = MergedConfig(layers=layers)
        assert cfg.get("only_in_base") == 42

    def test_get_default(self):
        cfg = MergedConfig(layers=[ConfigLayer("empty", {})])
        assert cfg.get("missing", "fallback") == "fallback"

    def test_get_default_none(self):
        cfg = MergedConfig(layers=[])
        assert cfg.get("anything") is None

    def test_get_nested_dotted_key(self):
        layers = [
            ConfigLayer("base", {"db": {"host": "localhost", "port": 5432}}),
            ConfigLayer("project", {"db": {"host": "prod.example.com"}}),
        ]
        cfg = MergedConfig(layers=layers)
        assert cfg.get_nested("db.host") == "prod.example.com"
        assert cfg.get_nested("db.port") == 5432

    def test_get_nested_missing_key(self):
        layers = [ConfigLayer("base", {"a": {"b": 1}})]
        cfg = MergedConfig(layers=layers)
        assert cfg.get_nested("a.b.c", "default") == "default"
        assert cfg.get_nested("x.y.z") is None

    def test_to_flat_dict(self):
        layers = [
            ConfigLayer("base", {"a": 1, "b": {"c": 2, "d": 3}}),
            ConfigLayer("project", {"b": {"c": 99}, "e": 5}),
        ]
        cfg = MergedConfig(layers=layers)
        flat = cfg.to_flat_dict()
        assert flat == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}

    def test_to_flat_dict_empty(self):
        cfg = MergedConfig(layers=[])
        assert cfg.to_flat_dict() == {}


# --- build_config ---

class TestBuildConfig:
    def test_layer_priority(self):
        layers = [
            ConfigLayer("default", {"timeout": 30}),
            ConfigLayer("user", {"timeout": 60}),
            ConfigLayer("project", {"timeout": 10}),
        ]
        cfg = build_config(layers)
        assert cfg.get("timeout") == 10  # last layer wins

    def test_empty_layers(self):
        cfg = build_config([])
        assert cfg.to_flat_dict() == {}

    def test_single_layer(self):
        cfg = build_config([ConfigLayer("only", {"k": "v"})])
        assert cfg.get("k") == "v"


# --- load_config_layer ---

class TestLoadConfigLayer:
    def test_load_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"model": "gpt-4", "temperature": 0.7}, f)
            f.flush()
            path = f.name
        try:
            layer = load_config_layer(path, "test")
            assert layer.name == "test"
            assert layer.values["model"] == "gpt-4"
            assert layer.values["temperature"] == 0.7
            assert layer.source == path
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        layer = load_config_layer("/nonexistent/path/config.json", "missing")
        assert layer.name == "missing"
        assert layer.values == {}

    def test_yaml_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("key: value\n")
            f.flush()
            path = f.name
        try:
            layer = load_config_layer(path)
            assert layer.values == {}
            assert layer.source == path
        finally:
            os.unlink(path)

    def test_default_name_from_stem(self):
        layer = load_config_layer("/some/path/myconfig.json", "")
        # File does not exist, so empty layer, but name should be stem
        assert layer.name == "myconfig"


# --- ConfigLayer roundtrip ---

class TestConfigLayerRoundtrip:
    def test_as_dict_from_dict(self):
        original = ConfigLayer("base", {"a": 1, "b": {"c": 2}}, "/path/to/file")
        d = original.as_dict()
        restored = ConfigLayer.from_dict(d)
        assert restored.name == original.name
        assert restored.values == original.values
        assert restored.source == original.source

    def test_empty_values(self):
        original = ConfigLayer("empty", {}, "")
        d = original.as_dict()
        restored = ConfigLayer.from_dict(d)
        assert restored.values == {}
        assert restored.source == ""


# --- MergedConfig roundtrip ---

class TestMergedConfigRoundtrip:
    def test_as_dict_from_dict(self):
        layers = [
            ConfigLayer("base", {"x": 1}, "base.json"),
            ConfigLayer("project", {"x": 2}, "project.json"),
        ]
        original = MergedConfig(layers=layers)
        d = original.as_dict()
        restored = MergedConfig.from_dict(d)
        assert len(restored.layers) == 2
        assert restored.layers[0].name == "base"
        assert restored.layers[1].values == {"x": 2}
        assert restored.get("x") == 2

    def test_empty_roundtrip(self):
        original = MergedConfig(layers=[])
        d = original.as_dict()
        restored = MergedConfig.from_dict(d)
        assert restored.layers == []
