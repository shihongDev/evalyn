"""Tests for plugin_system module."""

from __future__ import annotations

import pytest

from evalyn_sdk.plugin_system import (
    PluginInfo,
    PluginRegistry,
    create_plugin_info,
    discover_plugins,
    get_global_registry,
    reset_global_registry,
    validate_plugin,
)


# --- PluginInfo ---


def test_plugin_info_defaults():
    info = PluginInfo(name="test")
    assert info.version == "0.0.0"
    assert info.plugin_type == "metric"
    assert info.description == ""
    assert info.author == ""
    assert info.entry_point == ""


def test_plugin_info_as_dict():
    info = PluginInfo(name="foo", version="1.2.3", plugin_type="storage")
    d = info.as_dict()
    assert d["name"] == "foo"
    assert d["version"] == "1.2.3"
    assert d["plugin_type"] == "storage"


def test_plugin_info_from_dict():
    d = {"name": "bar", "version": "2.0.0", "plugin_type": "analyzer", "author": "me"}
    info = PluginInfo.from_dict(d)
    assert info.name == "bar"
    assert info.version == "2.0.0"
    assert info.plugin_type == "analyzer"
    assert info.author == "me"


def test_plugin_info_roundtrip():
    original = PluginInfo(
        name="roundtrip",
        version="3.1.4",
        plugin_type="instrumentor",
        description="A test plugin",
        author="tester",
        entry_point="mod:func",
    )
    restored = PluginInfo.from_dict(original.as_dict())
    assert restored.name == original.name
    assert restored.version == original.version
    assert restored.plugin_type == original.plugin_type
    assert restored.description == original.description
    assert restored.author == original.author
    assert restored.entry_point == original.entry_point


def test_plugin_info_from_dict_defaults():
    info = PluginInfo.from_dict({})
    assert info.name == ""
    assert info.version == "0.0.0"
    assert info.plugin_type == "metric"


# --- PluginRegistry ---


def test_register_plugin():
    reg = PluginRegistry()
    info = PluginInfo(name="p1", plugin_type="metric")
    reg.register(info)
    assert reg.is_registered("p1")


def test_register_duplicate_raises():
    reg = PluginRegistry()
    info = PluginInfo(name="dup")
    reg.register(info)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(PluginInfo(name="dup"))


def test_unregister_removes():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="rem"))
    assert reg.unregister("rem") is True
    assert not reg.is_registered("rem")


def test_unregister_missing_returns_false():
    reg = PluginRegistry()
    assert reg.unregister("nonexistent") is False


def test_get_existing():
    reg = PluginRegistry()
    info = PluginInfo(name="found", version="1.0.0")
    reg.register(info)
    result = reg.get("found")
    assert result is not None
    assert result.version == "1.0.0"


def test_get_missing():
    reg = PluginRegistry()
    assert reg.get("missing") is None


def test_list_all():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="a"))
    reg.register(PluginInfo(name="b"))
    reg.register(PluginInfo(name="c"))
    assert len(reg.list_all()) == 3


def test_list_by_type():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="m1", plugin_type="metric"))
    reg.register(PluginInfo(name="m2", plugin_type="metric"))
    reg.register(PluginInfo(name="s1", plugin_type="storage"))
    assert len(reg.list_by_type("metric")) == 2
    assert len(reg.list_by_type("storage")) == 1
    assert len(reg.list_by_type("analyzer")) == 0


def test_is_registered_true():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="exists"))
    assert reg.is_registered("exists") is True


def test_is_registered_false():
    reg = PluginRegistry()
    assert reg.is_registered("nope") is False


def test_count_property():
    reg = PluginRegistry()
    assert reg.count == 0
    reg.register(PluginInfo(name="x"))
    assert reg.count == 1
    reg.register(PluginInfo(name="y"))
    assert reg.count == 2


def test_empty_registry():
    reg = PluginRegistry()
    assert reg.count == 0
    assert reg.list_all() == []
    assert reg.list_by_type("metric") == []
    assert reg.get("anything") is None


def test_registry_as_dict():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="p1", version="1.0.0"))
    d = reg.as_dict()
    assert "plugins" in d
    assert "p1" in d["plugins"]
    assert d["plugins"]["p1"]["version"] == "1.0.0"


def test_registry_from_dict():
    d = {
        "plugins": {
            "alpha": {"name": "alpha", "version": "2.0.0", "plugin_type": "storage"},
        }
    }
    reg = PluginRegistry.from_dict(d)
    assert reg.is_registered("alpha")
    assert reg.get("alpha").plugin_type == "storage"


def test_registry_roundtrip():
    reg = PluginRegistry()
    reg.register(PluginInfo(name="r1", plugin_type="metric", version="1.0.0"))
    reg.register(PluginInfo(name="r2", plugin_type="storage", version="2.0.0"))
    restored = PluginRegistry.from_dict(reg.as_dict())
    assert restored.count == 2
    assert restored.get("r1").plugin_type == "metric"
    assert restored.get("r2").version == "2.0.0"


def test_registry_from_dict_empty():
    reg = PluginRegistry.from_dict({})
    assert reg.count == 0


# --- Global registry ---


def test_global_registry_singleton():
    reset_global_registry()
    r1 = get_global_registry()
    r2 = get_global_registry()
    assert r1 is r2


def test_reset_global_registry():
    reset_global_registry()
    reg = get_global_registry()
    reg.register(PluginInfo(name="temp"))
    assert reg.count == 1
    reset_global_registry()
    reg2 = get_global_registry()
    assert reg2.count == 0
    assert reg2 is not reg


# --- discover_plugins ---


def test_discover_plugins_returns_list():
    result = discover_plugins()
    assert isinstance(result, list)


def test_discover_plugins_nonexistent_group():
    result = discover_plugins(entry_point_group="evalyn.nonexistent.group.xyz")
    assert result == []


# --- validate_plugin ---


def test_validate_plugin_valid():
    info = PluginInfo(name="valid", version="1.0.0", plugin_type="metric")
    is_valid, errors = validate_plugin(info)
    assert is_valid is True
    assert errors == []


def test_validate_plugin_empty_name():
    info = PluginInfo(name="", version="1.0.0", plugin_type="metric")
    is_valid, errors = validate_plugin(info)
    assert is_valid is False
    assert any("name" in e for e in errors)


def test_validate_plugin_whitespace_name():
    info = PluginInfo(name="   ", version="1.0.0", plugin_type="metric")
    is_valid, errors = validate_plugin(info)
    assert is_valid is False


def test_validate_plugin_bad_type():
    info = PluginInfo(name="ok", version="1.0.0", plugin_type="unknown")
    is_valid, errors = validate_plugin(info)
    assert is_valid is False
    assert any("plugin_type" in e for e in errors)


def test_validate_plugin_bad_version():
    info = PluginInfo(name="ok", version="abc", plugin_type="metric")
    is_valid, errors = validate_plugin(info)
    assert is_valid is False
    assert any("version" in e for e in errors)


def test_validate_plugin_multiple_errors():
    info = PluginInfo(name="", version="nope", plugin_type="bad")
    is_valid, errors = validate_plugin(info)
    assert is_valid is False
    assert len(errors) == 3


def test_validate_all_valid_types():
    for ptype in ["metric", "instrumentor", "storage", "analyzer"]:
        info = PluginInfo(name="t", version="0.0.0", plugin_type=ptype)
        is_valid, _ = validate_plugin(info)
        assert is_valid is True


# --- create_plugin_info ---


def test_create_plugin_info_basic():
    info = create_plugin_info("myplug", "storage")
    assert info.name == "myplug"
    assert info.plugin_type == "storage"
    assert info.version == "0.0.0"


def test_create_plugin_info_with_kwargs():
    info = create_plugin_info(
        "myplug", "analyzer", version="5.0.0", author="dev", description="desc"
    )
    assert info.version == "5.0.0"
    assert info.author == "dev"
    assert info.description == "desc"
