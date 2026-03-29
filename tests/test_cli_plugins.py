"""Tests for the CLI plugin system."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.cli_plugins import (
    PluginInfo,
    PluginRegistry,
    create_default_registry,
    discover_entry_points,
    format_plugin_list,
)


# ---------------------------------------------------------------------------
# PluginInfo
# ---------------------------------------------------------------------------


class TestPluginInfo:
    def test_create(self):
        info = PluginInfo(name="my-cmd", module_path="my_pkg.cmd")
        assert info.name == "my-cmd"
        assert info.module_path == "my_pkg.cmd"
        assert info.description == ""
        assert info.version == ""
        assert info.enabled is True

    def test_create_full(self):
        info = PluginInfo(
            name="adv",
            module_path="adv.main",
            description="Advanced tools",
            version="1.2.0",
            enabled=False,
        )
        assert info.name == "adv"
        assert info.description == "Advanced tools"
        assert info.version == "1.2.0"
        assert info.enabled is False

    def test_as_dict(self):
        info = PluginInfo("x", "x.mod", "desc", "0.1", True)
        d = info.as_dict()
        assert d["name"] == "x"
        assert d["module_path"] == "x.mod"
        assert d["description"] == "desc"
        assert d["version"] == "0.1"
        assert d["enabled"] is True

    def test_from_dict(self):
        d = {
            "name": "foo",
            "module_path": "foo.bar",
            "description": "Foo plugin",
            "version": "2.0",
            "enabled": False,
        }
        info = PluginInfo.from_dict(d)
        assert info.name == "foo"
        assert info.module_path == "foo.bar"
        assert info.description == "Foo plugin"
        assert info.version == "2.0"
        assert info.enabled is False

    def test_from_dict_defaults(self):
        info = PluginInfo.from_dict({})
        assert info.name == ""
        assert info.module_path == ""
        assert info.description == ""
        assert info.version == ""
        assert info.enabled is True

    def test_roundtrip(self):
        original = PluginInfo("rt", "rt.mod", "round trip", "3.0", False)
        restored = PluginInfo.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.module_path == original.module_path
        assert restored.description == original.description
        assert restored.version == original.version
        assert restored.enabled == original.enabled


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_empty_registry(self):
        reg = PluginRegistry()
        assert reg.list_plugins() == []
        assert reg.list_enabled() == []
        assert reg.is_registered("anything") is False

    def test_register(self):
        reg = PluginRegistry()
        info = PluginInfo("cmd1", "cmd1.mod")
        reg.register(info)
        assert reg.is_registered("cmd1")
        assert reg.get("cmd1") is info

    def test_register_multiple(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("a", "a.mod"))
        reg.register(PluginInfo("b", "b.mod"))
        assert len(reg.list_plugins()) == 2

    def test_register_overwrite(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("dup", "first.mod", version="1.0"))
        reg.register(PluginInfo("dup", "second.mod", version="2.0"))
        assert reg.get("dup").module_path == "second.mod"
        assert len(reg.list_plugins()) == 1

    def test_unregister_existing(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("rm", "rm.mod"))
        assert reg.unregister("rm") is True
        assert reg.is_registered("rm") is False

    def test_unregister_nonexistent(self):
        reg = PluginRegistry()
        assert reg.unregister("ghost") is False

    def test_get_nonexistent(self):
        reg = PluginRegistry()
        assert reg.get("nope") is None

    def test_list_plugins(self):
        reg = PluginRegistry()
        p1 = PluginInfo("a", "a.mod")
        p2 = PluginInfo("b", "b.mod")
        reg.register(p1)
        reg.register(p2)
        plugins = reg.list_plugins()
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"a", "b"}

    def test_list_enabled(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("on", "on.mod", enabled=True))
        reg.register(PluginInfo("off", "off.mod", enabled=False))
        enabled = reg.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "on"

    def test_list_enabled_all_disabled(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("d1", "d1.mod", enabled=False))
        reg.register(PluginInfo("d2", "d2.mod", enabled=False))
        assert reg.list_enabled() == []

    def test_list_enabled_all_enabled(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("e1", "e1.mod", enabled=True))
        reg.register(PluginInfo("e2", "e2.mod", enabled=True))
        assert len(reg.list_enabled()) == 2

    def test_is_registered(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("check", "check.mod"))
        assert reg.is_registered("check") is True
        assert reg.is_registered("other") is False

    def test_register_then_unregister_then_check(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("tmp", "tmp.mod"))
        reg.unregister("tmp")
        assert reg.is_registered("tmp") is False
        assert reg.get("tmp") is None
        assert reg.list_plugins() == []

    def test_register_after_unregister(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("re", "first.mod"))
        reg.unregister("re")
        reg.register(PluginInfo("re", "second.mod"))
        assert reg.get("re").module_path == "second.mod"

    def test_list_plugins_returns_copy(self):
        reg = PluginRegistry()
        reg.register(PluginInfo("x", "x.mod"))
        lst = reg.list_plugins()
        lst.clear()
        assert len(reg.list_plugins()) == 1


# ---------------------------------------------------------------------------
# discover_entry_points
# ---------------------------------------------------------------------------


class TestDiscoverEntryPoints:
    def test_no_entry_points(self):
        # Default group unlikely to have real entries in test env
        # Should return empty list without raising
        result = discover_entry_points("evalyn.nonexistent_group_for_testing")
        assert result == []

    def test_handles_import_error(self):
        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", side_effect=ImportError):
            result = discover_entry_points()
            assert result == []

    def test_handles_generic_exception(self):
        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", side_effect=RuntimeError):
            result = discover_entry_points()
            assert result == []

    def test_with_mock_entry_points_select(self):
        mock_ep = MagicMock()
        mock_ep.name = "test-cmd"
        mock_ep.value = "test_pkg.commands:register"
        mock_ep.dist = MagicMock()
        mock_ep.dist.version = "1.0.0"

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points("evalyn.commands")
            assert len(result) == 1
            assert result[0].name == "test-cmd"
            assert result[0].module_path == "test_pkg.commands:register"
            assert result[0].version == "1.0.0"
            assert result[0].enabled is True

    def test_with_mock_entry_points_dict(self):
        mock_ep = MagicMock()
        mock_ep.name = "dict-cmd"
        mock_ep.value = "dict_pkg.cmd:main"
        mock_ep.dist = None

        mock_eps = {"evalyn.commands": [mock_ep]}
        # Remove select attribute to force dict path
        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points("evalyn.commands")
            assert len(result) == 1
            assert result[0].name == "dict-cmd"
            assert result[0].version == ""

    def test_with_mock_entry_points_list(self):
        mock_ep = MagicMock(spec=[])  # No select, not a dict
        mock_ep.name = "list-cmd"
        mock_ep.value = "list_pkg.cmd:run"
        mock_ep.group = "evalyn.commands"
        mock_ep.dist = None

        # Use a plain list (no select method, not a dict)
        mock_eps = [mock_ep]

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points("evalyn.commands")
            assert len(result) == 1
            assert result[0].name == "list-cmd"

    def test_filters_by_group_in_list_mode(self):
        ep1 = MagicMock(spec=[])
        ep1.name = "match"
        ep1.value = "a.b:c"
        ep1.group = "evalyn.commands"
        ep1.dist = None

        ep2 = MagicMock(spec=[])
        ep2.name = "other"
        ep2.value = "x.y:z"
        ep2.group = "other.group"
        ep2.dist = None

        mock_eps = [ep1, ep2]

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points("evalyn.commands")
            assert len(result) == 1
            assert result[0].name == "match"

    def test_version_extraction_failure(self):

        class BrokenDist:
            @property
            def version(self):
                raise AttributeError("no version")

        mock_ep = MagicMock()
        mock_ep.name = "broken-ver"
        mock_ep.value = "pkg:fn"
        mock_ep.dist = BrokenDist()

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points()
            assert len(result) == 1
            assert result[0].version == ""

    def test_multiple_entry_points(self):
        eps = []
        for i in range(3):
            ep = MagicMock()
            ep.name = f"cmd-{i}"
            ep.value = f"pkg{i}.mod:register"
            ep.dist = MagicMock()
            ep.dist.version = f"{i}.0.0"
            eps.append(ep)

        mock_eps = MagicMock()
        mock_eps.select.return_value = eps

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_entry_points()
            assert len(result) == 3
            names = [p.name for p in result]
            assert names == ["cmd-0", "cmd-1", "cmd-2"]


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_returns_registry(self):
        reg = create_default_registry()
        assert isinstance(reg, PluginRegistry)

    def test_populates_from_discovery(self):
        mock_ep = MagicMock()
        mock_ep.name = "auto-cmd"
        mock_ep.value = "auto.mod:fn"
        mock_ep.dist = MagicMock()
        mock_ep.dist.version = "0.1.0"

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            reg = create_default_registry()
            assert reg.is_registered("auto-cmd")
            info = reg.get("auto-cmd")
            assert info.module_path == "auto.mod:fn"

    def test_empty_when_no_plugins(self):
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch("evalyn_sdk.cli_plugins.importlib.metadata.entry_points", return_value=mock_eps):
            reg = create_default_registry()
            assert reg.list_plugins() == []


# ---------------------------------------------------------------------------
# format_plugin_list
# ---------------------------------------------------------------------------


class TestFormatPluginList:
    def test_empty_list(self):
        result = format_plugin_list([])
        assert "No plugins registered" in result

    def test_single_plugin(self):
        plugins = [PluginInfo("hello", "hello.mod", "A greeting plugin", "1.0.0", True)]
        result = format_plugin_list(plugins)
        assert "hello" in result
        assert "1.0.0" in result
        assert "Enabled" in result
        assert "A greeting plugin" in result

    def test_disabled_plugin(self):
        plugins = [PluginInfo("off", "off.mod", "Disabled one", "0.1", False)]
        result = format_plugin_list(plugins)
        assert "Disabled" in result

    def test_multiple_plugins(self):
        plugins = [
            PluginInfo("alpha", "alpha.mod", "First", "1.0", True),
            PluginInfo("beta", "beta.mod", "Second", "2.0", False),
            PluginInfo("gamma", "gamma.mod", "Third", "3.0", True),
        ]
        result = format_plugin_list(plugins)
        lines = result.strip().split("\n")
        # Header + separator + 3 plugins = 5 lines
        assert len(lines) == 5
        assert "alpha" in lines[2]
        assert "beta" in lines[3]
        assert "gamma" in lines[4]

    def test_header_present(self):
        plugins = [PluginInfo("x", "x.mod")]
        result = format_plugin_list(plugins)
        assert "Name" in result
        assert "Version" in result
        assert "Status" in result
        assert "Description" in result

    def test_separator_line(self):
        plugins = [PluginInfo("x", "x.mod")]
        result = format_plugin_list(plugins)
        lines = result.strip().split("\n")
        # Second line should be a separator
        assert set(lines[1].strip()) == {"-"}

    def test_alignment_with_long_name(self):
        plugins = [
            PluginInfo("short", "s.mod", "S", "1.0", True),
            PluginInfo("a-very-long-plugin-name", "l.mod", "L", "2.0", True),
        ]
        result = format_plugin_list(plugins)
        lines = result.strip().split("\n")
        # Both data lines should have the same column structure
        assert len(lines) == 4  # header + sep + 2 plugins

    def test_empty_version(self):
        plugins = [PluginInfo("noversion", "nv.mod", "No version")]
        result = format_plugin_list(plugins)
        assert "noversion" in result
        assert "Enabled" in result

    def test_empty_description(self):
        plugins = [PluginInfo("nodesc", "nd.mod")]
        result = format_plugin_list(plugins)
        assert "nodesc" in result
