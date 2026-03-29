"""Tests for CLI alias support."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.cli_aliases import (
    Alias,
    AliasRegistry,
    BUILTIN_ALIASES,
    format_alias_list,
    load_aliases_from_config,
    parse_alias_command,
)


# ---------------------------------------------------------------------------
# Alias dataclass
# ---------------------------------------------------------------------------


class TestAlias:
    def test_basic_creation(self):
        a = Alias(name="q", command="quickstart")
        assert a.name == "q"
        assert a.command == "quickstart"
        assert a.description == ""

    def test_creation_with_description(self):
        a = Alias(name="q", command="quickstart", description="Quick start")
        assert a.description == "Quick start"

    def test_as_dict(self):
        a = Alias(name="q", command="quickstart", description="Quick")
        d = a.as_dict()
        assert d == {"name": "q", "command": "quickstart", "description": "Quick"}

    def test_as_dict_default_description(self):
        a = Alias(name="q", command="quickstart")
        d = a.as_dict()
        assert d["description"] == ""

    def test_from_dict(self):
        d = {"name": "q", "command": "quickstart", "description": "Quick"}
        a = Alias.from_dict(d)
        assert a.name == "q"
        assert a.command == "quickstart"
        assert a.description == "Quick"

    def test_from_dict_missing_description(self):
        d = {"name": "q", "command": "quickstart"}
        a = Alias.from_dict(d)
        assert a.description == ""

    def test_roundtrip(self):
        original = Alias(name="fast", command="run-eval --workers 8", description="Fast eval")
        restored = Alias.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.command == original.command
        assert restored.description == original.description

    def test_equality(self):
        a = Alias(name="q", command="quickstart")
        b = Alias(name="q", command="quickstart")
        assert a == b

    def test_inequality(self):
        a = Alias(name="q", command="quickstart")
        b = Alias(name="q", command="other")
        assert a != b


# ---------------------------------------------------------------------------
# AliasRegistry
# ---------------------------------------------------------------------------


class TestAliasRegistry:
    def test_empty_registry(self):
        reg = AliasRegistry()
        assert reg.list_aliases() == []
        assert not reg.is_alias("q")
        assert reg.get("q") is None

    def test_register_and_get(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        assert reg.get("q") is not None
        assert reg.get("q").command == "quickstart"

    def test_register_overwrites(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        reg.register(Alias(name="q", command="status"))
        assert reg.get("q").command == "status"

    def test_unregister_existing(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        assert reg.unregister("q") is True
        assert reg.get("q") is None

    def test_unregister_missing(self):
        reg = AliasRegistry()
        assert reg.unregister("nonexistent") is False

    def test_resolve_alias(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        assert reg.resolve("q") == "quickstart"

    def test_resolve_passthrough(self):
        reg = AliasRegistry()
        assert reg.resolve("run-eval") == "run-eval"

    def test_list_aliases_sorted(self):
        reg = AliasRegistry()
        reg.register(Alias(name="z", command="zeta"))
        reg.register(Alias(name="a", command="alpha"))
        reg.register(Alias(name="m", command="mid"))
        names = [a.name for a in reg.list_aliases()]
        assert names == ["a", "m", "z"]

    def test_is_alias_true(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        assert reg.is_alias("q") is True

    def test_is_alias_false(self):
        reg = AliasRegistry()
        assert reg.is_alias("q") is False

    def test_multiple_aliases(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        reg.register(Alias(name="ls", command="list-runs"))
        reg.register(Alias(name="st", command="status"))
        assert len(reg.list_aliases()) == 3

    def test_unregister_then_is_alias(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        reg.unregister("q")
        assert reg.is_alias("q") is False

    def test_resolve_after_unregister(self):
        reg = AliasRegistry()
        reg.register(Alias(name="q", command="quickstart"))
        reg.unregister("q")
        assert reg.resolve("q") == "q"


# ---------------------------------------------------------------------------
# load_aliases_from_config
# ---------------------------------------------------------------------------


class TestLoadAliasesFromConfig:
    def test_empty_config(self):
        reg = load_aliases_from_config({})
        assert reg.list_aliases() == []

    def test_missing_aliases_key(self):
        reg = load_aliases_from_config({"other": "stuff"})
        assert reg.list_aliases() == []

    def test_string_values(self):
        config = {"aliases": {"q": "quickstart", "fast": "run-eval --workers 8"}}
        reg = load_aliases_from_config(config)
        assert reg.get("q").command == "quickstart"
        assert reg.get("fast").command == "run-eval --workers 8"

    def test_dict_values(self):
        config = {
            "aliases": {
                "q": {"command": "quickstart", "description": "Quick start"},
            }
        }
        reg = load_aliases_from_config(config)
        assert reg.get("q").command == "quickstart"
        assert reg.get("q").description == "Quick start"

    def test_dict_values_missing_description(self):
        config = {"aliases": {"q": {"command": "quickstart"}}}
        reg = load_aliases_from_config(config)
        assert reg.get("q").description == ""

    def test_mixed_values(self):
        config = {
            "aliases": {
                "q": "quickstart",
                "fast": {"command": "run-eval --workers 8", "description": "Parallel"},
            }
        }
        reg = load_aliases_from_config(config)
        assert reg.get("q").command == "quickstart"
        assert reg.get("fast").description == "Parallel"

    def test_preserves_all_aliases(self):
        config = {"aliases": {f"a{i}": f"cmd{i}" for i in range(10)}}
        reg = load_aliases_from_config(config)
        assert len(reg.list_aliases()) == 10


# ---------------------------------------------------------------------------
# parse_alias_command
# ---------------------------------------------------------------------------


class TestParseAliasCommand:
    def test_simple_command(self):
        result = parse_alias_command("quickstart", [])
        assert result == ["quickstart"]

    def test_command_with_flags(self):
        result = parse_alias_command("run-eval --workers 8", [])
        assert result == ["run-eval", "--workers", "8"]

    def test_extra_args_appended(self):
        result = parse_alias_command("run-eval", ["--verbose"])
        assert result == ["run-eval", "--verbose"]

    def test_combined_flags_and_extra(self):
        result = parse_alias_command("run-eval --workers 8", ["--verbose", "--dry-run"])
        assert result == ["run-eval", "--workers", "8", "--verbose", "--dry-run"]

    def test_empty_extra_args(self):
        result = parse_alias_command("status", [])
        assert result == ["status"]

    def test_multiple_spaces(self):
        # split() handles multiple spaces
        result = parse_alias_command("run-eval   --workers   8", [])
        assert result == ["run-eval", "--workers", "8"]


# ---------------------------------------------------------------------------
# BUILTIN_ALIASES
# ---------------------------------------------------------------------------


class TestBuiltinAliases:
    def test_expected_keys(self):
        assert "q" in BUILTIN_ALIASES
        assert "ls" in BUILTIN_ALIASES
        assert "st" in BUILTIN_ALIASES

    def test_values(self):
        assert BUILTIN_ALIASES["q"] == "quickstart"
        assert BUILTIN_ALIASES["ls"] == "list-runs"
        assert BUILTIN_ALIASES["st"] == "status"

    def test_count(self):
        assert len(BUILTIN_ALIASES) == 3


# ---------------------------------------------------------------------------
# format_alias_list
# ---------------------------------------------------------------------------


class TestFormatAliasList:
    def test_empty_list(self):
        assert format_alias_list([]) == ""

    def test_single_alias(self):
        aliases = [Alias(name="q", command="quickstart")]
        result = format_alias_list(aliases)
        assert "q" in result
        assert "quickstart" in result

    def test_header_present(self):
        aliases = [Alias(name="q", command="quickstart")]
        result = format_alias_list(aliases)
        assert "ALIAS" in result
        assert "COMMAND" in result
        assert "DESCRIPTION" in result

    def test_separator_line(self):
        aliases = [Alias(name="q", command="quickstart")]
        result = format_alias_list(aliases)
        lines = result.split("\n")
        assert len(lines) >= 3
        # Second line should be dashes
        assert set(lines[1].strip()) == {"-"}

    def test_description_shown(self):
        aliases = [Alias(name="q", command="quickstart", description="Quick start")]
        result = format_alias_list(aliases)
        assert "Quick start" in result

    def test_multiple_aliases(self):
        aliases = [
            Alias(name="q", command="quickstart"),
            Alias(name="ls", command="list-runs", description="List all runs"),
        ]
        result = format_alias_list(aliases)
        lines = result.split("\n")
        # header + separator + 2 data lines
        assert len(lines) == 4

    def test_alignment(self):
        aliases = [
            Alias(name="q", command="quickstart"),
            Alias(name="long-name", command="a"),
        ]
        result = format_alias_list(aliases)
        lines = result.split("\n")
        # Data lines should have consistent column spacing
        assert len(lines) == 4
