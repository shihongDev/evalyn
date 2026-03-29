"""Tests for the CLI color theme configuration module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.color_theme import (
    ANSI_COLORS,
    ANSI_RESET,
    DEFAULT_THEME,
    DARK_THEME,
    LIGHT_THEME,
    NO_COLOR_THEME,
    ColorDef,
    Theme,
    colorize,
    format_theme_preview,
    get_active_theme,
    load_theme_from_config,
    strip_ansi,
)


# ---------------------------------------------------------------------------
# ColorDef tests
# ---------------------------------------------------------------------------


class TestColorDef:
    def test_creation(self):
        cd = ColorDef(name="red", ansi_code="\033[31m", description="Red text")
        assert cd.name == "red"
        assert cd.ansi_code == "\033[31m"
        assert cd.description == "Red text"

    def test_default_description(self):
        cd = ColorDef(name="blue", ansi_code="\033[34m")
        assert cd.description == ""

    def test_as_dict(self):
        cd = ColorDef(name="green", ansi_code="\033[32m", description="Green")
        d = cd.as_dict()
        assert d == {"name": "green", "ansi_code": "\033[32m", "description": "Green"}

    def test_from_dict(self):
        d = {"name": "cyan", "ansi_code": "\033[36m", "description": "Cyan text"}
        cd = ColorDef.from_dict(d)
        assert cd.name == "cyan"
        assert cd.ansi_code == "\033[36m"
        assert cd.description == "Cyan text"

    def test_from_dict_missing_description(self):
        d = {"name": "bold", "ansi_code": "\033[1m"}
        cd = ColorDef.from_dict(d)
        assert cd.description == ""

    def test_roundtrip(self):
        cd = ColorDef(name="magenta", ansi_code="\033[35m", description="Purple-ish")
        assert ColorDef.from_dict(cd.as_dict()) == cd


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------


class TestTheme:
    def test_creation(self):
        t = Theme(name="test", colors={"info": "\033[34m"})
        assert t.name == "test"
        assert t.colors["info"] == "\033[34m"

    def test_default_colors(self):
        t = Theme(name="empty")
        assert t.colors == {}

    def test_as_dict(self):
        t = Theme(name="t", colors={"a": "1", "b": "2"})
        d = t.as_dict()
        assert d == {"name": "t", "colors": {"a": "1", "b": "2"}}

    def test_from_dict(self):
        d = {"name": "loaded", "colors": {"error": "\033[31m"}}
        t = Theme.from_dict(d)
        assert t.name == "loaded"
        assert t.colors == {"error": "\033[31m"}

    def test_from_dict_missing_colors(self):
        d = {"name": "bare"}
        t = Theme.from_dict(d)
        assert t.colors == {}

    def test_roundtrip(self):
        t = Theme(name="rt", colors={"x": "a", "y": "b"})
        assert Theme.from_dict(t.as_dict()) == t

    def test_colors_are_copied(self):
        """Ensure from_dict does not share the original dict reference."""
        original = {"error": "\033[31m"}
        d = {"name": "copy_test", "colors": original}
        t = Theme.from_dict(d)
        original["error"] = "changed"
        assert t.colors["error"] == "\033[31m"


# ---------------------------------------------------------------------------
# ANSI constants tests
# ---------------------------------------------------------------------------


class TestAnsiConstants:
    def test_reset_value(self):
        assert ANSI_RESET == "\033[0m"

    def test_colors_dict_has_required_keys(self):
        expected = {"red", "green", "yellow", "blue", "magenta", "cyan", "bold", "dim"}
        assert expected == set(ANSI_COLORS.keys())

    def test_all_codes_are_escape_sequences(self):
        for name, code in ANSI_COLORS.items():
            assert code.startswith("\033["), f"{name} does not start with ESC["

    def test_red_code(self):
        assert ANSI_COLORS["red"] == "\033[31m"

    def test_bold_code(self):
        assert ANSI_COLORS["bold"] == "\033[1m"


# ---------------------------------------------------------------------------
# Built-in theme tests
# ---------------------------------------------------------------------------


class TestBuiltinThemes:
    def test_default_theme_name(self):
        assert DEFAULT_THEME.name == "default"

    def test_default_theme_has_all_roles(self):
        roles = {"success", "error", "warning", "info", "header", "muted"}
        assert roles == set(DEFAULT_THEME.colors.keys())

    def test_dark_theme_name(self):
        assert DARK_THEME.name == "dark"

    def test_dark_theme_uses_bright_colors(self):
        # Bright colors use codes 90-97
        assert "\033[9" in DARK_THEME.colors["success"]

    def test_light_theme_name(self):
        assert LIGHT_THEME.name == "light"

    def test_light_theme_has_all_roles(self):
        roles = {"success", "error", "warning", "info", "header", "muted"}
        assert roles == set(LIGHT_THEME.colors.keys())

    def test_no_color_theme_all_empty(self):
        for role, code in NO_COLOR_THEME.colors.items():
            assert code == "", f"NO_COLOR_THEME role '{role}' has non-empty code"

    def test_no_color_theme_has_all_roles(self):
        roles = {"success", "error", "warning", "info", "header", "muted"}
        assert roles == set(NO_COLOR_THEME.colors.keys())


# ---------------------------------------------------------------------------
# colorize tests
# ---------------------------------------------------------------------------


class TestColorize:
    def test_wraps_text_with_ansi(self):
        result = colorize("hello", "success")
        assert result.startswith(ANSI_COLORS["green"])
        assert result.endswith(ANSI_RESET)
        assert "hello" in result

    def test_custom_theme(self):
        t = Theme(name="custom", colors={"info": "\033[96m"})
        result = colorize("msg", "info", theme=t)
        assert "\033[96m" in result
        assert "msg" in result

    def test_unknown_role_returns_plain(self):
        result = colorize("plain", "nonexistent")
        assert result == "plain"

    def test_respects_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        result = colorize("text", "error")
        assert result == "text"

    def test_no_color_env_unset(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        result = colorize("text", "error")
        assert "\033[" in result

    def test_default_theme_used_when_none(self):
        result = colorize("x", "error")
        assert ANSI_COLORS["red"] in result

    def test_empty_role_code_returns_plain(self):
        result = colorize("x", "success", theme=NO_COLOR_THEME)
        assert result == "x"


# ---------------------------------------------------------------------------
# get_active_theme tests
# ---------------------------------------------------------------------------


class TestGetActiveTheme:
    def test_empty_name_returns_default(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        t = get_active_theme()
        assert t.name == "default"

    def test_known_name(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        t = get_active_theme("dark")
        assert t.name == "dark"

    def test_unknown_name_returns_default(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        t = get_active_theme("nonexistent_theme")
        assert t.name == "default"

    def test_no_color_overrides(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        t = get_active_theme("dark")
        assert t.name == "no_color"

    def test_no_color_with_empty_name(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        t = get_active_theme("")
        assert t.name == "no_color"


# ---------------------------------------------------------------------------
# load_theme_from_config tests
# ---------------------------------------------------------------------------


class TestLoadThemeFromConfig:
    def test_full_config(self):
        cfg = {"name": "my_theme", "colors": {"info": "\033[96m"}}
        t = load_theme_from_config(cfg)
        assert t.name == "my_theme"
        assert t.colors["info"] == "\033[96m"

    def test_missing_name(self):
        cfg = {"colors": {"a": "b"}}
        t = load_theme_from_config(cfg)
        assert t.name == "custom"

    def test_missing_colors(self):
        cfg = {"name": "bare"}
        t = load_theme_from_config(cfg)
        assert t.colors == {}

    def test_empty_config(self):
        t = load_theme_from_config({})
        assert t.name == "custom"
        assert t.colors == {}

    def test_colors_are_copied(self):
        colors = {"a": "1"}
        cfg = {"name": "c", "colors": colors}
        t = load_theme_from_config(cfg)
        colors["a"] = "changed"
        assert t.colors["a"] == "1"


# ---------------------------------------------------------------------------
# strip_ansi tests
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_strips_color_codes(self):
        text = "\033[31mhello\033[0m"
        assert strip_ansi(text) == "hello"

    def test_strips_bold(self):
        text = "\033[1mBOLD\033[0m"
        assert strip_ansi(text) == "BOLD"

    def test_strips_compound_codes(self):
        text = "\033[1;97mheader\033[0m"
        assert strip_ansi(text) == "header"

    def test_no_codes_unchanged(self):
        assert strip_ansi("plain text") == "plain text"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_multiple_codes(self):
        text = "\033[31mred\033[0m and \033[32mgreen\033[0m"
        assert strip_ansi(text) == "red and green"


# ---------------------------------------------------------------------------
# format_theme_preview tests
# ---------------------------------------------------------------------------


class TestFormatThemePreview:
    def test_includes_theme_name(self):
        preview = format_theme_preview(DEFAULT_THEME)
        assert "Theme: default" in preview

    def test_includes_all_roles(self):
        preview = format_theme_preview(DEFAULT_THEME)
        for role in DEFAULT_THEME.colors:
            assert role in preview

    def test_colored_roles_have_ansi(self):
        preview = format_theme_preview(DEFAULT_THEME)
        assert "\033[" in preview

    def test_no_color_theme_has_no_ansi(self):
        preview = format_theme_preview(NO_COLOR_THEME)
        stripped = strip_ansi(preview)
        assert stripped == preview

    def test_multiline_output(self):
        preview = format_theme_preview(DEFAULT_THEME)
        lines = preview.split("\n")
        # Header line + one line per role
        assert len(lines) == 1 + len(DEFAULT_THEME.colors)
