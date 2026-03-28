"""Tests for dark mode / theme support module."""
from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.dark_mode import (
    ThemeColors,
    ThemeConfig,
    LIGHT_THEME,
    DARK_THEME,
    get_theme_colors,
    generate_css_variables,
    generate_theme_css,
    apply_theme_to_html,
    toggle_theme,
    list_available_themes,
)


# ---------------------------------------------------------------------------
# ThemeColors
# ---------------------------------------------------------------------------


class TestThemeColors:
    def test_as_dict_keys(self):
        d = LIGHT_THEME.as_dict()
        expected_keys = {
            "bg_primary", "bg_secondary", "text_primary", "text_secondary",
            "accent", "success", "warning", "error", "border",
        }
        assert set(d.keys()) == expected_keys

    def test_as_dict_values(self):
        d = DARK_THEME.as_dict()
        assert d["bg_primary"] == "#1A1A2E"
        assert d["text_primary"] == "#ECF0F1"

    def test_from_dict(self):
        d = LIGHT_THEME.as_dict()
        restored = ThemeColors.from_dict(d)
        assert restored.bg_primary == LIGHT_THEME.bg_primary
        assert restored.border == LIGHT_THEME.border

    def test_roundtrip(self):
        original = ThemeColors(
            "#000", "#111", "#fff", "#ccc", "#f00", "#0f0", "#ff0", "#f00", "#333",
        )
        restored = ThemeColors.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# ThemeConfig
# ---------------------------------------------------------------------------


class TestThemeConfig:
    def test_defaults(self):
        cfg = ThemeConfig()
        assert cfg.mode == "light"
        assert cfg.custom_colors is None

    def test_as_dict_no_custom(self):
        d = ThemeConfig(mode="dark").as_dict()
        assert d == {"mode": "dark"}

    def test_as_dict_with_custom(self):
        custom = ThemeColors(
            "#a", "#b", "#c", "#d", "#e", "#f", "#g", "#h", "#i",
        )
        d = ThemeConfig(mode="light", custom_colors=custom).as_dict()
        assert "custom_colors" in d
        assert d["custom_colors"]["bg_primary"] == "#a"

    def test_from_dict_simple(self):
        cfg = ThemeConfig.from_dict({"mode": "dark"})
        assert cfg.mode == "dark"
        assert cfg.custom_colors is None

    def test_from_dict_with_custom(self):
        raw = {
            "mode": "light",
            "custom_colors": DARK_THEME.as_dict(),
        }
        cfg = ThemeConfig.from_dict(raw)
        assert cfg.custom_colors is not None
        assert cfg.custom_colors.bg_primary == DARK_THEME.bg_primary

    def test_from_dict_defaults(self):
        cfg = ThemeConfig.from_dict({})
        assert cfg.mode == "light"


# ---------------------------------------------------------------------------
# get_theme_colors
# ---------------------------------------------------------------------------


class TestGetThemeColors:
    def test_light(self):
        colors = get_theme_colors(ThemeConfig(mode="light"))
        assert colors is LIGHT_THEME

    def test_dark(self):
        colors = get_theme_colors(ThemeConfig(mode="dark"))
        assert colors is DARK_THEME

    def test_auto_returns_light(self):
        colors = get_theme_colors(ThemeConfig(mode="auto"))
        assert colors is LIGHT_THEME

    def test_custom_overrides(self):
        custom = ThemeColors(
            "#000", "#111", "#fff", "#ccc", "#f00", "#0f0", "#ff0", "#f00", "#333",
        )
        colors = get_theme_colors(ThemeConfig(mode="dark", custom_colors=custom))
        assert colors is custom


# ---------------------------------------------------------------------------
# generate_css_variables
# ---------------------------------------------------------------------------


class TestGenerateCSSVariables:
    def test_has_all_properties(self):
        css = generate_css_variables(LIGHT_THEME)
        for prop in (
            "--bg-primary", "--bg-secondary", "--text-primary", "--text-secondary",
            "--accent", "--success", "--warning", "--error", "--border",
        ):
            assert prop in css

    def test_values_present(self):
        css = generate_css_variables(DARK_THEME)
        assert "#1A1A2E" in css
        assert "#ECF0F1" in css


# ---------------------------------------------------------------------------
# generate_theme_css
# ---------------------------------------------------------------------------


class TestGenerateThemeCSS:
    def test_light_has_root(self):
        css = generate_theme_css(ThemeConfig(mode="light"))
        assert ":root" in css
        assert "prefers-color-scheme" not in css

    def test_dark_has_root(self):
        css = generate_theme_css(ThemeConfig(mode="dark"))
        assert ":root" in css

    def test_auto_has_media_query(self):
        css = generate_theme_css(ThemeConfig(mode="auto"))
        assert "@media (prefers-color-scheme: dark)" in css

    def test_auto_has_both_palettes(self):
        css = generate_theme_css(ThemeConfig(mode="auto"))
        # Light palette in :root
        assert LIGHT_THEME.bg_primary in css
        # Dark palette inside media query
        assert DARK_THEME.bg_primary in css

    def test_body_uses_variables(self):
        css = generate_theme_css(ThemeConfig(mode="light"))
        assert "var(--bg-primary)" in css
        assert "var(--text-primary)" in css


# ---------------------------------------------------------------------------
# apply_theme_to_html
# ---------------------------------------------------------------------------


class TestApplyThemeToHtml:
    def test_injects_into_style_tag(self):
        html_in = "<html><head><style>.x{}</style></head><body></body></html>"
        result = apply_theme_to_html(html_in, ThemeConfig(mode="dark"))
        assert "--bg-primary" in result
        assert DARK_THEME.bg_primary in result

    def test_injects_before_head_close(self):
        html_in = "<html><head></head><body></body></html>"
        result = apply_theme_to_html(html_in, ThemeConfig(mode="light"))
        assert "--bg-primary" in result
        assert "<style>" in result

    def test_fallback_prepend(self):
        html_in = "<div>bare</div>"
        result = apply_theme_to_html(html_in, ThemeConfig(mode="dark"))
        assert "--bg-primary" in result
        assert "<div>bare</div>" in result

    def test_auto_mode_includes_media_query(self):
        html_in = "<html><head><style></style></head><body></body></html>"
        result = apply_theme_to_html(html_in, ThemeConfig(mode="auto"))
        assert "@media (prefers-color-scheme: dark)" in result


# ---------------------------------------------------------------------------
# toggle_theme
# ---------------------------------------------------------------------------


class TestToggleTheme:
    def test_light_to_dark(self):
        assert toggle_theme("light") == "dark"

    def test_dark_to_light(self):
        assert toggle_theme("dark") == "light"

    def test_unknown_to_light(self):
        # Anything other than "light" returns "light"
        assert toggle_theme("auto") == "light"


# ---------------------------------------------------------------------------
# list_available_themes
# ---------------------------------------------------------------------------


class TestListAvailableThemes:
    def test_has_three(self):
        themes = list_available_themes()
        assert len(themes) == 3

    def test_contents(self):
        themes = list_available_themes()
        assert "light" in themes
        assert "dark" in themes
        assert "auto" in themes

    def test_returns_new_list(self):
        a = list_available_themes()
        b = list_available_themes()
        assert a == b
        assert a is not b
