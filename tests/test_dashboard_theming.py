"""Tests for dashboard theming."""

import pytest
from evalyn_sdk.analysis.dashboard_theming import (
    Theme,
    CORPORATE,
    ACADEMIC,
    DARK_MODE,
    PRINT_FRIENDLY,
    DEFAULT,
    BUILTIN_THEMES,
    get_theme,
    list_themes,
)


# =============================================================================
# Theme dataclass
# =============================================================================

class TestTheme:
    def test_default_values(self):
        t = Theme(name="test")
        assert t.bg_primary == "#FFFBF7"
        assert t.status_pass == "#6B8E8E"
        assert t.text_primary == "#1A1A1A"
        assert len(t.chart_palette) == 8

    def test_css_variables(self):
        t = Theme(name="t")
        css = t.css_variables()
        assert "--bg-primary:" in css
        assert "--status-pass:" in css
        assert "--text-muted:" in css
        assert "--border-light:" in css

    def test_chart_js_colors(self):
        t = Theme(name="t")
        colors = t.chart_js_colors()
        assert "accent" in colors
        assert "success" in colors
        assert "error" in colors
        assert "bg" in colors
        assert "text" in colors
        assert colors["accent"] == t.accent_primary
        assert colors["success"] == t.status_pass

    def test_as_dict(self):
        t = Theme(name="myname", description="desc")
        d = t.as_dict()
        assert d["name"] == "myname"
        assert d["description"] == "desc"
        assert "chart_palette" in d
        assert isinstance(d["chart_palette"], list)
        assert "custom_css" in d

    def test_from_dict(self):
        data = {
            "name": "restored",
            "bg_primary": "#000000",
            "status_pass": "#00FF00",
            "chart_palette": ["#111", "#222"],
        }
        t = Theme.from_dict(data)
        assert t.name == "restored"
        assert t.bg_primary == "#000000"
        assert t.status_pass == "#00FF00"
        assert t.chart_palette == ["#111", "#222"]

    def test_from_dict_defaults(self):
        t = Theme.from_dict({"name": "minimal"})
        assert t.bg_primary == "#FFFBF7"
        assert t.text_primary == "#1A1A1A"

    def test_roundtrip(self):
        original = CORPORATE
        d = original.as_dict()
        restored = Theme.from_dict(d)
        assert restored.name == original.name
        assert restored.bg_primary == original.bg_primary
        assert restored.chart_palette == original.chart_palette
        assert restored.status_fail == original.status_fail

    def test_custom_css(self):
        t = Theme(name="branded", custom_css=".header { color: red; }")
        assert t.custom_css == ".header { color: red; }"
        d = t.as_dict()
        assert d["custom_css"] == ".header { color: red; }"


# =============================================================================
# Built-in themes
# =============================================================================

class TestBuiltinThemes:
    def test_five_builtins(self):
        assert len(BUILTIN_THEMES) == 5

    def test_corporate(self):
        assert CORPORATE.name == "corporate"
        assert CORPORATE.accent_primary == "#0D6EFD"
        assert CORPORATE.bg_primary == "#FFFFFF"

    def test_academic(self):
        assert ACADEMIC.name == "academic"
        assert ACADEMIC.accent_primary == "#37474F"

    def test_dark_mode(self):
        assert DARK_MODE.name == "dark-mode"
        assert DARK_MODE.bg_primary == "#1A1A2E"
        assert DARK_MODE.text_primary == "#ECF0F1"

    def test_print_friendly(self):
        assert PRINT_FRIENDLY.name == "print-friendly"
        assert PRINT_FRIENDLY.bg_primary == "#FFFFFF"
        assert PRINT_FRIENDLY.text_primary == "#000000"

    def test_default_matches_html_report(self):
        # Default theme should match existing html_report colors
        assert DEFAULT.bg_primary == "#FFFBF7"
        assert DEFAULT.accent_primary == "#D4A27F"
        assert DEFAULT.status_pass == "#6B8E8E"
        assert DEFAULT.status_fail == "#C97B63"

    def test_all_themes_have_8_chart_colors(self):
        for name, theme in BUILTIN_THEMES.items():
            assert len(theme.chart_palette) == 8, f"{name} has {len(theme.chart_palette)} colors"

    def test_all_themes_produce_valid_css(self):
        for name, theme in BUILTIN_THEMES.items():
            css = theme.css_variables()
            assert "--bg-primary:" in css
            assert "--text-primary:" in css


# =============================================================================
# get_theme
# =============================================================================

class TestGetTheme:
    def test_get_builtin(self):
        t = get_theme("corporate")
        assert t.name == "corporate"

    def test_get_default(self):
        t = get_theme("default")
        assert t.name == "default"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme("nonexistent")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="corporate"):
            get_theme("nope")

    def test_custom_theme(self):
        customs = {"brand": {"bg_primary": "#FF0000", "description": "Brand theme"}}
        t = get_theme("brand", custom_themes=customs)
        assert t.name == "brand"
        assert t.bg_primary == "#FF0000"
        assert t.description == "Brand theme"

    def test_custom_overrides_builtin(self):
        customs = {"corporate": {"bg_primary": "#123456"}}
        t = get_theme("corporate", custom_themes=customs)
        assert t.bg_primary == "#123456"

    def test_custom_inherits_defaults(self):
        customs = {"mine": {"accent_primary": "#AABBCC"}}
        t = get_theme("mine", custom_themes=customs)
        assert t.accent_primary == "#AABBCC"
        # Other fields get Theme defaults
        assert t.bg_primary == "#FFFBF7"
        # chart_palette should default to 8 colors, not empty
        assert len(t.chart_palette) == 8


# =============================================================================
# list_themes
# =============================================================================

class TestListThemes:
    def test_builtin_only(self):
        themes = list_themes()
        assert len(themes) == 5
        names = {t.name for t in themes}
        assert "default" in names
        assert "corporate" in names
        assert "dark-mode" in names

    def test_with_custom(self):
        customs = {"nightfall": {"bg_primary": "#000"}}
        themes = list_themes(custom_themes=customs)
        assert len(themes) == 6
        names = {t.name for t in themes}
        assert "nightfall" in names

    def test_sorted_by_name(self):
        themes = list_themes()
        names = [t.name for t in themes]
        assert names == sorted(names)
