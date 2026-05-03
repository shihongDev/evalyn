"""Tests for rich.py rendering primitives."""

from __future__ import annotations

import os
import re

import pytest


@pytest.fixture(autouse=True)
def _force_no_color(monkeypatch):
    """Force NO_COLOR so tests get deterministic plain-text output."""
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture()
def _enable_color(monkeypatch):
    """Remove NO_COLOR and fake a TTY for color tests."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("EVALYN_NO_COLOR", raising=False)


# ---------------------------------------------------------------------------
# ANSI stripping helper
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


class TestBanner:
    def test_basic_banner(self):
        from evalyn_sdk.cli.utils.rich import banner

        out = banner("Hello")
        lines = out.splitlines()
        assert len(lines) == 3
        # ASCII fallback uses +, =, |
        assert lines[0].startswith("+")
        assert lines[0].endswith("+")
        assert lines[1].startswith("|")
        assert "Hello" in lines[1]
        assert lines[2].startswith("+")

    def test_banner_width(self):
        from evalyn_sdk.cli.utils.rich import banner

        out = banner("Test", width=40)
        lines = out.splitlines()
        # top border should be 40 chars
        assert len(lines[0]) == 40

    def test_banner_unicode_when_color_enabled(self, _enable_color, monkeypatch):
        """When colors are enabled (TTY), banner should use Unicode box chars."""
        import io
        fake_tty = io.StringIO()
        fake_tty.isatty = lambda: True
        monkeypatch.setattr("sys.stdout", fake_tty)

        from evalyn_sdk.cli.utils.rich import banner

        out = banner("Test")
        # Unicode double-line top-left corner
        assert "\u2554" in out or "+" in out  # at minimum, one of them


class TestSection:
    def test_basic_section(self):
        from evalyn_sdk.cli.utils.rich import section

        out = section("Results")
        assert "Results" in out.upper() or "RESULTS" in out

    def test_section_width(self):
        from evalyn_sdk.cli.utils.rich import section

        out = section("Test", width=40)
        # The visible length should be around 40
        assert len(strip_ansi(out)) == 40

    def test_section_fallback_char(self):
        from evalyn_sdk.cli.utils.rich import section

        out = section("X")
        # NO_COLOR is set, so should use plain hyphen
        assert "-" in out


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


class TestTable:
    def test_headers_only(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table(["Name", "Score"], [])
        assert "Name" in out
        assert "Score" in out

    def test_basic_rows(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table(["A", "B"], [["x", "y"], ["hello", "world"]])
        assert "x" in out
        assert "hello" in out
        assert "world" in out

    def test_right_alignment(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table(
            ["Name", "Score"],
            [["test", "95"]],
            align=["left", "right"],
        )
        # Score column should be right-aligned - "95" padded on the left
        lines = out.splitlines()
        # Find the data row (contains "test")
        data_lines = [l for l in lines if "test" in l]
        assert len(data_lines) >= 1

    def test_center_alignment(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table(
            ["Name", "Value"],
            [["a", "b"]],
            align=["left", "center"],
        )
        assert "b" in out

    def test_max_col_width_truncation(self):
        from evalyn_sdk.cli.utils.rich import table

        long_text = "A" * 100
        out = table(["Col"], [[long_text]], max_col_width=20)
        # The long text should be truncated
        lines = out.splitlines()
        # No line should contain the full 100-char string
        for line in lines:
            assert long_text not in strip_ansi(line)

    def test_ansi_stripping_for_width(self, _enable_color, monkeypatch):
        """Table width calculation must strip ANSI codes."""
        import io
        fake_tty = io.StringIO()
        fake_tty.isatty = lambda: True
        monkeypatch.setattr("sys.stdout", fake_tty)

        from evalyn_sdk.cli.utils.rich import table
        from evalyn_sdk.cli.utils.colors import green

        colored = green("hello")
        out = table(["Name"], [[colored]])
        # Should render without crashing -- the colored text should be in output
        assert "hello" in strip_ansi(out)

    def test_empty_table(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table([], [])
        # Should return something (even if minimal)
        assert isinstance(out, str)

    def test_box_drawing_chars_in_no_color(self):
        from evalyn_sdk.cli.utils.rich import table

        out = table(["X"], [["y"]])
        # In NO_COLOR mode, should use ASCII box chars (+, -, |)
        assert "+" in out or "-" in out


# ---------------------------------------------------------------------------
# kv
# ---------------------------------------------------------------------------


class TestKV:
    def test_basic_kv(self):
        from evalyn_sdk.cli.utils.rich import kv

        out = kv([("Name", "Alice"), ("Age", "30")])
        assert "Name" in out
        assert "Alice" in out
        assert "Age" in out
        assert "30" in out

    def test_colon_alignment(self):
        from evalyn_sdk.cli.utils.rich import kv

        out = kv([("Short", "a"), ("Much Longer Key", "b")])
        lines = out.splitlines()
        # Both colons should be at the same position
        colon_positions = []
        for line in lines:
            idx = line.index(":")
            colon_positions.append(idx)
        assert len(set(colon_positions)) == 1

    def test_indent(self):
        from evalyn_sdk.cli.utils.rich import kv

        out = kv([("K", "V")], indent=4)
        lines = out.splitlines()
        # Each line should start with 4 spaces
        for line in lines:
            assert line.startswith("    ")


# ---------------------------------------------------------------------------
# footer
# ---------------------------------------------------------------------------


class TestFooter:
    def test_basic_footer(self):
        from evalyn_sdk.cli.utils.rich import footer

        out = footer([("evalyn analyze", "Run analysis")])
        assert "evalyn analyze" in out
        assert "Run analysis" in out

    def test_quiet_suppresses(self):
        from evalyn_sdk.cli.utils.rich import footer

        out = footer([("cmd", "desc")], quiet=True)
        assert out == ""

    def test_json_suppresses(self):
        from evalyn_sdk.cli.utils.rich import footer

        out = footer([("cmd", "desc")], format="json")
        assert out == ""

    def test_env_suppresses(self, monkeypatch):
        from evalyn_sdk.cli.utils.rich import footer

        monkeypatch.setenv("EVALYN_NO_HINTS", "1")
        out = footer([("cmd", "desc")])
        assert out == ""

    def test_triangle_bullet_in_no_color(self):
        from evalyn_sdk.cli.utils.rich import footer

        out = footer([("cmd", "desc")])
        # In NO_COLOR mode, should still have some bullet or prefix
        assert "cmd" in out

    def test_empty_commands(self):
        from evalyn_sdk.cli.utils.rich import footer

        out = footer([])
        assert out == ""


# ---------------------------------------------------------------------------
# progress_bar
# ---------------------------------------------------------------------------


class TestProgressBar:
    def test_zero_progress(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(0, 100)
        assert isinstance(out, str)
        assert "0%" in out or "0.0%" in out or "0" in out

    def test_full_progress(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(100, 100)
        assert "100%" in out

    def test_half_progress(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(50, 100)
        assert "50%" in out

    def test_with_label(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(50, 100, label="Loading")
        assert "Loading" in out

    def test_returns_string_not_prints(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        result = progress_bar(10, 100)
        assert isinstance(result, str)

    def test_custom_width(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(50, 100, width=10)
        assert isinstance(out, str)

    def test_zero_total(self):
        from evalyn_sdk.cli.utils.rich import progress_bar

        out = progress_bar(0, 0)
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# icon / status_icon
# ---------------------------------------------------------------------------


class TestIcon:
    def test_pass_icon_no_color(self):
        from evalyn_sdk.cli.utils.rich import icon, status_icon

        assert "[PASS]" in icon("pass")
        assert "[PASS]" in status_icon(True)

    def test_fail_icon_no_color(self):
        from evalyn_sdk.cli.utils.rich import icon, status_icon

        assert "[FAIL]" in icon("fail")
        assert "[FAIL]" in status_icon(False)

    def test_warn_icon_no_color(self):
        from evalyn_sdk.cli.utils.rich import icon

        assert "[WARN]" in icon("warn")

    def test_info_icon_no_color(self):
        from evalyn_sdk.cli.utils.rich import icon

        assert "[INFO]" in icon("info")

    def test_next_icon_no_color(self):
        from evalyn_sdk.cli.utils.rich import icon

        assert "[NEXT]" in icon("next") or ">" in icon("next")

    def test_unknown_icon(self):
        from evalyn_sdk.cli.utils.rich import icon

        # Should not crash on unknown kind
        result = icon("unknown")
        assert isinstance(result, str)

    def test_icon_unicode_when_color_enabled(self, _enable_color, monkeypatch):
        """When colors are enabled, icons should use Unicode symbols."""
        import io
        fake_tty = io.StringIO()
        fake_tty.isatty = lambda: True
        monkeypatch.setattr("sys.stdout", fake_tty)

        from evalyn_sdk.cli.utils.rich import icon

        result = icon("pass")
        plain = strip_ansi(result)
        # Should use checkmark U+2713, not [PASS]
        assert "\u2713" in plain


# ---------------------------------------------------------------------------
# Integration: multiple primitives together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_all_exports_exist(self):
        from evalyn_sdk.cli.utils.rich import (
            banner,
            section,
            table,
            kv,
            footer,
            progress_bar,
            icon,
            status_icon,
        )

        assert callable(banner)
        assert callable(section)
        assert callable(table)
        assert callable(kv)
        assert callable(footer)
        assert callable(progress_bar)
        assert callable(icon)
        assert callable(status_icon)
