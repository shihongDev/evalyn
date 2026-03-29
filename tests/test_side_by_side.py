"""Tests for side-by-side view module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.side_by_side import (
    Column,
    SideBySideConfig,
    pad_line,
    render_diff_side_by_side,
    render_header,
    render_side_by_side,
    truncate_line,
    wrap_text,
)


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


class TestColumn:
    def test_default_values(self):
        col = Column(title="test")
        assert col.lines == []
        assert col.width == 0

    def test_custom_values(self):
        col = Column(title="Results", lines=["line1", "line2"], width=40)
        assert col.title == "Results"
        assert len(col.lines) == 2
        assert col.width == 40

    def test_as_dict(self):
        col = Column(title="A", lines=["x"], width=10)
        d = col.as_dict()
        assert d == {"title": "A", "lines": ["x"], "width": 10}

    def test_from_dict_full(self):
        d = {"title": "B", "lines": ["y", "z"], "width": 20}
        col = Column.from_dict(d)
        assert col.title == "B"
        assert col.lines == ["y", "z"]
        assert col.width == 20

    def test_from_dict_defaults(self):
        d = {"title": "C"}
        col = Column.from_dict(d)
        assert col.lines == []
        assert col.width == 0

    def test_roundtrip(self):
        original = Column(title="round", lines=["a", "b"], width=30)
        rebuilt = Column.from_dict(original.as_dict())
        assert rebuilt.title == original.title
        assert rebuilt.lines == original.lines
        assert rebuilt.width == original.width

    def test_lines_are_copied(self):
        """as_dict should return a copy of lines, not a reference."""
        col = Column(title="test", lines=["a"])
        d = col.as_dict()
        d["lines"].append("b")
        assert len(col.lines) == 1


# ---------------------------------------------------------------------------
# SideBySideConfig
# ---------------------------------------------------------------------------


class TestSideBySideConfig:
    def test_default_values(self):
        cfg = SideBySideConfig()
        assert cfg.total_width == 80
        assert cfg.separator == " | "
        assert cfg.padding == 1

    def test_custom_values(self):
        cfg = SideBySideConfig(total_width=120, separator=" || ", padding=2)
        assert cfg.total_width == 120
        assert cfg.separator == " || "
        assert cfg.padding == 2

    def test_as_dict(self):
        cfg = SideBySideConfig()
        d = cfg.as_dict()
        assert d == {"total_width": 80, "separator": " | ", "padding": 1}

    def test_from_dict(self):
        d = {"total_width": 100, "separator": " : ", "padding": 0}
        cfg = SideBySideConfig.from_dict(d)
        assert cfg.total_width == 100
        assert cfg.separator == " : "
        assert cfg.padding == 0

    def test_from_dict_defaults(self):
        cfg = SideBySideConfig.from_dict({})
        assert cfg.total_width == 80

    def test_roundtrip(self):
        original = SideBySideConfig(total_width=60, separator=" - ", padding=3)
        rebuilt = SideBySideConfig.from_dict(original.as_dict())
        assert rebuilt.total_width == original.total_width
        assert rebuilt.separator == original.separator
        assert rebuilt.padding == original.padding


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------


class TestWrapText:
    def test_short_text_no_wrap(self):
        result = wrap_text("hello", 20)
        assert result == ["hello"]

    def test_exact_width(self):
        result = wrap_text("12345", 5)
        assert result == ["12345"]

    def test_wraps_at_word_boundary(self):
        result = wrap_text("hello world", 5)
        assert result == ["hello", "world"]

    def test_multiple_words(self):
        result = wrap_text("the quick brown fox", 10)
        assert result == ["the quick", "brown fox"]

    def test_long_word_force_break(self):
        result = wrap_text("abcdefghij", 5)
        assert result == ["abcde", "fghij"]

    def test_empty_text(self):
        result = wrap_text("", 10)
        assert result == [""]

    def test_preserves_newlines(self):
        result = wrap_text("line1\nline2", 20)
        assert result == ["line1", "line2"]

    def test_newline_with_wrap(self):
        result = wrap_text("hello world\nfoo bar", 5)
        assert result == ["hello", "world", "foo", "bar"]

    def test_zero_width(self):
        result = wrap_text("hello", 0)
        assert result == ["hello"]

    def test_single_space(self):
        result = wrap_text("a b c d", 3)
        assert result == ["a b", "c d"]

    def test_empty_paragraph_between_newlines(self):
        result = wrap_text("a\n\nb", 10)
        assert result == ["a", "", "b"]


# ---------------------------------------------------------------------------
# truncate_line
# ---------------------------------------------------------------------------


class TestTruncateLine:
    def test_short_line_unchanged(self):
        assert truncate_line("hi", 10) == "hi"

    def test_exact_width_unchanged(self):
        assert truncate_line("12345", 5) == "12345"

    def test_long_line_truncated(self):
        result = truncate_line("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_width_3(self):
        result = truncate_line("hello", 3)
        assert result == "..."

    def test_width_2(self):
        result = truncate_line("hello", 2)
        assert result == ".."

    def test_width_1(self):
        result = truncate_line("hello", 1)
        assert result == "."

    def test_width_0(self):
        result = truncate_line("hello", 0)
        assert result == ""

    def test_empty_line(self):
        assert truncate_line("", 10) == ""


# ---------------------------------------------------------------------------
# pad_line
# ---------------------------------------------------------------------------


class TestPadLine:
    def test_short_line_padded(self):
        result = pad_line("hi", 10)
        assert result == "hi        "
        assert len(result) == 10

    def test_exact_width_unchanged(self):
        result = pad_line("12345", 5)
        assert result == "12345"

    def test_longer_than_width(self):
        result = pad_line("hello world", 5)
        assert result == "hello world"

    def test_empty_line(self):
        result = pad_line("", 5)
        assert result == "     "

    def test_zero_width(self):
        result = pad_line("hi", 0)
        assert result == "hi"


# ---------------------------------------------------------------------------
# render_header
# ---------------------------------------------------------------------------


class TestRenderHeader:
    def test_basic_header(self):
        result = render_header("Left", "Right", 10, " | ")
        lines = result.split("\n")
        assert len(lines) == 2
        assert "Left" in lines[0]
        assert "Right" in lines[0]
        assert " | " in lines[0]

    def test_header_has_dashes(self):
        result = render_header("A", "B", 10, " | ")
        lines = result.split("\n")
        assert "----------" in lines[1]

    def test_header_width(self):
        result = render_header("A", "B", 10, " | ")
        lines = result.split("\n")
        expected_width = 10 + 3 + 10  # col + sep + col
        assert len(lines[0]) == expected_width
        assert len(lines[1]) == expected_width

    def test_long_title_truncated(self):
        result = render_header("A very long title", "B", 10, " | ")
        lines = result.split("\n")
        # Left title should be truncated to 10 chars
        left_part = lines[0].split(" | ")[0]
        assert len(left_part) == 10


# ---------------------------------------------------------------------------
# render_side_by_side
# ---------------------------------------------------------------------------


class TestRenderSideBySide:
    def test_basic_render(self):
        left = Column(title="Left", lines=["hello"])
        right = Column(title="Right", lines=["world"])
        result = render_side_by_side(left, right)
        assert "Left" in result
        assert "Right" in result
        assert "hello" in result
        assert "world" in result

    def test_default_config(self):
        left = Column(title="A", lines=["x"])
        right = Column(title="B", lines=["y"])
        result = render_side_by_side(left, right)
        # Should use default 80-width
        lines = result.split("\n")
        # Header line should be consistent width
        assert " | " in lines[0]

    def test_unequal_line_counts(self):
        left = Column(title="A", lines=["1", "2", "3"])
        right = Column(title="B", lines=["a"])
        result = render_side_by_side(left, right)
        lines = result.split("\n")
        # 2 header lines + 3 body lines
        assert len(lines) == 5

    def test_empty_columns(self):
        left = Column(title="A", lines=[])
        right = Column(title="B", lines=[])
        result = render_side_by_side(left, right)
        # Should still have header
        assert "A" in result
        assert "B" in result

    def test_custom_config(self):
        left = Column(title="L", lines=["data"])
        right = Column(title="R", lines=["info"])
        cfg = SideBySideConfig(total_width=40, separator=" : ")
        result = render_side_by_side(left, right, cfg)
        assert " : " in result

    def test_separator_in_output(self):
        left = Column(title="A", lines=["x"])
        right = Column(title="B", lines=["y"])
        cfg = SideBySideConfig(separator=" || ")
        result = render_side_by_side(left, right, cfg)
        assert " || " in result

    def test_fixed_left_width(self):
        left = Column(title="A", lines=["short"], width=15)
        right = Column(title="B", lines=["data"])
        result = render_side_by_side(left, right)
        lines = result.split("\n")
        # First content line after header
        header_line = lines[0]
        left_part = header_line.split(" | ")[0]
        assert len(left_part) == 15

    def test_long_lines_wrapped(self):
        long_line = "word " * 20
        left = Column(title="A", lines=[long_line.strip()])
        right = Column(title="B", lines=["short"])
        cfg = SideBySideConfig(total_width=40)
        result = render_side_by_side(left, right, cfg)
        lines = result.split("\n")
        # Should have more body lines than 1 due to wrapping
        body_lines = lines[2:]  # Skip header + dash
        assert len(body_lines) > 1


# ---------------------------------------------------------------------------
# render_diff_side_by_side
# ---------------------------------------------------------------------------


class TestRenderDiffSideBySide:
    def test_basic_diff(self):
        result = render_diff_side_by_side("hello\nworld", "hello\nearth")
        assert "A" in result
        assert "B" in result
        assert "hello" in result

    def test_custom_titles(self):
        result = render_diff_side_by_side("x", "y", title_a="Before", title_b="After")
        assert "Before" in result
        assert "After" in result

    def test_custom_width(self):
        result = render_diff_side_by_side("a", "b", width=40)
        lines = result.split("\n")
        # All lines should be approximately 40 chars wide
        # (col + separator + col)
        for line in lines:
            assert len(line) <= 42  # Some tolerance for separator

    def test_empty_texts(self):
        result = render_diff_side_by_side("", "")
        assert "A" in result
        assert "B" in result

    def test_multiline_comparison(self):
        text_a = "line1\nline2\nline3"
        text_b = "lineA\nlineB"
        result = render_diff_side_by_side(text_a, text_b)
        lines = result.split("\n")
        # 2 header + 3 body (max of left 3 lines, right 2 lines)
        assert len(lines) == 5

    def test_single_line_comparison(self):
        result = render_diff_side_by_side("only", "one")
        lines = result.split("\n")
        # 2 header + 1 body
        assert len(lines) == 3
