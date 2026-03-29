"""Tests for CLI output width control."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.output_width import (
    WidthConfig,
    compute_column_widths,
    format_progress_bar,
    format_table,
    get_effective_width,
    get_terminal_width,
    truncate_table_row,
    wrap_paragraph,
)


# ---------------------------------------------------------------------------
# WidthConfig dataclass
# ---------------------------------------------------------------------------


class TestWidthConfig:
    def test_defaults(self):
        c = WidthConfig()
        assert c.max_width == 0
        assert c.min_width == 40
        assert c.default_width == 80

    def test_custom_values(self):
        c = WidthConfig(max_width=120, min_width=20, default_width=100)
        assert c.max_width == 120
        assert c.min_width == 20

    def test_as_dict(self):
        c = WidthConfig(max_width=100)
        d = c.as_dict()
        assert d["max_width"] == 100
        assert d["min_width"] == 40
        assert d["default_width"] == 80

    def test_from_dict(self):
        c = WidthConfig.from_dict({"max_width": 60, "min_width": 30})
        assert c.max_width == 60
        assert c.min_width == 30

    def test_from_dict_defaults(self):
        c = WidthConfig.from_dict({})
        assert c.max_width == 0
        assert c.min_width == 40
        assert c.default_width == 80

    def test_roundtrip(self):
        original = WidthConfig(max_width=120, min_width=30, default_width=90)
        restored = WidthConfig.from_dict(original.as_dict())
        assert restored.max_width == original.max_width
        assert restored.min_width == original.min_width
        assert restored.default_width == original.default_width


# ---------------------------------------------------------------------------
# get_terminal_width
# ---------------------------------------------------------------------------


class TestGetTerminalWidth:
    def test_fallback_on_error(self):
        with patch("os.get_terminal_size", side_effect=OSError):
            assert get_terminal_width() == 80

    def test_returns_columns(self):
        mock_size = type("TermSize", (), {"columns": 120, "lines": 40})()
        with patch("os.get_terminal_size", return_value=mock_size):
            assert get_terminal_width() == 120


# ---------------------------------------------------------------------------
# get_effective_width
# ---------------------------------------------------------------------------


class TestGetEffectiveWidth:
    def test_auto_detect(self):
        mock_size = type("TermSize", (), {"columns": 100, "lines": 40})()
        with patch("os.get_terminal_size", return_value=mock_size):
            w = get_effective_width()
            assert w == 100

    def test_configured_width(self):
        config = WidthConfig(max_width=60)
        w = get_effective_width(config)
        assert w == 60

    def test_clamp_to_min(self):
        config = WidthConfig(max_width=10, min_width=40)
        w = get_effective_width(config)
        assert w == 40

    def test_auto_detect_clamped(self):
        mock_size = type("TermSize", (), {"columns": 20, "lines": 40})()
        with patch("os.get_terminal_size", return_value=mock_size):
            w = get_effective_width(WidthConfig(min_width=40))
            assert w == 40

    def test_none_config(self):
        mock_size = type("TermSize", (), {"columns": 90, "lines": 40})()
        with patch("os.get_terminal_size", return_value=mock_size):
            w = get_effective_width(None)
            assert w == 90


# ---------------------------------------------------------------------------
# truncate_table_row
# ---------------------------------------------------------------------------


class TestTruncateTableRow:
    def test_no_truncation(self):
        result = truncate_table_row(["abc", "def"], [10, 10])
        assert "abc" in result
        assert "def" in result

    def test_truncation_with_ellipsis(self):
        result = truncate_table_row(["abcdefghij"], [7])
        assert result == "abcd..."

    def test_very_short_width(self):
        result = truncate_table_row(["abcdef"], [3])
        assert result == "abc"

    def test_exact_fit(self):
        result = truncate_table_row(["abc"], [3])
        assert result == "abc"

    def test_custom_separator(self):
        result = truncate_table_row(["a", "b"], [5, 5], separator=" - ")
        assert " - " in result

    def test_pads_short_columns(self):
        result = truncate_table_row(["ab"], [5])
        assert len(result) == 5

    def test_multiple_columns(self):
        result = truncate_table_row(["name", "score", "grade"], [10, 10, 10])
        parts = result.split(" | ")
        assert len(parts) == 3

    def test_width_of_one(self):
        result = truncate_table_row(["hello"], [1])
        assert result == "h"


# ---------------------------------------------------------------------------
# compute_column_widths
# ---------------------------------------------------------------------------


class TestComputeColumnWidths:
    def test_everything_fits(self):
        widths = compute_column_widths(["A", "B"], [["x", "y"]], max_width=100)
        assert widths == [1, 1]

    def test_respects_min_per_column(self):
        widths = compute_column_widths(
            ["Name", "Score"],
            [["Alice", "95.5"]],
            max_width=10,
        )
        for w in widths:
            assert w >= 5

    def test_proportional_distribution(self):
        widths = compute_column_widths(
            ["name", "description"],
            [["short", "a very long description that needs space"]],
            max_width=40,
        )
        # description column should be wider than name column
        assert widths[1] > widths[0]

    def test_empty_headers(self):
        assert compute_column_widths([], [], max_width=80) == []

    def test_single_column(self):
        widths = compute_column_widths(["header"], [["value"]], max_width=80)
        assert len(widths) == 1

    def test_rows_wider_than_headers(self):
        widths = compute_column_widths(
            ["A", "B"],
            [["long_value_here", "another_long_value"]],
            max_width=200,
        )
        assert widths[0] >= len("long_value_here")
        assert widths[1] >= len("another_long_value")

    def test_narrow_width(self):
        widths = compute_column_widths(
            ["name", "score", "grade"],
            [["Alice", "95.5", "A+"]],
            max_width=20,
        )
        assert all(w >= 5 for w in widths)

    def test_all_empty_content(self):
        widths = compute_column_widths(["", ""], [["", ""]], max_width=80)
        assert len(widths) == 2
        # Should split evenly
        assert sum(widths) > 0


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


class TestFormatTable:
    def test_basic_table(self):
        config = WidthConfig(max_width=80)
        result = format_table(
            ["Name", "Score"],
            [["Alice", "95"], ["Bob", "87"]],
            config=config,
        )
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
        assert "Name" in lines[0]
        assert "-" in lines[1]
        assert "Alice" in lines[2]
        assert "Bob" in lines[3]

    def test_empty_headers(self):
        assert format_table([], []) == ""

    def test_empty_rows(self):
        config = WidthConfig(max_width=80)
        result = format_table(["A", "B"], [], config=config)
        lines = result.split("\n")
        assert len(lines) == 2  # header + separator

    def test_short_rows_padded(self):
        config = WidthConfig(max_width=80)
        result = format_table(
            ["A", "B", "C"],
            [["x"]],  # only 1 column in data
            config=config,
        )
        assert result  # should not crash

    def test_uses_config(self):
        narrow = WidthConfig(max_width=30)
        result = format_table(
            ["Name", "Description"],
            [["Alice", "A very long description that should be truncated"]],
            config=narrow,
        )
        lines = result.split("\n")
        # Verify no line exceeds max width too much (allow some flex for separators)
        for line in lines:
            assert len(line) <= 40  # some tolerance

    def test_consistent_columns(self):
        config = WidthConfig(max_width=80)
        result = format_table(
            ["A", "B"],
            [["x", "y"], ["longer", "value"]],
            config=config,
        )
        lines = result.split("\n")
        # All lines should have the same separator count
        sep_count = lines[0].count(" | ")
        for line in lines:
            assert line.count(" | ") == sep_count


# ---------------------------------------------------------------------------
# wrap_paragraph
# ---------------------------------------------------------------------------


class TestWrapParagraph:
    def test_short_text(self):
        result = wrap_paragraph("hello world", width=80)
        assert result == "hello world"

    def test_wraps_long_text(self):
        text = "word " * 30
        result = wrap_paragraph(text.strip(), width=40)
        lines = result.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 40

    def test_single_long_word(self):
        text = "a" * 100
        result = wrap_paragraph(text, width=50)
        # textwrap.fill breaks long words - all content preserved
        assert result.replace("\n", "") == text

    def test_preserves_content(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = wrap_paragraph(text, width=20)
        # All words should be present
        for word in text.split():
            assert word in result

    def test_explicit_width(self):
        text = "one two three four five six seven eight"
        result = wrap_paragraph(text, width=15)
        for line in result.split("\n"):
            assert len(line) <= 15

    def test_none_width_uses_terminal(self):
        mock_size = type("TermSize", (), {"columns": 50, "lines": 40})()
        with patch("evalyn_sdk.output_width.get_terminal_width", return_value=50):
            result = wrap_paragraph("x " * 100, width=None)
            for line in result.split("\n"):
                assert len(line) <= 50


# ---------------------------------------------------------------------------
# format_progress_bar
# ---------------------------------------------------------------------------


class TestFormatProgressBar:
    def test_zero_progress(self):
        bar = format_progress_bar(0.0, width=40)
        assert bar.startswith("[")
        assert "]" in bar
        assert "0%" in bar

    def test_full_progress(self):
        bar = format_progress_bar(1.0, width=40)
        assert "100%" in bar
        assert ">" not in bar  # full bar has no arrow

    def test_half_progress(self):
        bar = format_progress_bar(0.5, width=40)
        assert "50%" in bar
        assert ">" in bar

    def test_with_label(self):
        bar = format_progress_bar(0.75, width=50, label="loading")
        assert "75%" in bar
        assert "loading" in bar

    def test_clamp_over_one(self):
        bar = format_progress_bar(1.5, width=40)
        assert "100%" in bar

    def test_clamp_below_zero(self):
        bar = format_progress_bar(-0.5, width=40)
        assert "0%" in bar

    def test_starts_with_bracket(self):
        bar = format_progress_bar(0.3, width=40)
        assert bar[0] == "["

    def test_contains_closing_bracket(self):
        bar = format_progress_bar(0.3, width=40)
        assert "]" in bar

    def test_narrow_width(self):
        bar = format_progress_bar(0.5, width=15)
        assert "50%" in bar
        assert bar.startswith("[")

    def test_no_label_no_trailing_space(self):
        bar = format_progress_bar(0.5, width=40, label="")
        assert not bar.endswith(" ")

    def test_various_progress_values(self):
        for pct in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            bar = format_progress_bar(pct, width=50)
            assert bar.startswith("[")
            assert "]" in bar

    def test_none_width_uses_terminal(self):
        with patch("evalyn_sdk.output_width.get_effective_width", return_value=60):
            bar = format_progress_bar(0.5, width=None)
            assert "50%" in bar
