"""
Side-by-side view: render two text columns for terminal comparison.

Provides word-wrapping, truncation, and padding utilities for displaying
two outputs side by side in a fixed-width terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Column:
    """A single column of text with an optional fixed width."""

    title: str
    lines: list[str] = field(default_factory=list)
    width: int = 0  # 0 means auto

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "lines": list(self.lines),
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Column:
        return cls(
            title=data["title"],
            lines=list(data.get("lines", [])),
            width=data.get("width", 0),
        )


@dataclass
class SideBySideConfig:
    """Configuration for side-by-side rendering."""

    total_width: int = 80
    separator: str = " | "
    padding: int = 1

    def as_dict(self) -> dict:
        return {
            "total_width": self.total_width,
            "separator": self.separator,
            "padding": self.padding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SideBySideConfig:
        return cls(
            total_width=data.get("total_width", 80),
            separator=data.get("separator", " | "),
            padding=data.get("padding", 1),
        )


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap text to the given width.

    Splits on whitespace boundaries. Words longer than width are
    force-broken at the width boundary.
    """
    if width <= 0:
        return [text] if text else [""]

    if not text:
        return [""]

    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue

        words = paragraph.split()
        if not words:
            result.append("")
            continue

        current_line = ""
        for word in words:
            if not current_line:
                # Start a new line - force-break long words
                while len(word) > width:
                    result.append(word[:width])
                    word = word[width:]
                current_line = word
            elif len(current_line) + 1 + len(word) <= width:
                current_line += " " + word
            else:
                result.append(current_line)
                # Force-break long words
                while len(word) > width:
                    result.append(word[:width])
                    word = word[width:]
                current_line = word

        if current_line:
            result.append(current_line)

    return result if result else [""]


def truncate_line(line: str, width: int) -> str:
    """Truncate a line with '...' if it exceeds the given width."""
    if width <= 0:
        return ""
    if len(line) <= width:
        return line
    if width <= 3:
        return "." * width
    return line[: width - 3] + "..."


def pad_line(line: str, width: int) -> str:
    """Right-pad a line with spaces to reach the given width."""
    if len(line) >= width:
        return line
    return line + " " * (width - len(line))


# ---------------------------------------------------------------------------
# Header rendering
# ---------------------------------------------------------------------------


def render_header(
    left_title: str,
    right_title: str,
    col_width: int,
    separator: str,
) -> str:
    """Render column headers with a separator and underline.

    Returns a two-line string: titles on the first line, dashes on the second.
    """
    left = pad_line(truncate_line(left_title, col_width), col_width)
    right = pad_line(truncate_line(right_title, col_width), col_width)
    title_line = left + separator + right
    dash_line = "-" * col_width + separator + "-" * col_width
    return title_line + "\n" + dash_line


# ---------------------------------------------------------------------------
# Side-by-side rendering
# ---------------------------------------------------------------------------


def _compute_col_width(
    left: Column,
    right: Column,
    config: SideBySideConfig,
) -> int:
    """Compute the column width for each side.

    If a column specifies a fixed width, that is used; otherwise the
    available space is split evenly between the two columns after
    subtracting the separator.
    """
    sep_len = len(config.separator)
    available = config.total_width - sep_len

    if left.width > 0 and right.width > 0:
        # Both fixed - just use left.width (caller controls)
        return left.width
    if left.width > 0:
        return left.width
    if right.width > 0:
        return available - right.width

    return max(available // 2, 1)


def render_side_by_side(
    left: Column,
    right: Column,
    config: SideBySideConfig | None = None,
) -> str:
    """Render two columns side by side.

    Lines are wrapped/truncated to fit within the configured total width.
    """
    if config is None:
        config = SideBySideConfig()

    col_width = _compute_col_width(left, right, config)
    sep = config.separator

    # Wrap all lines in each column
    left_lines: list[str] = []
    for line in left.lines:
        left_lines.extend(wrap_text(line, col_width))

    right_lines: list[str] = []
    for line in right.lines:
        right_lines.extend(wrap_text(line, col_width))

    # Build header
    header = render_header(left.title, right.title, col_width, sep)

    # Build body - zip with padding
    max_rows = max(len(left_lines), len(right_lines))
    body_lines: list[str] = []
    for i in range(max_rows):
        l_text = left_lines[i] if i < len(left_lines) else ""
        r_text = right_lines[i] if i < len(right_lines) else ""
        l_text = pad_line(truncate_line(l_text, col_width), col_width)
        r_text = pad_line(truncate_line(r_text, col_width), col_width)
        body_lines.append(l_text + sep + r_text)

    return header + "\n" + "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Convenience diff view
# ---------------------------------------------------------------------------


def render_diff_side_by_side(
    text_a: str,
    text_b: str,
    title_a: str = "A",
    title_b: str = "B",
    width: int = 80,
) -> str:
    """Convenience function for comparing two text blocks side by side."""
    left = Column(title=title_a, lines=text_a.splitlines())
    right = Column(title=title_b, lines=text_b.splitlines())
    config = SideBySideConfig(total_width=width)
    return render_side_by_side(left, right, config)
