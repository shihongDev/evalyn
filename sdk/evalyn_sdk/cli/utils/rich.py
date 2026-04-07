"""Rich rendering primitives for CLI output.

Provides box-drawing tables, banners, sections, key-value blocks, footers,
progress bars, and semantic icons.  Falls back to ASCII when NO_COLOR is set
or stdout is not a TTY.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

from .colors import _colors_enabled, blue, bold, dim, green, red, yellow

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


def _visible_len(text: str) -> int:
    """Length of *text* ignoring ANSI escape codes."""
    return len(_strip_ansi(text))


def _use_unicode() -> bool:
    """Return True when Unicode box-drawing characters are safe to use."""
    return _colors_enabled()


def _truncate(text: str, max_width: int) -> str:
    """Truncate *text* (ANSI-aware) to *max_width* visible chars."""
    if _visible_len(text) <= max_width:
        return text
    plain = _strip_ansi(text)
    if max_width <= 3:
        return plain[:max_width]
    return plain[: max_width - 3] + "..."


def _pad(text: str, width: int, align: str = "left") -> str:
    """Pad *text* to *width* visible characters respecting alignment."""
    vis = _visible_len(text)
    pad_needed = max(0, width - vis)
    if align == "right":
        return " " * pad_needed + text
    if align == "center":
        left = pad_needed // 2
        right = pad_needed - left
        return " " * left + text + " " * right
    # left (default)
    return text + " " * pad_needed


# ---------------------------------------------------------------------------
# 1. banner
# ---------------------------------------------------------------------------


def banner(title: str, width: int = 60) -> str:
    """Double-line box header.

    Uses Unicode box-drawing characters when color/TTY is available,
    ASCII (+, =, |) otherwise.
    """
    if _use_unicode():
        tl, tr, bl, br = "\u2554", "\u2557", "\u255a", "\u255d"
        h, v = "\u2550", "\u2551"
    else:
        tl, tr, bl, br = "+", "+", "+", "+"
        h, v = "=", "|"

    inner = width - 2  # space between left and right border chars
    top = tl + h * inner + tr
    mid = v + title.center(inner) + v
    bot = bl + h * inner + br
    return "\n".join([top, mid, bot])


# ---------------------------------------------------------------------------
# 2. section
# ---------------------------------------------------------------------------


def section(title: str, width: int = 55) -> str:
    """Thin-line section separator: ``-- TITLE --------``."""
    if _use_unicode():
        line_char = "\u2500"
    else:
        line_char = "-"

    label = f" {title.upper()} "
    prefix = line_char * 2
    remaining = width - len(prefix) - len(label)
    suffix = line_char * max(0, remaining)
    return prefix + label + suffix


# ---------------------------------------------------------------------------
# 3. table
# ---------------------------------------------------------------------------


def table(
    headers: List[str],
    rows: List[List[str]],
    align: Optional[List[str]] = None,
    max_col_width: int = 50,
) -> str:
    """Box-drawing table with auto column widths.

    *align* is a per-column list of ``"left"`` (default), ``"right"``, or
    ``"center"``.  Columns wider than *max_col_width* are truncated.
    """
    if not headers:
        return ""

    ncols = len(headers)
    if align is None:
        align = ["left"] * ncols
    # Extend align list if shorter than headers
    while len(align) < ncols:
        align.append("left")

    # Truncate cell contents
    t_headers = [_truncate(h, max_col_width) for h in headers]
    t_rows = [
        [_truncate(str(cell), max_col_width) for cell in row] + [""] * max(0, ncols - len(row))
        for row in rows
    ]

    # Column widths (visible length)
    col_widths = [_visible_len(h) for h in t_headers]
    for row in t_rows:
        for i in range(ncols):
            if i < len(row):
                col_widths[i] = max(col_widths[i], _visible_len(row[i]))

    # Box characters
    if _use_unicode():
        tl, tr, bl, br = "\u250c", "\u2510", "\u2514", "\u2518"
        h, v = "\u2500", "\u2502"
        t_down, t_up, t_left, t_right = "\u252c", "\u2534", "\u2524", "\u251c"
        cross = "\u253c"
    else:
        tl, tr, bl, br = "+", "+", "+", "+"
        h, v = "-", "|"
        t_down, t_up, t_left, t_right = "+", "+", "+", "+"
        cross = "+"

    def h_line(left: str, mid: str, right: str) -> str:
        segments = [h * (w + 2) for w in col_widths]
        return left + mid.join(segments) + right

    def data_line(cells: List[str]) -> str:
        parts = []
        for i in range(ncols):
            cell = cells[i] if i < len(cells) else ""
            parts.append(" " + _pad(cell, col_widths[i], align[i]) + " ")
        return v + v.join(parts) + v

    lines = [
        h_line(tl, t_down, tr),
        data_line(t_headers),
        h_line(t_right, cross, t_left),
    ]
    for row in t_rows:
        lines.append(data_line(row))
    lines.append(h_line(bl, t_up, br))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. kv
# ---------------------------------------------------------------------------


def kv(pairs: Sequence[Tuple[str, str]], indent: int = 2) -> str:
    """Key-value block with aligned colons.

    Keys are dimmed when colors are enabled.
    """
    if not pairs:
        return ""

    max_key_len = max(_visible_len(k) for k, _ in pairs)
    prefix = " " * indent
    lines: list[str] = []
    for key, value in pairs:
        padded_key = _pad(key, max_key_len)
        display_key = dim(padded_key) if _colors_enabled() else padded_key
        lines.append(f"{prefix}{display_key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. footer
# ---------------------------------------------------------------------------


def footer(
    commands: Sequence[Tuple[str, str]],
    quiet: bool = False,
    format: str = "table",
) -> str:
    """Hint footer with triangle bullet.

    Suppressed when *quiet* is True, *format* is ``"json"``, or the
    ``EVALYN_NO_HINTS`` env var is set.
    """
    if not commands:
        return ""
    if quiet or format == "json":
        return ""
    if os.environ.get("EVALYN_NO_HINTS", "").lower() in ("1", "true"):
        return ""

    bullet = "\u25b8" if _use_unicode() else ">"

    max_cmd = max(len(cmd) for cmd, _ in commands)
    pad = max_cmd + 2
    lines = ["\nNext steps:"]
    for cmd, desc in commands:
        lines.append(f"  {bullet} {cmd:<{pad}} {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. progress_bar
# ---------------------------------------------------------------------------


def progress_bar(
    value: int,
    total: int,
    width: int = 20,
    label: str = "",
) -> str:
    """Return a text progress bar string (does not print).

    Uses U+2588 (full block) and U+2591 (light shade) when Unicode is
    available.  Colors: green >= 80%, yellow >= 50%, red < 50%.
    """
    if total <= 0:
        pct = 0.0
    else:
        pct = value / total

    filled = int(width * pct)
    filled = min(filled, width)

    if _use_unicode():
        fill_char = "\u2588"
        empty_char = "\u2591"
    else:
        fill_char = "#"
        empty_char = "."

    bar_str = fill_char * filled + empty_char * (width - filled)

    # Color the bar
    if _colors_enabled():
        if pct >= 0.8:
            bar_str = green(bar_str)
        elif pct >= 0.5:
            bar_str = yellow(bar_str)
        else:
            bar_str = red(bar_str)

    pct_str = f"{pct * 100:.0f}%"
    parts = []
    if label:
        parts.append(label)
    parts.append(f"[{bar_str}]")
    parts.append(pct_str)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 7. icon / status_icon
# ---------------------------------------------------------------------------

# Unicode icons and their NO_COLOR fallbacks
_ICONS_UNICODE = {
    "pass": ("\u2713", green),   # checkmark
    "fail": ("\u2717", red),     # ballot X
    "warn": ("~", yellow),
    "info": ("\u25cf", blue),    # filled circle
    "next": ("\u25b8", green),   # right triangle
}

_ICONS_ASCII = {
    "pass": "[PASS]",
    "fail": "[FAIL]",
    "warn": "[WARN]",
    "info": "[INFO]",
    "next": "[NEXT]",
}


def icon(kind: str) -> str:
    """Return a semantic icon string.

    Falls back to text labels like ``[PASS]`` when color/TTY is unavailable.
    """
    kind = kind.lower()
    if _use_unicode():
        char, color_fn = _ICONS_UNICODE.get(kind, (kind, lambda x: x))
        return color_fn(char)
    return _ICONS_ASCII.get(kind, f"[{kind.upper()}]")


def status_icon(passed: bool) -> str:
    """Return pass or fail icon based on boolean."""
    return icon("pass" if passed else "fail")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "banner",
    "section",
    "table",
    "kv",
    "footer",
    "progress_bar",
    "icon",
    "status_icon",
]
