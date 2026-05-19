"""
CLI color theme configuration.

User-configurable terminal color schemes using ANSI escape codes.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# ANSI Constants
# ---------------------------------------------------------------------------

ANSI_RESET: str = "\033[0m"

ANSI_COLORS: dict[str, str] = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ColorDef:
    """A named color definition with its ANSI escape code."""

    name: str
    ansi_code: str
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ansi_code": self.ansi_code,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColorDef:
        return cls(
            name=data["name"],
            ansi_code=data["ansi_code"],
            description=data.get("description", ""),
        )


@dataclass
class Theme:
    """A named color theme mapping roles to ANSI codes."""

    name: str
    colors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "colors": dict(self.colors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Theme:
        return cls(
            name=data["name"],
            colors=dict(data.get("colors", {})),
        )


# ---------------------------------------------------------------------------
# Built-in Themes
# ---------------------------------------------------------------------------

DEFAULT_THEME = Theme(
    name="default",
    colors={
        "success": ANSI_COLORS["green"],
        "error": ANSI_COLORS["red"],
        "warning": ANSI_COLORS["yellow"],
        "info": ANSI_COLORS["blue"],
        "header": ANSI_COLORS["bold"],
        "muted": ANSI_COLORS["dim"],
    },
)

DARK_THEME = Theme(
    name="dark",
    colors={
        "success": "\033[92m",   # bright green
        "error": "\033[91m",     # bright red
        "warning": "\033[93m",   # bright yellow
        "info": "\033[94m",      # bright blue
        "header": "\033[1;97m",  # bold bright white
        "muted": "\033[90m",     # dark gray
    },
)

LIGHT_THEME = Theme(
    name="light",
    colors={
        "success": "\033[32m",   # standard green
        "error": "\033[31m",     # standard red
        "warning": "\033[33m",   # standard yellow
        "info": "\033[34m",      # standard blue
        "header": "\033[1;30m",  # bold black
        "muted": "\033[37m",     # light gray
    },
)

NO_COLOR_THEME = Theme(
    name="no_color",
    colors={
        "success": "",
        "error": "",
        "warning": "",
        "info": "",
        "header": "",
        "muted": "",
    },
)

_BUILTIN_THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "no_color": NO_COLOR_THEME,
}


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def colorize(text: str, role: str, theme: Theme | None = None) -> str:
    """Wrap text in ANSI codes for the given role.

    If the NO_COLOR environment variable is set, returns plain text.
    Falls back to DEFAULT_THEME when no theme is provided.
    Returns plain text for unknown roles.
    """
    if os.environ.get("NO_COLOR"):
        return text
    if theme is None:
        theme = DEFAULT_THEME
    code = theme.colors.get(role, "")
    if not code:
        return text
    return f"{code}{text}{ANSI_RESET}"


def get_active_theme(theme_name: str = "") -> Theme:
    """Look up a theme by name.

    Respects the NO_COLOR environment variable - returns NO_COLOR_THEME
    when the variable is set regardless of the requested name.
    Falls back to DEFAULT_THEME for unknown names.
    """
    if os.environ.get("NO_COLOR"):
        return NO_COLOR_THEME
    if not theme_name:
        return DEFAULT_THEME
    return _BUILTIN_THEMES.get(theme_name, DEFAULT_THEME)


def load_theme_from_config(config: dict[str, Any]) -> Theme:
    """Load a theme from a configuration dict.

    Expected format: {"name": "my_theme", "colors": {"role": "ansi_code", ...}}
    Missing fields receive sensible defaults.
    """
    name = config.get("name", "custom")
    colors = config.get("colors", {})
    return Theme(name=name, colors=dict(colors))


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape codes from text."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def format_theme_preview(theme: Theme) -> str:
    """Show each role rendered in its assigned color.

    Returns a multi-line string with one role per line.
    """
    lines: list[str] = [f"Theme: {theme.name}"]
    for role, code in sorted(theme.colors.items()):
        if code:
            lines.append(f"  {code}{role}{ANSI_RESET}")
        else:
            lines.append(f"  {role}")
    return "\n".join(lines)
