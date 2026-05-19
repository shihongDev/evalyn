"""Dashboard theming for HTML evaluation reports.

Provides configurable color palettes and CSS variable sets for
html_report.py. Ships five built-in themes and supports custom
definitions loaded from evalyn.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Theme:
    """Color and style definitions for an HTML report."""

    name: str
    description: str = ""

    # Background colors
    bg_primary: str = "#FFFBF7"
    bg_secondary: str = "#FBF7F3"
    bg_tertiary: str = "#F5EDE6"
    bg_hover: str = "#F0E8E0"

    # Accent colors
    accent_primary: str = "#D4A27F"
    accent_secondary: str = "#C4836A"
    accent_muted: str = "#A89F97"

    # Status colors
    status_pass: str = "#6B8E8E"
    status_fail: str = "#C97B63"
    status_warn: str = "#D4A27F"

    # Text colors
    text_primary: str = "#1A1A1A"
    text_secondary: str = "#555555"
    text_muted: str = "#888888"

    # Border
    border_light: str = "#E8E0D8"
    border_strong: str = "#D4CCC4"

    # Chart palette (ordered list for multi-series charts)
    chart_palette: List[str] = field(default_factory=lambda: [
        "#D4A27F", "#6B8E8E", "#9B8AA6", "#C4836A",
        "#A89F97", "#8B7355", "#7BA3A3", "#B39DBA",
    ])

    # Custom CSS to inject after theme variables
    custom_css: str = ""

    def css_variables(self) -> str:
        """Generate CSS custom property declarations."""
        lines = [
            f"--bg-primary: {self.bg_primary};",
            f"--bg-secondary: {self.bg_secondary};",
            f"--bg-tertiary: {self.bg_tertiary};",
            f"--bg-hover: {self.bg_hover};",
            f"--accent-primary: {self.accent_primary};",
            f"--accent-secondary: {self.accent_secondary};",
            f"--accent-muted: {self.accent_muted};",
            f"--status-pass: {self.status_pass};",
            f"--status-fail: {self.status_fail};",
            f"--status-warn: {self.status_warn};",
            f"--text-primary: {self.text_primary};",
            f"--text-secondary: {self.text_secondary};",
            f"--text-muted: {self.text_muted};",
            f"--border-light: {self.border_light};",
            f"--border-strong: {self.border_strong};",
        ]
        return "\n            ".join(lines)

    def chart_js_colors(self) -> Dict[str, str]:
        """Color map used by Chart.js rendering in html_report."""
        return {
            "accent": self.accent_primary,
            "accentDim": self.accent_secondary,
            "secondary": self.accent_secondary,
            "error": self.status_fail,
            "border": self.border_light,
            "warning": self.status_warn,
            "success": self.status_pass,
            "muted": self.accent_muted,
            "gridLine": self.border_light,
            "bg": self.bg_primary,
            "text": self.text_primary,
            "textMuted": self.text_muted,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "bg_primary": self.bg_primary,
            "bg_secondary": self.bg_secondary,
            "bg_tertiary": self.bg_tertiary,
            "bg_hover": self.bg_hover,
            "accent_primary": self.accent_primary,
            "accent_secondary": self.accent_secondary,
            "accent_muted": self.accent_muted,
            "status_pass": self.status_pass,
            "status_fail": self.status_fail,
            "status_warn": self.status_warn,
            "text_primary": self.text_primary,
            "text_secondary": self.text_secondary,
            "text_muted": self.text_muted,
            "border_light": self.border_light,
            "border_strong": self.border_strong,
            "chart_palette": list(self.chart_palette),
            "custom_css": self.custom_css,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Theme:
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            bg_primary=data.get("bg_primary", "#FFFBF7"),
            bg_secondary=data.get("bg_secondary", "#FBF7F3"),
            bg_tertiary=data.get("bg_tertiary", "#F5EDE6"),
            bg_hover=data.get("bg_hover", "#F0E8E0"),
            accent_primary=data.get("accent_primary", "#D4A27F"),
            accent_secondary=data.get("accent_secondary", "#C4836A"),
            accent_muted=data.get("accent_muted", "#A89F97"),
            status_pass=data.get("status_pass", "#6B8E8E"),
            status_fail=data.get("status_fail", "#C97B63"),
            status_warn=data.get("status_warn", "#D4A27F"),
            text_primary=data.get("text_primary", "#1A1A1A"),
            text_secondary=data.get("text_secondary", "#555555"),
            text_muted=data.get("text_muted", "#888888"),
            border_light=data.get("border_light", "#E8E0D8"),
            border_strong=data.get("border_strong", "#D4CCC4"),
            chart_palette=data.get("chart_palette") or [
                "#D4A27F", "#6B8E8E", "#9B8AA6", "#C4836A",
                "#A89F97", "#8B7355", "#7BA3A3", "#B39DBA",
            ],
            custom_css=data.get("custom_css", ""),
        )


# ---- Built-in themes ----

CORPORATE = Theme(
    name="corporate",
    description="Clean professional look with blue accents",
    bg_primary="#FFFFFF",
    bg_secondary="#F8F9FA",
    bg_tertiary="#E9ECEF",
    bg_hover="#DEE2E6",
    accent_primary="#0D6EFD",
    accent_secondary="#0B5ED7",
    accent_muted="#6C757D",
    status_pass="#198754",
    status_fail="#DC3545",
    status_warn="#FFC107",
    text_primary="#212529",
    text_secondary="#495057",
    text_muted="#6C757D",
    border_light="#DEE2E6",
    border_strong="#CED4DA",
    chart_palette=[
        "#0D6EFD", "#198754", "#FFC107", "#DC3545",
        "#6F42C1", "#0DCAF0", "#FD7E14", "#20C997",
    ],
)

ACADEMIC = Theme(
    name="academic",
    description="Muted tones suited for papers and presentations",
    bg_primary="#FAFAFA",
    bg_secondary="#F5F5F5",
    bg_tertiary="#EEEEEE",
    bg_hover="#E0E0E0",
    accent_primary="#37474F",
    accent_secondary="#546E7A",
    accent_muted="#90A4AE",
    status_pass="#2E7D32",
    status_fail="#C62828",
    status_warn="#F9A825",
    text_primary="#212121",
    text_secondary="#424242",
    text_muted="#757575",
    border_light="#E0E0E0",
    border_strong="#BDBDBD",
    chart_palette=[
        "#37474F", "#2E7D32", "#C62828", "#F9A825",
        "#1565C0", "#6A1B9A", "#00838F", "#EF6C00",
    ],
)

DARK_MODE = Theme(
    name="dark-mode",
    description="Dark background for low-light viewing",
    bg_primary="#1A1A2E",
    bg_secondary="#16213E",
    bg_tertiary="#0F3460",
    bg_hover="#1A3A5C",
    accent_primary="#E94560",
    accent_secondary="#C23152",
    accent_muted="#7F8C8D",
    status_pass="#2ECC71",
    status_fail="#E74C3C",
    status_warn="#F39C12",
    text_primary="#ECF0F1",
    text_secondary="#BDC3C7",
    text_muted="#7F8C8D",
    border_light="#2C3E50",
    border_strong="#34495E",
    chart_palette=[
        "#E94560", "#2ECC71", "#3498DB", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#E74C3C",
    ],
)

PRINT_FRIENDLY = Theme(
    name="print-friendly",
    description="High contrast for printing on paper",
    bg_primary="#FFFFFF",
    bg_secondary="#FFFFFF",
    bg_tertiary="#F5F5F5",
    bg_hover="#EEEEEE",
    accent_primary="#333333",
    accent_secondary="#555555",
    accent_muted="#999999",
    status_pass="#006400",
    status_fail="#8B0000",
    status_warn="#8B8000",
    text_primary="#000000",
    text_secondary="#333333",
    text_muted="#666666",
    border_light="#CCCCCC",
    border_strong="#999999",
    chart_palette=[
        "#333333", "#006400", "#8B0000", "#8B8000",
        "#00008B", "#4B0082", "#006666", "#8B4513",
    ],
)

# Default theme matches the existing html_report.py colors
DEFAULT = Theme(
    name="default",
    description="Warm cream with terracotta accents (original evalyn style)",
)

BUILTIN_THEMES: Dict[str, Theme] = {
    "default": DEFAULT,
    "corporate": CORPORATE,
    "academic": ACADEMIC,
    "dark-mode": DARK_MODE,
    "print-friendly": PRINT_FRIENDLY,
}


def get_theme(
    name: str,
    custom_themes: Optional[Dict[str, Any]] = None,
) -> Theme:
    """Look up a theme by name.

    Checks custom_themes first (from evalyn.yaml), then built-ins.

    Args:
        name: Theme name.
        custom_themes: Dict of name -> theme config from evalyn.yaml.

    Returns:
        Theme instance.

    Raises:
        ValueError: If name not found.
    """
    if custom_themes and name in custom_themes:
        data = dict(custom_themes[name])
        data.setdefault("name", name)
        return Theme.from_dict(data)

    if name in BUILTIN_THEMES:
        return BUILTIN_THEMES[name]

    available = sorted(set(BUILTIN_THEMES) | set(custom_themes or {}))
    raise ValueError(
        f"Unknown theme '{name}'. Available: {', '.join(available)}"
    )


def list_themes(
    custom_themes: Optional[Dict[str, Any]] = None,
) -> List[Theme]:
    """Return all available themes (built-in + custom)."""
    themes = dict(BUILTIN_THEMES)
    if custom_themes:
        for name, data in custom_themes.items():
            d = dict(data)
            d.setdefault("name", name)
            themes[name] = Theme.from_dict(d)
    return [themes[k] for k in sorted(themes)]
