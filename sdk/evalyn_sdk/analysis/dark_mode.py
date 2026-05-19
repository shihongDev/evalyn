"""
Dark mode theme support for HTML evaluation dashboards.

Provides light, dark, and auto (prefers-color-scheme) themes via CSS
custom properties that can be injected into any HTML report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ThemeColors:
    """Complete color palette for a dashboard theme."""
    bg_primary: str
    bg_secondary: str
    text_primary: str
    text_secondary: str
    accent: str
    success: str
    warning: str
    error: str
    border: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "bg_primary": self.bg_primary,
            "bg_secondary": self.bg_secondary,
            "text_primary": self.text_primary,
            "text_secondary": self.text_secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "border": self.border,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> ThemeColors:
        return cls(
            bg_primary=data["bg_primary"],
            bg_secondary=data["bg_secondary"],
            text_primary=data["text_primary"],
            text_secondary=data["text_secondary"],
            accent=data["accent"],
            success=data["success"],
            warning=data["warning"],
            error=data["error"],
            border=data["border"],
        )


@dataclass
class ThemeConfig:
    """Theme configuration: mode plus optional custom overrides."""
    mode: str = "light"  # "light", "dark", or "auto"
    custom_colors: Optional[ThemeColors] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"mode": self.mode}
        if self.custom_colors is not None:
            result["custom_colors"] = self.custom_colors.as_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThemeConfig:
        custom = data.get("custom_colors")
        return cls(
            mode=data.get("mode", "light"),
            custom_colors=ThemeColors.from_dict(custom) if custom else None,
        )


# ---------------------------------------------------------------------------
# Built-in Themes
# ---------------------------------------------------------------------------

LIGHT_THEME = ThemeColors(
    "#FFFBF7", "#FBF7F3", "#1A1A1A", "#555555",
    "#D4A27F", "#6B8E8E", "#D4A27F", "#C97B63", "#E8E0D8",
)

DARK_THEME = ThemeColors(
    "#1A1A2E", "#16213E", "#ECF0F1", "#BDC3C7",
    "#E94560", "#2ECC71", "#F39C12", "#E74C3C", "#2C3E50",
)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_theme_colors(config: ThemeConfig) -> ThemeColors:
    """Return the color palette for the configured mode.

    Args:
        config: Theme configuration.

    Returns:
        ThemeColors for the requested mode. Custom colors take priority.
        "auto" falls back to the light palette (CSS handles runtime switching).
    """
    if config.custom_colors is not None:
        return config.custom_colors
    if config.mode == "dark":
        return DARK_THEME
    # "light" and "auto" both use the light palette as the base
    return LIGHT_THEME


def generate_css_variables(colors: ThemeColors) -> str:
    """Generate CSS custom property declarations from a color palette.

    Args:
        colors: Theme color values.

    Returns:
        A string of CSS custom property declarations (without selector wrapper).
    """
    return (
        f"  --bg-primary: {colors.bg_primary};\n"
        f"  --bg-secondary: {colors.bg_secondary};\n"
        f"  --text-primary: {colors.text_primary};\n"
        f"  --text-secondary: {colors.text_secondary};\n"
        f"  --accent: {colors.accent};\n"
        f"  --success: {colors.success};\n"
        f"  --warning: {colors.warning};\n"
        f"  --error: {colors.error};\n"
        f"  --border: {colors.border};\n"
    )


def generate_theme_css(config: ThemeConfig) -> str:
    """Generate full CSS block including media query for auto mode.

    Args:
        config: Theme configuration.

    Returns:
        Complete CSS string with :root variables and optional
        @media (prefers-color-scheme: dark) block for auto mode.
    """
    base_colors = get_theme_colors(config)
    css_parts: List[str] = []

    css_parts.append(f":root {{\n{generate_css_variables(base_colors)}}}\n")

    # Body defaults that consume the variables
    css_parts.append(
        "body {\n"
        "  background-color: var(--bg-primary);\n"
        "  color: var(--text-primary);\n"
        "}\n"
    )

    if config.mode == "auto":
        dark_vars = generate_css_variables(DARK_THEME)
        css_parts.append(
            f"@media (prefers-color-scheme: dark) {{\n"
            f"  :root {{\n{dark_vars}  }}\n"
            f"}}\n"
        )

    return "\n".join(css_parts)


def apply_theme_to_html(html: str, config: ThemeConfig) -> str:
    """Inject theme CSS into an HTML document.

    The CSS is inserted into the first ``<style>`` tag found, or added inside
    ``<head>`` if no ``<style>`` exists.

    Args:
        html: Source HTML string.
        config: Theme configuration.

    Returns:
        HTML string with theme CSS injected.
    """
    theme_css = generate_theme_css(config)

    # Try inserting after opening <style> tag
    style_idx = html.find("<style>")
    if style_idx != -1:
        insert_at = style_idx + len("<style>")
        return html[:insert_at] + "\n" + theme_css + html[insert_at:]

    # Fall back to inserting before </head>
    head_close = html.find("</head>")
    if head_close != -1:
        return html[:head_close] + f"<style>\n{theme_css}</style>\n" + html[head_close:]

    # Last resort: prepend
    return f"<style>\n{theme_css}</style>\n" + html


def toggle_theme(current: str) -> str:
    """Toggle between light and dark modes.

    Args:
        current: Current mode ("light" or "dark").

    Returns:
        The opposite mode.
    """
    if current == "light":
        return "dark"
    return "light"


def list_available_themes() -> List[str]:
    """Return the list of supported theme mode names.

    Returns:
        ["light", "dark", "auto"]
    """
    return ["light", "dark", "auto"]
