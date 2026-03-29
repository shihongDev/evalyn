"""
TUI data model and rendering framework.

Pure-Python, ASCII-based terminal UI with no external dependencies.
Provides dataclasses for TUI state management and functions for
rendering panels, layouts, status bars, and processing user actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class TUIConfig:
    """Display configuration for the TUI."""

    width: int = 80
    height: int = 24
    show_help: bool = True
    color_enabled: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "show_help": self.show_help,
            "color_enabled": self.color_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TUIConfig:
        return cls(
            width=data.get("width", 80),
            height=data.get("height", 24),
            show_help=data.get("show_help", True),
            color_enabled=data.get("color_enabled", True),
        )


@dataclass
class TUIPanel:
    """A single panel with title and content lines."""

    panel_id: str
    title: str
    content_lines: List[str] = field(default_factory=list)
    focused: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "content_lines": list(self.content_lines),
            "focused": self.focused,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TUIPanel:
        return cls(
            panel_id=data["panel_id"],
            title=data["title"],
            content_lines=list(data.get("content_lines", [])),
            focused=data.get("focused", False),
        )


@dataclass
class TUIState:
    """Full TUI state: panels, active selection, filter, status."""

    panels: List[TUIPanel] = field(default_factory=list)
    active_panel: int = 0
    filter_text: str = ""
    status_message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "panels": [p.as_dict() for p in self.panels],
            "active_panel": self.active_panel,
            "filter_text": self.filter_text,
            "status_message": self.status_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TUIState:
        return cls(
            panels=[TUIPanel.from_dict(p) for p in data.get("panels", [])],
            active_panel=data.get("active_panel", 0),
            filter_text=data.get("filter_text", ""),
            status_message=data.get("status_message", ""),
        )


@dataclass
class TUIAction:
    """A user action in the TUI."""

    action: str  # "navigate", "filter", "select", "quit", "refresh"
    value: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TUIAction:
        return cls(
            action=data["action"],
            value=data.get("value", ""),
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_panel(panel: TUIPanel, width: int, height: int) -> str:
    """Render a single panel with ASCII box border, title, and scrollable content.

    The panel occupies exactly *width* columns and *height* rows (including
    the top/bottom borders).  Content lines are truncated or padded to fit.
    """
    if width < 4:
        width = 4
    if height < 3:
        height = 3

    inner_w = width - 2  # space inside the vertical bars
    content_rows = height - 2  # rows available for content

    # Build title bar
    border_char = "=" if panel.focused else "-"
    if panel.title:
        title_display = f" {panel.title} "
        if len(title_display) > inner_w:
            title_display = title_display[: inner_w]
        left_pad = (inner_w - len(title_display)) // 2
        right_pad = inner_w - len(title_display) - left_pad
        top = "+" + border_char * left_pad + title_display + border_char * right_pad + "+"
    else:
        top = "+" + border_char * inner_w + "+"

    bottom = "+" + border_char * inner_w + "+"

    lines: List[str] = [top]
    for i in range(content_rows):
        if i < len(panel.content_lines):
            text = panel.content_lines[i]
            if len(text) > inner_w:
                text = text[: inner_w - 1] + "~"
            text = text.ljust(inner_w)
        else:
            text = " " * inner_w
        lines.append("|" + text + "|")
    lines.append(bottom)
    return "\n".join(lines)


def render_layout(state: TUIState, config: TUIConfig) -> str:
    """Render the full TUI layout.

    When there are 1-2 panels they are placed side by side.
    When there are 3+ panels, the first two share the top row and
    remaining panels are stacked below.  The status bar and optional
    help bar are appended at the bottom.
    """
    if not state.panels:
        return render_status_bar(state.status_message, config.width)

    # Mark focused panel
    panels = []
    for idx, p in enumerate(state.panels):
        copy = TUIPanel(
            panel_id=p.panel_id,
            title=p.title,
            content_lines=list(p.content_lines),
            focused=(idx == state.active_panel),
        )
        panels.append(copy)

    reserved = 1  # status bar
    if config.show_help:
        reserved += 1

    available_height = config.height - reserved

    output_lines: List[str] = []

    if len(panels) <= 2:
        # Side by side
        panel_w = config.width // len(panels)
        panel_h = max(available_height, 3)
        rendered = [render_panel(p, panel_w, panel_h) for p in panels]
        # Merge side by side
        split = [r.split("\n") for r in rendered]
        max_rows = max(len(s) for s in split)
        for row_idx in range(max_rows):
            row_parts = []
            for s in split:
                if row_idx < len(s):
                    row_parts.append(s[row_idx])
                else:
                    row_parts.append(" " * panel_w)
            output_lines.append("".join(row_parts))
    else:
        # First two side by side, rest stacked
        top_w = config.width // 2
        top_h = max(available_height // 2, 3)
        top_rendered = [render_panel(p, top_w, top_h) for p in panels[:2]]
        top_split = [r.split("\n") for r in top_rendered]
        max_rows = max(len(s) for s in top_split)
        for row_idx in range(max_rows):
            row_parts = []
            for s in top_split:
                if row_idx < len(s):
                    row_parts.append(s[row_idx])
                else:
                    row_parts.append(" " * top_w)
            output_lines.append("".join(row_parts))

        remaining = panels[2:]
        remaining_h = max((available_height - top_h) // len(remaining), 3)
        for p in remaining:
            rendered = render_panel(p, config.width, remaining_h)
            output_lines.extend(rendered.split("\n"))

    # Status bar
    output_lines.append(render_status_bar(state.status_message, config.width))

    # Help bar
    if config.show_help:
        output_lines.append(render_help_bar(config.width))

    return "\n".join(output_lines)


def render_status_bar(message: str, width: int) -> str:
    """Bottom status bar - inverse-style with the message left-aligned."""
    if width < 1:
        width = 1
    text = message[:width] if message else ""
    return text.ljust(width, "-")


def render_help_bar(width: int) -> str:
    """Render the help bar showing available keyboard shortcuts."""
    if width < 1:
        width = 1
    help_text = "[q]uit [f]ilter [n]ext [p]rev [r]efresh"
    if len(help_text) > width:
        help_text = help_text[:width]
    return help_text.ljust(width)


# ---------------------------------------------------------------------------
# Action Processing
# ---------------------------------------------------------------------------


def process_action(state: TUIState, action: TUIAction) -> TUIState:
    """Apply an action to the TUI state and return a new state.

    Actions:
      navigate - value "next" or "prev" to move between panels
      filter   - set filter_text to action.value
      select   - set status_message acknowledging the selection
      quit     - set status_message to "quit"
      refresh  - set status_message to "refreshed"
    """
    new_panels = [
        TUIPanel(
            panel_id=p.panel_id,
            title=p.title,
            content_lines=list(p.content_lines),
            focused=p.focused,
        )
        for p in state.panels
    ]
    new_state = TUIState(
        panels=new_panels,
        active_panel=state.active_panel,
        filter_text=state.filter_text,
        status_message=state.status_message,
    )

    if action.action == "navigate":
        n = len(new_state.panels)
        if n > 0:
            if action.value == "next":
                new_state.active_panel = (state.active_panel + 1) % n
            elif action.value == "prev":
                new_state.active_panel = (state.active_panel - 1) % n
            new_state.status_message = f"Panel {new_state.active_panel}"

    elif action.action == "filter":
        new_state.filter_text = action.value
        new_state.status_message = f"Filter: {action.value}" if action.value else "Filter cleared"

    elif action.action == "select":
        new_state.status_message = f"Selected: {action.value}" if action.value else "Selected"

    elif action.action == "quit":
        new_state.status_message = "quit"

    elif action.action == "refresh":
        new_state.status_message = "refreshed"

    return new_state


# ---------------------------------------------------------------------------
# Eval TUI Factory
# ---------------------------------------------------------------------------


def create_eval_tui(
    metrics: Dict[str, float],
    items: List[Dict[str, Any]],
    run_info: Dict[str, Any],
) -> TUIState:
    """Build a TUI state from evaluation data.

    Creates three panels:
      1. Metrics overview - one line per metric with score
      2. Items list - one line per item with summary
      3. Run details - key/value pairs from run_info
    """
    # Metrics panel
    metric_lines: List[str] = []
    for name, score in sorted(metrics.items()):
        metric_lines.append(f"  {name}: {score:.3f}")
    metrics_panel = TUIPanel(
        panel_id="metrics",
        title="Metrics",
        content_lines=metric_lines,
        focused=True,
    )

    # Items panel
    item_lines: List[str] = []
    for idx, item in enumerate(items):
        label = item.get("id", item.get("input", f"item_{idx}"))
        score_val = item.get("score", "")
        if isinstance(score_val, float):
            item_lines.append(f"  {label}: {score_val:.3f}")
        else:
            item_lines.append(f"  {label}: {score_val}")
    items_panel = TUIPanel(
        panel_id="items",
        title="Items",
        content_lines=item_lines,
    )

    # Details panel
    detail_lines: List[str] = []
    for key, val in sorted(run_info.items()):
        detail_lines.append(f"  {key}: {val}")
    details_panel = TUIPanel(
        panel_id="details",
        title="Details",
        content_lines=detail_lines,
    )

    return TUIState(
        panels=[metrics_panel, items_panel, details_panel],
        active_panel=0,
        status_message="Ready",
    )
