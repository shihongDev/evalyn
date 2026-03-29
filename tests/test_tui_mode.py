"""Tests for the TUI mode data model and rendering framework."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.tui_mode import (
    TUIConfig,
    TUIPanel,
    TUIState,
    TUIAction,
    render_panel,
    render_layout,
    render_status_bar,
    render_help_bar,
    process_action,
    create_eval_tui,
)


# ---------------------------------------------------------------------------
# TUIConfig
# ---------------------------------------------------------------------------


class TestTUIConfig:
    def test_defaults(self):
        cfg = TUIConfig()
        assert cfg.width == 80
        assert cfg.height == 24
        assert cfg.show_help is True
        assert cfg.color_enabled is True

    def test_custom_values(self):
        cfg = TUIConfig(width=120, height=40, show_help=False, color_enabled=False)
        assert cfg.width == 120
        assert cfg.height == 40
        assert cfg.show_help is False

    def test_as_dict(self):
        cfg = TUIConfig(width=100)
        d = cfg.as_dict()
        assert d["width"] == 100
        assert d["height"] == 24
        assert "show_help" in d
        assert "color_enabled" in d

    def test_from_dict(self):
        d = {"width": 60, "height": 20, "show_help": False, "color_enabled": False}
        cfg = TUIConfig.from_dict(d)
        assert cfg.width == 60
        assert cfg.show_help is False

    def test_from_dict_defaults(self):
        cfg = TUIConfig.from_dict({})
        assert cfg.width == 80
        assert cfg.height == 24

    def test_roundtrip(self):
        cfg = TUIConfig(width=50, height=10, show_help=False, color_enabled=False)
        restored = TUIConfig.from_dict(cfg.as_dict())
        assert restored.width == cfg.width
        assert restored.height == cfg.height
        assert restored.show_help == cfg.show_help
        assert restored.color_enabled == cfg.color_enabled


# ---------------------------------------------------------------------------
# TUIPanel
# ---------------------------------------------------------------------------


class TestTUIPanel:
    def test_basic_creation(self):
        p = TUIPanel(panel_id="a", title="Alpha")
        assert p.panel_id == "a"
        assert p.title == "Alpha"
        assert p.content_lines == []
        assert p.focused is False

    def test_with_content(self):
        lines = ["line 1", "line 2"]
        p = TUIPanel(panel_id="b", title="Beta", content_lines=lines, focused=True)
        assert len(p.content_lines) == 2
        assert p.focused is True

    def test_as_dict(self):
        p = TUIPanel(panel_id="c", title="C", content_lines=["x"], focused=True)
        d = p.as_dict()
        assert d["panel_id"] == "c"
        assert d["focused"] is True
        assert d["content_lines"] == ["x"]

    def test_from_dict(self):
        d = {"panel_id": "d", "title": "D", "content_lines": ["a", "b"], "focused": True}
        p = TUIPanel.from_dict(d)
        assert p.panel_id == "d"
        assert len(p.content_lines) == 2
        assert p.focused is True

    def test_from_dict_minimal(self):
        d = {"panel_id": "e", "title": "E"}
        p = TUIPanel.from_dict(d)
        assert p.content_lines == []
        assert p.focused is False

    def test_roundtrip(self):
        p = TUIPanel(panel_id="f", title="F", content_lines=["1", "2"], focused=True)
        restored = TUIPanel.from_dict(p.as_dict())
        assert restored.panel_id == p.panel_id
        assert restored.content_lines == p.content_lines


# ---------------------------------------------------------------------------
# TUIState
# ---------------------------------------------------------------------------


class TestTUIState:
    def test_defaults(self):
        s = TUIState()
        assert s.panels == []
        assert s.active_panel == 0
        assert s.filter_text == ""
        assert s.status_message == ""

    def test_with_panels(self):
        panels = [TUIPanel(panel_id="a", title="A"), TUIPanel(panel_id="b", title="B")]
        s = TUIState(panels=panels, active_panel=1, status_message="ok")
        assert len(s.panels) == 2
        assert s.active_panel == 1

    def test_as_dict(self):
        panels = [TUIPanel(panel_id="x", title="X")]
        s = TUIState(panels=panels, filter_text="test")
        d = s.as_dict()
        assert len(d["panels"]) == 1
        assert d["filter_text"] == "test"

    def test_from_dict(self):
        d = {
            "panels": [{"panel_id": "p", "title": "P"}],
            "active_panel": 0,
            "filter_text": "q",
            "status_message": "hi",
        }
        s = TUIState.from_dict(d)
        assert len(s.panels) == 1
        assert s.filter_text == "q"

    def test_roundtrip(self):
        panels = [TUIPanel(panel_id="r", title="R", content_lines=["a"])]
        s = TUIState(panels=panels, active_panel=0, filter_text="abc", status_message="msg")
        restored = TUIState.from_dict(s.as_dict())
        assert restored.filter_text == s.filter_text
        assert len(restored.panels) == 1


# ---------------------------------------------------------------------------
# TUIAction
# ---------------------------------------------------------------------------


class TestTUIAction:
    def test_basic(self):
        a = TUIAction(action="quit")
        assert a.action == "quit"
        assert a.value == ""

    def test_with_value(self):
        a = TUIAction(action="filter", value="hello")
        assert a.value == "hello"

    def test_as_dict(self):
        a = TUIAction(action="navigate", value="next")
        d = a.as_dict()
        assert d["action"] == "navigate"
        assert d["value"] == "next"

    def test_from_dict(self):
        a = TUIAction.from_dict({"action": "select", "value": "item_3"})
        assert a.action == "select"
        assert a.value == "item_3"

    def test_roundtrip(self):
        a = TUIAction(action="refresh", value="all")
        restored = TUIAction.from_dict(a.as_dict())
        assert restored.action == a.action
        assert restored.value == a.value


# ---------------------------------------------------------------------------
# render_panel
# ---------------------------------------------------------------------------


class TestRenderPanel:
    def test_empty_panel(self):
        p = TUIPanel(panel_id="e", title="Empty")
        result = render_panel(p, 20, 5)
        lines = result.split("\n")
        assert len(lines) == 5
        assert lines[0].startswith("+")
        assert lines[0].endswith("+")
        assert lines[-1].startswith("+")

    def test_content_rendered(self):
        p = TUIPanel(panel_id="c", title="Content", content_lines=["hello", "world"])
        result = render_panel(p, 20, 5)
        assert "hello" in result
        assert "world" in result

    def test_focused_uses_equals(self):
        p = TUIPanel(panel_id="f", title="Focus", focused=True)
        result = render_panel(p, 20, 5)
        first_line = result.split("\n")[0]
        assert "=" in first_line

    def test_unfocused_uses_dashes(self):
        p = TUIPanel(panel_id="u", title="Unfocused", focused=False)
        result = render_panel(p, 20, 5)
        first_line = result.split("\n")[0]
        assert "-" in first_line

    def test_long_content_truncated(self):
        p = TUIPanel(panel_id="t", title="T", content_lines=["a" * 100])
        result = render_panel(p, 20, 5)
        # Each inner line should be at most 18 chars (20 - 2 for borders)
        for line in result.split("\n"):
            assert len(line) <= 20

    def test_title_centered(self):
        p = TUIPanel(panel_id="t", title="Hi", focused=False)
        result = render_panel(p, 20, 4)
        first_line = result.split("\n")[0]
        assert " Hi " in first_line

    def test_minimum_dimensions(self):
        p = TUIPanel(panel_id="m", title="")
        result = render_panel(p, 2, 2)
        lines = result.split("\n")
        # Should clamp to min 4 width, 3 height
        assert len(lines) == 3

    def test_many_content_lines_exceed_height(self):
        lines = [f"line {i}" for i in range(50)]
        p = TUIPanel(panel_id="s", title="Scroll", content_lines=lines)
        result = render_panel(p, 30, 10)
        rendered_lines = result.split("\n")
        # 10 rows total: top border + 8 content + bottom border
        assert len(rendered_lines) == 10


# ---------------------------------------------------------------------------
# render_status_bar / render_help_bar
# ---------------------------------------------------------------------------


class TestStatusBar:
    def test_empty_message(self):
        bar = render_status_bar("", 40)
        assert len(bar) == 40
        assert bar == "-" * 40

    def test_with_message(self):
        bar = render_status_bar("Ready", 40)
        assert bar.startswith("Ready")
        assert len(bar) == 40

    def test_long_message_truncated(self):
        bar = render_status_bar("x" * 100, 20)
        assert len(bar) == 20


class TestHelpBar:
    def test_contains_shortcuts(self):
        bar = render_help_bar(80)
        assert "[q]uit" in bar
        assert "[f]ilter" in bar
        assert "[n]ext" in bar
        assert "[p]rev" in bar
        assert "[r]efresh" in bar

    def test_padded_to_width(self):
        bar = render_help_bar(80)
        assert len(bar) == 80

    def test_short_width_truncated(self):
        bar = render_help_bar(10)
        assert len(bar) == 10


# ---------------------------------------------------------------------------
# render_layout
# ---------------------------------------------------------------------------


class TestRenderLayout:
    def test_empty_panels(self):
        state = TUIState()
        cfg = TUIConfig(width=40, height=10)
        result = render_layout(state, cfg)
        # Just the status bar
        assert "-" in result

    def test_single_panel(self):
        panels = [TUIPanel(panel_id="a", title="A", content_lines=["hello"])]
        state = TUIState(panels=panels, status_message="ok")
        cfg = TUIConfig(width=40, height=12, show_help=True)
        result = render_layout(state, cfg)
        assert "hello" in result
        assert "[q]uit" in result

    def test_two_panels_side_by_side(self):
        panels = [
            TUIPanel(panel_id="a", title="Left"),
            TUIPanel(panel_id="b", title="Right"),
        ]
        state = TUIState(panels=panels)
        cfg = TUIConfig(width=40, height=12, show_help=False)
        result = render_layout(state, cfg)
        assert "Left" in result
        assert "Right" in result

    def test_three_panels_layout(self):
        panels = [
            TUIPanel(panel_id="a", title="One"),
            TUIPanel(panel_id="b", title="Two"),
            TUIPanel(panel_id="c", title="Three"),
        ]
        state = TUIState(panels=panels)
        cfg = TUIConfig(width=60, height=20, show_help=True)
        result = render_layout(state, cfg)
        assert "One" in result
        assert "Two" in result
        assert "Three" in result

    def test_help_bar_omitted_when_disabled(self):
        panels = [TUIPanel(panel_id="a", title="A")]
        state = TUIState(panels=panels)
        cfg = TUIConfig(width=40, height=10, show_help=False)
        result = render_layout(state, cfg)
        assert "[q]uit" not in result


# ---------------------------------------------------------------------------
# process_action
# ---------------------------------------------------------------------------


class TestProcessAction:
    def _make_state(self, n_panels: int = 3) -> TUIState:
        panels = [TUIPanel(panel_id=f"p{i}", title=f"P{i}") for i in range(n_panels)]
        return TUIState(panels=panels, active_panel=0)

    def test_navigate_next(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="navigate", value="next"))
        assert new.active_panel == 1

    def test_navigate_prev(self):
        state = self._make_state()
        state.active_panel = 0
        new = process_action(state, TUIAction(action="navigate", value="prev"))
        assert new.active_panel == 2  # wraps around

    def test_navigate_wraps_forward(self):
        state = self._make_state()
        state.active_panel = 2
        new = process_action(state, TUIAction(action="navigate", value="next"))
        assert new.active_panel == 0

    def test_filter_set(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="filter", value="test"))
        assert new.filter_text == "test"
        assert "test" in new.status_message.lower() or "filter" in new.status_message.lower()

    def test_filter_clear(self):
        state = self._make_state()
        state.filter_text = "old"
        new = process_action(state, TUIAction(action="filter", value=""))
        assert new.filter_text == ""

    def test_select(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="select", value="item_5"))
        assert "item_5" in new.status_message

    def test_quit(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="quit"))
        assert new.status_message == "quit"

    def test_refresh(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="refresh"))
        assert new.status_message == "refreshed"

    def test_immutable(self):
        state = self._make_state()
        new = process_action(state, TUIAction(action="navigate", value="next"))
        assert state.active_panel == 0
        assert new.active_panel == 1

    def test_navigate_empty_panels(self):
        state = TUIState()
        new = process_action(state, TUIAction(action="navigate", value="next"))
        assert new.active_panel == 0


# ---------------------------------------------------------------------------
# create_eval_tui
# ---------------------------------------------------------------------------


class TestCreateEvalTUI:
    def test_basic_creation(self):
        metrics = {"accuracy": 0.85, "relevance": 0.72}
        items = [{"id": "q1", "score": 0.9}, {"id": "q2", "score": 0.7}]
        run_info = {"model": "gpt-4", "dataset": "test_set"}
        state = create_eval_tui(metrics, items, run_info)
        assert len(state.panels) == 3
        assert state.active_panel == 0
        assert state.status_message == "Ready"

    def test_metrics_panel(self):
        metrics = {"accuracy": 0.85}
        state = create_eval_tui(metrics, [], {})
        panel = state.panels[0]
        assert panel.panel_id == "metrics"
        assert panel.title == "Metrics"
        assert any("accuracy" in line for line in panel.content_lines)
        assert any("0.850" in line for line in panel.content_lines)

    def test_items_panel(self):
        items = [{"id": "item_1", "score": 0.5}]
        state = create_eval_tui({}, items, {})
        panel = state.panels[1]
        assert panel.panel_id == "items"
        assert any("item_1" in line for line in panel.content_lines)

    def test_details_panel(self):
        run_info = {"model": "claude", "version": "3"}
        state = create_eval_tui({}, [], run_info)
        panel = state.panels[2]
        assert panel.panel_id == "details"
        assert any("model" in line for line in panel.content_lines)

    def test_first_panel_focused(self):
        state = create_eval_tui({"a": 1.0}, [], {})
        assert state.panels[0].focused is True
        assert state.panels[1].focused is False
        assert state.panels[2].focused is False

    def test_empty_data(self):
        state = create_eval_tui({}, [], {})
        assert len(state.panels) == 3
        assert state.panels[0].content_lines == []
        assert state.panels[1].content_lines == []
        assert state.panels[2].content_lines == []

    def test_item_without_id_uses_input(self):
        items = [{"input": "hello world"}]
        state = create_eval_tui({}, items, {})
        panel = state.panels[1]
        assert any("hello world" in line for line in panel.content_lines)

    def test_item_without_id_or_input_uses_index(self):
        items = [{"score": 0.5}]
        state = create_eval_tui({}, items, {})
        panel = state.panels[1]
        assert any("item_0" in line for line in panel.content_lines)
