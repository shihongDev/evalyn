"""Tests for web_dashboard module."""

from __future__ import annotations

from evalyn_sdk.web_dashboard import (
    Dashboard,
    DashboardConfig,
    DashboardPanel,
    build_eval_dashboard,
    generate_chart_panel,
    generate_metrics_panel,
    generate_table_panel,
    render_dashboard,
)


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig()
        assert cfg.title == "Evalyn Dashboard"
        assert cfg.theme == "dark"
        assert cfg.refresh_interval == 0

    def test_custom_values(self):
        cfg = DashboardConfig(title="My Dash", theme="light", refresh_interval=30)
        assert cfg.title == "My Dash"
        assert cfg.theme == "light"
        assert cfg.refresh_interval == 30

    def test_as_dict(self):
        cfg = DashboardConfig(title="Test", theme="dark", refresh_interval=10)
        d = cfg.as_dict()
        assert d["title"] == "Test"
        assert d["theme"] == "dark"
        assert d["refresh_interval"] == 10

    def test_from_dict(self):
        d = {"title": "From Dict", "theme": "light", "refresh_interval": 5}
        cfg = DashboardConfig.from_dict(d)
        assert cfg.title == "From Dict"
        assert cfg.theme == "light"
        assert cfg.refresh_interval == 5

    def test_from_dict_defaults(self):
        cfg = DashboardConfig.from_dict({})
        assert cfg.title == "Evalyn Dashboard"
        assert cfg.theme == "dark"
        assert cfg.refresh_interval == 0

    def test_roundtrip(self):
        cfg = DashboardConfig(title="RT", theme="light", refresh_interval=15)
        restored = DashboardConfig.from_dict(cfg.as_dict())
        assert restored.as_dict() == cfg.as_dict()


# ---------------------------------------------------------------------------
# DashboardPanel
# ---------------------------------------------------------------------------


class TestDashboardPanel:
    def test_defaults(self):
        panel = DashboardPanel(panel_id="p1", title="T", content_html="<p>hi</p>")
        assert panel.panel_type == "metrics"

    def test_custom_type(self):
        panel = DashboardPanel(
            panel_id="p2", title="T2", content_html="<p>x</p>", panel_type="chart"
        )
        assert panel.panel_type == "chart"

    def test_as_dict(self):
        panel = DashboardPanel(
            panel_id="x1", title="Title", content_html="<b>bold</b>", panel_type="table"
        )
        d = panel.as_dict()
        assert d["panel_id"] == "x1"
        assert d["title"] == "Title"
        assert d["content_html"] == "<b>bold</b>"
        assert d["panel_type"] == "table"

    def test_from_dict(self):
        d = {
            "panel_id": "abc",
            "title": "My Panel",
            "content_html": "<div></div>",
            "panel_type": "text",
        }
        panel = DashboardPanel.from_dict(d)
        assert panel.panel_id == "abc"
        assert panel.panel_type == "text"

    def test_from_dict_defaults(self):
        panel = DashboardPanel.from_dict({})
        assert panel.panel_id == ""
        assert panel.panel_type == "metrics"

    def test_roundtrip(self):
        panel = DashboardPanel(
            panel_id="rt1", title="RT", content_html="<p>rt</p>", panel_type="chart"
        )
        restored = DashboardPanel.from_dict(panel.as_dict())
        assert restored.as_dict() == panel.as_dict()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_empty(self):
        dash = Dashboard(config=DashboardConfig())
        assert dash.panels == []

    def test_with_panels(self):
        panels = [
            DashboardPanel(panel_id="p1", title="A", content_html="a"),
            DashboardPanel(panel_id="p2", title="B", content_html="b"),
        ]
        dash = Dashboard(config=DashboardConfig(), panels=panels)
        assert len(dash.panels) == 2

    def test_as_dict(self):
        dash = Dashboard(
            config=DashboardConfig(title="T"),
            panels=[DashboardPanel(panel_id="p1", title="X", content_html="y")],
        )
        d = dash.as_dict()
        assert d["config"]["title"] == "T"
        assert len(d["panels"]) == 1
        assert d["panels"][0]["panel_id"] == "p1"

    def test_from_dict(self):
        d = {
            "config": {"title": "Loaded", "theme": "light"},
            "panels": [
                {"panel_id": "z", "title": "Z", "content_html": "zz", "panel_type": "text"},
            ],
        }
        dash = Dashboard.from_dict(d)
        assert dash.config.title == "Loaded"
        assert dash.config.theme == "light"
        assert len(dash.panels) == 1
        assert dash.panels[0].panel_id == "z"

    def test_roundtrip(self):
        dash = Dashboard(
            config=DashboardConfig(title="RT", theme="light"),
            panels=[
                DashboardPanel(panel_id="a", title="A", content_html="aa", panel_type="chart"),
            ],
        )
        restored = Dashboard.from_dict(dash.as_dict())
        assert restored.as_dict() == dash.as_dict()


# ---------------------------------------------------------------------------
# generate_metrics_panel
# ---------------------------------------------------------------------------


class TestGenerateMetricsPanel:
    def test_basic(self):
        panel = generate_metrics_panel({"accuracy": 0.95, "f1": 0.87})
        assert panel.panel_type == "metrics"
        assert "accuracy" in panel.content_html
        assert "0.9500" in panel.content_html

    def test_empty_metrics(self):
        panel = generate_metrics_panel({})
        assert panel.panel_type == "metrics"
        assert "metrics-grid" in panel.content_html

    def test_html_escape(self):
        panel = generate_metrics_panel({"<script>alert(1)</script>": 0.5})
        assert "<script>" not in panel.content_html
        assert "&lt;script&gt;" in panel.content_html

    def test_high_score_color(self):
        panel = generate_metrics_panel({"acc": 0.95})
        assert "#4caf50" in panel.content_html

    def test_mid_score_color(self):
        panel = generate_metrics_panel({"acc": 0.65})
        assert "#ff9800" in panel.content_html

    def test_low_score_color(self):
        panel = generate_metrics_panel({"acc": 0.2})
        assert "#f44336" in panel.content_html


# ---------------------------------------------------------------------------
# generate_table_panel
# ---------------------------------------------------------------------------


class TestGenerateTablePanel:
    def test_basic(self):
        panel = generate_table_panel("Items", ["ID", "Score"], [["1", "0.9"], ["2", "0.3"]])
        assert panel.panel_type == "table"
        assert panel.title == "Items"
        assert "<th>" in panel.content_html
        assert "<td>" in panel.content_html

    def test_html_escape_headers(self):
        panel = generate_table_panel("T", ["<b>Name</b>"], [["val"]])
        assert "&lt;b&gt;" in panel.content_html
        assert "<b>Name</b>" not in panel.content_html

    def test_html_escape_cells(self):
        panel = generate_table_panel("T", ["Col"], [["<img src=x>"]])
        assert "&lt;img" in panel.content_html

    def test_empty_rows(self):
        panel = generate_table_panel("Empty", ["A", "B"], [])
        assert "<thead>" in panel.content_html
        assert "<tbody></tbody>" in panel.content_html

    def test_multiple_rows(self):
        rows = [["a", "b"], ["c", "d"], ["e", "f"]]
        panel = generate_table_panel("Multi", ["X", "Y"], rows)
        # 3 body rows + 1 header row = 4 total <tr> tags
        assert panel.content_html.count("<tr>") == 4


# ---------------------------------------------------------------------------
# generate_chart_panel
# ---------------------------------------------------------------------------


class TestGenerateChartPanel:
    def test_basic(self):
        panel = generate_chart_panel("Scores", {"a": 0.9, "b": 0.5})
        assert panel.panel_type == "chart"
        assert "<svg" in panel.content_html
        assert "a" in panel.content_html

    def test_empty_data(self):
        panel = generate_chart_panel("Empty", {})
        assert "No data" in panel.content_html

    def test_single_bar(self):
        panel = generate_chart_panel("Single", {"metric": 0.75})
        assert "<rect" in panel.content_html

    def test_html_escape_labels(self):
        panel = generate_chart_panel("XSS", {"<script>": 0.5})
        assert "&lt;script&gt;" in panel.content_html


# ---------------------------------------------------------------------------
# render_dashboard
# ---------------------------------------------------------------------------


class TestRenderDashboard:
    def test_dark_theme(self):
        dash = Dashboard(config=DashboardConfig(theme="dark"))
        html_out = render_dashboard(dash)
        assert "<!DOCTYPE html>" in html_out
        assert "#1e1e2e" in html_out

    def test_light_theme(self):
        dash = Dashboard(config=DashboardConfig(theme="light"))
        html_out = render_dashboard(dash)
        assert "#f5f5f5" in html_out

    def test_title_in_output(self):
        dash = Dashboard(config=DashboardConfig(title="My Results"))
        html_out = render_dashboard(dash)
        assert "My Results" in html_out
        assert "<title>My Results</title>" in html_out

    def test_title_escaped(self):
        dash = Dashboard(config=DashboardConfig(title="<script>xss</script>"))
        html_out = render_dashboard(dash)
        assert "<script>xss</script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_refresh_interval(self):
        dash = Dashboard(config=DashboardConfig(refresh_interval=30))
        html_out = render_dashboard(dash)
        assert 'http-equiv="refresh"' in html_out
        assert "30000" in html_out

    def test_no_refresh_by_default(self):
        dash = Dashboard(config=DashboardConfig())
        html_out = render_dashboard(dash)
        assert "http-equiv" not in html_out

    def test_panels_rendered(self):
        panels = [
            DashboardPanel(panel_id="p1", title="Stats", content_html="<p>data</p>"),
        ]
        dash = Dashboard(config=DashboardConfig(), panels=panels)
        html_out = render_dashboard(dash)
        assert "Stats" in html_out
        assert "<p>data</p>" in html_out

    def test_responsive_grid(self):
        dash = Dashboard(config=DashboardConfig())
        html_out = render_dashboard(dash)
        assert "dashboard-grid" in html_out
        assert "grid-template-columns" in html_out


# ---------------------------------------------------------------------------
# build_eval_dashboard
# ---------------------------------------------------------------------------


class TestBuildEvalDashboard:
    def test_basic(self):
        dash = build_eval_dashboard(
            metrics={"accuracy": 0.9},
            failed_items=[],
            run_info={"run_id": "r1", "dataset": "test-ds"},
        )
        assert isinstance(dash, Dashboard)
        assert "test-ds" in dash.config.title

    def test_includes_metrics_panel(self):
        dash = build_eval_dashboard(
            metrics={"acc": 0.8, "f1": 0.7},
            failed_items=[],
            run_info={},
        )
        types = [p.panel_type for p in dash.panels]
        assert "metrics" in types

    def test_includes_chart_panel(self):
        dash = build_eval_dashboard(
            metrics={"acc": 0.8},
            failed_items=[],
            run_info={},
        )
        types = [p.panel_type for p in dash.panels]
        assert "chart" in types

    def test_includes_failed_items(self):
        dash = build_eval_dashboard(
            metrics={"acc": 0.5},
            failed_items=[{"id": "item-1", "reason": "timeout"}],
            run_info={},
        )
        table_panels = [p for p in dash.panels if p.title == "Failed Items"]
        assert len(table_panels) == 1
        assert "item-1" in table_panels[0].content_html
        assert "timeout" in table_panels[0].content_html

    def test_no_failed_items_panel_when_empty(self):
        dash = build_eval_dashboard(
            metrics={"acc": 0.9},
            failed_items=[],
            run_info={},
        )
        titles = [p.title for p in dash.panels]
        assert "Failed Items" not in titles

    def test_run_info_table(self):
        dash = build_eval_dashboard(
            metrics={},
            failed_items=[],
            run_info={"run_id": "abc", "model": "gpt-4"},
        )
        info_panels = [p for p in dash.panels if p.title == "Run Info"]
        assert len(info_panels) == 1
        assert "abc" in info_panels[0].content_html
        assert "gpt-4" in info_panels[0].content_html

    def test_empty_everything(self):
        dash = build_eval_dashboard(metrics={}, failed_items=[], run_info={})
        assert isinstance(dash, Dashboard)
        assert len(dash.panels) == 0
