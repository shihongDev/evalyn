"""Tests for embeddable_widget module."""

from __future__ import annotations

from evalyn_sdk.embeddable_widget import (
    DEFAULT_CONFIG,
    WidgetConfig,
    WidgetData,
    generate_postmessage_script,
    generate_score_bar_svg,
    generate_widget_html,
)


# ---------------------------------------------------------------------------
# WidgetConfig
# ---------------------------------------------------------------------------


class TestWidgetConfig:
    def test_defaults(self):
        cfg = WidgetConfig()
        assert cfg.width == "100%"
        assert cfg.height == "400px"
        assert cfg.theme == "light"
        assert cfg.show_header is True
        assert cfg.show_footer is True
        assert cfg.charts == ["bar"]

    def test_as_dict(self):
        cfg = WidgetConfig(width="800px", theme="dark")
        d = cfg.as_dict()
        assert d["width"] == "800px"
        assert d["theme"] == "dark"

    def test_from_dict(self):
        d = {
            "width": "50%", "height": "300px", "charts": ["bar", "line"],
            "theme": "dark", "show_header": False, "show_footer": False,
        }
        cfg = WidgetConfig.from_dict(d)
        assert cfg.width == "50%"
        assert cfg.charts == ["bar", "line"]
        assert cfg.show_header is False

    def test_roundtrip(self):
        cfg = WidgetConfig(width="600px", height="500px", theme="dark", charts=["bar"])
        assert WidgetConfig.from_dict(cfg.as_dict()).as_dict() == cfg.as_dict()

    def test_from_dict_defaults(self):
        cfg = WidgetConfig.from_dict({})
        assert cfg.width == "100%"
        assert cfg.theme == "light"


# ---------------------------------------------------------------------------
# WidgetData
# ---------------------------------------------------------------------------


class TestWidgetData:
    def test_create(self):
        data = WidgetData(
            title="Test", metrics={"acc": 0.95},
            labels=["a"], series=[{"name": "s1"}],
            summary_text="All good",
        )
        assert data.title == "Test"
        assert data.metrics == {"acc": 0.95}

    def test_as_dict(self):
        data = WidgetData(title="T", metrics={"x": 1.0})
        d = data.as_dict()
        assert d["title"] == "T"
        assert d["metrics"] == {"x": 1.0}
        assert d["labels"] == []
        assert d["series"] == []
        assert d["summary_text"] == ""

    def test_from_dict(self):
        d = {
            "title": "My Widget",
            "metrics": {"a": 0.5, "b": 0.8},
            "labels": ["l1"],
            "series": [{"name": "s"}],
            "summary_text": "summary",
        }
        data = WidgetData.from_dict(d)
        assert data.title == "My Widget"
        assert data.labels == ["l1"]

    def test_roundtrip(self):
        data = WidgetData(
            title="RT", metrics={"m": 0.3},
            labels=["x"], series=[{"k": "v"}],
            summary_text="text",
        )
        assert WidgetData.from_dict(data.as_dict()).as_dict() == data.as_dict()

    def test_from_dict_defaults(self):
        data = WidgetData.from_dict({"title": "T"})
        assert data.metrics == {}
        assert data.labels == []
        assert data.summary_text == ""


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_is_widget_config(self):
        assert isinstance(DEFAULT_CONFIG, WidgetConfig)

    def test_has_sensible_defaults(self):
        assert DEFAULT_CONFIG.width == "100%"
        assert DEFAULT_CONFIG.theme == "light"
        assert DEFAULT_CONFIG.show_header is True


# ---------------------------------------------------------------------------
# generate_score_bar_svg
# ---------------------------------------------------------------------------


class TestGenerateScoreBarSvg:
    def test_empty_metrics(self):
        svg = generate_score_bar_svg({})
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "rect" not in svg

    def test_single_metric(self):
        svg = generate_score_bar_svg({"accuracy": 0.95})
        assert "<rect" in svg
        assert "accuracy" in svg
        assert "0.95" in svg

    def test_multiple_metrics(self):
        metrics = {"a": 0.5, "b": 0.8, "c": 0.3}
        svg = generate_score_bar_svg(metrics)
        for name in metrics:
            assert name in svg
        assert svg.count("<rect") == 3

    def test_custom_dimensions(self):
        svg = generate_score_bar_svg({"x": 1.0}, width=600, height=300)
        assert 'width="600"' in svg
        assert 'height="300"' in svg

    def test_zero_values(self):
        svg = generate_score_bar_svg({"zero": 0.0})
        assert "<svg" in svg
        assert "zero" in svg

    def test_all_zeros(self):
        svg = generate_score_bar_svg({"a": 0.0, "b": 0.0})
        assert "<svg" in svg

    def test_escapes_html_in_names(self):
        svg = generate_score_bar_svg({"<script>": 0.5})
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg

    def test_has_xmlns(self):
        svg = generate_score_bar_svg({"x": 1.0})
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg


# ---------------------------------------------------------------------------
# generate_postmessage_script
# ---------------------------------------------------------------------------


class TestGeneratePostmessageScript:
    def test_contains_script_tags(self):
        script = generate_postmessage_script()
        assert "<script>" in script
        assert "</script>" in script

    def test_sends_ready_message(self):
        script = generate_postmessage_script()
        assert "evalyn:ready" in script

    def test_listens_for_filter(self):
        script = generate_postmessage_script()
        assert "evalyn:filter" in script

    def test_uses_postmessage(self):
        script = generate_postmessage_script()
        assert "postMessage" in script

    def test_uses_addeventlistener(self):
        script = generate_postmessage_script()
        assert "addEventListener" in script


# ---------------------------------------------------------------------------
# generate_widget_html
# ---------------------------------------------------------------------------


class TestGenerateWidgetHtml:
    def _basic_data(self) -> WidgetData:
        return WidgetData(
            title="Test Dashboard",
            metrics={"accuracy": 0.92, "latency": 250.0},
            summary_text="2 metrics evaluated",
        )

    def test_is_valid_html(self):
        html_str = generate_widget_html(self._basic_data())
        assert "<!DOCTYPE html>" in html_str
        assert "<html" in html_str
        assert "</html>" in html_str
        assert "<head>" in html_str
        assert "</head>" in html_str
        assert "<body" in html_str
        assert "</body>" in html_str

    def test_contains_title(self):
        html_str = generate_widget_html(self._basic_data())
        assert "Test Dashboard" in html_str

    def test_contains_metrics(self):
        html_str = generate_widget_html(self._basic_data())
        assert "accuracy" in html_str
        assert "0.92" in html_str

    def test_contains_summary(self):
        html_str = generate_widget_html(self._basic_data())
        assert "2 metrics evaluated" in html_str

    def test_contains_svg_chart(self):
        html_str = generate_widget_html(self._basic_data())
        assert "<svg" in html_str

    def test_contains_postmessage_script(self):
        html_str = generate_widget_html(self._basic_data())
        assert "evalyn:ready" in html_str

    def test_contains_header_by_default(self):
        html_str = generate_widget_html(self._basic_data())
        assert 'class="header"' in html_str

    def test_contains_footer_by_default(self):
        html_str = generate_widget_html(self._basic_data())
        assert "Powered by Evalyn" in html_str

    def test_no_header_when_disabled(self):
        cfg = WidgetConfig(show_header=False)
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert 'class="header"' not in html_str

    def test_no_footer_when_disabled(self):
        cfg = WidgetConfig(show_footer=False)
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert "Powered by Evalyn" not in html_str

    def test_dark_theme_css(self):
        cfg = WidgetConfig(theme="dark")
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert "#1e1e1e" in html_str

    def test_light_theme_css(self):
        cfg = WidgetConfig(theme="light")
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert "#fff" in html_str

    def test_custom_dimensions(self):
        cfg = WidgetConfig(width="800px", height="600px")
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert "800px" in html_str
        assert "600px" in html_str

    def test_no_chart_when_excluded(self):
        cfg = WidgetConfig(charts=[])
        html_str = generate_widget_html(self._basic_data(), config=cfg)
        assert "<svg" not in html_str

    def test_uses_default_config_when_none(self):
        html_str = generate_widget_html(self._basic_data())
        # Should work fine and use default config
        assert "<!DOCTYPE html>" in html_str

    def test_escapes_title(self):
        data = WidgetData(title="<b>Danger</b>", metrics={})
        html_str = generate_widget_html(data)
        assert "<b>Danger</b>" not in html_str
        assert "&lt;b&gt;Danger&lt;/b&gt;" in html_str

    def test_escapes_summary_text(self):
        data = WidgetData(
            title="T", metrics={},
            summary_text='<img src="x" onerror="alert(1)">',
        )
        html_str = generate_widget_html(data)
        assert 'onerror="alert(1)"' not in html_str

    def test_escapes_metric_names(self):
        data = WidgetData(title="T", metrics={"<script>alert</script>": 0.5})
        html_str = generate_widget_html(data)
        assert "<script>alert</script>" not in html_str

    def test_empty_metrics_no_stats_bar_div(self):
        data = WidgetData(title="T", metrics={})
        html_str = generate_widget_html(data)
        assert 'class="stats-bar">' not in html_str

    def test_has_charset_meta(self):
        html_str = generate_widget_html(self._basic_data())
        assert 'charset="utf-8"' in html_str

    def test_has_viewport_meta(self):
        html_str = generate_widget_html(self._basic_data())
        assert "viewport" in html_str

    def test_no_external_dependencies(self):
        html_str = generate_widget_html(self._basic_data())
        # No link to external CSS or scripts
        assert '<link rel="stylesheet"' not in html_str
        assert 'src="http' not in html_str

    def test_inline_css_present(self):
        html_str = generate_widget_html(self._basic_data())
        assert "<style>" in html_str
        assert "</style>" in html_str

    def test_stats_bar_data_metric_attribute(self):
        html_str = generate_widget_html(self._basic_data())
        assert 'data-metric=' in html_str
