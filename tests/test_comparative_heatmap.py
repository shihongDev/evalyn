"""Tests for comparative heatmap analysis."""

import pytest
from evalyn_sdk.analysis.comparative_heatmap import (
    HeatmapCell,
    HeatmapData,
    HeatmapReport,
    build_item_metric_heatmap,
    build_run_metric_heatmap,
    compute_color_intensity,
    find_hotspots,
    find_coldspots,
    render_heatmap_ascii,
    render_heatmap_html,
)


# =============================================================================
# HeatmapCell
# =============================================================================


class TestHeatmapCell:
    def test_basic(self):
        c = HeatmapCell(row="item1", col="accuracy", value=0.9)
        assert c.row == "item1"
        assert c.col == "accuracy"
        assert c.value == 0.9
        assert c.color_intensity == 0.0

    def test_as_dict(self):
        c = HeatmapCell(row="r", col="c", value=0.5, color_intensity=0.7)
        d = c.as_dict()
        assert d["row"] == "r"
        assert d["value"] == 0.5
        assert d["color_intensity"] == 0.7


# =============================================================================
# HeatmapData
# =============================================================================


class TestHeatmapData:
    def test_empty(self):
        hd = HeatmapData()
        assert hd.cells == []
        assert hd.rows == []
        assert hd.cols == []

    def test_as_dict(self):
        hd = HeatmapData(
            cells=[HeatmapCell(row="a", col="b", value=1.0)],
            rows=["a"],
            cols=["b"],
            title="Test",
        )
        d = hd.as_dict()
        assert len(d["cells"]) == 1
        assert d["title"] == "Test"

    def test_from_dict(self):
        d = {
            "cells": [{"row": "x", "col": "y", "value": 0.5}],
            "rows": ["x"],
            "cols": ["y"],
            "title": "Restored",
        }
        hd = HeatmapData.from_dict(d)
        assert len(hd.cells) == 1
        assert hd.cells[0].row == "x"
        assert hd.title == "Restored"

    def test_from_dict_empty(self):
        hd = HeatmapData.from_dict({})
        assert hd.cells == []
        assert hd.rows == []

    def test_roundtrip(self):
        original = HeatmapData(
            cells=[HeatmapCell(row="r", col="c", value=0.8, color_intensity=0.6)],
            rows=["r"],
            cols=["c"],
            title="RT",
        )
        restored = HeatmapData.from_dict(original.as_dict())
        assert len(restored.cells) == 1
        assert restored.cells[0].value == 0.8
        assert restored.cells[0].color_intensity == 0.6
        assert restored.title == "RT"


# =============================================================================
# HeatmapReport
# =============================================================================


class TestHeatmapReport:
    def test_as_dict(self):
        report = HeatmapReport(
            data=HeatmapData(title="T"),
            hotspots=["a"],
            coldspots=["b"],
        )
        d = report.as_dict()
        assert d["hotspots"] == ["a"]
        assert d["coldspots"] == ["b"]
        assert d["data"]["title"] == "T"

    def test_from_dict(self):
        d = {
            "data": {"cells": [], "rows": [], "cols": [], "title": "X"},
            "hotspots": ["h"],
            "coldspots": ["c"],
        }
        report = HeatmapReport.from_dict(d)
        assert report.hotspots == ["h"]
        assert report.coldspots == ["c"]
        assert report.data.title == "X"

    def test_roundtrip(self):
        original = HeatmapReport(
            data=HeatmapData(rows=["r1"], cols=["c1"], title="Round"),
            hotspots=["r1"],
            coldspots=[],
        )
        restored = HeatmapReport.from_dict(original.as_dict())
        assert restored.data.title == "Round"
        assert restored.hotspots == ["r1"]

    def test_format_text(self):
        report = HeatmapReport(
            data=HeatmapData(
                cells=[HeatmapCell(row="a", col="b", value=0.9)],
                rows=["a"],
                cols=["b"],
                title="My Heatmap",
            ),
            hotspots=["a"],
            coldspots=["z"],
        )
        text = report.format_text()
        assert "My Heatmap" in text
        assert "Hotspots" in text
        assert "Coldspots" in text
        assert "a" in text

    def test_format_text_no_spots(self):
        report = HeatmapReport(data=HeatmapData(rows=["r"], cols=["c"], title="T"))
        text = report.format_text()
        assert "Hotspots" not in text
        assert "Coldspots" not in text


# =============================================================================
# build_item_metric_heatmap
# =============================================================================


class TestBuildItemMetricHeatmap:
    def test_rows_and_cols(self):
        scores = {
            "item1": {"accuracy": 0.9, "fluency": 0.8},
            "item2": {"accuracy": 0.7, "fluency": 0.6},
        }
        hd = build_item_metric_heatmap(scores)
        assert set(hd.rows) == {"item1", "item2"}
        assert set(hd.cols) == {"accuracy", "fluency"}
        assert len(hd.cells) == 4

    def test_empty_scores(self):
        hd = build_item_metric_heatmap({})
        assert hd.cells == []
        assert hd.rows == []

    def test_single_item(self):
        hd = build_item_metric_heatmap({"x": {"m": 0.5}})
        assert hd.rows == ["x"]
        assert hd.cols == ["m"]
        assert len(hd.cells) == 1

    def test_color_intensity_set(self):
        scores = {"a": {"m": 0.0}, "b": {"m": 1.0}}
        hd = build_item_metric_heatmap(scores)
        intensities = {c.row: c.color_intensity for c in hd.cells}
        assert intensities["a"] == 0.0
        assert intensities["b"] == 1.0

    def test_title(self):
        hd = build_item_metric_heatmap({"i": {"m": 0.5}})
        assert "Item" in hd.title


# =============================================================================
# build_run_metric_heatmap
# =============================================================================


class TestBuildRunMetricHeatmap:
    def test_basic(self):
        run_scores = {
            "run1": {"acc": 0.9},
            "run2": {"acc": 0.7},
        }
        hd = build_run_metric_heatmap(run_scores)
        assert set(hd.rows) == {"run1", "run2"}
        assert hd.cols == ["acc"]

    def test_empty(self):
        hd = build_run_metric_heatmap({})
        assert hd.cells == []

    def test_title(self):
        hd = build_run_metric_heatmap({"r": {"m": 0.5}})
        assert "Run" in hd.title


# =============================================================================
# compute_color_intensity
# =============================================================================


class TestComputeColorIntensity:
    def test_mid_range(self):
        assert compute_color_intensity(0.5, 0.0, 1.0) == 0.5

    def test_at_min(self):
        assert compute_color_intensity(0.0, 0.0, 1.0) == 0.0

    def test_at_max(self):
        assert compute_color_intensity(1.0, 0.0, 1.0) == 1.0

    def test_equal_min_max(self):
        result = compute_color_intensity(0.5, 0.5, 0.5)
        assert result == 0.5

    def test_clamp_below(self):
        result = compute_color_intensity(-1.0, 0.0, 1.0)
        assert result == 0.0

    def test_clamp_above(self):
        result = compute_color_intensity(2.0, 0.0, 1.0)
        assert result == 1.0

    def test_custom_range(self):
        result = compute_color_intensity(50.0, 0.0, 100.0)
        assert result == pytest.approx(0.5)


# =============================================================================
# find_hotspots / find_coldspots
# =============================================================================


class TestFindHotspots:
    def test_above_threshold(self):
        scores = {"hot": {"m1": 0.95, "m2": 0.92}, "cold": {"m1": 0.5, "m2": 0.3}}
        hd = build_item_metric_heatmap(scores)
        spots = find_hotspots(hd, threshold=0.9)
        assert "hot" in spots
        assert "cold" not in spots

    def test_empty_data(self):
        hd = HeatmapData()
        assert find_hotspots(hd) == []

    def test_no_hotspots(self):
        scores = {"a": {"m": 0.5}}
        hd = build_item_metric_heatmap(scores)
        assert find_hotspots(hd, threshold=0.9) == []


class TestFindColdspots:
    def test_below_threshold(self):
        scores = {"ok": {"m": 0.8}, "bad": {"m": 0.1}}
        hd = build_item_metric_heatmap(scores)
        spots = find_coldspots(hd, threshold=0.3)
        assert "bad" in spots
        assert "ok" not in spots

    def test_empty_data(self):
        hd = HeatmapData()
        assert find_coldspots(hd) == []

    def test_mixed_values(self):
        scores = {"item": {"m1": 0.9, "m2": 0.1}}
        hd = build_item_metric_heatmap(scores)
        spots = find_coldspots(hd, threshold=0.3)
        assert "item" in spots


# =============================================================================
# render_heatmap_ascii
# =============================================================================


class TestRenderHeatmapAscii:
    def test_output(self):
        scores = {"a": {"x": 0.9, "y": 0.1}}
        hd = build_item_metric_heatmap(scores)
        text = render_heatmap_ascii(hd)
        assert "a" in text
        assert len(text) > 0

    def test_empty(self):
        hd = HeatmapData()
        text = render_heatmap_ascii(hd)
        assert "empty" in text.lower()

    def test_title_shown(self):
        hd = HeatmapData(
            cells=[HeatmapCell(row="r", col="c", value=0.5, color_intensity=0.5)],
            rows=["r"],
            cols=["c"],
            title="MyTitle",
        )
        text = render_heatmap_ascii(hd)
        assert "MyTitle" in text


# =============================================================================
# render_heatmap_html
# =============================================================================


class TestRenderHeatmapHtml:
    def test_has_table(self):
        scores = {"i1": {"m1": 0.8}}
        hd = build_item_metric_heatmap(scores)
        html = render_heatmap_html(hd)
        assert "<table>" in html
        assert "</table>" in html

    def test_empty(self):
        hd = HeatmapData()
        html = render_heatmap_html(hd)
        assert "No heatmap data" in html

    def test_contains_values(self):
        scores = {"i": {"m": 0.75}}
        hd = build_item_metric_heatmap(scores)
        html = render_heatmap_html(hd)
        assert "0.75" in html

    def test_has_background_color(self):
        scores = {"i": {"m": 0.5}}
        hd = build_item_metric_heatmap(scores)
        html = render_heatmap_html(hd)
        assert "background-color" in html

    def test_caption_with_title(self):
        hd = HeatmapData(
            cells=[HeatmapCell(row="r", col="c", value=1.0, color_intensity=1.0)],
            rows=["r"],
            cols=["c"],
            title="Title Here",
        )
        html = render_heatmap_html(hd)
        assert "<caption>" in html
        assert "Title Here" in html
