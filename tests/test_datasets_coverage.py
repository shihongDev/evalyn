"""Tests for evalyn_sdk.datasets_coverage: annotation coverage mapping."""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_coverage import (
    CoverageItem,
    CoverageReport,
    CoverageStats,
    compute_coverage,
    compute_metric_coverage,
    find_coverage_gaps,
    prioritize_for_annotation,
    render_coverage_ascii,
    suggest_annotation_targets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _items(*ids: str):
    return [{"id": i} for i in ids]


def _annot(item_id: str, annotator_id: str = "ann1", metric_id: str | None = None):
    d = {"item_id": item_id, "annotator_id": annotator_id}
    if metric_id is not None:
        d["metric_id"] = metric_id
    return d


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_all_annotated(self):
        items = _items("a", "b")
        annots = [_annot("a"), _annot("b")]
        report = compute_coverage(items, annots)
        assert report.stats is not None
        assert report.stats.annotated_items == 2
        assert report.stats.unannotated_items == 0
        assert report.stats.coverage_rate == pytest.approx(1.0)

    def test_none_annotated(self):
        items = _items("a", "b", "c")
        report = compute_coverage(items, [])
        assert report.stats.annotated_items == 0
        assert report.stats.unannotated_items == 3
        assert report.stats.coverage_rate == pytest.approx(0.0)

    def test_partial(self):
        items = _items("a", "b", "c", "d")
        annots = [_annot("a"), _annot("b")]
        report = compute_coverage(items, annots)
        assert report.stats.annotated_items == 2
        assert report.stats.unannotated_items == 2
        assert report.stats.coverage_rate == pytest.approx(0.5)

    def test_multiple_annotators(self):
        items = _items("a")
        annots = [_annot("a", "ann1"), _annot("a", "ann2"), _annot("a", "ann3")]
        report = compute_coverage(items, annots)
        assert report.items[0].annotator_count == 3
        assert report.stats.avg_annotators == pytest.approx(3.0)

    def test_with_metrics(self):
        items = _items("a")
        annots = [
            _annot("a", "ann1", metric_id="accuracy"),
            _annot("a", "ann1", metric_id="fluency"),
        ]
        report = compute_coverage(items, annots)
        assert sorted(report.items[0].metrics_evaluated) == ["accuracy", "fluency"]

    def test_empty_items(self):
        report = compute_coverage([], [])
        assert report.stats.total_items == 0
        assert report.stats.coverage_rate == pytest.approx(0.0)
        assert report.stats.avg_annotators == pytest.approx(0.0)

    def test_gaps_populated(self):
        items = _items("a", "b", "c")
        annots = [_annot("a")]
        report = compute_coverage(items, annots)
        assert set(report.gaps) == {"b", "c"}


# ---------------------------------------------------------------------------
# find_coverage_gaps
# ---------------------------------------------------------------------------


class TestFindCoverageGaps:
    def test_basic(self):
        items = _items("a", "b", "c")
        annots = [_annot("b")]
        report = compute_coverage(items, annots)
        gaps = find_coverage_gaps(report)
        assert set(gaps) == {"a", "c"}

    def test_no_gaps(self):
        items = _items("a", "b")
        annots = [_annot("a"), _annot("b")]
        report = compute_coverage(items, annots)
        gaps = find_coverage_gaps(report)
        assert gaps == []

    def test_all_gaps(self):
        items = _items("x", "y")
        report = compute_coverage(items, [])
        gaps = find_coverage_gaps(report)
        assert set(gaps) == {"x", "y"}


# ---------------------------------------------------------------------------
# compute_metric_coverage
# ---------------------------------------------------------------------------


class TestComputeMetricCoverage:
    def test_basic(self):
        annots = [
            _annot("a", metric_id="acc"),
            _annot("b", metric_id="acc"),
            _annot("a", metric_id="flu"),
        ]
        result = compute_metric_coverage(annots, ["acc", "flu"])
        # 3 unique item_ids: a, b (from annotations with item_id)
        # Wait - all annotations have item_ids. unique item_ids = {a, b} => total = 2
        # acc covers {a, b} => 2/2 = 1.0
        # flu covers {a} => 1/2 = 0.5
        assert result["acc"] == pytest.approx(1.0)
        assert result["flu"] == pytest.approx(0.5)

    def test_empty_metrics(self):
        result = compute_metric_coverage([], [])
        assert result == {}

    def test_no_annotations_for_metric(self):
        annots = [_annot("a", metric_id="acc")]
        result = compute_metric_coverage(annots, ["acc", "missing"])
        assert result["acc"] == pytest.approx(1.0)
        assert result["missing"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# prioritize_for_annotation
# ---------------------------------------------------------------------------


class TestPrioritizeForAnnotation:
    def test_unannotated_first(self):
        items = _items("a", "b", "c")
        annots = [_annot("b", "ann1"), _annot("b", "ann2")]
        report = compute_coverage(items, annots)
        priority = prioritize_for_annotation(report, max_items=2)
        # a and c have 0 annotators, b has 2
        assert len(priority) == 2
        assert "b" not in priority

    def test_max_items(self):
        items = _items("a", "b", "c", "d", "e")
        report = compute_coverage(items, [])
        priority = prioritize_for_annotation(report, max_items=3)
        assert len(priority) == 3

    def test_sorted_by_count(self):
        items = _items("a", "b", "c")
        annots = [
            _annot("a", "ann1"),
            _annot("b", "ann1"),
            _annot("b", "ann2"),
        ]
        report = compute_coverage(items, annots)
        priority = prioritize_for_annotation(report, max_items=3)
        # c has 0, a has 1, b has 2
        assert priority[0] == "c"
        assert priority[1] == "a"
        assert priority[2] == "b"


# ---------------------------------------------------------------------------
# render_coverage_ascii
# ---------------------------------------------------------------------------


class TestRenderCoverageAscii:
    def test_output_contains_bar(self):
        items = _items("a", "b", "c", "d")
        annots = [_annot("a"), _annot("b")]
        report = compute_coverage(items, annots)
        output = render_coverage_ascii(report, width=20)
        assert "[" in output
        assert "]" in output
        assert "#" in output
        assert "." in output
        assert "50.0%" in output

    def test_full_coverage(self):
        items = _items("a", "b")
        annots = [_annot("a"), _annot("b")]
        report = compute_coverage(items, annots)
        output = render_coverage_ascii(report, width=10)
        assert "100.0%" in output

    def test_no_items(self):
        report = compute_coverage([], [])
        output = render_coverage_ascii(report)
        assert output == "[no items]"

    def test_with_metrics(self):
        items = _items("a", "b")
        annots = [_annot("a", metric_id="acc"), _annot("b", metric_id="acc")]
        report = compute_coverage(items, annots)
        output = render_coverage_ascii(report, width=20)
        assert "Per-metric" in output
        assert "acc" in output


# ---------------------------------------------------------------------------
# suggest_annotation_targets
# ---------------------------------------------------------------------------


class TestSuggestAnnotationTargets:
    def test_suggestions_for_unannotated(self):
        items = _items("a", "b", "c")
        annots = [_annot("a")]
        report = compute_coverage(items, annots)
        suggestions = suggest_annotation_targets(report)
        assert any("need first annotation" in s for s in suggestions)

    def test_no_suggestions_when_fully_covered(self):
        items = _items("a")
        annots = [_annot("a", "ann1"), _annot("a", "ann2")]
        report = compute_coverage(items, annots)
        suggestions = suggest_annotation_targets(report)
        # Fully covered, multiple annotators - should have no "need first" suggestion
        assert not any("need first annotation" in s for s in suggestions)

    def test_single_annotator_suggestion(self):
        items = _items("a", "b")
        annots = [_annot("a", "ann1"), _annot("b", "ann1")]
        report = compute_coverage(items, annots)
        suggestions = suggest_annotation_targets(report)
        assert any("only one annotator" in s for s in suggestions)

    def test_empty_report(self):
        report = CoverageReport(stats=None)
        suggestions = suggest_annotation_targets(report)
        assert suggestions == []


# ---------------------------------------------------------------------------
# CoverageStats
# ---------------------------------------------------------------------------


class TestCoverageStats:
    def test_format_text(self):
        stats = CoverageStats(
            total_items=10,
            annotated_items=7,
            unannotated_items=3,
            coverage_rate=0.7,
            avg_annotators=1.5,
            metrics_coverage={"accuracy": 0.9, "fluency": 0.5},
        )
        text = stats.format_text()
        assert "Coverage Stats" in text
        assert "Total items: 10" in text
        assert "Annotated: 7" in text
        assert "70.0%" in text
        assert "accuracy" in text
        assert "fluency" in text

    def test_as_dict(self):
        stats = CoverageStats(total_items=5, annotated_items=3, unannotated_items=2,
                              coverage_rate=0.6, avg_annotators=1.0)
        d = stats.as_dict()
        assert d["total_items"] == 5
        assert d["coverage_rate"] == 0.6


# ---------------------------------------------------------------------------
# CoverageReport
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def test_format_text(self):
        items = _items("a", "b", "c")
        annots = [_annot("a")]
        report = compute_coverage(items, annots)
        text = report.format_text()
        assert "Coverage Report" in text
        assert "Gaps" in text

    def test_as_dict_from_dict_roundtrip(self):
        items = _items("a", "b")
        annots = [_annot("a", metric_id="acc")]
        report = compute_coverage(items, annots)
        d = report.as_dict()
        restored = CoverageReport.from_dict(d)
        assert restored.stats.total_items == report.stats.total_items
        assert restored.stats.annotated_items == report.stats.annotated_items
        assert len(restored.items) == len(report.items)
        assert restored.gaps == report.gaps

    def test_from_dict_empty(self):
        report = CoverageReport.from_dict({})
        assert report.items == []
        assert report.stats is None
        assert report.gaps == []

    def test_from_dict_with_stats(self):
        d = {
            "items": [{"item_id": "x", "has_annotation": True, "annotator_count": 1,
                        "metrics_evaluated": []}],
            "stats": {"total_items": 1, "annotated_items": 1, "unannotated_items": 0,
                       "coverage_rate": 1.0, "avg_annotators": 1.0, "metrics_coverage": {}},
            "gaps": [],
        }
        report = CoverageReport.from_dict(d)
        assert report.stats.total_items == 1
        assert report.items[0].item_id == "x"


# ---------------------------------------------------------------------------
# CoverageItem
# ---------------------------------------------------------------------------


class TestCoverageItem:
    def test_as_dict(self):
        ci = CoverageItem(item_id="abc", has_annotation=True, annotator_count=2,
                          metrics_evaluated=["acc", "flu"])
        d = ci.as_dict()
        assert d["item_id"] == "abc"
        assert d["has_annotation"] is True
        assert d["annotator_count"] == 2
        assert d["metrics_evaluated"] == ["acc", "flu"]

    def test_defaults(self):
        ci = CoverageItem(item_id="x")
        assert ci.has_annotation is False
        assert ci.annotator_count == 0
        assert ci.metrics_evaluated == []
