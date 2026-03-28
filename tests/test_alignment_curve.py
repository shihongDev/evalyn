"""Tests for calibration alignment curve."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.alignment_curve import (
    AlignmentCurve,
    AlignmentPoint,
    build_alignment_curve,
    compute_roi,
    estimate_optimal_annotations,
    render_ascii_curve,
)


# -------------------------------------------------------------------
# AlignmentPoint
# -------------------------------------------------------------------

class TestAlignmentPoint:
    def test_defaults(self):
        p = AlignmentPoint(num_annotations=10, alignment_score=0.8)
        assert p.marginal_improvement == 0.0

    def test_as_dict(self):
        p = AlignmentPoint(num_annotations=5, alignment_score=0.6, marginal_improvement=0.1)
        d = p.as_dict()
        assert d["num_annotations"] == 5
        assert d["alignment_score"] == 0.6
        assert d["marginal_improvement"] == 0.1

    def test_as_dict_keys(self):
        p = AlignmentPoint(num_annotations=1, alignment_score=0.5)
        d = p.as_dict()
        assert set(d.keys()) == {"num_annotations", "alignment_score", "marginal_improvement"}


# -------------------------------------------------------------------
# AlignmentCurve - properties
# -------------------------------------------------------------------

class TestAlignmentCurveProperties:
    def test_max_alignment(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5),
                AlignmentPoint(num_annotations=10, alignment_score=0.9),
                AlignmentPoint(num_annotations=15, alignment_score=0.7),
            ]
        )
        assert curve.max_alignment == 0.9

    def test_max_alignment_empty(self):
        curve = AlignmentCurve()
        assert curve.max_alignment == 0.0

    def test_total_improvement(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.4),
                AlignmentPoint(num_annotations=10, alignment_score=0.6),
                AlignmentPoint(num_annotations=15, alignment_score=0.8),
            ]
        )
        assert abs(curve.total_improvement - 0.4) < 1e-9

    def test_total_improvement_single_point(self):
        curve = AlignmentCurve(
            points=[AlignmentPoint(num_annotations=5, alignment_score=0.5)]
        )
        assert curve.total_improvement == 0.0

    def test_total_improvement_empty(self):
        curve = AlignmentCurve()
        assert curve.total_improvement == 0.0

    def test_format_text_with_diminishing(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5),
                AlignmentPoint(num_annotations=10, alignment_score=0.8),
            ],
            diminishing_returns_at=10,
        )
        text = curve.format_text()
        assert "10 annotations" in text
        assert "2 points" in text

    def test_format_text_no_diminishing(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5),
                AlignmentPoint(num_annotations=10, alignment_score=0.8),
            ],
        )
        text = curve.format_text()
        assert "not reached" in text


# -------------------------------------------------------------------
# AlignmentCurve - as_dict / from_dict
# -------------------------------------------------------------------

class TestAlignmentCurveSerialize:
    def test_as_dict(self):
        curve = AlignmentCurve(
            points=[AlignmentPoint(num_annotations=5, alignment_score=0.6)],
            diminishing_returns_at=5,
        )
        d = curve.as_dict()
        assert len(d["points"]) == 1
        assert d["diminishing_returns_at"] == 5

    def test_from_dict_roundtrip(self):
        original = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5, marginal_improvement=0.0),
                AlignmentPoint(num_annotations=10, alignment_score=0.7, marginal_improvement=0.2),
            ],
            diminishing_returns_at=10,
        )
        d = original.as_dict()
        restored = AlignmentCurve.from_dict(d)
        assert len(restored.points) == 2
        assert restored.points[0].num_annotations == 5
        assert restored.points[1].alignment_score == 0.7
        assert restored.points[1].marginal_improvement == 0.2
        assert restored.diminishing_returns_at == 10

    def test_from_dict_empty(self):
        restored = AlignmentCurve.from_dict({})
        assert restored.points == []
        assert restored.diminishing_returns_at is None

    def test_from_dict_no_diminishing(self):
        restored = AlignmentCurve.from_dict({
            "points": [{"num_annotations": 5, "alignment_score": 0.5}],
        })
        assert len(restored.points) == 1
        assert restored.diminishing_returns_at is None


# -------------------------------------------------------------------
# build_alignment_curve
# -------------------------------------------------------------------

class TestBuildAlignmentCurve:
    def test_basic(self):
        pairs = [(5, 0.5), (10, 0.7), (15, 0.85)]
        curve = build_alignment_curve(pairs)
        assert len(curve.points) == 3
        assert curve.points[0].num_annotations == 5
        assert curve.points[1].num_annotations == 10
        assert curve.points[2].num_annotations == 15

    def test_marginal_improvement_computed(self):
        pairs = [(5, 0.5), (10, 0.7), (15, 0.85)]
        curve = build_alignment_curve(pairs)
        assert curve.points[0].marginal_improvement == 0.0
        assert abs(curve.points[1].marginal_improvement - 0.2) < 1e-9
        assert abs(curve.points[2].marginal_improvement - 0.15) < 1e-9

    def test_diminishing_returns_detected(self):
        # Improvement drops below 0.01 at the third pair
        pairs = [(5, 0.5), (10, 0.7), (15, 0.705), (20, 0.708)]
        curve = build_alignment_curve(pairs)
        assert curve.diminishing_returns_at == 15

    def test_no_diminishing_returns_all_improving(self):
        pairs = [(5, 0.3), (10, 0.5), (15, 0.7), (20, 0.9)]
        curve = build_alignment_curve(pairs)
        assert curve.diminishing_returns_at is None

    def test_empty_input(self):
        curve = build_alignment_curve([])
        assert len(curve.points) == 0
        assert curve.diminishing_returns_at is None

    def test_single_pair(self):
        curve = build_alignment_curve([(10, 0.8)])
        assert len(curve.points) == 1
        assert curve.points[0].marginal_improvement == 0.0
        assert curve.diminishing_returns_at is None

    def test_sorts_by_count(self):
        pairs = [(20, 0.9), (5, 0.3), (10, 0.6)]
        curve = build_alignment_curve(pairs)
        assert curve.points[0].num_annotations == 5
        assert curve.points[1].num_annotations == 10
        assert curve.points[2].num_annotations == 20


# -------------------------------------------------------------------
# render_ascii_curve
# -------------------------------------------------------------------

class TestRenderAsciiCurve:
    def test_produces_output(self):
        pairs = [(5, 0.5), (10, 0.7), (15, 0.85)]
        curve = build_alignment_curve(pairs)
        output = render_ascii_curve(curve)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "#" in output

    def test_contains_point_count(self):
        pairs = [(5, 0.5), (10, 0.7)]
        curve = build_alignment_curve(pairs)
        output = render_ascii_curve(curve)
        assert "2 points" in output

    def test_empty_curve(self):
        curve = AlignmentCurve()
        output = render_ascii_curve(curve)
        assert "no data" in output.lower()

    def test_single_point(self):
        curve = build_alignment_curve([(10, 0.5)])
        output = render_ascii_curve(curve)
        assert "#" in output

    def test_custom_dimensions(self):
        pairs = [(i * 5, 0.3 + i * 0.1) for i in range(5)]
        curve = build_alignment_curve(pairs)
        output = render_ascii_curve(curve, width=30, height=10)
        assert isinstance(output, str)
        assert "#" in output


# -------------------------------------------------------------------
# estimate_optimal_annotations
# -------------------------------------------------------------------

class TestEstimateOptimalAnnotations:
    def test_returns_diminishing_point(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5),
                AlignmentPoint(num_annotations=10, alignment_score=0.7),
            ],
            diminishing_returns_at=10,
        )
        assert estimate_optimal_annotations(curve) == 10

    def test_returns_max_if_no_diminishing(self):
        curve = AlignmentCurve(
            points=[
                AlignmentPoint(num_annotations=5, alignment_score=0.5),
                AlignmentPoint(num_annotations=20, alignment_score=0.9),
            ],
        )
        assert estimate_optimal_annotations(curve) == 20

    def test_empty_curve(self):
        curve = AlignmentCurve()
        assert estimate_optimal_annotations(curve) == 0


# -------------------------------------------------------------------
# compute_roi
# -------------------------------------------------------------------

class TestComputeRoi:
    def test_basic_calculation(self):
        pairs = [(5, 0.5), (10, 0.7), (15, 0.85)]
        curve = build_alignment_curve(pairs)
        roi = compute_roi(curve, cost_per_annotation=2.0)
        assert roi["total_cost"] == 30.0  # 15 * 2.0
        assert abs(roi["total_improvement"] - 0.35) < 1e-9
        assert roi["optimal_annotations"] == 15  # no diminishing returns
        assert roi["optimal_cost"] == 30.0

    def test_with_diminishing_returns(self):
        pairs = [(5, 0.5), (10, 0.7), (15, 0.705), (20, 0.708)]
        curve = build_alignment_curve(pairs)
        roi = compute_roi(curve, cost_per_annotation=1.0)
        assert roi["optimal_annotations"] == 15
        assert roi["optimal_cost"] == 15.0
        assert roi["total_cost"] == 20.0

    def test_empty_curve(self):
        curve = AlignmentCurve()
        roi = compute_roi(curve)
        assert roi["total_cost"] == 0.0
        assert roi["total_improvement"] == 0.0
        assert roi["cost_per_point"] == 0.0
        assert roi["optimal_annotations"] == 0
        assert roi["optimal_cost"] == 0.0

    def test_cost_per_point(self):
        pairs = [(10, 0.5), (20, 0.7)]
        curve = build_alignment_curve(pairs)
        roi = compute_roi(curve, cost_per_annotation=1.0)
        # total_cost = 20, improvement = 0.2, cost_per_point = 100.0
        assert abs(roi["cost_per_point"] - 100.0) < 1e-9

    def test_roi_keys(self):
        pairs = [(5, 0.5), (10, 0.7)]
        curve = build_alignment_curve(pairs)
        roi = compute_roi(curve)
        expected_keys = {
            "total_cost",
            "total_improvement",
            "cost_per_point",
            "optimal_annotations",
            "optimal_cost",
        }
        assert set(roi.keys()) == expected_keys
