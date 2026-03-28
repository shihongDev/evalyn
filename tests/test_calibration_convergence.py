"""Tests for calibration convergence visualization."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.convergence import (
    ConvergencePoint,
    ConvergenceTrace,
    ConvergenceTracker,
    diagnose_convergence,
    render_ascii_convergence,
)


# -------------------------------------------------------------------
# ConvergencePoint
# -------------------------------------------------------------------

class TestConvergencePoint:
    def test_defaults(self):
        p = ConvergencePoint(step=0, alignment_score=0.5)
        assert p.prompt_length == 0
        assert p.tokens_used == 0

    def test_as_dict(self):
        p = ConvergencePoint(step=3, alignment_score=0.8, prompt_length=100, tokens_used=500)
        d = p.as_dict()
        assert d["step"] == 3
        assert d["alignment_score"] == 0.8
        assert d["prompt_length"] == 100
        assert d["tokens_used"] == 500


# -------------------------------------------------------------------
# ConvergenceTracker
# -------------------------------------------------------------------

class TestConvergenceTracker:
    def test_records_steps(self):
        t = ConvergenceTracker("acc", "opro")
        t.record(0.5)
        t.record(0.6)
        t.record(0.7)
        trace = t.get_trace()
        assert len(trace.points) == 3
        assert trace.points[0].step == 0
        assert trace.points[1].step == 1
        assert trace.points[2].step == 2

    def test_records_with_metadata(self):
        t = ConvergenceTracker("acc", "opro")
        t.record(0.5, prompt_length=50, tokens_used=200)
        trace = t.get_trace()
        assert trace.points[0].prompt_length == 50
        assert trace.points[0].tokens_used == 200

    def test_get_trace_metadata(self):
        t = ConvergenceTracker("f1", "basic")
        trace = t.get_trace()
        assert trace.metric_id == "f1"
        assert trace.optimizer == "basic"

    def test_reset_clears(self):
        t = ConvergenceTracker("acc", "opro")
        t.record(0.5)
        t.record(0.6)
        t.reset()
        trace = t.get_trace()
        assert len(trace.points) == 0

    def test_reset_restarts_step_counter(self):
        t = ConvergenceTracker("acc", "opro")
        t.record(0.5)
        t.record(0.6)
        t.reset()
        t.record(0.7)
        trace = t.get_trace()
        assert trace.points[0].step == 0


# -------------------------------------------------------------------
# ConvergenceTrace - properties
# -------------------------------------------------------------------

class TestConvergenceTraceProperties:
    def test_best_score(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.9),
                ConvergencePoint(step=2, alignment_score=0.7),
            ],
        )
        assert trace.best_score == 0.9

    def test_best_step(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.9),
                ConvergencePoint(step=2, alignment_score=0.7),
            ],
        )
        assert trace.best_step == 1

    def test_converged_stable_last_3(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.6),
                ConvergencePoint(step=2, alignment_score=0.80),
                ConvergencePoint(step=3, alignment_score=0.81),
                ConvergencePoint(step=4, alignment_score=0.80),
            ],
        )
        assert trace.converged is True

    def test_not_converged_varying(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.9),
                ConvergencePoint(step=2, alignment_score=0.3),
            ],
        )
        assert trace.converged is False

    def test_not_converged_fewer_than_3_points(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.6),
            ],
        )
        assert trace.converged is False

    def test_improvement_positive(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.8),
            ],
        )
        assert abs(trace.improvement - 0.3) < 1e-9

    def test_improvement_negative(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.8),
                ConvergencePoint(step=1, alignment_score=0.5),
            ],
        )
        assert abs(trace.improvement - (-0.3)) < 1e-9

    def test_improvement_single_point(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[ConvergencePoint(step=0, alignment_score=0.5)],
        )
        assert trace.improvement == 0.0

    def test_empty_trace_defaults(self):
        trace = ConvergenceTrace(metric_id="m", optimizer="o")
        assert trace.best_score == 0.0
        assert trace.best_step == 0
        assert trace.converged is False
        assert trace.improvement == 0.0


# -------------------------------------------------------------------
# ConvergenceTrace - as_dict / from_dict
# -------------------------------------------------------------------

class TestConvergenceTraceSerialize:
    def test_as_dict(self):
        trace = ConvergenceTrace(
            metric_id="acc",
            optimizer="opro",
            points=[ConvergencePoint(step=0, alignment_score=0.5)],
        )
        d = trace.as_dict()
        assert d["metric_id"] == "acc"
        assert d["optimizer"] == "opro"
        assert len(d["points"]) == 1

    def test_from_dict_roundtrip(self):
        trace = ConvergenceTrace(
            metric_id="f1",
            optimizer="basic",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5, prompt_length=10, tokens_used=100),
                ConvergencePoint(step=1, alignment_score=0.7, prompt_length=20, tokens_used=200),
            ],
        )
        d = trace.as_dict()
        restored = ConvergenceTrace.from_dict(d)
        assert restored.metric_id == "f1"
        assert restored.optimizer == "basic"
        assert len(restored.points) == 2
        assert restored.points[0].alignment_score == 0.5
        assert restored.points[1].prompt_length == 20

    def test_from_dict_empty_points(self):
        restored = ConvergenceTrace.from_dict(
            {"metric_id": "m", "optimizer": "o"}
        )
        assert restored.points == []


# -------------------------------------------------------------------
# render_ascii_convergence
# -------------------------------------------------------------------

class TestRenderAsciiConvergence:
    def test_produces_output(self):
        trace = ConvergenceTrace(
            metric_id="acc",
            optimizer="opro",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.7),
                ConvergencePoint(step=2, alignment_score=0.8),
            ],
        )
        output = render_ascii_convergence(trace)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "#" in output

    def test_contains_metric_id(self):
        trace = ConvergenceTrace(
            metric_id="coherence",
            optimizer="basic",
            points=[ConvergencePoint(step=0, alignment_score=0.5)],
        )
        output = render_ascii_convergence(trace)
        assert "coherence" in output

    def test_empty_trace(self):
        trace = ConvergenceTrace(metric_id="m", optimizer="o")
        output = render_ascii_convergence(trace)
        assert "no data" in output.lower()

    def test_single_point(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[ConvergencePoint(step=0, alignment_score=0.5)],
        )
        output = render_ascii_convergence(trace)
        assert "#" in output

    def test_custom_dimensions(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=i, alignment_score=0.5 + i * 0.05)
                for i in range(5)
            ],
        )
        output = render_ascii_convergence(trace, width=30, height=10)
        assert isinstance(output, str)
        assert "#" in output


# -------------------------------------------------------------------
# diagnose_convergence
# -------------------------------------------------------------------

class TestDiagnoseConvergence:
    def test_converged_and_improving(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.7),
                ConvergencePoint(step=2, alignment_score=0.80),
                ConvergencePoint(step=3, alignment_score=0.81),
                ConvergencePoint(step=4, alignment_score=0.80),
            ],
        )
        result = diagnose_convergence(trace)
        assert result["converged"] is True
        assert result["improving"] is True
        assert "good" in result["recommendation"].lower() or "converged" in result["recommendation"].lower()

    def test_oscillating(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=i, alignment_score=0.5 + (0.3 if i % 2 == 0 else -0.3))
                for i in range(8)
            ],
        )
        result = diagnose_convergence(trace)
        assert result["oscillating"] is True
        assert "oscillat" in result["recommendation"].lower()

    def test_improving_not_converged(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.3),
                ConvergencePoint(step=1, alignment_score=0.5),
                ConvergencePoint(step=2, alignment_score=0.7),
                ConvergencePoint(step=3, alignment_score=0.9),
            ],
        )
        result = diagnose_convergence(trace)
        assert result["improving"] is True
        assert result["converged"] is False

    def test_empty_trace(self):
        trace = ConvergenceTrace(metric_id="m", optimizer="o")
        result = diagnose_convergence(trace)
        assert result["converged"] is False
        assert result["oscillating"] is False
        assert result["improving"] is False
        assert result["plateau_at_step"] is None

    def test_single_point(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[ConvergencePoint(step=0, alignment_score=0.5)],
        )
        result = diagnose_convergence(trace)
        assert result["converged"] is False
        assert "one data point" in result["recommendation"].lower()

    def test_plateau_detection(self):
        points = [ConvergencePoint(step=0, alignment_score=0.3)]
        # After step 0, scores plateau at ~0.8
        for i in range(1, 8):
            points.append(ConvergencePoint(step=i, alignment_score=0.80))
        trace = ConvergenceTrace(metric_id="m", optimizer="o", points=points)
        result = diagnose_convergence(trace)
        assert result["plateau_at_step"] is not None

    def test_result_keys(self):
        trace = ConvergenceTrace(
            metric_id="m",
            optimizer="o",
            points=[
                ConvergencePoint(step=0, alignment_score=0.5),
                ConvergencePoint(step=1, alignment_score=0.6),
            ],
        )
        result = diagnose_convergence(trace)
        assert "converged" in result
        assert "plateau_at_step" in result
        assert "oscillating" in result
        assert "improving" in result
        assert "recommendation" in result
