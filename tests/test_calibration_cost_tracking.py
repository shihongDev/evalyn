"""Tests for calibration cost tracking."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.cost_tracking import (
    CalibrationCostRecord,
    CalibrationCostTracker,
    PhaseCost,
    compare_calibration_costs,
    estimate_calibration_cost,
)


# -------------------------------------------------------------------
# PhaseCost
# -------------------------------------------------------------------

class TestPhaseCost:
    def test_defaults(self):
        pc = PhaseCost(phase="alignment")
        assert pc.tokens == 0
        assert pc.cost_usd == 0.0
        assert pc.calls == 0

    def test_as_dict(self):
        pc = PhaseCost(phase="validation", tokens=100, cost_usd=0.01, calls=2)
        d = pc.as_dict()
        assert d["phase"] == "validation"
        assert d["tokens"] == 100
        assert d["cost_usd"] == 0.01
        assert d["calls"] == 2


# -------------------------------------------------------------------
# CalibrationCostRecord
# -------------------------------------------------------------------

class TestCalibrationCostRecord:
    def test_as_dict_from_dict_roundtrip(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rec = CalibrationCostRecord(
            metric_id="acc",
            optimizer="opro",
            phases=[PhaseCost(phase="alignment", tokens=50, cost_usd=0.005, calls=1)],
            total_tokens=50,
            total_cost_usd=0.005,
            total_calls=1,
            timestamp=ts,
        )
        d = rec.as_dict()
        restored = CalibrationCostRecord.from_dict(d)
        assert restored.metric_id == "acc"
        assert restored.optimizer == "opro"
        assert len(restored.phases) == 1
        assert restored.phases[0].tokens == 50
        assert restored.total_tokens == 50
        assert restored.timestamp == ts

    def test_from_dict_missing_optional(self):
        rec = CalibrationCostRecord.from_dict(
            {"metric_id": "m", "optimizer": "o"}
        )
        assert rec.phases == []
        assert rec.total_tokens == 0
        assert rec.timestamp is None

    def test_format_text_contains_metric(self):
        rec = CalibrationCostRecord(
            metric_id="coherence",
            optimizer="basic",
            phases=[PhaseCost(phase="optimization", tokens=200, cost_usd=0.02, calls=3)],
            total_tokens=200,
            total_cost_usd=0.02,
            total_calls=3,
        )
        text = rec.format_text()
        assert "coherence" in text
        assert "basic" in text
        assert "200" in text
        assert "optimization" in text

    def test_format_text_no_phases(self):
        rec = CalibrationCostRecord(metric_id="x", optimizer="y")
        text = rec.format_text()
        assert "x" in text


# -------------------------------------------------------------------
# CalibrationCostTracker
# -------------------------------------------------------------------

class TestCalibrationCostTracker:
    def test_record_adds_to_correct_phase(self):
        t = CalibrationCostTracker("m1", "opro")
        t.record("alignment", 100, 0.01)
        t.record("alignment", 50, 0.005)
        pc = t.get_phase_cost("alignment")
        assert pc.tokens == 150
        assert pc.calls == 2
        assert abs(pc.cost_usd - 0.015) < 1e-9

    def test_record_multiple_phases(self):
        t = CalibrationCostTracker("m1", "opro")
        t.record("alignment", 100, 0.01)
        t.record("validation", 200, 0.02)
        assert t.get_phase_cost("alignment").tokens == 100
        assert t.get_phase_cost("validation").tokens == 200

    def test_get_phase_cost_missing(self):
        t = CalibrationCostTracker("m1", "opro")
        pc = t.get_phase_cost("nonexistent")
        assert pc.tokens == 0
        assert pc.phase == "nonexistent"

    def test_get_record_totals(self):
        t = CalibrationCostTracker("m1", "opro")
        t.record("alignment", 100, 0.01)
        t.record("optimization", 200, 0.02)
        t.record("validation", 50, 0.005)
        rec = t.get_record()
        assert rec.total_tokens == 350
        assert abs(rec.total_cost_usd - 0.035) < 1e-9
        assert rec.total_calls == 3
        assert rec.metric_id == "m1"
        assert rec.optimizer == "opro"
        assert rec.timestamp is not None

    def test_get_record_empty(self):
        t = CalibrationCostTracker("m1", "opro")
        rec = t.get_record()
        assert rec.total_tokens == 0
        assert rec.total_cost_usd == 0.0
        assert rec.total_calls == 0
        assert rec.phases == []

    def test_reset_clears(self):
        t = CalibrationCostTracker("m1", "opro")
        t.record("alignment", 100, 0.01)
        t.reset()
        assert t.get_phase_cost("alignment").tokens == 0
        rec = t.get_record()
        assert rec.total_tokens == 0

    def test_single_phase_record(self):
        t = CalibrationCostTracker("m1", "basic")
        t.record("alignment", 500, 0.05)
        rec = t.get_record()
        assert len(rec.phases) == 1
        assert rec.phases[0].phase == "alignment"


# -------------------------------------------------------------------
# compare_calibration_costs
# -------------------------------------------------------------------

class TestCompareCalibrationCosts:
    def test_finds_cheapest_and_most_expensive(self):
        records = [
            CalibrationCostRecord(metric_id="a", optimizer="o", total_cost_usd=0.10),
            CalibrationCostRecord(metric_id="b", optimizer="o", total_cost_usd=0.05),
            CalibrationCostRecord(metric_id="c", optimizer="o", total_cost_usd=0.20),
        ]
        result = compare_calibration_costs(records)
        assert result["cheapest"] == "b"
        assert result["most_expensive"] == "c"
        assert abs(result["average_cost"] - (0.10 + 0.05 + 0.20) / 3) < 1e-9

    def test_single_record(self):
        records = [
            CalibrationCostRecord(metric_id="only", optimizer="o", total_cost_usd=0.07),
        ]
        result = compare_calibration_costs(records)
        assert result["cheapest"] == "only"
        assert result["most_expensive"] == "only"
        assert abs(result["average_cost"] - 0.07) < 1e-9

    def test_empty_records(self):
        result = compare_calibration_costs([])
        assert result["cheapest"] is None
        assert result["most_expensive"] is None
        assert result["average_cost"] == 0.0


# -------------------------------------------------------------------
# estimate_calibration_cost
# -------------------------------------------------------------------

class TestEstimateCalibrationCost:
    def test_basic_estimate(self):
        cost = estimate_calibration_cost(num_items=10, num_iterations=10)
        # 10*10=100 calls, 100*500=50000 tokens, 50000/1000*0.001 = 0.05
        assert abs(cost - 0.05) < 1e-9

    def test_custom_parameters(self):
        cost = estimate_calibration_cost(
            num_items=5,
            num_iterations=2,
            avg_tokens_per_call=1000,
            cost_per_1k_tokens=0.01,
        )
        # 5*2=10 calls, 10*1000=10000 tokens, 10000/1000*0.01 = 0.10
        assert abs(cost - 0.10) < 1e-9

    def test_zero_items(self):
        cost = estimate_calibration_cost(num_items=0)
        assert cost == 0.0

    def test_zero_iterations(self):
        cost = estimate_calibration_cost(num_items=10, num_iterations=0)
        assert cost == 0.0
