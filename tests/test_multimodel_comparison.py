"""Tests for multi-model comparison module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.multimodel_comparison import (
    ModelResult,
    ModelComparison,
    MultiModelReport,
    build_model_result,
    compare_two_models,
    build_multimodel_report,
    rank_models,
    compute_cost_efficiency,
)


# ---------------------------------------------------------------------------
# build_model_result
# ---------------------------------------------------------------------------


class TestBuildModelResult:
    def test_basic_aggregation(self):
        r = build_model_result("gpt-4", [0.8, 0.9, 1.0])
        assert r.model == "gpt-4"
        assert r.item_count == 3
        assert abs(r.avg_score - 0.9) < 1e-9

    def test_pass_rate_calculation(self):
        """Scores >= 0.5 count as passed."""
        r = build_model_result("gpt-4", [0.3, 0.6, 0.8, 0.2])
        assert r.item_count == 4
        assert abs(r.pass_rate - 0.5) < 1e-9

    def test_latency_aggregation(self):
        r = build_model_result("gpt-4", [1.0], latencies=[100.0, 200.0, 300.0])
        assert abs(r.avg_latency_ms - 200.0) < 1e-9

    def test_cost_aggregation(self):
        r = build_model_result("gpt-4", [1.0], costs=[0.01, 0.02, 0.03])
        assert abs(r.avg_cost_usd - 0.02) < 1e-9

    def test_provider_set(self):
        r = build_model_result("gpt-4", [1.0], provider="openai")
        assert r.provider == "openai"

    def test_empty_scores(self):
        r = build_model_result("gpt-4", [])
        assert r.item_count == 0
        assert r.avg_score == 0.0
        assert r.pass_rate == 0.0

    def test_all_passing(self):
        r = build_model_result("gpt-4", [0.8, 0.7, 0.6, 0.9, 1.0])
        assert abs(r.pass_rate - 1.0) < 1e-9

    def test_none_passing(self):
        r = build_model_result("gpt-4", [0.1, 0.2, 0.3, 0.4])
        assert abs(r.pass_rate - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# compare_two_models
# ---------------------------------------------------------------------------


class TestCompareTwoModels:
    def test_a_wins_higher_score(self):
        a = ModelResult(model="gpt-4", avg_score=0.9, avg_cost_usd=0.05)
        b = ModelResult(model="claude", avg_score=0.7, avg_cost_usd=0.03)
        c = compare_two_models(a, b)
        assert c.winner == "gpt-4"
        assert c.score_delta > 0

    def test_b_wins_higher_score(self):
        a = ModelResult(model="gpt-4", avg_score=0.6, avg_cost_usd=0.05)
        b = ModelResult(model="claude", avg_score=0.9, avg_cost_usd=0.03)
        c = compare_two_models(a, b)
        assert c.winner == "claude"
        assert c.score_delta < 0

    def test_tie_broken_by_cost(self):
        a = ModelResult(model="gpt-4", avg_score=0.8, avg_cost_usd=0.02)
        b = ModelResult(model="claude", avg_score=0.8, avg_cost_usd=0.05)
        c = compare_two_models(a, b)
        assert c.winner == "gpt-4"

    def test_tie_broken_by_cost_b_cheaper(self):
        a = ModelResult(model="gpt-4", avg_score=0.8, avg_cost_usd=0.05)
        b = ModelResult(model="claude", avg_score=0.8, avg_cost_usd=0.02)
        c = compare_two_models(a, b)
        assert c.winner == "claude"

    def test_identical_models_default_to_a(self):
        a = ModelResult(model="gpt-4", avg_score=0.8, avg_cost_usd=0.03)
        b = ModelResult(model="claude", avg_score=0.8, avg_cost_usd=0.03)
        c = compare_two_models(a, b)
        assert c.winner == "gpt-4"

    def test_deltas_correct(self):
        a = ModelResult(model="gpt-4", avg_score=0.9, avg_cost_usd=0.05, avg_latency_ms=100.0)
        b = ModelResult(model="claude", avg_score=0.7, avg_cost_usd=0.03, avg_latency_ms=80.0)
        c = compare_two_models(a, b)
        assert abs(c.score_delta - 0.2) < 1e-9
        assert abs(c.cost_delta - 0.02) < 1e-9
        assert abs(c.latency_delta - 20.0) < 1e-9

    def test_format_text(self):
        c = ModelComparison(
            model_a="gpt-4", model_b="claude",
            score_delta=0.2, cost_delta=0.02, latency_delta=20.0, winner="gpt-4",
        )
        text = c.format_text()
        assert "gpt-4 vs claude" in text
        assert "Winner: gpt-4" in text


# ---------------------------------------------------------------------------
# build_multimodel_report
# ---------------------------------------------------------------------------


class TestBuildMultimodelReport:
    def test_full_report(self):
        results = [
            ModelResult(model="gpt-4", avg_score=0.9, avg_cost_usd=0.05, avg_latency_ms=200.0),
            ModelResult(model="claude", avg_score=0.85, avg_cost_usd=0.03, avg_latency_ms=150.0),
            ModelResult(model="gemini", avg_score=0.8, avg_cost_usd=0.01, avg_latency_ms=100.0),
        ]
        report = build_multimodel_report(results)
        assert report.best_quality == "gpt-4"
        assert report.best_cost == "gemini"
        assert report.best_latency == "gemini"
        # 3 models -> 3 pairwise comparisons
        assert len(report.comparisons) == 3

    def test_empty_results(self):
        report = build_multimodel_report([])
        assert report.models == []
        assert report.comparisons == []
        assert report.best_quality == ""

    def test_single_model(self):
        results = [ModelResult(model="gpt-4", avg_score=0.9)]
        report = build_multimodel_report(results)
        assert report.best_quality == "gpt-4"
        assert len(report.comparisons) == 0


# ---------------------------------------------------------------------------
# rank_models
# ---------------------------------------------------------------------------


class TestRankModels:
    def _make_results(self):
        return [
            ModelResult(model="low", avg_score=0.5, avg_cost_usd=0.01, avg_latency_ms=50.0),
            ModelResult(model="mid", avg_score=0.7, avg_cost_usd=0.03, avg_latency_ms=100.0),
            ModelResult(model="high", avg_score=0.9, avg_cost_usd=0.05, avg_latency_ms=200.0),
        ]

    def test_rank_by_score(self):
        ranked = rank_models(self._make_results(), by="score")
        assert [r.model for r in ranked] == ["high", "mid", "low"]

    def test_rank_by_cost(self):
        ranked = rank_models(self._make_results(), by="cost")
        assert [r.model for r in ranked] == ["low", "mid", "high"]

    def test_rank_by_latency(self):
        ranked = rank_models(self._make_results(), by="latency")
        assert [r.model for r in ranked] == ["low", "mid", "high"]

    def test_rank_unknown_key(self):
        results = self._make_results()
        ranked = rank_models(results, by="unknown")
        assert len(ranked) == 3  # returns copy, no error


# ---------------------------------------------------------------------------
# compute_cost_efficiency
# ---------------------------------------------------------------------------


class TestComputeCostEfficiency:
    def test_basic(self):
        results = [
            ModelResult(model="cheap", avg_score=0.8, avg_cost_usd=0.01),
            ModelResult(model="expensive", avg_score=0.9, avg_cost_usd=0.10),
        ]
        eff = compute_cost_efficiency(results)
        assert eff["cheap"] == 0.8 / 0.01
        assert eff["expensive"] == 0.9 / 0.10
        assert eff["cheap"] > eff["expensive"]

    def test_zero_cost(self):
        results = [ModelResult(model="free", avg_score=0.9, avg_cost_usd=0.0)]
        eff = compute_cost_efficiency(results)
        assert eff["free"] == 0.0


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_model_result_roundtrip(self):
        original = ModelResult(
            model="gpt-4", provider="openai", avg_score=0.85,
            pass_rate=0.9, avg_latency_ms=150.0, avg_cost_usd=0.04, item_count=100,
        )
        d = original.as_dict()
        restored = ModelResult.from_dict(d)
        assert restored.model == original.model
        assert restored.provider == original.provider
        assert restored.avg_score == original.avg_score
        assert restored.item_count == original.item_count

    def test_multimodel_report_roundtrip(self):
        report = MultiModelReport(
            models=[ModelResult(model="gpt-4", avg_score=0.9)],
            best_quality="gpt-4",
            best_cost="gpt-4",
            best_latency="gpt-4",
        )
        d = report.as_dict()
        restored = MultiModelReport.from_dict(d)
        assert restored.best_quality == "gpt-4"
        assert len(restored.models) == 1
        assert restored.models[0].model == "gpt-4"

    def test_model_comparison_as_dict(self):
        c = ModelComparison(model_a="a", model_b="b", score_delta=0.1, winner="a")
        d = c.as_dict()
        assert d["model_a"] == "a"
        assert d["winner"] == "a"


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_multimodel_report_format(self):
        results = [
            ModelResult(model="gpt-4", avg_score=0.9, avg_cost_usd=0.05, avg_latency_ms=200.0, item_count=10),
        ]
        report = build_multimodel_report(results)
        text = report.format_text()
        assert "MULTI-MODEL COMPARISON" in text
        assert "gpt-4" in text
        assert "Best quality" in text

    def test_empty_report_format(self):
        report = MultiModelReport()
        text = report.format_text()
        assert "MULTI-MODEL COMPARISON" in text
