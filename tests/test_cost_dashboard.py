"""Tests for cost tracking dashboard module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.cost_dashboard import (
    CostEntry,
    DailyCost,
    CostDashboard,
    aggregate_daily_costs,
    build_cost_dashboard,
    detect_cost_trend,
    render_cost_chart_ascii,
    find_cost_spikes,
    project_monthly_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entries(costs_by_date: dict[str, list[tuple[str, float, int, int]]]) -> list[CostEntry]:
    """Build CostEntry list from {date: [(model, cost, tokens, calls), ...]}."""
    entries = []
    for date, items in costs_by_date.items():
        for model, cost, tokens, calls in items:
            entries.append(CostEntry(date=date, model=model, cost_usd=cost, tokens=tokens, calls=calls))
    return entries


# ---------------------------------------------------------------------------
# aggregate_daily_costs
# ---------------------------------------------------------------------------


class TestAggregateDailyCosts:
    def test_groups_by_date(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
            "2026-03-02": [("gpt-4", 0.20, 2000, 10)],
        })
        daily = aggregate_daily_costs(entries)
        assert len(daily) == 2
        assert daily[0].date == "2026-03-01"
        assert daily[1].date == "2026-03-02"

    def test_sums_within_day(self):
        entries = _make_entries({
            "2026-03-01": [
                ("gpt-4", 0.10, 1000, 5),
                ("claude", 0.05, 500, 3),
            ],
        })
        daily = aggregate_daily_costs(entries)
        assert len(daily) == 1
        assert abs(daily[0].total_cost_usd - 0.15) < 1e-9
        assert daily[0].total_tokens == 1500
        assert daily[0].total_calls == 8

    def test_by_model_breakdown(self):
        entries = _make_entries({
            "2026-03-01": [
                ("gpt-4", 0.10, 1000, 5),
                ("claude", 0.05, 500, 3),
            ],
        })
        daily = aggregate_daily_costs(entries)
        assert abs(daily[0].by_model["gpt-4"] - 0.10) < 1e-9
        assert abs(daily[0].by_model["claude"] - 0.05) < 1e-9

    def test_sorted_by_date(self):
        entries = _make_entries({
            "2026-03-03": [("gpt-4", 0.30, 3000, 15)],
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
            "2026-03-02": [("gpt-4", 0.20, 2000, 10)],
        })
        daily = aggregate_daily_costs(entries)
        dates = [d.date for d in daily]
        assert dates == ["2026-03-01", "2026-03-02", "2026-03-03"]

    def test_empty_entries(self):
        daily = aggregate_daily_costs([])
        assert daily == []

    def test_same_model_same_day_accumulates(self):
        entries = [
            CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.05, tokens=500, calls=2),
            CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.03, tokens=300, calls=1),
        ]
        daily = aggregate_daily_costs(entries)
        assert len(daily) == 1
        assert abs(daily[0].by_model["gpt-4"] - 0.08) < 1e-9


# ---------------------------------------------------------------------------
# detect_cost_trend
# ---------------------------------------------------------------------------


class TestDetectCostTrend:
    def test_increasing(self):
        daily = [
            DailyCost(date=f"2026-03-0{i}", total_cost_usd=float(i))
            for i in range(1, 7)
        ]
        assert detect_cost_trend(daily) == "increasing"

    def test_decreasing(self):
        daily = [
            DailyCost(date=f"2026-03-0{i}", total_cost_usd=float(6 - i))
            for i in range(1, 7)
        ]
        assert detect_cost_trend(daily) == "decreasing"

    def test_stable(self):
        daily = [
            DailyCost(date=f"2026-03-0{i}", total_cost_usd=1.0)
            for i in range(1, 7)
        ]
        assert detect_cost_trend(daily) == "stable"

    def test_fewer_than_3_days(self):
        daily = [DailyCost(date="2026-03-01", total_cost_usd=1.0)]
        assert detect_cost_trend(daily) == "stable"

    def test_exactly_3_days_increasing(self):
        daily = [
            DailyCost(date="2026-03-01", total_cost_usd=1.0),
            DailyCost(date="2026-03-02", total_cost_usd=2.0),
            DailyCost(date="2026-03-03", total_cost_usd=3.0),
        ]
        # first 3 avg = 2.0, last 3 avg = 2.0 -> stable (same window)
        assert detect_cost_trend(daily) == "stable"

    def test_all_zero(self):
        daily = [
            DailyCost(date=f"2026-03-0{i}", total_cost_usd=0.0)
            for i in range(1, 7)
        ]
        assert detect_cost_trend(daily) == "stable"


# ---------------------------------------------------------------------------
# build_cost_dashboard
# ---------------------------------------------------------------------------


class TestBuildCostDashboard:
    def test_full_dashboard(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
            "2026-03-02": [("gpt-4", 0.20, 2000, 10)],
            "2026-03-03": [("claude", 0.15, 1500, 7)],
        })
        dash = build_cost_dashboard(entries)
        assert abs(dash.total_cost_usd - 0.45) < 1e-9
        assert dash.total_tokens == 4500
        assert abs(dash.avg_daily_cost - 0.15) < 1e-9
        assert len(dash.daily_costs) == 3

    def test_most_expensive_model(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.50, 5000, 20), ("claude", 0.10, 1000, 5)],
        })
        dash = build_cost_dashboard(entries)
        assert dash.most_expensive_model == "gpt-4"

    def test_empty_entries(self):
        dash = build_cost_dashboard([])
        assert dash.total_cost_usd == 0.0
        assert dash.daily_costs == []
        assert dash.trend == "stable"

    def test_single_day(self):
        entries = [CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.10, tokens=1000, calls=5)]
        dash = build_cost_dashboard(entries)
        assert len(dash.daily_costs) == 1
        assert abs(dash.avg_daily_cost - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# render_cost_chart_ascii
# ---------------------------------------------------------------------------


class TestRenderCostChartAscii:
    def test_output_contains_dates(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
            "2026-03-02": [("gpt-4", 0.20, 2000, 10)],
        })
        dash = build_cost_dashboard(entries)
        chart = render_cost_chart_ascii(dash)
        assert "2026-03-01" in chart
        assert "2026-03-02" in chart

    def test_output_contains_bars(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
        })
        dash = build_cost_dashboard(entries)
        chart = render_cost_chart_ascii(dash)
        assert "#" in chart

    def test_empty_dashboard(self):
        dash = CostDashboard()
        chart = render_cost_chart_ascii(dash)
        assert chart == "No data"

    def test_custom_width(self):
        entries = [CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.10, tokens=1000, calls=5)]
        dash = build_cost_dashboard(entries)
        chart = render_cost_chart_ascii(dash, width=40)
        assert "Daily Cost Chart" in chart

    def test_zero_cost_no_crash(self):
        entries = [CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.0, tokens=0, calls=0)]
        dash = build_cost_dashboard(entries)
        chart = render_cost_chart_ascii(dash)
        assert "2026-03-01" in chart


# ---------------------------------------------------------------------------
# find_cost_spikes
# ---------------------------------------------------------------------------


class TestFindCostSpikes:
    def test_detects_outlier(self):
        daily = [
            DailyCost(date="2026-03-01", total_cost_usd=1.0),
            DailyCost(date="2026-03-02", total_cost_usd=1.0),
            DailyCost(date="2026-03-03", total_cost_usd=1.0),
            DailyCost(date="2026-03-04", total_cost_usd=5.0),
        ]
        spikes = find_cost_spikes(daily)
        # avg = 2.0, threshold = 4.0, only day 4 (5.0) is a spike
        assert spikes == ["2026-03-04"]

    def test_no_spikes(self):
        daily = [
            DailyCost(date=f"2026-03-0{i}", total_cost_usd=1.0)
            for i in range(1, 5)
        ]
        spikes = find_cost_spikes(daily)
        assert spikes == []

    def test_custom_threshold(self):
        daily = [
            DailyCost(date="2026-03-01", total_cost_usd=1.0),
            DailyCost(date="2026-03-02", total_cost_usd=1.0),
            DailyCost(date="2026-03-03", total_cost_usd=2.0),
        ]
        # avg = 1.333, threshold at 1.5x = 2.0 -> day 3 is not a spike (equal)
        spikes = find_cost_spikes(daily, threshold_multiplier=1.5)
        assert "2026-03-03" not in spikes

    def test_empty(self):
        assert find_cost_spikes([]) == []


# ---------------------------------------------------------------------------
# project_monthly_cost
# ---------------------------------------------------------------------------


class TestProjectMonthlyCost:
    def test_basic_projection(self):
        dash = CostDashboard(avg_daily_cost=1.0)
        assert abs(project_monthly_cost(dash) - 30.0) < 1e-9

    def test_zero_cost(self):
        dash = CostDashboard(avg_daily_cost=0.0)
        assert project_monthly_cost(dash) == 0.0


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_cost_entry_roundtrip(self):
        original = CostEntry(date="2026-03-01", model="gpt-4", cost_usd=0.10, tokens=1000, calls=5)
        d = original.as_dict()
        restored = CostEntry.from_dict(d)
        assert restored.date == original.date
        assert restored.model == original.model
        assert restored.cost_usd == original.cost_usd
        assert restored.tokens == original.tokens
        assert restored.calls == original.calls

    def test_daily_cost_as_dict(self):
        dc = DailyCost(date="2026-03-01", total_cost_usd=0.15, total_tokens=1500, total_calls=8, by_model={"gpt-4": 0.10})
        d = dc.as_dict()
        assert d["date"] == "2026-03-01"
        assert d["by_model"]["gpt-4"] == 0.10

    def test_cost_dashboard_roundtrip(self):
        original = CostDashboard(
            total_cost_usd=0.45, total_tokens=4500,
            avg_daily_cost=0.15, trend="increasing", most_expensive_model="gpt-4",
        )
        d = original.as_dict()
        restored = CostDashboard.from_dict(d)
        assert restored.total_cost_usd == original.total_cost_usd
        assert restored.trend == original.trend
        assert restored.most_expensive_model == original.most_expensive_model


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_dashboard_format(self):
        entries = _make_entries({
            "2026-03-01": [("gpt-4", 0.10, 1000, 5)],
        })
        dash = build_cost_dashboard(entries)
        text = dash.format_text()
        assert "COST DASHBOARD" in text
        assert "gpt-4" in text
        assert "Trend:" in text

    def test_empty_dashboard_format(self):
        dash = CostDashboard()
        text = dash.format_text()
        assert "COST DASHBOARD" in text
        assert "stable" in text
