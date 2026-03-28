"""Tests for storage usage forecast module."""

from __future__ import annotations

from evalyn_sdk.storage.usage_forecast import (
    ForecastResult,
    UsageDataPoint,
    build_usage_forecast,
    compute_daily_growth,
    days_until_limit,
    forecast_size,
    render_forecast_chart,
    suggest_actions,
)


# ---------------------------------------------------------------------------
# UsageDataPoint
# ---------------------------------------------------------------------------


class TestUsageDataPoint:
    def test_as_dict(self):
        dp = UsageDataPoint(date="2026-01-01", size_bytes=1000, row_count=10)
        d = dp.as_dict()
        assert d["date"] == "2026-01-01"
        assert d["size_bytes"] == 1000
        assert d["row_count"] == 10

    def test_defaults(self):
        dp = UsageDataPoint(date="2026-01-01", size_bytes=500)
        assert dp.row_count == 0


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------


class TestForecastResult:
    def test_as_dict(self):
        fr = ForecastResult(
            current_size_bytes=1000,
            projected_30d_bytes=2000,
            daily_growth_bytes=33,
            days_until_limit=100,
            recommendation="OK",
        )
        d = fr.as_dict()
        assert d["current_size_bytes"] == 1000
        assert d["projected_30d_bytes"] == 2000
        assert d["daily_growth_bytes"] == 33
        assert d["days_until_limit"] == 100
        assert d["recommendation"] == "OK"

    def test_from_dict(self):
        d = {
            "current_size_bytes": 500,
            "projected_30d_bytes": 1500,
            "projected_90d_bytes": 3500,
            "daily_growth_bytes": 33,
            "days_until_limit": 50,
            "recommendation": "watch it",
        }
        fr = ForecastResult.from_dict(d)
        assert fr.current_size_bytes == 500
        assert fr.projected_30d_bytes == 1500
        assert fr.projected_90d_bytes == 3500
        assert fr.daily_growth_bytes == 33
        assert fr.days_until_limit == 50
        assert fr.recommendation == "watch it"

    def test_from_dict_defaults(self):
        fr = ForecastResult.from_dict({})
        assert fr.current_size_bytes == 0
        assert fr.days_until_limit is None
        assert fr.recommendation == ""

    def test_roundtrip(self):
        fr = ForecastResult(
            current_size_bytes=999,
            projected_30d_bytes=1999,
            projected_90d_bytes=3999,
            daily_growth_bytes=33,
            days_until_limit=42,
            recommendation="test",
        )
        restored = ForecastResult.from_dict(fr.as_dict())
        assert restored.current_size_bytes == fr.current_size_bytes
        assert restored.projected_30d_bytes == fr.projected_30d_bytes
        assert restored.projected_90d_bytes == fr.projected_90d_bytes
        assert restored.daily_growth_bytes == fr.daily_growth_bytes
        assert restored.days_until_limit == fr.days_until_limit
        assert restored.recommendation == fr.recommendation

    def test_format_text(self):
        fr = ForecastResult(
            current_size_bytes=1024,
            daily_growth_bytes=100,
            projected_30d_bytes=4024,
            projected_90d_bytes=10024,
            days_until_limit=50,
            recommendation="Monitor it",
        )
        text = fr.format_text()
        assert "Usage Forecast" in text
        assert "Days until limit: 50" in text
        assert "Monitor it" in text

    def test_format_text_no_limit(self):
        fr = ForecastResult(current_size_bytes=100, days_until_limit=None)
        text = fr.format_text()
        assert "Days until limit" not in text

    def test_format_text_no_recommendation(self):
        fr = ForecastResult(current_size_bytes=100, recommendation="")
        text = fr.format_text()
        assert "Recommendation" not in text


# ---------------------------------------------------------------------------
# compute_daily_growth
# ---------------------------------------------------------------------------


class TestComputeDailyGrowth:
    def test_linear_growth(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=1000 + i * 100)
            for i in range(5)
        ]
        growth = compute_daily_growth(points)
        assert growth == 100

    def test_flat(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=500)
            for i in range(5)
        ]
        growth = compute_daily_growth(points)
        assert growth == 0

    def test_single_point(self):
        points = [UsageDataPoint(date="2026-01-01", size_bytes=1000)]
        growth = compute_daily_growth(points)
        assert growth == 0

    def test_empty(self):
        growth = compute_daily_growth([])
        assert growth == 0

    def test_two_points(self):
        points = [
            UsageDataPoint(date="2026-01-01", size_bytes=1000),
            UsageDataPoint(date="2026-01-02", size_bytes=1500),
        ]
        growth = compute_daily_growth(points)
        assert growth == 500

    def test_decreasing(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=1000 - i * 50)
            for i in range(5)
        ]
        growth = compute_daily_growth(points)
        assert growth == -50


# ---------------------------------------------------------------------------
# forecast_size
# ---------------------------------------------------------------------------


class TestForecastSize:
    def test_basic(self):
        result = forecast_size(1000, 100, 10)
        assert result == 2000

    def test_zero_growth(self):
        result = forecast_size(1000, 0, 30)
        assert result == 1000

    def test_negative_growth(self):
        result = forecast_size(1000, -50, 10)
        assert result == 500


# ---------------------------------------------------------------------------
# days_until_limit
# ---------------------------------------------------------------------------


class TestDaysUntilLimit:
    def test_calculates(self):
        result = days_until_limit(500, 50, 1000)
        assert result == 10

    def test_no_growth(self):
        result = days_until_limit(500, 0, 1000)
        assert result is None

    def test_negative_growth(self):
        result = days_until_limit(500, -10, 1000)
        assert result is None

    def test_already_at_limit(self):
        result = days_until_limit(1000, 100, 1000)
        assert result == 0

    def test_already_over_limit(self):
        result = days_until_limit(1500, 100, 1000)
        assert result == 0


# ---------------------------------------------------------------------------
# build_usage_forecast
# ---------------------------------------------------------------------------


class TestBuildUsageForecast:
    def test_full(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=1000 + i * 100)
            for i in range(5)
        ]
        result = build_usage_forecast(points, limit_bytes=5000)
        assert result.current_size_bytes == 1400
        assert result.daily_growth_bytes == 100
        assert result.projected_30d_bytes == 1400 + 100 * 30
        assert result.projected_90d_bytes == 1400 + 100 * 90
        assert result.days_until_limit is not None
        assert result.recommendation != ""

    def test_empty_data(self):
        result = build_usage_forecast([])
        assert result.current_size_bytes == 0
        assert "No data" in result.recommendation

    def test_approaching_limit(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=900 + i * 10)
            for i in range(5)
        ]
        # Limit is very close
        result = build_usage_forecast(points, limit_bytes=950)
        assert "approaching" in result.recommendation.lower() or "action" in result.recommendation.lower()

    def test_stable_storage(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=500)
            for i in range(5)
        ]
        result = build_usage_forecast(points, limit_bytes=10000)
        assert result.daily_growth_bytes == 0
        assert result.days_until_limit is None
        assert "stable" in result.recommendation.lower() or "not growing" in result.recommendation.lower()


# ---------------------------------------------------------------------------
# render_forecast_chart
# ---------------------------------------------------------------------------


class TestRenderForecastChart:
    def test_output(self):
        points = [
            UsageDataPoint(date=f"2026-01-0{i+1}", size_bytes=1000 + i * 100)
            for i in range(3)
        ]
        chart = render_forecast_chart(points, forecast_days=5, width=60)
        assert "2026-01-01" in chart
        assert "#" in chart
        assert "." in chart  # projected portion
        assert "+1d" in chart

    def test_empty(self):
        chart = render_forecast_chart([])
        assert chart == "(no data)"

    def test_single_point(self):
        points = [UsageDataPoint(date="2026-01-01", size_bytes=1000)]
        chart = render_forecast_chart(points, forecast_days=3)
        assert "2026-01-01" in chart
        # With single point, growth is 0, so projections stay same
        lines = chart.strip().split("\n")
        assert len(lines) == 4  # 1 historical + 3 projected


# ---------------------------------------------------------------------------
# suggest_actions
# ---------------------------------------------------------------------------


class TestSuggestActions:
    def test_high_growth(self):
        fr = ForecastResult(
            current_size_bytes=100_000_000,
            daily_growth_bytes=15 * 1024 * 1024,  # 15 MB/day
            days_until_limit=20,
        )
        actions = suggest_actions(fr)
        assert len(actions) >= 2
        assert any("compaction" in a.lower() for a in actions)
        assert any("immediately" in a.lower() or "limit" in a.lower() for a in actions)

    def test_low_growth(self):
        fr = ForecastResult(
            current_size_bytes=100_000,
            daily_growth_bytes=500,  # 500 bytes/day
            days_until_limit=None,
        )
        actions = suggest_actions(fr)
        assert len(actions) >= 1
        assert any("monitor" in a.lower() for a in actions)

    def test_no_growth(self):
        fr = ForecastResult(
            current_size_bytes=100_000,
            daily_growth_bytes=0,
        )
        actions = suggest_actions(fr)
        assert len(actions) >= 1
        assert any("no action" in a.lower() for a in actions)

    def test_medium_growth(self):
        fr = ForecastResult(
            current_size_bytes=100_000_000,
            daily_growth_bytes=2 * 1024 * 1024,  # 2 MB/day
            days_until_limit=200,
        )
        actions = suggest_actions(fr)
        assert len(actions) >= 1
        assert any("retention" in a.lower() for a in actions)
