"""
Storage usage forecast: predict storage growth based on historical data points.

Provides pure functions to compute daily growth via linear regression, project
future storage size, estimate days until a limit, and render ASCII charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class UsageDataPoint:
    """A single storage usage measurement."""

    date: str
    size_bytes: int
    row_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
        }


@dataclass
class ForecastResult:
    """Result of a storage usage forecast."""

    current_size_bytes: int
    projected_30d_bytes: int = 0
    projected_90d_bytes: int = 0
    daily_growth_bytes: int = 0
    days_until_limit: Optional[int] = None
    recommendation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_size_bytes": self.current_size_bytes,
            "projected_30d_bytes": self.projected_30d_bytes,
            "projected_90d_bytes": self.projected_90d_bytes,
            "daily_growth_bytes": self.daily_growth_bytes,
            "days_until_limit": self.days_until_limit,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ForecastResult:
        return cls(
            current_size_bytes=data.get("current_size_bytes", 0),
            projected_30d_bytes=data.get("projected_30d_bytes", 0),
            projected_90d_bytes=data.get("projected_90d_bytes", 0),
            daily_growth_bytes=data.get("daily_growth_bytes", 0),
            days_until_limit=data.get("days_until_limit"),
            recommendation=data.get("recommendation", ""),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Usage Forecast")
        lines.append("=" * 40)
        lines.append(f"Current size: {_format_size(self.current_size_bytes)}")
        lines.append(f"Daily growth: {_format_size(self.daily_growth_bytes)}")
        lines.append(f"30-day projection: {_format_size(self.projected_30d_bytes)}")
        lines.append(f"90-day projection: {_format_size(self.projected_90d_bytes)}")
        if self.days_until_limit is not None:
            lines.append(f"Days until limit: {self.days_until_limit}")
        if self.recommendation:
            lines.append("")
            lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val < 0:
        return f"-{_format_size(-bytes_val)}"
    if bytes_val < 1024:
        return f"{bytes_val} B"
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    if bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_daily_growth(data_points: List[UsageDataPoint]) -> int:
    """Compute daily growth rate in bytes using linear regression.

    Uses the index of each data point as the x-value and size_bytes as y.
    Returns the slope rounded to an integer. Returns 0 for fewer than 2 points.
    """
    n = len(data_points)
    if n < 2:
        return 0

    # Simple linear regression: y = a + b*x
    # x values are 0, 1, 2, ... (index as proxy for days)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for i, dp in enumerate(data_points):
        x = float(i)
        y = float(dp.size_bytes)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return int(round(slope))


def forecast_size(current_bytes: int, daily_growth: int, days: int) -> int:
    """Project future storage size."""
    return current_bytes + daily_growth * days


def days_until_limit(
    current_bytes: int, daily_growth: int, limit_bytes: int
) -> Optional[int]:
    """How many days until limit_bytes is reached.

    Returns None if daily_growth is zero or negative (will never reach limit).
    """
    if daily_growth <= 0:
        return None
    remaining = limit_bytes - current_bytes
    if remaining <= 0:
        return 0
    return int(remaining / daily_growth)


def build_usage_forecast(
    data_points: List[UsageDataPoint], limit_bytes: int = 1_000_000_000
) -> ForecastResult:
    """Build a full storage usage forecast with recommendations."""
    if not data_points:
        return ForecastResult(current_size_bytes=0, recommendation="No data available")

    current = data_points[-1].size_bytes
    growth = compute_daily_growth(data_points)

    proj_30 = forecast_size(current, growth, 30)
    proj_90 = forecast_size(current, growth, 90)
    dtl = days_until_limit(current, growth, limit_bytes)

    # Build recommendation
    rec = ""
    if growth <= 0:
        rec = "Storage is stable or shrinking - no action needed"
    elif dtl is not None and dtl < 30:
        rec = "Storage limit approaching within 30 days - take action soon"
    elif dtl is not None and dtl < 90:
        rec = "Storage limit approaching within 90 days - plan cleanup"
    else:
        rec = "Growth is manageable - monitor periodically"

    return ForecastResult(
        current_size_bytes=current,
        projected_30d_bytes=proj_30,
        projected_90d_bytes=proj_90,
        daily_growth_bytes=growth,
        days_until_limit=dtl,
        recommendation=rec,
    )


def render_forecast_chart(
    data_points: List[UsageDataPoint], forecast_days: int = 30, width: int = 60
) -> str:
    """Render an ASCII chart showing historical and projected usage.

    Historical points use '#' and projected points use '.'.
    """
    if not data_points:
        return "(no data)"

    growth = compute_daily_growth(data_points)

    # Build combined series: historical + projected
    labels: List[str] = []
    values: List[int] = []
    kinds: List[str] = []  # "h" for historical, "p" for projected

    for dp in data_points:
        labels.append(dp.date)
        values.append(dp.size_bytes)
        kinds.append("h")

    last_size = data_points[-1].size_bytes
    for d in range(1, forecast_days + 1):
        projected = last_size + growth * d
        labels.append(f"+{d}d")
        values.append(max(projected, 0))
        kinds.append("p")

    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1

    # Determine label width
    max_label = max(len(lb) for lb in labels)
    bar_width = width - max_label - 2
    if bar_width < 10:
        bar_width = 10

    lines: List[str] = []
    for label, val, kind in zip(labels, values, kinds):
        bar_len = int((val / max_val) * bar_width)
        bar_len = max(bar_len, 0)
        char = "#" if kind == "h" else "."
        bar = char * bar_len
        padded = label.ljust(max_label)
        lines.append(f"{padded}  {bar} {_format_size(val)}")

    return "\n".join(lines)


def suggest_actions(forecast: ForecastResult) -> List[str]:
    """Suggest actions based on the forecast growth rate."""
    actions: List[str] = []

    if forecast.daily_growth_bytes <= 0:
        actions.append("No action needed - storage is not growing")
        return actions

    daily_mb = forecast.daily_growth_bytes / (1024 * 1024)

    if daily_mb > 10:
        actions.append("Run compaction to reclaim unused space")
        actions.append("Review retention policy - consider shorter retention")
        actions.append("Archive old data to external storage")
    elif daily_mb > 1:
        actions.append("Review retention policy for optimization")
        actions.append("Consider enabling auto-compaction")
    else:
        actions.append("Monitor growth - current rate is low")

    if forecast.days_until_limit is not None and forecast.days_until_limit < 30:
        actions.append("Increase storage limit or free space immediately")

    if forecast.days_until_limit is not None and forecast.days_until_limit < 90:
        actions.append("Plan storage expansion within the next quarter")

    return actions
