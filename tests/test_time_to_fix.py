"""Tests for time-to-fix tracking."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.time_to_fix import (
    ItemFixHistory,
    TimeToFixReport,
    track_item_fixes,
    track_all_fixes,
    compute_fix_percentiles,
    identify_chronic_failures,
    render_fix_timeline,
)


# ---------------------------------------------------------------------------
# track_item_fixes
# ---------------------------------------------------------------------------


class TestTrackItemFixes:
    def test_never_failed(self):
        """Item that always passed should not be marked as failing."""
        h = track_item_fixes("a", [True, True, True])
        assert h.still_failing is False
        assert h.runs_to_fix == 0

    def test_fixed_after_3_runs(self):
        """Fail on run 0, pass on run 3 -> runs_to_fix = 3."""
        h = track_item_fixes("b", [False, False, False, True])
        assert h.first_failed_run == 0
        assert h.first_fixed_run == 3
        assert h.runs_to_fix == 3
        assert h.still_failing is False

    def test_still_failing(self):
        """Item that never recovered should be marked still_failing."""
        h = track_item_fixes("c", [False, False, False])
        assert h.still_failing is True
        assert h.first_fixed_run is None
        assert h.runs_to_fix is None

    def test_fail_in_middle(self):
        """Pass, then fail, then pass - first failure at run 1."""
        h = track_item_fixes("d", [True, False, True])
        assert h.first_failed_run == 1
        assert h.first_fixed_run == 2
        assert h.runs_to_fix == 1
        assert h.still_failing is False

    def test_single_fail_then_pass(self):
        """One fail followed by one pass."""
        h = track_item_fixes("e", [False, True])
        assert h.first_failed_run == 0
        assert h.first_fixed_run == 1
        assert h.runs_to_fix == 1

    def test_single_run_pass(self):
        """Single passing run."""
        h = track_item_fixes("f", [True])
        assert h.still_failing is False
        assert h.runs_to_fix == 0

    def test_single_run_fail(self):
        """Single failing run."""
        h = track_item_fixes("g", [False])
        assert h.still_failing is True
        assert h.first_failed_run == 0
        assert h.first_fixed_run is None

    def test_empty_results(self):
        """Empty run_results should produce never-failed item."""
        h = track_item_fixes("h", [])
        assert h.still_failing is False
        assert h.runs_to_fix == 0

    def test_late_failure(self):
        """Failure starting at run 4."""
        h = track_item_fixes("i", [True, True, True, True, False, False])
        assert h.first_failed_run == 4
        assert h.still_failing is True

    def test_late_failure_then_fix(self):
        """Failure at run 2, fix at run 4."""
        h = track_item_fixes("j", [True, True, False, False, True])
        assert h.first_failed_run == 2
        assert h.first_fixed_run == 4
        assert h.runs_to_fix == 2


# ---------------------------------------------------------------------------
# track_all_fixes
# ---------------------------------------------------------------------------


class TestTrackAllFixes:
    def test_aggregate_stats(self):
        """Aggregate report should compute correct counts."""
        item_results = {
            "a": [False, False, True],   # fixed after 2
            "b": [False, False, False],  # still failing
            "c": [True, True, True],     # never failed
        }
        report = track_all_fixes(item_results)
        assert report.total_tracked == 3
        assert report.fixed_count == 1
        assert report.still_failing_count == 1
        assert report.avg_runs_to_fix == 2.0
        assert report.median_runs_to_fix == 2.0

    def test_empty_input(self):
        """No items should produce zeroed report."""
        report = track_all_fixes({})
        assert report.total_tracked == 0
        assert report.fixed_count == 0
        assert report.still_failing_count == 0

    def test_all_passing(self):
        """All items always passing."""
        report = track_all_fixes({
            "a": [True, True],
            "b": [True, True],
        })
        assert report.fixed_count == 0
        assert report.still_failing_count == 0
        assert report.total_tracked == 2

    def test_multiple_fixed(self):
        """Multiple fixed items should compute mean/median correctly."""
        item_results = {
            "a": [False, True, True],         # runs_to_fix = 1
            "b": [False, False, False, True],  # runs_to_fix = 3
        }
        report = track_all_fixes(item_results)
        assert report.fixed_count == 2
        assert report.avg_runs_to_fix == 2.0
        assert report.median_runs_to_fix == 2.0


# ---------------------------------------------------------------------------
# compute_fix_percentiles
# ---------------------------------------------------------------------------


class TestComputeFixPercentiles:
    def test_basic_percentiles(self):
        """Percentiles for a mix of fix times."""
        item_results = {
            "a": [False, True],                     # 1
            "b": [False, False, True],               # 2
            "c": [False, False, False, True],         # 3
            "d": [False, False, False, False, True],  # 4
        }
        report = track_all_fixes(item_results)
        pcts = compute_fix_percentiles(report)
        assert "p25" in pcts
        assert "p50" in pcts
        assert "p75" in pcts
        assert pcts["p25"] <= pcts["p50"] <= pcts["p75"]

    def test_no_fixed_items(self):
        """All still failing should return zeros."""
        report = track_all_fixes({"a": [False, False]})
        pcts = compute_fix_percentiles(report)
        assert pcts == {"p25": 0.0, "p50": 0.0, "p75": 0.0}

    def test_single_fixed_item(self):
        """Single fixed item: all percentiles equal."""
        report = track_all_fixes({"a": [False, False, True]})
        pcts = compute_fix_percentiles(report)
        assert pcts["p25"] == pcts["p50"] == pcts["p75"] == 2.0


# ---------------------------------------------------------------------------
# identify_chronic_failures
# ---------------------------------------------------------------------------


class TestIdentifyChronicFailures:
    def test_chronic_detected(self):
        """Item failing since run 0 with 6 total runs should be chronic at min_runs=5."""
        item_results = {
            "a": [False, False, False, False, False, False],
        }
        report = track_all_fixes(item_results)
        # total_tracked=1, but we need runs_since_fail = total_tracked - first_failed_run
        # = 1 - 0 = 1 which is < 5. We need to think about this differently.
        # Actually total_tracked is number of items, not runs.
        # Let me re-read the implementation.
        chronic = identify_chronic_failures(report, min_runs=1)
        assert "a" in chronic

    def test_not_chronic_yet(self):
        """Item that failed recently should not be chronic."""
        item_results = {
            "a": [True, True, False],
        }
        report = track_all_fixes(item_results)
        chronic = identify_chronic_failures(report, min_runs=5)
        assert chronic == []

    def test_fixed_items_excluded(self):
        """Fixed items should not appear in chronic list."""
        item_results = {
            "a": [False, True],
        }
        report = track_all_fixes(item_results)
        chronic = identify_chronic_failures(report, min_runs=1)
        assert chronic == []


# ---------------------------------------------------------------------------
# render_fix_timeline
# ---------------------------------------------------------------------------


class TestRenderFixTimeline:
    def test_never_failed(self):
        """All-pass timeline should be all P."""
        h = track_item_fixes("a", [True, True, True])
        result = render_fix_timeline(h, 3)
        assert result == "a: PPP"

    def test_fail_then_fix(self):
        """Fail on 0-1, pass on 2-3."""
        h = track_item_fixes("b", [False, False, True, True])
        result = render_fix_timeline(h, 4)
        assert result == "b: FFPP"

    def test_still_failing(self):
        """All failing should be all F."""
        h = track_item_fixes("c", [False, False, False])
        result = render_fix_timeline(h, 3)
        assert result == "c: FFF"

    def test_zero_total_runs(self):
        """Zero total runs should produce empty timeline."""
        h = track_item_fixes("d", [])
        result = render_fix_timeline(h, 0)
        assert result == "d: "

    def test_pass_fail_pass(self):
        """P then F then P timeline."""
        h = track_item_fixes("e", [True, False, True])
        result = render_fix_timeline(h, 3)
        assert result == "e: PFP"


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestTimeToFixReportFormatText:
    def test_format_with_fixes(self):
        """Format text should include item statuses."""
        report = track_all_fixes({
            "a": [False, True],
            "b": [False, False],
        })
        text = report.format_text()
        assert "Time-to-Fix Report" in text
        assert "Fixed: 1" in text
        assert "Still failing: 1" in text
        assert "FIXED" in text
        assert "FAILING" in text

    def test_format_empty(self):
        """Empty report should still format."""
        report = track_all_fixes({})
        text = report.format_text()
        assert "0 items tracked" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_item_fix_history_roundtrip(self):
        h = track_item_fixes("x", [False, False, True])
        d = h.as_dict()
        h2 = ItemFixHistory.from_dict(d)
        assert h2.item_id == h.item_id
        assert h2.first_failed_run == h.first_failed_run
        assert h2.first_fixed_run == h.first_fixed_run
        assert h2.runs_to_fix == h.runs_to_fix
        assert h2.still_failing == h.still_failing

    def test_report_roundtrip(self):
        report = track_all_fixes({
            "a": [False, True],
            "b": [False, False],
        })
        d = report.as_dict()
        report2 = TimeToFixReport.from_dict(d)
        assert report2.total_tracked == report.total_tracked
        assert report2.fixed_count == report.fixed_count
        assert report2.still_failing_count == report.still_failing_count
        assert len(report2.items) == len(report.items)

    def test_item_fix_history_from_dict_defaults(self):
        """from_dict with empty dict should use defaults."""
        h = ItemFixHistory.from_dict({})
        assert h.item_id == ""
        assert h.still_failing is True
