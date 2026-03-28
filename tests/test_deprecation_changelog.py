"""Tests for metric deprecation lifecycle and evaluation changelog."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import EvalRun, MetricResult
from evalyn_sdk.metrics.deprecation import (
    DeprecationRegistry, DeprecationEntry,
)
from evalyn_sdk.analysis.changelog import (
    generate_changelog, Changelog, ChangelogEntry,
)


def _result(mid="m", iid="i1", passed=True):
    return MetricResult(
        metric_id=mid, item_id=iid, call_id="c1",
        score=1.0 if passed else 0.0, passed=passed, details={},
    )


def _run(run_id, results=None):
    now = datetime.now(timezone.utc)
    return EvalRun(
        id=run_id, dataset_name="test", created_at=now,
        metric_results=results or [],
    )


# =============================================================================
# Deprecation tests
# =============================================================================

class TestDeprecationRegistry:
    def test_deprecate_and_check(self):
        reg = DeprecationRegistry()
        reg.deprecate("old_metric", replacement="new_metric", since="0.15.0")
        assert reg.is_deprecated("old_metric")
        assert not reg.is_deprecated("current_metric")

    def test_check_metrics_warns(self):
        reg = DeprecationRegistry()
        reg.deprecate("old", replacement="new")
        warnings = reg.check_metrics(["old", "current"])
        assert len(warnings) == 1
        assert "old" in warnings[0]
        assert "new" in warnings[0]

    def test_warn_only_once(self):
        reg = DeprecationRegistry()
        reg.deprecate("old")
        w1 = reg.check_metrics(["old"])
        w2 = reg.check_metrics(["old"])
        assert len(w1) == 1
        assert len(w2) == 0  # already warned

    def test_no_deprecated(self):
        reg = DeprecationRegistry()
        assert reg.check_metrics(["a", "b"]) == []

    def test_get_entry(self):
        reg = DeprecationRegistry()
        reg.deprecate("old", reason="outdated rubric")
        entry = reg.get_entry("old")
        assert entry is not None
        assert entry.reason == "outdated rubric"

    def test_get_entry_missing(self):
        reg = DeprecationRegistry()
        assert reg.get_entry("nonexistent") is None

    def test_list_deprecated(self):
        reg = DeprecationRegistry()
        reg.deprecate("b_metric")
        reg.deprecate("a_metric")
        listed = reg.list_deprecated()
        assert len(listed) == 2
        assert listed[0].metric_id == "a_metric"  # sorted

    def test_migration_guide(self):
        reg = DeprecationRegistry()
        reg.deprecate("old", replacement="new", reason="better rubric", sunset_date="2026-06-01")
        guide = reg.migration_guide()
        assert "old" in guide
        assert "new" in guide
        assert "2026-06-01" in guide

    def test_migration_guide_empty(self):
        reg = DeprecationRegistry()
        assert "No deprecated" in reg.migration_guide()

    def test_count(self):
        reg = DeprecationRegistry()
        assert reg.count == 0
        reg.deprecate("a")
        reg.deprecate("b")
        assert reg.count == 2

    def test_entry_format_warning(self):
        e = DeprecationEntry(
            metric_id="old", replacement="new",
            deprecated_since="0.15.0", sunset_date="2026-06-01",
        )
        warning = e.format_warning()
        assert "old" in warning
        assert "new" in warning
        assert "0.15.0" in warning

    def test_entry_as_dict(self):
        e = DeprecationEntry(metric_id="old", replacement="new")
        d = e.as_dict()
        assert d["metric_id"] == "old"
        assert d["replacement"] == "new"


# =============================================================================
# Changelog tests
# =============================================================================

class TestGenerateChangelog:
    def test_no_changes(self):
        results = [_result("a", "i1", True)]
        cl = generate_changelog(_run("r1", results), _run("r2", results))
        assert len(cl.regressions) == 0
        assert len(cl.improvements) == 0
        assert "unchanged" in cl.summary_line

    def test_regression(self):
        base = [_result("help", "i1", True)] * 10
        curr = [_result("help", f"i{i}", True) for i in range(3)] + [_result("help", f"i{i}", False) for i in range(3, 10)]
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        assert len(cl.regressions) == 1
        assert cl.regressions[0].metric_id == "help"

    def test_improvement(self):
        base = [_result("help", f"i{i}", i < 5) for i in range(10)]
        curr = [_result("help", f"i{i}", True) for i in range(10)]
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        assert len(cl.improvements) == 1

    def test_added_metric(self):
        base = [_result("a", "i1")]
        curr = [_result("a", "i1"), _result("b", "i1")]
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        assert len(cl.added) == 1
        assert cl.added[0].metric_id == "b"

    def test_removed_metric(self):
        base = [_result("a", "i1"), _result("b", "i1")]
        curr = [_result("a", "i1")]
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        assert len(cl.removed) == 1
        assert cl.removed[0].metric_id == "b"

    def test_summary_line(self):
        base = [_result("a", "i1", True)]
        curr = [_result("a", "i1", False)]
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        assert "regressed" in cl.summary_line

    def test_empty_runs(self):
        cl = generate_changelog(_run("r1"), _run("r2"))
        assert len(cl.entries) == 0
        assert "no changes" in cl.summary_line

    def test_as_dict(self):
        cl = generate_changelog(_run("r1"), _run("r2"))
        d = cl.as_dict()
        assert "baseline_run_id" in d
        assert "summary" in d
        assert "regressions" in d

    def test_format_text(self):
        base = [_result("help", "i1", True)] * 10
        curr = [_result("help", "i1", False)] * 10
        cl = generate_changelog(_run("r1", base), _run("r2", curr))
        text = cl.format_text()
        assert "Changelog" in text
        assert "help" in text

    def test_custom_threshold(self):
        base = [_result("a", "i1", True)] * 10
        curr = [_result("a", f"i{i}", True) for i in range(9)] + [_result("a", "i9", False)]
        # 10% drop with 15% threshold should NOT be flagged
        cl = generate_changelog(_run("r1", base), _run("r2", curr), regression_threshold=0.15)
        assert len(cl.regressions) == 0
