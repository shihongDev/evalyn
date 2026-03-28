"""Tests for run management: tagging, filtering, cleanup."""

import pytest
from datetime import datetime, timezone, timedelta
from evalyn_sdk.models import EvalRun, MetricResult, MetricSpec
from evalyn_sdk.evaluation.run_management import (
    filter_runs_by_tag,
    filter_runs_by_age,
    filter_runs_below_pass_rate,
    select_runs_for_cleanup,
    CleanupResult,
)


def _make_run(
    run_id: str = "run-1",
    name: str = None,
    pinned: bool = False,
    tags: list = None,
    created_at: datetime = None,
    pass_rate: float = None,
) -> EvalRun:
    now = created_at or datetime.now(timezone.utc)
    summary = {}
    if pass_rate is not None:
        summary = {"metrics": {"m1": {"pass_rate": pass_rate}}}
    return EvalRun(
        id=run_id,
        dataset_name="test",
        created_at=now,
        metric_results=[],
        name=name,
        pinned=pinned,
        tags=tags or [],
        summary=summary,
    )


class TestEvalRunTags:
    """Test EvalRun tag support."""

    def test_tags_default_empty(self):
        run = _make_run()
        assert run.tags == []

    def test_tags_set(self):
        run = _make_run(tags=["experiment-v1", "nightly"])
        assert run.tags == ["experiment-v1", "nightly"]

    def test_has_tag_true(self):
        run = _make_run(tags=["experiment-v1"])
        assert run.has_tag("experiment-v1") is True

    def test_has_tag_false(self):
        run = _make_run(tags=["experiment-v1"])
        assert run.has_tag("nonexistent") is False

    def test_has_tag_empty(self):
        run = _make_run()
        assert run.has_tag("anything") is False

    def test_tags_in_as_dict(self):
        run = _make_run(tags=["tag1", "tag2"])
        d = run.as_dict()
        assert d["tags"] == ["tag1", "tag2"]

    def test_tags_not_in_as_dict_when_empty(self):
        run = _make_run()
        d = run.as_dict()
        assert "tags" not in d

    def test_tags_roundtrip(self):
        run = _make_run(tags=["a", "b"])
        d = run.as_dict()
        restored = EvalRun.from_dict(d)
        assert restored.tags == ["a", "b"]

    def test_tags_from_dict_missing(self):
        d = {
            "id": "r1", "dataset_name": "ds",
            "metric_results": [],
        }
        run = EvalRun.from_dict(d)
        assert run.tags == []


class TestFilterByTag:
    """Test tag-based filtering."""

    def test_filter_matching(self):
        runs = [
            _make_run("r1", tags=["nightly"]),
            _make_run("r2", tags=["experiment"]),
            _make_run("r3", tags=["nightly", "ci"]),
        ]
        result = filter_runs_by_tag(runs, "nightly")
        assert len(result) == 2
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r3" in ids

    def test_filter_no_match(self):
        runs = [_make_run("r1", tags=["a"]), _make_run("r2", tags=["b"])]
        assert filter_runs_by_tag(runs, "c") == []

    def test_filter_empty_runs(self):
        assert filter_runs_by_tag([], "any") == []


class TestFilterByAge:
    """Test age-based filtering."""

    def test_old_runs_found(self):
        now = datetime.now(timezone.utc)
        runs = [
            _make_run("r1", created_at=now - timedelta(days=60)),
            _make_run("r2", created_at=now - timedelta(days=10)),
            _make_run("r3", created_at=now - timedelta(days=90)),
        ]
        old = filter_runs_by_age(runs, max_age_days=30)
        assert len(old) == 2
        ids = [r.id for r in old]
        assert "r1" in ids
        assert "r3" in ids

    def test_no_old_runs(self):
        now = datetime.now(timezone.utc)
        runs = [_make_run("r1", created_at=now - timedelta(days=5))]
        assert filter_runs_by_age(runs, max_age_days=30) == []

    def test_empty_runs(self):
        assert filter_runs_by_age([], max_age_days=30) == []


class TestFilterByPassRate:
    """Test pass-rate-based filtering."""

    def test_low_pass_rate_found(self):
        runs = [
            _make_run("r1", pass_rate=0.9),
            _make_run("r2", pass_rate=0.2),
            _make_run("r3", pass_rate=0.5),
        ]
        low = filter_runs_below_pass_rate(runs, min_pass_rate=0.5)
        assert len(low) == 1
        assert low[0].id == "r2"

    def test_all_above_threshold(self):
        runs = [_make_run("r1", pass_rate=0.9), _make_run("r2", pass_rate=0.8)]
        assert filter_runs_below_pass_rate(runs, min_pass_rate=0.5) == []

    def test_no_summary_skipped(self):
        runs = [_make_run("r1")]  # no pass_rate in summary
        assert filter_runs_below_pass_rate(runs, min_pass_rate=0.5) == []


class TestSelectForCleanup:
    """Test cleanup selection logic."""

    def test_cleanup_by_age(self):
        now = datetime.now(timezone.utc)
        runs = [
            _make_run("r1", created_at=now - timedelta(days=60)),
            _make_run("r2", created_at=now - timedelta(days=10)),
        ]
        result = select_runs_for_cleanup(runs, older_than_days=30)
        assert result.deleted_count == 1
        assert "r1" in result.deleted_ids
        assert result.kept_count == 1

    def test_cleanup_by_pass_rate(self):
        runs = [
            _make_run("r1", pass_rate=0.1),
            _make_run("r2", pass_rate=0.9),
        ]
        result = select_runs_for_cleanup(runs, below_pass_rate=0.5)
        assert result.deleted_count == 1
        assert "r1" in result.deleted_ids

    def test_cleanup_keeps_pinned(self):
        now = datetime.now(timezone.utc)
        runs = [
            _make_run("r1", created_at=now - timedelta(days=60), pinned=True),
            _make_run("r2", created_at=now - timedelta(days=60)),
        ]
        result = select_runs_for_cleanup(runs, older_than_days=30)
        assert result.deleted_count == 1
        assert "r2" in result.deleted_ids
        assert result.skipped_pinned == 1

    def test_cleanup_combined_criteria(self):
        now = datetime.now(timezone.utc)
        runs = [
            _make_run("r1", created_at=now - timedelta(days=60), pass_rate=0.9),
            _make_run("r2", created_at=now - timedelta(days=5), pass_rate=0.1),
            _make_run("r3", created_at=now - timedelta(days=60), pass_rate=0.1),
            _make_run("r4", created_at=now - timedelta(days=5), pass_rate=0.9),
        ]
        result = select_runs_for_cleanup(
            runs, older_than_days=30, below_pass_rate=0.5,
        )
        # r1: old (match), r2: low pass (match), r3: old + low (match), r4: neither
        assert result.deleted_count == 3
        assert result.kept_count == 1

    def test_cleanup_nothing_to_delete(self):
        runs = [_make_run("r1", pass_rate=0.9)]
        result = select_runs_for_cleanup(runs, below_pass_rate=0.5)
        assert result.deleted_count == 0
        assert result.kept_count == 1

    def test_cleanup_empty_runs(self):
        result = select_runs_for_cleanup([], older_than_days=30)
        assert result.deleted_count == 0
        assert result.total_scanned == 0

    def test_no_criteria_deletes_nothing(self):
        runs = [_make_run("r1"), _make_run("r2")]
        result = select_runs_for_cleanup(runs)
        assert result.deleted_count == 0
        assert result.kept_count == 2
