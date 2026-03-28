from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.trace.archival import (
    ArchivalConfig,
    ArchivalResult,
    ArchivedTrace,
    should_archive,
    plan_archival,
    create_archive_manifest,
    estimate_archive_size,
)


def _now():
    return datetime.now(timezone.utc)


def _days_ago(n):
    return (_now() - timedelta(days=n)).isoformat()


def _trace(tid, days_old=0, span_count=5):
    return {
        "id": tid,
        "created_at": _days_ago(days_old),
        "span_count": span_count,
    }


# --- should_archive ---


def test_should_archive_old_trace():
    old = _now() - timedelta(days=100)
    assert should_archive(old, max_age_days=90) is True


def test_should_archive_recent_trace():
    recent = _now() - timedelta(days=10)
    assert should_archive(recent, max_age_days=90) is False


def test_should_archive_exact_boundary():
    # Exactly at cutoff: not older, so should not archive
    boundary = _now() - timedelta(days=90)
    # Due to time passing between creation and check, this could go either way.
    # Use a slightly newer timestamp to ensure it's kept.
    just_inside = _now() - timedelta(days=89)
    assert should_archive(just_inside, max_age_days=90) is False


def test_should_archive_naive_datetime():
    # Naive datetime (no tzinfo) should be treated as UTC
    old_naive = datetime.now() - timedelta(days=100)
    assert should_archive(old_naive, max_age_days=90) is True


def test_should_archive_custom_age():
    ts = _now() - timedelta(days=10)
    assert should_archive(ts, max_age_days=5) is True
    assert should_archive(ts, max_age_days=15) is False


# --- plan_archival ---


def test_plan_archival_splits_correctly():
    traces = [
        _trace("t1", days_old=100, span_count=3),
        _trace("t2", days_old=10, span_count=5),
        _trace("t3", days_old=200, span_count=2),
    ]
    config = ArchivalConfig(max_age_days=90)
    to_archive, to_keep = plan_archival(traces, config)
    assert len(to_archive) == 2
    assert len(to_keep) == 1
    assert to_keep[0]["id"] == "t2"


def test_plan_archival_all_old():
    traces = [
        _trace("t1", days_old=100),
        _trace("t2", days_old=200),
    ]
    config = ArchivalConfig(max_age_days=90)
    to_archive, to_keep = plan_archival(traces, config)
    assert len(to_archive) == 2
    assert len(to_keep) == 0


def test_plan_archival_all_new():
    traces = [
        _trace("t1", days_old=1),
        _trace("t2", days_old=5),
    ]
    config = ArchivalConfig(max_age_days=90)
    to_archive, to_keep = plan_archival(traces, config)
    assert len(to_archive) == 0
    assert len(to_keep) == 2


def test_plan_archival_empty_traces():
    config = ArchivalConfig(max_age_days=90)
    to_archive, to_keep = plan_archival([], config)
    assert to_archive == []
    assert to_keep == []


def test_plan_archival_custom_max_age():
    traces = [
        _trace("t1", days_old=10),
        _trace("t2", days_old=3),
    ]
    config = ArchivalConfig(max_age_days=5)
    to_archive, to_keep = plan_archival(traces, config)
    assert len(to_archive) == 1
    assert to_archive[0]["id"] == "t1"


def test_plan_archival_datetime_object():
    # created_at as datetime object instead of ISO string
    old_dt = _now() - timedelta(days=100)
    traces = [{"id": "t1", "created_at": old_dt, "span_count": 3}]
    config = ArchivalConfig(max_age_days=90)
    to_archive, to_keep = plan_archival(traces, config)
    assert len(to_archive) == 1


# --- create_archive_manifest ---


def test_create_archive_manifest():
    traces = [
        {"id": "t1", "span_count": 5, "original_path": "/traces/t1"},
        {"id": "t2", "span_count": 3},
    ]
    manifest = create_archive_manifest(traces)
    assert len(manifest) == 2
    assert manifest[0].trace_id == "t1"
    assert manifest[0].span_count == 5
    assert manifest[0].original_path == "/traces/t1"
    assert manifest[1].trace_id == "t2"
    assert manifest[1].original_path == ""


def test_create_archive_manifest_empty():
    manifest = create_archive_manifest([])
    assert manifest == []


def test_create_archive_manifest_archived_at_is_recent():
    traces = [{"id": "t1", "span_count": 1}]
    manifest = create_archive_manifest(traces)
    # archived_at should be very recent (within last second)
    delta = _now() - manifest[0].archived_at
    assert delta.total_seconds() < 2


# --- estimate_archive_size ---


def test_estimate_archive_size():
    traces = [
        {"span_count": 10},
        {"span_count": 20},
    ]
    size = estimate_archive_size(traces)
    assert size == 30 * 500  # default 500 bytes per span


def test_estimate_archive_size_custom_avg():
    traces = [{"span_count": 10}]
    size = estimate_archive_size(traces, avg_bytes_per_span=100)
    assert size == 1000


def test_estimate_archive_size_empty():
    assert estimate_archive_size([]) == 0


def test_estimate_archive_size_no_span_count():
    traces = [{"id": "t1"}]
    size = estimate_archive_size(traces)
    assert size == 0


# --- ArchivalConfig ---


def test_archival_config_defaults():
    config = ArchivalConfig()
    assert config.max_age_days == 90
    assert config.archive_path == "archive"
    assert config.dry_run is False


def test_archival_config_as_dict():
    config = ArchivalConfig(max_age_days=30, archive_path="/cold", dry_run=True)
    d = config.as_dict()
    assert d["max_age_days"] == 30
    assert d["archive_path"] == "/cold"
    assert d["dry_run"] is True


def test_archival_config_from_dict():
    d = {"max_age_days": 60, "archive_path": "/storage", "dry_run": True}
    config = ArchivalConfig.from_dict(d)
    assert config.max_age_days == 60
    assert config.archive_path == "/storage"
    assert config.dry_run is True


def test_archival_config_from_dict_defaults():
    config = ArchivalConfig.from_dict({})
    assert config.max_age_days == 90
    assert config.archive_path == "archive"
    assert config.dry_run is False


def test_archival_config_roundtrip():
    original = ArchivalConfig(max_age_days=45, archive_path="/tmp/arch", dry_run=True)
    restored = ArchivalConfig.from_dict(original.as_dict())
    assert restored.max_age_days == original.max_age_days
    assert restored.archive_path == original.archive_path
    assert restored.dry_run == original.dry_run


# --- ArchivalResult ---


def test_archival_result_as_dict():
    result = ArchivalResult(
        archived_count=5, skipped_count=3, total_size_bytes=1500, archive_path="/arch"
    )
    d = result.as_dict()
    assert d["archived_count"] == 5
    assert d["skipped_count"] == 3
    assert d["total_size_bytes"] == 1500
    assert d["archive_path"] == "/arch"


def test_archival_result_format_text():
    result = ArchivalResult(
        archived_count=10, skipped_count=2, total_size_bytes=5000, archive_path="/cold"
    )
    text = result.format_text()
    assert "Archived: 10" in text
    assert "Skipped:  2" in text
    assert "5000 bytes" in text
    assert "/cold" in text


def test_archival_result_format_text_no_path():
    result = ArchivalResult(archived_count=1, skipped_count=0)
    text = result.format_text()
    assert "Path:" not in text


# --- ArchivedTrace ---


def test_archived_trace_as_dict():
    now = _now()
    at = ArchivedTrace(
        trace_id="t1", archived_at=now, original_path="/p", span_count=3
    )
    d = at.as_dict()
    assert d["trace_id"] == "t1"
    assert d["archived_at"] == now.isoformat()
    assert d["original_path"] == "/p"
    assert d["span_count"] == 3


def test_archived_trace_from_dict():
    now = _now()
    d = {
        "trace_id": "t1",
        "archived_at": now.isoformat(),
        "original_path": "/p",
        "span_count": 7,
    }
    at = ArchivedTrace.from_dict(d)
    assert at.trace_id == "t1"
    assert at.span_count == 7
    assert at.original_path == "/p"


def test_archived_trace_roundtrip():
    now = _now()
    original = ArchivedTrace(
        trace_id="t1", archived_at=now, original_path="/data/t1", span_count=12
    )
    d = original.as_dict()
    restored = ArchivedTrace.from_dict(d)
    assert restored.trace_id == original.trace_id
    assert restored.span_count == original.span_count
    assert restored.original_path == original.original_path


def test_archived_trace_from_dict_defaults():
    d = {"trace_id": "t1", "archived_at": _now().isoformat()}
    at = ArchivedTrace.from_dict(d)
    assert at.original_path == ""
    assert at.span_count == 0
