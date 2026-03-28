from __future__ import annotations

from datetime import datetime, timedelta, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.orphan_recovery import (
    OrphanSpan,
    RecoveryResult,
    find_orphan_spans,
    match_by_name,
    match_by_timestamp,
    reattach_orphans,
    recover_orphans,
)


def _span(sid, name="s", span_type="llm_call", parent_id=None, start_offset_ms=0, duration_ms=100, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(id=sid, name=name, span_type=span_type, parent_id=parent_id,
                start_time=t, end_time=t + timedelta(milliseconds=duration_ms), attributes=attrs)


# -- find_orphan_spans --


class TestFindOrphanSpans:
    def test_detects_orphans(self):
        root = _span("root")
        child = _span("child", parent_id="root")
        orphan = _span("orphan", parent_id="missing_parent")
        result = find_orphan_spans([root, child, orphan])
        assert len(result) == 1
        assert result[0].id == "orphan"

    def test_no_orphans(self):
        root = _span("root")
        child = _span("child", parent_id="root")
        result = find_orphan_spans([root, child])
        assert result == []

    def test_root_spans_not_orphans(self):
        root = _span("root")  # parent_id is None
        result = find_orphan_spans([root])
        assert result == []

    def test_empty_list(self):
        assert find_orphan_spans([]) == []

    def test_all_orphans(self):
        s1 = _span("s1", parent_id="gone1")
        s2 = _span("s2", parent_id="gone2")
        result = find_orphan_spans([s1, s2])
        assert len(result) == 2

    def test_multiple_orphans_mixed(self):
        root = _span("root")
        valid = _span("valid", parent_id="root")
        orphan1 = _span("o1", parent_id="missing1")
        orphan2 = _span("o2", parent_id="missing2")
        result = find_orphan_spans([root, valid, orphan1, orphan2])
        ids = {s.id for s in result}
        assert ids == {"o1", "o2"}


# -- match_by_timestamp --


class TestMatchByTimestamp:
    def test_closest_match(self):
        orphan = _span("o", start_offset_ms=500)
        c1 = _span("c1", start_offset_ms=0, duration_ms=200)
        c2 = _span("c2", start_offset_ms=400, duration_ms=200)
        result = match_by_timestamp(orphan, [c1, c2])
        assert result is not None
        sid, conf = result
        assert sid == "c2"
        assert conf > 0.0

    def test_within_range_zero_gap(self):
        orphan = _span("o", start_offset_ms=150)
        c = _span("c", start_offset_ms=100, duration_ms=200)
        result = match_by_timestamp(orphan, [c])
        assert result is not None
        sid, conf = result
        assert sid == "c"
        assert conf == 1.0

    def test_too_far(self):
        orphan = _span("o", start_offset_ms=5000)
        c = _span("c", start_offset_ms=0, duration_ms=100)
        result = match_by_timestamp(orphan, [c], max_gap_ms=1000.0)
        assert result is None

    def test_empty_candidates(self):
        orphan = _span("o")
        assert match_by_timestamp(orphan, []) is None

    def test_confidence_decreases_with_distance(self):
        orphan = _span("o", start_offset_ms=600)
        c = _span("c", start_offset_ms=0, duration_ms=100)
        result = match_by_timestamp(orphan, [c], max_gap_ms=1000.0)
        assert result is not None
        _, conf = result
        assert 0.0 < conf < 1.0

    def test_exact_boundary_match(self):
        orphan = _span("o", start_offset_ms=200)
        c = _span("c", start_offset_ms=100, duration_ms=100)  # ends at 200
        result = match_by_timestamp(orphan, [c])
        assert result is not None
        _, conf = result
        assert conf == 1.0


# -- match_by_name --


class TestMatchByName:
    def test_same_type(self):
        orphan = _span("o", span_type="llm_call")
        c = _span("c", span_type="llm_call")
        result = match_by_name(orphan, [c])
        assert result is not None
        sid, conf = result
        assert sid == "c"
        assert conf == 0.5

    def test_no_match(self):
        orphan = _span("o", span_type="llm_call")
        c = _span("c", span_type="tool_call")
        assert match_by_name(orphan, [c]) is None

    def test_empty_candidates(self):
        orphan = _span("o")
        assert match_by_name(orphan, []) is None

    def test_picks_closest_in_time(self):
        orphan = _span("o", span_type="llm_call", start_offset_ms=500)
        c1 = _span("c1", span_type="llm_call", start_offset_ms=0)
        c2 = _span("c2", span_type="llm_call", start_offset_ms=400)
        result = match_by_name(orphan, [c1, c2])
        assert result is not None
        sid, _ = result
        assert sid == "c2"

    def test_confidence_fixed_at_half(self):
        orphan = _span("o", span_type="custom")
        c = _span("c", span_type="custom", start_offset_ms=0)
        result = match_by_name(orphan, [c])
        assert result is not None
        _, conf = result
        assert conf == 0.5


# -- recover_orphans --


class TestRecoverOrphans:
    def test_combines_methods(self):
        # orphan1: close in time -> timestamp match
        orphan1 = _span("o1", span_type="llm_call", start_offset_ms=150)
        # orphan2: far in time but same type -> name match
        orphan2 = _span("o2", span_type="tool_call", start_offset_ms=5000)
        active = [
            _span("a1", span_type="llm_call", start_offset_ms=100, duration_ms=100),
            _span("a2", span_type="tool_call", start_offset_ms=0, duration_ms=100),
        ]
        result = recover_orphans([orphan1, orphan2], active, max_gap_ms=1000.0)
        assert result.total_orphans == 2
        assert result.recovered == 2
        assert result.truly_lost == 0

        methods = {m.match_method for m in result.matches}
        assert "timestamp" in methods
        assert "name" in methods

    def test_all_lost(self):
        orphan = _span("o", span_type="custom", start_offset_ms=50000)
        active = [_span("a", span_type="llm_call", start_offset_ms=0)]
        result = recover_orphans([orphan], active, max_gap_ms=100.0)
        assert result.recovered == 0
        assert result.truly_lost == 1
        assert result.recovery_rate == 0.0

    def test_empty_orphans(self):
        result = recover_orphans([], [_span("a")])
        assert result.total_orphans == 0
        assert result.recovered == 0
        assert result.recovery_rate == 0.0

    def test_empty_active(self):
        orphan = _span("o")
        result = recover_orphans([orphan], [])
        assert result.truly_lost == 1

    def test_recovery_rate_calculation(self):
        orphans = [
            _span("o1", start_offset_ms=50),
            _span("o2", span_type="custom", start_offset_ms=50000),
        ]
        active = [_span("a", start_offset_ms=0, duration_ms=200)]
        result = recover_orphans(orphans, active, max_gap_ms=1000.0)
        assert result.recovery_rate == 0.5


# -- reattach_orphans --


class TestReattachOrphans:
    def test_updates_parent_id(self):
        orphan = _span("o", parent_id="missing")
        root = _span("root")
        recovery = RecoveryResult(
            total_orphans=1,
            recovered=1,
            truly_lost=0,
            recovery_rate=1.0,
            matches=[
                OrphanSpan(span=orphan, matched_to="root", match_confidence=0.9, match_method="timestamp"),
            ],
        )
        result = reattach_orphans([orphan, root], recovery)
        updated = [s for s in result if s.id == "o"][0]
        assert updated.parent_id == "root"

    def test_doesnt_mutate_originals(self):
        orphan = _span("o", parent_id="missing")
        recovery = RecoveryResult(
            total_orphans=1,
            recovered=1,
            truly_lost=0,
            recovery_rate=1.0,
            matches=[
                OrphanSpan(span=orphan, matched_to="new_parent", match_confidence=0.8, match_method="timestamp"),
            ],
        )
        reattach_orphans([orphan], recovery)
        assert orphan.parent_id == "missing"

    def test_unmatched_spans_unchanged(self):
        normal = _span("n", parent_id="existing")
        recovery = RecoveryResult(
            total_orphans=0, recovered=0, truly_lost=0, recovery_rate=0.0
        )
        result = reattach_orphans([normal], recovery)
        assert result[0].parent_id == "existing"

    def test_empty_spans(self):
        recovery = RecoveryResult(
            total_orphans=0, recovered=0, truly_lost=0, recovery_rate=0.0
        )
        assert reattach_orphans([], recovery) == []


# -- RecoveryResult --


class TestRecoveryResult:
    def test_as_dict_from_dict_roundtrip(self):
        s = _span("o1", parent_id="missing")
        original = RecoveryResult(
            total_orphans=2,
            recovered=1,
            truly_lost=1,
            recovery_rate=0.5,
            matches=[
                OrphanSpan(span=s, matched_to="target", match_confidence=0.8, match_method="timestamp"),
            ],
        )
        d = original.as_dict()
        restored = RecoveryResult.from_dict(d)
        assert restored.total_orphans == original.total_orphans
        assert restored.recovered == original.recovered
        assert restored.truly_lost == original.truly_lost
        assert restored.recovery_rate == original.recovery_rate
        assert len(restored.matches) == 1
        assert restored.matches[0].matched_to == "target"
        assert restored.matches[0].match_confidence == 0.8
        assert restored.matches[0].match_method == "timestamp"

    def test_format_text_output(self):
        s = _span("o1")
        result = RecoveryResult(
            total_orphans=1,
            recovered=1,
            truly_lost=0,
            recovery_rate=1.0,
            matches=[
                OrphanSpan(span=s, matched_to="parent1", match_confidence=0.95, match_method="timestamp"),
            ],
        )
        text = result.format_text()
        assert "1/1 recovered" in text
        assert "parent1" in text
        assert "timestamp" in text

    def test_format_text_lost(self):
        s = _span("lost1")
        result = RecoveryResult(
            total_orphans=1,
            recovered=0,
            truly_lost=1,
            recovery_rate=0.0,
            matches=[OrphanSpan(span=s)],
        )
        text = result.format_text()
        assert "lost" in text
        assert "Truly lost: 1" in text

    def test_empty_matches(self):
        result = RecoveryResult(
            total_orphans=0,
            recovered=0,
            truly_lost=0,
            recovery_rate=0.0,
        )
        d = result.as_dict()
        assert d["matches"] == []


# -- OrphanSpan --


class TestOrphanSpan:
    def test_as_dict(self):
        s = _span("o1")
        o = OrphanSpan(span=s, matched_to="target", match_confidence=0.9, match_method="timestamp")
        d = o.as_dict()
        assert d["matched_to"] == "target"
        assert d["match_confidence"] == 0.9
        assert d["match_method"] == "timestamp"
        assert d["span"]["id"] == "o1"

    def test_defaults(self):
        s = _span("o1")
        o = OrphanSpan(span=s)
        assert o.matched_to is None
        assert o.match_confidence == 0.0
        assert o.match_method == ""
