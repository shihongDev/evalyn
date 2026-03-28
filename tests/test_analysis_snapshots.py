"""Tests for analysis_snapshots module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.analysis_snapshots import (
    AnalysisSnapshot,
    SnapshotComparison,
    SnapshotStore,
    compare_snapshots,
    create_snapshot,
    find_trend,
    render_snapshot_timeline,
)


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    def test_auto_id(self):
        s = create_snapshot("run-1", {"acc": 0.9})
        assert s.snapshot_id  # non-empty
        assert len(s.snapshot_id) == 36  # UUID format

    def test_auto_timestamp(self):
        s = create_snapshot("run-1", {"acc": 0.9})
        assert "T" in s.timestamp  # ISO format

    def test_run_id_preserved(self):
        s = create_snapshot("run-42", {"acc": 0.5})
        assert s.run_id == "run-42"

    def test_metrics_copied(self):
        orig = {"acc": 0.9, "f1": 0.8}
        s = create_snapshot("r", orig)
        assert s.metrics == {"acc": 0.9, "f1": 0.8}
        # Mutation of original should not affect snapshot
        orig["acc"] = 0.0
        assert s.metrics["acc"] == 0.9

    def test_pass_rate_default(self):
        s = create_snapshot("r", {})
        assert s.pass_rate == 0.0

    def test_pass_rate_custom(self):
        s = create_snapshot("r", {}, pass_rate=0.75)
        assert s.pass_rate == 0.75

    def test_total_items_default(self):
        s = create_snapshot("r", {})
        assert s.total_items == 0

    def test_total_items_custom(self):
        s = create_snapshot("r", {}, total_items=50)
        assert s.total_items == 50


# ---------------------------------------------------------------------------
# AnalysisSnapshot as_dict / from_dict
# ---------------------------------------------------------------------------


class TestAnalysisSnapshotRoundtrip:
    def test_roundtrip(self):
        s = create_snapshot("run-1", {"acc": 0.9, "f1": 0.85}, pass_rate=0.8, total_items=10)
        d = s.as_dict()
        s2 = AnalysisSnapshot.from_dict(d)
        assert s2.snapshot_id == s.snapshot_id
        assert s2.timestamp == s.timestamp
        assert s2.run_id == s.run_id
        assert s2.metrics == s.metrics
        assert s2.pass_rate == s.pass_rate
        assert s2.total_items == s.total_items

    def test_from_dict_defaults(self):
        d = {"snapshot_id": "abc", "timestamp": "2026-01-01T00:00:00"}
        s = AnalysisSnapshot.from_dict(d)
        assert s.run_id == ""
        assert s.metrics == {}
        assert s.pass_rate == 0.0
        assert s.total_items == 0
        assert s.metadata == {}

    def test_metadata_roundtrip(self):
        s = AnalysisSnapshot(
            snapshot_id="x", timestamp="t", metadata={"tag": "nightly"}
        )
        d = s.as_dict()
        s2 = AnalysisSnapshot.from_dict(d)
        assert s2.metadata == {"tag": "nightly"}


# ---------------------------------------------------------------------------
# SnapshotStore
# ---------------------------------------------------------------------------


class TestSnapshotStore:
    def test_empty_store(self):
        store = SnapshotStore()
        assert store.count == 0
        assert store.get("missing") is None
        assert store.get_latest() is None
        assert store.list_all() == []

    def test_add_and_get(self):
        store = SnapshotStore()
        s = create_snapshot("r1", {"acc": 0.9})
        store.add(s)
        assert store.count == 1
        assert store.get(s.snapshot_id) is s

    def test_get_returns_none_for_unknown(self):
        store = SnapshotStore()
        store.add(create_snapshot("r", {}))
        assert store.get("no-such-id") is None

    def test_get_latest(self):
        store = SnapshotStore()
        s1 = create_snapshot("r1", {"acc": 0.5})
        s2 = create_snapshot("r2", {"acc": 0.9})
        store.add(s1)
        store.add(s2)
        assert store.get_latest() is s2

    def test_list_all(self):
        store = SnapshotStore()
        s1 = create_snapshot("r1", {})
        s2 = create_snapshot("r2", {})
        store.add(s1)
        store.add(s2)
        result = store.list_all()
        assert len(result) == 2
        assert result[0] is s1
        assert result[1] is s2

    def test_count_property(self):
        store = SnapshotStore()
        for i in range(5):
            store.add(create_snapshot(f"r{i}", {}))
        assert store.count == 5

    def test_single_snapshot_store(self):
        store = SnapshotStore()
        s = create_snapshot("r", {"x": 1.0})
        store.add(s)
        assert store.get_latest() is s
        assert store.list_all() == [s]


# ---------------------------------------------------------------------------
# SnapshotStore serialization
# ---------------------------------------------------------------------------


class TestSnapshotStoreSerialization:
    def test_as_dict_from_dict_roundtrip(self):
        store = SnapshotStore()
        store.add(create_snapshot("r1", {"acc": 0.9}))
        store.add(create_snapshot("r2", {"acc": 0.8}))
        d = store.as_dict()
        store2 = SnapshotStore.from_dict(d)
        assert store2.count == 2
        assert store2.list_all()[0].run_id == "r1"
        assert store2.list_all()[1].run_id == "r2"

    def test_to_json_from_json_roundtrip(self):
        store = SnapshotStore()
        store.add(create_snapshot("r1", {"f1": 0.7}, pass_rate=0.6, total_items=20))
        json_str = store.to_json()
        store2 = SnapshotStore.from_json(json_str)
        assert store2.count == 1
        s = store2.list_all()[0]
        assert s.run_id == "r1"
        assert s.metrics == {"f1": 0.7}
        assert s.pass_rate == 0.6
        assert s.total_items == 20

    def test_to_json_is_valid_json(self):
        store = SnapshotStore()
        store.add(create_snapshot("r", {}))
        parsed = json.loads(store.to_json())
        assert "snapshots" in parsed

    def test_empty_store_roundtrip(self):
        store = SnapshotStore()
        store2 = SnapshotStore.from_json(store.to_json())
        assert store2.count == 0


# ---------------------------------------------------------------------------
# compare_snapshots
# ---------------------------------------------------------------------------


class TestCompareSnapshots:
    def _make(self, sid: str, metrics: dict, pass_rate: float = 0.0) -> AnalysisSnapshot:
        return AnalysisSnapshot(
            snapshot_id=sid, timestamp="t", metrics=metrics, pass_rate=pass_rate
        )

    def test_improved_metric(self):
        a = self._make("a", {"acc": 0.7}, pass_rate=0.7)
        b = self._make("b", {"acc": 0.9}, pass_rate=0.9)
        cmp = compare_snapshots(a, b)
        assert "acc" in cmp.improved_metrics
        assert cmp.degraded_metrics == []
        assert cmp.pass_rate_delta == pytest.approx(0.2)

    def test_degraded_metric(self):
        a = self._make("a", {"acc": 0.9})
        b = self._make("b", {"acc": 0.6})
        cmp = compare_snapshots(a, b)
        assert "acc" in cmp.degraded_metrics
        assert cmp.improved_metrics == []

    def test_unchanged_within_threshold(self):
        a = self._make("a", {"acc": 0.80})
        b = self._make("b", {"acc": 0.805})
        cmp = compare_snapshots(a, b, threshold=0.01)
        assert cmp.improved_metrics == []
        assert cmp.degraded_metrics == []

    def test_metric_deltas(self):
        a = self._make("a", {"x": 0.5, "y": 0.3})
        b = self._make("b", {"x": 0.8, "y": 0.1})
        cmp = compare_snapshots(a, b)
        assert cmp.metric_deltas["x"] == pytest.approx(0.3)
        assert cmp.metric_deltas["y"] == pytest.approx(-0.2)

    def test_pass_rate_delta(self):
        a = self._make("a", {}, pass_rate=0.5)
        b = self._make("b", {}, pass_rate=0.8)
        cmp = compare_snapshots(a, b)
        assert cmp.pass_rate_delta == pytest.approx(0.3)

    def test_new_metric_in_b(self):
        a = self._make("a", {"x": 0.5})
        b = self._make("b", {"x": 0.5, "y": 0.9})
        cmp = compare_snapshots(a, b)
        # y goes from 0.0 (missing) to 0.9
        assert "y" in cmp.improved_metrics

    def test_metric_missing_in_b(self):
        a = self._make("a", {"x": 0.5, "y": 0.9})
        b = self._make("b", {"x": 0.5})
        cmp = compare_snapshots(a, b)
        # y goes from 0.9 to 0.0 (missing)
        assert "y" in cmp.degraded_metrics


# ---------------------------------------------------------------------------
# SnapshotComparison format_text
# ---------------------------------------------------------------------------


class TestSnapshotComparisonFormat:
    def test_format_text_contains_ids(self):
        cmp = SnapshotComparison(
            snapshot_a_id="snap-a",
            snapshot_b_id="snap-b",
            pass_rate_delta=0.1,
            improved_metrics=["acc"],
        )
        text = cmp.format_text()
        assert "snap-a" in text
        assert "snap-b" in text
        assert "Improved" in text

    def test_format_text_degraded(self):
        cmp = SnapshotComparison(
            snapshot_a_id="a",
            snapshot_b_id="b",
            degraded_metrics=["f1"],
        )
        text = cmp.format_text()
        assert "Degraded" in text
        assert "f1" in text

    def test_as_dict(self):
        cmp = SnapshotComparison(
            snapshot_a_id="a",
            snapshot_b_id="b",
            metric_deltas={"x": 0.1},
            pass_rate_delta=0.05,
            improved_metrics=["x"],
            degraded_metrics=[],
        )
        d = cmp.as_dict()
        assert d["snapshot_a_id"] == "a"
        assert d["metric_deltas"] == {"x": 0.1}


# ---------------------------------------------------------------------------
# find_trend
# ---------------------------------------------------------------------------


class TestFindTrend:
    def test_extracts_series(self):
        snaps = [
            AnalysisSnapshot(snapshot_id=str(i), timestamp="t", metrics={"acc": 0.5 + i * 0.1})
            for i in range(4)
        ]
        trend = find_trend(snaps, "acc")
        assert len(trend) == 4
        assert trend[0] == pytest.approx(0.5)
        assert trend[3] == pytest.approx(0.8)

    def test_missing_metric_returns_zero(self):
        snaps = [
            AnalysisSnapshot(snapshot_id="1", timestamp="t", metrics={"acc": 0.9}),
            AnalysisSnapshot(snapshot_id="2", timestamp="t", metrics={}),
        ]
        trend = find_trend(snaps, "acc")
        assert trend == [pytest.approx(0.9), 0.0]

    def test_empty_snapshots(self):
        assert find_trend([], "acc") == []


# ---------------------------------------------------------------------------
# render_snapshot_timeline
# ---------------------------------------------------------------------------


class TestRenderTimeline:
    def test_output_has_header(self):
        snaps = [
            AnalysisSnapshot(snapshot_id="s", timestamp="t", metrics={"acc": 0.5})
        ]
        text = render_snapshot_timeline(snaps, "acc")
        assert "Timeline: acc" in text

    def test_output_has_bars(self):
        snaps = [
            AnalysisSnapshot(snapshot_id=str(i), timestamp="t", metrics={"acc": 0.1 * (i + 1)})
            for i in range(3)
        ]
        text = render_snapshot_timeline(snaps, "acc")
        assert "#" in text

    def test_empty_snapshots(self):
        text = render_snapshot_timeline([], "acc")
        assert "No data" in text

    def test_single_snapshot(self):
        snaps = [AnalysisSnapshot(snapshot_id="s", timestamp="t", metrics={"x": 1.0})]
        text = render_snapshot_timeline(snaps, "x")
        lines = text.strip().split("\n")
        assert len(lines) == 2  # header + one bar


# Need pytest for approx
import pytest
