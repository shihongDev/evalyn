"""Tests for datasets_lineage module."""

import pytest
from evalyn_sdk.datasets_lineage import (
    LineageRecord,
    LineageChain,
    LineageReport,
    record_lineage,
    build_lineage_chain,
    build_lineage_report,
    find_orphaned_items,
    render_lineage_tree,
    compute_source_distribution,
)


# ---------------------------------------------------------------------------
# record_lineage
# ---------------------------------------------------------------------------


class TestRecordLineage:
    def test_creates_record(self):
        rec = record_lineage("item-1", "trace", source_id="t-1")
        assert rec.item_id == "item-1"
        assert rec.source_type == "trace"
        assert rec.source_id == "t-1"
        assert rec.created_at != ""

    def test_default_source_type(self):
        rec = record_lineage("item-2", "manual")
        assert rec.source_type == "manual"

    def test_parent_and_transformation(self):
        rec = record_lineage(
            "item-3", "run", parent_id="item-2", transformation="augmented"
        )
        assert rec.parent_item_id == "item-2"
        assert rec.transformation == "augmented"

    def test_timestamp_is_iso(self):
        rec = record_lineage("item-4", "import")
        # ISO format contains "T" separator
        assert "T" in rec.created_at


# ---------------------------------------------------------------------------
# build_lineage_chain
# ---------------------------------------------------------------------------


class TestBuildLineageChain:
    def test_single_record(self):
        rec = LineageRecord(item_id="a", source_type="trace", source_id="t1")
        chain = build_lineage_chain("a", [rec])
        assert chain.item_id == "a"
        assert chain.depth == 1
        assert len(chain.records) == 1

    def test_multi_level(self):
        r1 = LineageRecord(item_id="c", source_type="run", parent_item_id="b")
        r2 = LineageRecord(item_id="b", source_type="trace", parent_item_id="a")
        r3 = LineageRecord(item_id="a", source_type="import")
        chain = build_lineage_chain("c", [r1, r2, r3])
        assert chain.depth == 3
        assert chain.records[0].item_id == "c"
        assert chain.records[1].item_id == "b"
        assert chain.records[2].item_id == "a"

    def test_no_matching_record(self):
        rec = LineageRecord(item_id="x", source_type="trace")
        chain = build_lineage_chain("y", [rec])
        assert chain.depth == 0
        assert chain.records == []

    def test_cycle_protection(self):
        r1 = LineageRecord(item_id="a", source_type="trace", parent_item_id="b")
        r2 = LineageRecord(item_id="b", source_type="trace", parent_item_id="a")
        chain = build_lineage_chain("a", [r1, r2])
        # Should not loop forever; depth should be exactly 2
        assert chain.depth == 2

    def test_empty_records(self):
        chain = build_lineage_chain("a", [])
        assert chain.depth == 0
        assert chain.records == []


# ---------------------------------------------------------------------------
# build_lineage_report
# ---------------------------------------------------------------------------


class TestBuildLineageReport:
    def test_aggregates(self):
        records = [
            LineageRecord(item_id="a", source_type="trace"),
            LineageRecord(item_id="b", source_type="run"),
            LineageRecord(item_id="c", source_type="trace"),
        ]
        report = build_lineage_report(records)
        assert report.total_items == 3
        assert len(report.chains) == 3
        assert report.source_distribution["trace"] == 2
        assert report.source_distribution["run"] == 1

    def test_empty_records(self):
        report = build_lineage_report([])
        assert report.total_items == 0
        assert report.chains == []


# ---------------------------------------------------------------------------
# find_orphaned_items
# ---------------------------------------------------------------------------


class TestFindOrphanedItems:
    def test_finds_orphans(self):
        records = [LineageRecord(item_id="a", source_type="trace")]
        orphans = find_orphaned_items(["a", "b", "c"], records)
        assert orphans == ["b", "c"]

    def test_no_orphans(self):
        records = [
            LineageRecord(item_id="a", source_type="trace"),
            LineageRecord(item_id="b", source_type="run"),
        ]
        orphans = find_orphaned_items(["a", "b"], records)
        assert orphans == []

    def test_all_orphans(self):
        orphans = find_orphaned_items(["x", "y"], [])
        assert orphans == ["x", "y"]

    def test_empty_both(self):
        orphans = find_orphaned_items([], [])
        assert orphans == []


# ---------------------------------------------------------------------------
# render_lineage_tree
# ---------------------------------------------------------------------------


class TestRenderLineageTree:
    def test_single_node(self):
        rec = LineageRecord(item_id="a", source_type="trace", source_id="t1")
        chain = LineageChain(item_id="a", records=[rec], depth=1)
        tree = render_lineage_tree(chain)
        assert "a" in tree
        assert "trace:t1" in tree

    def test_multi_level(self):
        r1 = LineageRecord(item_id="c", source_type="run")
        r2 = LineageRecord(item_id="b", source_type="trace")
        chain = LineageChain(item_id="c", records=[r1, r2], depth=2)
        tree = render_lineage_tree(chain)
        assert "c" in tree
        assert "b" in tree

    def test_empty_chain(self):
        chain = LineageChain(item_id="a", records=[], depth=0)
        tree = render_lineage_tree(chain)
        assert "no lineage" in tree

    def test_transformation_shown(self):
        rec = LineageRecord(
            item_id="a", source_type="run", transformation="augmented"
        )
        chain = LineageChain(item_id="a", records=[rec], depth=1)
        tree = render_lineage_tree(chain)
        assert "[augmented]" in tree


# ---------------------------------------------------------------------------
# compute_source_distribution
# ---------------------------------------------------------------------------


class TestComputeSourceDistribution:
    def test_basic(self):
        records = [
            LineageRecord(item_id="1", source_type="trace"),
            LineageRecord(item_id="2", source_type="trace"),
            LineageRecord(item_id="3", source_type="manual"),
        ]
        dist = compute_source_distribution(records)
        assert dist == {"trace": 2, "manual": 1}

    def test_empty(self):
        dist = compute_source_distribution([])
        assert dist == {}

    def test_single_type(self):
        records = [
            LineageRecord(item_id="a", source_type="import"),
            LineageRecord(item_id="b", source_type="import"),
        ]
        dist = compute_source_distribution(records)
        assert dist == {"import": 2}


# ---------------------------------------------------------------------------
# LineageRecord from_dict
# ---------------------------------------------------------------------------


class TestLineageRecordFromDict:
    def test_roundtrip(self):
        original = record_lineage("x", "trace", source_id="s1")
        d = original.as_dict()
        restored = LineageRecord.from_dict(d)
        assert restored.item_id == original.item_id
        assert restored.source_type == original.source_type
        assert restored.source_id == original.source_id
        assert restored.created_at == original.created_at

    def test_from_empty_dict(self):
        rec = LineageRecord.from_dict({})
        assert rec.item_id == ""
        assert rec.source_type == "trace"


# ---------------------------------------------------------------------------
# LineageReport format_text / as_dict / from_dict
# ---------------------------------------------------------------------------


class TestLineageReportFormatText:
    def test_basic(self):
        records = [
            LineageRecord(item_id="a", source_type="trace"),
            LineageRecord(item_id="b", source_type="run"),
        ]
        report = build_lineage_report(records)
        text = report.format_text()
        assert "2 items" in text
        assert "trace" in text

    def test_orphaned_shown(self):
        report = LineageReport(
            total_items=1,
            orphaned_items=["z"],
        )
        text = report.format_text()
        assert "z" in text


class TestLineageReportRoundtrip:
    def test_as_dict_from_dict(self):
        records = [
            LineageRecord(item_id="a", source_type="trace", source_id="t1"),
            LineageRecord(item_id="b", source_type="run"),
        ]
        original = build_lineage_report(records)
        d = original.as_dict()
        restored = LineageReport.from_dict(d)
        assert restored.total_items == original.total_items
        assert len(restored.chains) == len(original.chains)
        assert restored.source_distribution == original.source_distribution

    def test_from_empty_dict(self):
        restored = LineageReport.from_dict({})
        assert restored.total_items == 0
        assert restored.chains == []


# ---------------------------------------------------------------------------
# LineageChain as_dict / format_text
# ---------------------------------------------------------------------------


class TestLineageChain:
    def test_as_dict(self):
        rec = LineageRecord(item_id="a", source_type="trace")
        chain = LineageChain(item_id="a", records=[rec], depth=1)
        d = chain.as_dict()
        assert d["item_id"] == "a"
        assert d["depth"] == 1
        assert len(d["records"]) == 1

    def test_format_text(self):
        rec = LineageRecord(item_id="a", source_type="trace", source_id="t1")
        chain = LineageChain(item_id="a", records=[rec], depth=1)
        text = chain.format_text()
        assert "a" in text
        assert "trace:t1" in text
