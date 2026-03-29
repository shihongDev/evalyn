"""Tests for provider_diversity module."""

from __future__ import annotations

import pytest

from evalyn_sdk.provider_diversity import (
    DiversityResult,
    ProviderConfig,
    ProviderPool,
    ProviderResult,
    compare_provider_quality,
    format_provider_report,
    merge_provider_results,
)


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_defaults(self):
        pc = ProviderConfig(provider_id="a", name="Alpha")
        assert pc.model == ""
        assert pc.weight == 1.0
        assert pc.enabled is True

    def test_custom_fields(self):
        pc = ProviderConfig("b", "Beta", model="gpt-4o", weight=2.5, enabled=False)
        assert pc.provider_id == "b"
        assert pc.name == "Beta"
        assert pc.model == "gpt-4o"
        assert pc.weight == 2.5
        assert pc.enabled is False

    def test_as_dict(self):
        pc = ProviderConfig("x", "X", model="m", weight=3.0, enabled=True)
        d = pc.as_dict()
        assert d["provider_id"] == "x"
        assert d["model"] == "m"
        assert d["weight"] == 3.0
        assert d["enabled"] is True

    def test_from_dict_full(self):
        d = {
            "provider_id": "z",
            "name": "Zeta",
            "model": "claude",
            "weight": 0.5,
            "enabled": False,
        }
        pc = ProviderConfig.from_dict(d)
        assert pc.provider_id == "z"
        assert pc.name == "Zeta"
        assert pc.model == "claude"
        assert pc.weight == 0.5
        assert pc.enabled is False

    def test_from_dict_defaults(self):
        d = {"provider_id": "q"}
        pc = ProviderConfig.from_dict(d)
        assert pc.name == "q"
        assert pc.model == ""
        assert pc.weight == 1.0
        assert pc.enabled is True

    def test_roundtrip(self):
        pc = ProviderConfig("rt", "Roundtrip", "model-x", 4.0, False)
        assert ProviderConfig.from_dict(pc.as_dict()).as_dict() == pc.as_dict()


# ---------------------------------------------------------------------------
# ProviderResult
# ---------------------------------------------------------------------------


class TestProviderResult:
    def test_basic(self):
        pr = ProviderResult(items=[{"a": 1}], provider_id="p1", item_count=1)
        assert pr.quality_score == 0.0

    def test_as_dict(self):
        pr = ProviderResult([{"x": 2}], "p2", 1, 0.85)
        d = pr.as_dict()
        assert d["provider_id"] == "p2"
        assert d["quality_score"] == 0.85
        assert len(d["items"]) == 1

    def test_from_dict(self):
        d = {
            "items": [{"k": "v"}],
            "provider_id": "p3",
            "item_count": 1,
            "quality_score": 0.9,
        }
        pr = ProviderResult.from_dict(d)
        assert pr.provider_id == "p3"
        assert pr.quality_score == 0.9

    def test_from_dict_defaults(self):
        d = {"provider_id": "p4"}
        pr = ProviderResult.from_dict(d)
        assert pr.items == []
        assert pr.item_count == 0
        assert pr.quality_score == 0.0

    def test_roundtrip(self):
        pr = ProviderResult([{"a": 1}, {"b": 2}], "p5", 2, 0.7)
        assert ProviderResult.from_dict(pr.as_dict()).as_dict() == pr.as_dict()


# ---------------------------------------------------------------------------
# DiversityResult
# ---------------------------------------------------------------------------


class TestDiversityResult:
    def test_basic(self):
        dr = DiversityResult(all_items=[], per_provider=[], total_items=0, providers_used=0)
        assert dr.total_items == 0

    def test_as_dict(self):
        pr = ProviderResult([{"i": 1}], "px", 1, 0.5)
        dr = DiversityResult([{"i": 1}], [pr], 1, 1)
        d = dr.as_dict()
        assert d["total_items"] == 1
        assert len(d["per_provider"]) == 1

    def test_from_dict(self):
        d = {
            "all_items": [{"v": 1}],
            "per_provider": [
                {"items": [{"v": 1}], "provider_id": "z", "item_count": 1}
            ],
            "total_items": 1,
            "providers_used": 1,
        }
        dr = DiversityResult.from_dict(d)
        assert dr.providers_used == 1
        assert dr.per_provider[0].provider_id == "z"

    def test_roundtrip(self):
        pr = ProviderResult([{"x": 1}], "rr", 1, 0.3)
        dr = DiversityResult([{"x": 1}], [pr], 1, 1)
        d2 = DiversityResult.from_dict(dr.as_dict())
        assert d2.as_dict() == dr.as_dict()


# ---------------------------------------------------------------------------
# ProviderPool
# ---------------------------------------------------------------------------


def _make_pool(*specs):
    """Helper: specs are (id, weight, enabled) tuples."""
    providers = [
        ProviderConfig(provider_id=s[0], name=s[0], weight=s[1], enabled=s[2])
        for s in specs
    ]
    return ProviderPool(providers)


class TestProviderPool:
    def test_list_enabled(self):
        pool = _make_pool(("a", 1, True), ("b", 1, False), ("c", 1, True))
        enabled = pool.list_enabled()
        ids = [p.provider_id for p in enabled]
        assert ids == ["a", "c"]

    def test_get_provider_found(self):
        pool = _make_pool(("x", 1, True))
        assert pool.get_provider("x") is not None

    def test_get_provider_not_found(self):
        pool = _make_pool(("x", 1, True))
        assert pool.get_provider("missing") is None

    def test_next_provider_single(self):
        pool = _make_pool(("only", 1, True))
        p = pool.next_provider()
        assert p.provider_id == "only"

    def test_next_provider_round_robin(self):
        pool = _make_pool(("a", 1, True), ("b", 1, True))
        ids = [pool.next_provider().provider_id for _ in range(4)]
        assert ids == ["a", "b", "a", "b"]

    def test_next_provider_skips_disabled(self):
        pool = _make_pool(("a", 1, True), ("b", 1, False))
        for _ in range(5):
            assert pool.next_provider().provider_id == "a"

    def test_next_provider_no_enabled_raises(self):
        pool = _make_pool(("a", 1, False))
        with pytest.raises(ValueError, match="No enabled"):
            pool.next_provider()

    def test_next_provider_weighted(self):
        # Weight=2 should appear twice as often as weight=1
        pool = _make_pool(("heavy", 2, True), ("light", 1, True))
        ids = [pool.next_provider().provider_id for _ in range(3)]
        assert ids.count("heavy") == 2
        assert ids.count("light") == 1

    def test_allocate_items_equal_weight(self):
        pool = _make_pool(("a", 1, True), ("b", 1, True))
        alloc = pool.allocate_items(10)
        assert alloc["a"] == 5
        assert alloc["b"] == 5

    def test_allocate_items_unequal_weight(self):
        pool = _make_pool(("a", 3, True), ("b", 1, True))
        alloc = pool.allocate_items(100)
        assert alloc["a"] == 75
        assert alloc["b"] == 25

    def test_allocate_items_total_preserved(self):
        pool = _make_pool(("a", 1, True), ("b", 1, True), ("c", 1, True))
        alloc = pool.allocate_items(10)
        assert sum(alloc.values()) == 10

    def test_allocate_items_remainder_distributed(self):
        pool = _make_pool(("a", 1, True), ("b", 1, True), ("c", 1, True))
        alloc = pool.allocate_items(7)
        assert sum(alloc.values()) == 7

    def test_allocate_items_skips_disabled(self):
        pool = _make_pool(("a", 1, True), ("b", 1, False))
        alloc = pool.allocate_items(10)
        assert "b" not in alloc
        assert alloc["a"] == 10

    def test_allocate_items_no_enabled(self):
        pool = _make_pool(("a", 1, False))
        assert pool.allocate_items(10) == {}

    def test_allocate_items_zero(self):
        pool = _make_pool(("a", 1, True), ("b", 1, True))
        alloc = pool.allocate_items(0)
        assert alloc["a"] == 0
        assert alloc["b"] == 0

    def test_allocate_items_single_provider(self):
        pool = _make_pool(("solo", 5, True))
        alloc = pool.allocate_items(42)
        assert alloc["solo"] == 42


# ---------------------------------------------------------------------------
# merge_provider_results
# ---------------------------------------------------------------------------


class TestMergeProviderResults:
    def test_empty(self):
        dr = merge_provider_results([])
        assert dr.total_items == 0
        assert dr.providers_used == 0

    def test_single_provider(self):
        pr = ProviderResult([{"id": "1"}, {"id": "2"}], "p1", 2, 0.8)
        dr = merge_provider_results([pr])
        assert dr.total_items == 2
        assert dr.providers_used == 1
        assert dr.all_items[0]["metadata"]["_provider_id"] == "p1"

    def test_multiple_providers(self):
        r1 = ProviderResult([{"v": 1}], "a", 1, 0.9)
        r2 = ProviderResult([{"v": 2}, {"v": 3}], "b", 2, 0.7)
        dr = merge_provider_results([r1, r2])
        assert dr.total_items == 3
        assert dr.providers_used == 2

    def test_metadata_tag_preserved(self):
        item = {"id": "x", "metadata": {"existing": True}}
        pr = ProviderResult([item], "tag_test", 1)
        dr = merge_provider_results([pr])
        meta = dr.all_items[0]["metadata"]
        assert meta["existing"] is True
        assert meta["_provider_id"] == "tag_test"

    def test_metadata_created_when_missing(self):
        pr = ProviderResult([{"id": "bare"}], "p", 1)
        dr = merge_provider_results([pr])
        assert dr.all_items[0]["metadata"]["_provider_id"] == "p"


# ---------------------------------------------------------------------------
# compare_provider_quality
# ---------------------------------------------------------------------------


class TestCompareProviderQuality:
    def test_empty(self):
        assert compare_provider_quality([]) == {}

    def test_basic(self):
        r1 = ProviderResult([], "a", 0, 0.9)
        r2 = ProviderResult([], "b", 0, 0.7)
        q = compare_provider_quality([r1, r2])
        assert q == {"a": 0.9, "b": 0.7}


# ---------------------------------------------------------------------------
# format_provider_report
# ---------------------------------------------------------------------------


class TestFormatProviderReport:
    def test_contains_header(self):
        dr = DiversityResult([], [], 0, 0)
        report = format_provider_report(dr)
        assert "Provider Diversity Report" in report

    def test_contains_provider_details(self):
        pr = ProviderResult([{"x": 1}], "gemini", 1, 0.95)
        dr = DiversityResult([{"x": 1}], [pr], 1, 1)
        report = format_provider_report(dr)
        assert "gemini" in report
        assert "0.95" in report
        assert "100.0%" in report

    def test_multiple_providers_listed(self):
        r1 = ProviderResult([], "a", 3, 0.8)
        r2 = ProviderResult([], "b", 7, 0.6)
        dr = DiversityResult([], [r1, r2], 10, 2)
        report = format_provider_report(dr)
        assert "a" in report
        assert "b" in report
        assert "Providers used: 2" in report
