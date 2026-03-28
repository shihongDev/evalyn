"""Tests for calibration scope control."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.scope_control import (
    ScopeFilter,
    ScopeResult,
    apply_scope_filter,
    create_exclude_scope,
    create_include_scope,
    create_metadata_scope,
    describe_scope,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _items() -> list:
    return [
        {"id": "a", "metadata": {"domain": "math", "level": "easy"}},
        {"id": "b", "metadata": {"domain": "science", "level": "hard"}},
        {"id": "c", "metadata": {"domain": "math", "level": "hard"}},
        {"id": "d", "metadata": {"domain": "science", "level": "easy"}},
        {"id": "e", "metadata": {"domain": "math", "level": "medium"}},
    ]


# -------------------------------------------------------------------
# ScopeFilter - as_dict / from_dict
# -------------------------------------------------------------------

class TestScopeFilterSerialization:
    def test_as_dict(self):
        sf = ScopeFilter(include_ids=["a"], exclude_ids=["b"], metadata_key="k", metadata_value="v", max_items=10)
        d = sf.as_dict()
        assert d["include_ids"] == ["a"]
        assert d["exclude_ids"] == ["b"]
        assert d["metadata_key"] == "k"
        assert d["metadata_value"] == "v"
        assert d["max_items"] == 10

    def test_from_dict_roundtrip(self):
        sf = ScopeFilter(include_ids=["x"], max_items=5)
        restored = ScopeFilter.from_dict(sf.as_dict())
        assert restored.include_ids == ["x"]
        assert restored.max_items == 5
        assert restored.exclude_ids == []
        assert restored.metadata_key == ""

    def test_from_dict_defaults(self):
        restored = ScopeFilter.from_dict({})
        assert restored.include_ids == []
        assert restored.exclude_ids == []
        assert restored.metadata_key == ""
        assert restored.metadata_value == ""
        assert restored.max_items is None


# -------------------------------------------------------------------
# ScopeResult
# -------------------------------------------------------------------

class TestScopeResult:
    def test_as_dict(self):
        sr = ScopeResult(original_count=10, filtered_count=5, filter_description="include 3 ids")
        d = sr.as_dict()
        assert d["original_count"] == 10
        assert d["filtered_count"] == 5
        assert d["filter_description"] == "include 3 ids"

    def test_format_text(self):
        sr = ScopeResult(original_count=10, filtered_count=3, filter_description="include 2 ids")
        text = sr.format_text()
        assert "3/10" in text
        assert "include 2 ids" in text


# -------------------------------------------------------------------
# apply_scope_filter - include_ids
# -------------------------------------------------------------------

class TestIncludeIds:
    def test_include_filters(self):
        items = _items()
        filtered, result = apply_scope_filter(items, ScopeFilter(include_ids=["a", "c"]))
        assert len(filtered) == 2
        assert {it["id"] for it in filtered} == {"a", "c"}
        assert result.original_count == 5
        assert result.filtered_count == 2

    def test_include_nonexistent_ids(self):
        items = _items()
        filtered, result = apply_scope_filter(items, ScopeFilter(include_ids=["z"]))
        assert len(filtered) == 0
        assert result.filtered_count == 0


# -------------------------------------------------------------------
# apply_scope_filter - exclude_ids
# -------------------------------------------------------------------

class TestExcludeIds:
    def test_exclude_removes(self):
        items = _items()
        filtered, result = apply_scope_filter(items, ScopeFilter(exclude_ids=["b", "d"]))
        assert len(filtered) == 3
        ids = {it["id"] for it in filtered}
        assert "b" not in ids
        assert "d" not in ids

    def test_exclude_nonexistent_no_effect(self):
        items = _items()
        filtered, _ = apply_scope_filter(items, ScopeFilter(exclude_ids=["z"]))
        assert len(filtered) == 5


# -------------------------------------------------------------------
# apply_scope_filter - metadata filter
# -------------------------------------------------------------------

class TestMetadataFilter:
    def test_metadata_matches(self):
        items = _items()
        scope = ScopeFilter(metadata_key="domain", metadata_value="math")
        filtered, result = apply_scope_filter(items, scope)
        assert len(filtered) == 3
        assert all(it["metadata"]["domain"] == "math" for it in filtered)

    def test_metadata_no_match(self):
        items = _items()
        scope = ScopeFilter(metadata_key="domain", metadata_value="history")
        filtered, _ = apply_scope_filter(items, scope)
        assert len(filtered) == 0

    def test_metadata_missing_key(self):
        items = [{"id": "a", "metadata": {}}, {"id": "b"}]
        scope = ScopeFilter(metadata_key="x", metadata_value="y")
        filtered, _ = apply_scope_filter(items, scope)
        assert len(filtered) == 0


# -------------------------------------------------------------------
# apply_scope_filter - max_items
# -------------------------------------------------------------------

class TestMaxItems:
    def test_limits_count(self):
        items = _items()
        filtered, result = apply_scope_filter(items, ScopeFilter(max_items=2))
        assert len(filtered) == 2
        assert result.filtered_count == 2

    def test_max_items_larger_than_list(self):
        items = _items()
        filtered, _ = apply_scope_filter(items, ScopeFilter(max_items=100))
        assert len(filtered) == 5


# -------------------------------------------------------------------
# apply_scope_filter - combined filters
# -------------------------------------------------------------------

class TestCombinedFilters:
    def test_include_then_exclude(self):
        items = _items()
        scope = ScopeFilter(include_ids=["a", "b", "c"], exclude_ids=["b"])
        filtered, _ = apply_scope_filter(items, scope)
        assert {it["id"] for it in filtered} == {"a", "c"}

    def test_metadata_then_max(self):
        items = _items()
        scope = ScopeFilter(metadata_key="domain", metadata_value="math", max_items=2)
        filtered, result = apply_scope_filter(items, scope)
        assert len(filtered) == 2
        assert all(it["metadata"]["domain"] == "math" for it in filtered)

    def test_all_filters(self):
        items = _items()
        scope = ScopeFilter(
            include_ids=["a", "b", "c", "e"],
            exclude_ids=["b"],
            metadata_key="domain",
            metadata_value="math",
            max_items=1,
        )
        filtered, result = apply_scope_filter(items, scope)
        assert len(filtered) == 1
        assert result.filtered_count == 1


# -------------------------------------------------------------------
# apply_scope_filter - no filters (pass all)
# -------------------------------------------------------------------

class TestNoFilters:
    def test_no_filter_passes_all(self):
        items = _items()
        filtered, result = apply_scope_filter(items, ScopeFilter())
        assert len(filtered) == 5
        assert result.original_count == 5
        assert result.filtered_count == 5

    def test_empty_items(self):
        filtered, result = apply_scope_filter([], ScopeFilter())
        assert len(filtered) == 0
        assert result.original_count == 0
        assert result.filtered_count == 0


# -------------------------------------------------------------------
# Factory functions
# -------------------------------------------------------------------

class TestFactories:
    def test_create_include_scope(self):
        sf = create_include_scope(["a", "b"])
        assert sf.include_ids == ["a", "b"]
        assert sf.exclude_ids == []
        assert sf.metadata_key == ""
        assert sf.max_items is None

    def test_create_exclude_scope(self):
        sf = create_exclude_scope(["x"])
        assert sf.exclude_ids == ["x"]
        assert sf.include_ids == []

    def test_create_metadata_scope(self):
        sf = create_metadata_scope("domain", "math")
        assert sf.metadata_key == "domain"
        assert sf.metadata_value == "math"
        assert sf.include_ids == []


# -------------------------------------------------------------------
# describe_scope
# -------------------------------------------------------------------

class TestDescribeScope:
    def test_no_filter(self):
        assert describe_scope(ScopeFilter()) == "no filter"

    def test_include_only(self):
        desc = describe_scope(ScopeFilter(include_ids=["a", "b"]))
        assert "include 2 ids" in desc

    def test_exclude_only(self):
        desc = describe_scope(ScopeFilter(exclude_ids=["x"]))
        assert "exclude 1 ids" in desc

    def test_metadata_only(self):
        desc = describe_scope(ScopeFilter(metadata_key="k", metadata_value="v"))
        assert "metadata k=v" in desc

    def test_max_items_only(self):
        desc = describe_scope(ScopeFilter(max_items=10))
        assert "max 10 items" in desc

    def test_combined(self):
        sf = ScopeFilter(include_ids=["a"], exclude_ids=["b"], max_items=5)
        desc = describe_scope(sf)
        assert "include" in desc
        assert "exclude" in desc
        assert "max" in desc
