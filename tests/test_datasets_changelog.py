"""
Tests for evalyn_sdk.datasets_changelog.

Run with: uv run pytest tests/test_datasets_changelog.py -v
"""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.datasets_changelog import (
    ChangelogEntry,
    DatasetChangelog,
    create_changelog,
    record_build,
    record_filter,
    record_merge,
    render_changelog,
)


# ---------------------------------------------------------------------------
# ChangelogEntry
# ---------------------------------------------------------------------------


class TestChangelogEntry:
    def test_as_dict_basic(self):
        entry = ChangelogEntry(
            operation="build",
            timestamp="2026-01-01T00:00:00+00:00",
            parameters={"source": "file.jsonl"},
            items_affected=10,
            description="Built dataset",
        )
        d = entry.as_dict()
        assert d["operation"] == "build"
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["parameters"] == {"source": "file.jsonl"}
        assert d["items_affected"] == 10
        assert d["description"] == "Built dataset"

    def test_from_dict_basic(self):
        d = {
            "operation": "filter",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "parameters": {"filter": "length > 10"},
            "items_affected": 5,
            "description": "Filtered short items",
        }
        entry = ChangelogEntry.from_dict(d)
        assert entry.operation == "filter"
        assert entry.items_affected == 5

    def test_from_dict_defaults(self):
        d = {"operation": "build", "timestamp": "2026-01-01T00:00:00+00:00"}
        entry = ChangelogEntry.from_dict(d)
        assert entry.parameters == {}
        assert entry.items_affected == 0
        assert entry.description == ""

    def test_as_dict_from_dict_roundtrip(self):
        entry = ChangelogEntry(
            operation="merge",
            timestamp="2026-03-01T12:00:00+00:00",
            parameters={"sources": ["a", "b"]},
            items_affected=100,
            description="Merged two sets",
        )
        restored = ChangelogEntry.from_dict(entry.as_dict())
        assert restored.operation == entry.operation
        assert restored.timestamp == entry.timestamp
        assert restored.parameters == entry.parameters
        assert restored.items_affected == entry.items_affected
        assert restored.description == entry.description


# ---------------------------------------------------------------------------
# DatasetChangelog
# ---------------------------------------------------------------------------


class TestDatasetChangelog:
    def test_create_empty(self):
        cl = DatasetChangelog()
        assert cl.count == 0
        assert cl.entries == []
        assert cl.dataset_name == ""

    def test_add_entry_increments_count(self):
        cl = DatasetChangelog(dataset_name="test")
        cl.add_entry("build", items_affected=5)
        assert cl.count == 1
        cl.add_entry("filter", items_affected=2)
        assert cl.count == 2

    def test_add_entry_auto_timestamp(self):
        cl = DatasetChangelog()
        cl.add_entry("build")
        entry = cl.entries[0]
        assert entry.timestamp  # non-empty
        assert "T" in entry.timestamp  # ISO format

    def test_add_entry_with_parameters(self):
        cl = DatasetChangelog()
        cl.add_entry("build", parameters={"source": "data.jsonl"}, items_affected=10)
        assert cl.entries[0].parameters == {"source": "data.jsonl"}
        assert cl.entries[0].items_affected == 10

    def test_add_entry_none_parameters_becomes_empty_dict(self):
        cl = DatasetChangelog()
        cl.add_entry("build", parameters=None)
        assert cl.entries[0].parameters == {}

    def test_get_entries_by_operation(self):
        cl = DatasetChangelog()
        cl.add_entry("build")
        cl.add_entry("filter")
        cl.add_entry("build")
        cl.add_entry("merge")
        builds = cl.get_entries_by_operation("build")
        assert len(builds) == 2
        assert all(e.operation == "build" for e in builds)

    def test_get_entries_by_operation_no_match(self):
        cl = DatasetChangelog()
        cl.add_entry("build")
        assert cl.get_entries_by_operation("delete") == []

    def test_latest_property_empty(self):
        cl = DatasetChangelog()
        assert cl.latest is None

    def test_latest_property_single(self):
        cl = DatasetChangelog()
        cl.add_entry("build", description="first")
        assert cl.latest is not None
        assert cl.latest.description == "first"

    def test_latest_property_multiple(self):
        cl = DatasetChangelog()
        cl.add_entry("build", description="first")
        cl.add_entry("filter", description="second")
        cl.add_entry("merge", description="third")
        assert cl.latest is not None
        assert cl.latest.description == "third"

    def test_count_property(self):
        cl = DatasetChangelog()
        assert cl.count == 0
        for i in range(5):
            cl.add_entry("op")
        assert cl.count == 5

    def test_as_dict(self):
        cl = DatasetChangelog(dataset_name="my_ds")
        cl.add_entry("build", items_affected=3)
        d = cl.as_dict()
        assert d["dataset_name"] == "my_ds"
        assert len(d["entries"]) == 1
        assert d["entries"][0]["operation"] == "build"

    def test_from_dict(self):
        d = {
            "dataset_name": "restored",
            "entries": [
                {"operation": "build", "timestamp": "2026-01-01T00:00:00+00:00"},
                {"operation": "filter", "timestamp": "2026-01-02T00:00:00+00:00"},
            ],
        }
        cl = DatasetChangelog.from_dict(d)
        assert cl.dataset_name == "restored"
        assert cl.count == 2

    def test_as_dict_from_dict_roundtrip(self):
        cl = DatasetChangelog(dataset_name="roundtrip")
        cl.add_entry("build", parameters={"x": 1}, items_affected=10, description="b")
        cl.add_entry("filter", items_affected=3, description="f")
        restored = DatasetChangelog.from_dict(cl.as_dict())
        assert restored.dataset_name == cl.dataset_name
        assert restored.count == cl.count
        for orig, rest in zip(cl.entries, restored.entries):
            assert orig.operation == rest.operation
            assert orig.parameters == rest.parameters

    def test_to_json_from_json_roundtrip(self):
        cl = DatasetChangelog(dataset_name="json_test")
        cl.add_entry("build", items_affected=7)
        cl.add_entry("merge", parameters={"sources": ["a", "b"]})
        json_str = cl.to_json()
        restored = DatasetChangelog.from_json(json_str)
        assert restored.dataset_name == "json_test"
        assert restored.count == 2
        assert restored.entries[0].items_affected == 7

    def test_to_json_is_valid_json(self):
        cl = DatasetChangelog(dataset_name="valid")
        cl.add_entry("build")
        parsed = json.loads(cl.to_json())
        assert "dataset_name" in parsed
        assert "entries" in parsed

    def test_format_text_empty(self):
        cl = DatasetChangelog(dataset_name="empty_ds")
        text = cl.format_text()
        assert "empty_ds" in text
        assert "Total entries: 0" in text

    def test_format_text_with_entries(self):
        cl = DatasetChangelog(dataset_name="fmt")
        cl.add_entry("build", items_affected=10, description="Built from source")
        text = cl.format_text()
        assert "build" in text
        assert "Built from source" in text
        assert "Items affected: 10" in text


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateChangelog:
    def test_create_empty(self):
        cl = create_changelog()
        assert cl.count == 0
        assert cl.dataset_name == ""

    def test_create_with_name(self):
        cl = create_changelog("my_dataset")
        assert cl.dataset_name == "my_dataset"
        assert cl.count == 0


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


class TestRecordBuild:
    def test_record_build_basic(self):
        cl = create_changelog("ds")
        result = record_build(cl, source="file.jsonl", item_count=50)
        assert result is cl
        assert cl.count == 1
        assert cl.latest.operation == "build"
        assert cl.latest.items_affected == 50
        assert "file.jsonl" in cl.latest.description

    def test_record_build_with_params(self):
        cl = create_changelog()
        record_build(cl, source="api", item_count=20, params={"format": "json"})
        assert cl.latest.parameters["source"] == "api"
        assert cl.latest.parameters["format"] == "json"


class TestRecordFilter:
    def test_record_filter_basic(self):
        cl = create_changelog()
        record_filter(cl, filter_desc="length > 10", before_count=100, after_count=80)
        assert cl.count == 1
        assert cl.latest.operation == "filter"
        assert cl.latest.items_affected == 20
        assert "100" in cl.latest.description
        assert "80" in cl.latest.description

    def test_record_filter_returns_changelog(self):
        cl = create_changelog()
        result = record_filter(cl, "x", 10, 5)
        assert result is cl


class TestRecordMerge:
    def test_record_merge_basic(self):
        cl = create_changelog()
        record_merge(cl, sources=["a.jsonl", "b.jsonl"], result_count=200)
        assert cl.count == 1
        assert cl.latest.operation == "merge"
        assert cl.latest.items_affected == 200
        assert "2 sources" in cl.latest.description

    def test_record_merge_returns_changelog(self):
        cl = create_changelog()
        result = record_merge(cl, ["x"], 10)
        assert result is cl


# ---------------------------------------------------------------------------
# render_changelog
# ---------------------------------------------------------------------------


class TestRenderChangelog:
    def test_render_empty(self):
        cl = create_changelog("empty")
        text = render_changelog(cl)
        assert "empty" in text
        assert "(no entries)" in text

    def test_render_with_entries(self):
        cl = create_changelog("render_test")
        record_build(cl, "src.jsonl", 10)
        record_filter(cl, "quality", 10, 8)
        text = render_changelog(cl)
        assert "render_test" in text
        assert "build" in text
        assert "filter" in text

    def test_render_max_entries(self):
        cl = create_changelog("many")
        for i in range(30):
            cl.add_entry("op", description=f"entry-{i}")
        text = render_changelog(cl, max_entries=5)
        # Should contain only the last 5 entries
        assert "entry-29" in text
        assert "entry-25" in text
        assert "entry-24" not in text

    def test_render_max_entries_larger_than_total(self):
        cl = create_changelog("few")
        cl.add_entry("build", description="only one")
        text = render_changelog(cl, max_entries=100)
        assert "only one" in text
