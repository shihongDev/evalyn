"""
Tests for storage multi-DB queries.

Run with: uv run pytest tests/test_multi_db.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.multi_db import (
    CrossDBQueryResult,
    DatabaseSource,
    MultiDBConfig,
    MultiDBManager,
    MultiDBReport,
    create_multi_db,
    find_common_records,
)


# ---------------------------------------------------------------------------
# DatabaseSource
# ---------------------------------------------------------------------------


class TestDatabaseSource:
    def test_defaults(self):
        s = DatabaseSource(name="prod")
        assert s.path == ""
        assert s.db_type == "sqlite"
        assert s.description == ""

    def test_as_dict(self):
        s = DatabaseSource(name="prod", path="/data/prod.db", description="Production")
        d = s.as_dict()
        assert d["name"] == "prod"
        assert d["path"] == "/data/prod.db"
        assert d["description"] == "Production"

    def test_from_dict(self):
        d = {"name": "test", "path": "/data/test.db", "db_type": "postgres"}
        s = DatabaseSource.from_dict(d)
        assert s.name == "test"
        assert s.db_type == "postgres"

    def test_from_dict_empty(self):
        s = DatabaseSource.from_dict({})
        assert s.name == ""
        assert s.db_type == "sqlite"

    def test_roundtrip(self):
        s = DatabaseSource(name="staging", path="/db/staging.db", db_type="sqlite", description="Staging env")
        restored = DatabaseSource.from_dict(s.as_dict())
        assert restored.name == s.name
        assert restored.path == s.path
        assert restored.db_type == s.db_type
        assert restored.description == s.description


# ---------------------------------------------------------------------------
# MultiDBConfig
# ---------------------------------------------------------------------------


class TestMultiDBConfig:
    def test_defaults(self):
        cfg = MultiDBConfig()
        assert cfg.sources == []
        assert cfg.default_source == ""

    def test_as_dict(self):
        cfg = MultiDBConfig(
            sources=[DatabaseSource(name="a"), DatabaseSource(name="b")],
            default_source="a",
        )
        d = cfg.as_dict()
        assert len(d["sources"]) == 2
        assert d["default_source"] == "a"

    def test_from_dict(self):
        d = {
            "sources": [{"name": "x", "path": "/x.db"}],
            "default_source": "x",
        }
        cfg = MultiDBConfig.from_dict(d)
        assert len(cfg.sources) == 1
        assert cfg.sources[0].name == "x"
        assert cfg.default_source == "x"

    def test_from_dict_empty(self):
        cfg = MultiDBConfig.from_dict({})
        assert cfg.sources == []
        assert cfg.default_source == ""

    def test_roundtrip(self):
        cfg = MultiDBConfig(
            sources=[DatabaseSource(name="prod", path="/prod.db")],
            default_source="prod",
        )
        restored = MultiDBConfig.from_dict(cfg.as_dict())
        assert len(restored.sources) == 1
        assert restored.sources[0].name == "prod"
        assert restored.default_source == "prod"


# ---------------------------------------------------------------------------
# CrossDBQueryResult
# ---------------------------------------------------------------------------


class TestCrossDBQueryResult:
    def test_defaults(self):
        r = CrossDBQueryResult(source="prod")
        assert r.records == []
        assert r.count == 0
        assert r.duration_ms == 0.0

    def test_as_dict(self):
        r = CrossDBQueryResult(
            source="test",
            records=[{"id": "1"}],
            count=1,
            duration_ms=5.5,
        )
        d = r.as_dict()
        assert d["source"] == "test"
        assert d["count"] == 1
        assert len(d["records"]) == 1


# ---------------------------------------------------------------------------
# MultiDBReport
# ---------------------------------------------------------------------------


class TestMultiDBReport:
    def test_defaults(self):
        rpt = MultiDBReport()
        assert rpt.results == []
        assert rpt.total_records == 0
        assert rpt.sources_queried == []

    def test_as_dict(self):
        rpt = MultiDBReport(
            results=[CrossDBQueryResult(source="prod", count=5)],
            total_records=5,
            sources_queried=["prod"],
        )
        d = rpt.as_dict()
        assert d["total_records"] == 5
        assert len(d["results"]) == 1

    def test_from_dict(self):
        d = {
            "results": [{"source": "a", "records": [{"id": "1"}], "count": 1, "duration_ms": 2.0}],
            "total_records": 1,
            "sources_queried": ["a"],
        }
        rpt = MultiDBReport.from_dict(d)
        assert rpt.total_records == 1
        assert rpt.results[0].source == "a"
        assert rpt.results[0].count == 1

    def test_from_dict_empty(self):
        rpt = MultiDBReport.from_dict({})
        assert rpt.results == []
        assert rpt.total_records == 0

    def test_roundtrip(self):
        rpt = MultiDBReport(
            results=[CrossDBQueryResult(source="x", records=[{"id": "r1"}], count=1, duration_ms=3.0)],
            total_records=1,
            sources_queried=["x"],
        )
        restored = MultiDBReport.from_dict(rpt.as_dict())
        assert restored.total_records == rpt.total_records
        assert len(restored.results) == 1
        assert restored.results[0].source == "x"

    def test_format_text(self):
        rpt = MultiDBReport(
            results=[
                CrossDBQueryResult(source="prod", count=10, duration_ms=5.0),
                CrossDBQueryResult(source="test", count=3, duration_ms=2.1),
            ],
            total_records=13,
            sources_queried=["prod", "test"],
        )
        text = rpt.format_text()
        assert "Multi-DB Report" in text
        assert "prod, test" in text
        assert "13" in text
        assert "[prod]" in text
        assert "[test]" in text

    def test_format_text_empty(self):
        rpt = MultiDBReport()
        text = rpt.format_text()
        assert "none" in text


# ---------------------------------------------------------------------------
# MultiDBManager
# ---------------------------------------------------------------------------


class TestMultiDBManager:
    def _make_manager(self):
        cfg = MultiDBConfig(sources=[
            DatabaseSource(name="prod", path="/prod.db"),
            DatabaseSource(name="test", path="/test.db"),
        ])
        return MultiDBManager(cfg)

    def test_list_sources(self):
        mgr = self._make_manager()
        sources = mgr.list_sources()
        assert len(sources) == 2
        assert sources[0].name == "prod"

    def test_add_source(self):
        mgr = self._make_manager()
        mgr.add_source(DatabaseSource(name="staging"))
        assert len(mgr.list_sources()) == 3

    def test_remove_source_exists(self):
        mgr = self._make_manager()
        assert mgr.remove_source("test") is True
        assert len(mgr.list_sources()) == 1

    def test_remove_source_not_found(self):
        mgr = self._make_manager()
        assert mgr.remove_source("nonexistent") is False
        assert len(mgr.list_sources()) == 2

    def test_query_all_filters(self):
        mgr = self._make_manager()
        records = {
            "prod": [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.3}],
            "test": [{"id": "3", "score": 0.8}, {"id": "4", "score": 0.1}],
        }
        report = mgr.query_all(lambda r: r.get("score", 0) > 0.5, records)
        assert report.total_records == 2
        assert len(report.sources_queried) == 2
        # prod has 1 match (score 0.9), test has 1 match (score 0.8)
        prod_result = report.results[0]
        test_result = report.results[1]
        assert prod_result.count == 1
        assert test_result.count == 1

    def test_query_all_no_matches(self):
        mgr = self._make_manager()
        records = {
            "prod": [{"id": "1", "score": 0.1}],
            "test": [{"id": "2", "score": 0.2}],
        }
        report = mgr.query_all(lambda r: r.get("score", 0) > 0.99, records)
        assert report.total_records == 0

    def test_query_all_missing_source(self):
        mgr = self._make_manager()
        # Only provide records for prod, not test
        records = {"prod": [{"id": "1", "val": 1}]}
        report = mgr.query_all(lambda r: True, records)
        assert report.total_records == 1
        assert report.results[1].count == 0  # test has no records

    def test_compare_sources(self):
        mgr = self._make_manager()
        records = {
            "prod": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "test": [{"id": "b"}, {"id": "c"}, {"id": "d"}],
        }
        cmp = mgr.compare_sources(records)
        assert cmp["counts"]["prod"] == 3
        assert cmp["counts"]["test"] == 3
        assert sorted(cmp["common_ids"]) == ["b", "c"]
        assert cmp["unique_per_source"]["prod"] == ["a"]
        assert cmp["unique_per_source"]["test"] == ["d"]

    def test_compare_sources_no_overlap(self):
        mgr = self._make_manager()
        records = {
            "prod": [{"id": "1"}],
            "test": [{"id": "2"}],
        }
        cmp = mgr.compare_sources(records)
        assert cmp["common_ids"] == []
        assert cmp["unique_per_source"]["prod"] == ["1"]
        assert cmp["unique_per_source"]["test"] == ["2"]

    def test_merge_results(self):
        mgr = self._make_manager()
        results = [
            CrossDBQueryResult(source="prod", records=[{"id": "1"}, {"id": "2"}], count=2),
            CrossDBQueryResult(source="test", records=[{"id": "3"}], count=1),
        ]
        merged = mgr.merge_results(results)
        assert len(merged) == 3
        assert merged[0]["_source"] == "prod"
        assert merged[2]["_source"] == "test"

    def test_merge_results_empty(self):
        mgr = self._make_manager()
        merged = mgr.merge_results([])
        assert merged == []

    def test_merge_does_not_mutate_original(self):
        mgr = self._make_manager()
        original = {"id": "1", "val": "x"}
        results = [CrossDBQueryResult(source="prod", records=[original], count=1)]
        merged = mgr.merge_results(results)
        assert "_source" in merged[0]
        assert "_source" not in original


# ---------------------------------------------------------------------------
# Factory & Utility Functions
# ---------------------------------------------------------------------------


class TestFactoryAndFunctions:
    def test_create_multi_db(self):
        mgr = create_multi_db([
            {"name": "prod", "path": "/prod.db"},
            {"name": "test", "path": "/test.db"},
        ])
        assert len(mgr.list_sources()) == 2

    def test_create_multi_db_empty(self):
        mgr = create_multi_db([])
        assert len(mgr.list_sources()) == 0

    def test_find_common_records(self):
        a = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        b = [{"id": "2"}, {"id": "3"}, {"id": "4"}]
        common = find_common_records(a, b)
        assert common == ["2", "3"]

    def test_find_common_records_no_overlap(self):
        a = [{"id": "1"}]
        b = [{"id": "2"}]
        assert find_common_records(a, b) == []

    def test_find_common_records_custom_key(self):
        a = [{"name": "alice"}, {"name": "bob"}]
        b = [{"name": "bob"}, {"name": "carol"}]
        common = find_common_records(a, b, key="name")
        assert common == ["bob"]

    def test_find_common_records_empty(self):
        assert find_common_records([], []) == []

    def test_find_common_records_missing_key(self):
        a = [{"id": "1"}, {"other": "x"}]
        b = [{"id": "1"}, {"id": "2"}]
        common = find_common_records(a, b)
        assert common == ["1"]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_source_manager(self):
        cfg = MultiDBConfig(sources=[DatabaseSource(name="only")])
        mgr = MultiDBManager(cfg)
        records = {"only": [{"id": "1"}, {"id": "2"}]}
        report = mgr.query_all(lambda r: True, records)
        assert report.total_records == 2
        assert report.sources_queried == ["only"]

    def test_empty_sources_manager(self):
        cfg = MultiDBConfig()
        mgr = MultiDBManager(cfg)
        report = mgr.query_all(lambda r: True, {})
        assert report.total_records == 0
        assert report.sources_queried == []

    def test_compare_empty_sources(self):
        cfg = MultiDBConfig()
        mgr = MultiDBManager(cfg)
        cmp = mgr.compare_sources({})
        assert cmp["counts"] == {}
        assert cmp["common_ids"] == []
        assert cmp["unique_per_source"] == {}
