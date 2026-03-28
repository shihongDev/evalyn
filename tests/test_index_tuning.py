"""
Tests for evalyn_sdk.storage.index_tuning.

Run with: uv run pytest tests/test_index_tuning.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.index_tuning import (
    ExistingIndex,
    IndexSuggestion,
    IndexTuningReport,
    build_tuning_report,
    detect_redundant_indexes,
    generate_create_index_sql,
    parse_query_columns,
    suggest_indexes,
)


# ---------------------------------------------------------------------------
# parse_query_columns
# ---------------------------------------------------------------------------


class TestParseQueryColumns:
    def test_simple_where(self):
        cols = parse_query_columns("status = 'active'")
        assert "status" in cols

    def test_multiple_columns(self):
        cols = parse_query_columns("status = 'active' AND age > 30")
        assert "status" in cols
        assert "age" in cols

    def test_like_operator(self):
        cols = parse_query_columns("name LIKE '%test%'")
        assert "name" in cols

    def test_in_operator(self):
        cols = parse_query_columns("category IN ('a', 'b', 'c')")
        assert "category" in cols

    def test_is_null(self):
        cols = parse_query_columns("deleted_at IS NULL")
        assert "deleted_at" in cols

    def test_comparison_operators(self):
        cols = parse_query_columns("price >= 100 AND qty <= 50")
        assert "price" in cols
        assert "qty" in cols

    def test_empty_query(self):
        assert parse_query_columns("") == []

    def test_whitespace_only(self):
        assert parse_query_columns("   ") == []

    def test_no_duplicates(self):
        cols = parse_query_columns("status = 'a' AND status = 'b'")
        assert cols.count("status") == 1

    def test_not_equal(self):
        cols = parse_query_columns("status != 'deleted'")
        assert "status" in cols


# ---------------------------------------------------------------------------
# IndexSuggestion
# ---------------------------------------------------------------------------


class TestIndexSuggestion:
    def test_as_dict(self):
        s = IndexSuggestion(
            table="runs",
            columns=["status"],
            index_name="idx_runs_status",
            reason="frequently filtered",
            estimated_speedup="moderate",
        )
        d = s.as_dict()
        assert d["table"] == "runs"
        assert d["columns"] == ["status"]
        assert d["index_name"] == "idx_runs_status"
        assert d["reason"] == "frequently filtered"
        assert d["estimated_speedup"] == "moderate"

    def test_defaults(self):
        s = IndexSuggestion(table="runs", columns=["status"])
        assert s.index_name == ""
        assert s.reason == ""
        assert s.estimated_speedup == ""


# ---------------------------------------------------------------------------
# ExistingIndex
# ---------------------------------------------------------------------------


class TestExistingIndex:
    def test_as_dict(self):
        idx = ExistingIndex(name="idx_runs_id", table="runs", columns=["id"], unique=True)
        d = idx.as_dict()
        assert d["name"] == "idx_runs_id"
        assert d["table"] == "runs"
        assert d["columns"] == ["id"]
        assert d["unique"] is True

    def test_from_dict(self):
        d = {"name": "idx_runs_id", "table": "runs", "columns": ["id"], "unique": True}
        idx = ExistingIndex.from_dict(d)
        assert idx.name == "idx_runs_id"
        assert idx.table == "runs"
        assert idx.columns == ["id"]
        assert idx.unique is True

    def test_from_dict_defaults(self):
        idx = ExistingIndex.from_dict({})
        assert idx.name == ""
        assert idx.table == ""
        assert idx.columns == []
        assert idx.unique is False

    def test_roundtrip(self):
        original = ExistingIndex(name="idx_x", table="t", columns=["a", "b"], unique=False)
        restored = ExistingIndex.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.table == original.table
        assert restored.columns == original.columns
        assert restored.unique == original.unique


# ---------------------------------------------------------------------------
# IndexTuningReport
# ---------------------------------------------------------------------------


class TestIndexTuningReport:
    def test_as_dict(self):
        report = IndexTuningReport(
            suggestions=[IndexSuggestion(table="t", columns=["a"])],
            existing=[ExistingIndex(name="idx_t_b", table="t", columns=["b"])],
            redundant=["idx_old"],
            total_suggested=1,
        )
        d = report.as_dict()
        assert len(d["suggestions"]) == 1
        assert len(d["existing"]) == 1
        assert d["redundant"] == ["idx_old"]
        assert d["total_suggested"] == 1

    def test_from_dict(self):
        d = {
            "suggestions": [
                {"table": "t", "columns": ["a"], "index_name": "idx_t_a", "reason": "test", "estimated_speedup": "low"}
            ],
            "existing": [
                {"name": "idx_t_b", "table": "t", "columns": ["b"], "unique": False}
            ],
            "redundant": ["idx_old"],
            "total_suggested": 1,
        }
        report = IndexTuningReport.from_dict(d)
        assert len(report.suggestions) == 1
        assert report.suggestions[0].table == "t"
        assert len(report.existing) == 1
        assert report.redundant == ["idx_old"]
        assert report.total_suggested == 1

    def test_from_dict_defaults(self):
        report = IndexTuningReport.from_dict({})
        assert report.suggestions == []
        assert report.existing == []
        assert report.redundant == []
        assert report.total_suggested == 0

    def test_roundtrip(self):
        original = IndexTuningReport(
            suggestions=[IndexSuggestion(table="t", columns=["a"], index_name="idx_t_a")],
            existing=[ExistingIndex(name="idx_t_b", table="t", columns=["b"])],
            redundant=["idx_r"],
            total_suggested=1,
        )
        restored = IndexTuningReport.from_dict(original.as_dict())
        assert restored.total_suggested == original.total_suggested
        assert len(restored.suggestions) == len(original.suggestions)
        assert len(restored.existing) == len(original.existing)
        assert restored.redundant == original.redundant

    def test_format_text_with_suggestions(self):
        report = IndexTuningReport(
            suggestions=[
                IndexSuggestion(
                    table="runs",
                    columns=["status"],
                    index_name="idx_runs_status",
                    reason="frequently filtered",
                    estimated_speedup="moderate",
                )
            ],
            existing=[ExistingIndex(name="idx_runs_id", table="runs", columns=["id"])],
            redundant=["idx_old"],
            total_suggested=1,
        )
        text = report.format_text()
        assert "Index Tuning Report" in text
        assert "Suggested indexes: 1" in text
        assert "idx_runs_status" in text
        assert "frequently filtered" in text
        assert "moderate" in text
        assert "idx_old" in text

    def test_format_text_empty(self):
        report = IndexTuningReport()
        text = report.format_text()
        assert "Suggested indexes: 0" in text
        assert "Redundant indexes: 0" in text


# ---------------------------------------------------------------------------
# suggest_indexes
# ---------------------------------------------------------------------------


class TestSuggestIndexes:
    def test_where_clause(self):
        patterns = ["status = 'active'"]
        suggestions = suggest_indexes("runs", patterns)
        assert len(suggestions) >= 1
        col_names = [s.columns[0] for s in suggestions]
        assert "status" in col_names

    def test_multiple_patterns_frequency(self):
        patterns = [
            "status = 'active'",
            "status = 'failed' AND created_at > '2026-01-01'",
        ]
        suggestions = suggest_indexes("runs", patterns)
        # status appears twice, should be first (highest frequency)
        assert suggestions[0].columns == ["status"]

    def test_skips_existing(self):
        patterns = ["status = 'active'"]
        existing = [ExistingIndex(name="idx_runs_status", table="runs", columns=["status"])]
        suggestions = suggest_indexes("runs", patterns, existing)
        col_names = [s.columns[0] for s in suggestions]
        assert "status" not in col_names

    def test_skips_existing_different_table(self):
        """Existing index on different table should not affect suggestions."""
        patterns = ["status = 'active'"]
        existing = [ExistingIndex(name="idx_other_status", table="other", columns=["status"])]
        suggestions = suggest_indexes("runs", patterns, existing)
        col_names = [s.columns[0] for s in suggestions]
        assert "status" in col_names

    def test_empty_patterns(self):
        suggestions = suggest_indexes("runs", [])
        assert suggestions == []

    def test_no_existing_indexes(self):
        patterns = ["name = 'test'"]
        suggestions = suggest_indexes("runs", patterns)
        assert len(suggestions) >= 1

    def test_index_name_generated(self):
        patterns = ["status = 'active'"]
        suggestions = suggest_indexes("runs", patterns)
        assert suggestions[0].index_name == "idx_runs_status"

    def test_reason_included(self):
        patterns = ["status = 'active'"]
        suggestions = suggest_indexes("runs", patterns)
        assert "status" in suggestions[0].reason
        assert "1" in suggestions[0].reason


# ---------------------------------------------------------------------------
# detect_redundant_indexes
# ---------------------------------------------------------------------------


class TestDetectRedundantIndexes:
    def test_prefix_subset(self):
        indexes = [
            ExistingIndex(name="idx_a", table="t", columns=["a"]),
            ExistingIndex(name="idx_a_b", table="t", columns=["a", "b"]),
        ]
        redundant = detect_redundant_indexes(indexes)
        assert "idx_a" in redundant

    def test_no_redundant(self):
        indexes = [
            ExistingIndex(name="idx_a", table="t", columns=["a"]),
            ExistingIndex(name="idx_b", table="t", columns=["b"]),
        ]
        redundant = detect_redundant_indexes(indexes)
        assert redundant == []

    def test_different_tables(self):
        """Indexes on different tables are never redundant of each other."""
        indexes = [
            ExistingIndex(name="idx_t1_a", table="t1", columns=["a"]),
            ExistingIndex(name="idx_t2_a_b", table="t2", columns=["a", "b"]),
        ]
        redundant = detect_redundant_indexes(indexes)
        assert redundant == []

    def test_empty_list(self):
        assert detect_redundant_indexes([]) == []

    def test_single_index(self):
        indexes = [ExistingIndex(name="idx_a", table="t", columns=["a"])]
        assert detect_redundant_indexes(indexes) == []

    def test_identical_not_redundant(self):
        """Indexes with same columns (same length) are not prefix-subsets."""
        indexes = [
            ExistingIndex(name="idx_a1", table="t", columns=["a"]),
            ExistingIndex(name="idx_a2", table="t", columns=["a"]),
        ]
        redundant = detect_redundant_indexes(indexes)
        assert redundant == []

    def test_case_insensitive(self):
        indexes = [
            ExistingIndex(name="idx_a", table="t", columns=["A"]),
            ExistingIndex(name="idx_a_b", table="t", columns=["a", "b"]),
        ]
        redundant = detect_redundant_indexes(indexes)
        assert "idx_a" in redundant


# ---------------------------------------------------------------------------
# generate_create_index_sql
# ---------------------------------------------------------------------------


class TestGenerateCreateIndexSql:
    def test_basic(self):
        s = IndexSuggestion(table="runs", columns=["status"], index_name="idx_runs_status")
        sql = generate_create_index_sql(s)
        assert sql == "CREATE INDEX idx_runs_status ON runs (status);"

    def test_composite(self):
        s = IndexSuggestion(table="runs", columns=["status", "created_at"], index_name="idx_runs_status_created")
        sql = generate_create_index_sql(s)
        assert sql == "CREATE INDEX idx_runs_status_created ON runs (status, created_at);"

    def test_auto_name(self):
        s = IndexSuggestion(table="runs", columns=["status"])
        sql = generate_create_index_sql(s)
        assert "idx_runs_status" in sql
        assert sql.startswith("CREATE INDEX")
        assert sql.endswith(";")


# ---------------------------------------------------------------------------
# build_tuning_report
# ---------------------------------------------------------------------------


class TestBuildTuningReport:
    def test_basic(self):
        suggestions = [IndexSuggestion(table="runs", columns=["status"])]
        existing = [
            ExistingIndex(name="idx_a", table="t", columns=["a"]),
            ExistingIndex(name="idx_a_b", table="t", columns=["a", "b"]),
        ]
        report = build_tuning_report(suggestions, existing)
        assert report.total_suggested == 1
        assert len(report.suggestions) == 1
        assert len(report.existing) == 2
        assert "idx_a" in report.redundant

    def test_no_suggestions(self):
        existing = [ExistingIndex(name="idx_a", table="t", columns=["a"])]
        report = build_tuning_report([], existing)
        assert report.total_suggested == 0
        assert report.suggestions == []

    def test_no_existing(self):
        suggestions = [IndexSuggestion(table="t", columns=["a"])]
        report = build_tuning_report(suggestions, [])
        assert report.total_suggested == 1
        assert report.redundant == []

    def test_empty(self):
        report = build_tuning_report([], [])
        assert report.total_suggested == 0
        assert report.suggestions == []
        assert report.existing == []
        assert report.redundant == []
