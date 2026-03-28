"""
Tests for evalyn_sdk.storage.query_logging.

Run with: uv run pytest tests/test_query_logging.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.query_logging import (
    QueryLog,
    QueryLogger,
    QueryStats,
    analyze_query_patterns,
    detect_operation,
    detect_table,
    suggest_optimizations,
)


# ---------------------------------------------------------------------------
# QueryLog
# ---------------------------------------------------------------------------


class TestQueryLog:
    def test_as_dict(self):
        log = QueryLog(
            query="SELECT * FROM users",
            duration_ms=12.5,
            timestamp="2026-03-28T00:00:00+00:00",
            table="users",
            operation="SELECT",
        )
        d = log.as_dict()
        assert d["query"] == "SELECT * FROM users"
        assert d["duration_ms"] == 12.5
        assert d["table"] == "users"
        assert d["operation"] == "SELECT"

    def test_defaults(self):
        log = QueryLog(query="SELECT 1")
        assert log.duration_ms == 0.0
        assert log.timestamp == ""
        assert log.table == ""
        assert log.operation == ""


# ---------------------------------------------------------------------------
# QueryStats
# ---------------------------------------------------------------------------


class TestQueryStats:
    def test_as_dict_from_dict_roundtrip(self):
        stats = QueryStats(
            total_queries=10,
            avg_duration_ms=25.5,
            slowest_query="SELECT * FROM big_table",
            slowest_duration_ms=200.0,
            by_operation={"SELECT": 8, "INSERT": 2},
            by_table={"big_table": 6, "small_table": 4},
        )
        d = stats.as_dict()
        restored = QueryStats.from_dict(d)
        assert restored.total_queries == 10
        assert restored.avg_duration_ms == 25.5
        assert restored.slowest_query == "SELECT * FROM big_table"
        assert restored.slowest_duration_ms == 200.0
        assert restored.by_operation == {"SELECT": 8, "INSERT": 2}
        assert restored.by_table == {"big_table": 6, "small_table": 4}

    def test_from_dict_defaults(self):
        stats = QueryStats.from_dict({})
        assert stats.total_queries == 0
        assert stats.avg_duration_ms == 0.0
        assert stats.slowest_query == ""
        assert stats.by_operation == {}

    def test_format_text_basic(self):
        stats = QueryStats(
            total_queries=5,
            avg_duration_ms=10.0,
            slowest_query="SELECT * FROM t",
            slowest_duration_ms=50.0,
            by_operation={"SELECT": 5},
            by_table={"t": 5},
        )
        text = stats.format_text()
        assert "Query Stats" in text
        assert "Total queries: 5" in text
        assert "Avg duration: 10.00 ms" in text
        assert "Slowest query: SELECT * FROM t" in text
        assert "SELECT: 5" in text
        assert "t: 5" in text

    def test_format_text_empty(self):
        stats = QueryStats()
        text = stats.format_text()
        assert "Total queries: 0" in text


# ---------------------------------------------------------------------------
# detect_operation
# ---------------------------------------------------------------------------


class TestDetectOperation:
    def test_select(self):
        assert detect_operation("SELECT * FROM users") == "SELECT"

    def test_insert(self):
        assert detect_operation("INSERT INTO users VALUES (1)") == "INSERT"

    def test_update(self):
        assert detect_operation("UPDATE users SET name='x'") == "UPDATE"

    def test_delete(self):
        assert detect_operation("DELETE FROM users WHERE id=1") == "DELETE"

    def test_lowercase(self):
        assert detect_operation("select * from users") == "SELECT"

    def test_leading_whitespace(self):
        assert detect_operation("  SELECT 1") == "SELECT"

    def test_unknown(self):
        assert detect_operation("CREATE TABLE t (id INT)") == "OTHER"

    def test_empty(self):
        assert detect_operation("") == "OTHER"


# ---------------------------------------------------------------------------
# detect_table
# ---------------------------------------------------------------------------


class TestDetectTable:
    def test_select_from(self):
        assert detect_table("SELECT * FROM users WHERE id=1") == "users"

    def test_insert_into(self):
        assert detect_table("INSERT INTO orders VALUES (1)") == "orders"

    def test_update(self):
        assert detect_table("UPDATE products SET price=10") == "products"

    def test_delete_from(self):
        assert detect_table("DELETE FROM logs WHERE ts < '2026'") == "logs"

    def test_no_table(self):
        assert detect_table("SELECT 1") == ""

    def test_empty(self):
        assert detect_table("") == ""


# ---------------------------------------------------------------------------
# QueryLogger
# ---------------------------------------------------------------------------


class TestQueryLogger:
    def test_log_and_get_logs(self):
        logger = QueryLogger()
        logger.log("SELECT * FROM users", duration_ms=10.0)
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0].query == "SELECT * FROM users"
        assert logs[0].operation == "SELECT"
        assert logs[0].table == "users"
        assert logs[0].duration_ms == 10.0
        assert logs[0].timestamp != ""

    def test_count_property(self):
        logger = QueryLogger()
        assert logger.count == 0
        logger.log("SELECT 1")
        assert logger.count == 1
        logger.log("SELECT 2")
        assert logger.count == 2

    def test_clear(self):
        logger = QueryLogger()
        logger.log("SELECT 1")
        logger.log("SELECT 2")
        assert logger.count == 2
        logger.clear()
        assert logger.count == 0
        assert logger.get_logs() == []

    def test_max_logs_limit(self):
        logger = QueryLogger(max_logs=3)
        for i in range(5):
            logger.log(f"SELECT {i}")
        assert logger.count == 3
        # Should keep the last 3
        queries = [log.query for log in logger.get_logs()]
        assert queries == ["SELECT 2", "SELECT 3", "SELECT 4"]

    def test_get_slow_queries(self):
        logger = QueryLogger()
        logger.log("SELECT 1", duration_ms=50.0)
        logger.log("SELECT 2", duration_ms=150.0)
        logger.log("SELECT 3", duration_ms=200.0)
        slow = logger.get_slow_queries(threshold_ms=100.0)
        assert len(slow) == 2
        assert slow[0].duration_ms == 150.0
        assert slow[1].duration_ms == 200.0

    def test_get_slow_queries_none_slow(self):
        logger = QueryLogger()
        logger.log("SELECT 1", duration_ms=10.0)
        slow = logger.get_slow_queries(threshold_ms=100.0)
        assert slow == []

    def test_get_stats(self):
        logger = QueryLogger()
        logger.log("SELECT * FROM users", duration_ms=10.0)
        logger.log("INSERT INTO users VALUES (1)", duration_ms=30.0)
        logger.log("SELECT * FROM orders", duration_ms=20.0)
        stats = logger.get_stats()
        assert stats.total_queries == 3
        assert stats.avg_duration_ms == 20.0
        assert stats.slowest_query == "INSERT INTO users VALUES (1)"
        assert stats.slowest_duration_ms == 30.0
        assert stats.by_operation["SELECT"] == 2
        assert stats.by_operation["INSERT"] == 1
        assert stats.by_table["users"] == 2
        assert stats.by_table["orders"] == 1

    def test_get_stats_empty(self):
        logger = QueryLogger()
        stats = logger.get_stats()
        assert stats.total_queries == 0
        assert stats.avg_duration_ms == 0.0
        assert stats.slowest_query == ""

    def test_get_logs_returns_copy(self):
        logger = QueryLogger()
        logger.log("SELECT 1")
        logs = logger.get_logs()
        logs.clear()
        assert logger.count == 1


# ---------------------------------------------------------------------------
# analyze_query_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeQueryPatterns:
    def test_repeated_queries(self):
        logs = [
            QueryLog(query="SELECT * FROM users", table="users", operation="SELECT"),
            QueryLog(query="SELECT * FROM users", table="users", operation="SELECT"),
            QueryLog(query="INSERT INTO orders VALUES (1)", table="orders", operation="INSERT"),
        ]
        result = analyze_query_patterns(logs)
        assert result["repeated_queries"]["SELECT * FROM users"] == 2
        assert "INSERT INTO orders VALUES (1)" not in result["repeated_queries"]

    def test_most_accessed_tables(self):
        logs = [
            QueryLog(query="q1", table="users", operation="SELECT"),
            QueryLog(query="q2", table="users", operation="SELECT"),
            QueryLog(query="q3", table="orders", operation="INSERT"),
        ]
        result = analyze_query_patterns(logs)
        assert result["most_accessed_tables"]["users"] == 2
        assert result["most_accessed_tables"]["orders"] == 1

    def test_operation_distribution(self):
        logs = [
            QueryLog(query="q1", operation="SELECT"),
            QueryLog(query="q2", operation="SELECT"),
            QueryLog(query="q3", operation="INSERT"),
        ]
        result = analyze_query_patterns(logs)
        assert result["operation_distribution"]["SELECT"] == 2
        assert result["operation_distribution"]["INSERT"] == 1

    def test_empty_logs(self):
        result = analyze_query_patterns([])
        assert result["repeated_queries"] == {}
        assert result["most_accessed_tables"] == {}
        assert result["operation_distribution"] == {}


# ---------------------------------------------------------------------------
# suggest_optimizations
# ---------------------------------------------------------------------------


class TestSuggestOptimizations:
    def test_slow_query_suggestion(self):
        stats = QueryStats(
            total_queries=10,
            slowest_duration_ms=1500.0,
            slowest_query="SELECT * FROM big",
        )
        suggestions = suggest_optimizations(stats)
        assert any("1s" in s for s in suggestions)

    def test_high_avg_duration(self):
        stats = QueryStats(total_queries=10, avg_duration_ms=600.0)
        suggestions = suggest_optimizations(stats)
        assert any("average" in s.lower() for s in suggestions)

    def test_high_volume(self):
        stats = QueryStats(total_queries=600)
        suggestions = suggest_optimizations(stats)
        assert any("volume" in s.lower() or "batch" in s.lower() for s in suggestions)

    def test_read_heavy(self):
        stats = QueryStats(
            total_queries=100,
            by_operation={"SELECT": 95, "INSERT": 5},
        )
        suggestions = suggest_optimizations(stats)
        assert any("read" in s.lower() for s in suggestions)

    def test_single_table(self):
        stats = QueryStats(
            total_queries=200,
            by_table={"users": 200},
        )
        suggestions = suggest_optimizations(stats)
        assert any("one table" in s.lower() for s in suggestions)

    def test_no_suggestions_for_healthy_stats(self):
        stats = QueryStats(
            total_queries=10,
            avg_duration_ms=5.0,
            slowest_duration_ms=20.0,
            by_operation={"SELECT": 5, "INSERT": 5},
            by_table={"a": 5, "b": 5},
        )
        suggestions = suggest_optimizations(stats)
        assert suggestions == []
