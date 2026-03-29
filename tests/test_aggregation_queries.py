"""Tests for the aggregation queries module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.aggregation_queries import (
    AggregationQuery,
    AggregationResult,
    ProjectStats,
    execute_query,
    aggregate_by_project,
    aggregate_by_date,
    aggregate_by_model,
    filter_records,
    format_aggregation_result,
    format_project_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"project": "alpha", "model": "gpt-4", "tokens": 100, "cost": 0.01, "latency": 0.5, "timestamp": "2026-01-01T10:00:00Z"},
    {"project": "alpha", "model": "gpt-4", "tokens": 200, "cost": 0.02, "latency": 0.6, "timestamp": "2026-01-01T11:00:00Z"},
    {"project": "alpha", "model": "claude-3", "tokens": 150, "cost": 0.015, "latency": 0.4, "timestamp": "2026-01-02T10:00:00Z"},
    {"project": "beta", "model": "gpt-4", "tokens": 300, "cost": 0.03, "latency": 0.8, "timestamp": "2026-01-02T12:00:00Z"},
    {"project": "beta", "model": "claude-3", "tokens": 250, "cost": 0.025, "latency": 0.3, "timestamp": "2026-01-03T09:00:00Z"},
]


# ---------------------------------------------------------------------------
# AggregationQuery
# ---------------------------------------------------------------------------


class TestAggregationQuery:
    def test_defaults(self):
        q = AggregationQuery(group_by=["model"], metric="cost", operation="sum")
        assert q.date_field == ""
        assert q.filters == {}

    def test_as_dict(self):
        q = AggregationQuery(group_by=["a", "b"], metric="x", operation="mean", filters={"k": "v"})
        d = q.as_dict()
        assert d["group_by"] == ["a", "b"]
        assert d["filters"] == {"k": "v"}

    def test_from_dict(self):
        d = {"group_by": ["model"], "metric": "tokens", "operation": "sum"}
        q = AggregationQuery.from_dict(d)
        assert q.group_by == ["model"]
        assert q.operation == "sum"

    def test_roundtrip(self):
        q = AggregationQuery(["a"], "m", "max", {"f": 1}, "ts", "2026-01-01", "2026-12-31")
        assert AggregationQuery.from_dict(q.as_dict()).as_dict() == q.as_dict()

    def test_from_dict_defaults(self):
        q = AggregationQuery.from_dict({})
        assert q.group_by == []
        assert q.operation == "count"


# ---------------------------------------------------------------------------
# AggregationResult
# ---------------------------------------------------------------------------


class TestAggregationResult:
    def test_as_dict(self):
        q = AggregationQuery(["model"], "cost", "sum")
        r = AggregationResult(groups=[{"model": "gpt-4", "value": 0.05}], total_records=3, query=q)
        d = r.as_dict()
        assert d["total_records"] == 3
        assert len(d["groups"]) == 1

    def test_from_dict(self):
        d = {
            "groups": [{"model": "x", "value": 1.0}],
            "total_records": 5,
            "query": {"group_by": ["model"], "metric": "cost", "operation": "sum"},
        }
        r = AggregationResult.from_dict(d)
        assert r.total_records == 5
        assert r.query.operation == "sum"

    def test_roundtrip(self):
        q = AggregationQuery(["a"], "m", "count")
        r = AggregationResult([{"a": "b", "value": 3.0}], 10, q)
        assert AggregationResult.from_dict(r.as_dict()).as_dict() == r.as_dict()


# ---------------------------------------------------------------------------
# ProjectStats
# ---------------------------------------------------------------------------


class TestProjectStats:
    def test_as_dict(self):
        s = ProjectStats("proj", 10, 5000, 1.5, {"gpt-4": 8, "claude": 2}, ("2026-01-01", "2026-01-31"))
        d = s.as_dict()
        assert d["project_name"] == "proj"
        assert d["date_range"] == ["2026-01-01", "2026-01-31"]

    def test_from_dict(self):
        d = {
            "project_name": "x",
            "total_traces": 5,
            "total_tokens": 1000,
            "total_cost": 0.5,
            "model_breakdown": {"m": 5},
            "date_range": ["a", "b"],
        }
        s = ProjectStats.from_dict(d)
        assert s.project_name == "x"
        assert s.date_range == ("a", "b")

    def test_roundtrip(self):
        s = ProjectStats("p", 1, 2, 3.0, {"m": 1}, ("d1", "d2"))
        rebuilt = ProjectStats.from_dict(s.as_dict())
        assert rebuilt.project_name == s.project_name
        assert rebuilt.date_range == s.date_range

    def test_from_dict_empty_date_range(self):
        s = ProjectStats.from_dict({"date_range": []})
        assert s.date_range == ("", "")


# ---------------------------------------------------------------------------
# filter_records
# ---------------------------------------------------------------------------


class TestFilterRecords:
    def test_no_filters(self):
        result = filter_records(SAMPLE_RECORDS, {})
        assert len(result) == 5

    def test_single_filter(self):
        result = filter_records(SAMPLE_RECORDS, {"project": "alpha"})
        assert len(result) == 3

    def test_multiple_filters(self):
        result = filter_records(SAMPLE_RECORDS, {"project": "alpha", "model": "gpt-4"})
        assert len(result) == 2

    def test_no_match(self):
        result = filter_records(SAMPLE_RECORDS, {"project": "gamma"})
        assert len(result) == 0

    def test_date_from(self):
        result = filter_records(SAMPLE_RECORDS, {}, date_field="timestamp", date_from="2026-01-02T00:00:00Z")
        assert len(result) == 3

    def test_date_to(self):
        result = filter_records(SAMPLE_RECORDS, {}, date_field="timestamp", date_to="2026-01-01T23:59:59Z")
        assert len(result) == 2

    def test_date_range(self):
        result = filter_records(
            SAMPLE_RECORDS, {},
            date_field="timestamp",
            date_from="2026-01-02T00:00:00Z",
            date_to="2026-01-02T23:59:59Z",
        )
        assert len(result) == 2

    def test_missing_date_field_excluded(self):
        records = [{"id": 1}, {"id": 2, "timestamp": "2026-01-01"}]
        result = filter_records(records, {}, date_field="timestamp", date_from="2025-01-01")
        assert len(result) == 1

    def test_missing_filter_field_excluded(self):
        records = [{"a": 1}, {"b": 2}]
        result = filter_records(records, {"a": 1})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


class TestExecuteQuery:
    def test_sum_by_project(self):
        q = AggregationQuery(group_by=["project"], metric="cost", operation="sum")
        r = execute_query(SAMPLE_RECORDS, q)
        by_proj = {g["project"]: g["value"] for g in r.groups}
        assert abs(by_proj["alpha"] - 0.045) < 0.0001
        assert abs(by_proj["beta"] - 0.055) < 0.0001

    def test_mean_by_model(self):
        q = AggregationQuery(group_by=["model"], metric="latency", operation="mean")
        r = execute_query(SAMPLE_RECORDS, q)
        by_model = {g["model"]: g["value"] for g in r.groups}
        assert abs(by_model["gpt-4"] - (0.5 + 0.6 + 0.8) / 3) < 0.001

    def test_count_by_project(self):
        q = AggregationQuery(group_by=["project"], metric="", operation="count")
        r = execute_query(SAMPLE_RECORDS, q)
        by_proj = {g["project"]: g["value"] for g in r.groups}
        assert by_proj["alpha"] == 3.0
        assert by_proj["beta"] == 2.0

    def test_min_by_model(self):
        q = AggregationQuery(group_by=["model"], metric="tokens", operation="min")
        r = execute_query(SAMPLE_RECORDS, q)
        by_model = {g["model"]: g["value"] for g in r.groups}
        assert by_model["gpt-4"] == 100.0

    def test_max_by_model(self):
        q = AggregationQuery(group_by=["model"], metric="tokens", operation="max")
        r = execute_query(SAMPLE_RECORDS, q)
        by_model = {g["model"]: g["value"] for g in r.groups}
        assert by_model["gpt-4"] == 300.0

    def test_with_filters(self):
        q = AggregationQuery(group_by=["model"], metric="cost", operation="sum", filters={"project": "alpha"})
        r = execute_query(SAMPLE_RECORDS, q)
        assert r.total_records == 3

    def test_with_date_range(self):
        q = AggregationQuery(
            group_by=["project"], metric="tokens", operation="sum",
            date_field="timestamp", date_from="2026-01-02T00:00:00Z", date_to="2026-01-02T23:59:59Z",
        )
        r = execute_query(SAMPLE_RECORDS, q)
        assert r.total_records == 2

    def test_missing_metric_skipped(self):
        records = [{"group": "a", "val": 10}, {"group": "a"}, {"group": "b", "val": 20}]
        q = AggregationQuery(group_by=["group"], metric="val", operation="sum")
        r = execute_query(records, q)
        by_g = {g["group"]: g["value"] for g in r.groups}
        assert by_g["a"] == 10.0
        assert by_g["b"] == 20.0

    def test_empty_records(self):
        q = AggregationQuery(group_by=["x"], metric="y", operation="sum")
        r = execute_query([], q)
        assert r.groups == []
        assert r.total_records == 0

    def test_multi_group_by(self):
        q = AggregationQuery(group_by=["project", "model"], metric="cost", operation="sum")
        r = execute_query(SAMPLE_RECORDS, q)
        assert len(r.groups) == 4  # alpha/gpt-4, alpha/claude-3, beta/gpt-4, beta/claude-3

    def test_unknown_operation_returns_zero(self):
        q = AggregationQuery(group_by=["project"], metric="cost", operation="unknown_op")
        r = execute_query(SAMPLE_RECORDS, q)
        for g in r.groups:
            assert g["value"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_by_project
# ---------------------------------------------------------------------------


class TestAggregateByProject:
    def test_basic(self):
        stats = aggregate_by_project(SAMPLE_RECORDS)
        assert len(stats) == 2
        names = {s.project_name for s in stats}
        assert names == {"alpha", "beta"}

    def test_token_totals(self):
        stats = aggregate_by_project(SAMPLE_RECORDS)
        alpha = [s for s in stats if s.project_name == "alpha"][0]
        assert alpha.total_tokens == 450  # 100+200+150

    def test_cost_totals(self):
        stats = aggregate_by_project(SAMPLE_RECORDS)
        beta = [s for s in stats if s.project_name == "beta"][0]
        assert abs(beta.total_cost - 0.055) < 0.0001

    def test_model_breakdown(self):
        stats = aggregate_by_project(SAMPLE_RECORDS)
        alpha = [s for s in stats if s.project_name == "alpha"][0]
        assert alpha.model_breakdown["gpt-4"] == 2
        assert alpha.model_breakdown["claude-3"] == 1

    def test_date_range(self):
        stats = aggregate_by_project(SAMPLE_RECORDS)
        alpha = [s for s in stats if s.project_name == "alpha"][0]
        assert alpha.date_range[0] == "2026-01-01T10:00:00Z"
        assert alpha.date_range[1] == "2026-01-02T10:00:00Z"

    def test_custom_project_field(self):
        records = [{"team": "x", "tokens": 10, "cost": 0.1}, {"team": "y", "tokens": 20, "cost": 0.2}]
        stats = aggregate_by_project(records, project_field="team")
        assert len(stats) == 2

    def test_missing_project_field(self):
        records = [{"tokens": 10, "cost": 0.1}]
        stats = aggregate_by_project(records)
        assert stats[0].project_name == "unknown"

    def test_empty_records(self):
        assert aggregate_by_project([]) == []


# ---------------------------------------------------------------------------
# aggregate_by_date
# ---------------------------------------------------------------------------


class TestAggregateByDate:
    def test_by_day(self):
        result = aggregate_by_date(SAMPLE_RECORDS, interval="day")
        assert "2026-01-01" in result
        assert "2026-01-02" in result
        assert "2026-01-03" in result
        assert result["2026-01-01"]["count"] == 2

    def test_by_month(self):
        records = SAMPLE_RECORDS + [
            {"timestamp": "2026-02-01T10:00:00Z", "cost": 0.1, "tokens": 500},
        ]
        result = aggregate_by_date(records, interval="month")
        assert "2026-01" in result
        assert "2026-02" in result
        assert result["2026-01"]["count"] == 5

    def test_by_week(self):
        result = aggregate_by_date(SAMPLE_RECORDS, interval="week")
        # All records are in the same week (2026-01-01 is a Thursday)
        assert len(result) >= 1

    def test_totals(self):
        result = aggregate_by_date(SAMPLE_RECORDS, interval="day")
        day1 = result["2026-01-01"]
        assert day1["count"] == 2
        assert day1["total_tokens"] == 300  # 100+200
        assert abs(day1["total_cost"] - 0.03) < 0.0001

    def test_missing_timestamp_skipped(self):
        records = [{"cost": 0.1}, {"timestamp": "2026-01-01", "cost": 0.2}]
        result = aggregate_by_date(records, interval="day")
        assert len(result) == 1

    def test_custom_date_field(self):
        records = [{"created": "2026-03-01", "cost": 1.0, "tokens": 50}]
        result = aggregate_by_date(records, date_field="created", interval="day")
        assert "2026-03-01" in result

    def test_empty_records(self):
        assert aggregate_by_date([]) == {}


# ---------------------------------------------------------------------------
# aggregate_by_model
# ---------------------------------------------------------------------------


class TestAggregateByModel:
    def test_basic(self):
        result = aggregate_by_model(SAMPLE_RECORDS)
        assert "gpt-4" in result
        assert "claude-3" in result

    def test_counts(self):
        result = aggregate_by_model(SAMPLE_RECORDS)
        assert result["gpt-4"]["count"] == 3
        assert result["claude-3"]["count"] == 2

    def test_total_tokens(self):
        result = aggregate_by_model(SAMPLE_RECORDS)
        assert result["gpt-4"]["total_tokens"] == 600  # 100+200+300

    def test_total_cost(self):
        result = aggregate_by_model(SAMPLE_RECORDS)
        assert abs(result["gpt-4"]["total_cost"] - 0.06) < 0.0001

    def test_avg_latency(self):
        result = aggregate_by_model(SAMPLE_RECORDS)
        expected = (0.5 + 0.6 + 0.8) / 3
        assert abs(result["gpt-4"]["avg_latency"] - expected) < 0.001

    def test_missing_model_skipped(self):
        records = [{"tokens": 100, "cost": 0.1}, {"model": "x", "tokens": 50, "cost": 0.05}]
        result = aggregate_by_model(records)
        assert len(result) == 1

    def test_custom_model_field(self):
        records = [{"engine": "davinci", "tokens": 100, "cost": 0.1, "latency": 0.5}]
        result = aggregate_by_model(records, model_field="engine")
        assert "davinci" in result

    def test_no_latency_field(self):
        records = [{"model": "x", "tokens": 100, "cost": 0.1}]
        result = aggregate_by_model(records)
        assert result["x"]["avg_latency"] == 0.0

    def test_empty_records(self):
        assert aggregate_by_model([]) == {}


# ---------------------------------------------------------------------------
# format_aggregation_result
# ---------------------------------------------------------------------------


class TestFormatAggregationResult:
    def test_basic_format(self):
        q = AggregationQuery(["model"], "cost", "sum")
        r = AggregationResult([{"model": "gpt-4", "value": 0.06}], 3, q)
        text = format_aggregation_result(r)
        assert "gpt-4" in text
        assert "0.06" in text
        assert "Total records: 3" in text

    def test_empty_groups(self):
        q = AggregationQuery(["x"], "y", "sum")
        r = AggregationResult([], 0, q)
        text = format_aggregation_result(r)
        assert "No results" in text

    def test_multiple_rows(self):
        q = AggregationQuery(["model"], "cost", "sum")
        r = AggregationResult(
            [{"model": "gpt-4", "value": 0.06}, {"model": "claude", "value": 0.04}],
            5, q,
        )
        text = format_aggregation_result(r)
        assert "gpt-4" in text
        assert "claude" in text

    def test_header_present(self):
        q = AggregationQuery(["project"], "cost", "sum")
        r = AggregationResult([{"project": "alpha", "value": 1.0}], 1, q)
        text = format_aggregation_result(r)
        lines = text.strip().split("\n")
        assert "project" in lines[0]
        assert "value" in lines[0]


# ---------------------------------------------------------------------------
# format_project_stats
# ---------------------------------------------------------------------------


class TestFormatProjectStats:
    def test_basic(self):
        stats = [ProjectStats("alpha", 10, 5000, 1.5, {"gpt-4": 8}, ("2026-01-01", "2026-01-31"))]
        text = format_project_stats(stats)
        assert "alpha" in text
        assert "5000" in text
        assert "1.5" in text

    def test_empty(self):
        text = format_project_stats([])
        assert "No project data" in text

    def test_multiple_projects(self):
        stats = [
            ProjectStats("a", 1, 100, 0.1, {}, ("d1", "d2")),
            ProjectStats("b", 2, 200, 0.2, {}, ("d3", "d4")),
        ]
        text = format_project_stats(stats)
        assert "a" in text
        assert "b" in text
        assert "Project Summary" in text

    def test_no_date_range(self):
        stats = [ProjectStats("p", 1, 1, 0.0, {}, ("", ""))]
        text = format_project_stats(stats)
        assert "n/a" in text
