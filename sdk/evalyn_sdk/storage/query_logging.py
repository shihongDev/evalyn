"""Storage query logging: log SQL queries for performance debugging.

Provides dataclasses and pure functions to record queries, compute
aggregate statistics, analyze patterns, and suggest optimizations.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class QueryLog:
    """Single recorded SQL query with timing metadata."""

    query: str
    duration_ms: float = 0.0
    timestamp: str = ""
    table: str = ""
    operation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "table": self.table,
            "operation": self.operation,
        }


@dataclass
class QueryStats:
    """Aggregate statistics across logged queries."""

    total_queries: int = 0
    avg_duration_ms: float = 0.0
    slowest_query: str = ""
    slowest_duration_ms: float = 0.0
    by_operation: Dict[str, int] = field(default_factory=dict)
    by_table: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "avg_duration_ms": self.avg_duration_ms,
            "slowest_query": self.slowest_query,
            "slowest_duration_ms": self.slowest_duration_ms,
            "by_operation": dict(self.by_operation),
            "by_table": dict(self.by_table),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryStats:
        return cls(
            total_queries=data.get("total_queries", 0),
            avg_duration_ms=data.get("avg_duration_ms", 0.0),
            slowest_query=data.get("slowest_query", ""),
            slowest_duration_ms=data.get("slowest_duration_ms", 0.0),
            by_operation=dict(data.get("by_operation", {})),
            by_table=dict(data.get("by_table", {})),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Query Stats")
        lines.append("=" * 40)
        lines.append(f"Total queries: {self.total_queries}")
        lines.append(f"Avg duration: {self.avg_duration_ms:.2f} ms")
        if self.slowest_query:
            lines.append(f"Slowest query: {self.slowest_query}")
            lines.append(f"Slowest duration: {self.slowest_duration_ms:.2f} ms")
        if self.by_operation:
            lines.append("")
            lines.append("By operation:")
            for op, count in sorted(self.by_operation.items()):
                lines.append(f"  {op}: {count}")
        if self.by_table:
            lines.append("")
            lines.append("By table:")
            for table, count in sorted(self.by_table.items()):
                lines.append(f"  {table}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QueryLogger
# ---------------------------------------------------------------------------


class QueryLogger:
    """Accumulates query logs with a configurable size limit."""

    def __init__(self, max_logs: int = 1000) -> None:
        self._logs: List[QueryLog] = []
        self._max_logs = max_logs

    def log(self, query: str, duration_ms: float = 0.0) -> None:
        """Record a query. Auto-detects operation and table."""
        operation = detect_operation(query)
        table = detect_table(query)
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = QueryLog(
            query=query,
            duration_ms=duration_ms,
            timestamp=timestamp,
            table=table,
            operation=operation,
        )
        self._logs.append(entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

    def get_logs(self) -> List[QueryLog]:
        """Return all recorded logs."""
        return list(self._logs)

    def get_slow_queries(self, threshold_ms: float = 100.0) -> List[QueryLog]:
        """Return queries slower than threshold."""
        return [log for log in self._logs if log.duration_ms > threshold_ms]

    def get_stats(self) -> QueryStats:
        """Compute aggregate statistics over all recorded logs."""
        if not self._logs:
            return QueryStats()

        total = len(self._logs)
        total_duration = sum(log.duration_ms for log in self._logs)
        avg = total_duration / total

        slowest = max(self._logs, key=lambda l: l.duration_ms)

        by_operation: Dict[str, int] = defaultdict(int)
        by_table: Dict[str, int] = defaultdict(int)
        for log in self._logs:
            if log.operation:
                by_operation[log.operation] += 1
            if log.table:
                by_table[log.table] += 1

        return QueryStats(
            total_queries=total,
            avg_duration_ms=round(avg, 2),
            slowest_query=slowest.query,
            slowest_duration_ms=slowest.duration_ms,
            by_operation=dict(by_operation),
            by_table=dict(by_table),
        )

    def clear(self) -> None:
        """Clear all logs."""
        self._logs.clear()

    @property
    def count(self) -> int:
        """Number of recorded logs."""
        return len(self._logs)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

_OPERATION_PATTERN = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE
)


def detect_operation(query: str) -> str:
    """Extract the SQL operation (SELECT/INSERT/UPDATE/DELETE/OTHER)."""
    match = _OPERATION_PATTERN.match(query)
    if match:
        return match.group(1).upper()
    return "OTHER"


_TABLE_PATTERNS = [
    re.compile(r"\bFROM\s+(\w+)", re.IGNORECASE),
    re.compile(r"\bINTO\s+(\w+)", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+(\w+)", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\s+(\w+)", re.IGNORECASE),
]


def detect_table(query: str) -> str:
    """Extract the primary table name from a SQL query."""
    for pattern in _TABLE_PATTERNS:
        match = pattern.search(query)
        if match:
            return match.group(1)
    return ""


def analyze_query_patterns(logs: List[QueryLog]) -> Dict[str, Any]:
    """Find repeated queries, most accessed tables, and operation distribution."""
    if not logs:
        return {
            "repeated_queries": {},
            "most_accessed_tables": {},
            "operation_distribution": {},
        }

    query_counts: Dict[str, int] = defaultdict(int)
    table_counts: Dict[str, int] = defaultdict(int)
    operation_counts: Dict[str, int] = defaultdict(int)

    for log in logs:
        query_counts[log.query] += 1
        if log.table:
            table_counts[log.table] += 1
        if log.operation:
            operation_counts[log.operation] += 1

    repeated = {q: c for q, c in query_counts.items() if c > 1}

    return {
        "repeated_queries": dict(repeated),
        "most_accessed_tables": dict(table_counts),
        "operation_distribution": dict(operation_counts),
    }


def suggest_optimizations(stats: QueryStats) -> List[str]:
    """Generate optimization suggestions based on query statistics."""
    suggestions: List[str] = []

    if stats.slowest_duration_ms > 1000:
        suggestions.append(
            "Slowest query exceeds 1s - consider adding indexes or optimizing"
        )

    if stats.avg_duration_ms > 500:
        suggestions.append(
            "High average query duration - review query patterns"
        )

    if stats.total_queries > 500:
        suggestions.append(
            "High query volume - consider batching or caching"
        )

    select_count = stats.by_operation.get("SELECT", 0)
    if stats.total_queries > 0 and select_count / stats.total_queries > 0.9:
        suggestions.append(
            "Read-heavy workload - consider read replicas or caching"
        )

    if len(stats.by_table) == 1 and stats.total_queries > 100:
        suggestions.append(
            "All queries target one table - check for missing joins"
        )

    return suggestions
