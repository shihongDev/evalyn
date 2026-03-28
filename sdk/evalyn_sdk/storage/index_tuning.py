"""
Storage index tuning: auto-create indexes based on common query patterns.

Provides pure functions to analyze query patterns, suggest indexes, detect
redundant indexes, and generate CREATE INDEX SQL statements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class IndexSuggestion:
    """A suggested index for a table based on query pattern analysis."""

    table: str
    columns: List[str]
    index_name: str = ""
    reason: str = ""
    estimated_speedup: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "columns": self.columns,
            "index_name": self.index_name,
            "reason": self.reason,
            "estimated_speedup": self.estimated_speedup,
        }


@dataclass
class ExistingIndex:
    """An index that already exists in the database."""

    name: str
    table: str
    columns: List[str]
    unique: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "columns": self.columns,
            "unique": self.unique,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExistingIndex:
        return cls(
            name=data.get("name", ""),
            table=data.get("table", ""),
            columns=data.get("columns", []),
            unique=data.get("unique", False),
        )


@dataclass
class IndexTuningReport:
    """Full index tuning report with suggestions and redundancy info."""

    suggestions: List[IndexSuggestion] = field(default_factory=list)
    existing: List[ExistingIndex] = field(default_factory=list)
    redundant: List[str] = field(default_factory=list)
    total_suggested: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suggestions": [s.as_dict() for s in self.suggestions],
            "existing": [e.as_dict() for e in self.existing],
            "redundant": self.redundant,
            "total_suggested": self.total_suggested,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IndexTuningReport:
        suggestions = [
            IndexSuggestion(
                table=s.get("table", ""),
                columns=s.get("columns", []),
                index_name=s.get("index_name", ""),
                reason=s.get("reason", ""),
                estimated_speedup=s.get("estimated_speedup", ""),
            )
            for s in data.get("suggestions", [])
        ]
        existing = [
            ExistingIndex.from_dict(e) for e in data.get("existing", [])
        ]
        return cls(
            suggestions=suggestions,
            existing=existing,
            redundant=data.get("redundant", []),
            total_suggested=data.get("total_suggested", 0),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Index Tuning Report")
        lines.append("=" * 40)
        lines.append(f"Existing indexes: {len(self.existing)}")
        lines.append(f"Suggested indexes: {self.total_suggested}")
        lines.append(f"Redundant indexes: {len(self.redundant)}")

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for s in self.suggestions:
                cols = ", ".join(s.columns)
                name = s.index_name or "unnamed"
                lines.append(f"  {name}: {s.table}({cols})")
                if s.reason:
                    lines.append(f"    Reason: {s.reason}")
                if s.estimated_speedup:
                    lines.append(f"    Estimated speedup: {s.estimated_speedup}")

        if self.redundant:
            lines.append("")
            lines.append("Redundant indexes:")
            for r in self.redundant:
                lines.append(f"  - {r}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def parse_query_columns(query: str) -> List[str]:
    """Extract column names from a WHERE clause string (simple regex).

    Looks for patterns like: column = value, column > value, column IN (...),
    column LIKE value, column IS NULL, etc.
    """
    if not query or not query.strip():
        return []

    # Match column names before comparison operators
    # Handles: col = val, col > val, col < val, col >= val, col <= val,
    # col != val, col <> val, col LIKE val, col IN (...), col IS NULL,
    # col BETWEEN val AND val
    pattern = r'\b(\w+)\s*(?:=|!=|<>|>=|<=|>|<|(?:NOT\s+)?(?:LIKE|IN|BETWEEN|IS)\b)'
    matches = re.findall(pattern, query, re.IGNORECASE)

    # Filter out SQL keywords that might be captured
    sql_keywords = {
        "and", "or", "not", "where", "select", "from", "join", "on",
        "set", "insert", "update", "delete", "null", "is", "in", "like",
        "between", "having", "group", "order", "by", "limit", "offset",
        "true", "false", "case", "when", "then", "else", "end",
    }

    seen: set[str] = set()
    result: List[str] = []
    for col in matches:
        lower = col.lower()
        if lower not in sql_keywords and lower not in seen:
            seen.add(lower)
            result.append(col)

    return result


def suggest_indexes(
    table_name: str,
    query_patterns: List[str],
    existing_indexes: List[ExistingIndex] | None = None,
) -> List[IndexSuggestion]:
    """Analyze query patterns and suggest indexes.

    Examines WHERE clauses and JOIN conditions to find frequently filtered
    columns. Skips columns already covered by existing indexes.
    """
    if existing_indexes is None:
        existing_indexes = []

    if not query_patterns:
        return []

    # Collect column frequencies across all query patterns
    column_counts: Dict[str, int] = {}
    for pattern in query_patterns:
        cols = parse_query_columns(pattern)
        for col in cols:
            column_counts[col] = column_counts.get(col, 0) + 1

    if not column_counts:
        return []

    # Build set of already-indexed column sets for this table
    indexed_columns: set[str] = set()
    for idx in existing_indexes:
        if idx.table == table_name:
            for col in idx.columns:
                indexed_columns.add(col.lower())

    # Generate suggestions for unindexed columns, sorted by frequency
    suggestions: List[IndexSuggestion] = []
    sorted_cols = sorted(column_counts.items(), key=lambda x: x[1], reverse=True)

    for col, count in sorted_cols:
        if col.lower() in indexed_columns:
            continue

        idx_name = f"idx_{table_name}_{col}"
        reason = f"column '{col}' used in {count} query pattern(s)"
        speedup = "moderate" if count >= 2 else "low"

        suggestions.append(
            IndexSuggestion(
                table=table_name,
                columns=[col],
                index_name=idx_name,
                reason=reason,
                estimated_speedup=speedup,
            )
        )

    return suggestions


def detect_redundant_indexes(indexes: List[ExistingIndex]) -> List[str]:
    """Find indexes that are prefix-subsets of other indexes.

    An index on (a) is redundant if another index on (a, b) exists for the
    same table, since the composite index covers the single-column case.
    """
    if not indexes:
        return []

    redundant: List[str] = []

    for i, idx_a in enumerate(indexes):
        cols_a = [c.lower() for c in idx_a.columns]
        if not cols_a:
            continue

        for j, idx_b in enumerate(indexes):
            if i == j:
                continue
            if idx_a.table != idx_b.table:
                continue

            cols_b = [c.lower() for c in idx_b.columns]
            if not cols_b:
                continue

            # Check if cols_a is a strict prefix of cols_b
            if len(cols_a) < len(cols_b) and cols_b[: len(cols_a)] == cols_a:
                if idx_a.name not in redundant:
                    redundant.append(idx_a.name)
                break

    return redundant


def generate_create_index_sql(suggestion: IndexSuggestion) -> str:
    """Generate a CREATE INDEX SQL statement from a suggestion."""
    name = suggestion.index_name or f"idx_{suggestion.table}_{'_'.join(suggestion.columns)}"
    cols = ", ".join(suggestion.columns)
    return f"CREATE INDEX {name} ON {suggestion.table} ({cols});"


def build_tuning_report(
    suggestions: List[IndexSuggestion],
    existing: List[ExistingIndex],
) -> IndexTuningReport:
    """Build a full index tuning report."""
    redundant = detect_redundant_indexes(existing)
    return IndexTuningReport(
        suggestions=suggestions,
        existing=existing,
        redundant=redundant,
        total_suggested=len(suggestions),
    )
