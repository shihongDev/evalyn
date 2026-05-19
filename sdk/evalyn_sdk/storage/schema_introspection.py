"""Storage schema introspection: show current database schema and statistics.

Provides dataclasses and pure functions to inspect table schemas, render
ASCII reports, compare schemas, and suggest optimizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ColumnInfo:
    """Metadata for a single database column."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
        }


@dataclass
class TableSchema:
    """Schema for a single database table."""

    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    index_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.as_dict() for c in self.columns],
            "row_count": self.row_count,
            "index_count": self.index_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableSchema:
        columns = [
            ColumnInfo(
                name=c["name"],
                data_type=c.get("data_type", ""),
                nullable=c.get("nullable", True),
                primary_key=c.get("primary_key", False),
            )
            for c in data.get("columns", [])
        ]
        return cls(
            name=data["name"],
            columns=columns,
            row_count=data.get("row_count", 0),
            index_count=data.get("index_count", 0),
        )


@dataclass
class SchemaReport:
    """Aggregated report of all table schemas."""

    tables: list[TableSchema] = field(default_factory=list)
    total_tables: int = 0
    total_columns: int = 0
    total_rows: int = 0
    total_indexes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tables": [t.as_dict() for t in self.tables],
            "total_tables": self.total_tables,
            "total_columns": self.total_columns,
            "total_rows": self.total_rows,
            "total_indexes": self.total_indexes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaReport:
        tables = [
            TableSchema.from_dict(t) for t in data.get("tables", [])
        ]
        return cls(
            tables=tables,
            total_tables=data.get("total_tables", 0),
            total_columns=data.get("total_columns", 0),
            total_rows=data.get("total_rows", 0),
            total_indexes=data.get("total_indexes", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Schema Report")
        lines.append("-" * 40)
        lines.append(f"Tables: {self.total_tables}")
        lines.append(f"Total columns: {self.total_columns}")
        lines.append(f"Total rows: {self.total_rows}")
        lines.append(f"Total indexes: {self.total_indexes}")
        if self.tables:
            lines.append("")
            for table in self.tables:
                lines.append(f"  {table.name}: {len(table.columns)} cols, {table.row_count} rows")
                for col in table.columns:
                    pk = " [PK]" if col.primary_key else ""
                    null = " NULL" if col.nullable else " NOT NULL"
                    lines.append(f"    - {col.name} ({col.data_type}){null}{pk}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_table_schema(
    name: str,
    columns: list[dict[str, Any]],
    row_count: int = 0,
) -> TableSchema:
    """Build a TableSchema from column dicts.

    Each column dict should have "name", "type", and optionally "nullable"
    and "pk" keys.
    """
    col_infos = [
        ColumnInfo(
            name=c["name"],
            data_type=c.get("type", ""),
            nullable=c.get("nullable", True),
            primary_key=c.get("pk", False),
        )
        for c in columns
    ]
    return TableSchema(name=name, columns=col_infos, row_count=row_count)


def build_schema_report(tables: list[TableSchema]) -> SchemaReport:
    """Aggregate a list of TableSchemas into a SchemaReport."""
    total_cols = sum(len(t.columns) for t in tables)
    total_rows = sum(t.row_count for t in tables)
    total_indexes = sum(t.index_count for t in tables)
    return SchemaReport(
        tables=tables,
        total_tables=len(tables),
        total_columns=total_cols,
        total_rows=total_rows,
        total_indexes=total_indexes,
    )


def render_schema_ascii(report: SchemaReport) -> str:
    """Render an ASCII table showing table name, column count, and row count."""
    header = f"{'Table':<20} {'Columns':>8} {'Rows':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for table in report.tables:
        lines.append(
            f"{table.name:<20} {len(table.columns):>8} {table.row_count:>10}"
        )
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<20} {report.total_columns:>8} {report.total_rows:>10}"
    )
    return "\n".join(lines)


def compare_schemas(
    schema_a: SchemaReport,
    schema_b: SchemaReport,
) -> dict[str, Any]:
    """Compare two schemas. Find added, removed, and changed tables."""
    names_a = {t.name for t in schema_a.tables}
    names_b = {t.name for t in schema_b.tables}

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)

    tables_a = {t.name: t for t in schema_a.tables}
    tables_b = {t.name: t for t in schema_b.tables}

    changed: list[dict[str, Any]] = []
    for name in sorted(names_a & names_b):
        ta = tables_a[name]
        tb = tables_b[name]
        cols_a = {c.name for c in ta.columns}
        cols_b = {c.name for c in tb.columns}
        added_cols = sorted(cols_b - cols_a)
        removed_cols = sorted(cols_a - cols_b)
        if added_cols or removed_cols or ta.row_count != tb.row_count:
            changed.append({
                "table": name,
                "added_columns": added_cols,
                "removed_columns": removed_cols,
                "row_count_a": ta.row_count,
                "row_count_b": tb.row_count,
            })

    return {
        "added_tables": added,
        "removed_tables": removed,
        "changed_tables": changed,
    }


def suggest_schema_optimizations(report: SchemaReport) -> list[str]:
    """Return optimization suggestions based on schema structure."""
    suggestions: list[str] = []
    for table in report.tables:
        if table.index_count == 0 and table.row_count > 0:
            suggestions.append(
                f"Table '{table.name}' has no indexes but contains {table.row_count} rows"
            )
        has_pk = any(c.primary_key for c in table.columns)
        if not has_pk and table.columns:
            suggestions.append(
                f"Table '{table.name}' has no primary key column"
            )
        all_nullable = all(c.nullable for c in table.columns) if table.columns else False
        if all_nullable and len(table.columns) > 1:
            suggestions.append(
                f"Table '{table.name}' has all nullable columns - consider adding NOT NULL constraints"
            )
    return suggestions
