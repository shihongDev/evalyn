"""
Columnar export framework for evaluation results.

Defines schema, data conversion, and export helpers. Actual parquet
writing requires pyarrow; CSV and JSON fallbacks are always available.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParquetSchema:
    """Schema describing columns for a parquet dataset."""

    columns: List[Tuple[str, str]]  # (name, type) pairs
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "columns": [list(c) for c in self.columns],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParquetSchema:
        columns = [tuple(c) for c in data["columns"]]
        return cls(
            columns=columns,
            description=data.get("description", ""),
        )


@dataclass
class ParquetRow:
    """A single row of column values."""

    values: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"values": dict(self.values)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParquetRow:
        return cls(values=data["values"])


@dataclass
class ParquetDataset:
    """A dataset consisting of a schema and rows."""

    schema: ParquetSchema
    rows: List[ParquetRow]
    row_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.as_dict(),
            "rows": [r.as_dict() for r in self.rows],
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParquetDataset:
        schema = ParquetSchema.from_dict(data["schema"])
        rows = [ParquetRow.from_dict(r) for r in data["rows"]]
        return cls(
            schema=schema,
            rows=rows,
            row_count=data["row_count"],
        )


# ---------------------------------------------------------------------------
# Standard schema for evaluation results
# ---------------------------------------------------------------------------

EVAL_RESULT_SCHEMA = ParquetSchema(
    columns=[
        ("item_id", "str"),
        ("metric_id", "str"),
        ("score", "float"),
        ("passed", "bool"),
        ("input_text", "str"),
        ("output_text", "str"),
        ("run_id", "str"),
        ("timestamp", "str"),
    ],
    description="Standard schema for evalyn evaluation results.",
)


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


def convert_results_to_rows(
    results: List[Dict[str, Any]],
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[ParquetRow]:
    """Convert evalyn result dicts to ParquetRow objects.

    Joins with items by item_id when items are provided, pulling
    input_text and output_text from matching items.
    """
    items_lookup: Dict[str, Dict[str, Any]] = {}
    if items:
        for item in items:
            iid = item.get("item_id", "")
            if iid:
                items_lookup[iid] = item

    rows: List[ParquetRow] = []
    for r in results:
        item_id = r.get("item_id", "")
        item_data = items_lookup.get(item_id, {})
        values: Dict[str, Any] = {
            "item_id": item_id,
            "metric_id": r.get("metric_id", ""),
            "score": r.get("score", 0.0),
            "passed": r.get("passed", False),
            "input_text": item_data.get("input_text", r.get("input_text", "")),
            "output_text": item_data.get("output_text", r.get("output_text", "")),
            "run_id": r.get("run_id", ""),
            "timestamp": r.get("timestamp", ""),
        }
        rows.append(ParquetRow(values=values))
    return rows


def build_parquet_dataset(
    results: List[Dict[str, Any]],
    items: Optional[List[Dict[str, Any]]] = None,
    schema: Optional[ParquetSchema] = None,
) -> ParquetDataset:
    """Build a ParquetDataset from result dicts and optional items."""
    if schema is None:
        schema = EVAL_RESULT_SCHEMA
    rows = convert_results_to_rows(results, items)
    return ParquetDataset(schema=schema, rows=rows, row_count=len(rows))


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_as_csv_fallback(dataset: ParquetDataset) -> str:
    """Export dataset as CSV string (fallback when pyarrow unavailable)."""
    col_names = [c[0] for c in dataset.schema.columns]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(col_names)
    for row in dataset.rows:
        writer.writerow([row.values.get(c, "") for c in col_names])
    return buf.getvalue()


def export_as_json_rows(dataset: ParquetDataset) -> str:
    """Export dataset as a JSON array of row dicts."""
    dicts = [row.values for row in dataset.rows]
    return json.dumps(dicts, indent=2)


def check_pyarrow_available() -> bool:
    """Return True if pyarrow can be imported, False otherwise."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def format_schema_info(schema: ParquetSchema) -> str:
    """Format a human-readable description of the schema."""
    lines: List[str] = []
    if schema.description:
        lines.append(schema.description)
    lines.append("Columns:")
    for name, col_type in schema.columns:
        lines.append(f"  {name} ({col_type})")
    return "\n".join(lines)
