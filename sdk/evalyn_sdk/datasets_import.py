"""
External format import: CSV, JSONL, and in-memory dicts.

Maps external column schemas (HuggingFace, LMSYS Arena, custom CSV)
into evalyn-native DatasetItem-compatible dicts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ImportConfig:
    """Configuration for importing from an external format."""

    format: str  # "csv" / "jsonl" / "huggingface" / "lmsys"
    file_path: str = ""
    input_column: str = "input"
    output_column: str = "output"
    id_column: str = "id"
    delimiter: str = ","

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "file_path": self.file_path,
            "input_column": self.input_column,
            "output_column": self.output_column,
            "id_column": self.id_column,
            "delimiter": self.delimiter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImportConfig:
        return cls(
            format=data["format"],
            file_path=data.get("file_path", ""),
            input_column=data.get("input_column", "input"),
            output_column=data.get("output_column", "output"),
            id_column=data.get("id_column", "id"),
            delimiter=data.get("delimiter", ","),
        )


@dataclass
class ImportedItem:
    """A single item produced by the import process."""

    id: str
    input_text: str
    output_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source_format: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "metadata": self.metadata,
            "source_format": self.source_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImportedItem:
        return cls(
            id=data["id"],
            input_text=data["input_text"],
            output_text=data.get("output_text", ""),
            metadata=data.get("metadata", {}),
            source_format=data.get("source_format", ""),
        )


@dataclass
class ImportResult:
    """Outcome of an import operation."""

    items: list[ImportedItem] = field(default_factory=list)
    total_imported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    source_format: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [it.as_dict() for it in self.items],
            "total_imported": self.total_imported,
            "skipped": self.skipped,
            "errors": self.errors,
            "source_format": self.source_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImportResult:
        return cls(
            items=[ImportedItem.from_dict(d) for d in data.get("items", [])],
            total_imported=data.get("total_imported", 0),
            skipped=data.get("skipped", 0),
            errors=data.get("errors", []),
            source_format=data.get("source_format", ""),
        )

    def format_text(self) -> str:
        lines = [
            "Import Result",
            f"  Source format: {self.source_format}",
            f"  Imported:      {self.total_imported}",
            f"  Skipped:       {self.skipped}",
            f"  Errors:        {len(self.errors)}",
        ]
        for err in self.errors[:5]:
            lines.append(f"    - {err}")
        if len(self.errors) > 5:
            lines.append(f"    ... and {len(self.errors) - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Import Functions
# ---------------------------------------------------------------------------


def import_from_csv(file_path: str, config: ImportConfig) -> ImportResult:
    """Read a CSV file and map columns according to config."""
    items: list[ImportedItem] = []
    errors: list[str] = []
    skipped = 0
    row_num = 0

    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=config.delimiter)
            for row_num, row in enumerate(reader, start=1):
                input_val = row.get(config.input_column, "")
                if not input_val:
                    skipped += 1
                    continue
                item_id = row.get(config.id_column, str(row_num))
                output_val = row.get(config.output_column, "")
                # Remaining columns go into metadata
                meta_keys = set(row.keys()) - {
                    config.input_column,
                    config.output_column,
                    config.id_column,
                }
                metadata = {k: row[k] for k in sorted(meta_keys) if row.get(k)}
                items.append(
                    ImportedItem(
                        id=item_id,
                        input_text=input_val,
                        output_text=output_val,
                        metadata=metadata,
                        source_format="csv",
                    )
                )
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as exc:
        errors.append(f"CSV read error: {exc}")

    return ImportResult(
        items=items,
        total_imported=len(items),
        skipped=skipped,
        errors=errors,
        source_format="csv",
    )


def import_from_jsonl(file_path: str, config: ImportConfig) -> ImportResult:
    """Read a JSONL file line by line, mapping fields via config."""
    items: list[ImportedItem] = []
    errors: list[str] = []
    skipped = 0

    try:
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Line {line_num}: {exc}")
                    skipped += 1
                    continue

                input_val = record.get(config.input_column, "")
                if not input_val:
                    skipped += 1
                    continue

                item_id = record.get(config.id_column, str(line_num))
                output_val = record.get(config.output_column, "")
                meta_keys = set(record.keys()) - {
                    config.input_column,
                    config.output_column,
                    config.id_column,
                }
                metadata = {k: record[k] for k in sorted(meta_keys)}
                items.append(
                    ImportedItem(
                        id=item_id,
                        input_text=input_val,
                        output_text=output_val,
                        metadata=metadata,
                        source_format="jsonl",
                    )
                )
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as exc:
        errors.append(f"JSONL read error: {exc}")

    return ImportResult(
        items=items,
        total_imported=len(items),
        skipped=skipped,
        errors=errors,
        source_format="jsonl",
    )


def import_from_dict_list(
    records: list[dict[str, Any]], config: ImportConfig
) -> ImportResult:
    """Import from a list of in-memory dicts."""
    items: list[ImportedItem] = []
    errors: list[str] = []
    skipped = 0

    for idx, record in enumerate(records):
        input_val = record.get(config.input_column, "")
        if not input_val:
            skipped += 1
            continue

        item_id = record.get(config.id_column, str(idx + 1))
        output_val = record.get(config.output_column, "")
        meta_keys = set(record.keys()) - {
            config.input_column,
            config.output_column,
            config.id_column,
        }
        metadata = {k: record[k] for k in sorted(meta_keys)}
        items.append(
            ImportedItem(
                id=item_id,
                input_text=input_val,
                output_text=output_val,
                metadata=metadata,
                source_format=config.format,
            )
        )

    return ImportResult(
        items=items,
        total_imported=len(items),
        skipped=skipped,
        errors=errors,
        source_format=config.format,
    )


def detect_format(file_path: str) -> str:
    """Auto-detect format from file extension."""
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    return "unknown"


def validate_import(result: ImportResult) -> tuple[bool, list[str]]:
    """Validate that all imported items have required fields (id, input_text)."""
    issues: list[str] = []

    for item in result.items:
        if not item.id:
            issues.append("Item missing id")
        if not item.input_text:
            issues.append(f"Item {item.id}: missing input_text")

    valid = len(issues) == 0
    return valid, issues


def convert_to_evalyn_format(items: list[ImportedItem]) -> list[dict[str, Any]]:
    """Convert ImportedItem list to evalyn DatasetItem-compatible dicts."""
    result: list[dict[str, Any]] = []
    for item in items:
        entry: dict[str, Any] = {
            "id": item.id,
            "input": {"text": item.input_text},
        }
        if item.output_text:
            entry["output"] = item.output_text
        if item.metadata:
            entry["metadata"] = item.metadata
        result.append(entry)
    return result
