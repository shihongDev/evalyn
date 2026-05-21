from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormatDetection:
    """Result of format detection."""

    format: str = "unknown"  # json / jsonl / csv / yaml / tsv / unknown
    confidence: float = 0.0  # 0-1
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class AutoloadResult:
    """Result of autoloading a dataset file."""

    items: list[dict[str, Any]] = field(default_factory=list)
    format_detected: str = ""
    total_loaded: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": list(self.items),
            "format_detected": self.format_detected,
            "total_loaded": self.total_loaded,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoloadResult:
        return cls(
            items=data.get("items", []),
            format_detected=data.get("format_detected", ""),
            total_loaded=data.get("total_loaded", 0),
            errors=data.get("errors", []),
        )

    def format_text(self) -> str:
        lines = [
            "Autoload Result",
            "-" * 40,
            f"  Format: {self.format_detected}",
            f"  Total loaded: {self.total_loaded}",
        ]
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                lines.append(f"    - {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extension mapping
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".yaml": "yaml",
    ".yml": "yaml",
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def detect_format_from_extension(file_path: str) -> FormatDetection:
    """Detect format from file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    fmt = _EXTENSION_MAP.get(ext, "unknown")
    if fmt == "unknown":
        return FormatDetection(
            format="unknown", confidence=0.0, evidence=f"unknown extension: {ext}"
        )

    return FormatDetection(
        format=fmt,
        confidence=0.9,
        evidence=f"extension: {ext}",
    )


def detect_format_from_content(content: str) -> FormatDetection:
    """Detect format from content inspection."""
    stripped = content.strip()
    if not stripped:
        return FormatDetection(format="unknown", confidence=0.0, evidence="empty content")

    # Check JSON (array or object)
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return FormatDetection(format="json", confidence=0.95, evidence="valid JSON")
        except json.JSONDecodeError:
            pass

    # Check JSONL (first line is valid JSON object)
    first_line = stripped.split("\n", 1)[0].strip()
    if first_line.startswith("{"):
        try:
            json.loads(first_line)
            return FormatDetection(
                format="jsonl", confidence=0.85, evidence="first line is JSON object"
            )
        except json.JSONDecodeError:
            pass

    # Check TSV (tabs in first line, multiple columns)
    if "\t" in first_line:
        parts = first_line.split("\t")
        if len(parts) >= 2:
            return FormatDetection(format="tsv", confidence=0.7, evidence="tab-separated columns")

    # Check CSV (commas in first line, multiple columns)
    if "," in first_line:
        try:
            reader = csv.reader(io.StringIO(stripped))
            row = next(reader)
            if len(row) >= 2:
                return FormatDetection(
                    format="csv", confidence=0.7, evidence="comma-separated columns"
                )
        except Exception:
            pass

    # Check YAML (key: value pattern)
    if ":" in first_line and not first_line.startswith("{"):
        return FormatDetection(format="yaml", confidence=0.5, evidence="key-colon-value pattern")

    return FormatDetection(format="unknown", confidence=0.0, evidence="no format detected")


def detect_format(file_path: str, content: str | None = None) -> FormatDetection:
    """Combine extension and content detection.

    Extension detection is preferred when available.
    Content detection is used as fallback or confirmation.
    """
    ext_result = detect_format_from_extension(file_path)

    if content is not None:
        content_result = detect_format_from_content(content)
    else:
        # Try reading the file
        try:
            with open(file_path, encoding="utf-8") as f:
                raw = f.read(4096)
            content_result = detect_format_from_content(raw)
        except OSError:
            content_result = FormatDetection()

    # If extension gives a known format, prefer it
    if ext_result.format != "unknown":
        # Boost confidence if content agrees
        if content_result.format == ext_result.format:
            return FormatDetection(
                format=ext_result.format,
                confidence=min(ext_result.confidence + 0.05, 1.0),
                evidence=f"{ext_result.evidence}; {content_result.evidence}",
            )
        return ext_result

    # Fall back to content detection
    return content_result


def _load_json(content: str) -> list[dict[str, Any]]:
    """Load JSON content. Handles both array and single object."""
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _load_jsonl(content: str) -> list[dict[str, Any]]:
    """Load JSONL content (one JSON object per line)."""
    items: list[dict[str, Any]] = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def _load_csv(content: str) -> list[dict[str, Any]]:
    """Load CSV content with header row."""
    reader = csv.DictReader(io.StringIO(content))
    return [dict(row) for row in reader]


def autoload(file_path: str) -> AutoloadResult:
    """Auto-detect format and load a dataset file.

    Tries JSON, JSONL, CSV in order based on detected format.
    """
    errors: list[str] = []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return AutoloadResult(errors=[f"cannot read file: {exc}"])

    if not content.strip():
        return AutoloadResult(format_detected="unknown", errors=["empty file"])

    detection = detect_format(file_path, content)
    fmt = detection.format

    # Try the detected format first, then fall back
    loaders = [
        ("json", _load_json),
        ("jsonl", _load_jsonl),
        ("csv", _load_csv),
    ]

    # Put detected format first
    ordered = [(n, fn) for n, fn in loaders if n == fmt]
    ordered += [(n, fn) for n, fn in loaders if n != fmt]

    for loader_name, loader_fn in ordered:
        try:
            items = loader_fn(content)
            if items:
                return AutoloadResult(
                    items=items,
                    format_detected=loader_name,
                    total_loaded=len(items),
                )
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    return AutoloadResult(
        format_detected=fmt if fmt != "unknown" else "",
        errors=errors or ["no items loaded"],
    )


def autoload_from_string(content: str, hint: str = "") -> AutoloadResult:
    """Load from string content with optional format hint.

    Args:
        content: Raw string content.
        hint: Format hint (e.g. "json", "csv"). If empty, auto-detects.
    """
    errors: list[str] = []

    if not content.strip():
        return AutoloadResult(format_detected="unknown", errors=["empty content"])

    if hint:
        fmt = hint
    else:
        detection = detect_format_from_content(content)
        fmt = detection.format

    loaders = [
        ("json", _load_json),
        ("jsonl", _load_jsonl),
        ("csv", _load_csv),
    ]

    # Put hinted/detected format first
    ordered = [(n, fn) for n, fn in loaders if n == fmt]
    ordered += [(n, fn) for n, fn in loaders if n != fmt]

    for loader_name, loader_fn in ordered:
        try:
            items = loader_fn(content)
            if items:
                return AutoloadResult(
                    items=items,
                    format_detected=loader_name,
                    total_loaded=len(items),
                )
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    return AutoloadResult(
        format_detected=fmt if fmt != "unknown" else "",
        errors=errors or ["no items loaded"],
    )


def supported_formats() -> list[str]:
    """List of supported dataset formats."""
    return ["json", "jsonl", "csv", "tsv", "yaml"]
