"""Dataset creation from production HTTP request/response logs.

Parses log lines into structured entries, extracts LLM input/output content,
and produces importable dataset items for evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LogEntry:
    """A single parsed HTTP log entry."""

    timestamp: str = ""
    method: str = "POST"
    url: str = ""
    request_body: str = ""
    response_body: str = ""
    status_code: int = 200
    duration_ms: float = 0.0
    headers: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "method": self.method,
            "url": self.url,
            "request_body": self.request_body,
            "response_body": self.response_body,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogEntry:
        return cls(
            timestamp=data.get("timestamp", ""),
            method=data.get("method", "POST"),
            url=data.get("url", ""),
            request_body=data.get("request_body", ""),
            response_body=data.get("response_body", ""),
            status_code=data.get("status_code", 200),
            duration_ms=data.get("duration_ms", 0.0),
            headers=data.get("headers", {}),
        )


@dataclass
class ImportedLogItem:
    """A dataset item derived from a log entry."""

    id: str = ""
    input_text: str = ""
    output_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImportedLogItem:
        return cls(
            id=data.get("id", ""),
            input_text=data.get("input_text", ""),
            output_text=data.get("output_text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LogImportResult:
    """Summary of a log import operation."""

    items: List[ImportedLogItem] = field(default_factory=list)
    total_logs: int = 0
    imported: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "total_logs": self.total_logs,
            "imported": self.imported,
            "skipped": self.skipped,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogImportResult:
        return cls(
            items=[ImportedLogItem.from_dict(i) for i in data.get("items", [])],
            total_logs=data.get("total_logs", 0),
            imported=data.get("imported", 0),
            skipped=data.get("skipped", 0),
            errors=data.get("errors", []),
        )

    def format_text(self) -> str:
        lines = [
            f"Log Import: {self.imported}/{self.total_logs} imported, {self.skipped} skipped",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        return "\n".join(lines)


def parse_log_entry(line: str, format: str = "json") -> Optional[LogEntry]:
    """Parse a single log line into a LogEntry.

    Returns None on parse error.
    """
    line = line.strip()
    if not line:
        return None

    if format == "json":
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return LogEntry(
            timestamp=str(data.get("timestamp", "")),
            method=str(data.get("method", "POST")),
            url=str(data.get("url", "")),
            request_body=str(data.get("request_body", "")),
            response_body=str(data.get("response_body", "")),
            status_code=int(data.get("status_code", 200)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            headers=data.get("headers", {}),
        )

    return None


def extract_llm_content(log: LogEntry) -> Tuple[str, str]:
    """Extract LLM input and output text from a log entry.

    Looks for common LLM API patterns in request/response bodies:
    - Request: "messages" list or "prompt" field
    - Response: "choices" list or "text" field
    """
    input_text = ""
    output_text = ""

    # Parse request body
    try:
        req = json.loads(log.request_body) if log.request_body else {}
    except (json.JSONDecodeError, TypeError):
        req = {}

    if isinstance(req, dict):
        if "messages" in req and isinstance(req["messages"], list):
            parts = []
            for msg in req["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    parts.append(str(msg["content"]))
            input_text = "\n".join(parts)
        elif "prompt" in req:
            input_text = str(req["prompt"])

    # Parse response body
    try:
        resp = json.loads(log.response_body) if log.response_body else {}
    except (json.JSONDecodeError, TypeError):
        resp = {}

    if isinstance(resp, dict):
        if "choices" in resp and isinstance(resp["choices"], list):
            parts = []
            for choice in resp["choices"]:
                if isinstance(choice, dict):
                    msg = choice.get("message", {})
                    if isinstance(msg, dict) and "content" in msg:
                        parts.append(str(msg["content"]))
                    elif "text" in choice:
                        parts.append(str(choice["text"]))
            output_text = "\n".join(parts)
        elif "text" in resp:
            output_text = str(resp["text"])

    return input_text, output_text


def import_from_logs(
    log_lines: List[str],
    format: str = "json",
    min_status: int = 200,
    max_status: int = 299,
) -> LogImportResult:
    """Parse log lines and convert to dataset items.

    Filters by status code range. Lines that fail to parse are counted as errors.
    Lines with status codes outside the range are skipped.
    """
    result = LogImportResult(total_logs=len(log_lines))

    for i, line in enumerate(log_lines):
        entry = parse_log_entry(line, format=format)
        if entry is None:
            result.errors.append(f"Line {i}: parse error")
            continue

        if not (min_status <= entry.status_code <= max_status):
            result.skipped += 1
            continue

        input_text, output_text = extract_llm_content(entry)
        item_id = hashlib.sha256(
            f"{entry.timestamp}:{entry.url}:{i}".encode()
        ).hexdigest()[:16]

        result.items.append(
            ImportedLogItem(
                id=item_id,
                input_text=input_text,
                output_text=output_text,
                metadata={
                    "url": entry.url,
                    "method": entry.method,
                    "status_code": entry.status_code,
                    "duration_ms": entry.duration_ms,
                    "timestamp": entry.timestamp,
                },
            )
        )
        result.imported += 1

    return result


def detect_log_format(sample: str) -> str:
    """Auto-detect the format of a log sample.

    Returns "json", "csv", or "text".
    """
    sample = sample.strip()
    if not sample:
        return "text"

    # Check first non-empty line
    first_line = sample.split("\n")[0].strip()

    # JSON: starts with { or [
    if first_line.startswith("{") or first_line.startswith("["):
        try:
            json.loads(first_line)
            return "json"
        except (json.JSONDecodeError, TypeError):
            pass

    # CSV: contains commas and no obvious JSON
    if "," in first_line and not first_line.startswith("{"):
        parts = first_line.split(",")
        if len(parts) >= 3:
            return "csv"

    return "text"


def filter_logs_by_endpoint(
    logs: List[LogEntry], url_pattern: str
) -> List[LogEntry]:
    """Filter log entries by URL substring match."""
    return [log for log in logs if url_pattern in log.url]
