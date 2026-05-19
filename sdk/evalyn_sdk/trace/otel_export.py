from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models import Span


@dataclass
class OTelSpan:
    """An evalyn span converted to OpenTelemetry-compatible format."""

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    operation_name: str = ""
    start_time_unix_ns: int = 0
    duration_ns: int = 0
    service_name: str = "evalyn"
    tags: Dict[str, str] = field(default_factory=dict)
    status: str = "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time_unix_ns": self.start_time_unix_ns,
            "duration_ns": self.duration_ns,
            "service_name": self.service_name,
            "tags": dict(self.tags),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OTelSpan:
        return cls(
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            parent_span_id=data.get("parent_span_id", ""),
            operation_name=data.get("operation_name", ""),
            start_time_unix_ns=data.get("start_time_unix_ns", 0),
            duration_ns=data.get("duration_ns", 0),
            service_name=data.get("service_name", "evalyn"),
            tags=data.get("tags", {}),
            status=data.get("status", "ok"),
        )


@dataclass
class ExportConfig:
    """Configuration for exporting spans to an OTel backend."""

    format: str = "otlp"  # "otlp", "jaeger", "zipkin"
    endpoint: str = ""
    service_name: str = "evalyn"
    headers: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "endpoint": self.endpoint,
            "service_name": self.service_name,
            "headers": dict(self.headers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExportConfig:
        return cls(
            format=data.get("format", "otlp"),
            endpoint=data.get("endpoint", ""),
            service_name=data.get("service_name", "evalyn"),
            headers=data.get("headers", {}),
        )


@dataclass
class ExportResult:
    """Result of an export operation."""

    exported_count: int = 0
    failed_count: int = 0
    format: str = ""
    endpoint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exported_count": self.exported_count,
            "failed_count": self.failed_count,
            "format": self.format,
            "endpoint": self.endpoint,
        }

    def format_text(self) -> str:
        lines = [
            f"Exported: {self.exported_count}",
            f"Failed: {self.failed_count}",
            f"Format: {self.format}",
        ]
        if self.endpoint:
            lines.append(f"Endpoint: {self.endpoint}")
        return "\n".join(lines)


def _datetime_to_unix_ns(dt: Any) -> int:
    """Convert a datetime to unix nanoseconds. Returns 0 if dt is None."""
    if dt is None:
        return 0
    return int(dt.timestamp() * 1_000_000_000)


def convert_span_to_otel(span: Span, service_name: str = "evalyn") -> OTelSpan:
    """Convert an evalyn Span to OTel format."""
    start_ns = _datetime_to_unix_ns(span.start_time)
    end_ns = _datetime_to_unix_ns(span.end_time)
    duration_ns = end_ns - start_ns if end_ns and start_ns else 0

    tags: Dict[str, str] = {}
    tags["span_type"] = span.span_type
    for key, value in span.attributes.items():
        tags[str(key)] = str(value)

    return OTelSpan(
        trace_id=span.id,
        span_id=span.id,
        parent_span_id=span.parent_id or "",
        operation_name=span.name,
        start_time_unix_ns=start_ns,
        duration_ns=duration_ns,
        service_name=service_name,
        tags=tags,
        status=span.status,
    )


def convert_spans_to_otel(
    spans: List[Span], service_name: str = "evalyn"
) -> List[OTelSpan]:
    """Batch convert evalyn Spans to OTel format."""
    return [convert_span_to_otel(s, service_name) for s in spans]


def format_as_otlp_json(otel_spans: List[OTelSpan]) -> str:
    """Format as OTLP JSON (resource spans format)."""
    if not otel_spans:
        return json.dumps({"resourceSpans": []}, indent=2)

    # Group spans by service_name
    by_service: Dict[str, List[OTelSpan]] = {}
    for s in otel_spans:
        by_service.setdefault(s.service_name, []).append(s)

    resource_spans = []
    for svc, spans in by_service.items():
        scope_spans = []
        for s in spans:
            scope_spans.append(
                {
                    "traceId": s.trace_id,
                    "spanId": s.span_id,
                    "parentSpanId": s.parent_span_id,
                    "name": s.operation_name,
                    "startTimeUnixNano": s.start_time_unix_ns,
                    "endTimeUnixNano": s.start_time_unix_ns + s.duration_ns,
                    "status": {"code": 1 if s.status == "ok" else 2},
                    "attributes": [
                        {"key": k, "value": {"stringValue": v}}
                        for k, v in s.tags.items()
                    ],
                }
            )
        resource_spans.append(
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": svc},
                        }
                    ]
                },
                "scopeSpans": [{"spans": scope_spans}],
            }
        )

    return json.dumps({"resourceSpans": resource_spans}, indent=2)


def format_as_jaeger_json(otel_spans: List[OTelSpan]) -> str:
    """Format as Jaeger-compatible JSON."""
    if not otel_spans:
        return json.dumps({"data": []}, indent=2)

    # Group by service
    by_service: Dict[str, List[OTelSpan]] = {}
    for s in otel_spans:
        by_service.setdefault(s.service_name, []).append(s)

    data = []
    for svc, spans in by_service.items():
        jaeger_spans = []
        for s in spans:
            jaeger_spans.append(
                {
                    "traceID": s.trace_id,
                    "spanID": s.span_id,
                    "operationName": s.operation_name,
                    "references": (
                        [
                            {
                                "refType": "CHILD_OF",
                                "traceID": s.trace_id,
                                "spanID": s.parent_span_id,
                            }
                        ]
                        if s.parent_span_id
                        else []
                    ),
                    "startTime": s.start_time_unix_ns // 1000,  # microseconds
                    "duration": s.duration_ns // 1000,
                    "tags": [
                        {"key": k, "type": "string", "value": v}
                        for k, v in s.tags.items()
                    ],
                }
            )
        data.append(
            {
                "traceID": spans[0].trace_id if spans else "",
                "spans": jaeger_spans,
                "processes": {
                    "p1": {
                        "serviceName": svc,
                        "tags": [],
                    }
                },
            }
        )

    return json.dumps({"data": data}, indent=2)


def format_as_zipkin_json(otel_spans: List[OTelSpan]) -> str:
    """Format as Zipkin JSON array."""
    if not otel_spans:
        return json.dumps([], indent=2)

    zipkin_spans = []
    for s in otel_spans:
        span_obj: Dict[str, Any] = {
            "traceId": s.trace_id,
            "id": s.span_id,
            "name": s.operation_name,
            "timestamp": s.start_time_unix_ns // 1000,  # microseconds
            "duration": s.duration_ns // 1000,
            "localEndpoint": {"serviceName": s.service_name},
            "tags": dict(s.tags),
        }
        if s.parent_span_id:
            span_obj["parentId"] = s.parent_span_id
        zipkin_spans.append(span_obj)

    return json.dumps(zipkin_spans, indent=2)


def export_to_file(
    otel_spans: List[OTelSpan], file_path: str, format: str = "otlp"
) -> ExportResult:
    """Write spans to file in specified format."""
    formatters = {
        "otlp": format_as_otlp_json,
        "jaeger": format_as_jaeger_json,
        "zipkin": format_as_zipkin_json,
    }

    formatter = formatters.get(format)
    if formatter is None:
        return ExportResult(
            exported_count=0,
            failed_count=len(otel_spans),
            format=format,
            endpoint=file_path,
        )

    content = formatter(otel_spans)
    with open(file_path, "w") as f:
        f.write(content)

    return ExportResult(
        exported_count=len(otel_spans),
        failed_count=0,
        format=format,
        endpoint=file_path,
    )
