from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceContext:
    """W3C Trace Context for distributed trace propagation."""

    trace_id: str
    span_id: str
    sampled: bool = True
    trace_state: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "sampled": self.sampled,
            "trace_state": dict(self.trace_state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceContext:
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            sampled=data.get("sampled", True),
            trace_state=data.get("trace_state", {}),
        )


@dataclass
class PropagationResult:
    """Result of injecting or extracting trace context."""

    injected: bool
    extracted: bool
    context: TraceContext | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "injected": self.injected,
            "extracted": self.extracted,
            "context": self.context.as_dict() if self.context else None,
        }


# Regex for validating traceparent: version-traceid-spanid-flags
_TRACEPARENT_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def generate_trace_id() -> str:
    """Generate a random 32-char hex string (16 bytes)."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a random 16-char hex string (8 bytes)."""
    return secrets.token_hex(8)


def create_traceparent(ctx: TraceContext) -> str:
    """Format W3C traceparent header.

    Format: "00-{trace_id}-{span_id}-{flags}"
    flags = "01" if sampled else "00"
    """
    flags = "01" if ctx.sampled else "00"
    return f"00-{ctx.trace_id}-{ctx.span_id}-{flags}"


def parse_traceparent(header: str) -> TraceContext | None:
    """Parse a W3C traceparent header. Return None if malformed."""
    match = _TRACEPARENT_RE.match(header.strip())
    if not match:
        return None
    _version, trace_id, span_id, flags = match.groups()
    sampled = flags[-1] == "1"
    return TraceContext(trace_id=trace_id, span_id=span_id, sampled=sampled)


def create_tracestate(ctx: TraceContext) -> str:
    """Format tracestate header: 'key1=value1,key2=value2'."""
    if not ctx.trace_state:
        return ""
    return ",".join(f"{k}={v}" for k, v in ctx.trace_state.items())


def parse_tracestate(header: str) -> dict[str, str]:
    """Parse tracestate header into a dict."""
    result: dict[str, str] = {}
    if not header or not header.strip():
        return result
    for pair in header.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def inject_headers(ctx: TraceContext, headers: dict[str, str]) -> dict[str, str]:
    """Add traceparent and tracestate to a headers dict. Return new dict."""
    out = dict(headers)
    out["traceparent"] = create_traceparent(ctx)
    state = create_tracestate(ctx)
    if state:
        out["tracestate"] = state
    return out


def extract_context(headers: dict[str, str]) -> PropagationResult:
    """Extract TraceContext from headers.

    Looks for 'traceparent' (and optionally 'tracestate') in the headers.
    Returns a PropagationResult with extracted=True if successful.
    """
    traceparent = headers.get("traceparent", "")
    if not traceparent:
        return PropagationResult(injected=False, extracted=False, context=None)

    ctx = parse_traceparent(traceparent)
    if ctx is None:
        return PropagationResult(injected=False, extracted=False, context=None)

    tracestate = headers.get("tracestate", "")
    if tracestate:
        ctx.trace_state = parse_tracestate(tracestate)

    return PropagationResult(injected=False, extracted=True, context=ctx)
