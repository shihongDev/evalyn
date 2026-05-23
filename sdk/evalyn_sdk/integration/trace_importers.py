"""
Trace importers: convert external provider logs to evalyn FunctionCall + Span.

Supports:
- OpenAI chat completion JSONL logs (request/response/timestamp/duration_ms shape)
- Anthropic messages JSONL logs (request/response/timestamp/duration_ms shape)
- Generic JSONL (input/output/model with optional token counts)

Each input record yields a single FunctionCall containing one child llm_call Span.
Records with missing required fields are skipped with a warning rather than
aborting the whole import, so partial logs can still be ingested.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..models import FunctionCall, Span, _parse_datetime, now_utc
from ..trace.instrumentation.providers._shared import calculate_cost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield (line_number, parsed_record) tuples from a JSONL file.

    Skips empty lines. Warns and skips lines that fail JSON parsing.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"Skipping malformed JSON at {p}:{lineno}: {exc}",
                    stacklevel=2,
                )
                continue


def _parse_ts_or_now(raw: str | None):
    """Parse an ISO timestamp; fall back to now() if missing or invalid."""
    if not raw:
        return now_utc()
    try:
        dt = _parse_datetime(raw)
        return dt or now_utc()
    except (ValueError, TypeError):
        return now_utc()


def _build_call_with_span(
    *,
    function_name: str,
    project: str | None,
    model: str,
    input_payload: Any,
    output_payload: Any,
    input_tokens: int,
    output_tokens: int,
    started_at,
    duration_ms: float,
    provider: str,
    extra_metadata: dict[str, Any] | None = None,
    extra_span_attrs: dict[str, Any] | None = None,
) -> FunctionCall:
    """Assemble a FunctionCall with a single child llm_call Span.

    Mirrors what `log_llm_call` would produce for a live trace so imported
    records show up identically in `evalyn list-calls`, `show-trace`, etc.
    """
    ended_at = started_at + timedelta(milliseconds=duration_ms)
    cost = calculate_cost(model, input_tokens, output_tokens)

    metadata: dict[str, Any] = {"imported_from": provider}
    if project:
        metadata["project_name"] = project
    if extra_metadata:
        metadata.update(extra_metadata)

    call = FunctionCall.new(
        function_name=function_name,
        inputs={"input": input_payload, "model": model},
        session_id=None,
        metadata=metadata,
    )
    call.started_at = started_at
    call.ended_at = ended_at
    call.duration_ms = duration_ms
    call.output = output_payload

    span = Span.new(
        name=f"{provider}:{model}",
        span_type="llm_call",
        parent_id=None,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
    span.start_time = started_at
    span.end_time = ended_at
    span.status = "ok"
    if extra_span_attrs:
        span.attributes.update(extra_span_attrs)

    call.add_span(span)
    return call


# ---------------------------------------------------------------------------
# OpenAI chat completions JSONL
# ---------------------------------------------------------------------------


def import_openai_jsonl(
    path: str | Path,
    *,
    project: str | None = None,
    function_name: str = "openai.chat.completions",
) -> list[FunctionCall]:
    """Parse an OpenAI chat completions JSONL log into FunctionCalls.

    Expected record shape:
        {
          "request":  {"model": "...", "messages": [...]},
          "response": {"choices": [{"message": {"content": "..."}}],
                       "usage": {"prompt_tokens": N, "completion_tokens": N}},
          "timestamp": "...",
          "duration_ms": N
        }
    """
    calls: list[FunctionCall] = []
    p = Path(path)

    for lineno, record in _iter_jsonl(p):
        try:
            request = record.get("request") or {}
            response = record.get("response") or {}

            model = request.get("model")
            messages = request.get("messages")
            if not model or not messages:
                warnings.warn(
                    f"Skipping {p}:{lineno}: missing request.model or request.messages",
                    stacklevel=2,
                )
                continue

            choices = response.get("choices") or []
            content = ""
            finish_reason = None
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content") or ""
                finish_reason = first.get("finish_reason")

            usage = response.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)

            started_at = _parse_ts_or_now(record.get("timestamp"))
            duration_ms = float(record.get("duration_ms") or 0)

            extra_span_attrs: dict[str, Any] = {
                "request": {"messages": str(messages)[:500]},
                "response": {"content": str(content)[:500]},
            }
            if finish_reason is not None:
                extra_span_attrs["response"]["finish_reason"] = finish_reason

            call = _build_call_with_span(
                function_name=function_name,
                project=project,
                model=model,
                input_payload=messages,
                output_payload=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                started_at=started_at,
                duration_ms=duration_ms,
                provider="openai",
                extra_span_attrs=extra_span_attrs,
            )
            calls.append(call)
        except Exception as exc:  # noqa: BLE001 - defensive: never abort import
            warnings.warn(
                f"Skipping {p}:{lineno} due to error: {exc}",
                stacklevel=2,
            )
            continue

    return calls


# ---------------------------------------------------------------------------
# Anthropic messages JSONL
# ---------------------------------------------------------------------------


def import_anthropic_jsonl(
    path: str | Path,
    *,
    project: str | None = None,
    function_name: str = "anthropic.messages",
) -> list[FunctionCall]:
    """Parse an Anthropic messages JSONL log into FunctionCalls.

    Expected record shape:
        {
          "request":  {"model": "...", "messages": [...], "max_tokens": N},
          "response": {"content": [{"type": "text", "text": "..."}],
                       "usage": {"input_tokens": N, "output_tokens": N}},
          "timestamp": "...",
          "duration_ms": N
        }
    """
    calls: list[FunctionCall] = []
    p = Path(path)

    for lineno, record in _iter_jsonl(p):
        try:
            request = record.get("request") or {}
            response = record.get("response") or {}

            model = request.get("model")
            messages = request.get("messages")
            if not model or not messages:
                warnings.warn(
                    f"Skipping {p}:{lineno}: missing request.model or request.messages",
                    stacklevel=2,
                )
                continue

            # Concatenate all text blocks; ignore tool_use / image blocks for now.
            content_blocks = response.get("content") or []
            texts: list[str] = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_val = block.get("text") or ""
                    if text_val:
                        texts.append(text_val)
            content = " ".join(texts)

            usage = response.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)

            started_at = _parse_ts_or_now(record.get("timestamp"))
            duration_ms = float(record.get("duration_ms") or 0)

            extra_span_attrs: dict[str, Any] = {
                "request": {"messages": str(messages)[:500]},
                "response": {"content": str(content)[:1000]},
            }
            if cache_creation:
                extra_span_attrs["cache_creation_input_tokens"] = cache_creation
            if cache_read:
                extra_span_attrs["cache_read_input_tokens"] = cache_read

            extra_metadata: dict[str, Any] = {}
            if "max_tokens" in request:
                extra_metadata["max_tokens"] = request["max_tokens"]

            call = _build_call_with_span(
                function_name=function_name,
                project=project,
                model=model,
                input_payload=messages,
                output_payload=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                started_at=started_at,
                duration_ms=duration_ms,
                provider="anthropic",
                extra_metadata=extra_metadata or None,
                extra_span_attrs=extra_span_attrs,
            )
            calls.append(call)
        except Exception as exc:  # noqa: BLE001 - defensive: never abort import
            warnings.warn(
                f"Skipping {p}:{lineno} due to error: {exc}",
                stacklevel=2,
            )
            continue

    return calls


# ---------------------------------------------------------------------------
# Generic JSONL
# ---------------------------------------------------------------------------


def import_generic_jsonl(
    path: str | Path,
    *,
    project: str | None = None,
    function_name: str = "generic.llm_call",
) -> list[FunctionCall]:
    """Parse a generic JSONL log into FunctionCalls.

    Expected record shape (most fields optional, input/output required):
        {
          "input": "...",
          "output": "...",
          "model": "...",
          "input_tokens": N,
          "output_tokens": N,
          "metadata": {...},
          "timestamp": "...",
          "duration_ms": N
        }
    """
    calls: list[FunctionCall] = []
    p = Path(path)

    for lineno, record in _iter_jsonl(p):
        try:
            input_val = record.get("input")
            output_val = record.get("output")
            if input_val is None or output_val is None:
                warnings.warn(
                    f"Skipping {p}:{lineno}: missing input or output",
                    stacklevel=2,
                )
                continue

            model = record.get("model") or "unknown"
            input_tokens = int(record.get("input_tokens") or 0)
            output_tokens = int(record.get("output_tokens") or 0)

            started_at = _parse_ts_or_now(record.get("timestamp"))
            duration_ms = float(record.get("duration_ms") or 0)

            user_metadata = record.get("metadata") or {}
            extra_metadata: dict[str, Any] = {}
            if isinstance(user_metadata, dict):
                extra_metadata.update(user_metadata)

            extra_span_attrs = {
                "request": {"input": str(input_val)[:500]},
                "response": {"output": str(output_val)[:1000]},
            }

            call = _build_call_with_span(
                function_name=function_name,
                project=project,
                model=model,
                input_payload=input_val,
                output_payload=output_val,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                started_at=started_at,
                duration_ms=duration_ms,
                provider="generic",
                extra_metadata=extra_metadata or None,
                extra_span_attrs=extra_span_attrs,
            )
            calls.append(call)
        except Exception as exc:  # noqa: BLE001 - defensive: never abort import
            warnings.warn(
                f"Skipping {p}:{lineno} due to error: {exc}",
                stacklevel=2,
            )
            continue

    return calls


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


FORMAT_IMPORTERS = {
    "openai": import_openai_jsonl,
    "anthropic": import_anthropic_jsonl,
    "generic": import_generic_jsonl,
}


def import_traces(
    path: str | Path,
    *,
    fmt: str,
    project: str | None = None,
    function_name: str | None = None,
) -> list[FunctionCall]:
    """Dispatch to the correct importer based on `fmt`."""
    importer = FORMAT_IMPORTERS.get(fmt)
    if importer is None:
        raise ValueError(
            f"Unsupported format: {fmt!r}. Expected one of: {sorted(FORMAT_IMPORTERS)}"
        )
    kwargs: dict[str, Any] = {"project": project}
    if function_name:
        kwargs["function_name"] = function_name
    return importer(path, **kwargs)


__all__ = [
    "FORMAT_IMPORTERS",
    "import_anthropic_jsonl",
    "import_generic_jsonl",
    "import_openai_jsonl",
    "import_traces",
]
