"""Replay captured LLM calls against a different model.

Extracts LLM call spans from a captured FunctionCall trace, re-runs the
prompts against a target model, and produces a side-by-side comparison of
the original vs replayed output (cost, latency, tokens, text).
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import FunctionCall, Span
from ..trace.instrumentation.providers._shared import calculate_cost

# Type alias for an LLM-call function (used to inject mocks in tests).
# Signature: (provider, model, messages) -> dict with keys:
#   text, input_tokens, output_tokens
LLMCaller = Callable[[str, str, list[dict[str, Any]]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplayableSpan:
    """An LLM call extracted from a trace that can be replayed."""

    span_id: str
    span_name: str
    original_provider: str
    original_model: str
    messages: list[dict[str, Any]]  # Parsed messages list
    original_input_tokens: int
    original_output_tokens: int
    original_cost_usd: float
    original_duration_ms: float
    original_output: str  # Text response from original call
    raw_request: Any = None  # Original request payload (debug)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayResult:
    """Result of replaying a single span against a new model."""

    span_id: str
    new_provider: str
    new_model: str
    new_output: str
    new_input_tokens: int
    new_output_tokens: int
    new_cost_usd: float
    new_duration_ms: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayPair:
    """A side-by-side pair of original + replayed span."""

    original: ReplayableSpan
    replayed: ReplayResult

    @property
    def cost_delta_usd(self) -> float:
        return self.replayed.new_cost_usd - self.original.original_cost_usd

    @property
    def latency_delta_ms(self) -> float:
        return self.replayed.new_duration_ms - self.original.original_duration_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original.as_dict(),
            "replayed": self.replayed.as_dict(),
            "cost_delta_usd": self.cost_delta_usd,
            "latency_delta_ms": self.latency_delta_ms,
        }


@dataclass
class ReplayComparison:
    """Full comparison of an original trace vs its replay."""

    call_id: str
    function_name: str
    target_provider: str
    target_model: str
    pairs: list[ReplayPair] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_original_cost_usd(self) -> float:
        return sum(p.original.original_cost_usd for p in self.pairs)

    @property
    def total_replayed_cost_usd(self) -> float:
        return sum(p.replayed.new_cost_usd for p in self.pairs)

    @property
    def total_cost_delta_usd(self) -> float:
        return self.total_replayed_cost_usd - self.total_original_cost_usd

    @property
    def total_original_latency_ms(self) -> float:
        return sum(p.original.original_duration_ms for p in self.pairs)

    @property
    def total_replayed_latency_ms(self) -> float:
        return sum(p.replayed.new_duration_ms for p in self.pairs)

    @property
    def total_latency_delta_ms(self) -> float:
        return self.total_replayed_latency_ms - self.total_original_latency_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "function_name": self.function_name,
            "target_provider": self.target_provider,
            "target_model": self.target_model,
            "pairs": [p.as_dict() for p in self.pairs],
            "skipped": list(self.skipped),
            "totals": {
                "original_cost_usd": self.total_original_cost_usd,
                "replayed_cost_usd": self.total_replayed_cost_usd,
                "cost_delta_usd": self.total_cost_delta_usd,
                "original_latency_ms": self.total_original_latency_ms,
                "replayed_latency_ms": self.total_replayed_latency_ms,
                "latency_delta_ms": self.total_latency_delta_ms,
            },
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _parse_messages_field(raw: Any) -> list[dict[str, Any]]:
    """Best-effort parse of the captured 'messages' field.

    Instrumentors store messages as `str(messages)[:500]` (often truncated
    Python repr of a list of dicts). We try, in order:
    1. Already-a-list -> return as-is
    2. JSON parse
    3. ast.literal_eval (handles single-quoted dicts; safe - data only)
    Returns [] on failure (caller treats this as "unusable").
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []

    # Try JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal parsing (handles `[{'role': 'user', ...}]` form).
    # ast.literal_eval is safe - it only parses literals, no code execution.
    # Only succeeds if the captured string was not truncated mid-structure.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict)]
    except (ValueError, SyntaxError):
        pass

    return []


def _extract_messages_from_span(span: Span) -> list[dict[str, Any]]:
    """Pull the messages list out of an llm_call span's attributes.request."""
    attrs = span.attributes or {}
    request = attrs.get("request")
    if isinstance(request, dict):
        return _parse_messages_field(request.get("messages"))
    return _parse_messages_field(request)


def _extract_output_text_from_span(span: Span) -> str:
    """Pull the text response from an llm_call span's attributes.response."""
    attrs = span.attributes or {}
    response = attrs.get("response")
    if isinstance(response, dict):
        content = response.get("content")
        if content is not None:
            return str(content)
    if response is not None:
        return str(response)
    # Fallbacks used by some providers
    for key in ("output", "output_preview", "text"):
        val = attrs.get(key)
        if val:
            return str(val)
    return ""


def extract_replayable_spans(call: FunctionCall) -> list[ReplayableSpan]:
    """Find all llm_call spans in the call and project them to ReplayableSpan.

    Spans without usable messages (parse failure / empty request) are skipped
    silently - callers should compare len(extract_replayable_spans(call)) vs
    the raw llm_call count to detect drops.
    """
    out: list[ReplayableSpan] = []
    for span in call.spans or []:
        if span.span_type != "llm_call":
            continue
        messages = _extract_messages_from_span(span)
        if not messages:
            continue

        attrs = span.attributes or {}
        provider = str(attrs.get("provider", "unknown"))
        model = str(attrs.get("model", "unknown"))
        input_tokens = int(attrs.get("input_tokens", 0) or 0)
        output_tokens = int(attrs.get("output_tokens", 0) or 0)
        cost_usd = float(attrs.get("cost_usd", 0.0) or 0.0)
        duration_ms = float(span.duration_ms or 0.0)

        out.append(
            ReplayableSpan(
                span_id=span.id,
                span_name=span.name,
                original_provider=provider,
                original_model=model,
                messages=messages,
                original_input_tokens=input_tokens,
                original_output_tokens=output_tokens,
                original_cost_usd=cost_usd,
                original_duration_ms=duration_ms,
                original_output=_extract_output_text_from_span(span),
                raw_request=attrs.get("request"),
            )
        )
    return out


def count_unreplayable_llm_spans(call: FunctionCall) -> int:
    """Count llm_call spans that were dropped due to unparseable messages."""
    total = sum(1 for s in (call.spans or []) if s.span_type == "llm_call")
    return total - len(extract_replayable_spans(call))


# ---------------------------------------------------------------------------
# Replay execution
# ---------------------------------------------------------------------------


def _messages_to_prompt(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Flatten a messages list into (user_prompt, system_instruction).

    Our LLM clients use generate_with_usage(prompt, system_instruction=...).
    We collapse all user/assistant turns into a single prompt and surface the
    first system message separately for prefix caching.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI-style content blocks
            text_chunks = []
            for block in content:
                if isinstance(block, dict):
                    text_chunks.append(str(block.get("text", block)))
                else:
                    text_chunks.append(str(block))
            content = "\n".join(text_chunks)
        content = str(content)
        if role == "system":
            system_parts.append(content)
        else:
            prefix = f"{role}: " if role != "user" else ""
            user_parts.append(f"{prefix}{content}")

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(user_parts) if user_parts else ""
    return prompt, system_instruction


def _default_llm_caller(
    provider: str, model: str, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Default replay backend - uses the same LLM clients judges use."""
    from ..utils.api_client import create_llm_client

    client = create_llm_client(provider=provider, model=model)
    prompt, system_instruction = _messages_to_prompt(messages)
    result = client.generate_with_usage(prompt, system_instruction=system_instruction)
    return {
        "text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def replay_span(
    span: ReplayableSpan,
    target_model: str,
    target_provider: str,
    *,
    caller: LLMCaller | None = None,
) -> ReplayResult:
    """Re-execute one captured llm_call against (target_provider, target_model).

    The optional *caller* lets tests inject a deterministic stub. Production
    callers pass None to get the real LLM client.
    """
    fn = caller or _default_llm_caller
    start = time.time()
    try:
        result = fn(target_provider, target_model, span.messages)
    except Exception as exc:  # noqa: BLE001 - surface any provider failure
        duration_ms = (time.time() - start) * 1000
        return ReplayResult(
            span_id=span.span_id,
            new_provider=target_provider,
            new_model=target_model,
            new_output="",
            new_input_tokens=0,
            new_output_tokens=0,
            new_cost_usd=0.0,
            new_duration_ms=duration_ms,
            error=str(exc),
        )
    duration_ms = (time.time() - start) * 1000

    input_tokens = int(result.get("input_tokens", 0) or 0)
    output_tokens = int(result.get("output_tokens", 0) or 0)
    new_cost = calculate_cost(target_model, input_tokens, output_tokens)

    return ReplayResult(
        span_id=span.span_id,
        new_provider=target_provider,
        new_model=target_model,
        new_output=str(result.get("text", "")),
        new_input_tokens=input_tokens,
        new_output_tokens=output_tokens,
        new_cost_usd=new_cost,
        new_duration_ms=duration_ms,
        error=None,
    )


def build_replay_comparison(
    call: FunctionCall,
    target_provider: str,
    target_model: str,
    *,
    caller: LLMCaller | None = None,
) -> ReplayComparison:
    """End-to-end: extract spans, replay each, return a ReplayComparison."""
    replayable_ids: set[str] = set()
    pairs: list[ReplayPair] = []
    for replayable in extract_replayable_spans(call):
        replayable_ids.add(replayable.span_id)
        result = replay_span(replayable, target_model, target_provider, caller=caller)
        pairs.append(ReplayPair(original=replayable, replayed=result))

    skipped = [
        {"span_id": span.id, "span_name": span.name, "reason": "no_usable_messages"}
        for span in (call.spans or [])
        if span.span_type == "llm_call" and span.id not in replayable_ids
    ]

    return ReplayComparison(
        call_id=call.id,
        function_name=call.function_name,
        target_provider=target_provider,
        target_model=target_model,
        pairs=pairs,
        skipped=skipped,
    )


__all__ = [
    "LLMCaller",
    "ReplayableSpan",
    "ReplayResult",
    "ReplayPair",
    "ReplayComparison",
    "extract_replayable_spans",
    "count_unreplayable_llm_spans",
    "replay_span",
    "build_replay_comparison",
]
