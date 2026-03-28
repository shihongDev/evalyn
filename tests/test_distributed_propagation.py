from __future__ import annotations

from evalyn_sdk.trace.distributed_propagation import (
    TraceContext,
    PropagationResult,
    generate_trace_id,
    generate_span_id,
    create_traceparent,
    parse_traceparent,
    create_tracestate,
    parse_tracestate,
    inject_headers,
    extract_context,
)


# -- generate_trace_id --

def test_generate_trace_id_length():
    tid = generate_trace_id()
    assert len(tid) == 32


def test_generate_trace_id_hex():
    tid = generate_trace_id()
    int(tid, 16)  # raises ValueError if not valid hex


def test_generate_trace_id_unique():
    ids = {generate_trace_id() for _ in range(50)}
    assert len(ids) == 50


# -- generate_span_id --

def test_generate_span_id_length():
    sid = generate_span_id()
    assert len(sid) == 16


def test_generate_span_id_hex():
    sid = generate_span_id()
    int(sid, 16)


def test_generate_span_id_unique():
    ids = {generate_span_id() for _ in range(50)}
    assert len(ids) == 50


# -- create_traceparent --

def test_create_traceparent_sampled():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, sampled=True)
    header = create_traceparent(ctx)
    assert header == f"00-{'a' * 32}-{'b' * 16}-01"


def test_create_traceparent_not_sampled():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, sampled=False)
    header = create_traceparent(ctx)
    assert header == f"00-{'a' * 32}-{'b' * 16}-00"


# -- parse_traceparent --

def test_parse_traceparent_valid():
    header = f"00-{'a' * 32}-{'b' * 16}-01"
    ctx = parse_traceparent(header)
    assert ctx is not None
    assert ctx.trace_id == "a" * 32
    assert ctx.span_id == "b" * 16
    assert ctx.sampled is True


def test_parse_traceparent_not_sampled():
    header = f"00-{'c' * 32}-{'d' * 16}-00"
    ctx = parse_traceparent(header)
    assert ctx is not None
    assert ctx.sampled is False


def test_parse_traceparent_invalid_empty():
    assert parse_traceparent("") is None


def test_parse_traceparent_invalid_short():
    assert parse_traceparent("00-abc-def-01") is None


def test_parse_traceparent_invalid_chars():
    bad = f"00-{'g' * 32}-{'h' * 16}-01"
    assert parse_traceparent(bad) is None


def test_parse_traceparent_roundtrip():
    ctx = TraceContext(trace_id=generate_trace_id(), span_id=generate_span_id())
    header = create_traceparent(ctx)
    parsed = parse_traceparent(header)
    assert parsed is not None
    assert parsed.trace_id == ctx.trace_id
    assert parsed.span_id == ctx.span_id
    assert parsed.sampled == ctx.sampled


# -- create_tracestate / parse_tracestate --

def test_create_tracestate_empty():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    assert create_tracestate(ctx) == ""


def test_create_tracestate_single():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, trace_state={"vendor": "val"})
    assert create_tracestate(ctx) == "vendor=val"


def test_create_tracestate_multiple():
    ctx = TraceContext(
        trace_id="a" * 32,
        span_id="b" * 16,
        trace_state={"k1": "v1", "k2": "v2"},
    )
    state = create_tracestate(ctx)
    parsed = parse_tracestate(state)
    assert parsed == {"k1": "v1", "k2": "v2"}


def test_parse_tracestate_empty():
    assert parse_tracestate("") == {}


def test_parse_tracestate_whitespace():
    assert parse_tracestate("   ") == {}


def test_parse_tracestate_valid():
    result = parse_tracestate("foo=bar,baz=qux")
    assert result == {"foo": "bar", "baz": "qux"}


def test_parse_tracestate_roundtrip():
    original = {"vendor1": "abc", "vendor2": "xyz"}
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, trace_state=original)
    header = create_tracestate(ctx)
    parsed = parse_tracestate(header)
    assert parsed == original


# -- inject_headers --

def test_inject_headers_adds_traceparent():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, sampled=True)
    result = inject_headers(ctx, {})
    assert "traceparent" in result
    assert result["traceparent"] == f"00-{'a' * 32}-{'b' * 16}-01"


def test_inject_headers_adds_tracestate():
    ctx = TraceContext(
        trace_id="a" * 32, span_id="b" * 16, trace_state={"vendor": "val"},
    )
    result = inject_headers(ctx, {})
    assert "tracestate" in result
    assert result["tracestate"] == "vendor=val"


def test_inject_headers_no_tracestate_when_empty():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    result = inject_headers(ctx, {})
    assert "tracestate" not in result


def test_inject_headers_preserves_existing():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    result = inject_headers(ctx, {"Authorization": "Bearer token"})
    assert result["Authorization"] == "Bearer token"
    assert "traceparent" in result


def test_inject_headers_returns_new_dict():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    original = {"X-Custom": "value"}
    result = inject_headers(ctx, original)
    assert result is not original


# -- extract_context --

def test_extract_context_valid():
    headers = {"traceparent": f"00-{'a' * 32}-{'b' * 16}-01"}
    result = extract_context(headers)
    assert result.extracted is True
    assert result.context is not None
    assert result.context.trace_id == "a" * 32


def test_extract_context_with_tracestate():
    headers = {
        "traceparent": f"00-{'a' * 32}-{'b' * 16}-01",
        "tracestate": "vendor=val",
    }
    result = extract_context(headers)
    assert result.extracted is True
    assert result.context is not None
    assert result.context.trace_state == {"vendor": "val"}


def test_extract_context_empty_headers():
    result = extract_context({})
    assert result.extracted is False
    assert result.context is None


def test_extract_context_invalid_traceparent():
    result = extract_context({"traceparent": "garbage"})
    assert result.extracted is False
    assert result.context is None


# -- TraceContext as_dict / from_dict --

def test_trace_context_as_dict():
    ctx = TraceContext(
        trace_id="a" * 32, span_id="b" * 16, sampled=False, trace_state={"k": "v"},
    )
    d = ctx.as_dict()
    assert d["trace_id"] == "a" * 32
    assert d["span_id"] == "b" * 16
    assert d["sampled"] is False
    assert d["trace_state"] == {"k": "v"}


def test_trace_context_from_dict():
    d = {"trace_id": "a" * 32, "span_id": "b" * 16, "sampled": True, "trace_state": {"x": "y"}}
    ctx = TraceContext.from_dict(d)
    assert ctx.trace_id == "a" * 32
    assert ctx.trace_state == {"x": "y"}


def test_trace_context_roundtrip():
    ctx = TraceContext(
        trace_id=generate_trace_id(),
        span_id=generate_span_id(),
        sampled=False,
        trace_state={"evalyn": "v1"},
    )
    restored = TraceContext.from_dict(ctx.as_dict())
    assert restored.trace_id == ctx.trace_id
    assert restored.span_id == ctx.span_id
    assert restored.sampled == ctx.sampled
    assert restored.trace_state == ctx.trace_state


# -- PropagationResult --

def test_propagation_result_as_dict_no_context():
    pr = PropagationResult(injected=False, extracted=False)
    d = pr.as_dict()
    assert d["context"] is None


def test_propagation_result_as_dict_with_context():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    pr = PropagationResult(injected=True, extracted=True, context=ctx)
    d = pr.as_dict()
    assert d["injected"] is True
    assert d["context"]["trace_id"] == "a" * 32


# -- Sampled flag toggling --

def test_sampled_true_by_default():
    ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    assert ctx.sampled is True


def test_sampled_flag_in_traceparent():
    for sampled, expected_flag in [(True, "01"), (False, "00")]:
        ctx = TraceContext(trace_id="a" * 32, span_id="b" * 16, sampled=sampled)
        header = create_traceparent(ctx)
        assert header.endswith(f"-{expected_flag}")


# -- Full inject/extract roundtrip --

def test_full_roundtrip():
    ctx = TraceContext(
        trace_id=generate_trace_id(),
        span_id=generate_span_id(),
        sampled=True,
        trace_state={"evalyn": "test", "upstream": "abc"},
    )
    headers = inject_headers(ctx, {"Content-Type": "application/json"})
    result = extract_context(headers)
    assert result.extracted is True
    assert result.context is not None
    assert result.context.trace_id == ctx.trace_id
    assert result.context.span_id == ctx.span_id
    assert result.context.sampled == ctx.sampled
    assert result.context.trace_state == ctx.trace_state
