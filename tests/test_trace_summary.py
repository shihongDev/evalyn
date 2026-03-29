"""Tests for trace summary generation module."""
from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.trace_summary import (
    TraceSummaryConfig,
    TraceSummary,
    build_summary_prompt,
    extract_trace_highlights,
    build_heuristic_summary,
    parse_llm_summary_response,
    format_trace_summary,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_span(
    name: str = "test_span",
    span_type: str = "custom",
    status: str = "ok",
    duration_ms: float = 100.0,
    attributes: dict | None = None,
) -> dict:
    return {
        "id": f"span-{name}",
        "name": name,
        "span_type": span_type,
        "status": status,
        "duration_ms": duration_ms,
        "attributes": attributes or {},
    }


def _make_trace(spans: list[dict] | None = None, trace_id: str = "trace-001") -> dict:
    return {
        "trace_id": trace_id,
        "spans": spans or [],
    }


# ---------------------------------------------------------------------------
# TraceSummaryConfig
# ---------------------------------------------------------------------------


class TestTraceSummaryConfig:
    def test_defaults(self):
        cfg = TraceSummaryConfig()
        assert cfg.max_spans == 20
        assert cfg.include_tool_calls is True
        assert cfg.include_latencies is True
        assert cfg.summary_style == "brief"

    def test_custom_values(self):
        cfg = TraceSummaryConfig(max_spans=50, include_tool_calls=False, summary_style="detailed")
        assert cfg.max_spans == 50
        assert cfg.include_tool_calls is False
        assert cfg.summary_style == "detailed"

    def test_as_dict(self):
        cfg = TraceSummaryConfig(max_spans=10, summary_style="technical")
        d = cfg.as_dict()
        assert d["max_spans"] == 10
        assert d["summary_style"] == "technical"
        assert d["include_tool_calls"] is True

    def test_from_dict(self):
        d = {"max_spans": 30, "summary_style": "detailed", "include_latencies": False}
        cfg = TraceSummaryConfig.from_dict(d)
        assert cfg.max_spans == 30
        assert cfg.summary_style == "detailed"
        assert cfg.include_latencies is False

    def test_from_dict_defaults(self):
        cfg = TraceSummaryConfig.from_dict({})
        assert cfg.max_spans == 20
        assert cfg.include_tool_calls is True

    def test_roundtrip(self):
        original = TraceSummaryConfig(max_spans=15, include_tool_calls=False, summary_style="technical")
        restored = TraceSummaryConfig.from_dict(original.as_dict())
        assert restored.max_spans == original.max_spans
        assert restored.include_tool_calls == original.include_tool_calls
        assert restored.summary_style == original.summary_style


# ---------------------------------------------------------------------------
# TraceSummary
# ---------------------------------------------------------------------------


class TestTraceSummary:
    def test_defaults(self):
        ts = TraceSummary(trace_id="t1", summary_text="hello")
        assert ts.trace_id == "t1"
        assert ts.key_findings == []
        assert ts.span_count == 0
        assert ts.total_latency_ms == 0.0
        assert ts.generated_at == ""

    def test_as_dict(self):
        ts = TraceSummary(
            trace_id="t1",
            summary_text="summary",
            key_findings=["finding1"],
            span_count=5,
            total_latency_ms=1234.5,
            generated_at="2026-01-01T00:00:00",
        )
        d = ts.as_dict()
        assert d["trace_id"] == "t1"
        assert d["key_findings"] == ["finding1"]
        assert d["span_count"] == 5

    def test_from_dict(self):
        d = {
            "trace_id": "t2",
            "summary_text": "text",
            "key_findings": ["a", "b"],
            "span_count": 3,
            "total_latency_ms": 500.0,
        }
        ts = TraceSummary.from_dict(d)
        assert ts.trace_id == "t2"
        assert ts.key_findings == ["a", "b"]
        assert ts.total_latency_ms == 500.0

    def test_from_dict_defaults(self):
        ts = TraceSummary.from_dict({})
        assert ts.trace_id == ""
        assert ts.summary_text == ""
        assert ts.span_count == 0

    def test_roundtrip(self):
        original = TraceSummary(
            trace_id="t3",
            summary_text="s",
            key_findings=["x"],
            span_count=2,
            total_latency_ms=99.9,
            generated_at="ts",
        )
        restored = TraceSummary.from_dict(original.as_dict())
        assert restored.trace_id == original.trace_id
        assert restored.key_findings == original.key_findings


# ---------------------------------------------------------------------------
# build_summary_prompt
# ---------------------------------------------------------------------------


class TestBuildSummaryPrompt:
    def test_basic_prompt(self):
        trace = _make_trace([_make_span("s1", "llm_call"), _make_span("s2", "tool_call")])
        prompt = build_summary_prompt(trace, TraceSummaryConfig())
        assert "trace-001" in prompt
        assert "2 spans" in prompt
        assert "llm_call" in prompt
        assert "tool_call" in prompt

    def test_includes_tool_calls(self):
        trace = _make_trace([_make_span("search_tool", "tool_call")])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(include_tool_calls=True))
        assert "search_tool" in prompt

    def test_excludes_tool_calls_when_disabled(self):
        trace = _make_trace([_make_span("search_tool", "tool_call")])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(include_tool_calls=False))
        assert "Tool calls" not in prompt

    def test_includes_latency(self):
        trace = _make_trace([_make_span("s1", duration_ms=500.0)])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(include_latencies=True))
        assert "500ms" in prompt

    def test_excludes_latency_when_disabled(self):
        trace = _make_trace([_make_span("s1", duration_ms=500.0)])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(include_latencies=False))
        assert "Total latency" not in prompt

    def test_includes_errors(self):
        trace = _make_trace([
            _make_span("bad_span", status="error", attributes={"error": "timeout"}),
        ])
        prompt = build_summary_prompt(trace, TraceSummaryConfig())
        assert "Errors" in prompt
        assert "timeout" in prompt

    def test_respects_max_spans(self):
        spans = [_make_span(f"s{i}") for i in range(50)]
        trace = _make_trace(spans)
        prompt = build_summary_prompt(trace, TraceSummaryConfig(max_spans=5))
        assert "5 spans" in prompt

    def test_brief_style_instruction(self):
        trace = _make_trace([_make_span()])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(summary_style="brief"))
        assert "2-3 sentence" in prompt

    def test_detailed_style_instruction(self):
        trace = _make_trace([_make_span()])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(summary_style="detailed"))
        assert "detailed paragraph" in prompt

    def test_technical_style_instruction(self):
        trace = _make_trace([_make_span()])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(summary_style="technical"))
        assert "technical summary" in prompt

    def test_empty_trace(self):
        trace = _make_trace([])
        prompt = build_summary_prompt(trace, TraceSummaryConfig())
        assert "0 spans" in prompt

    def test_slow_spans_mentioned(self):
        trace = _make_trace([_make_span("slow_one", duration_ms=2000.0)])
        prompt = build_summary_prompt(trace, TraceSummaryConfig())
        assert "Slow spans" in prompt
        assert "slow_one" in prompt

    def test_no_tool_calls_says_none(self):
        trace = _make_trace([_make_span("s1", "llm_call")])
        prompt = build_summary_prompt(trace, TraceSummaryConfig(include_tool_calls=True))
        assert "Tool calls: none" in prompt

    def test_bullet_point_instruction(self):
        trace = _make_trace([_make_span()])
        prompt = build_summary_prompt(trace, TraceSummaryConfig())
        assert "bullet points" in prompt.lower() or "- " in prompt


# ---------------------------------------------------------------------------
# extract_trace_highlights
# ---------------------------------------------------------------------------


class TestExtractTraceHighlights:
    def test_detects_errors(self):
        trace = _make_trace([
            _make_span("fail_span", status="error", attributes={"error": "connection refused"}),
        ])
        highlights = extract_trace_highlights(trace)
        assert any("Error" in h for h in highlights)
        assert any("connection refused" in h for h in highlights)

    def test_detects_slow_spans(self):
        trace = _make_trace([_make_span("slow_op", duration_ms=2500.0)])
        highlights = extract_trace_highlights(trace)
        assert any("Slow span" in h for h in highlights)
        assert any("slow_op" in h for h in highlights)

    def test_detects_tool_failures(self):
        trace = _make_trace([
            _make_span("broken_tool", span_type="tool_call", status="error"),
        ])
        highlights = extract_trace_highlights(trace)
        assert any("Tool failure" in h for h in highlights)

    def test_detects_repeated_spans(self):
        trace = _make_trace([
            _make_span("retry_op"),
            _make_span("retry_op"),
            _make_span("retry_op"),
        ])
        highlights = extract_trace_highlights(trace)
        assert any("Repeated span" in h and "retry_op" in h for h in highlights)

    def test_empty_trace_highlight(self):
        trace = _make_trace([])
        highlights = extract_trace_highlights(trace)
        assert any("Empty trace" in h for h in highlights)

    def test_clean_trace_no_errors(self):
        trace = _make_trace([_make_span("good_span", duration_ms=50.0)])
        highlights = extract_trace_highlights(trace)
        assert not any("Error" in h for h in highlights)
        assert not any("Slow" in h for h in highlights)

    def test_multiple_highlights(self):
        trace = _make_trace([
            _make_span("err_span", status="error", attributes={"error": "fail"}),
            _make_span("slow_span", duration_ms=5000.0),
        ])
        highlights = extract_trace_highlights(trace)
        assert len(highlights) >= 2


# ---------------------------------------------------------------------------
# build_heuristic_summary
# ---------------------------------------------------------------------------


class TestBuildHeuristicSummary:
    def test_basic_summary(self):
        trace = _make_trace([
            _make_span("llm1", "llm_call", duration_ms=200.0),
            _make_span("tool1", "tool_call", duration_ms=300.0),
        ])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert summary.trace_id == "trace-001"
        assert summary.span_count == 2
        assert "llm_call" in summary.summary_text
        assert "tool_call" in summary.summary_text

    def test_includes_latency_in_text(self):
        trace = _make_trace([_make_span("s1", duration_ms=500.0)])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert "500ms" in summary.summary_text

    def test_excludes_latency_when_disabled(self):
        trace = _make_trace([_make_span("s1", duration_ms=500.0)])
        summary = build_heuristic_summary(trace, TraceSummaryConfig(include_latencies=False))
        assert "500ms" not in summary.summary_text

    def test_includes_tool_names(self):
        trace = _make_trace([_make_span("web_search", "tool_call")])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert "web_search" in summary.summary_text

    def test_error_count_in_text(self):
        trace = _make_trace([
            _make_span("err1", status="error", attributes={"error": "fail1"}),
            _make_span("err2", status="error", attributes={"error": "fail2"}),
        ])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert "2 error" in summary.summary_text

    def test_key_findings_populated(self):
        trace = _make_trace([
            _make_span("s1", status="error", attributes={"error": "oops"}),
        ])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert len(summary.key_findings) > 0

    def test_generated_at_set(self):
        trace = _make_trace([_make_span()])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert summary.generated_at != ""

    def test_total_latency_computed(self):
        trace = _make_trace([
            _make_span("s1", duration_ms=100.0),
            _make_span("s2", duration_ms=200.0),
        ])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert summary.total_latency_ms == 300.0

    def test_empty_trace(self):
        trace = _make_trace([])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert summary.span_count == 0
        assert summary.total_latency_ms == 0.0
        assert "0 spans" in summary.summary_text

    def test_respects_max_spans(self):
        spans = [_make_span(f"s{i}", duration_ms=10.0) for i in range(50)]
        trace = _make_trace(spans)
        summary = build_heuristic_summary(trace, TraceSummaryConfig(max_spans=5))
        assert summary.span_count == 5

    def test_no_errors_finding(self):
        trace = _make_trace([_make_span("ok_span")])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert any("No errors" in f for f in summary.key_findings)

    def test_all_fast_finding(self):
        trace = _make_trace([_make_span("fast", duration_ms=50.0)])
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert any("within 1s" in f for f in summary.key_findings)

    def test_uses_id_field_as_fallback(self):
        trace = {"id": "fallback-id", "spans": [_make_span()]}
        summary = build_heuristic_summary(trace, TraceSummaryConfig())
        assert summary.trace_id == "fallback-id"


# ---------------------------------------------------------------------------
# parse_llm_summary_response
# ---------------------------------------------------------------------------


class TestParseLlmSummaryResponse:
    def test_extracts_bullet_points(self):
        response = "This trace was successful.\n- Fast execution\n- No errors found"
        summary = parse_llm_summary_response(response, "t1")
        assert summary.key_findings == ["Fast execution", "No errors found"]

    def test_summary_text_excludes_bullets(self):
        response = "Overview of trace.\n- Finding 1\n- Finding 2"
        summary = parse_llm_summary_response(response, "t1")
        assert "Overview" in summary.summary_text
        assert "Finding 1" not in summary.summary_text

    def test_no_bullet_points(self):
        response = "Simple summary with no bullets."
        summary = parse_llm_summary_response(response, "t1")
        assert summary.key_findings == []
        assert "Simple summary" in summary.summary_text

    def test_asterisk_bullets(self):
        response = "Summary text.\n* Point A\n* Point B"
        summary = parse_llm_summary_response(response, "t1")
        assert "Point A" in summary.key_findings
        assert "Point B" in summary.key_findings

    def test_mixed_content(self):
        response = "Para 1.\n- Bullet 1\nPara 2.\n- Bullet 2"
        summary = parse_llm_summary_response(response, "t1")
        assert len(summary.key_findings) == 2
        assert "Para 1" in summary.summary_text

    def test_trace_id_set(self):
        summary = parse_llm_summary_response("text", "my-trace")
        assert summary.trace_id == "my-trace"

    def test_generated_at_set(self):
        summary = parse_llm_summary_response("text", "t1")
        assert summary.generated_at != ""

    def test_empty_response(self):
        summary = parse_llm_summary_response("", "t1")
        assert summary.summary_text == ""
        assert summary.key_findings == []

    def test_whitespace_handling(self):
        response = "  Summary.  \n  - Finding with spaces  \n"
        summary = parse_llm_summary_response(response, "t1")
        assert "Finding with spaces" in summary.key_findings


# ---------------------------------------------------------------------------
# format_trace_summary
# ---------------------------------------------------------------------------


class TestFormatTraceSummary:
    def test_includes_trace_id(self):
        ts = TraceSummary(trace_id="t-123", summary_text="All good.")
        output = format_trace_summary(ts)
        assert "t-123" in output

    def test_includes_summary_text(self):
        ts = TraceSummary(trace_id="t1", summary_text="Trace completed successfully.")
        output = format_trace_summary(ts)
        assert "Trace completed successfully." in output

    def test_includes_key_findings(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", key_findings=["Fast", "Clean"])
        output = format_trace_summary(ts)
        assert "Key Findings" in output
        assert "Fast" in output
        assert "Clean" in output

    def test_includes_span_count(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", span_count=15)
        output = format_trace_summary(ts)
        assert "15" in output

    def test_includes_latency(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", total_latency_ms=1234.0)
        output = format_trace_summary(ts)
        assert "1234ms" in output

    def test_no_latency_when_zero(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", total_latency_ms=0.0)
        output = format_trace_summary(ts)
        assert "Total Latency" not in output

    def test_no_findings_section_when_empty(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", key_findings=[])
        output = format_trace_summary(ts)
        assert "Key Findings" not in output

    def test_includes_generated_at(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok", generated_at="2026-01-01")
        output = format_trace_summary(ts)
        assert "2026-01-01" in output

    def test_separator_line(self):
        ts = TraceSummary(trace_id="t1", summary_text="ok")
        output = format_trace_summary(ts)
        assert "=" in output
