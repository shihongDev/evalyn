"""Trace commands: list-calls, show-call, show-trace, show-projects, delete-traces.

This module provides CLI commands for inspecting and managing traced function calls
stored in the database.

Commands:
- list-calls: List captured function calls with filtering by project/simulation
- show-call: Show detailed information about a specific call including inputs, outputs, metadata
- show-trace: Show hierarchical span tree for a traced call (Phoenix-style visualization)
- show-projects: Show summary of projects and their traces
- delete-traces: Delete the latest N traces from storage

Typical workflow:
1. Run your agent with @eval decorator to capture traces
2. Use 'evalyn show-projects' to see available projects
3. Use 'evalyn list-calls --project <name>' to see calls for a project
4. Use 'evalyn show-call --id <id>' to inspect a specific call
5. Use 'evalyn show-trace --id <id>' to visualize the span tree
6. Use 'evalyn delete-traces -n 5 --db test' to clean up test traces
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...storage import SQLiteStorage

from ..utils.command_common import (
    resolve_call_id,
    resolve_call_id_or_last,
    try_resolve_call_id,
)
from ..utils.errors import fatal_error
from ..utils.formatters import format_duration, trim_timestamp
from ..utils.hints import HintCollector
from ..utils.rich import banner, kv, section, table as rich_table, footer, icon, status_icon
from ..utils.validation import extract_project_id

from ...storage.sqlite import DB_PATHS


def _get_storage(args: argparse.Namespace) -> SQLiteStorage:
    """Get storage based on --db flag."""
    from ...storage import SQLiteStorage

    db = getattr(args, "db", None)
    if db and db in DB_PATHS:
        return SQLiteStorage(DB_PATHS[db])
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    return tracer.storage


# ---------------------------------------------------------------------------
# Trace formatting helpers (extracted from cmd_show_call for reuse)
# ---------------------------------------------------------------------------


def _format_value(value: Any, max_len: int = 300) -> str:
    """Format a value for display, truncating if needed."""
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + "..."
    try:
        text = json.dumps(value, indent=2)
        return text if len(text) <= max_len else text[:max_len] + "..."
    except Exception:
        text = str(value)
        return text if len(text) <= max_len else text[:max_len] + "..."


def _truncate(text: Any, max_len: int = 120) -> str:
    """Truncate text to max_len characters."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _normalize_span_time(raw: Any) -> float | None:
    """Normalize span timestamp to Unix seconds."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1e15:
            return value / 1e9
        if value > 1e12:
            return value / 1e6
        if value > 1e10:
            return value / 1e3
        return value
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            if raw.isdigit():
                return _normalize_span_time(int(raw))
            return _normalize_span_time(float(raw))
        except ValueError:
            pass
        try:
            import datetime as _dt

            return _dt.datetime.fromisoformat(raw).timestamp()
        except Exception:
            return None
    return None


def _span_duration_ms(span: dict) -> float | None:
    """Calculate span duration in milliseconds."""
    start_ts = _normalize_span_time(span.get("start_time"))
    end_ts = _normalize_span_time(span.get("end_time"))
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, (end_ts - start_ts) * 1000)


def _span_status(span: dict) -> str:
    """Get span status as normalized string."""
    status = span.get("status")
    if status is None:
        return "UNSET"
    text = str(status).upper()
    if "ERROR" in text:
        return "ERROR"
    if "OK" in text:
        return "OK"
    return text


def _detect_call_turns(inputs: Any) -> tuple[str, int]:
    """Infer single vs multi-turn interaction shape from call inputs."""
    if isinstance(inputs, dict):
        kwargs = inputs.get("kwargs", {})
        for key in ("messages", "history", "conversation", "turns"):
            val = kwargs.get(key)
            if isinstance(val, list):
                return ("multi" if len(val) > 1 else "single"), len(val)
        for arg in inputs.get("args", []):
            if isinstance(arg, list):
                return ("multi" if len(arg) > 1 else "single"), len(arg)
    return "single", 1


def _count_call_trace_events(call, kinds: list[str]) -> int:
    """Count trace events whose kind contains any provided substring."""
    return sum(1 for ev in (call.trace or []) if any(k in ev.kind.lower() for k in kinds))


def _count_call_llm_calls(call) -> int:
    """Count LLM calls using both legacy trace events and modern spans."""
    event_count = _count_call_trace_events(
        call,
        ["completion", "request", "llm", "gemini", "openai", "anthropic"],
    )
    span_count = sum(1 for s in (call.spans or []) if s.span_type == "llm_call")
    return max(event_count, span_count)


def _count_call_tool_calls(call) -> int:
    """Count tool calls using both legacy trace events and modern spans."""
    event_count = _count_call_trace_events(call, ["tool"])
    span_count = sum(1 for s in (call.spans or []) if s.span_type == "tool_call")
    return max(event_count, span_count)


def _print_show_call_header(call) -> None:
    """Print top-level call summary header."""
    status = "ERROR" if call.error else "OK"
    turn_label, turns = _detect_call_turns(call.inputs)
    llm_calls = _count_call_llm_calls(call)
    tool_events = _count_call_tool_calls(call)
    dur_str = f"{call.duration_ms:.2f} ms" if call.duration_ms is not None else "N/A"

    print()
    print(banner("CALL DETAILS"))
    print(kv([
        ("ID", call.id),
        ("Function", call.function_name),
        ("Status", f"{status} {status_icon(status == 'OK')}"),
        ("Session", str(call.session_id)),
        ("Started", str(call.started_at)),
        ("Ended", str(call.ended_at)),
        ("Duration", dur_str),
        ("Turns", f"{turn_label} ({turns})"),
        ("LLM calls", f"{llm_calls} | tool_events: {tool_events}"),
    ]))


def _print_show_call_inputs(call) -> None:
    """Print call input args/kwargs."""
    print()
    print(section("INPUT"))
    if not call.inputs:
        print("  <none>")
        return
    args_list = call.inputs.get("args", [])
    kwargs = call.inputs.get("kwargs", {})
    if args_list:
        print("  args:")
        for idx, arg in enumerate(args_list):
            print(f"    - [{idx}] {_format_value(arg)}")
    if kwargs:
        print("  kwargs:")
        for key, value in kwargs.items():
            print(f"    - {key}: {_format_value(value)}")
    if not args_list and not kwargs:
        print("  <empty>")


def _print_show_call_output(call) -> None:
    """Print call error or output preview."""
    if call.error:
        print()
        print(section("ERROR"))
        print(call.error)
        return

    print()
    print(section("OUTPUT"))
    output_text = str(call.output or "")
    print(kv([
        ("Type", type(call.output).__name__),
        ("Length", f"{len(output_text)} chars"),
        ("Preview", _format_value(output_text, max_len=1000)),
    ]))


def _print_show_call_metadata(meta: dict) -> None:
    """Print call metadata with nested dict support."""
    print()
    print(section("METADATA"))
    for key, value in meta.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_val in value.items():
                if sub_key == "source":
                    src = sub_val or ""
                    src_preview = src if len(src) <= 1200 else src[:1200] + "..."
                    print("    source:")
                    for line in src_preview.splitlines()[:40]:
                        print(f"      {line}")
                else:
                    print(f"    - {sub_key}: {_format_value(sub_val, max_len=400)}")
            continue
        print(f"  - {key}: {_format_value(value, max_len=400)}")


def _format_trace_event_inline(value: Any, max_len: int = 100) -> str:
    """Format trace event values for inline timeline summaries."""
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, separators=(",", ":"))
        else:
            text = str(value)
    except Exception:
        text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _summarize_trace_event_detail(detail: Any, max_items: int = 4) -> str:
    """Build a compact summary string for trace event details."""
    if not detail:
        return ""
    parts = []
    if isinstance(detail, dict):
        model = detail.get("model")
        if model:
            parts.append(f"model={_truncate(model, 40)}")
        tool = detail.get("tool") or detail.get("name")
        if tool:
            parts.append(f"tool={_truncate(tool, 40)}")
        if "config" in detail and len(parts) < max_items:
            cfg = detail.get("config") or {}
            tools = []
            raw_tools = cfg.get("tools")
            if isinstance(raw_tools, list):
                for tool_entry in raw_tools:
                    if isinstance(tool_entry, dict):
                        tools.extend(list(tool_entry.keys()))
                    else:
                        tools.append(str(tool_entry))
            if tools:
                parts.append(f"tools={','.join(tools)}")
            else:
                parts.append(f"config_keys={list(cfg.keys())}")
        for key in ("status", "status_code", "elapsed_ms", "duration_ms", "count", "length"):
            if key in detail and len(parts) < max_items:
                parts.append(f"{key}={_format_trace_event_inline(detail.get(key), 40)}")
        for key in ("error", "url"):
            if key in detail and len(parts) < max_items:
                parts.append(f"{key}={_truncate(detail.get(key), 60)}")
        for key in ("contents", "messages", "prompt", "prompt_excerpt"):
            if key in detail and len(parts) < max_items:
                parts.append(f"{key}={_truncate(detail.get(key), 80)}")
    if not parts:
        parts.append(_truncate(_format_trace_event_inline(detail, 120), 120))
    return " ".join(parts[:max_items])


def _print_show_call_events(call) -> None:
    """Print legacy trace events summary and timeline."""
    if not call.trace:
        return

    total = len(call.trace)
    reqs = sum(1 for ev in call.trace if ev.kind.lower().endswith(".request"))
    resps = sum(1 for ev in call.trace if ev.kind.lower().endswith(".response"))
    tool_cnt = sum(1 for ev in call.trace if "tool" in ev.kind.lower())

    print()
    print(section(f"EVENTS ({total})"))
    print(f"  total={total} | requests={reqs} | responses={resps} | tool_events={tool_cnt}")

    kind_counts: dict[str, int] = {}
    for ev in call.trace:
        kind_counts[ev.kind] = kind_counts.get(ev.kind, 0) + 1
    for kind, count in sorted(kind_counts.items()):
        print(f"  - {kind}: {count}")

    headers = ["idx", "t+ms", "delta_ms", "kind", "summary"]
    rows: list[list[str]] = []
    prev_ts = None
    for idx, ev in enumerate(call.trace, start=1):
        elapsed_ms = (
            (ev.timestamp - call.started_at).total_seconds() * 1000 if call.started_at else 0.0
        )
        delta_ms = (ev.timestamp - prev_ts).total_seconds() * 1000 if prev_ts else 0.0
        summary = _summarize_trace_event_detail(ev.detail or {})
        rows.append([
            str(idx),
            f"{elapsed_ms:.1f}",
            f"{delta_ms:.1f}",
            ev.kind,
            summary,
        ])
        prev_ts = ev.timestamp
    print(rich_table(headers, rows, align=["right", "right", "right", "left", "left"]))


def _print_show_call_sparse_span_timeline(call) -> None:
    """Print chronological span timeline when traces are sparse."""
    if len(call.trace) > 1 or not call.spans or len(call.spans) <= 1:
        return

    print()
    print(section("SPAN TIMELINE (CHRONOLOGICAL)"))

    sorted_spans = sorted([s for s in call.spans if s.start_time], key=lambda s: s.start_time)
    for idx, span in enumerate(sorted_spans[:50], start=1):
        elapsed_ms = 0.0
        if span.start_time and call.started_at:
            elapsed_ms = (span.start_time - call.started_at).total_seconds() * 1000

        dur_str = f"{span.duration_ms:.0f}ms" if span.duration_ms is not None else "N/A"
        attrs = span.attributes or {}
        summary_parts = []
        if span.span_type == "llm_call":
            model = attrs.get("model", "")
            if model:
                summary_parts.append(f"model={model}")
        elif span.span_type == "tool_call":
            tool = attrs.get("tool_name", span.name)
            summary_parts.append(f"tool={tool}")
        elif span.span_type == "user_message":
            content = attrs.get("content", "")[:50]
            summary_parts.append(f"input={content}")
        summary = " ".join(summary_parts)
        print(
            f"{idx:3} | +{elapsed_ms:8.0f}ms | {span.span_type:12} | {span.name:20} | {dur_str:8} | {summary[:60]}"
        )

    if len(sorted_spans) > 50:
        print(f"  ... and {len(sorted_spans) - 50} more spans")


def _build_span_tree(spans: list) -> list[dict]:
    """Build a sorted span tree from a list of spans for display."""
    by_id = {s.id: {"span": s, "children": []} for s in spans}
    roots: list[dict] = []
    for s in spans:
        node = by_id[s.id]
        if s.parent_id and s.parent_id in by_id:
            by_id[s.parent_id]["children"].append(node)
        else:
            roots.append(node)

    def _sort_children(node: dict) -> None:
        node["children"].sort(key=lambda n: n["span"].start_time or 0)
        for child in node["children"]:
            _sort_children(child)

    for root in roots:
        _sort_children(root)
    return roots


def _call_span_tokens_info(attrs: dict) -> str:
    """Format compact token counts for span tree labels."""
    input_t = attrs.get("input_tokens", 0)
    output_t = attrs.get("output_tokens", 0)
    if input_t or output_t:
        return f" [{input_t}>{output_t} tok]"
    return ""


def _format_call_span_label(span) -> str:
    """Format a span label for show-call span tree."""
    dur = format_duration(span.duration_ms)
    tokens = _call_span_tokens_info(span.attributes or {})
    if span.span_type == "llm_call":
        return f"llm: {span.name}{tokens} ({dur})"
    if span.span_type == "tool_call":
        return f"tool: {span.name} ({dur})"
    if span.span_type == "node":
        return f"node: {span.name.replace('node:', '')} ({dur})"
    if span.span_type == "graph":
        return f"graph: {span.name.replace('graph:', '')} ({dur})"
    return f"{span.name} ({dur})"


def _render_call_span_node(node: dict, prefix: str = "", is_last: bool = True) -> None:
    """Render one node in the show-call span tree."""
    span = node["span"]
    connector = "`-- " if is_last else "|-- "
    print(f"{prefix}{connector}{_format_call_span_label(span)}")
    child_prefix = prefix + ("    " if is_last else "|   ")
    children = node["children"]
    for i, child in enumerate(children):
        _render_call_span_node(child, child_prefix, i == len(children) - 1)


def _print_show_call_spans(call) -> None:
    """Print span tree and summary from call.spans."""
    if not call.spans:
        return

    roots = _build_span_tree(call.spans or [])
    print()
    print(section("SPAN TREE"))
    for i, root in enumerate(roots):
        _render_call_span_node(root, "", i == len(roots) - 1)

    llm_count = sum(1 for s in call.spans if s.span_type == "llm_call")
    tool_count = sum(1 for s in call.spans if s.span_type == "tool_call")
    node_count = sum(1 for s in call.spans if s.span_type == "node")
    print(f"\n  {len(call.spans)} spans | {llm_count} LLM | {tool_count} tools | {node_count} nodes")


def _render_show_call_table(call) -> None:
    """Render the human-readable show-call output."""
    _print_show_call_header(call)
    _print_show_call_inputs(call)
    _print_show_call_output(call)
    if call.metadata:
        _print_show_call_metadata(call.metadata)
    _print_show_call_events(call)
    _print_show_call_sparse_span_timeline(call)
    _print_show_call_spans(call)
    print()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def cmd_list_calls(args: argparse.Namespace) -> None:
    """List captured function calls."""
    storage = _get_storage(args)
    base_limit = args.limit + getattr(args, "offset", 0)

    # Push function_name filter to SQL where possible
    func_filter = getattr(args, "function", None)
    project = getattr(args, "project", None)

    # Only overfetch for filters that can't be pushed to SQL
    has_post_filters = (
        getattr(args, "error_only", False)
        or getattr(args, "simulation", False)
        or getattr(args, "production", False)
    )
    fetch_limit = base_limit * 2 if has_post_filters else base_limit + 50

    all_calls = (
        storage.list_calls(
            limit=fetch_limit,
            project=project,
            function_name=func_filter,
            lightweight=True,
        )
        if storage
        else []
    )
    calls = list(all_calls)

    # Post-filters for conditions not handled in SQL
    if getattr(args, "error_only", False) and calls:
        calls = [c for c in calls if c.error]

    if hasattr(args, "simulation") and args.simulation and calls:
        calls = [
            c for c in calls
            if (c.metadata if isinstance(c.metadata, dict) else {}).get("is_simulation", False)
        ]
    elif hasattr(args, "production") and args.production and calls:
        calls = [
            c for c in calls
            if not (c.metadata if isinstance(c.metadata, dict) else {}).get("is_simulation", False)
        ]

    # Sorting
    sort_field = getattr(args, "sort", "started_at")
    sort_reverse = True  # Default descending (newest first)
    if sort_field.startswith("+"):
        sort_field = sort_field[1:]
        sort_reverse = False
    elif sort_field.startswith("-"):
        sort_field = sort_field[1:]
        sort_reverse = True

    def get_sort_key(call):
        if sort_field == "started_at":
            return call.started_at or ""
        elif sort_field == "duration":
            return call.duration_ms or 0
        elif sort_field == "function":
            return call.function_name or ""
        elif sort_field == "status":
            return 1 if call.error else 0
        return call.started_at or ""

    calls.sort(key=get_sort_key, reverse=sort_reverse)

    # Track total before pagination for "more available" indicator
    total_after_filter = len(calls)

    # Pagination: apply offset and limit
    offset = getattr(args, "offset", 0)
    if offset > 0:
        calls = calls[offset:]
    calls = calls[: args.limit]

    output_format = getattr(args, "format", "table")

    if not calls:
        if output_format == "json":
            print(json.dumps({"calls": [], "total": 0, "showing": 0, "offset": offset}))
        else:
            print("No calls found.")
        return

    # Calculate "more available"
    more_available = max(0, total_after_filter - offset - len(calls))

    def _extract_call_meta(call):
        """Extract common display fields from call metadata once."""
        meta = call.metadata if isinstance(call.metadata, dict) else {}
        code = meta.get("code", {}) if meta else {}
        return {
            "project": extract_project_id(call.metadata) or "",
            "version": meta.get("version", ""),
            "is_sim": meta.get("is_simulation", False),
            "code": code,
            "file_path": code.get("file_path") if isinstance(code, dict) else None,
            "status": "ERROR" if call.error else "OK",
        }

    # JSON output mode
    if output_format == "json":
        result_calls = []
        for call in calls:
            m = _extract_call_meta(call)
            result_calls.append(
                {
                    "id": call.id,
                    "function": call.function_name,
                    "project": m["project"],
                    "version": m["version"],
                    "is_simulation": m["is_sim"],
                    "status": m["status"],
                    "file": m["file_path"],
                    "started_at": call.started_at.isoformat()
                    if call.started_at
                    else None,
                    "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                    "duration_ms": call.duration_ms,
                }
            )
        result = {
            "calls": result_calls,
            "total": total_after_filter,
            "showing": len(calls),
            "offset": offset,
            "more_available": more_available,
        }
        print(json.dumps(result, indent=2))
        return

    # Table output mode
    headers = ["ID", "Function", "Project", "Status", "Duration", "Started"]
    align = ["left", "left", "left", "left", "left", "left"]
    rows = []
    for call in calls:
        m = _extract_call_meta(call)
        short_id = call.id[:8]
        status = icon("pass") if m["status"] == "OK" else icon("fail")
        rows.append([
            short_id,
            call.function_name or "",
            m["project"],
            status,
            format_duration(call.duration_ms),
            trim_timestamp(call.started_at),
        ])

    print(banner("TRACED CALLS"))
    print(rich_table(headers, rows, align=align))

    # Show "more available" indicator
    if more_available > 0:
        next_offset = offset + len(calls)
        print(
            f"\n({more_available} more available. Use --offset {next_offset} to see next page)"
        )

    # Show hints
    hint_items = []
    if calls:
        short_id = calls[0].id[:8]
        hint_items.append((f"evalyn show-call --id {short_id}", "See call details"))
    hint_text = footer(hint_items, quiet=getattr(args, "quiet", False), format=output_format)
    if hint_text:
        print(hint_text)


def cmd_show_call(args: argparse.Namespace) -> None:
    """Show detailed information about a specific call."""
    storage = _get_storage(args)
    output_format = getattr(args, "format", "table")

    if not storage:
        fatal_error("No storage configured")

    call_id = resolve_call_id_or_last(
        storage,
        input_id=args.id,
        use_last=getattr(args, "last", False),
    )

    call = storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")

    # JSON output mode
    if output_format == "json":
        print(json.dumps(call.as_dict(), indent=2, default=str))
        return
    _render_show_call_table(call)
    hints = HintCollector(quiet=getattr(args, "quiet", False), format=output_format)
    if call.spans:
        hints.add(
            f"evalyn show-trace --id {call.id[:8]} --verbose",
            "See span tree with details",
            options=[
                ("--format json", "Machine-readable output"),
            ],
        )
    hints.render()


def _show_trace_format_tokens(attrs: dict) -> str:
    """Format token usage suffix for show-trace labels."""
    input_t = attrs.get("input_tokens", 0)
    output_t = attrs.get("output_tokens", 0)
    tool_t = attrs.get("tool_tokens", 0)
    if input_t or output_t:
        if tool_t:
            return f" [{input_t}→{output_t} +{tool_t} tool tokens]"
        return f" [{input_t}→{output_t} tokens]"
    return ""


def _show_trace_status_icon(status: str) -> str:
    """Map span status to a compact icon."""
    if status == "error":
        return "[X]"
    if status == "ok":
        return "[OK]"
    return "[-]"


def _show_trace_grounding_lines(attrs: dict, prefix: str) -> list[str]:
    """Format grounding metadata lines for LLM spans."""
    lines: list[str] = []
    queries = attrs.get("search_queries")
    if queries:
        lines.append(f"{prefix}    [Search] Queries: {', '.join(queries[:3])}")
        if len(queries) > 3:
            lines.append(f"{prefix}       ... and {len(queries) - 3} more")
    sources = attrs.get("sources")
    if sources:
        lines.append(f"{prefix}    [Sources]:")
        for src in sources[:3]:
            title = src.get("title", "")[:50]
            uri = src.get("uri", "")
            lines.append(f"{prefix}       - {title}")
            if uri:
                lines.append(f"{prefix}         {uri[:60]}...")
        if len(sources) > 3:
            lines.append(f"{prefix}       ... and {len(sources) - 3} more")
    return lines


def _print_show_trace_no_spans(call) -> None:
    """Fallback output when spans are unavailable."""
    print(f"\nTrace: {call.function_name} ({format_duration(call.duration_ms)})")
    print("  <no spans captured>")
    print("\n  Tip: Re-run with latest evalyn_sdk to capture spans.")
    if call.trace:
        print(f"\n  Trace Events ({len(call.trace)}):")
        for ev in call.trace[:20]:
            print(f"    - {ev.kind}")
        if len(call.trace) > 20:
            print(f"    ... and {len(call.trace) - 20} more")



def _show_trace_limits(full_output: bool) -> dict[str, Optional[int]]:
    """Return truncation limits for verbose trace rendering."""
    return {
        "input": None if full_output else 800,
        "output": None if full_output else 1000,
        "thinking": None if full_output else 500,
    }


def _truncate_with_indicator(text: str, limit: Optional[int]) -> str:
    """Truncate text and add a visible truncation marker."""
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + " [...truncated]"


def _render_show_trace_verbose_details(span, prefix: str, limits: dict[str, Optional[int]]) -> None:
    """Render verbose details block for a span."""
    attrs = span.attributes or {}
    detail_prefix = prefix + "    "

    if span.span_type == "user_message":
        content = attrs.get("content", "")
        if content:
            print(
                f"{detail_prefix}content: {_truncate_with_indicator(str(content), limits['input'])}"
            )
        return

    if span.span_type == "llm_call":
        model = attrs.get("model", "")
        if model:
            print(f"{detail_prefix}model: {model}")

        request = attrs.get("request")
        if request:
            messages = request.get("messages", "") if isinstance(request, dict) else str(request)
            if messages:
                print(
                    f"{detail_prefix}request: {_truncate_with_indicator(str(messages), limits['input'])}"
                )

        response = attrs.get("response")
        if response:
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            if content:
                print(
                    f"{detail_prefix}response: {_truncate_with_indicator(str(content), limits['output'])}"
                )

        output = attrs.get("output") or attrs.get("output_preview", "")
        if output and not response:
            print(
                f"{detail_prefix}output: {_truncate_with_indicator(str(output), limits['output'])}"
            )

        thinking = attrs.get("thinking")
        if thinking:
            print(
                f"{detail_prefix}thinking: {_truncate_with_indicator(str(thinking), limits['thinking'])}"
            )
        return

    if span.span_type == "tool_call":
        tool_input = attrs.get("input", "")
        if tool_input:
            print(
                f"{detail_prefix}input: {_truncate_with_indicator(str(tool_input), limits['input'])}"
            )
        tool_output = attrs.get("output", "")
        if tool_output:
            print(
                f"{detail_prefix}output: {_truncate_with_indicator(str(tool_output), limits['output'])}"
            )
        subagent = attrs.get("executing_subagent")
        if subagent:
            print(f"{detail_prefix}subagent: {subagent}")


def _count_show_trace_descendants(node: dict) -> int:
    """Count descendants in the span tree."""
    count = len(node["children"])
    for child in node["children"]:
        count += _count_show_trace_descendants(child)
    return count


def _show_trace_span_label(span) -> str:
    """Format span label for show-trace tree."""
    duration = format_duration(span.duration_ms)
    status = _show_trace_status_icon(span.status)
    tokens = _show_trace_format_tokens(span.attributes or {})

    if span.span_type == "session":
        return f"{span.name} ({duration})"
    if span.span_type == "graph":
        return f"graph:{span.name.replace('graph:', '')} ({duration})"
    if span.span_type == "node":
        return f"node:{span.name.replace('node:', '')} ({duration})"
    if span.span_type == "llm_call":
        return f"llm_call {span.name}{tokens} ({duration})"
    if span.span_type == "tool_call":
        return f"tool_call {span.name} ({duration})"
    if span.span_type == "scorer":
        score = (span.attributes or {}).get("score", "?")
        return f"{status} {span.name} (score: {score})"
    return f"{span.name} ({duration})"


def _render_show_trace_node(
    node: dict,
    *,
    prefix: str,
    is_last: bool,
    depth: int,
    max_depth: Optional[int],
    verbose: bool,
    truncated_count: list[int],
    limits: dict[str, Optional[int]],
) -> None:
    """Render a span node with optional verbose details and depth truncation."""
    if max_depth is not None and depth >= max_depth:
        truncated_count[0] += 1 + _count_show_trace_descendants(node)
        return

    span = node["span"]
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{_show_trace_span_label(span)}")

    if verbose:
        detail_prefix = prefix + ("    " if is_last else "│   ")
        _render_show_trace_verbose_details(span, detail_prefix, limits)

    if span.span_type == "llm_call":
        for line in _show_trace_grounding_lines(span.attributes or {}, prefix):
            print(line)

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = node["children"]
    for i, child in enumerate(children):
        _render_show_trace_node(
            child,
            prefix=child_prefix,
            is_last=i == len(children) - 1,
            depth=depth + 1,
            max_depth=max_depth,
            verbose=verbose,
            truncated_count=truncated_count,
            limits=limits,
        )


def _print_show_trace_header(call) -> None:
    """Print show-trace header lines."""
    is_ok = not call.error
    si = status_icon(is_ok)
    title = f"TRACE: {call.function_name} ({format_duration(call.duration_ms)}) {si}"
    print()
    print(banner(title))
    pairs = [("Call ID", call.id)]
    if call.session_id:
        pairs.append(("Session", str(call.session_id)))
    print(kv(pairs))
    print()


def _print_show_trace_summary_and_hints(
    *,
    call,
    spans: list,
    args: argparse.Namespace,
    verbose: bool,
    full_output: bool,
    truncated_count: list[int],
) -> None:
    """Print post-tree summary and helpful hints."""
    llm_count = sum(1 for s in spans if s.span_type == "llm_call")
    tool_count = sum(1 for s in spans if s.span_type == "tool_call")
    node_count = sum(1 for s in spans if s.span_type == "node")

    summary_line = (
        f"\nSummary: {len(spans)} spans | {llm_count} LLM calls | {tool_count} tool calls | {node_count} nodes"
    )
    if truncated_count[0] > 0:
        summary_line += f" | ({truncated_count[0]} spans hidden, use --max-depth to see more)"
    print(summary_line)

    quiet = getattr(args, "quiet", False)
    hint_cmds: list[tuple[str, str]] = []
    if not verbose and (llm_count > 0 or tool_count > 0):
        hint_cmds.append(("evalyn show-trace --id ... --verbose", "See input/output details for each span"))

    meta = call.metadata if isinstance(call.metadata, dict) else {}
    project = meta.get("project_id") or meta.get("project_name")
    if verbose and not full_output:
        hint_cmds.append((f"evalyn show-span --call-id {call.id[:8]} --span <name>", "See full span content"))
    if project and verbose:
        hint_cmds.append((f"evalyn build-dataset --project {project}", "Build a dataset from traces"))
    foot = footer(hint_cmds, quiet=quiet)
    if foot:
        print(foot)


def _render_show_trace(call, args: argparse.Namespace) -> None:
    """Render the human-readable trace tree view."""
    spans = call.spans or []
    if not spans:
        _print_show_trace_no_spans(call)
        return

    roots = _build_span_tree(spans)
    max_depth = getattr(args, "max_depth", None)
    verbose = getattr(args, "verbose", False)
    full_output = getattr(args, "full", False)
    truncated_count = [0]
    limits = _show_trace_limits(full_output)

    _print_show_trace_header(call)
    for i, root in enumerate(roots):
        _render_show_trace_node(
            root,
            prefix="",
            is_last=i == len(roots) - 1,
            depth=0,
            max_depth=max_depth,
            verbose=verbose,
            truncated_count=truncated_count,
            limits=limits,
        )

    _print_show_trace_summary_and_hints(
        call=call,
        spans=spans,
        args=args,
        verbose=verbose,
        full_output=full_output,
        truncated_count=truncated_count,
    )


def cmd_show_trace(args: argparse.Namespace) -> None:
    """Show hierarchical span tree for a traced call (Phoenix-style visualization)."""
    storage = _get_storage(args)
    if not storage:
        fatal_error("No storage configured")

    call_id = resolve_call_id_or_last(
        storage,
        input_id=args.id,
        use_last=getattr(args, "last", False),
    )

    call = storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")
    _render_show_trace(call, args)


def cmd_show_span(args: argparse.Namespace) -> None:
    """Show detailed information about a specific span."""
    storage = _get_storage(args)
    output_format = getattr(args, "format", "table")

    if not storage:
        fatal_error("No storage configured")

    # Need call ID and span name/index
    if not args.call_id:
        fatal_error("Must specify --call-id")

    call_id = resolve_call_id(storage, args.call_id)

    call = storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")

    spans = call.spans or []
    if not spans:
        fatal_error("No spans found in this call")

    # Find span by name or index
    target_span = None
    if args.span:
        # Try as index first
        try:
            idx = int(args.span)
            if 0 <= idx < len(spans):
                target_span = spans[idx]
        except ValueError:
            pass

        # Try as name match
        if not target_span:
            for s in spans:
                if args.span.lower() in s.name.lower():
                    target_span = s
                    break

    if not target_span:
        # List available spans
        print("Available spans:")
        for i, s in enumerate(spans[:50]):
            print(f"  [{i}] {s.span_type}: {s.name}")
        if len(spans) > 50:
            print(f"  ... and {len(spans) - 50} more")
        fatal_error(f"Span '{args.span}' not found. Use index or name substring.")

    # JSON output
    if output_format == "json":
        span_dict = {
            "id": target_span.id,
            "name": target_span.name,
            "span_type": target_span.span_type,
            "parent_id": target_span.parent_id,
            "start_time": str(target_span.start_time),
            "end_time": str(target_span.end_time),
            "duration_ms": target_span.duration_ms,
            "status": target_span.status,
            "attributes": target_span.attributes,
        }
        print(json.dumps(span_dict, indent=2, default=str))
        return

    # Table output
    dur_str = f"{target_span.duration_ms:.2f}ms" if target_span.duration_ms else "n/a"
    print()
    print(banner("SPAN DETAILS"))
    print(kv([
        ("Name", target_span.name),
        ("ID", target_span.id),
        ("Type", target_span.span_type),
        ("Status", target_span.status),
        ("Parent", target_span.parent_id or "(root)"),
        ("Started", str(target_span.start_time)),
        ("Ended", str(target_span.end_time)),
        ("Duration", dur_str),
    ]))

    attrs = target_span.attributes or {}
    if attrs:
        print()
        print(section("ATTRIBUTES"))
        for key, value in attrs.items():
            value_str = str(value)
            if len(value_str) > 1000:
                value_str = value_str[:1000] + "..."
            # Multi-line values get special formatting
            if "\n" in value_str or len(value_str) > 100:
                print(f"  {key}:")
                for line in value_str.split("\n")[:20]:
                    print(f"    {line[:200]}")
                if len(value_str.split("\n")) > 20:
                    print(f"    ... ({len(value_str.split(chr(10))) - 20} more lines)")
            else:
                print(f"  {key}: {value_str}")
    else:
        print("\n  Attributes: (none)")

    # Show hints
    quiet = getattr(args, "quiet", False)
    meta = call.metadata if isinstance(call.metadata, dict) else {}
    project = meta.get("project_id") or meta.get("project_name")
    hint_cmds: list[tuple[str, str]] = []
    if project:
        hint_cmds.append((f"evalyn build-dataset --project {project}", "Build a dataset from traces"))
    foot = footer(hint_cmds, quiet=quiet, format=output_format)
    if foot:
        print(foot)
    print()


def cmd_show_projects(args: argparse.Namespace) -> None:
    """Show summary of projects and their traces."""
    storage = _get_storage(args)
    if not storage:
        fatal_error("No storage configured")
    calls = storage.list_calls(limit=args.limit, lightweight=True)
    summary = {}
    for call in calls:
        meta = call.metadata if isinstance(call.metadata, dict) else {}
        project = (
            meta.get("project_id")
            or meta.get("project_name")
            or call.function_name
            or "unknown"
        )
        version = meta.get("version") or ""
        key = (project, version)
        rec = summary.setdefault(
            key,
            {
                "total": 0,
                "errors": 0,
                "first": call.started_at,
                "last": call.started_at,
            },
        )
        rec["total"] += 1
        if call.error:
            rec["errors"] += 1
        if call.started_at and rec["first"] and call.started_at < rec["first"]:
            rec["first"] = call.started_at
        if call.started_at and rec["last"] and call.started_at > rec["last"]:
            rec["last"] = call.started_at

    headers = ["Project", "Calls", "Errors", "Last Active"]
    align = ["left", "right", "right", "left"]
    rows = []
    first_project = None
    for (project, version), rec in summary.items():
        if first_project is None:
            first_project = project
        rows.append([
            project,
            str(rec["total"]),
            str(rec["errors"]),
            trim_timestamp(rec["last"]),
        ])

    print(banner("PROJECTS"))
    print(rich_table(headers, rows, align=align))

    # Show hints
    hint_items = []
    if first_project:
        hint_items.append((f"evalyn list-calls --project {first_project}", "See calls for this project"))
    hint_text = footer(hint_items, quiet=getattr(args, "quiet", False))
    if hint_text:
        print(hint_text)


# ---------------------------------------------------------------------------
# Delete traces
# ---------------------------------------------------------------------------


def cmd_delete_traces(args: argparse.Namespace) -> None:
    """Delete traces from storage by ID or latest N."""
    db = getattr(args, "db", "test")
    n = getattr(args, "n", None)
    ids = getattr(args, "id", None) or []
    yes = getattr(args, "yes", False)

    if db not in DB_PATHS:
        fatal_error(f"Unknown database: {db}")

    from ...storage import SQLiteStorage

    storage = SQLiteStorage(DB_PATHS[db])

    # Determine which calls to delete
    if ids:
        # Delete by specific IDs
        calls = []
        for short_id in ids:
            full_id = try_resolve_call_id(storage, short_id)
            if not full_id:
                print(f"Warning: Could not resolve ID {short_id}")
                continue
            call = storage.get_call(full_id)
            if call:
                calls.append(call)
            else:
                print(f"Warning: Call {short_id} not found")
    else:
        # Delete latest N (default: 1)
        limit = n if n is not None else 1
        calls = storage.list_calls(limit=limit, lightweight=True)

    if not calls:
        print(f"No traces found in {db} database.")
        return

    # Show what will be deleted
    print(f"\nTraces to delete from {db} database:\n")
    print("ID | Function | Started")
    print("-" * 60)
    for call in calls:
        started = (
            call.started_at.strftime("%Y-%m-%d %H:%M:%S") if call.started_at else "N/A"
        )
        print(f"{call.id[:8]} | {call.function_name or 'N/A'} | {started}")

    print(f"\nTotal: {len(calls)} trace(s)")

    # Extra warning for prod
    if db == "prod":
        print("\n*** WARNING: You are about to delete from PRODUCTION database! ***")

    # Confirm unless --yes
    if not yes:
        confirm = (
            input(f"\nDelete {len(calls)} trace(s) from {db}? [y/N]: ").strip().lower()
        )
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    # Delete
    call_ids = [call.id for call in calls]
    deleted = storage.delete_calls(call_ids)
    print(f"\n{icon('pass')} Deleted {deleted} trace(s) from {db} database.")


def _add_db_arg(parser) -> None:
    """Add --db argument to parser."""
    parser.add_argument(
        "--db",
        choices=["prod", "test"],
        help="Database to use: prod (default) or test",
    )


def register_commands(subparsers) -> None:
    """Register trace-related commands."""
    # list-calls
    p = subparsers.add_parser("list-calls", help="List captured function calls")
    _add_db_arg(p)
    p.add_argument("--limit", type=int, default=20, help="Number of calls to list")
    p.add_argument(
        "--offset", type=int, default=0, help="Skip first N results (pagination)"
    )
    p.add_argument("--project", help="Filter by project name")
    p.add_argument("--function", help="Filter by function name (substring match)")
    p.add_argument(
        "--error-only", action="store_true", help="Show only calls with errors"
    )
    p.add_argument(
        "--sort",
        default="started_at",
        help="Sort by field: started_at, duration, function, status. Prefix with + for ascending, - for descending (default: -started_at)",
    )
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.add_argument(
        "--simulation", action="store_true", help="Show only simulation calls"
    )
    p.add_argument(
        "--production", action="store_true", help="Show only production calls"
    )
    p.set_defaults(func=cmd_list_calls)

    # show-call
    p = subparsers.add_parser("show-call", help="Show details of a specific call")
    _add_db_arg(p)
    p.add_argument("--id", help="Call ID to show")
    p.add_argument("--last", action="store_true", help="Show the most recent call")
    p.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    p.set_defaults(func=cmd_show_call)

    # show-trace
    p = subparsers.add_parser("show-trace", help="Show hierarchical span tree")
    _add_db_arg(p)
    p.add_argument("--id", help="Call ID to show")
    p.add_argument("--last", action="store_true", help="Show the most recent call")
    p.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth of span tree to display (default: unlimited)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed input/output for each span",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Show full content without truncation (use with --verbose)",
    )
    p.set_defaults(func=cmd_show_trace)

    # show-span
    p = subparsers.add_parser("show-span", help="Show details of a specific span")
    _add_db_arg(p)
    p.add_argument("--call-id", "--id", required=True, help="Call ID containing the span")
    p.add_argument("--span", required=True, help="Span name or index to show")
    p.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    p.set_defaults(func=cmd_show_span)

    # show-projects
    p = subparsers.add_parser("show-projects", help="Show project summary")
    _add_db_arg(p)
    p.add_argument("--limit", type=int, default=1000, help="Max calls to scan")
    p.set_defaults(func=cmd_show_projects)

    # delete-traces
    p = subparsers.add_parser("delete-traces", help="Delete traces from storage")
    p.add_argument("--id", nargs="+", help="Trace ID(s) to delete (supports short IDs)")
    p.add_argument(
        "-n", type=int, help="Number of latest traces to delete (default: 1 if no --id)"
    )
    p.add_argument(
        "--db",
        choices=["test", "prod"],
        default="test",
        help="Database to use (default: test)",
    )
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p.set_defaults(func=cmd_delete_traces)


__all__ = [
    "cmd_list_calls",
    "cmd_show_call",
    "cmd_show_trace",
    "cmd_show_span",
    "cmd_show_projects",
    "cmd_delete_traces",
    "register_commands",
]
