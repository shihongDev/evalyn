"""Trace viewing commands: list-calls, show-call, show-trace, show-projects.

This module provides CLI commands for inspecting traced function calls stored in the database.
These commands are the entry point for understanding what your agent is doing.

Commands:
- list-calls: List captured function calls with filtering by project/simulation
- show-call: Show detailed information about a specific call including inputs, outputs, metadata
- show-trace: Show hierarchical span tree for a traced call (Phoenix-style visualization)
- show-projects: Show summary of projects and their traces

Typical workflow:
1. Run your agent with @eval decorator to capture traces
2. Use 'evalyn show-projects' to see available projects
3. Use 'evalyn list-calls --project <name>' to see calls for a project
4. Use 'evalyn show-call --id <id>' to inspect a specific call
5. Use 'evalyn show-trace --id <id>' to visualize the span tree
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ...decorators import get_default_tracer
from ..utils.errors import fatal_error
from ..utils.hints import print_hint
from ..utils.validation import extract_project_id


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
        if value > 1e12:
            return value / 1e9
        if value > 1e10:
            return value / 1e9
        if value > 1e6:
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


def _format_dur(ms: float | None) -> str:
    """Format duration for display."""
    if ms is None:
        return "?"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def cmd_list_calls(args: argparse.Namespace) -> None:
    """List captured function calls."""
    tracer = get_default_tracer()
    # Fetch more calls for filtering (we'll slice later for pagination)
    fetch_limit = args.limit + getattr(args, "offset", 0) + 100
    all_calls = tracer.storage.list_calls(limit=fetch_limit) if tracer.storage else []
    calls = list(all_calls)

    # Filter by project
    if args.project and calls:
        filtered = []
        for c in calls:
            pid = extract_project_id(c.metadata)
            if pid == args.project:
                filtered.append(c)
        calls = filtered

    # Filter by function name
    if getattr(args, "function", None) and calls:
        func_filter = args.function.lower()
        calls = [c for c in calls if func_filter in c.function_name.lower()]

    # Filter by error-only
    if getattr(args, "error_only", False) and calls:
        calls = [c for c in calls if c.error]

    # Filter by simulation/production if specified
    if hasattr(args, "simulation") and args.simulation and calls:
        filtered = []
        for c in calls:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            if meta.get("is_simulation", False):
                filtered.append(c)
        calls = filtered
    elif hasattr(args, "production") and args.production and calls:
        filtered = []
        for c in calls:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            if not meta.get("is_simulation", False):
                filtered.append(c)
        calls = filtered

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

    # JSON output mode
    if output_format == "json":
        result_calls = []
        for call in calls:
            code = (
                call.metadata.get("code", {}) if isinstance(call.metadata, dict) else {}
            )
            project = extract_project_id(call.metadata) or ""
            version = ""
            if isinstance(call.metadata, dict):
                version = call.metadata.get("version", "")
            is_sim = (
                call.metadata.get("is_simulation", False)
                if isinstance(call.metadata, dict)
                else False
            )
            result_calls.append(
                {
                    "id": call.id,
                    "function": call.function_name,
                    "project": project,
                    "version": version,
                    "is_simulation": is_sim,
                    "status": "ERROR" if call.error else "OK",
                    "file": code.get("file_path") if isinstance(code, dict) else None,
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
    headers = [
        "id",
        "function",
        "project",
        "version",
        "sim",
        "status",
        "file",
        "started_at",
        "ended_at",
        "duration_ms",
    ]
    print(" | ".join(headers))
    print("-" * 140)

    def _short_path(path: Any, max_len: int = 48) -> str:
        if not isinstance(path, str) or not path.strip():
            return ""
        raw = path.strip()
        display = raw
        try:
            rel = os.path.relpath(raw, os.getcwd())
            if rel and not rel.startswith("..") and not os.path.isabs(rel):
                display = rel
        except Exception:
            display = raw

        if len(display) <= max_len:
            return display

        base = os.path.basename(raw)
        parent = os.path.basename(os.path.dirname(raw))
        compact = os.path.join(parent, base) if parent else base
        if len(compact) <= max_len:
            return compact
        return "..." + compact[-(max_len - 3) :]

    for call in calls:
        status = "ERROR" if call.error else "OK"
        code = call.metadata.get("code", {}) if isinstance(call.metadata, dict) else {}
        project = ""
        version = ""
        is_sim = False
        project = extract_project_id(call.metadata) or ""
        if isinstance(call.metadata, dict):
            version = call.metadata.get("version", "")
            is_sim = call.metadata.get("is_simulation", False)
        file_path = code.get("file_path") if isinstance(code, dict) else None
        # Use short ID (first 8 chars) for easier copy-paste
        short_id = call.id[:8]
        row = [
            short_id,
            call.function_name,
            project,
            version,
            "Y" if is_sim else "",
            status,
            _short_path(file_path),
            str(call.started_at),
            str(call.ended_at),
            f"{call.duration_ms:.2f}",
        ]
        print(" | ".join(row))

    # Show "more available" indicator
    if more_available > 0:
        next_offset = offset + len(calls)
        print(
            f"\n({more_available} more available. Use --offset {next_offset} to see next page)"
        )

    # Show hint with first call ID (guard against empty list after filtering)
    if calls:
        # Use short ID (first 8 chars) in hint for easier copy-paste
        short_id = calls[0].id[:8]
        print_hint(
            f"To see details, run: evalyn show-call --id {short_id}",
            quiet=getattr(args, "quiet", False),
            format=output_format,
        )


def cmd_show_call(args: argparse.Namespace) -> None:
    """Show detailed information about a specific call."""
    tracer = get_default_tracer()
    output_format = getattr(args, "format", "table")

    if not tracer.storage:
        fatal_error("No storage configured")

    # Handle --last flag or --id
    if getattr(args, "last", False):
        calls = tracer.storage.list_calls(limit=1)
        if not calls:
            fatal_error("No calls found")
        call_id = calls[0].id
    elif args.id:
        # Resolve short ID to full ID (supports prefixes like '6cf21eb3')
        input_id = args.id
        if hasattr(tracer.storage, "resolve_call_id"):
            resolved = tracer.storage.resolve_call_id(input_id)
            if resolved:
                call_id = resolved
            else:
                fatal_error(
                    f"No call found matching '{input_id}'",
                    "Use more characters for a unique match",
                )
        else:
            call_id = input_id
    else:
        fatal_error("Must specify --id or --last")

    call = tracer.storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")

    # JSON output mode
    if output_format == "json":
        print(json.dumps(call.as_dict(), indent=2, default=str))
        return

    status = "ERROR" if call.error else "OK"

    def _detect_turns(inputs) -> tuple[str, int]:
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

    def _count_events(kinds: list[str]) -> int:
        return sum(1 for ev in call.trace if any(k in ev.kind.lower() for k in kinds))

    turn_label, turns = _detect_turns(call.inputs)
    llm_calls = _count_events(["gemini.request", "openai.request", "llm.request"])
    tool_events = _count_events(["tool"])

    print("\n================ Call Details ================")
    print(f"id       : {call.id}")
    print(f"function : {call.function_name}")
    print(f"status   : {status}")
    print(f"session  : {call.session_id}")
    print(f"started  : {call.started_at}")
    print(f"ended    : {call.ended_at}")
    print(f"duration : {call.duration_ms:.2f} ms")
    print(f"turns    : {turn_label} ({turns})")
    print(f"llm_calls: {llm_calls} | tool_events: {tool_events}")

    print("\nInputs:")
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

    if call.error:
        print("\nError:")
        print(call.error)
    else:
        print("\nOutput:")
        output_text = str(call.output or "")
        print(f"  type   : {type(call.output).__name__}")
        print(f"  length : {len(output_text)} chars")
        print(f"  preview: {_format_value(output_text, max_len=1000)}")

    if call.metadata:

        def _print_metadata(meta: dict) -> None:
            print("\nMetadata:")
            for key, value in meta.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_val in value.items():
                        if sub_key == "source":
                            src = sub_val or ""
                            src_preview = (
                                src if len(src) <= 1200 else src[:1200] + "..."
                            )
                            print("    source:")
                            for line in src_preview.splitlines()[:40]:
                                print(f"      {line}")
                        else:
                            print(
                                f"    - {sub_key}: {_format_value(sub_val, max_len=400)}"
                            )
                else:
                    print(f"  - {key}: {_format_value(value, max_len=400)}")

        _print_metadata(call.metadata)

    def _span_attr_summary(attrs, max_items=3):
        if not isinstance(attrs, dict) or not attrs:
            return ""
        preferred = [
            "model",
            "llm.model",
            "tool",
            "tool.name",
            "http.method",
            "http.url",
            "rpc.system",
        ]
        parts = []
        seen = set()

        def _add(key, value):
            if key in seen:
                return
            seen.add(key)
            if value is None:
                return
            text = str(value)
            text = text if len(text) <= 60 else text[:60] + "..."
            parts.append(f"{key}={text}")

        for key in preferred:
            if key in attrs:
                _add(key, attrs.get(key))
                if len(parts) >= max_items:
                    return " ".join(parts)

        for key in sorted(attrs.keys()):
            if key.startswith("evalyn."):
                continue
            _add(key, attrs.get(key))
            if len(parts) >= max_items:
                break
        return " ".join(parts)

    def _print_span_tree(spans, call_start_ts):
        by_id = {s.get("span_id"): s for s in spans if s.get("span_id")}
        children = {span_id: [] for span_id in by_id}
        for span in spans:
            span_id = span.get("span_id")
            parent_id = span.get("parent_span_id")
            if parent_id in by_id and span_id in by_id:
                children[parent_id].append(span_id)

        def _sort_key(span_id):
            span = by_id.get(span_id, {})
            return _normalize_span_time(span.get("start_time")) or 0.0

        for parent_id in children:
            children[parent_id].sort(key=_sort_key)

        roots = [
            sid
            for sid, span in by_id.items()
            if span.get("parent_span_id") not in by_id
        ]
        roots.sort(key=_sort_key)

        def _render(span_id, prefix, is_last):
            span = by_id[span_id]
            dur_ms = _span_duration_ms(span)
            status = _span_status(span)
            attr_summary = _span_attr_summary(span.get("attributes", {}))
            start_ts = _normalize_span_time(span.get("start_time"))
            rel_ms = None
            if start_ts is not None and call_start_ts is not None:
                rel_ms = (start_ts - call_start_ts) * 1000
            dur_text = f"{dur_ms:.1f}ms" if dur_ms is not None else "n/a"
            rel_text = f"+{rel_ms:.1f}ms" if rel_ms is not None else "n/a"
            branch = "`- " if is_last else "|- "
            line = (
                f"{prefix}{branch}{span.get('name')} ({status}, {dur_text}, {rel_text})"
            )
            if attr_summary:
                line += f" {attr_summary}"
            print(line)
            next_prefix = prefix + ("   " if is_last else "|  ")
            child_ids = children.get(span_id, [])
            for idx, child_id in enumerate(child_ids):
                _render(child_id, next_prefix, idx == len(child_ids) - 1)

        for idx, root_id in enumerate(roots):
            _render(root_id, "", idx == len(roots) - 1)

    if call.trace:

        def _format_time(ev):
            try:
                delta = (ev.timestamp - call.started_at).total_seconds()
                return f"+{delta:0.3f}s"
            except Exception:
                return str(ev.timestamp)

        def _truncate(text, max_len=120):
            if text is None:
                return ""
            text = str(text)
            return text if len(text) <= max_len else text[:max_len] + "..."

        def _format_inline(value, max_len=100):
            try:
                if isinstance(value, (dict, list)):
                    text = json.dumps(value, separators=(",", ":"))
                else:
                    text = str(value)
            except Exception:
                text = str(value)
            return text if len(text) <= max_len else text[:max_len] + "..."

        def _summarize_detail(detail, max_items=4):
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
                        for t in raw_tools:
                            if isinstance(t, dict):
                                tools.extend(list(t.keys()))
                            else:
                                tools.append(str(t))
                    if tools:
                        parts.append(f"tools={','.join(tools)}")
                    else:
                        parts.append(f"config_keys={list(cfg.keys())}")
                for key in (
                    "status",
                    "status_code",
                    "elapsed_ms",
                    "duration_ms",
                    "count",
                    "length",
                ):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_format_inline(detail.get(key), 40)}")
                for key in ("error", "url"):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_truncate(detail.get(key), 60)}")
                for key in ("contents", "messages", "prompt", "prompt_excerpt"):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_truncate(detail.get(key), 80)}")
            if not parts:
                parts.append(_truncate(_format_inline(detail, 120), 120))
            return " ".join(parts[:max_items])

        print("\nEvents summary:")
        total = len(call.trace)
        reqs = sum(1 for ev in call.trace if ev.kind.lower().endswith(".request"))
        resps = sum(1 for ev in call.trace if ev.kind.lower().endswith(".response"))
        tool_cnt = sum(1 for ev in call.trace if "tool" in ev.kind.lower())
        print(
            f"  total={total} | requests={reqs} | responses={resps} | tool_events={tool_cnt}"
        )
        kind_counts = {}
        for ev in call.trace:
            kind_counts[ev.kind] = kind_counts.get(ev.kind, 0) + 1
        for kind, count in sorted(kind_counts.items()):
            print(f"  - {kind}: {count}")

        print("\nEvents timeline:")
        header = ["idx", "t+ms", "delta_ms", "kind", "summary"]
        print(" | ".join(header))
        print("-" * 140)
        prev_ts = None
        for idx, ev in enumerate(call.trace, start=1):
            elapsed_ms = (
                (ev.timestamp - call.started_at).total_seconds() * 1000
                if call.started_at
                else 0.0
            )
            delta_ms = (
                (ev.timestamp - prev_ts).total_seconds() * 1000 if prev_ts else 0.0
            )
            summary = _summarize_detail(ev.detail or {})
            print(
                f"{idx} | {elapsed_ms:7.1f} | {delta_ms:7.1f} | {ev.kind} | {summary}"
            )
            prev_ts = ev.timestamp

    # Show hierarchical spans from call.spans
    if call.spans:

        def _tokens_info(attrs):
            input_t = attrs.get("input_tokens", 0)
            output_t = attrs.get("output_tokens", 0)
            if input_t or output_t:
                return f" [{input_t}>{output_t} tok]"
            return ""

        def _status_icon(status):
            if status == "error":
                return "X"
            elif status == "ok":
                return "V"
            return "o"

        # Build tree structure
        by_id = {s.id: {"span": s, "children": []} for s in call.spans}
        roots = []
        for s in call.spans:
            node = by_id[s.id]
            if s.parent_id and s.parent_id in by_id:
                by_id[s.parent_id]["children"].append(node)
            else:
                roots.append(node)

        # Sort children by start_time
        def sort_children(node):
            node["children"].sort(key=lambda n: n["span"].start_time or 0)
            for child in node["children"]:
                sort_children(child)

        for root in roots:
            sort_children(root)

        # Render tree
        def render_node(node, prefix="", is_last=True):
            span = node["span"]
            connector = "`-- " if is_last else "|-- "
            dur = _format_dur(span.duration_ms)
            tokens = _tokens_info(span.attributes or {})

            if span.span_type == "llm_call":
                label = f"llm: {span.name}{tokens} ({dur})"
            elif span.span_type == "tool_call":
                label = f"tool: {span.name} ({dur})"
            elif span.span_type == "node":
                label = f"node: {span.name.replace('node:', '')} ({dur})"
            elif span.span_type == "graph":
                label = f"graph: {span.name.replace('graph:', '')} ({dur})"
            else:
                label = f"{span.name} ({dur})"

            print(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            children = node["children"]
            for i, child in enumerate(children):
                render_node(child, child_prefix, i == len(children) - 1)

        print("\nSpan Tree:")
        for i, root in enumerate(roots):
            render_node(root, "", i == len(roots) - 1)

        # Summary
        llm_count = sum(1 for s in call.spans if s.span_type == "llm_call")
        tool_count = sum(1 for s in call.spans if s.span_type == "tool_call")
        node_count = sum(1 for s in call.spans if s.span_type == "node")
        print(
            f"\n  {len(call.spans)} spans | {llm_count} LLM | {tool_count} tools | {node_count} nodes"
        )

    print("=============================================\n")

    # Show hint to view span tree
    if call.spans:
        print_hint(
            f"To see span tree, run: evalyn show-trace --id {call.id[:8]}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_show_trace(args: argparse.Namespace) -> None:
    """Show hierarchical span tree for a traced call (Phoenix-style visualization)."""
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    # Handle --last flag or --id
    if getattr(args, "last", False):
        calls = tracer.storage.list_calls(limit=1)
        if not calls:
            fatal_error("No calls found")
        call_id = calls[0].id
    elif args.id:
        # Resolve short ID to full ID (supports prefixes like '6cf21eb3')
        input_id = args.id
        if hasattr(tracer.storage, "resolve_call_id"):
            resolved = tracer.storage.resolve_call_id(input_id)
            if resolved:
                call_id = resolved
            else:
                fatal_error(
                    f"No call found matching '{input_id}'",
                    "Use more characters for a unique match",
                )
        else:
            call_id = input_id
    else:
        fatal_error("Must specify --id or --last")

    call = tracer.storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")

    def _format_duration(ms: float) -> str:
        if ms is None:
            return "?"
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms / 1000:.1f}s"

    def _format_tokens(attrs: dict) -> str:
        """Format token info if present."""
        input_t = attrs.get("input_tokens", 0)
        output_t = attrs.get("output_tokens", 0)
        tool_t = attrs.get("tool_tokens", 0)
        if input_t or output_t:
            if tool_t:
                return f" [{input_t}→{output_t} +{tool_t} tool tokens]"
            return f" [{input_t}→{output_t} tokens]"
        return ""

    def _format_grounding(attrs: dict, prefix: str) -> list:
        """Format grounding metadata (search queries, sources) as extra lines."""
        lines = []
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

    def _status_icon(status: str) -> str:
        if status == "error":
            return "[X]"
        elif status == "ok":
            return "[OK]"
        return "[-]"

    # Build span tree from call.spans
    spans = call.spans or []
    if not spans:
        # Try to build from trace events (backwards compat)
        print(f"\nTrace: {call.function_name} ({_format_duration(call.duration_ms)})")
        print("  <no spans captured>")
        print("\n  Tip: Re-run with latest evalyn_sdk to capture spans.")

        # Show trace events as fallback
        if call.trace:
            print(f"\n  Trace Events ({len(call.trace)}):")
            for ev in call.trace[:20]:
                print(f"    - {ev.kind}")
            if len(call.trace) > 20:
                print(f"    ... and {len(call.trace) - 20} more")
        return

    # Build tree structure
    by_id = {s.id: {"span": s, "children": []} for s in spans}
    roots = []

    for s in spans:
        node = by_id[s.id]
        if s.parent_id and s.parent_id in by_id:
            by_id[s.parent_id]["children"].append(node)
        else:
            roots.append(node)

    # Sort children by start_time
    def sort_children(node):
        node["children"].sort(key=lambda n: n["span"].start_time)
        for child in node["children"]:
            sort_children(child)

    for root in roots:
        sort_children(root)

    # Render tree with optional depth limit
    max_depth = getattr(args, "max_depth", None)
    truncated_count = [0]  # Use list to allow mutation in nested function

    def render_node(node, prefix="", is_last=True, depth=0):
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            truncated_count[0] += 1 + _count_descendants(node)
            return

        span = node["span"]
        connector = "└── " if is_last else "├── "
        duration = _format_duration(span.duration_ms)
        status = _status_icon(span.status)
        tokens = _format_tokens(span.attributes)

        # Format name based on span type
        if span.span_type == "session":
            label = f"{span.name} ({duration})"
        elif span.span_type == "graph":
            label = f"graph:{span.name.replace('graph:', '')} ({duration})"
        elif span.span_type == "node":
            label = f"node:{span.name.replace('node:', '')} ({duration})"
        elif span.span_type == "llm_call":
            label = f"llm_call {span.name}{tokens} ({duration})"
        elif span.span_type == "tool_call":
            label = f"tool_call {span.name} ({duration})"
        elif span.span_type == "scorer":
            score = span.attributes.get("score", "?")
            label = f"{status} {span.name} (score: {score})"
        else:
            label = f"{span.name} ({duration})"

        print(f"{prefix}{connector}{label}")

        # Show grounding info for LLM calls with search/sources
        if span.span_type == "llm_call":
            grounding_lines = _format_grounding(span.attributes, prefix)
            for line in grounding_lines:
                print(line)

        # Render children
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = node["children"]
        for i, child in enumerate(children):
            render_node(child, child_prefix, i == len(children) - 1, depth + 1)

    def _count_descendants(node):
        count = len(node["children"])
        for child in node["children"]:
            count += _count_descendants(child)
        return count

    # Print header
    status = "ERROR" if call.error else "OK"
    print(
        f"\nTrace: {call.function_name} ({_format_duration(call.duration_ms)}) [{status}]"
    )
    print(f"Call ID: {call.id}")
    if call.session_id:
        print(f"Session: {call.session_id}")
    print()

    # Render all root spans
    for i, root in enumerate(roots):
        render_node(root, "", i == len(roots) - 1)

    # Summary
    llm_count = sum(1 for s in spans if s.span_type == "llm_call")
    tool_count = sum(1 for s in spans if s.span_type == "tool_call")
    node_count = sum(1 for s in spans if s.span_type == "node")

    summary_line = f"\nSummary: {len(spans)} spans | {llm_count} LLM calls | {tool_count} tool calls | {node_count} nodes"
    if truncated_count[0] > 0:
        summary_line += (
            f" | ({truncated_count[0]} spans hidden, use --max-depth to see more)"
        )
    print(summary_line)

    # Show hint to build dataset
    meta = call.metadata if isinstance(call.metadata, dict) else {}
    project = meta.get("project_id") or meta.get("project_name")
    if project:
        print_hint(
            f"To build a dataset, run: evalyn build-dataset --project {project}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_show_projects(args: argparse.Namespace) -> None:
    """Show summary of projects and their traces."""
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")
    calls = tracer.storage.list_calls(limit=args.limit)
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

    headers = ["project", "version", "calls", "errors", "first", "last"]
    print(" | ".join(headers))
    print("-" * 120)
    first_project = None
    for (project, version), rec in summary.items():
        if first_project is None:
            first_project = project
        row = [
            project,
            version,
            str(rec["total"]),
            str(rec["errors"]),
            str(rec["first"]),
            str(rec["last"]),
        ]
        print(" | ".join(row))

    # Show hint to list calls for a project
    if first_project:
        print_hint(
            f"To see calls, run: evalyn list-calls --project {first_project}",
            quiet=getattr(args, "quiet", False),
        )


def register_commands(subparsers) -> None:
    """Register trace-related commands."""
    # list-calls
    p = subparsers.add_parser("list-calls", help="List captured function calls")
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
    p.add_argument("--id", help="Call ID to show")
    p.add_argument("--last", action="store_true", help="Show the most recent call")
    p.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    p.set_defaults(func=cmd_show_call)

    # show-trace
    p = subparsers.add_parser("show-trace", help="Show hierarchical span tree")
    p.add_argument("--id", help="Call ID to show")
    p.add_argument("--last", action="store_true", help="Show the most recent call")
    p.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth of span tree to display (default: unlimited)",
    )
    p.set_defaults(func=cmd_show_trace)

    # show-projects
    p = subparsers.add_parser("show-projects", help="Show project summary")
    p.add_argument("--limit", type=int, default=1000, help="Max calls to scan")
    p.set_defaults(func=cmd_show_projects)


__all__ = [
    "cmd_list_calls",
    "cmd_show_call",
    "cmd_show_trace",
    "cmd_show_projects",
    "register_commands",
]
