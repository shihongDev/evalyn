from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class Spinner:
    """Simple CLI spinner for long-running operations."""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._running = False
        self._thread: threading.Thread | None = None
        self._chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _spin(self) -> None:
        idx = 0
        while self._running:
            char = self._chars[idx % len(self._chars)]
            sys.stderr.write(f"\r{char} {self.message}...")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.1)
        sys.stderr.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stderr.flush()

    def __enter__(self) -> "Spinner":
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

from .annotations import import_annotations
from .calibration import CalibrationEngine
from .datasets import load_dataset, save_dataset_with_meta, build_dataset_from_storage
from .decorators import get_default_tracer
from .metrics.objective import register_builtin_metrics
from .metrics.registry import MetricRegistry
from .metrics.templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .runner import EvalRunner
from .metrics.suggester import HeuristicSuggester, LLMSuggester, LLMRegistrySelector, TemplateSelector, render_selection_prompt_with_templates
from .tracing import EvalTracer
from .models import MetricSpec
from datetime import datetime

# Light bundles for quick manual selection (no LLM).
BUNDLES: dict[str, list[str]] = {
    "summarization": [
        "latency_ms",
        "token_overlap_f1",
        "rouge_l",
        "rouge_1",
        "hallucination_risk",
        "clarity_readability",
        "conciseness",
        "toxicity_safety",
    ],
    "orchestrator": [
        "latency_ms",
        "tool_call_count",
        "llm_call_count",
        "llm_error_rate",
        "tool_success_ratio",
        "hallucination_risk",
        "instruction_following",
        "policy_guardrails",
    ],
    "research-agent": [
        "latency_ms",
        "url_count",
        "hallucination_risk",
        "factual_consistency",
        "helpfulness_accuracy",
        "tool_success_ratio",
        "clarity_readability",
        "tone_alignment",
    ],
}


def _get_module_callables(module: Any) -> List[str]:
    """Get all callable names from a module for error suggestions."""
    import inspect
    callables = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if callable(obj) and not inspect.isclass(obj):
            callables.append(name)
    return sorted(callables)


def _suggest_similar(name: str, candidates: List[str], max_suggestions: int = 3) -> List[str]:
    """Find similar names using simple substring matching."""
    name_lower = name.lower()
    # Exact prefix match first
    prefix_matches = [c for c in candidates if c.lower().startswith(name_lower)]
    if prefix_matches:
        return prefix_matches[:max_suggestions]
    # Substring match
    substr_matches = [c for c in candidates if name_lower in c.lower() or c.lower() in name_lower]
    return substr_matches[:max_suggestions]


def _load_callable(target: str) -> Callable[..., Any]:
    """
    Load a function reference. Supports:
    - path/to/file.py:function_name   (preferred, no PYTHONPATH fuss)
    - module.path:function_name       (fallback)
    """
    if ":" not in target:
        raise ValueError(
            f"Target must be 'path/to/file.py:function' or 'module:function'\n"
            f"Got: {target}\n"
            f"Example: evalyn suggest-metrics --target example_agent/agent.py:run_agent"
        )

    left, func_name = target.split(":", 1)

    def _get_attr_with_suggestions(module: Any, name: str, module_path: str) -> Callable[..., Any]:
        """Get attribute with helpful error message if not found."""
        if hasattr(module, name):
            return getattr(module, name)

        available = _get_module_callables(module)
        similar = _suggest_similar(name, available)

        error_msg = f"Function '{name}' not found in {module_path}"
        if similar:
            error_msg += f"\n\nDid you mean one of these?\n"
            for s in similar:
                error_msg += f"  - {left}:{s}\n"
        elif available:
            error_msg += f"\n\nAvailable functions:\n"
            for fn in available[:10]:
                error_msg += f"  - {fn}\n"
            if len(available) > 10:
                error_msg += f"  ... and {len(available) - 10} more\n"
        raise AttributeError(error_msg)

    # Path-based load first
    if left.endswith(".py") or os.path.sep in left:
        path = os.path.abspath(left if left.endswith(".py") else left + ".py")
        if not os.path.isfile(path):
            raise ImportError(
                f"Cannot find file: {path}\n"
                f"Make sure the file path is correct and the file exists."
            )
        mod_name = os.path.splitext(os.path.basename(path))[0]
        # Ensure package imports inside the file can resolve (e.g., `from pkg.module import x`)
        pkg_dir = os.path.dirname(path)
        pkg_init = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(pkg_init):
            sys.path.insert(0, os.path.dirname(pkg_dir))
        else:
            sys.path.insert(0, pkg_dir)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from path: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(
                f"Failed to load module from {path}:\n{type(e).__name__}: {e}"
            ) from e
        return _get_attr_with_suggestions(module, func_name, path)

    # Fallback: dotted module import
    try:
        module = importlib.import_module(left)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Module '{left}' not found.\n"
            f"If using a file path, make sure it ends with .py\n"
            f"Example: evalyn suggest-metrics --target example_agent/agent.py:run_agent"
        ) from e
    return _get_attr_with_suggestions(module, func_name, left)


def _ollama_caller(model: str) -> Callable[[str], List[dict]]:
    def _call(prompt: str) -> List[dict]:
        if not shutil.which("ollama"):
            raise RuntimeError("ollama CLI not found. Install Ollama or choose --llm-mode api.")
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout.strip()
        try:
            return json.loads(text)
        except Exception:
            return []

    return _call


def _parse_json_array(text: str) -> List[dict]:
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _openai_caller(model: str, api_base: Optional[str] = None, api_key: Optional[str] = None) -> Callable[[str], List[dict]]:
    def _call(prompt: str) -> List[dict]:
        # Gemini shortcut using google-genai if model name starts with "gemini"
        if model.lower().startswith("gemini"):
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY for Gemini model.")
            try:
                from google.genai import Client  # type: ignore
            except Exception as exc:
                raise RuntimeError("google-genai package not installed. Install with: pip install google-genai") from exc

            try:
                client = Client(api_key=key)
                guard = (
                    "Return ONLY a JSON array of metric objects. "
                    'Each object: {"id": "metric_id", "config": {...}}. '
                    "No prose. If unsure, return [].\n\n"
                )
                full_prompt = guard + prompt
                resp = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config={"temperature": 0},
                )
                text = getattr(resp, "text", None) or ""
                parsed = _parse_json_array(text)
                if parsed:
                    return parsed
                raise RuntimeError(f"Gemini call returned non-JSON: {text[:200]}")
            except Exception as exc:
                raise RuntimeError(f"Gemini call failed: {exc}") from exc

        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package not installed. Install with extras: pip install -e \"sdk[llm]\"") from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=key, base_url=api_base) if (key or api_base) else openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        parsed = _parse_json_array(text)
        return parsed

    return _call


def _with_spinner(caller: Callable[[str], List[dict]], message: str = "Calling LLM") -> Callable[[str], List[dict]]:
    """Wrap a caller function with a spinner for visual feedback."""
    def _wrapped(prompt: str) -> List[dict]:
        with Spinner(message):
            return caller(prompt)
    return _wrapped


def _build_llm_caller(args: argparse.Namespace) -> Callable[[str], List[dict]]:
    model_name = getattr(args, "model", "LLM")
    spinner_msg = f"Querying {model_name}"
    if args.llm_mode == "local":
        return _with_spinner(_ollama_caller(args.model), spinner_msg)
    if args.llm_caller:
        return _with_spinner(_load_callable(args.llm_caller), spinner_msg)
    return _with_spinner(_openai_caller(args.model, api_base=args.api_base, api_key=args.api_key), spinner_msg)


def _extract_code_meta(tracer: EvalTracer, fn: Callable[..., Any]) -> Optional[dict]:
    """
    Try to reuse cached code metadata from the tracer; fall back to inspect if not present.
    """
    meta = getattr(tracer, "_function_meta_cache", {}).get(id(fn))  # type: ignore[attr-defined]
    if meta:
        return meta
    try:
        import inspect
        import hashlib

        source = inspect.getsource(fn)
        return {
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "doc": inspect.getdoc(fn),
            "signature": str(inspect.signature(fn)),
            "source": source,
            "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "file_path": inspect.getsourcefile(fn),
        }
    except Exception:
        return None


def cmd_list_calls(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    calls = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []
    if args.project and calls:
        filtered = []
        for c in calls:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            pid = meta.get("project_id") or meta.get("project_name")
            if pid == args.project:
                filtered.append(c)
        calls = filtered

    output_format = getattr(args, "format", "table")

    if not calls:
        if output_format == "json":
            print("[]")
        else:
            print("No calls found.")
        return

    # JSON output mode
    if output_format == "json":
        result = []
        for call in calls:
            code = call.metadata.get("code", {}) if isinstance(call.metadata, dict) else {}
            project = ""
            version = ""
            if isinstance(call.metadata, dict):
                project = call.metadata.get("project_id") or call.metadata.get("project_name") or ""
                version = call.metadata.get("version", "")
            result.append({
                "id": call.id,
                "function": call.function_name,
                "project": project,
                "version": version,
                "status": "ERROR" if call.error else "OK",
                "file": code.get("file_path") if isinstance(code, dict) else None,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_ms": call.duration_ms,
            })
        print(json.dumps(result, indent=2))
        return

    # Table output mode
    headers = ["id", "function", "project", "version", "status", "file", "start", "end", "duration_ms"]
    print(" | ".join(headers))
    print("-" * 120)

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
        if isinstance(call.metadata, dict):
            project = call.metadata.get("project_id") or call.metadata.get("project_name") or ""
            version = call.metadata.get("version", "")
        file_path = code.get("file_path") if isinstance(code, dict) else None
        row = [
            call.id,
            call.function_name,
            project,
            version,
            status,
            _short_path(file_path),
            str(call.started_at),
            str(call.ended_at),
            f"{call.duration_ms:.2f}",
        ]
        print(" | ".join(row))


def cmd_show_call(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    call = tracer.storage.get_call(args.id)
    if not call:
        print(f"No call found with id={args.id}")
        return
    status = "ERROR" if call.error else "OK"

    def _format_value(value, max_len=300):
        if isinstance(value, str):
            return value if len(value) <= max_len else value[:max_len] + "..."
        try:
            text = json.dumps(value, indent=2)
            return text if len(text) <= max_len else text[:max_len] + "..."
        except Exception:
            text = str(value)
            return text if len(text) <= max_len else text[:max_len] + "..."

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
    args = call.inputs.get("args", [])
    kwargs = call.inputs.get("kwargs", {})
    if args:
        print("  args:")
        for idx, arg in enumerate(args):
            print(f"    - [{idx}] {_format_value(arg)}")
    if kwargs:
        print("  kwargs:")
        for key, value in kwargs.items():
            print(f"    - {key}: {_format_value(value)}")
    if not args and not kwargs:
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
                            src_preview = src if len(src) <= 1200 else src[:1200] + "..."
                            print("    source:")
                            for line in src_preview.splitlines()[:40]:
                                print(f"      {line}")
                        else:
                            print(f"    - {sub_key}: {_format_value(sub_val, max_len=400)}")
                else:
                    print(f"  - {key}: {_format_value(value, max_len=400)}")

        _print_metadata(call.metadata)

    def _normalize_span_time(raw):
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

    def _span_duration_ms(span):
        start_ts = _normalize_span_time(span.get("start_time"))
        end_ts = _normalize_span_time(span.get("end_time"))
        if start_ts is None or end_ts is None:
            return None
        return max(0.0, (end_ts - start_ts) * 1000)

    def _span_status(span):
        status = span.get("status")
        if status is None:
            return "UNSET"
        text = str(status).upper()
        if "ERROR" in text:
            return "ERROR"
        if "OK" in text:
            return "OK"
        return text

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

        roots = [sid for sid, span in by_id.items() if span.get("parent_span_id") not in by_id]
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
            line = f"{prefix}{branch}{span.get('name')} ({status}, {dur_text}, {rel_text})"
            if attr_summary:
                line += f" {attr_summary}"
            print(line)
            next_prefix = prefix + ("   " if is_last else "|  ")
            child_ids = children.get(span_id, [])
            for idx, child_id in enumerate(child_ids):
                _render(child_id, next_prefix, idx == len(child_ids) - 1)

        for idx, root_id in enumerate(roots):
            _render(root_id, "", idx == len(roots) - 1)

    spans = tracer.storage.list_spans(call.id) if tracer.storage else []
    if spans:
        print("\nSpan tree (OTel):")
        call_start_ts = call.started_at.timestamp() if call.started_at else None
        _print_span_tree(spans, call_start_ts)

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
                for key in ("status", "status_code", "elapsed_ms", "duration_ms", "count", "length"):
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
        print(f"  total={total} | requests={reqs} | responses={resps} | tool_events={tool_cnt}")
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
            elapsed_ms = (ev.timestamp - call.started_at).total_seconds() * 1000 if call.started_at else 0.0
            delta_ms = (
                (ev.timestamp - prev_ts).total_seconds() * 1000 if prev_ts else 0.0
            )
            summary = _summarize_detail(ev.detail or {})
            print(f"{idx} | {elapsed_ms:7.1f} | {delta_ms:7.1f} | {ev.kind} | {summary}")
            prev_ts = ev.timestamp

    if not spans:
        print("\nSpan tree (OTel):")
        print("  <no spans found>")
        print("  Tip: set EVALYN_OTEL_EXPORTER=sqlite and re-run to capture span data locally.")

    print("\nOTel:")
    print(" Spans are emitted alongside this call (exporter-configured).")
    print("=============================================\n")


def _resolve_dataset_and_metrics(
    dataset_arg: str,
    metrics_arg: Optional[str],
    metrics_all: bool = False
) -> tuple[Path, List[Path]]:
    """
    Resolve dataset file and metrics file paths.

    Args:
        dataset_arg: Path to dataset file or directory
        metrics_arg: Comma-separated paths to metrics files, or None for auto-detect
        metrics_all: If True, use all metrics files in the metrics/ folder

    Returns:
        Tuple of (dataset_file, list_of_metrics_paths)
    """
    dataset_path = Path(dataset_arg)

    # If dataset is a directory, look for dataset.jsonl inside
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        dataset_file = dataset_dir / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_dir / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"No dataset.jsonl or dataset.json found in {dataset_dir}")
    else:
        dataset_file = dataset_path
        dataset_dir = dataset_path.parent

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    metrics_paths: List[Path] = []

    # Option 1: Use all metrics from metrics/ folder
    if metrics_all:
        metrics_dir = dataset_dir / "metrics"
        if metrics_dir.exists():
            for json_file in sorted(metrics_dir.glob("*.json")):
                metrics_paths.append(json_file)
        if not metrics_paths:
            raise FileNotFoundError(f"No metrics files found in {metrics_dir}")
        print(f"Using all metrics files: {len(metrics_paths)} files from {metrics_dir}")
        return dataset_file, metrics_paths

    # Option 2: Explicit metrics argument (supports comma-separated paths)
    if metrics_arg:
        for path_str in metrics_arg.split(","):
            path_str = path_str.strip()
            if path_str:
                metrics_path = Path(path_str)
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
                metrics_paths.append(metrics_path)
        if metrics_paths:
            return dataset_file, metrics_paths

    # Option 3: Auto-detect from meta.json
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No --metrics specified and no meta.json found in {dataset_dir}.\n"
            "Either specify --metrics explicitly or run 'evalyn suggest-metrics' first."
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse meta.json: {e}")

    active_set = meta.get("active_metric_set")
    if not active_set:
        raise ValueError(
            f"No active_metric_set in meta.json. Run 'evalyn suggest-metrics' to select metrics."
        )

    # Find the metrics file from metric_sets
    metric_sets = meta.get("metric_sets", [])
    matching = [m for m in metric_sets if m.get("name") == active_set]
    if not matching:
        raise ValueError(f"Metric set '{active_set}' not found in meta.json metric_sets.")

    metrics_rel = matching[0].get("file")
    if not metrics_rel:
        raise ValueError(f"No file path for metric set '{active_set}' in meta.json.")

    metrics_path = dataset_dir / metrics_rel
    print(f"Auto-detected metrics: {metrics_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    return dataset_file, [metrics_path]


class ProgressBar:
    """Simple progress bar for evaluation."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self._current = 0
        self._current_metric = ""
        self._current_type = ""

    def update(self, current: int, total: int, metric: str, metric_type: str) -> None:
        self._current = current
        self._current_metric = metric
        self._current_type = metric_type
        self._render()

    def _render(self) -> None:
        pct = self._current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)
        type_label = "[obj]" if self._current_type == "objective" else "[llm]"
        line = f"\r[{bar}] {self._current}/{self.total} {type_label} {self._current_metric[:20]:<20}"
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()


def cmd_run_eval(args: argparse.Namespace) -> None:
    """Run evaluation using pre-computed traces from dataset and metrics from JSON file(s)."""
    output_format = getattr(args, "format", "table")
    metrics_all = getattr(args, "metrics_all", False)

    # Resolve dataset and metrics paths
    try:
        dataset_file, metrics_paths = _resolve_dataset_and_metrics(
            args.dataset, args.metrics, metrics_all=metrics_all
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(str(dataset_file))
    dataset_list = list(dataset)  # Convert to list for counting

    # Load and merge metrics from all files (deduplicate by ID)
    all_metrics_data: Dict[str, dict] = {}  # id -> spec_data
    for metrics_path in metrics_paths:
        try:
            file_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(file_data, list):
                if output_format != "json":
                    print(f"Warning: Skipping {metrics_path} - not a JSON array")
                continue
            for spec_data in file_data:
                metric_id = spec_data.get("id")
                if metric_id and metric_id not in all_metrics_data:
                    all_metrics_data[metric_id] = spec_data
        except Exception as e:
            if output_format != "json":
                print(f"Warning: Failed to load {metrics_path}: {e}")

    if not all_metrics_data:
        print("Error: No valid metrics loaded from files")
        sys.exit(1)

    if output_format != "json" and len(metrics_paths) > 1:
        print(f"Merged {len(all_metrics_data)} unique metrics from {len(metrics_paths)} files")

    # Build metrics from specs
    from .metrics.factory import build_objective_metric, build_subjective_metric
    metrics = []
    objective_count = 0
    subjective_count = 0

    skipped_metrics = []
    for spec_data in all_metrics_data.values():
        spec = MetricSpec(
            id=spec_data["id"],
            name=spec_data.get("name", spec_data["id"]),
            type=spec_data["type"],
            description=spec_data.get("description", ""),
            config=spec_data.get("config", {}),
        )
        try:
            if spec.type == "objective":
                metric = build_objective_metric(spec.id, spec.config)
                objective_count += 1
            else:
                # Pass description for custom/brainstormed metrics
                metric = build_subjective_metric(
                    spec.id,
                    spec.config,
                    description=spec.description,
                )
                subjective_count += 1
            if metric:
                metrics.append(metric)
        except KeyError as e:
            # Unknown metric ID - skip with warning
            skipped_metrics.append((spec.id, spec.type, str(e)))
        except Exception as e:
            if output_format != "json":
                print(f"Warning: Failed to build metric '{spec.id}': {e}")

    if skipped_metrics and output_format != "json":
        print(f"Skipped {len(skipped_metrics)} unknown metrics:")
        for mid, mtype, reason in skipped_metrics:
            if mtype == "objective":
                print(f"  - {mid} [objective]: Custom objective metrics not supported. Use 'evalyn list-metrics' to see available templates.")
            else:
                print(f"  - {mid}: {reason}")

    if not metrics:
        print("Error: No valid metrics loaded from file")
        sys.exit(1)

    if output_format != "json":
        print(f"Loaded {len(metrics)} metrics ({objective_count} objective, {subjective_count} subjective)")
        print(f"Dataset: {len(dataset_list)} items")

        # Check for API key if subjective metrics are present
        if subjective_count > 0:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if not gemini_key and not openai_key:
                print()
                print("⚠️  Warning: No API key found for LLM judges.")
                print("   Set GEMINI_API_KEY or OPENAI_API_KEY to enable subjective metrics.")
                print("   Continuing anyway, but LLM judge scores will fail.")
            elif gemini_key and len(gemini_key) < 10:
                print()
                print("⚠️  Warning: GEMINI_API_KEY appears to be invalid (too short).")

        print()

    # Get tracer with storage
    tracer = get_default_tracer()
    if not tracer.storage:
        print("Error: No storage configured")
        sys.exit(1)

    # Create progress bar
    total_evals = len(dataset_list) * len(metrics)
    progress = ProgressBar(total_evals) if output_format != "json" else None

    def progress_callback(current: int, total: int, metric: str, metric_type: str) -> None:
        if progress:
            progress.update(current, total, metric, metric_type)

    # Run evaluation using cached traces (with synthetic call support)
    from .runner import EvalRunner, save_eval_run_json
    runner = EvalRunner(
        target_fn=lambda: None,  # Dummy function, won't be called
        metrics=metrics,
        dataset_name=args.dataset_name or dataset_file.stem,
        tracer=tracer,
        instrument=False,  # Don't instrument since we're not calling the function
        progress_callback=progress_callback if output_format != "json" else None,
    )
    run = runner.run_dataset(dataset_list, use_synthetic=True)

    if progress:
        progress.finish()

    # Save eval run as JSON in dataset folder
    dataset_dir = dataset_file.parent
    run_json_path = save_eval_run_json(run, dataset_dir)

    # JSON output
    if output_format == "json":
        result = {
            "id": run.id,
            "dataset_name": run.dataset_name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "summary": run.summary,
            "run_file": str(run_json_path),
            "metric_results": [
                {
                    "metric_id": r.metric_id,
                    "item_id": r.item_id,
                    "call_id": r.call_id,
                    "score": r.score,
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in run.metric_results
            ],
        }
        print(json.dumps(result, indent=2))
        return

    # Table output
    print(f"\nEval run {run.id}")
    print(f"Dataset: {run.dataset_name}")
    print(f"Run saved to: {run_json_path}")
    print()

    # Build metric type lookup
    metric_types = {m.spec.id: m.spec.type for m in metrics}

    # Check for API errors in results
    api_errors_by_metric: Dict[str, int] = {}
    for result in run.metric_results:
        if result.details and isinstance(result.details, dict):
            reason = result.details.get("reason", "")
            if "API" in reason and ("error" in reason.lower() or "failed" in reason.lower()):
                api_errors_by_metric[result.metric_id] = api_errors_by_metric.get(result.metric_id, 0) + 1

    print("Results:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Type':<10} {'Count':<6} {'Avg Score':<12} {'Pass Rate':<10} {'Errors':<8}")
    print("-" * 80)

    for metric_id, stats in run.summary.get("metrics", {}).items():
        mtype = metric_types.get(metric_id, "?")
        type_label = "obj" if mtype == "objective" else "llm"
        avg_score = stats.get('avg_score')
        avg_score_str = f"{avg_score:.4f}" if avg_score is not None else "N/A"
        pass_rate = stats.get('pass_rate')
        pass_rate_str = f"{pass_rate*100:.1f}%" if pass_rate is not None else "N/A"
        error_count = api_errors_by_metric.get(metric_id, 0)
        error_str = f"{error_count}" if error_count > 0 else "-"
        print(f"{metric_id:<25} [{type_label:<3}]    {stats['count']:<6} {avg_score_str:<12} {pass_rate_str:<10} {error_str:<8}")

    print("-" * 80)

    # Show API error summary if any
    total_api_errors = sum(api_errors_by_metric.values())
    if total_api_errors > 0:
        print(f"\n⚠️  {total_api_errors} API error(s) detected in LLM judge results.")
        print("   Check that GEMINI_API_KEY or OPENAI_API_KEY is set and valid.")
        print("   Re-run with a valid API key to get accurate LLM judge scores.")

    if run.summary.get("failed_items"):
        print(f"Failed items: {run.summary['failed_items']}")


def _dataset_has_reference(dataset_path: Optional[Path]) -> bool:
    """
    Check if a dataset has SEPARATE reference/golden-standard values for comparison.

    Reference-based metrics (ROUGE, BLEU, token_overlap) need TWO text values:
    1. output - what the model produced
    2. reference - what a human says is the correct answer (golden standard)

    This function checks for explicit reference fields that are SEPARATE from output:
    - human_label.reference (new format)
    - metadata.reference or metadata.golden (explicit reference)

    NOTE: The old 'expected' field often contains the model output (not a separate reference),
    so we don't count it unless there's also an 'output' field (indicating expected is actually
    the golden standard).

    Returns True if at least one item has a separate reference value, False otherwise.
    """
    if not dataset_path:
        return False

    # Find the actual dataset file
    if dataset_path.is_file():
        dataset_file = dataset_path
    else:
        if (dataset_path / "dataset.jsonl").exists():
            dataset_file = dataset_path / "dataset.jsonl"
        elif (dataset_path / "dataset.json").exists():
            dataset_file = dataset_path / "dataset.json"
        else:
            return False

    try:
        items = load_dataset(str(dataset_file))
        for item in items:
            # Check for human_label with reference (this is the clear signal)
            if hasattr(item, 'human_label') and item.human_label:
                if isinstance(item.human_label, dict) and item.human_label.get('reference'):
                    return True

            # Check metadata for explicit reference/golden fields
            if hasattr(item, 'metadata') and item.metadata:
                if item.metadata.get('reference') or item.metadata.get('golden') or item.metadata.get('golden_answer'):
                    return True

            # If BOTH output AND expected exist AND they are DIFFERENT, then expected is the golden standard
            has_output = hasattr(item, 'output') and item.output is not None
            has_expected = hasattr(item, 'expected') and item.expected is not None
            if has_output and has_expected:
                # Only count as reference if they're actually different values
                # (if same, expected is just a copy of output for backward compatibility)
                if item.output != item.expected:
                    return True

        return False
    except Exception:
        return False


def cmd_suggest_metrics(args: argparse.Namespace) -> None:
    # Validate dataset path FIRST before doing any expensive work
    dataset_path_obj: Optional[Path] = None
    if args.dataset:
        dataset_path_obj = Path(args.dataset)
        if dataset_path_obj.is_file():
            # It's a file (e.g., dataset.jsonl)
            if not dataset_path_obj.exists():
                print(f"Error: Dataset file not found: {dataset_path_obj}")
                print("Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.")
                sys.exit(1)
        else:
            # It's a directory - should exist and contain a dataset
            if not dataset_path_obj.exists():
                print(f"Error: Dataset directory not found: {dataset_path_obj}")
                print("Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.")
                sys.exit(1)
            # Check if directory has a dataset file
            has_dataset = (dataset_path_obj / "dataset.jsonl").exists() or (dataset_path_obj / "dataset.json").exists()
            if not has_dataset:
                print(f"Warning: Directory exists but no dataset.jsonl or dataset.json found in: {dataset_path_obj}")
                print("Proceeding anyway, but this may not be a valid dataset directory.")

    # Check if dataset has reference values
    has_reference = _dataset_has_reference(dataset_path_obj)
    if dataset_path_obj and not has_reference:
        print("Note: Dataset has no reference/expected values. Reference-based metrics (ROUGE, BLEU, etc.) excluded.")

    target_fn = _load_callable(args.target)
    metric_mode_hint = getattr(target_fn, "_evalyn_metric_mode", None)
    metric_bundle_hint = getattr(target_fn, "_evalyn_metric_bundle", None)
    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.num_traces) if tracer.storage else []

    selected_mode = args.mode
    if selected_mode == "auto":
        selected_mode = metric_mode_hint or "llm-registry"
    bundle_name = args.bundle or metric_bundle_hint
    max_metrics = args.num_metrics if args.num_metrics and args.num_metrics > 0 else None

    def _print_spec(spec: MetricSpec) -> None:
        why = getattr(spec, "why", "") or ""
        suffix = f" | why: {why}" if why else ""
        print(f"- {spec.id} [{spec.type}] :: {spec.description}{suffix}")

    def _save_metrics(specs: List[MetricSpec]) -> None:
        if not args.dataset:
            return
        # Dataset path already validated at the beginning of cmd_suggest_metrics
        dataset_path = Path(args.dataset)
        if dataset_path.is_file():
            dataset_dir = dataset_path.parent
        else:
            dataset_dir = dataset_path

        metrics_dir = dataset_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        if args.metrics_name:
            metrics_name = args.metrics_name
        elif selected_mode == "bundle":
            metrics_name = f"bundle-{(bundle_name or 'bundle')}"
        else:
            metrics_name = f"{selected_mode}-{ts}"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", metrics_name).strip("-")
        metrics_file = metrics_dir / f"{safe_name}.json"

        payload = []
        for spec in specs:
            payload.append(
                {
                    "id": spec.id,
                    "type": spec.type,
                    "description": spec.description,
                    "config": spec.config,
                    "why": getattr(spec, "why", ""),
                }
            )
        metrics_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

        meta_path = dataset_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        else:
            meta = {}
        metric_sets = meta.get("metric_sets") if isinstance(meta.get("metric_sets"), list) else []
        entry = {
            "name": safe_name,
            "file": f"metrics/{metrics_file.name}",
            "mode": selected_mode,
            "created_at": datetime.utcnow().isoformat(),
            "num_metrics": len(payload),
        }
        metric_sets = [m for m in metric_sets if m.get("name") != safe_name]
        metric_sets.append(entry)
        meta["metric_sets"] = metric_sets
        meta["active_metric_set"] = safe_name
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Saved metrics to {metrics_file}")

    if selected_mode == "bundle":
        bundle = (bundle_name or "").lower()
        ids = BUNDLES.get(bundle)
        if not ids:
            print(f"Unknown bundle '{args.bundle}'. Available: {', '.join(BUNDLES.keys())}")
            return
        tpl_map = {t["id"]: t for t in OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES}
        specs = []
        skipped_ref_metrics = []
        for mid in ids:
            tpl = tpl_map.get(mid)
            if tpl:
                # Skip reference-based metrics if no reference available
                if tpl.get("requires_reference", False) and not has_reference:
                    skipped_ref_metrics.append(mid)
                    continue
                specs.append(
                    MetricSpec(
                        id=tpl["id"],
                        name=tpl["id"],
                        type=tpl["type"],
                        description=tpl.get("description", ""),
                        config=tpl.get("config", {}),
                    )
                )
        if skipped_ref_metrics:
            print(f"Skipped reference-based metrics (no expected values): {', '.join(skipped_ref_metrics)}")
        if max_metrics:
            specs = specs[:max_metrics]
        for spec in specs:
            _print_spec(spec)
        _save_metrics(specs)
        return

    if selected_mode == "llm-registry":
        caller = _build_llm_caller(args)
        selector = TemplateSelector(caller, OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES, has_reference=has_reference)
        selected = selector.select(
            target_fn,
            traces=traces,
            code_meta=_extract_code_meta(tracer, target_fn),
            desired_count=max_metrics,
        )
        specs = selected
        if max_metrics:
            specs = specs[:max_metrics]
        for spec in specs:
            _print_spec(spec)
        _save_metrics(specs)
        return

    if selected_mode == "llm-brainstorm":
        caller = _build_llm_caller(args)
        suggester = LLMSuggester(caller=caller)
        specs = suggester.suggest(target_fn, traces, desired_count=max_metrics)
        if max_metrics:
            specs = specs[:max_metrics]
        if not specs:
            print("No metrics were returned by the LLM (brainstorm mode).")
        else:
            for spec in specs:
                _print_spec(spec)
            _save_metrics(specs)
        return

    suggester = HeuristicSuggester(has_reference=has_reference)
    specs = suggester.suggest(target_fn, traces)
    if max_metrics:
        specs = specs[:max_metrics]
    for spec in specs:
        _print_spec(spec)
    _save_metrics(specs)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return

    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    items = build_dataset_from_storage(
        tracer.storage,
        function_name=None,  # prefer project-based grouping
        project_id=args.project,
        project_name=args.project,
        version=args.version,
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
        limit=args.limit,
        success_only=not args.include_errors,
        include_metadata=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    proj = args.project or "all"
    ver = args.version or "v0"
    dataset_name = f"{proj}-{ver}-{ts}"

    dataset_dir = args.output or os.path.join("data", dataset_name)
    dataset_file = "dataset.jsonl"
    if args.output and args.output.endswith(".jsonl"):
        dataset_dir = os.path.dirname(args.output) or "."
        dataset_file = os.path.basename(args.output)

    functions = sorted(
        {item.metadata.get("function") for item in items if isinstance(item.metadata, dict) and item.metadata.get("function")}
    )
    function_name = functions[0] if len(functions) == 1 else None

    input_keys = set()
    meta_keys = set()
    for item in items:
        if isinstance(item.inputs, dict):
            input_keys.update(item.inputs.keys())
        if isinstance(item.metadata, dict):
            meta_keys.update(item.metadata.keys())

    meta = {
        "dataset_name": dataset_name,
        "created_at": datetime.utcnow().isoformat(),
        "project": args.project,
        "version": args.version,
        "function": function_name,
        "filters": {
            "since": args.since,
            "until": args.until,
            "limit": args.limit,
            "include_errors": bool(args.include_errors),
        },
        "counts": {"items": len(items)},
        "schema": {"inputs_keys": sorted(input_keys), "metadata_keys": sorted(meta_keys)},
    }

    dataset_path = save_dataset_with_meta(items, dataset_dir, meta, dataset_filename=dataset_file)
    print(f"Wrote {len(items)} items to {dataset_path}")


def cmd_show_projects(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    calls = tracer.storage.list_calls(limit=args.limit)
    summary = {}
    for call in calls:
        meta = call.metadata if isinstance(call.metadata, dict) else {}
        project = meta.get("project_id") or meta.get("project_name") or call.function_name or "unknown"
        version = meta.get("version") or ""
        key = (project, version)
        rec = summary.setdefault(key, {"total": 0, "errors": 0, "first": call.started_at, "last": call.started_at})
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
    for (project, version), rec in summary.items():
        row = [
            project,
            version,
            str(rec["total"]),
            str(rec["errors"]),
            str(rec["first"]),
            str(rec["last"]),
        ]
        print(" | ".join(row))

def cmd_list_runs(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    runs = tracer.storage.list_eval_runs(limit=args.limit)
    output_format = getattr(args, "format", "table")

    if not runs:
        if output_format == "json":
            print("[]")
        else:
            print("No eval runs found.")
        return

    if output_format == "json":
        result = []
        for run in runs:
            result.append({
                "id": run.id,
                "dataset_name": run.dataset_name,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "metrics_count": len(run.metrics),
                "results_count": len(run.metric_results),
                "summary": run.summary,
            })
        print(json.dumps(result, indent=2))
        return

    headers = ["id", "dataset", "created_at", "metrics", "results"]
    print(" | ".join(headers))
    print("-" * 120)
    for run in runs:
        row = [
            run.id,
            run.dataset_name,
            str(run.created_at),
            str(len(run.metrics)),
            str(len(run.metric_results)),
        ]
        print(" | ".join(row))


def cmd_show_run(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    run = tracer.storage.get_eval_run(args.id)
    if not run:
        print(f"No eval run found with id={args.id}")
        return
    print(f"\n=== Eval Run {run.id} ===")
    print(f"Dataset: {run.dataset_name}")
    print("Metrics summary:")
    for mid, stats in run.summary.get("metrics", {}).items():
        print(
            f" - {mid:<18} count={stats['count']:<3} "
            f"avg_score={stats['avg_score']!s:<10} pass_rate={stats['pass_rate']!s:<10}"
        )
    if run.summary.get("failed_items"):
        print(f"Failed items: {run.summary['failed_items']}")
    print("\nMetric results:")
    for res in run.metric_results:
        print(
            f" [{res.metric_id}] item={res.item_id} call={res.call_id} "
            f"score={res.score} passed={res.passed} details={res.details}"
        )


def cmd_import_annotations(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    anns = import_annotations(args.path)
    tracer.storage.store_annotations(anns)
    print(f"Imported {len(anns)} annotations into storage.")


def cmd_export_for_annotation(args: argparse.Namespace) -> None:
    """Export dataset items with eval results for human annotation."""
    from .models import AnnotationItem, HumanLabel

    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        dataset_file = dataset_path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_path / "dataset.json"
    else:
        dataset_file = dataset_path

    if not dataset_file.exists():
        print(f"Error: Dataset not found: {dataset_file}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(str(dataset_file))
    dataset_list = list(dataset)

    # Get eval run (if specified or use latest)
    run = None
    if args.run_id:
        run = tracer.storage.get_eval_run(args.run_id)
        if not run:
            print(f"Warning: Eval run '{args.run_id}' not found. Exporting without eval results.")
    else:
        # Try to get the latest run
        runs = tracer.storage.list_eval_runs(limit=1)
        if runs:
            run = runs[0]
            print(f"Using latest eval run: {run.id}")

    # Build item_id -> eval_results mapping
    eval_results_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if run:
        for result in run.metric_results:
            item_id = result.item_id
            if item_id not in eval_results_map:
                eval_results_map[item_id] = {}
            eval_results_map[item_id][result.metric_id] = {
                "passed": result.passed,
                "score": result.score,
                "reason": result.details.get("reason", ""),
            }

    # Build annotation items
    annotation_items = []
    for item in dataset_list:
        eval_results = eval_results_map.get(item.id, {})

        # Check for existing human_label
        human_label = None
        if item.human_label:
            if isinstance(item.human_label, dict):
                human_label = HumanLabel.from_dict(item.human_label)
            else:
                human_label = item.human_label

        ann_item = AnnotationItem(
            id=item.id,
            input=item.input,
            output=item.output,
            eval_results=eval_results,
            human_label=human_label,
            metadata=item.metadata,
        )
        annotation_items.append(ann_item)

    # Write output
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for ann_item in annotation_items:
            f.write(json.dumps(ann_item.as_dict(), ensure_ascii=False) + "\n")

    # Summary stats
    with_evals = sum(1 for item in annotation_items if item.eval_results)
    with_labels = sum(1 for item in annotation_items if item.human_label)

    print(f"Exported {len(annotation_items)} items to {output_path}")
    print(f"  - With eval results: {with_evals}")
    print(f"  - With human labels: {with_labels}")
    print(f"  - Awaiting annotation: {len(annotation_items) - with_labels}")


def cmd_annotation_stats(args: argparse.Namespace) -> None:
    """Show annotation coverage statistics."""
    from .models import AnnotationItem, HumanLabel

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        # Look for annotation file or dataset file
        ann_file = dataset_path / "annotations.jsonl"
        if ann_file.exists():
            data_file = ann_file
        else:
            data_file = dataset_path / "dataset.jsonl"
            if not data_file.exists():
                data_file = dataset_path / "dataset.json"
    else:
        data_file = dataset_path

    if not data_file.exists():
        print(f"Error: File not found: {data_file}")
        sys.exit(1)

    # Load data
    items = []
    raw = data_file.read_text(encoding="utf-8").strip()
    if not raw:
        print("Empty file.")
        return

    for line in raw.splitlines():
        if line.strip():
            data = json.loads(line)
            items.append(AnnotationItem.from_dict(data))

    if not items:
        print("No items found.")
        return

    # Calculate stats
    total = len(items)
    with_labels = sum(1 for item in items if item.human_label)
    with_evals = sum(1 for item in items if item.eval_results)
    coverage = with_labels / total if total > 0 else 0

    # Per-metric stats
    metric_stats: Dict[str, Dict[str, int]] = {}
    for item in items:
        for metric_id, result in item.eval_results.items():
            if metric_id not in metric_stats:
                metric_stats[metric_id] = {"total": 0, "passed": 0, "failed": 0}
            metric_stats[metric_id]["total"] += 1
            if result.get("passed") is True:
                metric_stats[metric_id]["passed"] += 1
            elif result.get("passed") is False:
                metric_stats[metric_id]["failed"] += 1

    # Human agreement with LLM (if we have both)
    agreement_stats: Dict[str, Dict[str, int]] = {}
    for item in items:
        if not item.human_label or not item.eval_results:
            continue
        human_passed = item.human_label.passed
        for metric_id, result in item.eval_results.items():
            llm_passed = result.get("passed")
            if llm_passed is None:
                continue
            if metric_id not in agreement_stats:
                agreement_stats[metric_id] = {"agree": 0, "disagree": 0, "fp": 0, "fn": 0}
            if human_passed == llm_passed:
                agreement_stats[metric_id]["agree"] += 1
            else:
                agreement_stats[metric_id]["disagree"] += 1
                if llm_passed and not human_passed:
                    agreement_stats[metric_id]["fp"] += 1  # LLM says pass, human says fail
                else:
                    agreement_stats[metric_id]["fn"] += 1  # LLM says fail, human says pass

    # Print report
    print("\n" + "=" * 60)
    print("ANNOTATION COVERAGE REPORT")
    print("=" * 60)
    print(f"\nTotal items:        {total}")
    print(f"With human labels:  {with_labels} ({coverage*100:.1f}%)")
    print(f"With eval results:  {with_evals}")
    print(f"Awaiting annotation: {total - with_labels}")

    if metric_stats:
        print("\n" + "-" * 60)
        print("EVAL RESULTS BY METRIC")
        print("-" * 60)
        print(f"{'Metric':<25} {'Total':<8} {'Pass':<8} {'Fail':<8} {'Pass %':<8}")
        print("-" * 60)
        for metric_id, stats in sorted(metric_stats.items()):
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{metric_id:<25} {stats['total']:<8} {stats['passed']:<8} {stats['failed']:<8} {pass_rate*100:.1f}%")

    if agreement_stats:
        print("\n" + "-" * 60)
        print("HUMAN vs LLM AGREEMENT")
        print("-" * 60)
        print(f"{'Metric':<25} {'Agree':<8} {'Disagree':<8} {'FP':<8} {'FN':<8} {'Agr %':<8}")
        print("-" * 60)
        for metric_id, stats in sorted(agreement_stats.items()):
            total_compared = stats["agree"] + stats["disagree"]
            agr_rate = stats["agree"] / total_compared if total_compared > 0 else 0
            print(f"{metric_id:<25} {stats['agree']:<8} {stats['disagree']:<8} {stats['fp']:<8} {stats['fn']:<8} {agr_rate*100:.1f}%")
        print("\nFP = False Positive (LLM=PASS, Human=FAIL)")
        print("FN = False Negative (LLM=FAIL, Human=PASS)")

    print("\n" + "=" * 60)


def cmd_calibrate(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return

    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        print("No eval runs available.")
        return

    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        print(f"No metric results found for metric_id={args.metric_id} in run {run.id}")
        return

    anns = import_annotations(args.annotations)
    engine = CalibrationEngine(judge_name=args.metric_id, current_threshold=args.threshold)
    record = engine.calibrate(metric_results, anns)

    print(f"Calibration for metric '{args.metric_id}' on run {run.id}")
    print(f"- disagreement_rate: {record.adjustments['disagreement_rate']:.3f}")
    print(f"- suggested_threshold: {record.adjustments['suggested_threshold']:.3f}")
    print(f"- current_threshold: {record.adjustments['current_threshold']:.3f}")


def cmd_select_metrics(args: argparse.Namespace) -> None:
    target_fn = _load_callable(args.target)
    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []

    if args.llm_caller:
        caller = _load_callable(args.llm_caller)
        selector = TemplateSelector(caller, OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES)
        selected = selector.select(target_fn, traces=traces, code_meta=_extract_code_meta(tracer, target_fn))
    else:
        # fallback to heuristic/built-in registry selection
        registry = MetricRegistry()
        register_builtin_metrics(registry)
        selector = LLMRegistrySelector(lambda prompt: [])
        selected = registry.list()

    print("Selected metrics:")
    for spec in selected:
        print(f"- {spec.id}: [{spec.type}] config={getattr(spec, 'config', {})}")


def cmd_list_metrics(args: argparse.Namespace) -> None:
    def _compact(value, max_len: int = 60) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _compact_text(value: Any, max_len: int = 55) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _config_summary(cfg: Any) -> str:
        if not cfg:
            return "{}"
        if not isinstance(cfg, dict):
            return _compact(cfg, 55)
        parts: list[str] = []
        for key in sorted(cfg.keys()):
            val = cfg.get(key)
            if key == "rubric" and isinstance(val, list):
                parts.append(f"rubric[{len(val)}]")
                continue
            if key == "policy" and isinstance(val, str):
                parts.append(f"policy[{len(val)}]")
                continue
            if key == "schema" and isinstance(val, dict):
                parts.append(f"schema[{len(val)}]")
                continue
            if isinstance(val, (str, int, float, bool)) or val is None:
                parts.append(f"{key}={_compact_text(val, 18)}")
            else:
                parts.append(f"{key}=…")
        text = ", ".join(parts)
        return text if text else "{}"

    def _print_table(title: str, templates: list[dict]) -> None:
        print(title)
        headers = ["id", "category", "inputs", "config", "desc"]
        rows = []
        for tpl in templates:
            inputs = tpl.get("inputs") or tpl.get("signals") or []
            rows.append(
                [
                    tpl.get("id", ""),
                    tpl.get("category", ""),
                    _compact(inputs, 45),
                    _config_summary(tpl.get("config", {})),
                    _compact_text(tpl.get("description", ""), 55),
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    _print_table("\nObjective metrics:", OBJECTIVE_TEMPLATES)
    _print_table("\nSubjective metrics:", SUBJECTIVE_TEMPLATES)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evalyn",
        description="Evalyn CLI - Instrument, trace, and evaluate LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evalyn list-calls --limit 20              List recent traced calls
  evalyn show-call --id <call_id>           Show details for a specific call
  evalyn build-dataset --project myproj     Build dataset from traces
  evalyn suggest-metrics --target app.py:func --mode basic
                                            Suggest metrics (fast heuristic)
  evalyn run-eval --dataset data/myproj/dataset.jsonl --metrics data/myproj/metrics/basic.json
                                            Run evaluation on dataset

For more info on a command: evalyn <command> --help
""",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command")

    def _print_ascii_help():
        art = r"""
         ______  __      __    /\       _     __     __  __   __   
        |  ____| \ \    / /   /  \     | |    \ \   / /  | \ | |  
        | |__     \ \  / /   / /\ \    | |     \ \_/ /   |  \| |  
              Evalyn CLI — Streamlined Evaluation Framework
        |  __|     \ \/ /   / ____ \   | |      \   /    | . ` |  
        | |____     \  /   / /    \ \  | |____   | |     | |\  |  
        |______|     \/   /_/      \_\ |______|  |_|     |_| \_|  
          
        """
        print("================================================================================")
        print(art)
        print("================================================================================")
        parser.print_help()

    help_parser = subparsers.add_parser("help", help="Show available commands and examples", add_help=False)
    help_parser.set_defaults(func=lambda args: _print_ascii_help())

    list_parser = subparsers.add_parser("list-calls", help="List recent traced calls")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of calls to display")
    list_parser.add_argument("--project", help="Filter by project_id/project_name in call metadata")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    list_parser.set_defaults(func=cmd_list_calls)

    run_parser = subparsers.add_parser("run-eval", help="Run evaluation on dataset using specified metrics")
    run_parser.add_argument("--dataset", required=True, help="Path to JSON/JSONL dataset file or directory containing dataset.jsonl")
    run_parser.add_argument("--metrics", help="Path to metrics JSON file(s), comma-separated for multiple (auto-detected from meta.json if omitted)")
    run_parser.add_argument("--metrics-all", action="store_true", help="Use all metrics files from the metrics/ folder")
    run_parser.add_argument("--dataset-name", help="Name for the eval run (defaults to dataset filename)")
    run_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    run_parser.set_defaults(func=cmd_run_eval)

    suggest_parser = subparsers.add_parser("suggest-metrics", help="Suggest metrics for a target function")
    suggest_parser.add_argument("--target", required=True, help="Callable to analyze in the form module:function")
    suggest_parser.add_argument("--num-traces", type=int, default=5, help="How many recent traces to include as examples")
    suggest_parser.add_argument("--num-metrics", type=int, help="Maximum number of metrics to return")
    suggest_parser.add_argument(
        "--mode",
        choices=["auto", "llm-registry", "llm-brainstorm", "bundle", "basic"],
        default="auto",
        help="auto (use function's metric_mode if set), llm-registry (LLM picks from registry), llm-brainstorm (free-form), bundle (preset), or basic heuristic.",
    )
    suggest_parser.add_argument(
        "--llm-mode",
        choices=["local", "api"],
        default="api",
        help="When using LLM modes: choose local (ollama) or api (OpenAI/Gemini-compatible).",
    )
    suggest_parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Model name (e.g., gemini-2.5-flash-lite, gpt-4, llama3.1 for Ollama)")
    suggest_parser.add_argument("--api-base", help="Custom API base URL for --llm-mode api (optional)")
    suggest_parser.add_argument("--api-key", help="API key override for --llm-mode api (optional)")
    suggest_parser.add_argument(
        "--llm-caller",
        help="Optional callable path that accepts a prompt string and returns a list of metric dicts",
    )
    suggest_parser.add_argument("--bundle", help="Bundle name when --mode bundle (e.g., summarization, orchestrator, research-agent)")
    suggest_parser.add_argument("--dataset", help="Dataset directory (or dataset.jsonl/meta.json) to save metrics into")
    suggest_parser.add_argument("--metrics-name", help="Metrics set name when saving to a dataset")
    suggest_parser.set_defaults(func=cmd_suggest_metrics)

    build_ds = subparsers.add_parser("build-dataset", help="Build dataset from stored traces")
    build_ds.add_argument("--output", help="Path to write dataset JSONL (default: data/<project>-<version>-<timestamp>.jsonl)")
    build_ds.add_argument("--project", help="Filter by metadata.project_id or project_name (recommended grouping)")
    build_ds.add_argument("--version", help="Filter by metadata.version")
    build_ds.add_argument("--since", help="ISO timestamp lower bound for started_at")
    build_ds.add_argument("--until", help="ISO timestamp upper bound for started_at")
    build_ds.add_argument("--limit", type=int, default=500, help="Max number of items to include (after filtering)")
    build_ds.add_argument("--include-errors", action="store_true", help="Include errored calls (default: skip)")
    build_ds.set_defaults(func=cmd_build_dataset)

    show_projects = subparsers.add_parser("show-projects", help="Summaries per project/version")
    show_projects.add_argument("--limit", type=int, default=5000, help="How many calls to scan")
    show_projects.set_defaults(func=cmd_show_projects)

    runs_parser = subparsers.add_parser("list-runs", help="List stored eval runs")
    runs_parser.add_argument("--limit", type=int, default=10)
    runs_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    runs_parser.set_defaults(func=cmd_list_runs)

    show_call = subparsers.add_parser("show-call", help="Show details for a traced call")
    show_call.add_argument("--id", required=True, help="Call id to display")
    show_call.set_defaults(func=cmd_show_call)

    show_run = subparsers.add_parser("show-run", help="Show details for an eval run")
    show_run.add_argument("--id", required=True, help="Eval run id to display")
    show_run.set_defaults(func=cmd_show_run)

    import_ann = subparsers.add_parser("import-annotations", help="Import annotations from a JSONL file")
    import_ann.add_argument("--path", required=True, help="Path to annotations JSONL")
    import_ann.set_defaults(func=cmd_import_annotations)

    export_ann = subparsers.add_parser("export-for-annotation", help="Export dataset with eval results for human annotation")
    export_ann.add_argument("--dataset", required=True, help="Path to dataset directory or dataset.jsonl file")
    export_ann.add_argument("--output", required=True, help="Output path for annotation JSONL file")
    export_ann.add_argument("--run-id", help="Specific eval run ID to use (defaults to latest)")
    export_ann.set_defaults(func=cmd_export_for_annotation)

    ann_stats = subparsers.add_parser("annotation-stats", help="Show annotation coverage statistics")
    ann_stats.add_argument("--dataset", required=True, help="Path to annotations.jsonl or dataset directory")
    ann_stats.set_defaults(func=cmd_annotation_stats)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate a subjective metric using human annotations")
    calibrate_parser.add_argument("--metric-id", required=True, help="Metric ID to calibrate (usually the judge metric id)")
    calibrate_parser.add_argument("--annotations", required=True, help="Path to annotations JSONL (target_id must match call_id)")
    calibrate_parser.add_argument("--run-id", help="Eval run id to calibrate; defaults to latest run")
    calibrate_parser.add_argument("--threshold", type=float, default=0.5, help="Current threshold for pass/fail")
    calibrate_parser.set_defaults(func=cmd_calibrate)

    select_parser = subparsers.add_parser("select-metrics", help="LLM-guided selection from metric registry")
    select_parser.add_argument("--target", required=True, help="Callable to analyze in the form module:function")
    select_parser.add_argument("--llm-caller", required=True, help="Callable that accepts a prompt and returns metric ids or dicts")
    select_parser.add_argument("--limit", type=int, default=5, help="Recent traces to include as examples")
    select_parser.set_defaults(func=cmd_select_metrics)

    list_metrics = subparsers.add_parser("list-metrics", help="List available metric templates (objective + subjective)")
    list_metrics.set_defaults(func=cmd_list_metrics)

    args = parser.parse_args(argv)

    # Handle --version flag
    if args.version:
        from . import __version__
        print(f"evalyn {__version__}")
        return

    # Require a command if --version wasn't used
    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except BrokenPipeError:
        # Allow piping to tools that close stdout early (e.g., `head`, `Select-Object -First`).
        try:
            sys.stdout.close()
        finally:
            return


if __name__ == "__main__":
    main()
