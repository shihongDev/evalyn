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
from .calibration import CalibrationEngine, GEPAConfig, GEPA_AVAILABLE, save_calibration, load_optimized_prompt
from .span_annotation import (
    SpanAnnotation, ANNOTATION_SCHEMAS, extract_spans_from_trace,
    get_annotation_prompts, SpanType,
)


# ---------------------------------------------------------------------------
# Config file support
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATHS = [".evalynrc", "evalyn.yaml", "evalyn.yml", ".evalyn.yaml"]


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(value, str):
        # Expand ${VAR} or $VAR patterns
        import re
        def replace_env(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        return re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)', replace_env, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config() -> Dict[str, Any]:
    """Load configuration from evalyn.yaml or .evalynrc if present."""
    for config_path in DEFAULT_CONFIG_PATHS:
        path = Path(config_path)
        if path.exists():
            try:
                import yaml  # Optional dependency
                with open(path) as f:
                    config = yaml.safe_load(f) or {}
                    # Expand environment variables
                    config = _expand_env_vars(config)
                    return config
            except ImportError:
                # Try JSON format if yaml not available
                try:
                    with open(path) as f:
                        config = json.load(f)
                        config = _expand_env_vars(config)
                        return config
                except Exception:
                    pass
            except Exception:
                pass
    return {}


def get_config_default(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get nested config value with fallback."""
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------
def find_latest_dataset(data_dir: str = "data") -> Optional[Path]:
    """Find the most recently modified dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    # Find directories containing dataset.jsonl
    dataset_dirs = []
    for d in data_path.iterdir():
        if d.is_dir() and (d / "dataset.jsonl").exists():
            dataset_dirs.append(d)

    if not dataset_dirs:
        return None

    # Sort by modification time (most recent first)
    dataset_dirs.sort(key=lambda d: (d / "dataset.jsonl").stat().st_mtime, reverse=True)
    return dataset_dirs[0]


def resolve_dataset_path(dataset_arg: Optional[str], use_latest: bool = False, config: Optional[Dict] = None) -> Optional[Path]:
    """Resolve dataset path from argument, --latest flag, or config."""
    if dataset_arg:
        path = Path(dataset_arg)
        if path.is_file():
            return path.parent
        return path

    if use_latest:
        return find_latest_dataset()

    if config:
        default_dataset = get_config_default(config, "defaults", "dataset")
        if default_dataset:
            return Path(default_dataset)

    return None


# ---------------------------------------------------------------------------
# Progress indicator for long operations
# ---------------------------------------------------------------------------
class ProgressIndicator:
    """Progress indicator for operations with known step count."""

    def __init__(self, total: int, message: str = "Processing", width: int = 30):
        self.total = total
        self.current = 0
        self.message = message
        self.width = width
        self._start_time = time.time()

    def update(self, current: Optional[int] = None, extra: str = ""):
        """Update progress display."""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self._start_time
        eta = ""
        if pct > 0 and pct < 1:
            remaining = (elapsed / pct) * (1 - pct)
            eta = f" ETA: {int(remaining)}s"

        extra_str = f" {extra}" if extra else ""
        sys.stderr.write(f"\r{self.message}: [{bar}] {self.current}/{self.total} ({pct:.0%}){eta}{extra_str}  ")
        sys.stderr.flush()

    def finish(self, message: str = "Done"):
        """Complete the progress indicator."""
        elapsed = time.time() - self._start_time
        sys.stderr.write(f"\r{self.message}: {message} ({elapsed:.1f}s)" + " " * 20 + "\n")
        sys.stderr.flush()
from .datasets import load_dataset, save_dataset_with_meta, build_dataset_from_storage
from .simulator import UserSimulator, AgentSimulator, SimulationConfig
from .decorators import get_default_tracer
from .metrics.objective import register_builtin_metrics
from .metrics.registry import MetricRegistry
from .metrics.templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .runner import EvalRunner
from .metrics.suggester import HeuristicSuggester, LLMSuggester, LLMRegistrySelector, TemplateSelector, render_selection_prompt_with_templates
from .tracing import EvalTracer
from .models import MetricSpec
from datetime import datetime, timezone

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

    # Filter by simulation/production if specified
    if hasattr(args, 'simulation') and args.simulation and calls:
        filtered = []
        for c in calls:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            if meta.get("is_simulation", False):
                filtered.append(c)
        calls = filtered
    elif hasattr(args, 'production') and args.production and calls:
        filtered = []
        for c in calls:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            if not meta.get("is_simulation", False):
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
            is_sim = call.metadata.get("is_simulation", False) if isinstance(call.metadata, dict) else False
            result.append({
                "id": call.id,
                "function": call.function_name,
                "project": project,
                "version": version,
                "is_simulation": is_sim,
                "status": "ERROR" if call.error else "OK",
                "file": code.get("file_path") if isinstance(code, dict) else None,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_ms": call.duration_ms,
            })
        print(json.dumps(result, indent=2))
        return

    # Table output mode
    headers = ["id", "function", "project", "version", "sim?", "status", "file", "start", "end", "duration_ms"]
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
        if isinstance(call.metadata, dict):
            project = call.metadata.get("project_id") or call.metadata.get("project_name") or ""
            version = call.metadata.get("version", "")
            is_sim = call.metadata.get("is_simulation", False)
        file_path = code.get("file_path") if isinstance(code, dict) else None
        row = [
            call.id,
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
    config = load_config()

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, 'dataset', None)
    use_latest = getattr(args, 'latest', False)
    dataset_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not dataset_path:
        print("Error: No dataset specified. Use --dataset <path> or --latest")
        sys.exit(1)

    # Resolve dataset and metrics paths
    try:
        dataset_file, metrics_paths = _resolve_dataset_and_metrics(
            str(dataset_path), args.metrics, metrics_all=metrics_all
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
    from .calibration import load_optimized_prompt
    metrics = []
    objective_count = 0
    subjective_count = 0
    calibrated_count = 0

    # Check if we should use calibrated prompts
    use_calibrated = getattr(args, 'use_calibrated', False)

    skipped_metrics = []
    for spec_data in all_metrics_data.values():
        spec = MetricSpec(
            id=spec_data["id"],
            name=spec_data.get("name", spec_data["id"]),
            type=spec_data["type"],
            description=spec_data.get("description", ""),
            config=spec_data.get("config", {}),
        )

        # Load calibrated prompt for subjective metrics if --use-calibrated is set
        if use_calibrated and spec.type == "subjective":
            try:
                optimized_prompt = load_optimized_prompt(str(dataset_dir), spec.id)
                if optimized_prompt:
                    # Update config with optimized prompt
                    spec.config = dict(spec.config or {})
                    spec.config["prompt"] = optimized_prompt
                    calibrated_count += 1
                    if output_format != "json":
                        print(f"  Using calibrated prompt for {spec.id}")
            except Exception as e:
                if output_format != "json":
                    print(f"  Warning: Could not load calibrated prompt for {spec.id}: {e}")

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
        metrics_summary = f"Loaded {len(metrics)} metrics ({objective_count} objective, {subjective_count} subjective"
        if calibrated_count > 0:
            metrics_summary += f", {calibrated_count} calibrated"
        metrics_summary += ")"
        print(metrics_summary)
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
    tracer = get_default_tracer()

    # Validate: need either --project or --target
    if not args.project and not args.target:
        print("Error: Either --project or --target is required.")
        print("\nUsage:")
        print("  evalyn suggest-metrics --project <name>    # Suggest based on project traces")
        print("  evalyn suggest-metrics --target <path>     # Suggest based on function code")
        print("\nTo see available projects:")
        print("  evalyn show-projects")
        sys.exit(1)

    # Validate dataset path if provided
    dataset_path_obj: Optional[Path] = None
    if args.dataset:
        dataset_path_obj = Path(args.dataset)
        if dataset_path_obj.is_file():
            if not dataset_path_obj.exists():
                print(f"Error: Dataset file not found: {dataset_path_obj}")
                print("Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.")
                sys.exit(1)
        else:
            if not dataset_path_obj.exists():
                print(f"Error: Dataset directory not found: {dataset_path_obj}")
                print("Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.")
                sys.exit(1)
            has_dataset = (dataset_path_obj / "dataset.jsonl").exists() or (dataset_path_obj / "dataset.json").exists()
            if not has_dataset:
                print(f"Warning: Directory exists but no dataset.jsonl or dataset.json found in: {dataset_path_obj}")

    # Check if dataset has reference values
    has_reference = _dataset_has_reference(dataset_path_obj)
    if dataset_path_obj and not has_reference:
        print("Note: Dataset has no reference/expected values. Reference-based metrics (ROUGE, BLEU, etc.) excluded.")

    # Load traces and function info based on --project or --target
    target_fn = None
    metric_mode_hint = None
    metric_bundle_hint = None
    traces = []
    function_name = "unknown"
    function_signature = ""
    function_docstring = ""

    if args.project:
        # Project-based: load traces from storage
        if not tracer.storage:
            print("Error: No storage configured. Cannot load project traces.")
            sys.exit(1)

        # Load all calls for this project
        all_calls = tracer.storage.list_calls(limit=500)
        project_traces = []
        for call in all_calls:
            meta = call.metadata if isinstance(call.metadata, dict) else {}
            call_project = meta.get("project_id") or meta.get("project_name") or call.function_name
            call_version = meta.get("version") or ""

            if call_project == args.project:
                if args.version and call_version != args.version:
                    continue
                project_traces.append(call)

        if not project_traces:
            print(f"Error: No traces found for project '{args.project}'")
            if args.version:
                print(f"  (filtered by version: {args.version})")
            print("\nAvailable projects:")
            print("  evalyn show-projects")
            sys.exit(1)

        traces = project_traces[:args.num_traces]
        print(f"Found {len(project_traces)} traces for project '{args.project}'")
        if args.version:
            print(f"  (version: {args.version})")

        # Extract function info from the first trace
        first_call = project_traces[0]
        function_name = first_call.function_name
        meta = first_call.metadata if isinstance(first_call.metadata, dict) else {}
        function_signature = meta.get("signature", "")
        function_docstring = meta.get("docstring", "")

        # Create a placeholder function for the suggester
        def _placeholder_fn(*args, **kwargs):
            pass
        _placeholder_fn.__name__ = function_name
        _placeholder_fn.__doc__ = function_docstring
        target_fn = _placeholder_fn

    else:
        # Target-based: load callable directly
        target_fn = _load_callable(args.target)
        metric_mode_hint = getattr(target_fn, "_evalyn_metric_mode", None)
        metric_bundle_hint = getattr(target_fn, "_evalyn_metric_bundle", None)
        traces = tracer.storage.list_calls(limit=args.num_traces) if tracer.storage else []
        function_name = target_fn.__name__

    selected_mode = args.mode
    if selected_mode == "auto":
        selected_mode = metric_mode_hint or "llm-registry"
    bundle_name = args.bundle or metric_bundle_hint
    max_metrics = args.num_metrics if args.num_metrics and args.num_metrics > 0 else None

    # Get scope filter
    scope_filter = getattr(args, 'scope', 'all')
    if scope_filter == 'all':
        scope_filter = None

    def _filter_by_scope(templates: list) -> list:
        """Filter templates by scope."""
        if not scope_filter:
            return templates
        return [t for t in templates if t.get("scope", "overall") == scope_filter]

    if scope_filter:
        print(f"Filtering metrics by scope: {scope_filter}")

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

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if args.metrics_name:
            metrics_name = args.metrics_name
        elif selected_mode == "bundle":
            metrics_name = f"bundle-{(bundle_name or 'bundle')}"
        else:
            metrics_name = f"{selected_mode}-{ts}"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", metrics_name).strip("-")
        metrics_file = metrics_dir / f"{safe_name}.json"

        # Validate objective metrics - filter out custom ones that won't work
        valid_objective_ids = {t["id"] for t in OBJECTIVE_TEMPLATES}
        invalid_objectives = [s for s in specs if s.type == "objective" and s.id not in valid_objective_ids]
        if invalid_objectives:
            print(f"Removed {len(invalid_objectives)} unsupported custom objective metric(s):")
            for s in invalid_objectives:
                print(f"  - {s.id}: Use 'evalyn list-metrics --type objective' to see valid IDs")
            specs = [s for s in specs if not (s.type == "objective" and s.id not in valid_objective_ids)]

        if not specs:
            print("No valid metrics to save.")
            return

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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
        all_templates = _filter_by_scope(OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES)
        tpl_map = {t["id"]: t for t in all_templates}
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
        filtered_templates = _filter_by_scope(OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES)
        selector = TemplateSelector(caller, filtered_templates, has_reference=has_reference)
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
        specs = suggester.suggest(target_fn, traces, desired_count=max_metrics, scope=scope_filter)
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

    # Filter by scope if specified (lookup scope from templates)
    if scope_filter:
        all_templates = OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES
        template_scope = {t["id"]: t.get("scope", "overall") for t in all_templates}
        specs = [s for s in specs if template_scope.get(s.id, "overall") == scope_filter]

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
        simulation_only=getattr(args, 'simulation', False),
        production_only=getattr(args, 'production', False),
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
        limit=args.limit,
        success_only=not args.include_errors,
        include_metadata=True,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
        "created_at": datetime.now(timezone.utc).isoformat(),
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


def cmd_annotate_spans(args: argparse.Namespace) -> None:
    """Interactive span-level annotation interface."""
    import uuid

    config = load_config()

    # Resolve dataset path
    dataset_arg = getattr(args, 'dataset', None)
    use_latest = getattr(args, 'latest', False)
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not resolved_path:
        print("Error: No dataset specified. Use --dataset <path> or --latest")
        sys.exit(1)

    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        data_file = dataset_path / "dataset.jsonl"
        if not data_file.exists():
            data_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        data_file = dataset_path

    if not data_file.exists():
        print(f"Error: Dataset file not found: {data_file}")
        sys.exit(1)

    # Load dataset items to get call_ids
    dataset_items = load_dataset(data_file)
    if not dataset_items:
        print("No items in dataset.")
        return

    # Get storage to fetch full calls
    tracer = get_default_tracer()
    if not tracer.storage:
        print("Error: No storage configured. Cannot retrieve call traces.")
        sys.exit(1)

    # Get span_type filter
    span_type_filter = getattr(args, 'span_type', 'all')
    if span_type_filter == 'all':
        span_type_filter = None

    # Output path for span annotations
    output_path = Path(args.output) if args.output else dataset_dir / "span_annotations.jsonl"

    # Load existing span annotations
    existing_annotations: Dict[str, SpanAnnotation] = {}
    if output_path.exists() and not getattr(args, 'restart', False):
        try:
            for line in output_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    data = json.loads(line)
                    ann = SpanAnnotation.from_dict(data)
                    existing_annotations[ann.span_id] = ann
        except Exception:
            pass

    # Collect all spans to annotate
    all_spans = []
    for item in dataset_items:
        call_id = item.metadata.get("call_id", item.id)

        # Fetch the full call from storage
        call = tracer.storage.get_call(call_id)
        if not call:
            continue

        # Extract spans from the call
        spans = extract_spans_from_trace(call)

        # Filter by span_type if specified
        if span_type_filter:
            spans = [s for s in spans if s["span_type"] == span_type_filter]

        # Skip already annotated spans
        spans = [s for s in spans if s["span_id"] not in existing_annotations]

        for span in spans:
            span["call"] = call  # Attach call for context
            all_spans.append(span)

    if not all_spans:
        print("No spans to annotate. All spans already annotated or no matching spans found.")
        if span_type_filter:
            print(f"Tip: Filter was set to '{span_type_filter}'. Try --span-type all")
        return

    # Group spans by type for summary
    by_type = {}
    for span in all_spans:
        st = span["span_type"]
        by_type[st] = by_type.get(st, 0) + 1

    print("\n" + "=" * 70)
    print("SPAN ANNOTATION MODE")
    print("=" * 70)
    print(f"Dataset: {data_file}")
    print(f"Spans to annotate: {len(all_spans)}")
    print(f"Already annotated: {len(existing_annotations)}")
    print(f"Output: {output_path}")
    print("\nSpan types:")
    for st, count in sorted(by_type.items()):
        print(f"  {st}: {count}")
    print("\nCommands: [y/n/1-5] answer prompts  [s]kip span  [q]uit")
    print("=" * 70)

    annotations: List[SpanAnnotation] = list(existing_annotations.values())

    def save_annotations() -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")

    def truncate(text: str, max_len: int = 300) -> str:
        text = str(text) if text else ""
        text = text.replace("\n", " ").strip()
        return text if len(text) <= max_len else text[:max_len] + "..."

    def get_bool_input(prompt: str) -> Optional[bool]:
        """Get yes/no input."""
        while True:
            try:
                val = input(f"  {prompt} [y/n/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if val in ("y", "yes", "1", "true"):
                return True
            if val in ("n", "no", "0", "false"):
                return False
            if val in ("s", "skip", ""):
                return None
            print("  Invalid. Use y(es), n(o), or s(kip)")

    def get_int_input(prompt: str, min_val: int, max_val: int) -> Optional[int]:
        """Get integer input within range."""
        while True:
            try:
                val = input(f"  {prompt} [{min_val}-{max_val}/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if val in ("s", "skip", ""):
                return None
            try:
                num = int(val)
                if min_val <= num <= max_val:
                    return num
                print(f"  Invalid. Use {min_val}-{max_val} or s(kip)")
            except ValueError:
                print(f"  Invalid. Use {min_val}-{max_val} or s(kip)")

    def get_str_input(prompt: str) -> str:
        """Get string input."""
        try:
            return input(f"  {prompt} ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    idx = 0
    total = len(all_spans)

    while idx < total:
        span = all_spans[idx]
        call = span["call"]
        span_type = span["span_type"]
        span_id = span["span_id"]

        print(f"\n{'─' * 70}")
        print(f"Span {idx + 1}/{total} [{span_type.upper()}]")
        print(f"Call: {call.function_name} [{call.id[:12]}...]")
        print("─" * 70)

        # Show span context
        print(f"\n📋 {span['summary']}")

        if span_type == "overall":
            # Show input/output for overall
            print(f"\n📥 INPUT: {truncate(json.dumps(span.get('input', {}), ensure_ascii=False), 200)}")
            print(f"📤 OUTPUT: {truncate(str(span.get('output', '')), 300)}")
        elif span.get("detail"):
            # Show detail for other span types
            detail = span["detail"]
            if span_type == "llm_call":
                print(f"  Model: {detail.get('model', 'unknown')}")
                print(f"  Tokens: {detail.get('input_tokens', '?')} in / {detail.get('output_tokens', '?')} out")
                if detail.get('cost'):
                    print(f"  Cost: ${detail.get('cost', 0):.6f}")
                if detail.get('response_excerpt'):
                    print(f"  Response: {truncate(detail.get('response_excerpt', ''), 200)}")
            elif span_type == "tool_call":
                print(f"  Tool: {detail.get('tool_name', 'unknown')}")
                if detail.get('args'):
                    print(f"  Args: {truncate(json.dumps(detail.get('args', {}), ensure_ascii=False), 150)}")
                if detail.get('result'):
                    print(f"  Result: {truncate(str(detail.get('result', '')), 150)}")
            elif span_type == "reasoning":
                if detail.get('content'):
                    print(f"  Content: {truncate(str(detail.get('content', '')), 200)}")
            elif span_type == "retrieval":
                if detail.get('query'):
                    print(f"  Query: {truncate(str(detail.get('query', '')), 100)}")
                if detail.get('results_count'):
                    print(f"  Results: {detail.get('results_count')} documents")

        print("─" * 70)

        # Get annotation prompts for this span type
        prompts = get_annotation_prompts(span_type)
        schema_cls = ANNOTATION_SCHEMAS[span_type]
        annotation_values = {}

        print(f"\nAnnotate this {span_type}:")

        quit_requested = False
        skip_span = False

        for prompt_info in prompts:
            field = prompt_info["field"]
            question = prompt_info["question"]
            field_type = prompt_info["type"]

            if field_type == "bool":
                val = get_bool_input(question)
                if val is None:
                    # Check for quit
                    try:
                        check = input("  Skip this span? [y/n/q]: ").strip().lower()
                        if check in ("q", "quit"):
                            quit_requested = True
                            break
                        if check in ("y", "yes"):
                            skip_span = True
                            break
                    except (EOFError, KeyboardInterrupt):
                        quit_requested = True
                        break
                    continue  # Skip this field
                annotation_values[field] = val
            elif field_type == "int":
                range_info = prompt_info.get("range", (1, 5))
                val = get_int_input(question, range_info[0], range_info[1])
                if val is not None:
                    annotation_values[field] = val
            elif field_type == "str":
                val = get_str_input(question)
                if val:
                    annotation_values[field] = val

        if quit_requested:
            save_annotations()
            print(f"\nSaved {len(annotations)} span annotations to {output_path}")
            return

        if skip_span:
            print("Skipped.")
            idx += 1
            continue

        # Create annotation if we have any values
        if annotation_values:
            annotation_schema = schema_cls(**annotation_values)
            span_ann = SpanAnnotation(
                id=f"span-ann-{uuid.uuid4().hex[:8]}",
                call_id=call.id,
                span_id=span_id,
                span_type=span_type,
                annotation=annotation_schema,
                annotator=getattr(args, 'annotator', None) or "human",
            )
            annotations.append(span_ann)
            save_annotations()
            print(f"✓ Saved annotation for {span_type}")
        else:
            print("No annotation recorded (all fields skipped).")

        idx += 1

    # Final save
    save_annotations()
    print("\n" + "=" * 70)
    print("SPAN ANNOTATION COMPLETE")
    print(f"Total annotated: {len(annotations)}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


def cmd_annotate(args: argparse.Namespace) -> None:
    """Interactive CLI annotation interface with per-metric support."""
    from .models import AnnotationItem, HumanLabel, Annotation, MetricLabel

    # Check for span annotation mode
    if getattr(args, 'spans', False):
        return cmd_annotate_spans(args)

    config = load_config()

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, 'dataset', None)
    use_latest = getattr(args, 'latest', False)
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not resolved_path:
        print("Error: No dataset specified. Use --dataset <path> or --latest")
        sys.exit(1)

    # Resolve dataset path
    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        data_file = dataset_path / "dataset.jsonl"
        if not data_file.exists():
            data_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        data_file = dataset_path

    if not data_file.exists():
        print(f"Error: Dataset file not found: {data_file}")
        sys.exit(1)

    # Load dataset items
    dataset_items = load_dataset(data_file)
    if not dataset_items:
        print("No items in dataset.")
        return

    # Get eval run for LLM judge results
    tracer = get_default_tracer()
    run = None
    if args.run_id and tracer.storage:
        run = tracer.storage.get_eval_run(args.run_id)
    elif tracer.storage:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None

    # Build eval results lookup by call_id
    eval_results_by_call: Dict[str, Dict[str, Any]] = {}
    if run:
        for result in run.metric_results:
            if result.call_id not in eval_results_by_call:
                eval_results_by_call[result.call_id] = {}
            eval_results_by_call[result.call_id][result.metric_id] = {
                "score": result.score,
                "passed": result.passed,
                "reason": result.details.get("reason", "") if result.details else "",
            }

    # Load existing annotations if any
    output_path = Path(args.output) if args.output else dataset_dir / "annotations.jsonl"
    existing_annotations: Dict[str, Annotation] = {}
    if output_path.exists() and not args.restart:
        try:
            for line in output_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    data = json.loads(line)
                    ann = Annotation.from_dict(data)
                    existing_annotations[ann.target_id] = ann
        except Exception:
            pass

    # Filter items based on options
    items_to_annotate = []
    for item in dataset_items:
        call_id = item.metadata.get("call_id", item.id)

        # Skip if already annotated (unless --restart)
        if call_id in existing_annotations and not args.restart:
            continue

        items_to_annotate.append(item)

    if not items_to_annotate:
        print("No items to annotate. All items already have annotations.")
        print(f"Use --restart to re-annotate, or check {output_path}")
        return

    total = len(items_to_annotate)
    annotated_count = len(existing_annotations)
    annotations: List[Annotation] = list(existing_annotations.values())
    per_metric_mode = args.per_metric

    print("\n" + "=" * 70)
    print("INTERACTIVE ANNOTATION" + (" (Per-Metric Mode)" if per_metric_mode else ""))
    print("=" * 70)
    print(f"Dataset: {data_file}")
    print(f"Items to annotate: {total}")
    print(f"Already annotated: {annotated_count}")
    if run:
        print(f"Using eval run: {run.id[:8]}...")
    print(f"Output: {output_path}")
    if per_metric_mode:
        print("\nPer-metric commands:")
        print("  [a]gree with LLM  [d]isagree (flip)  [s]kip metric")
    else:
        print("\nCommands: [y]es/pass  [n]o/fail  [s]kip  [v]iew full  [q]uit")
    print("=" * 70)

    def truncate(text: str, max_len: int = 500) -> str:
        text = str(text) if text else ""
        text = text.replace("\n", " ").strip()
        return text if len(text) <= max_len else text[:max_len] + "..."

    def display_item(idx: int, item: DatasetItem, show_metric_numbers: bool = False) -> List[tuple]:
        """Display item and return list of (metric_id, llm_passed, reason) for subjective metrics."""
        call_id = item.metadata.get("call_id", item.id)
        print(f"\n{'─' * 70}")
        print(f"Item {idx + 1}/{total} [{call_id[:12]}...]")
        print("─" * 70)

        # Input
        input_text = json.dumps(item.input, ensure_ascii=False, indent=2) if item.input else "(no input)"
        print(f"\n📥 INPUT:")
        if len(input_text) > 300:
            print(f"   {truncate(input_text, 300)}")
        else:
            for line in input_text.split("\n")[:5]:
                print(f"   {line}")

        # Output
        output_text = str(item.output) if item.output else "(no output)"
        print(f"\n📤 OUTPUT:")
        print(f"   {truncate(output_text, 500)}")

        # LLM Judge results
        eval_data = eval_results_by_call.get(call_id, {})
        subjective_metrics = []
        if eval_data:
            print(f"\n🤖 LLM JUDGE RESULTS:")
            metric_num = 1
            for metric_id, result in eval_data.items():
                passed = result.get("passed")
                if passed is None:
                    continue
                status = "✅ PASS" if passed else "❌ FAIL"
                reason = result.get("reason", "")
                if show_metric_numbers:
                    print(f"   [{metric_num}] {metric_id}: {status}")
                else:
                    print(f"   {metric_id}: {status}")
                if reason:
                    print(f"       Reason: {truncate(reason, 200)}")
                subjective_metrics.append((metric_id, passed, reason))
                metric_num += 1

        print("─" * 70)
        return subjective_metrics

    def get_confidence() -> Optional[int]:
        """Get confidence score 1-5 from user."""
        try:
            conf_input = input("Confidence (1-5, Enter to skip): ").strip()
            if not conf_input:
                return None
            conf = int(conf_input)
            if 1 <= conf <= 5:
                return conf
            print("Invalid. Use 1-5.")
            return get_confidence()
        except ValueError:
            print("Invalid. Use 1-5 or Enter to skip.")
            return get_confidence()
        except (EOFError, KeyboardInterrupt):
            return None

    def save_annotations() -> None:
        """Save all annotations to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")

    def annotate_per_metric(item: DatasetItem, subjective_metrics: List[tuple]) -> Optional[Annotation]:
        """Per-metric annotation flow."""
        call_id = item.metadata.get("call_id", item.id)
        metric_labels: Dict[str, MetricLabel] = {}

        if not subjective_metrics:
            print("No subjective metrics to annotate for this item.")
            return None

        print("\nAnnotate each metric ([a]gree, [d]isagree/flip, [s]kip, [q]uit):")

        for i, (metric_id, llm_passed, reason) in enumerate(subjective_metrics, 1):
            status = "✅ PASS" if llm_passed else "❌ FAIL"
            print(f"\n  [{i}/{len(subjective_metrics)}] {metric_id}: LLM says {status}")
            if reason:
                print(f"      Reason: {truncate(reason, 150)}")

            while True:
                try:
                    choice = input(f"  Your verdict [a/d/s/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return None

                if choice in ("q", "quit"):
                    return None  # Signal to quit

                if choice in ("s", "skip"):
                    break  # Skip this metric

                if choice in ("a", "agree", "y", "yes"):
                    # Agree with LLM
                    metric_labels[metric_id] = MetricLabel(
                        metric_id=metric_id,
                        agree_with_llm=True,
                        human_label=llm_passed,
                        notes="",
                    )
                    print(f"      → Agreed: {status}")
                    break

                if choice in ("d", "disagree", "n", "no", "flip"):
                    # Disagree - flip the label
                    human_label = not llm_passed
                    human_status = "✅ PASS" if human_label else "❌ FAIL"
                    try:
                        notes = input(f"      Notes (why disagree?): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        notes = ""
                    metric_labels[metric_id] = MetricLabel(
                        metric_id=metric_id,
                        agree_with_llm=False,
                        human_label=human_label,
                        notes=notes,
                    )
                    print(f"      → Disagreed: Human says {human_status}")
                    break

                print("      Invalid. Use: a(gree), d(isagree), s(kip), q(uit)")

        if not metric_labels:
            print("No metrics annotated.")
            return None

        # Calculate overall label from metric labels
        human_passes = [ml.human_label for ml in metric_labels.values()]
        overall_passed = all(human_passes) if human_passes else True

        # Get confidence
        confidence = get_confidence()

        # Get overall notes
        try:
            overall_notes = input("Overall notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            overall_notes = ""

        return Annotation(
            id=f"ann-{call_id[:8]}-{len(annotations)}",
            target_id=call_id,
            label=overall_passed,
            rationale=overall_notes if overall_notes else None,
            annotator=args.annotator or "human",
            source="human",
            confidence=confidence,
            metric_labels=metric_labels,
        )

    def annotate_simple(item: DatasetItem) -> Optional[Annotation]:
        """Simple overall pass/fail annotation flow."""
        call_id = item.metadata.get("call_id", item.id)

        while True:
            try:
                user_input = input("\nPass? [y/n/s/v/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if user_input in ("q", "quit"):
                return None  # Signal to quit

            if user_input in ("s", "skip"):
                return "skip"  # Signal to skip

            if user_input in ("v", "view"):
                # Show full output
                print("\n" + "=" * 70)
                print("FULL OUTPUT:")
                print("=" * 70)
                print(str(item.output) if item.output else "(no output)")
                print("=" * 70)
                continue

            if user_input in ("y", "yes", "1", "p", "pass"):
                passed = True
            elif user_input in ("n", "no", "0", "f", "fail"):
                passed = False
            else:
                print("Invalid input. Use: y(es), n(o), s(kip), v(iew), q(uit)")
                continue

            # Get confidence
            confidence = get_confidence()

            # Get optional notes
            try:
                notes = input("Notes (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                notes = ""

            return Annotation(
                id=f"ann-{call_id[:8]}-{len(annotations)}",
                target_id=call_id,
                label=passed,
                rationale=notes if notes else None,
                annotator=args.annotator or "human",
                source="human",
                confidence=confidence,
            )

    # Main annotation loop
    idx = 0
    while idx < total:
        item = items_to_annotate[idx]
        call_id = item.metadata.get("call_id", item.id)

        subjective_metrics = display_item(idx, item, show_metric_numbers=per_metric_mode)

        if per_metric_mode:
            ann = annotate_per_metric(item, subjective_metrics)
            if ann is None:
                # Check if user wants to quit
                try:
                    quit_check = input("\nQuit? [y/n]: ").strip().lower()
                    if quit_check in ("y", "yes"):
                        save_annotations()
                        print(f"\nSaved {len(annotations)} annotations to {output_path}")
                        return
                except (EOFError, KeyboardInterrupt):
                    save_annotations()
                    print(f"\nSaved {len(annotations)} annotations to {output_path}")
                    return
                idx += 1
                continue
        else:
            ann = annotate_simple(item)
            if ann is None:
                save_annotations()
                print(f"\nSaved {len(annotations)} annotations to {output_path}")
                return
            if ann == "skip":
                print("Skipped.")
                idx += 1
                continue

        # Save annotation
        annotations.append(ann)
        save_annotations()

        # Show summary
        if per_metric_mode:
            agrees = sum(1 for ml in ann.metric_labels.values() if ml.agree_with_llm)
            total_metrics = len(ann.metric_labels)
            print(f"\n✓ Saved: {agrees}/{total_metrics} agree with LLM, overall={('✅ PASS' if ann.label else '❌ FAIL')}")
        else:
            status = "✅ PASS" if ann.label else "❌ FAIL"
            conf_str = f", confidence={ann.confidence}" if ann.confidence else ""
            print(f"Saved: {status}{conf_str}" + (f" - {ann.rationale}" if ann.rationale else ""))

        idx += 1

    # Final save
    save_annotations()
    print("\n" + "=" * 70)
    print(f"ANNOTATION COMPLETE")
    print(f"Total annotated: {len(annotations)}")
    print(f"Saved to: {output_path}")
    print("=" * 70)
    print(f"\nNext step: evalyn calibrate --metric-id <metric> --annotations {output_path}")


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

    # Load annotations
    anns = import_annotations(args.annotations)

    # Get current rubric and preamble from metric config if available
    current_rubric: List[str] = []
    current_preamble: str = ""
    for metric_spec in run.metrics:
        if metric_spec.id == args.metric_id:
            cfg = metric_spec.config or {}
            # Extract rubric (evaluation criteria)
            rubric_val = cfg.get("rubric", [])
            if isinstance(rubric_val, list):
                current_rubric = [str(r) for r in rubric_val]
            # Extract preamble (base prompt before rubric)
            preamble_val = cfg.get("prompt", "")
            if isinstance(preamble_val, str):
                current_preamble = preamble_val
            break

    # Load dataset items for context (if dataset path provided)
    config = load_config()
    dataset_items: Optional[List[DatasetItem]] = None
    dataset_dir: Optional[Path] = None

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, 'dataset', None)
    use_latest = getattr(args, 'latest', False)
    resolved_dataset = resolve_dataset_path(dataset_arg, use_latest, config)

    if resolved_dataset:
        dataset_dir = Path(resolved_dataset)
        dataset_file = dataset_dir / "dataset.jsonl" if dataset_dir.is_dir() else dataset_dir
        if dataset_file.exists():
            dataset_items = load_dataset(dataset_file)
            if dataset_dir.is_file():
                dataset_dir = dataset_dir.parent

    # Build GEPA config if using GEPA optimizer
    gepa_config = None
    if args.optimizer == "gepa":
        if not GEPA_AVAILABLE:
            print("Error: GEPA is not installed. Install with: pip install gepa")
            return
        gepa_config = GEPAConfig(
            task_lm=args.gepa_task_lm,
            reflection_lm=args.gepa_reflection_lm,
            max_metric_calls=args.gepa_max_calls,
        )

    # Run enhanced calibration
    engine = CalibrationEngine(
        judge_name=args.metric_id,
        current_threshold=args.threshold,
        current_rubric=current_rubric,
        current_preamble=current_preamble,
        optimize_prompts=not args.no_optimize,
        optimizer_model=args.model,
        optimizer_type=args.optimizer,
        gepa_config=gepa_config,
    )

    # Use spinner for long-running operations (especially GEPA)
    if args.optimizer == "gepa" and not args.no_optimize:
        spinner_msg = f"Running GEPA optimization (max {args.gepa_max_calls} calls)"
        with Spinner(spinner_msg):
            record = engine.calibrate(metric_results, anns, dataset_items)
    else:
        record = engine.calibrate(metric_results, anns, dataset_items)

    # Display results
    print(f"\n{'='*60}")
    print(f"CALIBRATION REPORT: {args.metric_id}")
    print(f"{'='*60}")
    print(f"Eval Run: {run.id}")
    print(f"Samples:  {record.adjustments.get('alignment_metrics', {}).get('total_samples', 0)}")

    # Alignment metrics
    alignment = record.adjustments.get("alignment_metrics", {})
    if alignment:
        print(f"\n--- ALIGNMENT METRICS ---")
        print(f"Accuracy:       {alignment.get('accuracy', 0):.1%}")
        print(f"Precision:      {alignment.get('precision', 0):.1%}")
        print(f"Recall:         {alignment.get('recall', 0):.1%}")
        print(f"F1 Score:       {alignment.get('f1', 0):.1%}")
        print(f"Specificity:    {alignment.get('specificity', 0):.1%}")
        print(f"Cohen's Kappa:  {alignment.get('cohens_kappa', 0):.3f}")

        # Confusion matrix
        cm = alignment.get("confusion_matrix", {})
        if cm:
            print(f"\nConfusion Matrix:")
            print(f"                   Human PASS  Human FAIL")
            print(f"  Judge PASS       {cm.get('true_positive', 0):^10}  {cm.get('false_positive', 0):^10}")
            print(f"  Judge FAIL       {cm.get('false_negative', 0):^10}  {cm.get('true_negative', 0):^10}")

    # Disagreement patterns
    disagreements = record.adjustments.get("disagreement_patterns", {})
    if disagreements:
        fp_count = disagreements.get("false_positive_count", 0)
        fn_count = disagreements.get("false_negative_count", 0)
        if fp_count > 0 or fn_count > 0:
            print(f"\n--- DISAGREEMENT PATTERNS ---")
            print(f"False Positives (judge too lenient): {fp_count}")
            print(f"False Negatives (judge too strict):  {fn_count}")

            if args.show_examples:
                fp_examples = disagreements.get("false_positive_examples", [])[:3]
                fn_examples = disagreements.get("false_negative_examples", [])[:3]

                if fp_examples:
                    print(f"\nFalse Positive Examples:")
                    for i, ex in enumerate(fp_examples, 1):
                        print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
                        print(f"     Judge reason: {ex.get('judge_reason', '')[:80]}...")
                        if ex.get("human_notes"):
                            print(f"     Human notes:  {ex.get('human_notes', '')[:80]}...")

                if fn_examples:
                    print(f"\nFalse Negative Examples:")
                    for i, ex in enumerate(fn_examples, 1):
                        print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
                        print(f"     Judge reason: {ex.get('judge_reason', '')[:80]}...")
                        if ex.get("human_notes"):
                            print(f"     Human notes:  {ex.get('human_notes', '')[:80]}...")

    # Threshold suggestion
    print(f"\n--- THRESHOLD ---")
    print(f"Current:   {record.adjustments.get('current_threshold', 0.5):.3f}")
    print(f"Suggested: {record.adjustments.get('suggested_threshold', 0.5):.3f}")

    # Prompt optimization results
    optimizer_type = record.adjustments.get("optimizer_type", "llm")
    optimization = record.adjustments.get("prompt_optimization", {})
    if optimization:
        print(f"\n--- PROMPT OPTIMIZATION ---")
        print(f"Optimizer:             {optimizer_type.upper()}")
        print(f"Estimated improvement: {optimization.get('estimated_improvement', 'unknown')}")

        reasoning = optimization.get("improvement_reasoning", "")
        if reasoning:
            # Word-wrap the reasoning
            words = reasoning.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 70:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            print(f"\nReasoning:")
            for line in lines:
                print(f"  {line}")

        additions = optimization.get("suggested_additions", [])
        if additions:
            print(f"\nSuggested ADDITIONS to rubric:")
            for a in additions:
                print(f"  + {a}")

        removals = optimization.get("suggested_removals", [])
        if removals:
            print(f"\nSuggested REMOVALS from rubric:")
            for r in removals:
                print(f"  - {r}")

        # Show optimized preamble (for GEPA)
        optimized_preamble = optimization.get("optimized_preamble", "")
        if optimized_preamble:
            print(f"\nOPTIMIZED PREAMBLE:")
            # Show first 200 chars
            preview = optimized_preamble[:200] + "..." if len(optimized_preamble) > 200 else optimized_preamble
            for line in preview.split("\n"):
                print(f"  {line}")

        improved = optimization.get("improved_rubric", [])
        if improved:
            print(f"\nRUBRIC (unchanged):")
            for i, criterion in enumerate(improved, 1):
                print(f"  {i}. {criterion}")

    # Validation results
    validation = record.adjustments.get("validation", {})
    if validation:
        print(f"\n--- VALIDATION RESULTS ---")

        is_better = validation.get("is_better", False)
        original_f1 = validation.get("original_f1", 0.0)
        optimized_f1 = validation.get("optimized_f1", 0.0)
        improvement_delta = validation.get("improvement_delta", 0.0)
        confidence = validation.get("confidence", "unknown")
        recommendation = validation.get("recommendation", "uncertain")
        val_samples = validation.get("validation_samples", 0)

        # Status indicator
        if is_better and improvement_delta > 0.05:
            status_icon = "✅ SUCCESS"
            status_msg = "Optimized prompt is SIGNIFICANTLY BETTER"
        elif is_better:
            status_icon = "✅ SUCCESS"
            status_msg = "Optimized prompt is BETTER"
        elif improvement_delta < -0.05:
            status_icon = "❌ DEGRADED"
            status_msg = "Optimized prompt is SIGNIFICANTLY WORSE"
        elif improvement_delta < 0:
            status_icon = "⚠️  DEGRADED"
            status_msg = "Optimized prompt is WORSE"
        else:
            status_icon = "➖ UNCERTAIN"
            status_msg = "No significant difference"

        print(f"{status_icon} - {status_msg}")
        print()
        print(f"Original F1:     {original_f1:.3f}")
        print(f"Optimized F1:    {optimized_f1:.3f}")

        if improvement_delta > 0:
            print(f"Improvement:     +{improvement_delta:.3f} (+{improvement_delta*100:.1f}%)")
        elif improvement_delta < 0:
            print(f"Degradation:     {improvement_delta:.3f} ({improvement_delta*100:.1f}%)")
        else:
            print(f"Change:          {improvement_delta:.3f}")

        print(f"Validation size: {val_samples} samples")
        print(f"Confidence:      {confidence.upper()}")
        print()

        # Recommendation
        if recommendation == "use_optimized":
            print(f"💡 RECOMMENDATION: USE OPTIMIZED PROMPT")
            print(f"   Next: evalyn run-eval --latest --use-calibrated")
        elif recommendation == "keep_original":
            print(f"💡 RECOMMENDATION: KEEP ORIGINAL PROMPT")
            print(f"   The optimized prompt did not improve performance.")
        else:
            print(f"💡 RECOMMENDATION: UNCERTAIN")
            print(f"   Consider testing both prompts manually.")

    # Save calibration record
    saved_files = {}

    # Auto-save to dataset's calibrations folder if dataset was resolved
    if dataset_dir and dataset_dir.exists():
        try:
            saved_files = save_calibration(record, str(dataset_dir), args.metric_id)
            print(f"\n--- SAVED FILES ---")
            print(f"Calibration: {saved_files.get('calibration', 'N/A')}")
            if saved_files.get('preamble'):
                print(f"Preamble:    {saved_files.get('preamble')}")
            if saved_files.get('full_prompt'):
                print(f"Full prompt: {saved_files.get('full_prompt')}")
        except Exception as e:
                print(f"\nWarning: Could not save to calibrations folder: {e}")

    # Also save to explicit output path if specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(record.as_dict(), f, indent=2, default=str)
        print(f"\nCalibration record also saved to: {output_path}")

    print(f"\n{'='*60}")


def cmd_list_calibrations(args: argparse.Namespace) -> None:
    """List calibration records for a dataset."""
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, 'dataset', None),
        getattr(args, 'latest', False),
        config
    )

    if not dataset_path:
        print("Error: No dataset specified. Use --dataset or --latest")
        return

    calibrations_dir = dataset_path / "calibrations"
    if not calibrations_dir.exists():
        print(f"No calibrations found in {dataset_path}")
        return

    # Collect all calibration records
    calibrations = []
    for metric_dir in calibrations_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        metric_id = metric_dir.name
        for cal_file in metric_dir.glob("*.json"):
            if cal_file.name.startswith("."):
                continue
            try:
                with open(cal_file) as f:
                    record = json.load(f)
                    # Parse timestamp from filename (e.g., 20250101_120000_gepa.json)
                    parts = cal_file.stem.split("_")
                    timestamp = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
                    optimizer = parts[2] if len(parts) >= 3 else "unknown"

                    alignment = record.get("adjustments", {}).get("alignment_metrics", {})
                    calibrations.append({
                        "metric_id": metric_id,
                        "timestamp": timestamp,
                        "optimizer": optimizer,
                        "accuracy": alignment.get("accuracy", 0),
                        "f1": alignment.get("f1", 0),
                        "kappa": alignment.get("cohens_kappa", 0),
                        "samples": alignment.get("total_samples", 0),
                        "path": str(cal_file),
                    })
            except Exception:
                pass

    if not calibrations:
        print(f"No calibration records found in {calibrations_dir}")
        return

    # Sort by timestamp (most recent first)
    calibrations.sort(key=lambda x: x["timestamp"], reverse=True)

    # Output format
    output_format = getattr(args, 'format', 'table')
    if output_format == "json":
        print(json.dumps(calibrations, indent=2))
        return

    # Table format
    print(f"\nCalibrations in {dataset_path.name}:")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Timestamp':<17} {'Optimizer':<8} {'Acc':<7} {'F1':<7} {'Kappa':<7} {'N':<5}")
    print(f"{'-'*80}")
    for cal in calibrations:
        print(f"{cal['metric_id']:<25} {cal['timestamp']:<17} {cal['optimizer']:<8} "
              f"{cal['accuracy']:.1%}   {cal['f1']:.1%}   {cal['kappa']:.3f}  {cal['samples']:<5}")

    # Show prompt files if any
    print(f"\n{'='*80}")
    print("Optimized prompts:")
    for metric_dir in calibrations_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        prompts_dir = metric_dir / "prompts"
        if prompts_dir.exists():
            full_prompts = list(prompts_dir.glob("*_full.txt"))
            if full_prompts:
                latest = sorted(full_prompts, reverse=True)[0]
                print(f"  {metric_dir.name}: {latest}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of a dataset including items, metrics, runs, annotations, and calibrations."""
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, 'dataset', None),
        getattr(args, 'latest', False),
        config
    )

    if not dataset_path:
        # Try to show available datasets
        print("No dataset specified. Available datasets:")
        data_dir = Path("data")
        if data_dir.exists():
            datasets = [d for d in data_dir.iterdir() if d.is_dir() and (d / "dataset.jsonl").exists()]
            datasets.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            for d in datasets[:10]:
                mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {d.name:<40} (modified: {mtime})")
            if len(datasets) > 10:
                print(f"  ... and {len(datasets) - 10} more")
        print("\nUse: evalyn status --dataset <path> or evalyn status --latest")
        return

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"DATASET STATUS: {dataset_path.name}")
    print(f"{'='*60}")
    print(f"Path: {dataset_path}")

    # Dataset items
    dataset_file = dataset_path / "dataset.jsonl"
    item_count = 0
    if dataset_file.exists():
        with open(dataset_file) as f:
            item_count = sum(1 for _ in f)
    print(f"\n--- DATASET ---")
    print(f"Items: {item_count}")

    # Meta info
    meta_file = dataset_path / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
            if "project" in meta:
                print(f"Project: {meta.get('project')}")
            if "version" in meta:
                print(f"Version: {meta.get('version')}")

    # Metrics
    metrics_dir = dataset_path / "metrics"
    print(f"\n--- METRICS ---")
    if metrics_dir.exists():
        metric_files = list(metrics_dir.glob("*.json"))
        print(f"Metric sets: {len(metric_files)}")
        for mf in sorted(metric_files):
            try:
                with open(mf) as f:
                    metrics = json.load(f)
                    count = len(metrics) if isinstance(metrics, list) else 0
                    print(f"  {mf.name}: {count} metrics")
            except Exception:
                print(f"  {mf.name}: (error reading)")
    else:
        print("No metrics defined yet")
        print("  → Run: evalyn suggest-metrics --target <func> --dataset " + str(dataset_path))

    # Eval runs
    eval_runs_dir = dataset_path / "eval_runs"
    print(f"\n--- EVAL RUNS ---")
    if eval_runs_dir.exists():
        run_files = list(eval_runs_dir.glob("*.json"))
        print(f"Eval runs: {len(run_files)}")
        # Show latest 3
        for rf in sorted(run_files, reverse=True)[:3]:
            try:
                with open(rf) as f:
                    run = json.load(f)
                    created = run.get("created_at", "")[:19]
                    results_count = len(run.get("metric_results", []))
                    print(f"  {rf.stem}: {results_count} results ({created})")
            except Exception:
                pass
    else:
        print("No eval runs yet")
        print("  → Run: evalyn run-eval --dataset " + str(dataset_path))

    # Annotations
    annotations_file = dataset_path / "annotations.jsonl"
    print(f"\n--- ANNOTATIONS ---")
    if annotations_file.exists():
        with open(annotations_file) as f:
            ann_count = sum(1 for _ in f)
        coverage = f"{ann_count}/{item_count}" if item_count > 0 else str(ann_count)
        pct = f" ({ann_count/item_count:.0%})" if item_count > 0 else ""
        print(f"Annotated: {coverage}{pct}")
    else:
        print("No annotations yet")
        print("  → Run: evalyn annotate --dataset " + str(dataset_path))

    # Calibrations
    calibrations_dir = dataset_path / "calibrations"
    print(f"\n--- CALIBRATIONS ---")
    if calibrations_dir.exists():
        cal_count = 0
        metrics_with_cal = []
        for metric_dir in calibrations_dir.iterdir():
            if metric_dir.is_dir():
                cals = list(metric_dir.glob("*.json"))
                if cals:
                    cal_count += len(cals)
                    metrics_with_cal.append(metric_dir.name)
        if cal_count > 0:
            print(f"Calibrations: {cal_count} across {len(metrics_with_cal)} metrics")
            for m in metrics_with_cal[:5]:
                prompts_dir = calibrations_dir / m / "prompts"
                has_prompt = "✓ prompt" if prompts_dir.exists() and list(prompts_dir.glob("*_full.txt")) else ""
                print(f"  {m} {has_prompt}")
        else:
            print("No calibrations yet")
    else:
        print("No calibrations yet")
        print("  → Run: evalyn calibrate --metric-id <metric> --annotations ... --dataset " + str(dataset_path))

    # Suggested next step
    print(f"\n{'='*60}")
    print("SUGGESTED NEXT STEP:")
    if not metrics_dir.exists() or not list(metrics_dir.glob("*.json")):
        print("  evalyn suggest-metrics --target <module:func> --dataset " + str(dataset_path))
    elif not eval_runs_dir.exists() or not list(eval_runs_dir.glob("*.json")):
        print("  evalyn run-eval --dataset " + str(dataset_path))
    elif not annotations_file.exists():
        print("  evalyn annotate --dataset " + str(dataset_path))
    elif not calibrations_dir.exists():
        print("  evalyn calibrate --metric-id <metric> --annotations " + str(annotations_file) + " --dataset " + str(dataset_path))
    else:
        print("  All steps complete! Consider:")
        print("  - Re-run eval with optimized prompts")
        print("  - Generate synthetic data: evalyn simulate --dataset " + str(dataset_path))


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
        headers = ["id", "scope", "category", "config", "desc"]
        rows = []
        for tpl in templates:
            rows.append(
                [
                    tpl.get("id", ""),
                    tpl.get("scope", "overall"),
                    tpl.get("category", ""),
                    _config_summary(tpl.get("config", {})),
                    _compact_text(tpl.get("description", ""), 50),
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


def cmd_simulate(args: argparse.Namespace) -> None:
    """Generate synthetic test data using LLM-based user simulation."""
    from .models import DatasetItem
    from .decorators import eval as eval_decorator

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        dataset_file = dataset_path / "dataset.jsonl"
    else:
        dataset_file = dataset_path
        dataset_path = dataset_file.parent

    if not dataset_file.exists():
        print(f"Error: Dataset not found at {dataset_file}")
        return

    # Load seed dataset
    seed_items = load_dataset(dataset_file)
    if not seed_items:
        print("Error: No items found in seed dataset")
        return

    print(f"Loaded {len(seed_items)} seed items from {dataset_file}")

    # Load target function if provided
    target_fn = None
    if args.target:
        try:
            target_fn = _load_callable(args.target)
            print(f"Loaded target function: {args.target}")
        except Exception as e:
            print(f"Warning: Could not load target function: {e}")
            print("Simulation will generate queries only (no agent execution)")

    # Parse modes
    modes = [m.strip() for m in args.modes.split(",")]
    valid_modes = {"similar", "outlier"}
    modes = [m for m in modes if m in valid_modes]
    if not modes:
        modes = ["similar", "outlier"]

    print(f"Simulation modes: {modes}")

    # Build config
    config = SimulationConfig(
        num_similar=args.num_similar,
        num_outlier=args.num_outlier,
        model=args.model,
        temperature_similar=args.temp_similar,
        temperature_outlier=args.temp_outlier,
        max_seed_items=args.max_seeds,
    )

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: create sim-<timestamp> folder inside dataset directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = dataset_path / f"simulations"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if target_fn:
        # Wrap target function with is_simulation=True for all simulation traces
        target_fn_wrapped = eval_decorator(is_simulation=True)(target_fn)

        # Full simulation: generate + run agent
        simulator = AgentSimulator(
            target_fn=target_fn_wrapped,
            config=config,
            model=args.model,
        )

        with Spinner("Running agent simulation"):
            results = simulator.run(
                seed_dataset=seed_items,
                output_dir=output_dir,
                modes=modes,
            )

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        for mode, path in results.items():
            # Count items
            dataset_file = path / "dataset.jsonl"
            if dataset_file.exists():
                with open(dataset_file) as f:
                    count = sum(1 for _ in f)
                print(f"  {mode}: {count} items -> {path}")
            else:
                print(f"  {mode}: -> {path}")
    else:
        # Query generation only (no target function)
        user_sim = UserSimulator(model=args.model)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for mode in modes:
            print(f"\nGenerating {mode} queries...")
            with Spinner(f"Generating {mode} queries"):
                if mode == "similar":
                    generated = user_sim.generate_similar(
                        seed_items[:config.max_seed_items],
                        num_per_seed=config.num_similar,
                    )
                else:
                    generated = user_sim.generate_outliers(
                        seed_items[:config.max_seed_items],
                        num_per_seed=config.num_outlier,
                    )

            if not generated:
                print(f"  No queries generated for mode={mode}")
                continue

            # Save generated queries
            mode_dir = output_dir / f"queries-{mode}-{timestamp}"
            mode_dir.mkdir(parents=True, exist_ok=True)

            queries_file = mode_dir / "queries.jsonl"
            with open(queries_file, "w", encoding="utf-8") as f:
                for gq in generated:
                    f.write(json.dumps({
                        "query": gq.query,
                        "mode": gq.mode,
                        "seed_id": gq.seed_id,
                        "generation_reason": gq.generation_reason,
                    }, ensure_ascii=False) + "\n")

            # Save meta
            meta = {
                "type": "generated_queries",
                "mode": mode,
                "created_at": datetime.now().isoformat(),
                "seed_dataset": str(dataset_file),
                "num_queries": len(generated),
                "config": {
                    "model": config.model,
                    "num_per_seed": config.num_similar if mode == "similar" else config.num_outlier,
                    "temperature": config.temperature_similar if mode == "similar" else config.temperature_outlier,
                },
            }
            with open(mode_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Generated {len(generated)} {mode} queries -> {mode_dir}")

        print(f"\n{'='*60}")
        print("QUERY GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"To run these queries through your agent, use --target flag")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize configuration file with default settings."""
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return

    template = """# Evalyn Configuration

# API Keys (use env vars or set directly)
api_keys:
  gemini: "${GEMINI_API_KEY}"
  openai: "${OPENAI_API_KEY}"

# Default model for LLM operations
model: "gemini-2.5-flash-lite"

# Default project settings
defaults:
  project: null      # e.g., "myproject"
  version: null      # e.g., "v1"
"""
    with open(output_path, "w") as f:
        f.write(template)

    print(f"Created {output_path}")
    print(f"\nSet your API key:")
    print(f"  export GEMINI_API_KEY='your-key'")
    print(f"  # or edit {output_path} directly")


def cmd_one_click(args: argparse.Namespace) -> None:
    """Run the complete evaluation pipeline from dataset building to calibrated evaluation."""
    from datetime import datetime
    from .storage import SQLiteStorage

    # If no version specified, query available versions and prompt
    if not args.version:
        storage = SQLiteStorage()
        calls = storage.list_calls(limit=500)
        versions = set()
        for call in calls:
            meta = call.metadata or {}
            if meta.get("project_name") == args.project or meta.get("project_id") == args.project:
                v = meta.get("version")
                if v:
                    versions.add(v)

        if len(versions) == 0:
            print(f"No versions found for project '{args.project}'")
            print("Proceeding without version filter (all traces)...")
        elif len(versions) == 1:
            args.version = list(versions)[0]
            print(f"Auto-selected version: {args.version}")
        else:
            print(f"\nAvailable versions for project '{args.project}':")
            version_list = sorted(versions)
            for i, v in enumerate(version_list, 1):
                print(f"  {i}. {v}")
            print(f"  {len(version_list) + 1}. [all versions]")

            try:
                choice = input("\nSelect version (number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(version_list):
                    args.version = version_list[idx]
                elif idx == len(version_list):
                    args.version = None  # all versions
                else:
                    print("Invalid selection, using all versions")
            except (ValueError, EOFError):
                print("Invalid input, using all versions")

    # Print header
    print("\n" + "=" * 70)
    print(" " * 15 + "EVALYN ONE-CLICK EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nProject:  {args.project}")
    if args.target:
        print(f"Target:   {args.target}")
    if args.version:
        print(f"Version:  {args.version}")
    print(f"Mode:     {args.metric_mode}" + (f" ({args.model})" if args.metric_mode != "basic" else ""))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        version_str = f"-{args.version}" if args.version else ""
        output_dir = Path("data") / f"{args.project}{version_str}-{timestamp}-oneclick"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:   {output_dir}")
    print("\n" + "-" * 70 + "\n")

    if args.dry_run:
        print("DRY RUN MODE - showing what would be done:\n")

    # Initialize state tracking
    state = {
        "started_at": datetime.now().isoformat(),
        "config": vars(args),
        "steps": {},
        "output_dir": str(output_dir)
    }

    try:
        # Step 1: Build Dataset
        print("[1/7] Building Dataset")
        dataset_dir = output_dir / "1_dataset"
        dataset_dir.mkdir(exist_ok=True)

        if args.dry_run:
            print(f"  → Would build dataset with: project={args.project}, limit={args.dataset_limit}")
            print(f"  → Would save to: {dataset_dir}/dataset.jsonl\n")
        else:
            from .datasets import build_dataset_from_storage, save_dataset_with_meta
            from .storage import SQLiteStorage
            from datetime import datetime as dt

            # Parse date filters
            since = dt.fromisoformat(args.since) if args.since else None
            until = dt.fromisoformat(args.until) if args.until else None

            storage = SQLiteStorage()
            items = build_dataset_from_storage(
                storage,
                project_name=args.project,
                version=args.version,
                production_only=args.production_only,
                simulation_only=args.simulation_only,
                since=since,
                until=until,
                limit=args.dataset_limit,
                success_only=True,
                include_metadata=True
            )

            if not items:
                print(f"  ✗ No traces found matching filters")
                return

            meta = {
                "project": args.project,
                "version": args.version or "all",
                "created_at": datetime.now().isoformat(),
                "filters": {
                    "production_only": args.production_only,
                    "simulation_only": args.simulation_only,
                    "since": args.since,
                    "until": args.until,
                },
                "item_count": len(items)
            }

            dataset_path = save_dataset_with_meta(items, dataset_dir, meta)
            print(f"  ✓ Found {len(items)} items")
            print(f"  ✓ Saved to: {dataset_path}\n")

            state["steps"]["1_dataset"] = {
                "status": "success",
                "output": str(dataset_path),
                "item_count": len(items)
            }

        # Step 2: Suggest Metrics
        print("[2/7] Suggesting Metrics")
        metrics_dir = output_dir / "2_metrics"
        metrics_dir.mkdir(exist_ok=True)

        if args.dry_run:
            print(f"  → Would suggest metrics with mode={args.metric_mode}")
            print(f"  → Would save to: {metrics_dir}/suggested.json\n")
        else:
            # Call suggest-metrics logic
            metrics_path = metrics_dir / "suggested.json"

            # Create a mock args object for suggest_metrics
            suggest_args = argparse.Namespace(
                target=args.target,
                num_traces=5,
                num_metrics=None,
                mode=args.metric_mode,
                llm_mode=args.llm_mode if args.metric_mode in ["llm-registry", "llm-brainstorm"] else None,
                model=args.model,
                api_base=None,
                api_key=None,
                llm_caller=None,
                bundle=args.bundle,
                dataset=str(dataset_dir),
                metrics_name="suggested"
            )

            # Run suggest metrics (this will save to dataset/metrics/suggested.json)
            from .metrics.suggester import HeuristicSuggester, LLMSuggester, LLMRegistrySelector
            from .metrics.registry import MetricRegistry

            # Load target function if provided
            target_fn = None
            if args.target:
                target_fn = load_target_from_spec(args.target)
                if not target_fn:
                    print(f"  ⚠ Could not load target function: {args.target}")
                    print(f"  → Using trace-based suggestions only")

            # Get sample traces
            from .storage import SQLiteStorage
            storage = SQLiteStorage()
            calls = storage.list_calls(limit=5)

            # Suggest metrics based on mode
            if args.metric_mode == "basic":
                suggester = HeuristicSuggester()
                metric_specs = suggester.suggest(target_fn, calls)
            elif args.metric_mode == "llm-registry":
                selector = LLMRegistrySelector(model=args.model, llm_mode=args.llm_mode)
                metric_specs = selector.select_metrics(target_fn, calls, num_metrics=args.num_metrics)
            elif args.metric_mode == "llm-brainstorm":
                suggester = LLMSuggester(model=args.model, llm_mode=args.llm_mode)
                metric_specs = suggester.suggest(target_fn, calls, num_metrics=args.num_metrics)
            else:  # bundle
                from .metrics.bundles import get_bundle_metrics
                metric_specs = get_bundle_metrics(args.bundle) if args.bundle else []

            # Save metrics
            payload = []
            for spec in metric_specs:
                payload.append({
                    "id": spec.id,
                    "name": getattr(spec, "name", spec.id),
                    "type": spec.type,
                    "description": spec.description,
                    "config": spec.config,
                    "why": getattr(spec, "why", ""),
                })
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)

            obj_count = sum(1 for spec in metric_specs if spec.type == "objective")
            subj_count = sum(1 for spec in metric_specs if spec.type == "subjective")

            print(f"  ✓ Selected {len(metric_specs)} metrics ({obj_count} objective, {subj_count} subjective)")
            for spec in metric_specs:
                print(f"    - {spec.id} ({spec.type})")
            print(f"  ✓ Saved to: {metrics_path}\n")

            state["steps"]["2_metrics"] = {
                "status": "success",
                "output": str(metrics_path),
                "total": len(metric_specs),
                "objective": obj_count,
                "subjective": subj_count
            }

        # Step 3: Run Initial Evaluation
        print("[3/7] Running Initial Evaluation")
        eval_dir = output_dir / "3_initial_eval"
        eval_dir.mkdir(exist_ok=True)

        if args.dry_run:
            print(f"  → Would run evaluation on {args.dataset_limit} items")
            print(f"  → Would save to: {eval_dir}/\n")
        else:
            # Run evaluation
            from .runner import EvalRunner
            from .datasets import load_dataset
            from .metrics.factory import build_objective_metric, build_subjective_metric

            # Load metrics from file
            with open(metrics_path) as f:
                metrics_data = json.load(f)

            metrics = []
            for spec_data in metrics_data:
                spec = MetricSpec(
                    id=spec_data["id"],
                    name=spec_data.get("name", spec_data["id"]),
                    type=spec_data["type"],
                    description=spec_data.get("description", ""),
                    config=spec_data.get("config", {}),
                )
                try:
                    if spec.type == "objective":
                        m = build_objective_metric(spec.id, spec.config)
                    else:
                        m = build_subjective_metric(spec.id, spec.config)
                    if m:
                        metrics.append(m)
                except Exception:
                    pass

            items = list(load_dataset(dataset_path))
            runner = EvalRunner(
                target_fn=target_fn or (lambda: None),
                metrics=metrics,
                dataset_name=args.project,
                instrument=False,  # Don't re-run, use existing outputs
            )
            eval_run = runner.run_dataset(items, use_synthetic=True)

            # Save run
            run_path = eval_dir / f"run_{timestamp}_{eval_run.id[:8]}.json"
            with open(run_path, "w") as f:
                json.dump(eval_run.as_dict(), f, indent=2)

            print(f"  ✓ Evaluated {len(items)} items")
            print(f"  RESULTS:")
            for metric_id, summary in eval_run.summary.items():
                if "pass_rate" in summary:
                    print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
                elif "avg" in summary:
                    print(f"    {metric_id}: avg={summary['avg']:.1f}")
            print(f"  ✓ Saved to: {run_path}\n")

            state["steps"]["3_initial_eval"] = {
                "status": "success",
                "output": str(run_path),
                "run_id": eval_run.id
            }

        # Step 4: Human Annotation (optional)
        if args.skip_annotation:
            print("[4/7] Human Annotation")
            print("  ⏭️  SKIPPED (--skip-annotation)\n")
            state["steps"]["4_annotation"] = {"status": "skipped"}
        else:
            print("[4/7] Human Annotation")
            if args.dry_run:
                print(f"  → Would annotate {args.annotation_limit} items")
                print(f"  → Mode: {'per-metric' if args.per_metric else 'overall'}\n")
            else:
                ann_dir = output_dir / "4_annotations"
                ann_dir.mkdir(exist_ok=True)
                ann_path = ann_dir / "annotations.jsonl"

                print(f"  → Annotating {args.annotation_limit} items...")
                print(f"  → Interactive annotation mode")
                print(f"  → Press Ctrl+C to skip this step\n")

                try:
                    # Call interactive annotation
                    ann_args = argparse.Namespace(
                        dataset=str(dataset_dir),
                        latest=False,
                        run_id=None,
                        output=str(ann_path),
                        annotator="human",
                        restart=False,
                        per_metric=args.per_metric,
                        only_disagreements=False
                    )
                    cmd_annotate(ann_args)

                    # Count annotations
                    if ann_path.exists():
                        with open(ann_path) as f:
                            ann_count = sum(1 for _ in f)
                        print(f"  ✓ Completed {ann_count} annotations")
                        print(f"  ✓ Saved to: {ann_path}\n")
                        state["steps"]["4_annotation"] = {
                            "status": "success",
                            "output": str(ann_path),
                            "count": ann_count
                        }
                    else:
                        print("  ⏭️  No annotations created\n")
                        state["steps"]["4_annotation"] = {"status": "skipped"}
                except KeyboardInterrupt:
                    print("\n  ⏭️  Annotation interrupted by user\n")
                    state["steps"]["4_annotation"] = {"status": "interrupted"}

        # Step 5: Calibrate LLM Judges (optional)
        has_annotations = (output_dir / "4_annotations" / "annotations.jsonl").exists()

        if args.skip_calibration or not has_annotations:
            print("[5/7] Calibrating LLM Judges")
            if args.skip_calibration:
                print("  ⏭️  SKIPPED (--skip-calibration)\n")
            else:
                print("  ⏭️  SKIPPED (requires annotations)\n")
            state["steps"]["5_calibration"] = {"status": "skipped"}
        else:
            print("[5/7] Calibrating LLM Judges")
            if args.dry_run:
                print(f"  → Would calibrate subjective metrics")
                print(f"  → Optimizer: {args.optimizer}\n")
            else:
                cal_dir = output_dir / "5_calibrations"
                cal_dir.mkdir(exist_ok=True)

                ann_path = output_dir / "4_annotations" / "annotations.jsonl"

                # Get subjective metrics
                subj_metrics = [spec for spec in metric_specs if spec.type == "subjective"]

                print(f"  → Calibrating {len(subj_metrics)} subjective metrics...\n")

                for spec in subj_metrics:
                    print(f"  [{spec.id}]")

                    # Run calibration
                    cal_args = argparse.Namespace(
                        metric_id=spec.id,
                        annotations=str(ann_path),
                        run_id=None,
                        threshold=0.5,
                        dataset=str(dataset_dir),
                        latest=False,
                        no_optimize=False,
                        optimizer=args.optimizer,
                        model=args.model,
                        gepa_task_lm="gemini/gemini-2.5-flash",
                        gepa_reflection_lm="gemini/gemini-2.5-flash",
                        gepa_max_calls=150,
                        show_examples=False,
                        output=None,
                        format="table"
                    )

                    try:
                        cmd_calibrate(cal_args)
                        print()
                    except Exception as e:
                        print(f"    ✗ Calibration failed: {e}\n")

                state["steps"]["5_calibration"] = {
                    "status": "success",
                    "metrics_calibrated": len(subj_metrics)
                }

        # Step 6: Re-evaluate with Calibrated Prompts (optional)
        has_calibrations = (output_dir / "5_calibrations").exists()

        if not has_calibrations:
            print("[6/7] Re-evaluating with Calibrated Prompts")
            print("  ⏭️  SKIPPED (no calibrations)\n")
            state["steps"]["6_calibrated_eval"] = {"status": "skipped"}
        else:
            print("[6/7] Re-evaluating with Calibrated Prompts")
            if args.dry_run:
                print(f"  → Would re-run evaluation with calibrated prompts\n")
            else:
                eval2_dir = output_dir / "6_calibrated_eval"
                eval2_dir.mkdir(exist_ok=True)

                # Re-run evaluation with calibrated prompts
                from .runner import EvalRunner
                from .datasets import load_dataset
                from .metrics.factory import build_objective_metric, build_subjective_metric
                from .calibration import load_optimized_prompt

                # Load metrics from file
                with open(metrics_path) as f:
                    metrics_data = json.load(f)

                metrics = []
                calibrated_count = 0
                for spec_data in metrics_data:
                    spec = MetricSpec(
                        id=spec_data["id"],
                        name=spec_data.get("name", spec_data["id"]),
                        type=spec_data["type"],
                        description=spec_data.get("description", ""),
                        config=spec_data.get("config", {}),
                    )
                    # Load calibrated prompt for subjective metrics
                    if spec.type == "subjective":
                        optimized_prompt = load_optimized_prompt(str(dataset_dir), spec.id)
                        if optimized_prompt:
                            spec.config = dict(spec.config or {})
                            spec.config["prompt"] = optimized_prompt
                            calibrated_count += 1
                    try:
                        if spec.type == "objective":
                            m = build_objective_metric(spec.id, spec.config)
                        else:
                            m = build_subjective_metric(spec.id, spec.config)
                        if m:
                            metrics.append(m)
                    except Exception:
                        pass

                items = list(load_dataset(dataset_path))
                runner = EvalRunner(
                    target_fn=target_fn or (lambda: None),
                    metrics=metrics,
                    dataset_name=args.project,
                    instrument=False,
                )
                eval_run2 = runner.run_dataset(items, use_synthetic=True)

                # Save run
                run_path2 = eval2_dir / f"run_{timestamp}_{eval_run2.id[:8]}.json"
                with open(run_path2, "w") as f:
                    json.dump(eval_run2.as_dict(), f, indent=2)

                print(f"  ✓ Used {calibrated_count} calibrated prompts")
                print(f"  ✓ Evaluated {len(items)} items")
                print(f"  RESULTS:")
                for metric_id, summary in eval_run2.summary.items():
                    if "pass_rate" in summary:
                        print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
                print(f"  ✓ Saved to: {run_path2}\n")

                state["steps"]["6_calibrated_eval"] = {
                    "status": "success",
                    "output": str(run_path2),
                    "calibrated_count": calibrated_count
                }

        # Step 7: Generate Simulations (optional)
        if not args.enable_simulation:
            print("[7/7] Generating Simulations")
            print("  ⏭️  SKIPPED (use --enable-simulation to enable)\n")
            state["steps"]["7_simulation"] = {"status": "skipped"}
        elif not args.target:
            print("[7/7] Generating Simulations")
            print("  ⏭️  SKIPPED (--target required for simulation)\n")
            state["steps"]["7_simulation"] = {"status": "skipped", "reason": "no target"}
        else:
            print("[7/7] Generating Simulations")
            if args.dry_run:
                print(f"  → Would generate simulations: modes={args.simulation_modes}\n")
            else:
                sim_dir = output_dir / "7_simulations"
                sim_dir.mkdir(exist_ok=True)

                # Run simulation
                sim_args = argparse.Namespace(
                    dataset=str(dataset_dir),
                    target=args.target,
                    output=str(sim_dir),
                    modes=args.simulation_modes,
                    num_similar=args.num_similar,
                    num_outlier=args.num_outlier,
                    max_seeds=args.max_sim_seeds,
                    model=args.model,
                    temp_similar=0.3,
                    temp_outlier=0.8
                )

                try:
                    cmd_simulate(sim_args)
                    print(f"  ✓ Simulations generated")
                    print(f"  ✓ Saved to: {sim_dir}/\n")
                    state["steps"]["7_simulation"] = {
                        "status": "success",
                        "output": str(sim_dir)
                    }
                except Exception as e:
                    print(f"  ✗ Simulation failed: {e}\n")
                    state["steps"]["7_simulation"] = {"status": "failed", "error": str(e)}

        # Save pipeline state
        state["completed_at"] = datetime.now().isoformat()
        state_path = output_dir / "pipeline_summary.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        # Final summary
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print(f"\nSummary:")
        for step_name, step_info in state["steps"].items():
            status = step_info.get("status", "unknown")
            status_icon = "✓" if status == "success" else ("⏭️" if status == "skipped" else "✗")
            print(f"  {status_icon} {step_name}: {status}")

        print(f"\nNext steps:")
        if "6_calibrated_eval" in state["steps"] and state["steps"]["6_calibrated_eval"]["status"] == "success":
            print(f"  1. Review results: cat {state['steps']['6_calibrated_eval']['output']}")
        elif "3_initial_eval" in state["steps"]:
            print(f"  1. Review results: cat {state['steps']['3_initial_eval']['output']}")
        print(f"  2. View full summary: cat {state_path}")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print(f"Partial results saved to: {output_dir}")
        print(f"Resume or inspect: cd {output_dir}\n")
    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {e}")
        print(f"Partial results saved to: {output_dir}\n")
        import traceback
        if args.verbose:
            traceback.print_exc()


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

        # Dont change this ascii art
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
    list_parser.add_argument("--simulation", action="store_true", help="Show only simulation traces")
    list_parser.add_argument("--production", action="store_true", help="Show only production traces")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    list_parser.set_defaults(func=cmd_list_calls)

    run_parser = subparsers.add_parser("run-eval", help="Run evaluation on dataset using specified metrics")
    run_parser.add_argument("--dataset", help="Path to JSON/JSONL dataset file or directory containing dataset.jsonl")
    run_parser.add_argument("--latest", action="store_true", help="Use the most recently modified dataset")
    run_parser.add_argument("--metrics", help="Path to metrics JSON file(s), comma-separated for multiple (auto-detected from meta.json if omitted)")
    run_parser.add_argument("--metrics-all", action="store_true", help="Use all metrics files from the metrics/ folder")
    run_parser.add_argument("--use-calibrated", action="store_true", help="Use calibrated prompts for subjective metrics (if available)")
    run_parser.add_argument("--dataset-name", help="Name for the eval run (defaults to dataset filename)")
    run_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    run_parser.set_defaults(func=cmd_run_eval)

    suggest_parser = subparsers.add_parser("suggest-metrics", help="Suggest metrics for a project or target function")
    suggest_parser.add_argument("--project", help="Project name (use 'evalyn show-projects' to see available projects)")
    suggest_parser.add_argument("--version", help="Filter by version (optional, used with --project)")
    suggest_parser.add_argument("--target", help="Callable to analyze in the form module:function (alternative to --project)")
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
    suggest_parser.add_argument(
        "--scope",
        choices=["all", "overall", "llm_call", "tool_call", "trace"],
        default="all",
        help="Filter metrics by scope: overall (final output), llm_call (per LLM), tool_call (per tool), trace (aggregates), or all.",
    )
    suggest_parser.set_defaults(func=cmd_suggest_metrics)

    build_ds = subparsers.add_parser("build-dataset", help="Build dataset from stored traces")
    build_ds.add_argument("--output", help="Path to write dataset JSONL (default: data/<project>-<version>-<timestamp>.jsonl)")
    build_ds.add_argument("--project", help="Filter by metadata.project_id or project_name (recommended grouping)")
    build_ds.add_argument("--version", help="Filter by metadata.version")
    build_ds.add_argument("--simulation", action="store_true", help="Include only simulation traces")
    build_ds.add_argument("--production", action="store_true", help="Include only production traces")
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

    annotate_parser = subparsers.add_parser("annotate", help="Interactive CLI for annotating dataset items")
    annotate_parser.add_argument("--dataset", help="Path to dataset directory or dataset.jsonl file")
    annotate_parser.add_argument("--latest", action="store_true", help="Use the most recently modified dataset")
    annotate_parser.add_argument("--run-id", help="Eval run ID to show LLM judge results (defaults to latest)")
    annotate_parser.add_argument("--output", help="Output path for annotations (defaults to <dataset>/annotations.jsonl)")
    annotate_parser.add_argument("--annotator", default="human", help="Annotator name/id (default: human)")
    annotate_parser.add_argument("--restart", action="store_true", help="Restart annotation from scratch (ignore existing)")
    annotate_parser.add_argument("--per-metric", action="store_true", help="Annotate each metric separately (agree/disagree with LLM)")
    annotate_parser.add_argument("--spans", action="store_true", help="Annotate individual spans (LLM calls, tool calls, etc.)")
    annotate_parser.add_argument("--span-type", choices=["all", "llm_call", "tool_call", "reasoning", "retrieval"], default="all", help="Filter span types to annotate")
    annotate_parser.add_argument("--only-disagreements", action="store_true", help="Only show items where LLM and prior human labels disagree")
    annotate_parser.set_defaults(func=cmd_annotate)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate a subjective metric using human annotations")
    calibrate_parser.add_argument("--metric-id", required=True, help="Metric ID to calibrate (usually the judge metric id)")
    calibrate_parser.add_argument("--annotations", required=True, help="Path to annotations JSONL (target_id must match call_id)")
    calibrate_parser.add_argument("--run-id", help="Eval run id to calibrate; defaults to latest run")
    calibrate_parser.add_argument("--threshold", type=float, default=0.5, help="Current threshold for pass/fail")
    calibrate_parser.add_argument("--dataset", help="Path to dataset folder (provides input/output context for optimization)")
    calibrate_parser.add_argument("--latest", action="store_true", help="Use the most recently modified dataset")
    calibrate_parser.add_argument("--no-optimize", action="store_true", help="Skip LLM-based prompt optimization")
    calibrate_parser.add_argument("--optimizer", choices=["llm", "gepa"], default="llm", help="Optimization method: 'llm' (default) or 'gepa' (evolutionary)")
    calibrate_parser.add_argument("--model", default="gemini-2.5-flash-lite", help="LLM model for prompt optimization (llm mode)")
    # GEPA-specific options
    calibrate_parser.add_argument("--gepa-task-lm", default="gemini/gemini-2.5-flash", help="Task model for GEPA (model being optimized)")
    calibrate_parser.add_argument("--gepa-reflection-lm", default="gemini/gemini-2.5-flash", help="Reflection model for GEPA (strong model for reflection)")
    calibrate_parser.add_argument("--gepa-max-calls", type=int, default=150, help="Max metric calls budget for GEPA optimization")
    calibrate_parser.add_argument("--show-examples", action="store_true", help="Show example disagreement cases")
    calibrate_parser.add_argument("--output", help="Path to save calibration record JSON")
    calibrate_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    calibrate_parser.set_defaults(func=cmd_calibrate)

    simulate_parser = subparsers.add_parser("simulate", help="Generate synthetic test data using LLM-based user simulation")
    simulate_parser.add_argument("--dataset", required=True, help="Path to seed dataset directory or dataset.jsonl")
    simulate_parser.add_argument("--target", help="Target function to run queries against (module:func or path/to/file.py:func)")
    simulate_parser.add_argument("--output", help="Output directory for simulated data (default: <dataset>/simulations/)")
    simulate_parser.add_argument("--modes", default="similar,outlier", help="Simulation modes: similar,outlier (comma-separated)")
    simulate_parser.add_argument("--num-similar", type=int, default=3, help="Number of similar variations per seed item (default: 3)")
    simulate_parser.add_argument("--num-outlier", type=int, default=1, help="Number of outlier/edge cases per seed item (default: 1)")
    simulate_parser.add_argument("--max-seeds", type=int, default=50, help="Maximum seed items to use (default: 50)")
    simulate_parser.add_argument("--model", default="gemini-2.5-flash-lite", help="LLM model for query generation")
    simulate_parser.add_argument("--temp-similar", type=float, default=0.3, help="Temperature for similar queries (default: 0.3)")
    simulate_parser.add_argument("--temp-outlier", type=float, default=0.8, help="Temperature for outlier queries (default: 0.8)")
    simulate_parser.set_defaults(func=cmd_simulate)

    select_parser = subparsers.add_parser("select-metrics", help="LLM-guided selection from metric registry")
    select_parser.add_argument("--target", required=True, help="Callable to analyze in the form module:function")
    select_parser.add_argument("--llm-caller", required=True, help="Callable that accepts a prompt and returns metric ids or dicts")
    select_parser.add_argument("--limit", type=int, default=5, help="Recent traces to include as examples")
    select_parser.set_defaults(func=cmd_select_metrics)

    list_metrics = subparsers.add_parser("list-metrics", help="List available metric templates (objective + subjective)")
    list_metrics.set_defaults(func=cmd_list_metrics)

    # New commands
    status_parser = subparsers.add_parser("status", help="Show status of a dataset (items, metrics, runs, annotations, calibrations)")
    status_parser.add_argument("--dataset", help="Path to dataset directory")
    status_parser.add_argument("--latest", action="store_true", help="Use the most recently modified dataset")
    status_parser.set_defaults(func=cmd_status)

    list_cal_parser = subparsers.add_parser("list-calibrations", help="List calibration records for a dataset")
    list_cal_parser.add_argument("--dataset", help="Path to dataset directory")
    list_cal_parser.add_argument("--latest", action="store_true", help="Use the most recently modified dataset")
    list_cal_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")
    list_cal_parser.set_defaults(func=cmd_list_calibrations)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument("--output", default="evalyn.yaml", help="Output path for config file (default: evalyn.yaml)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config file")
    init_parser.set_defaults(func=cmd_init)

    # One-click pipeline
    oneclick_parser = subparsers.add_parser("one-click", help="Run complete evaluation pipeline (dataset -> metrics -> eval -> annotate -> calibrate)")
    oneclick_parser.add_argument("--project", required=True, help="Project name to filter traces")
    oneclick_parser.add_argument("--target", help="Target function (file.py:func or module:func). Optional - if not provided, uses existing trace outputs")
    oneclick_parser.add_argument("--version", help="Version filter (default: all versions)")
    oneclick_parser.add_argument("--production-only", action="store_true", help="Use only production traces")
    oneclick_parser.add_argument("--simulation-only", action="store_true", help="Use only simulation traces")
    oneclick_parser.add_argument("--output-dir", help="Custom output directory (default: auto-generated)")

    # Dataset options
    oneclick_parser.add_argument("--dataset-limit", type=int, default=100, help="Max dataset items (default: 100)")
    oneclick_parser.add_argument("--since", help="Filter traces since date (ISO format)")
    oneclick_parser.add_argument("--until", help="Filter traces until date (ISO format)")

    # Metrics options
    oneclick_parser.add_argument("--metric-mode", choices=["basic", "llm-registry", "llm-brainstorm", "bundle"], default="basic", help="Metric selection mode (default: basic)")
    oneclick_parser.add_argument("--llm-mode", choices=["api", "local"], default="api", help="LLM mode for metric selection (default: api)")
    oneclick_parser.add_argument("--model", default="gemini-2.5-flash-lite", help="LLM model name (default: gemini-2.5-flash-lite)")
    oneclick_parser.add_argument("--bundle", help="Bundle name (if metric-mode=bundle)")

    # Annotation options
    oneclick_parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation step (default: false)")
    oneclick_parser.add_argument("--annotation-limit", type=int, default=20, help="Max items to annotate (default: 20)")
    oneclick_parser.add_argument("--per-metric", action="store_true", help="Use per-metric annotation mode")

    # Calibration options
    oneclick_parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step (default: false)")
    oneclick_parser.add_argument("--optimizer", choices=["llm", "gepa"], default="llm", help="Prompt optimization method (default: llm)")
    oneclick_parser.add_argument("--calibrate-all-metrics", action="store_true", help="Calibrate all subjective metrics (default: only poorly-aligned)")

    # Simulation options
    oneclick_parser.add_argument("--enable-simulation", action="store_true", help="Enable simulation step (default: false)")
    oneclick_parser.add_argument("--simulation-modes", default="similar", help="Simulation modes: similar,outlier (default: similar)")
    oneclick_parser.add_argument("--num-similar", type=int, default=3, help="Similar queries per seed (default: 3)")
    oneclick_parser.add_argument("--num-outlier", type=int, default=2, help="Outlier queries per seed (default: 2)")
    oneclick_parser.add_argument("--max-sim-seeds", type=int, default=10, help="Max seeds for simulation (default: 10)")

    # Behavior options
    oneclick_parser.add_argument("--auto-yes", action="store_true", help="Skip all confirmation prompts (default: false)")
    oneclick_parser.add_argument("--verbose", action="store_true", help="Show detailed logs (default: false)")
    oneclick_parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing (default: false)")
    oneclick_parser.set_defaults(func=cmd_one_click)

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
