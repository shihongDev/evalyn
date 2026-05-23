"""Replay command: re-run a captured trace's prompts against a different model.

Answers "Should I switch to model X?" by extracting the LLM prompts from a
captured trace and re-executing them against a target model, then printing a
side-by-side comparison of cost, latency, and output.
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Tracing"

# Default visible params in the dashboard form.
ESSENTIAL = {"model", "provider"}

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...storage import SQLiteStorage

from ...evaluation.replay import ReplayComparison, build_replay_comparison
from ...storage.sqlite import DB_PATHS
from ..utils.command_common import resolve_call_id
from ..utils.errors import fatal_error
from ..utils.formatters import format_duration
from ..utils.rich import banner, kv, section


def _get_storage(args: argparse.Namespace) -> SQLiteStorage:
    """Get storage based on --db flag (same pattern as traces.py)."""
    from ...storage import SQLiteStorage

    db = getattr(args, "db", None)
    if db and db in DB_PATHS:
        return SQLiteStorage(DB_PATHS[db])
    from ...decorators import get_default_tracer

    return get_default_tracer().storage


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = 300) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + " [...truncated]"


def _format_cost(usd: float) -> str:
    if usd == 0:
        return "$0.0000"
    if abs(usd) < 0.0001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def _format_delta_cost(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{_format_cost(delta)}"


def _format_delta_ms(delta: float) -> str:
    return f"{'+' if delta >= 0 else ''}{delta:.1f} ms"


def _render_table(comparison: ReplayComparison) -> str:
    """Build the human-readable side-by-side comparison."""
    lines: list[str] = []
    lines.append("")
    lines.append(banner("REPLAY COMPARISON"))
    lines.append(
        kv(
            [
                ("Call", comparison.call_id),
                ("Function", comparison.function_name),
                ("Target provider", comparison.target_provider),
                ("Target model", comparison.target_model),
                ("LLM spans replayed", str(len(comparison.pairs))),
                ("LLM spans skipped", str(len(comparison.skipped))),
            ]
        )
    )

    if not comparison.pairs:
        lines.append("")
        lines.append("  No replayable LLM spans found in this trace.")
        if comparison.skipped:
            lines.append(
                f"  ({len(comparison.skipped)} llm_call span(s) had unparseable messages.)"
            )
        return "\n".join(lines)

    for idx, pair in enumerate(comparison.pairs, start=1):
        orig = pair.original
        rep = pair.replayed
        lines.append("")
        lines.append(section(f"LLM CALL {idx}: {orig.span_name}"))
        lines.append(
            "  Original ({provider}/{model}, {cost}, {dur}, {in_tok}->{out_tok} tok):".format(
                provider=orig.original_provider,
                model=orig.original_model,
                cost=_format_cost(orig.original_cost_usd),
                dur=format_duration(orig.original_duration_ms),
                in_tok=orig.original_input_tokens,
                out_tok=orig.original_output_tokens,
            )
        )
        lines.append(f"    {_truncate(orig.original_output, max_len=400)}")
        lines.append("")
        if rep.error:
            lines.append(
                "  Replayed ({provider}/{model}, ERROR after {dur}):".format(
                    provider=rep.new_provider,
                    model=rep.new_model,
                    dur=format_duration(rep.new_duration_ms),
                )
            )
            lines.append(f"    Error: {rep.error}")
        else:
            lines.append(
                "  Replayed ({provider}/{model}, {cost}, {dur}, {in_tok}->{out_tok} tok):".format(
                    provider=rep.new_provider,
                    model=rep.new_model,
                    cost=_format_cost(rep.new_cost_usd),
                    dur=format_duration(rep.new_duration_ms),
                    in_tok=rep.new_input_tokens,
                    out_tok=rep.new_output_tokens,
                )
            )
            lines.append(f"    {_truncate(rep.new_output, max_len=400)}")
            lines.append("")
            lines.append(
                "  Delta: cost {cost_d} | latency {lat_d}".format(
                    cost_d=_format_delta_cost(pair.cost_delta_usd),
                    lat_d=_format_delta_ms(pair.latency_delta_ms),
                )
            )

    lines.append("")
    lines.append(section("TOTALS"))
    lines.append(
        kv(
            [
                (
                    "Original cost",
                    _format_cost(comparison.total_original_cost_usd),
                ),
                (
                    "Replayed cost",
                    _format_cost(comparison.total_replayed_cost_usd),
                ),
                ("Cost delta", _format_delta_cost(comparison.total_cost_delta_usd)),
                (
                    "Original latency",
                    format_duration(comparison.total_original_latency_ms),
                ),
                (
                    "Replayed latency",
                    format_duration(comparison.total_replayed_latency_ms),
                ),
                (
                    "Latency delta",
                    _format_delta_ms(comparison.total_latency_delta_ms),
                ),
            ]
        )
    )
    lines.append("")
    return "\n".join(lines)


def _render_markdown(comparison: ReplayComparison) -> str:
    """Markdown-formatted comparison (for reports / sharing)."""
    lines = []
    lines.append(f"# Replay comparison: {comparison.call_id}")
    lines.append("")
    lines.append(f"- Function: `{comparison.function_name}`")
    lines.append(f"- Target: `{comparison.target_provider}/{comparison.target_model}`")
    lines.append(f"- Spans replayed: {len(comparison.pairs)}")
    lines.append(f"- Spans skipped: {len(comparison.skipped)}")
    lines.append("")
    for idx, pair in enumerate(comparison.pairs, start=1):
        orig = pair.original
        rep = pair.replayed
        lines.append(f"## LLM call {idx}: {orig.span_name}")
        lines.append("")
        lines.append(
            f"**Original** (`{orig.original_provider}/{orig.original_model}`, "
            f"{_format_cost(orig.original_cost_usd)}, "
            f"{format_duration(orig.original_duration_ms)}):"
        )
        lines.append("")
        lines.append("```")
        lines.append(_truncate(orig.original_output, max_len=600))
        lines.append("```")
        lines.append("")
        if rep.error:
            lines.append(f"**Replayed** (`{rep.new_provider}/{rep.new_model}`): ERROR")
            lines.append("")
            lines.append(f"`{rep.error}`")
        else:
            lines.append(
                f"**Replayed** (`{rep.new_provider}/{rep.new_model}`, "
                f"{_format_cost(rep.new_cost_usd)}, "
                f"{format_duration(rep.new_duration_ms)}):"
            )
            lines.append("")
            lines.append("```")
            lines.append(_truncate(rep.new_output, max_len=600))
            lines.append("```")
            lines.append("")
            lines.append(
                f"Delta: cost {_format_delta_cost(pair.cost_delta_usd)}, "
                f"latency {_format_delta_ms(pair.latency_delta_ms)}"
            )
        lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Original cost: {_format_cost(comparison.total_original_cost_usd)}")
    lines.append(f"- Replayed cost: {_format_cost(comparison.total_replayed_cost_usd)}")
    lines.append(f"- Cost delta: {_format_delta_cost(comparison.total_cost_delta_usd)}")
    lines.append(
        f"- Original latency: {format_duration(comparison.total_original_latency_ms)}"
    )
    lines.append(
        f"- Replayed latency: {format_duration(comparison.total_replayed_latency_ms)}"
    )
    lines.append(f"- Latency delta: {_format_delta_ms(comparison.total_latency_delta_ms)}")
    return "\n".join(lines)


def _render_json(comparison: ReplayComparison) -> str:
    return json.dumps(comparison.as_dict(), indent=2, default=str)


def _render_output(comparison: ReplayComparison, fmt: str) -> str:
    if fmt == "json":
        return _render_json(comparison)
    if fmt == "markdown":
        return _render_markdown(comparison)
    return _render_table(comparison)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a captured trace's LLM calls against a different model."""
    if not args.id:
        fatal_error("Must specify --id <call_id>")
    if not args.model:
        fatal_error("Must specify --model <target_model>")

    storage = _get_storage(args)
    if not storage:
        fatal_error("No storage configured")

    call_id = resolve_call_id(storage, args.id, strict_prefix_match=True)
    call = storage.get_call(call_id)
    if not call:
        fatal_error(f"No call found with id={call_id}")

    if not call.spans:
        fatal_error(
            "Call has no captured spans to replay",
            "Re-run with latest evalyn_sdk to capture LLM spans",
        )

    comparison = build_replay_comparison(
        call,
        target_provider=args.provider,
        target_model=args.model,
    )

    output = _render_output(comparison, args.format)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote replay comparison to {out_path}")
    else:
        print(output)

    unreplayable = len(comparison.skipped)
    if unreplayable and args.format != "json":
        print(
            f"\nNote: {unreplayable} llm_call span(s) were skipped (no parseable messages)."
        )


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "replay",
        help="Re-run a captured trace's LLM calls against a different model",
    )
    p.add_argument("--id", required=True, help="Call ID (short prefix OK) to replay")
    p.add_argument(
        "--model",
        required=True,
        help="Target model to replay against (e.g. gpt-4o-mini, claude-haiku-4-5)",
    )
    p.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "ollama"],
        default="gemini",
        help="Target LLM provider (default: gemini)",
    )
    p.add_argument(
        "--db",
        choices=["prod", "test"],
        help="Database to use: prod (default) or test",
    )
    p.add_argument(
        "--output",
        help="Write the comparison to this file instead of stdout",
    )
    p.add_argument(
        "--format",
        choices=["table", "json", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_replay)


__all__ = ["cmd_replay", "register_commands"]
