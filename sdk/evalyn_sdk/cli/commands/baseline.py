"""Baseline management commands: pin a run as a canonical reference.

Subcommands:
    evalyn baseline set    --run <id_or_path> [--name <label>]
    evalyn baseline show   [--name <label>]
    evalyn baseline list
    evalyn baseline clear  [--name <label>] [--all]

Pointers live in .evalyn/baselines.json (one entry per name). Other commands
(notably ``evalyn drift``) consume the pointers to compare new runs against
the pinned reference.
"""

from __future__ import annotations

# Dashboard catalog group.
GROUP = "Analysis"
ESSENTIAL = {"run", "name"}

import argparse
import json
from pathlib import Path

from ...analysis.baseline import (
    DEFAULT_BASELINE_NAME,
    BaselinePointer,
    clear_baseline,
    list_baselines,
    load_baseline,
    now_iso,
    save_baseline,
)
from ..utils.command_common import load_eval_run_for_command
from ..utils.errors import fatal_error


def _resolve_run(run_arg: str) -> tuple[object, Path | None]:
    """Resolve a --run argument (path or run ID) into (EvalRun, file_path)."""
    from ...models import EvalRun

    candidate = Path(run_arg)
    if candidate.exists():
        results_file = candidate / "results.json" if candidate.is_dir() else candidate
        if not results_file.exists():
            fatal_error(f"No results.json found at: {candidate}")
        with open(results_file, encoding="utf-8") as f:
            run = EvalRun.from_dict(json.load(f))
        return run, results_file

    loaded = load_eval_run_for_command(
        run_id=run_arg,
        error_message=f"Could not resolve run '{run_arg}'",
        error_hint="Pass a run ID from 'evalyn list-runs' or a path to results.json",
    )
    return loaded.run, loaded.run_file_path


def cmd_baseline_set(args: argparse.Namespace) -> None:
    """Pin a run as the named baseline."""
    if not args.run:
        fatal_error("--run is required", "evalyn baseline set --run <id_or_path>")

    run, run_path = _resolve_run(args.run)
    pointer = BaselinePointer(
        name=args.name or DEFAULT_BASELINE_NAME,
        run_id=run.id,
        run_path=str(run_path) if run_path else "",
        dataset_name=run.dataset_name,
        pinned_at=now_iso(),
    )
    out = save_baseline(pointer, state_dir=args.state_dir)
    print(f"Pinned baseline '{pointer.name}' -> run {pointer.run_id[:12]}")
    print(f"  dataset:  {pointer.dataset_name}")
    if pointer.run_path:
        print(f"  source:   {pointer.run_path}")
    print(f"  pinned:   {pointer.pinned_at}")
    print(f"  catalog:  {out}")


def cmd_baseline_show(args: argparse.Namespace) -> None:
    """Display a single baseline pointer."""
    name = args.name or DEFAULT_BASELINE_NAME
    pointer = load_baseline(name, state_dir=args.state_dir)
    if pointer is None:
        fatal_error(
            f"No baseline named '{name}'",
            "Pin one with: evalyn baseline set --run <id_or_path>",
        )
    print(f"Baseline: {pointer.name}")
    print(f"  run_id:       {pointer.run_id}")
    print(f"  dataset:      {pointer.dataset_name}")
    print(f"  pinned_at:    {pointer.pinned_at}")
    print(f"  source_path:  {pointer.run_path or '(unknown)'}")


def cmd_baseline_list(args: argparse.Namespace) -> None:
    """List all baselines stored in the catalog."""
    from ..utils.rich import table

    pointers = list_baselines(state_dir=args.state_dir)
    if not pointers:
        print("No baselines pinned yet.")
        print("Pin one with: evalyn baseline set --run <id_or_path>")
        return
    rows = [
        [p.name, p.run_id[:12], p.dataset_name, p.pinned_at[:19]]
        for p in pointers
    ]
    print(table(headers=["NAME", "RUN_ID", "DATASET", "PINNED_AT"], rows=rows))


def cmd_baseline_clear(args: argparse.Namespace) -> None:
    """Remove one or all baseline pointers."""
    if args.all:
        removed = clear_baseline(None, state_dir=args.state_dir)
        if removed:
            print("Cleared all baselines.")
        else:
            print("No baselines to clear.")
        return

    name = args.name or DEFAULT_BASELINE_NAME
    removed = clear_baseline(name, state_dir=args.state_dir)
    if removed:
        print(f"Cleared baseline '{name}'.")
    else:
        fatal_error(f"No baseline named '{name}'")


_BASELINE_HANDLERS = {
    "set": cmd_baseline_set,
    "show": cmd_baseline_show,
    "list": cmd_baseline_list,
    "clear": cmd_baseline_clear,
}


def cmd_baseline(args: argparse.Namespace) -> None:
    """Dispatch to the right baseline subcommand."""
    sub = getattr(args, "baseline_action", None)
    if sub is None:
        fatal_error(
            "Specify a baseline subcommand",
            "Try: evalyn baseline set|show|list|clear --help",
        )
    _BASELINE_HANDLERS[sub](args)


_STATE_DIR_HELP = "Directory holding baselines.json (default: ./.evalyn/)"
_NAME_HELP = f"Baseline label (default: {DEFAULT_BASELINE_NAME})"


def register_commands(subparsers) -> None:
    """Register the ``baseline`` command and its subcommands."""
    p = subparsers.add_parser(
        "baseline",
        help="Pin an eval run as a canonical baseline for drift comparisons",
    )
    sub = p.add_subparsers(dest="baseline_action")

    p_set = sub.add_parser("set", help="Pin a run as the named baseline")
    p_set.add_argument("--run", required=True, help="Run ID or path to results.json")
    p_set.add_argument("--name", default=DEFAULT_BASELINE_NAME, help=_NAME_HELP)
    p_set.add_argument("--state-dir", help=_STATE_DIR_HELP)

    p_show = sub.add_parser("show", help="Show a pinned baseline pointer")
    p_show.add_argument("--name", default=DEFAULT_BASELINE_NAME, help=_NAME_HELP)
    p_show.add_argument("--state-dir", help=_STATE_DIR_HELP)

    p_list = sub.add_parser("list", help="List all pinned baselines")
    p_list.add_argument("--state-dir", help=_STATE_DIR_HELP)

    p_clear = sub.add_parser("clear", help="Remove a baseline (or all with --all)")
    p_clear.add_argument("--name", default=DEFAULT_BASELINE_NAME, help=_NAME_HELP)
    p_clear.add_argument("--all", action="store_true", help="Remove every baseline")
    p_clear.add_argument("--state-dir", help=_STATE_DIR_HELP)

    # The top-level dispatcher reads args.func; route the whole group through cmd_baseline.
    for parser in (p, p_set, p_show, p_list, p_clear):
        parser.set_defaults(func=cmd_baseline)


__all__ = [
    "cmd_baseline",
    "cmd_baseline_set",
    "cmd_baseline_show",
    "cmd_baseline_list",
    "cmd_baseline_clear",
    "register_commands",
]
