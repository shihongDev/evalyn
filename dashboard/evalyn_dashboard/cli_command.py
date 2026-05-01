"""Entry-point hook for the evalyn ``dashboard`` subcommand.

Exposed via the ``evalyn.commands`` entry point in :mod:`pyproject` so the
core ``evalyn`` CLI discovers and registers the dashboard subcommand only
when the ``evalyn-dashboard`` package is installed.

Phase 0 ships a stub that prints a placeholder message. The real launcher
(uvicorn + browser) lands in Phase 1 lane A1.
"""

from __future__ import annotations

import argparse


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``dashboard`` subcommand on the given subparsers."""

    p = subparsers.add_parser(
        "dashboard",
        help="Launch the evalyn dashboard (localhost IDE)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7401)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_dashboard)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Run the dashboard (Phase 0 placeholder)."""

    print("Dashboard not yet implemented (Phase 1 lane A1)")
    return 0


__all__ = ["register_commands", "cmd_dashboard"]
