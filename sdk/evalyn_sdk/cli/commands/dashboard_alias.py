"""Deprecation alias for the legacy ``evalyn dashboard`` (static report) command.

This module is registered under the ``dashboard`` subcommand only when the
``evalyn-dashboard`` plugin (which provides the new IDE) is NOT installed.
Plugin entry-point discovery in :mod:`evalyn_sdk.cli.main` takes precedence
over this module: when the plugin is present, the new IDE replaces this
alias entirely.

Invoking the alias prints a deprecation warning to stderr and forwards to
:func:`evalyn_sdk.cli.commands.report.cmd_report`.
"""

from __future__ import annotations

import argparse
import sys

from .report import cmd_report

_DEPRECATION_MESSAGE = (
    "[deprecated] 'evalyn dashboard' (static report) renamed to 'evalyn report'. "
    "The 'dashboard' name now refers to the new IDE; install it via "
    "'pip install evalyn-dashboard'. This alias will be removed in v3.0."
)


def cmd_dashboard_alias(args: argparse.Namespace) -> None:
    """Print a deprecation warning then run the new ``report`` command."""
    print(_DEPRECATION_MESSAGE, file=sys.stderr)
    cmd_report(args)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``dashboard`` deprecation alias.

    The alias mirrors the old flag surface so existing scripts keep working.
    """

    p = subparsers.add_parser(
        "dashboard",
        help=(
            "[deprecated] alias for 'evalyn report'; the new IDE ships in "
            "the 'evalyn-dashboard' package"
        ),
    )
    p.add_argument("--run", help="Eval run ID to analyze")
    p.add_argument("--dataset", help="Dataset path (uses latest run)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    p.add_argument(
        "--output",
        help="Output file path (default: .evalyn/report.html)",
    )
    p.set_defaults(func=cmd_dashboard_alias)


__all__ = ["cmd_dashboard_alias", "register_commands"]
