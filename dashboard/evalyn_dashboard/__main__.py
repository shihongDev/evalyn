"""Allow ``python -m evalyn_dashboard`` to launch the dashboard CLI.

The real implementation lands in Phase 1 lane A1. For Phase 0 this is a
placeholder that delegates to :func:`cli_command.cmd_dashboard`.
"""

from __future__ import annotations

import argparse
import sys

from .cli_command import cmd_dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evalyn_dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7401)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    return cmd_dashboard(args)


if __name__ == "__main__":
    sys.exit(main())
