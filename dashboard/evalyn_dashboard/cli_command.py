"""Entry-point hook for the evalyn ``dashboard`` subcommand.

Wired into the core CLI via the ``evalyn.commands`` entry point declared
in ``dashboard/pyproject.toml``. The core ``evalyn`` CLI discovers and
registers the ``dashboard`` subparser only when ``evalyn-dashboard`` is
installed.

Phase 1 lane A1 deliverables:
- ``A1.3`` localhost binding guard with ``--unsafe-bind`` escape hatch.
- ``A1.4`` browser auto-open + uvicorn launcher (lives in ``server.py``).
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
from typing import Any


_LOOPBACK_HOSTNAMES = {"localhost", ""}


def _is_loopback_host(host: str) -> bool:
    """Return True when ``host`` resolves to a loopback address.

    Handles literal IPs (``127.0.0.1``, ``::1``), the ``localhost``
    hostname, and other names that resolve to loopback. Falls back to
    False on resolution failure.
    """

    if host.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            if ipaddress.ip_address(addr).is_loopback:
                return True
        except ValueError:
            continue
    return False


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``dashboard`` subcommand on the given subparsers."""

    p = subparsers.add_parser(
        "dashboard",
        help="Launch the evalyn dashboard (localhost IDE)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7401)
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab on startup.",
    )
    p.add_argument(
        "--unsafe-bind",
        action="store_true",
        help=(
            "Allow binding to a non-loopback host. The dashboard has no "
            "auth or TLS; only enable on trusted networks."
        ),
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Developer mode: skip browser auto-open and skip the bundled "
            "static frontend (Vite dev server is expected on :5173)."
        ),
    )
    p.set_defaults(func=cmd_dashboard)


def _run_server(
    *,
    host: str,
    port: int,
    no_browser: bool,
    dev: bool,
) -> int:
    """Indirection so tests can stub the actual uvicorn run.

    The real implementation is :func:`evalyn_dashboard.server.run_server`;
    importing it lazily keeps unit tests fast (uvicorn pulls in a lot).
    """

    from .server import run_server

    return run_server(host=host, port=port, no_browser=no_browser, dev=dev)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the dashboard, enforcing the loopback binding guard."""

    host: str = args.host
    unsafe: bool = bool(getattr(args, "unsafe_bind", False))

    if not _is_loopback_host(host) and not unsafe:
        print(
            (
                "[error] refusing to bind to non-loopback host "
                f"{host!r}: the dashboard has no auth or TLS. "
                "Use 127.0.0.1 (default), or pass --unsafe-bind to "
                "override on a trusted network."
            ),
            file=sys.stderr,
        )
        return 2

    if not _is_loopback_host(host) and unsafe:
        print(
            (
                f"[warning] binding evalyn dashboard to {host!r} via "
                "--unsafe-bind. No authentication, no TLS. Anyone who "
                "can reach this host can run arbitrary CLI commands."
            ),
            file=sys.stderr,
        )

    return _run_server(
        host=host,
        port=int(args.port),
        no_browser=bool(args.no_browser),
        dev=bool(getattr(args, "dev", False)),
    )


__all__ = [
    "register_commands",
    "cmd_dashboard",
    "_is_loopback_host",
]
