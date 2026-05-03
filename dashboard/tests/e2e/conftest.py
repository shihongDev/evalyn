"""Playwright fixtures for the dashboard E2E suite (Lane D1.1).

The single happy-path test in ``test_happy_path.py`` needs:

1. A running dashboard server on a random localhost port.
2. The pre-built React bundle served at ``/static`` (must be built before
   the test runs - CI does ``npm run build`` first).
3. A Playwright ``page`` fixture (provided by ``pytest-playwright``).

This conftest contributes (1): a ``session``-scoped fixture that spawns
``uvicorn`` in a subprocess, polls ``/api/health`` until it answers
(timeout 10s, fail-fast), yields the base URL, and tears down with
SIGTERM + SIGKILL grace.

Why subprocess and not ``TestClient``? Playwright drives a real browser
which can't talk to ``TestClient``'s in-process ASGI transport - it
needs a TCP socket. So we run the server out-of-process and let the
browser dial it.

The ``base_url`` fixture is also wired into ``pytest-playwright``'s
built-in ``base_url`` knob via the ``pytest-base-url`` plugin so that
``page.goto("/")`` resolves correctly without hard-coding the port.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

try:
    import httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dev extra missing
    httpx = None  # type: ignore[assignment]


SERVER_READY_TIMEOUT_SEC = 10.0
SERVER_POLL_INTERVAL_SEC = 0.1


def _pick_free_port() -> int:
    """Bind ephemeral, read assigned port, release. Race-free enough for CI."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float) -> None:
    """Poll ``/api/health`` until 200 or fail with the last error."""
    if httpx is None:
        raise RuntimeError(
            "httpx required for E2E health polling. Install dashboard[dev]."
        )
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/api/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - polling phase
            last_exc = exc
        time.sleep(SERVER_POLL_INTERVAL_SEC)
    raise RuntimeError(
        f"dashboard server did not answer /api/health within {timeout}s "
        f"(last error: {last_exc!r})"
    )


@pytest.fixture(scope="session")
def dashboard_server() -> Iterator[str]:
    """Spawn the dashboard server and yield its base URL.

    Starts ``python -m evalyn_dashboard --host 127.0.0.1 --port <free>
    --no-browser`` so the production code path is exercised.
    """
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    # Run the package as a script so we exercise the same launch path as
    # ``evalyn dashboard``. The ``--no-browser`` flag keeps CI headless.
    cmd = [
        sys.executable,
        "-m",
        "evalyn_dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-browser",
    ]

    env = os.environ.copy()
    # Force unbuffered Python so any startup error reaches stderr promptly.
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_health(base_url, SERVER_READY_TIMEOUT_SEC)
    except Exception:
        # Surface server output to make CI diagnosis possible.
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        raise RuntimeError(
            f"dashboard server failed to start. Output:\n{out}"
        )

    try:
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


@pytest.fixture(scope="session")
def base_url(dashboard_server: str) -> str:
    """Override ``pytest-base-url``'s default with our spawned server's URL.

    ``pytest-playwright`` reads this fixture to seed ``page.goto("/")``.
    """
    return dashboard_server
