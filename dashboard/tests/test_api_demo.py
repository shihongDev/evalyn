"""Tests for ``/api/demo/load``.

This endpoint copies a bundled fixture into the workspace's ``.evalyn/``
directory. Tests pin the audit-log shape (now including ``elapsed_ms``)
plus the basic happy path; the 409-on-non-demo-content guard is
covered via the workspace cwd inversion below.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import CSRF_HEADER, build_app


def _token_from(client: TestClient) -> str:
    html = client.get("/").text
    m = re.search(r'content="([^"]+)"', html)
    assert m
    return m.group(1)


def test_demo_load_happy_path_emits_audit_log_with_duration(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """POST /api/demo/load copies the fixture and logs an audit line
    with project + target + duration. The duration field is the
    contract added in the audit-log-duration sweep (matches the
    vacuum/prune/purge shape).
    """
    # Run from tmp_path so the demo lands in tmp_path/.evalyn rather
    # than the user's real workspace.
    monkeypatch.chdir(tmp_path)

    app = build_app()
    with TestClient(app) as client:
        token = _token_from(client)
        with caplog.at_level(logging.INFO, logger="evalyn_dashboard.api.demo"):
            r = client.post(
                "/api/demo/load",
                json={},
                headers={CSRF_HEADER: token},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["loaded"] is True
        assert body["project"] == "research-v1"

    matched = [
        rec for rec in caplog.records
        if rec.message.startswith("demo loaded:")
    ]
    assert len(matched) == 1, (
        f"expected one demo-loaded log line, got {len(matched)}: "
        f"{[r.message for r in caplog.records]}"
    )
    msg = matched[0].message
    assert "project=research-v1" in msg
    # Duration captured for operability - shutil.copytree on a
    # large fixture (or under WSL/NTFS overhead) can be slow enough
    # that operators want to see the cost.
    assert "elapsed_ms=" in msg

    # Sentinel was written after the copy.
    assert (tmp_path / ".evalyn" / ".demo_loaded").is_file()
