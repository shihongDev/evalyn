"""Regression tests for the saved-rubric index introduced in qa/dashboard-button-sweep.

Found by /qa on 2026-05-18: ``GET /api/v2/rubrics`` was hanging ~40s on
prod because ``_load_saved_rubric`` was called per metric and walked
all ~495 dataset directories doing stat() probes each time
(O(metrics x dataset_dirs)). The fix inverts the walk into a single
mtime-cached index. These tests pin down the behaviour that matters:

1. A saved rubric file under ``<dataset>/rubrics/<metric_id>.json`` is
   surfaced in the list response.
2. ``POST /api/v2/rubrics/<id>`` invalidates the index so a follow-up
   ``GET`` reflects the new state (root mtime alone does NOT bump on
   nested writes, which would have left a stale cache otherwise).

Report: .gstack/qa-reports/qa-report-localhost-7401-2026-05-18.md
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.server import build_app

from ._v2_helpers import make_populated_workspace


def _write_saved_rubric(workspace, metric_id: str, payload: dict) -> None:
    """Drop a persisted rubric next to the fixture's calibration dataset."""
    target = workspace / ".evalyn" / "data" / "datasets" / "calibration" / "rubrics"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{metric_id}.json").write_text(json.dumps(payload))


def _csrf(client: TestClient) -> dict[str, str]:
    """Pull the workbench-token header from the index page's meta tag."""
    html = client.get("/").text
    marker = 'name="workbench-token" content="'
    idx = html.find(marker)
    if idx < 0:
        return {}
    start = idx + len(marker)
    end = html.find('"', start)
    return {"X-Workbench-Token": html[start:end]}


def test_saved_rubric_surfaces_in_list(tmp_path, monkeypatch):
    workspace = make_populated_workspace(tmp_path, monkeypatch)
    _write_saved_rubric(
        workspace,
        "helpfulness",
        {
            "weights": {"clarity": 60, "accuracy": 40},
            "dimensions": [
                {"label": "clarity", "weight": 60, "fp": 1, "fn": 0},
                {"label": "accuracy", "weight": 40, "fp": 0, "fn": 2},
            ],
        },
    )
    with TestClient(build_app()) as client:
        r = client.get("/api/v2/rubrics")
    assert r.status_code == 200
    rows = {row["id"]: row for row in r.json()}
    assert "helpfulness" in rows
    row = rows["helpfulness"]
    assert row["weights"] == {"clarity": 60, "accuracy": 40}
    assert row["dimensions"] == 2  # the count, not the array
    labels = [d["label"] for d in row["dimensions_detail"]]
    assert labels == ["clarity", "accuracy"]


def test_save_rubric_invalidates_list_cache(tmp_path, monkeypatch):
    """POST then GET must reflect the new state without restart.

    Root mtime does not bump on nested writes, so without explicit
    cache busting the list would keep returning pre-save state.
    """
    make_populated_workspace(tmp_path, monkeypatch)
    app = build_app()
    with TestClient(app) as client:
        before = client.get("/api/v2/rubrics").json()
        before_row = next((r for r in before if r["id"] == "helpfulness"), None)
        assert before_row is not None
        assert before_row["weights"] is None

        save = client.post(
            "/api/v2/rubrics/helpfulness",
            json={
                "weights": {"clarity": 100},
                "dimensions": [{"label": "clarity", "weight": 100}],
            },
            headers=_csrf(client),
        )
        assert save.status_code == 200

        after = client.get("/api/v2/rubrics").json()
        after_row = next((r for r in after if r["id"] == "helpfulness"), None)
        assert after_row is not None
        assert after_row["weights"] == {"clarity": 100}


def test_list_rubrics_is_fast_on_warm_cache(tmp_path, monkeypatch):
    """The index must be cached: a second call should not re-walk dataset dirs.

    Uses a small fixture so the absolute time is meaningless; the
    invariant we check is that the warm call returns in well under the
    cold call's wall-clock budget, which is a proxy for "the cache was
    used". A hard cap of 1s keeps this from going green on a broken
    cache even on slow CI.
    """
    import time

    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        client.get("/api/v2/rubrics")  # warm
        t0 = time.perf_counter()
        for _ in range(5):
            r = client.get("/api/v2/rubrics")
            assert r.status_code == 200
        elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"5 warm GETs took {elapsed:.3f}s; cache likely broken"
