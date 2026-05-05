"""Tests for ``/api/v2/annotation`` - human annotation sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalyn_dashboard.api.v2._shared import _clear_caches_for_tests
from evalyn_dashboard.server import build_app

from ._v2_helpers import make_empty_workspace, make_populated_workspace


def _csrf(client: TestClient) -> dict[str, str]:
    """Pull the CSRF token out of the index page meta tag."""
    html = client.get("/").text
    marker = 'name="workbench-token" content="'
    idx = html.find(marker)
    if idx < 0:
        return {}
    start = idx + len(marker)
    end = html.find('"', start)
    return {"X-Workbench-Token": html[start:end]}


def _first_run_id(client: TestClient) -> str:
    """Pick the first run id from /api/v2/experiments."""
    rows = client.get("/api/v2/experiments").json()
    assert rows, "fixture must have at least one run"
    return rows[0]["id"]


def _first_metric_id(client: TestClient, run_id: str) -> str:
    """Pick the first metric id observed in a run."""
    detail = client.get(f"/api/v2/experiments/{run_id}").json()
    metrics = detail.get("sub_metrics") or []
    assert metrics, f"run {run_id} has no metrics"
    return metrics[0]["label"]


# ---------------------------------------------------------------------------
# Empty workspace
# ---------------------------------------------------------------------------


def test_list_sessions_empty_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_empty_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.get("/api/v2/annotation/sessions")
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


def test_get_session_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_empty_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.get("/api/v2/annotation/sessions/ann-bogus")
    assert r.status_code == 404


def test_create_session_source_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.post(
            "/api/v2/annotation/sessions",
            json={"source_kind": "run", "source_id": "does-not-exist"},
            headers=_csrf(client),
        )
    assert r.status_code == 404


def test_create_session_invalid_source_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.post(
            "/api/v2/annotation/sessions",
            json={"source_kind": "wat", "source_id": "x"},
            headers=_csrf(client),
        )
    assert r.status_code == 422


def test_create_session_path_traversal_blocked(tmp_path, monkeypatch):
    """Crafted source_id with traversal chars must not escape the dataset roots."""
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.post(
            "/api/v2/annotation/sessions",
            json={"source_kind": "dataset", "source_id": "../../etc"},
            headers=_csrf(client),
        )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Happy path: create -> list items -> verdict -> finalize
# ---------------------------------------------------------------------------


def test_session_lifecycle_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)

        # Create
        r = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "run",
                "source_id": run_id,
                "metric_ids": [metric_id],
                "annotator_id": "tester",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        meta = r.json()
        sid = meta["id"]
        assert meta["status"] == "in_progress"
        assert meta["metric_ids"] == [metric_id]
        assert meta["items_total"] >= 1
        assert meta["annotator_id"] == "tester"

        # List shows it
        sessions = client.get("/api/v2/annotation/sessions").json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

        # Items batch with pre-labels
        items_resp = client.get(
            f"/api/v2/annotation/sessions/{sid}/items?limit=3"
        ).json()
        assert items_resp["session_id"] == sid
        assert items_resp["total"] == meta["items_total"]
        assert len(items_resp["items"]) == min(3, meta["items_total"])
        first_item = items_resp["items"][0]
        assert "ai_labels" in first_item
        assert any(
            entry["metric_id"] == metric_id for entry in first_item["ai_labels"]
        )
        assert first_item["annotated"] is False

        # Submit verdict
        verdict_body = {
            "item_id": first_item["item_id"],
            "labels": [
                {
                    "metric_id": metric_id,
                    "label": "pass",
                    "used_ai_verdict": True,
                }
            ],
            "skipped_metrics": [],
            "note": None,
        }
        r2 = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=verdict_body,
            headers=headers,
        )
        assert r2.status_code == 200, r2.text
        prog = r2.json()
        assert prog["items_done"] == 1

        # Resubmitting same item is idempotent (count stays at 1)
        r3 = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=verdict_body,
            headers=headers,
        )
        assert r3.status_code == 200
        assert r3.json()["items_done"] == 1

        # Finalize merges into annotations.jsonl
        r4 = client.post(
            f"/api/v2/annotation/sessions/{sid}/finalize",
            headers=headers,
        )
        assert r4.status_code == 200, r4.text
        assert r4.json()["merged"] >= 1

        # Re-finalize is idempotent (no duplicate writes)
        r5 = client.post(
            f"/api/v2/annotation/sessions/{sid}/finalize",
            headers=headers,
        )
        assert r5.status_code == 200
        assert r5.json()["merged"] == 0

    # Files exist on disk where expected
    sessions_dir = tmp_path / ".evalyn" / "data" / "datasets" / "calibration" / "annotation_sessions"
    assert (sessions_dir / f"{sid}.json").exists()
    assert (sessions_dir / f"{sid}.jsonl").exists()
    canonical = tmp_path / ".evalyn" / "data" / "datasets" / "calibration" / "annotations.jsonl"
    assert canonical.exists()
    # The fixture ships a pre-existing annotations.jsonl with a different
    # shape (target_id instead of item_id). Finalize appends our records
    # without touching legacy ones; find ours by session_id.
    lines = [json.loads(l) for l in canonical.read_text().splitlines() if l.strip()]
    ours = [r for r in lines if r.get("session_id") == sid]
    assert ours and ours[0]["item_id"] == first_item["item_id"]


def test_verdict_unknown_session(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        r = client.post(
            "/api/v2/annotation/sessions/ann-nope/verdict",
            json={"item_id": "x", "labels": []},
            headers=_csrf(client),
        )
    assert r.status_code == 404


def test_verdict_unknown_item_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)
        meta = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "run",
                "source_id": run_id,
                "metric_ids": [metric_id],
            },
            headers=headers,
        ).json()
        r = client.post(
            f"/api/v2/annotation/sessions/{meta['id']}/verdict",
            json={
                "item_id": "definitely-not-in-this-run",
                "labels": [
                    {"metric_id": metric_id, "label": "pass", "used_ai_verdict": False}
                ],
            },
            headers=headers,
        )
    assert r.status_code == 422


def test_verdict_bad_label_value(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)
        meta = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "run",
                "source_id": run_id,
                "metric_ids": [metric_id],
            },
            headers=headers,
        ).json()
        items = client.get(f"/api/v2/annotation/sessions/{meta['id']}/items?limit=1").json()
        iid = items["items"][0]["item_id"]
        r = client.post(
            f"/api/v2/annotation/sessions/{meta['id']}/verdict",
            json={
                "item_id": iid,
                "labels": [
                    {"metric_id": metric_id, "label": "wat", "used_ai_verdict": False}
                ],
            },
            headers=headers,
        )
    assert r.status_code == 422


def test_abandon_session(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)
        meta = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "run",
                "source_id": run_id,
                "metric_ids": [metric_id],
            },
            headers=headers,
        ).json()
        sid = meta["id"]

        r = client.delete(f"/api/v2/annotation/sessions/{sid}", headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "abandoned"

        # Posting a verdict after abandon → 409
        r2 = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json={
                "item_id": meta["item_ids"][0],
                "labels": [
                    {"metric_id": metric_id, "label": "pass", "used_ai_verdict": False}
                ],
            },
            headers=headers,
        )
        assert r2.status_code == 409


def test_resume_replays_log(tmp_path, monkeypatch):
    """If session.json is missing or stale, items_done should self-heal from the log."""
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)
        meta = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "run",
                "source_id": run_id,
                "metric_ids": [metric_id],
            },
            headers=headers,
        ).json()
        sid = meta["id"]
        items = client.get(f"/api/v2/annotation/sessions/{sid}/items?limit=2").json()

        # Append a malformed line directly to the log to simulate corruption.
        ds_dir = tmp_path / ".evalyn" / "data" / "datasets" / "calibration"
        log_path = ds_dir / "annotation_sessions" / f"{sid}.jsonl"
        with log_path.open("a") as f:
            f.write("not-json garbage line\n")

        # Submit a real verdict - the bad line is logged + skipped, but the
        # real one is counted.
        r = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json={
                "item_id": items["items"][0]["item_id"],
                "labels": [
                    {"metric_id": metric_id, "label": "fail", "used_ai_verdict": False}
                ],
            },
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["items_done"] == 1


def test_dataset_source_works(tmp_path, monkeypatch):
    """Source kind 'dataset' uses the dataset's items, not a run's."""
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        # Need any metric. Use one observed in any run.
        run_id = _first_run_id(client)
        metric_id = _first_metric_id(client, run_id)
        headers = _csrf(client)
        r = client.post(
            "/api/v2/annotation/sessions",
            json={
                "source_kind": "dataset",
                "source_id": "calibration",
                "metric_ids": [metric_id],
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        meta = r.json()
        assert meta["source_kind"] == "dataset"
        assert meta["items_total"] >= 1


# ---------------------------------------------------------------------------
# Evidence: per-verdict text snippets the annotator highlighted
# ---------------------------------------------------------------------------


def _bootstrap_session(client: TestClient) -> tuple[str, str, str, dict]:
    """Helper: create a session, return (sid, metric_id, item_id, headers)."""
    run_id = _first_run_id(client)
    metric_id = _first_metric_id(client, run_id)
    headers = _csrf(client)
    r = client.post(
        "/api/v2/annotation/sessions",
        json={
            "source_kind": "run",
            "source_id": run_id,
            "metric_ids": [metric_id],
            "annotator_id": "tester",
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    items = client.get(f"/api/v2/annotation/sessions/{sid}/items?limit=1").json()
    item_id = items["items"][0]["item_id"]
    return sid, metric_id, item_id, headers


def test_evidence_round_trip(tmp_path, monkeypatch):
    """Submitted evidence is replayed in the items GET response."""
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        sid, metric_id, item_id, headers = _bootstrap_session(client)
        body = {
            "item_id": item_id,
            "labels": [
                {"metric_id": metric_id, "label": "fail", "used_ai_verdict": False}
            ],
            "evidence": [
                {"snippet": "hallucinated date", "metric_id": metric_id, "note": "wrong year"},
                {"snippet": "missing citation", "metric_id": None},
            ],
        }
        r = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=body,
            headers=headers,
        )
        assert r.status_code == 200, r.text

        # Items GET returns the evidence we just stored
        items = client.get(f"/api/v2/annotation/sessions/{sid}/items?limit=1").json()
        first = items["items"][0]
        assert first["item_id"] == item_id
        assert first["evidence"] == body["evidence"]


def test_evidence_rejects_empty_snippet(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        sid, metric_id, item_id, headers = _bootstrap_session(client)
        body = {
            "item_id": item_id,
            "labels": [
                {"metric_id": metric_id, "label": "pass", "used_ai_verdict": False}
            ],
            "evidence": [{"snippet": "   ", "metric_id": metric_id}],
        }
        r = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=body,
            headers=headers,
        )
        assert r.status_code == 422
        assert "snippet" in r.json()["detail"]


def test_evidence_rejects_unknown_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        sid, metric_id, item_id, headers = _bootstrap_session(client)
        body = {
            "item_id": item_id,
            "labels": [
                {"metric_id": metric_id, "label": "pass", "used_ai_verdict": False}
            ],
            "evidence": [{"snippet": "x", "metric_id": "no-such-metric"}],
        }
        r = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=body,
            headers=headers,
        )
        assert r.status_code == 422
        assert "metric_id" in r.json()["detail"]


def test_evidence_optional_omitted_works(tmp_path, monkeypatch):
    """Old clients that don't send evidence still work; field defaults to []."""
    monkeypatch.setenv("EVALYN_DASHBOARD_TEST_NO_CACHE", "1")
    _clear_caches_for_tests()
    make_populated_workspace(tmp_path, monkeypatch)
    with TestClient(build_app()) as client:
        sid, metric_id, item_id, headers = _bootstrap_session(client)
        body = {
            "item_id": item_id,
            "labels": [
                {"metric_id": metric_id, "label": "pass", "used_ai_verdict": True}
            ],
        }
        r = client.post(
            f"/api/v2/annotation/sessions/{sid}/verdict",
            json=body,
            headers=headers,
        )
        assert r.status_code == 200, r.text
        items = client.get(f"/api/v2/annotation/sessions/{sid}/items?limit=1").json()
        assert items["items"][0]["evidence"] == []
