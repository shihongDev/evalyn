"""``/api/runs`` router.

GET ``/api/runs`` returns a list of run metadata sourced from
``.evalyn/data/eval_runs/<dataset>/<run_id>/results.json``. Empty list
if no runs exist.

GET ``/api/runs/{run_id}`` returns the parsed results.json for one run.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


def _runs_root() -> Path:
    return (Path.cwd() / ".evalyn" / "data" / "eval_runs").resolve()


def _summarize_run(run_dir: Path) -> dict | None:
    """Build a RunMeta-shaped dict from a run directory.

    Returns None if the directory does not contain a parseable
    results.json.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    summary = data.get("summary") or {}
    return {
        "id": run_dir.name,
        "dataset": run_dir.parent.name,
        "pass": float(summary.get("pass_rate", 0.0)),
        "at": data.get("started_at") or data.get("ended_at"),
    }


@router.get("")
async def list_runs() -> JSONResponse:
    """Return all run metadata, newest first."""
    root = _runs_root()
    if not root.exists() or not root.is_dir():
        return JSONResponse([])

    runs: list[dict] = []
    try:
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_dir in dataset_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                meta = _summarize_run(run_dir)
                if meta is not None:
                    runs.append(meta)
    except OSError:
        return JSONResponse([])

    runs.sort(key=lambda r: r.get("at") or "", reverse=True)
    return JSONResponse(runs)


@router.get("/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    """Return the full results.json for one run."""
    root = _runs_root()
    if not root.exists():
        raise HTTPException(404, ".evalyn/data/eval_runs/ does not exist")

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        candidate = dataset_dir / run_id / "results.json"
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(500, f"failed to parse run: {exc}") from exc
            return JSONResponse(data)
    raise HTTPException(404, f"run {run_id} not found")
