"""``/api/runs`` router.

GET ``/api/runs`` returns a list of run metadata sourced from the SDK's
canonical layout:
``.evalyn/data/datasets/<dataset_name>/eval_runs/<run_id>/results.json``.
Empty list if no runs exist.

GET ``/api/runs/{run_id}`` returns the parsed results.json for one run.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


def _datasets_root() -> Path:
    """Return ``.evalyn/data/datasets/`` resolved against ``cwd``.

    Per SDK convention (see ``sdk/evalyn_sdk/analysis/core.py``), runs
    live under ``<datasets_root>/<dataset>/eval_runs/<run>/``. The
    dashboard previously walked a flat ``.evalyn/data/eval_runs/`` layout
    that does not exist on disk.
    """
    return (Path.cwd() / ".evalyn" / "data" / "datasets").resolve()


def _iter_run_dirs(root: Path):
    """Yield each run directory under ``<datasets_root>``.

    Layout: ``<root>/<dataset>/eval_runs/<run>/``.
    """
    if not root.exists() or not root.is_dir():
        return
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        eval_runs_dir = dataset_dir / "eval_runs"
        if not eval_runs_dir.is_dir():
            continue
        for run_dir in eval_runs_dir.iterdir():
            if run_dir.is_dir():
                yield run_dir


def _summarize_run(run_dir: Path) -> dict | None:
    """Build a RunMeta-shaped dict from a run directory.

    Returns None if the directory does not contain a parseable
    results.json. ``run_dir.parent`` is the ``eval_runs`` folder, so the
    dataset name lives one more level up.
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
        "dataset": run_dir.parent.parent.name,
        "pass": float(summary.get("pass_rate", 0.0)),
        "at": data.get("started_at") or data.get("ended_at"),
    }


@router.get("")
async def list_runs() -> JSONResponse:
    """Return all run metadata, newest first."""
    runs: list[dict] = []
    try:
        for run_dir in _iter_run_dirs(_datasets_root()):
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
    root = _datasets_root()
    if not root.exists():
        raise HTTPException(404, ".evalyn/data/datasets/ does not exist")
    for run_dir in _iter_run_dirs(root):
        if run_dir.name != run_id:
            continue
        candidate = run_dir / "results.json"
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, f"failed to parse run: {exc}") from exc
        return JSONResponse(data)
    raise HTTPException(404, f"run {run_id} not found")
