"""``/api/demo`` router.

POST ``/api/demo/load`` copies the bundled ``demo_fixture/`` directory tree
into ``.evalyn/`` (relative to the server's current working directory).

The dashboard's "Load demo" CTA on the Welcome view calls this endpoint
when the workspace is empty. After the copy succeeds we write a
``.evalyn/.demo_loaded`` sentinel file so the UI can show a "Demo data
loaded" banner.

Status codes:
- 200: copy succeeded or workspace was already loaded with the demo.
  Body: ``{"loaded": true, "project": "research-v1"}``.
- 409: ``.evalyn/data/datasets/`` already contains a non-demo dataset
  directory (anything other than ``research-v1``). The user has real
  work in this workspace; we refuse to clobber it.
- 500: filesystem error during copy.

Idempotency: if ``.evalyn/.demo_loaded`` already exists, or the only
existing dataset is ``research-v1``, the copy is run again with
``dirs_exist_ok=True`` so the fixture is restored to its canonical
state. The sentinel is written AFTER the copy completes so a partial
copy never leaves us in a "loaded but actually broken" state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


# Name of the demo project's dataset; used to decide whether existing
# .evalyn/ content is "demo" or "real user work".
DEMO_DATASET_NAME = "research-v1"

# Sentinel file path (relative to .evalyn/). Presence indicates the
# workspace was populated by /api/demo/load.
SENTINEL_REL = ".demo_loaded"


def _fixture_root() -> Path:
    """Resolve the bundled fixture directory (sibling of this module)."""
    return (Path(__file__).resolve().parent.parent / "demo_fixture").resolve()


def _evalyn_root() -> Path:
    """Resolve ``.evalyn/`` relative to the server's cwd."""
    return (Path.cwd() / ".evalyn").resolve()


def _has_non_demo_content(evalyn_dir: Path) -> bool:
    """Return True if ``.evalyn/data/datasets/`` contains a non-demo project.

    Treats the workspace as "real user work" only when at least one
    dataset directory exists whose name is not the demo's. Files outside
    ``data/datasets/`` (settings, credentials, logs) do not count: the
    demo never overwrites those, and refusing to load the demo just
    because a stray file is present would be too strict.

    A pre-existing demo sentinel disables the check entirely so repeat
    loads remain a no-op upgrade rather than a 409.
    """
    if (evalyn_dir / SENTINEL_REL).exists():
        return False
    datasets_dir = evalyn_dir / "data" / "datasets"
    if not datasets_dir.is_dir():
        return False
    try:
        for entry in datasets_dir.iterdir():
            if entry.is_dir() and entry.name != DEMO_DATASET_NAME:
                return True
    except OSError:
        # If we can't read the directory, fail safe: assume the user
        # has content and refuse to clobber it.
        return True
    return False


@router.post("/load")
async def load_demo() -> JSONResponse:
    """Copy the bundled demo fixture into ``.evalyn/``."""
    fixture = _fixture_root()
    if not fixture.is_dir():
        raise HTTPException(
            500,
            f"demo fixture not found at {fixture}",
        )

    target = _evalyn_root()
    if _has_non_demo_content(target):
        raise HTTPException(
            409,
            ".evalyn/ already contains non-demo content; refusing to overwrite",
        )

    try:
        target.mkdir(parents=True, exist_ok=True)
        # ``dirs_exist_ok=True`` makes the copy idempotent: re-running
        # restores the fixture to its canonical state without erroring
        # on existing demo files.
        shutil.copytree(fixture, target, dirs_exist_ok=True)
    except OSError as exc:
        raise HTTPException(500, f"failed to copy demo fixture: {exc}") from exc

    # Write the sentinel last so a partial copy never leaves us
    # claiming the demo is loaded when it isn't.
    try:
        (target / SENTINEL_REL).write_text(
            "Demo data loaded by POST /api/demo/load.\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(500, f"failed to write demo sentinel: {exc}") from exc

    return JSONResponse({"loaded": True, "project": DEMO_DATASET_NAME})
