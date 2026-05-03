"""``/api/files`` router.

GET ``/api/files/tree`` returns the .evalyn/ directory layout (folders +
files) under the server's current working directory, projected to the
``FileNode`` shape the frontend expects. Returns an empty list if the
directory does not exist (fresh checkout, never run).

GET ``/api/files/read?path=`` returns the raw text of one file inside
.evalyn/. Path-traversal protected: rejects paths that escape the
.evalyn/ root.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter()


_MAX_TREE_NODES = 2000
_MAX_READ_BYTES = 1_000_000


def _evalyn_root() -> Path:
    """Resolve the .evalyn/ directory relative to the server's cwd."""
    return (Path.cwd() / ".evalyn").resolve()


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}K"
    return f"{n / (1024 * 1024):.1f}M"


def _to_file_node(path: Path, budget: list[int]) -> dict | None:
    """Recursively build a FileNode dict. ``budget`` is a single-element
    list used as a mutable counter for the total node cap."""
    if budget[0] <= 0:
        return None
    budget[0] -= 1

    name = path.name
    if path.is_dir():
        children: list[dict] = []
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            entries = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            child = _to_file_node(entry, budget)
            if child is not None:
                children.append(child)
        return {"name": name, "kind": "folder", "children": children}

    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"name": name, "kind": "file", "meta": _human_size(size)}


@router.get("/tree")
async def file_tree() -> JSONResponse:
    """Return the .evalyn/ tree as a list of top-level FileNodes."""
    root = _evalyn_root()
    if not root.exists() or not root.is_dir():
        return JSONResponse([])

    budget = [_MAX_TREE_NODES]
    top: list[dict] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except OSError:
        return JSONResponse([])
    for entry in entries:
        if entry.name.startswith("."):
            continue
        node = _to_file_node(entry, budget)
        if node is not None:
            top.append(node)
    return JSONResponse(top)


@router.get("/read")
async def read_file(path: str = Query(..., min_length=1)) -> PlainTextResponse:
    """Return the raw text of one file inside .evalyn/. Path-traversal protected."""
    root = _evalyn_root()
    if not root.exists():
        raise HTTPException(404, ".evalyn/ does not exist")

    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "path escapes .evalyn/ root") from exc

    if not target.exists() or not target.is_file():
        raise HTTPException(404, "file not found")
    if target.stat().st_size > _MAX_READ_BYTES:
        raise HTTPException(413, f"file too large (>{_MAX_READ_BYTES // 1024}K)")

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(500, f"read failed: {exc}") from exc
    return PlainTextResponse(text)
