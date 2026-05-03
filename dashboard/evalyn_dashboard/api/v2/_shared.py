"""Shared helpers for v2 routers.

Centralises path resolution, run loading, and common formatting so each
router stays readable. The contract for each endpoint lives in
``dashboard/frontend/src/v2/api/types.ts`` - keep the JSON shapes there
in sync with what these helpers feed back to the routers.

Caching strategy
----------------
Several module-private mtime-keyed caches sit in front of the disk
walks. They are intentionally process-scoped (no TTL, no eviction):
``results.json``/``dataset.jsonl`` are read-mostly so a cache that
invalidates only when the file's ``mtime`` changes is exactly right -
fresh writes win automatically, repeated reads stay free.

Tests that need a clean slate can call :func:`_clear_caches_for_tests`
or set the ``EVALYN_DASHBOARD_TEST_NO_CACHE=1`` env var (every cache
read short-circuits when the env var is set).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

INVERSE_HINTS = ("hallucin", "refus_incorrect")

# ---------------------------------------------------------------------------
# Module-private caches (mtime-keyed). See module docstring for semantics.
# ---------------------------------------------------------------------------

_run_cache: dict[Path, tuple[float, dict]] = {}
_dataset_items_cache: dict[Path, tuple[float, dict[str, dict]]] = {}
_run_dirs_cache: dict[Path, tuple[float, list[Path]]] = {}
_dataset_dirs_cache: dict[Path, tuple[float, list[Path]]] = {}
# Whole-list cache for ``load_all_runs`` keyed on every dataset root's
# mtime so add/remove of a dataset (or a new eval_runs subdir) busts it.
_all_runs_cache: dict[tuple, list[dict]] = {}


def _caches_disabled() -> bool:
    """True when tests have opted out of caching via env var."""
    return os.environ.get("EVALYN_DASHBOARD_TEST_NO_CACHE") == "1"


def _clear_caches_for_tests() -> None:
    """Drop all cached state. Tests that mutate fixtures call this.

    Also clears caches living in sibling modules (datasets.py, rubrics.py)
    by importing them lazily; doing so here keeps the test seam in one
    place rather than scattering ``import``s across every test file.
    """
    _run_cache.clear()
    _dataset_items_cache.clear()
    _run_dirs_cache.clear()
    _dataset_dirs_cache.clear()
    _all_runs_cache.clear()
    try:
        from . import datasets as _datasets_mod  # noqa: WPS433 - intentional
        _datasets_mod._coverage_cache.clear()
        _datasets_mod._meta_cache.clear()
        _datasets_mod._response_cache.clear()
    except ImportError:
        pass
    try:
        from . import rubrics as _rubrics_mod  # noqa: WPS433 - intentional
        _rubrics_mod._calibration_index_cache.clear()
    except ImportError:
        pass


def parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp tolerating a trailing ``Z``. Returns ``None`` on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def dataset_roots() -> list[Path]:
    """Return every dataset root the dashboard should read from.

    Two SDK conventions exist in the wild:
      * ``data/prod/datasets/<name>/`` - the canonical SDK production
        layout (see ``sdk/evalyn_sdk/api.py``); this is where ``run-eval``
        actually writes when no override is set.
      * ``.evalyn/data/datasets/<name>/`` - the dashboard demo fixture
        layout used by ``POST /api/demo/load``.

    We walk both. Order is prod-first so that a dataset name colliding
    across both locations resolves to the prod copy. Non-existent roots
    are filtered out so callers can blindly iterate.
    """
    cwd = Path.cwd()
    candidates = [
        cwd / "data" / "prod" / "datasets",
        cwd / ".evalyn" / "data" / "datasets",
    ]
    return [p.resolve() for p in candidates if p.is_dir()]


def datasets_root() -> Path:
    """First existing dataset root, or the demo path as a placeholder.

    Kept for legacy callers that need a single ``Path`` (e.g. for 404
    error messages). New code should prefer ``dataset_roots()``.
    """
    roots = dataset_roots()
    return roots[0] if roots else (Path.cwd() / ".evalyn" / "data" / "datasets").resolve()


def evalyn_yaml_path() -> Path:
    """Path to the optional ``.evalyn/evalyn.yaml`` project config file."""
    return (Path.cwd() / ".evalyn" / "evalyn.yaml").resolve()


def project_meta() -> tuple[str, str | None]:
    """Return ``(project_name, version)`` from ``evalyn.yaml`` if present.

    Returns ``version=None`` (not a fabricated ``"v0.1"``) when no version
    is set on disk so the UI can render ``-`` rather than a fake string.
    The walker is intentionally minimal so we don't pull in a yaml
    dependency just for two scalars under ``project:``.
    """
    name = Path.cwd().name or "evalyn"
    version: str | None = None
    p = evalyn_yaml_path()
    if not p.exists():
        return name, version
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("evalyn.yaml read failed at %s: %s", p, exc)
        return name, version
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            in_project = stripped[:-1] == "project"
            continue
        if in_project and ":" in stripped:
            key, _, value = stripped.partition(":")
            value = value.strip().strip("\"'")
            if key.strip() == "name" and value:
                name = value
            elif key.strip() == "version" and value:
                version = value
    return name, version


def list_dataset_dirs(root: Path) -> list[Path]:
    """Return cached list of dataset subdirectories under ``root``.

    Used by every endpoint that walks ``data/prod/datasets/*``. Prod has
    491 dataset folders; raw ``iterdir()`` is ~1.9s on WSL/NTFS - so we
    cache by the root's mtime (changes on add/remove of a dataset). The
    list is sorted so callers don't have to.
    """
    if not root.is_dir():
        return []
    if _caches_disabled():
        return sorted(p for p in root.iterdir() if p.is_dir())
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return sorted(p for p in root.iterdir() if p.is_dir())
    cached = _dataset_dirs_cache.get(root)
    if cached and cached[0] == mtime:
        return cached[1]
    out = sorted(p for p in root.iterdir() if p.is_dir())
    _dataset_dirs_cache[root] = (mtime, out)
    return out


def _list_run_dirs_for_root(root: Path) -> list[Path]:
    """Return cached list of run dirs under ``root``, mtime-invalidated.

    ``root.stat().st_mtime`` changes when a dataset directory is added
    or removed - cheap signal that catches the only mutation we care
    about (new dataset folders). Per-dataset run additions don't bump
    the root mtime but a new dataset typically arrives before its first
    run anyway, so the cache reaches steady state quickly. Callers that
    need stronger freshness can invalidate via ``_clear_caches_for_tests``.
    """
    if not root.is_dir():
        return []
    if _caches_disabled():
        return _walk_run_dirs(root)
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return _walk_run_dirs(root)
    cached = _run_dirs_cache.get(root)
    if cached and cached[0] == mtime:
        return cached[1]
    out = _walk_run_dirs(root)
    _run_dirs_cache[root] = (mtime, out)
    return out


def _walk_run_dirs(root: Path) -> list[Path]:
    """Filesystem walk that ``_list_run_dirs_for_root`` memoizes."""
    out: list[Path] = []
    for dataset_dir in list_dataset_dirs(root):
        eval_runs_dir = dataset_dir / "eval_runs"
        if not eval_runs_dir.is_dir():
            continue
        for run_dir in eval_runs_dir.iterdir():
            if run_dir.is_dir():
                out.append(run_dir)
    return out


def iter_run_dirs(root: Path | None = None) -> Iterable[Path]:
    """Yield each ``<dataset>/eval_runs/<run>`` dir.

    When ``root`` is ``None`` (default) walks every root from
    :func:`dataset_roots`. When a specific root is passed we only walk
    that one, preserving back-compat with callers that already scope
    themselves to one location.

    Backed by a per-root mtime cache so repeat callers (every v2
    endpoint) skip the directory walk entirely on warm requests.
    """
    roots = [root] if root is not None else dataset_roots()
    for r in roots:
        for run_dir in _list_run_dirs_for_root(r):
            yield run_dir


def load_run(run_dir: Path) -> dict | None:
    """Parse ``run_dir/results.json``. Returns ``None`` on missing/invalid.

    Cached by ``mtime`` of the file - a fresh write (run promoted, file
    rewritten) invalidates automatically. Cold parse cost on prod data
    is ~3s for 147 runs (~21MB); warm hit is dict lookup.

    A logged-but-skipped run silently shrinks aggregate denominators
    (Quality, item counts, sub-metrics). Logging at WARN gives operators
    a breadcrumb when numbers look off; the right long-term fix is to
    surface a per-route ``data_warnings`` field.
    """
    p = run_dir / "results.json"
    try:
        st = p.stat()
    except OSError:
        return None
    if not _caches_disabled():
        cached = _run_cache.get(p)
        if cached and cached[0] == st.st_mtime:
            return cached[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("results.json read failed at %s: %s", p, exc)
        return None
    if not _caches_disabled():
        _run_cache[p] = (st.st_mtime, data)
    return data


def _all_runs_cache_key(roots: list[Path]) -> tuple | None:
    """Return ``((root, mtime), ...)`` snapshot or ``None`` on stat failure."""
    parts: list[tuple] = []
    for r in roots:
        try:
            parts.append((r, r.stat().st_mtime))
        except OSError:
            return None
    return tuple(parts)


def load_all_runs(root: Path | None = None) -> list[dict]:
    """Return all parseable runs as dicts, each augmented with ``_dir``.

    Each entry is the raw ``results.json`` plus:
      * ``_dir``: ``Path`` to the run directory
      * ``_dataset``: dataset folder name (immediate parent of eval_runs)
      * ``_run_dir_name``: directory basename (used as the v2 ``id``)

    Walks every root from :func:`dataset_roots` when ``root`` is ``None``
    so prod runs and demo fixture runs both surface in the dashboard.

    Sorted oldest first by ``created_at``; callers can re-sort.

    Cached at the list level keyed on every dataset root's mtime. Even
    with per-file ``load_run`` caching, the per-call ``stat()`` overhead
    on 147 runs hits ~500ms on WSL; memoising the assembled list cuts
    that to a dict lookup. The cache invalidates as soon as a dataset
    is added or removed (root mtime bumps).
    """
    roots = [root] if root is not None else dataset_roots()
    key: tuple | None = None
    if not _caches_disabled():
        key = _all_runs_cache_key(roots)
        if key is not None and key in _all_runs_cache:
            return _all_runs_cache[key]
    out: list[dict] = []
    for run_dir in iter_run_dirs(root):
        data = load_run(run_dir)
        if data is None:
            continue
        data["_dir"] = run_dir
        data["_dataset"] = run_dir.parent.parent.name
        data["_run_dir_name"] = run_dir.name
        out.append(data)
    out.sort(key=lambda d: d.get("created_at") or "")
    if key is not None and not _caches_disabled():
        _all_runs_cache[key] = out
    return out


def total_cost(run: dict) -> float | None:
    """Return run total cost in USD or ``None`` if absent.

    Newer runs use ``total_cost_usd``; the v2 contract calls it
    ``total_cost`` so accept both for forward compatibility.
    """
    usage = run.get("usage_summary") or {}
    cost = usage.get("total_cost")
    if cost is None:
        cost = usage.get("total_cost_usd")
    return float(cost) if cost is not None else None


def fmt_cost(cost: float | None) -> str:
    """Format ``cost`` as ``$X.YY``; ``$0.00`` when missing or zero."""
    if cost is None or cost <= 0:
        return "$0.00"
    return f"${cost:.2f}"


def fmt_duration(seconds: float | None) -> str:
    """Format seconds as ``Xm Ys`` or ``Ys``; ``-`` when missing."""
    if seconds is None:
        return "-"
    s = int(round(float(seconds)))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def is_inverse_metric(metric_id: str) -> bool:
    """True for hallucination-style metrics (lower score = better)."""
    mid = metric_id.lower()
    return any(h in mid for h in INVERSE_HINTS)


def run_id(run: dict) -> str:
    """Return the externally addressable run id (the directory name)."""
    return run.get("_run_dir_name") or run.get("id") or "unknown"


def run_pass_rate(run: dict) -> float | None:
    """Return overall pass rate (0..1) or ``None``.

    Prefers ``summary.pass_rate`` when set. Older runs only carry
    per-metric stats under ``summary.metrics`` so we fall back to
    averaging those, then to a fresh walk over ``metric_results``.
    """
    s = run.get("summary") or {}
    pr = s.get("pass_rate")
    if pr is not None:
        return float(pr)
    metrics = s.get("metrics") or {}
    rates = [
        m.get("pass_rate") if isinstance(m, dict) else None
        for m in metrics.values()
    ]
    rates = [float(x) for x in rates if isinstance(x, (int, float))]
    if rates:
        return sum(rates) / len(rates)
    # Last resort: derive from metric_results.
    item_pass: dict[str, bool] = {}
    for r in run.get("metric_results") or []:
        iid = r.get("item_id")
        if iid is None:
            continue
        item_pass.setdefault(iid, True)
        if r.get("passed") is False:
            item_pass[iid] = False
    if not item_pass:
        return None
    return sum(1 for v in item_pass.values() if v) / len(item_pass)


def run_status(run: dict, default: str = "completed") -> str:
    """Map ``summary.status`` to a v2 ``RunStatus`` string."""
    status = (run.get("summary") or {}).get("status")
    if not status:
        return default
    if status in {"completed", "running", "warn", "failed", "queued"}:
        return status
    if status == "in_progress":
        return "running"
    return default


def cumulative_pass_series(run: dict, n_points: int = 50) -> tuple[list[str], list[float]]:
    """Return ``(x_labels, cumulative_pass_pct)`` for items in arrival order.

    Pass per item = 1 if every metric for that item passed, else 0.
    Down-samples to roughly ``n_points`` points for sparkline use.
    """
    item_pass: dict[str, bool] = {}
    item_order: list[str] = []
    for r in run.get("metric_results") or []:
        iid = r.get("item_id")
        if iid is None:
            continue
        if iid not in item_pass:
            item_pass[iid] = True
            item_order.append(iid)
        passed = r.get("passed")
        if passed is False:
            item_pass[iid] = False

    if not item_order:
        return [], []

    cumulative: list[float] = []
    seen_pass = 0
    for idx, iid in enumerate(item_order, start=1):
        if item_pass[iid]:
            seen_pass += 1
        cumulative.append(round(100.0 * seen_pass / idx, 2))

    if len(cumulative) <= n_points:
        return [str(i + 1) for i in range(len(cumulative))], cumulative

    # Down-sample evenly.
    step = len(cumulative) / n_points
    sampled: list[float] = []
    labels: list[str] = []
    for i in range(n_points):
        idx = min(int(round(i * step)), len(cumulative) - 1)
        sampled.append(cumulative[idx])
        labels.append(str(idx + 1))
    return labels, sampled


def run_dataset_dir(run: dict) -> Path:
    """Return the dataset directory containing this run."""
    rd: Path = run["_dir"]
    return rd.parent.parent


def load_dataset_items(dataset_dir: Path) -> dict[str, dict]:
    """Return ``{item_id: item_dict}`` from ``dataset_dir/dataset.jsonl``.

    Empty dict if the file is missing or unreadable. Per-line parse
    failures are logged with line numbers so operators can spot a
    corrupted dataset (silent-failure hunter punch list). Cached by
    file mtime for the same reasons as :func:`load_run`.
    """
    p = dataset_dir / "dataset.jsonl"
    try:
        st = p.stat()
    except OSError:
        return {}
    if not _caches_disabled():
        cached = _dataset_items_cache.get(p)
        if cached and cached[0] == st.st_mtime:
            return cached[1]
    out: dict[str, dict] = {}
    try:
        with p.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("dataset.jsonl line %d at %s: %s", line_num, p, exc)
                    continue
                iid = obj.get("id")
                if iid:
                    out[iid] = obj
    except OSError as exc:
        logger.warning("dataset.jsonl read failed at %s: %s", p, exc)
        return {}
    if not _caches_disabled():
        _dataset_items_cache[p] = (st.st_mtime, out)
    return out


def input_text(item: dict | None) -> str:
    """Best-effort flattening of an item's ``input`` to a short string.

    Handles three common shapes: plain string, ``{question/prompt/...}``,
    or ``{kwargs: {question: ...}}`` from the SDK's ``call_id`` capture.
    """
    if not isinstance(item, dict):
        return ""
    inp = item.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, dict):
        kw = inp.get("kwargs") if isinstance(inp.get("kwargs"), dict) else None
        for source in (kw, inp):
            if not isinstance(source, dict):
                continue
            for key in ("question", "prompt", "input", "query", "user"):
                v = source.get(key)
                if isinstance(v, str):
                    return v
    return ""
