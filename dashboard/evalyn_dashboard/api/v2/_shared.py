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

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Cache keys the FE uses with ``useV2Resource``. Listed centrally so
# both the watcher loop and tests share the same vocabulary; the FE
# matches by prefix so e.g. ``"experiments"`` invalidates both
# ``"experiments"`` (list) and ``"experiment:<id>"`` (detail) entries.
V2_CACHE_KEYS: tuple[str, ...] = (
    "home",
    "experiments",
    "experiment",  # prefix matches "experiment:<id>" + items pages
    "datasets",
    "dataset",     # prefix matches "dataset:<name>"
    "rubrics",
    "rubric",      # prefix matches "rubric:<id>"
    "reviewQueue",
    "weeklyReport",
)

# Watcher state. ``_watcher_task`` is single-process so re-calling
# ``start_watcher`` from a second ``build_app`` (e.g. in tests) is a
# no-op rather than spawning duplicate pollers.
_LAST_ROOTS_SIG: tuple[tuple[str, float], ...] | None = None
_WATCHER_TASK: asyncio.Task | None = None
_WATCHER_INTERVAL_S: float = 3.0

INVERSE_HINTS = ("hallucin", "refus_incorrect")

# Calibration / review band thresholds. Kept here (not in review.py) so the
# shared ``calibration_suggestions`` helper - consumed by both review.py and
# home.py - has a single source of truth. review.py re-imports these.
PRIMARY_LO = 0.35
PRIMARY_HI = 0.65
SOFT_LO = 0.3
SOFT_HI = 0.7
CALIBRATION_VERDICT_THRESHOLD = 10
CALIBRATION_SUGGESTIONS_CAP = 5

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
# Per-route response snapshot caches keyed on the same root-mtime
# signature. Routers that compute heavy aggregates (home, etc.) read
# from here on warm hits to avoid re-walking runs every request.
_response_snapshot_cache: dict[str, dict[tuple, dict]] = {}
# Per-reviews-file cache. Walking dataset dirs to find ``reviews/``
# subdirs is expensive (~750ms cold on WSL); cache parsed verdict
# tuples keyed on ``(path_str, mtime)`` so warm requests skip the read.
# Verdicts are append-only so the file mtime catches every new line.
_verdicts_file_cache: dict[tuple[str, float], list[dict]] = {}
# Per-root list of ``(reviews_jsonl_path, dataset_name)`` keyed on the
# root's mtime - saves thousands of ``scandir`` syscalls per warm request
# when only a handful of dataset dirs actually have a ``reviews/`` subdir.
_reviews_dirs_cache: dict[Path, tuple[float, list[tuple[Path, str]]]] = {}


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
    _response_snapshot_cache.clear()
    _verdicts_file_cache.clear()
    _reviews_dirs_cache.clear()
    _clear_run_dirs_snapshots()
    _clear_reviews_files_snapshots()
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
        _rubrics_mod._saved_rubric_index_cache.clear()
    except ImportError:
        pass
    try:
        from . import review as _review_mod  # noqa: WPS433 - intentional
        _review_mod._clear_review_caches_for_tests()
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


def _scandir_subdirs(root: Path) -> list[Path]:
    """Enumerate immediate subdirectories of ``root`` via ``os.scandir``.

    ``Path.iterdir()`` followed by ``Path.is_dir()`` issues TWO syscalls
    per entry on POSIX (one to list, one to stat). ``os.scandir()``
    returns ``DirEntry`` objects whose ``is_dir()`` uses the directory
    listing's cached type info on Linux/macOS and a single cached stat
    on Windows. For prod's 484+ dataset roots on WSL/NTFS this halves
    cold-start FS work; the warm path was already snappy.

    Sorted by name so callers don't have to. ``follow_symlinks=False``
    avoids surprise stats on broken links (a dataset folder could
    legitimately contain link entries that should NOT be recursed
    into; here we only care about real subdirs anyway).
    """
    out: list[Path] = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    out.append(Path(entry.path))
    except OSError:
        return []
    out.sort()
    return out


def list_dataset_dirs(root: Path) -> list[Path]:
    """Return cached list of dataset subdirectories under ``root``.

    Used by every endpoint that walks ``data/prod/datasets/*``. Prod has
    491 dataset folders; raw ``iterdir()`` was ~1.9s on WSL/NTFS - so
    we cache by the root's mtime (changes on add/remove of a dataset).
    The cold path itself was reduced ~2x by switching to ``os.scandir``
    via :func:`_scandir_subdirs` (one syscall per entry instead of two).
    The list is sorted so callers don't have to.
    """
    if not root.is_dir():
        return []
    if _caches_disabled():
        return _scandir_subdirs(root)
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return _scandir_subdirs(root)
    cached = _dataset_dirs_cache.get(root)
    if cached and cached[0] == mtime:
        return cached[1]
    out = _scandir_subdirs(root)
    _dataset_dirs_cache[root] = (mtime, out)
    return out


def _run_dirs_snapshot_dir() -> Path:
    """Directory holding on-disk run-dir snapshots, one file per root."""
    return Path.cwd() / ".evalyn" / "cache" / "run_dirs"


def _run_dirs_snapshot_path(root: Path) -> Path:
    """Snapshot file for ``root``, named by a stable hash of its absolute path.

    Hashing the path keeps file names short and filesystem-safe across
    Windows/Linux while still being unique per root (we walk both prod
    and demo roots). The snapshot itself records the absolute path so a
    cross-checkout collision would self-detect.
    """
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return _run_dirs_snapshot_dir() / f"{digest}.json"


def _load_run_dirs_snapshot(root: Path, mtime: float) -> list[Path] | None:
    """Read the persisted run-dir list for ``root`` if it matches ``mtime``.

    Returns ``None`` on miss, mismatch, or any I/O / parse error - callers
    fall back to the live filesystem walk and rewrite the snapshot.
    """
    path = _run_dirs_snapshot_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("run_dirs snapshot parse failed at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("root") != str(root):
        return None
    if payload.get("mtime") != mtime:
        return None
    paths = payload.get("paths")
    if not isinstance(paths, list):
        return None
    out: list[Path] = []
    for p in paths:
        if not isinstance(p, str):
            return None
        candidate = Path(p)
        # If a dir vanished since the snapshot was written, skip it
        # silently rather than poison the cache; the next mtime bump
        # rebuilds from scratch.
        if candidate.is_dir():
            out.append(candidate)
    return out


def _save_run_dirs_snapshot(root: Path, mtime: float, paths: list[Path]) -> None:
    """Atomically persist the run-dir list for ``root`` keyed on ``mtime``.

    Best-effort: any I/O error is logged at WARN and swallowed - the
    in-memory cache is still valid, we just take the slow path next boot.
    Atomic via ``NamedTemporaryFile`` + ``os.replace`` so a partial
    write can't corrupt a future read.
    """
    snap_dir = _run_dirs_snapshot_dir()
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("run_dirs snapshot mkdir failed at %s: %s", snap_dir, exc)
        return
    payload = {
        "root": str(root),
        "mtime": mtime,
        "paths": [str(p) for p in paths],
    }
    final = _run_dirs_snapshot_path(root)
    try:
        # NamedTemporaryFile writes to the same dir so os.replace is atomic.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(snap_dir),
            prefix=".tmp-",
            suffix=".json",
        ) as f:
            json.dump(payload, f)
            tmp_name = f.name
        os.replace(tmp_name, final)
    except OSError as exc:
        logger.warning("run_dirs snapshot write failed at %s: %s", final, exc)


def _clear_run_dirs_snapshots() -> None:
    """Remove every persisted run-dir snapshot. Test-only seam."""
    snap_dir = _run_dirs_snapshot_dir()
    if not snap_dir.is_dir():
        return
    for entry in snap_dir.iterdir():
        try:
            entry.unlink()
        except OSError:
            pass


def _list_run_dirs_for_root(root: Path) -> list[Path]:
    """Return cached list of run dirs under ``root``, mtime-invalidated.

    ``root.stat().st_mtime`` changes when a dataset directory is added
    or removed - cheap signal that catches the only mutation we care
    about (new dataset folders). Per-dataset run additions don't bump
    the root mtime but a new dataset typically arrives before its first
    run anyway, so the cache reaches steady state quickly. Callers that
    need stronger freshness can invalidate via ``_clear_caches_for_tests``.

    Cache layers (cheapest first):
      1. Process memory (``_run_dirs_cache``).
      2. On-disk snapshot under ``.evalyn/cache/run_dirs/`` keyed on the
         root's mtime - survives process restarts so the first request
         after a fresh dashboard launch skips the multi-second WSL/NTFS
         iterdir walk over potentially hundreds of empty dataset dirs.
      3. Live filesystem walk via :func:`_walk_run_dirs`.
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
    snapshot = _load_run_dirs_snapshot(root, mtime)
    if snapshot is not None:
        _run_dirs_cache[root] = (mtime, snapshot)
        return snapshot
    out = _walk_run_dirs(root)
    _run_dirs_cache[root] = (mtime, out)
    _save_run_dirs_snapshot(root, mtime, out)
    return out


def _walk_run_dirs(root: Path) -> list[Path]:
    """Filesystem walk that ``_list_run_dirs_for_root`` memoizes.

    Cold-path optimization: each ``eval_runs`` subdir is enumerated via
    ``os.scandir`` (cached ``is_dir`` info from the listing) instead of
    ``Path.iterdir()`` + ``Path.is_dir()`` per entry. The outer presence
    check on ``eval_runs`` itself is folded into a single ``scandir``
    call wrapped in ``OSError`` - if the directory is missing or
    inaccessible we skip silently, same semantics as the previous
    ``is_dir()``-then-``iterdir()`` two-stat pattern.
    """
    out: list[Path] = []
    for dataset_dir in list_dataset_dirs(root):
        eval_runs_dir = dataset_dir / "eval_runs"
        try:
            with os.scandir(eval_runs_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        out.append(Path(entry.path))
        except (FileNotFoundError, NotADirectoryError):
            # eval_runs is absent or got replaced by a regular file
            # (rare). Same as the old ``is_dir()`` False branch.
            continue
        except OSError:
            # Permission denied / broken symlink / WSL hiccup - log
            # nothing (this is a hot path) and skip.
            continue
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


def roots_signature() -> tuple[tuple[str, float], ...]:
    """Return ``((root_path_str, mtime), ...)`` snapshot of every dataset root.

    Stable cache key for whole-route response snapshots (e.g. /home).
    Roots that fail to stat are silently skipped so the key is still
    formable. ``str(root)`` (not the ``Path``) makes the key picklable
    and hashable in a portable form.
    """
    sig: list[tuple[str, float]] = []
    for r in dataset_roots():
        try:
            sig.append((str(r), r.stat().st_mtime))
        except OSError:
            continue
    return tuple(sig)


def get_cached_snapshot(route: str) -> dict | None:
    """Return a cached response snapshot for ``route`` or ``None``.

    Returns ``None`` when caches are disabled (test seam) or when no
    matching snapshot exists for the current roots signature. Caller
    is expected to (re)compute and call :func:`set_cached_snapshot`.
    """
    if _caches_disabled():
        return None
    bucket = _response_snapshot_cache.get(route)
    if not bucket:
        return None
    return bucket.get(roots_signature())


def set_cached_snapshot(route: str, snapshot: dict) -> None:
    """Store a fully-computed response ``snapshot`` for ``route``.

    Keyed on the current :func:`roots_signature`; invalidates as soon as
    any dataset root's mtime changes (new run dir created, dataset
    added/removed). No-ops when caches are disabled.
    """
    if _caches_disabled():
        return
    _response_snapshot_cache.setdefault(route, {})[roots_signature()] = snapshot


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


def cost_or_zero(run: dict) -> float:
    """Return ``total_cost(run)`` or ``0.0`` when missing.

    Helper for aggregations where ``None`` cost should not propagate
    (e.g. summing across runs, where missing == free run).
    """
    c = total_cost(run)
    return float(c) if c is not None else 0.0


def daily_cost_buckets(
    runs: Iterable[dict], anchor: datetime, days: int
) -> list[float]:
    """Bucket per-run cost into ``days`` daily buckets ending at ``anchor``.

    Each bucket sums ``cost_or_zero`` for runs whose ``created_at`` falls
    on that calendar date (UTC). Returns a list of length ``days``,
    oldest day first. Days with no runs contribute 0.0.

    Inspects: ``created_at`` (parsed via :func:`parse_iso`); runs without
    a parseable timestamp are skipped (not silently bucketed into "today")
    so callers can spot data quality issues via run counts.
    """
    end_date = anchor.date()
    start_date = end_date - timedelta(days=days - 1)
    buckets = [0.0] * days
    for run in runs:
        dt = parse_iso(run.get("created_at"))
        if dt is None:
            logger.warning("daily_cost_buckets: run with unparseable created_at")
            continue
        run_date = dt.date()
        if run_date < start_date or run_date > end_date:
            continue
        idx = (run_date - start_date).days
        buckets[idx] += cost_or_zero(run)
    return buckets


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


def has_any_calibration() -> bool:
    """True iff any dataset under any root has a ``calibrations/`` subdir.

    Walks every dataset directory once; cheap (mtime-cached list) but not
    memoised because callers (currently the home attention builder) only
    call it once per request and it short-circuits on the first hit.
    """
    for root in dataset_roots():
        for ds in list_dataset_dirs(root):
            calib = ds / "calibrations"
            if calib.is_dir():
                try:
                    if any(calib.iterdir()):
                        return True
                except OSError as exc:
                    logger.warning("calibrations iter failed at %s: %s", calib, exc)
                    continue
    return False


def median(values: list[float]) -> float | None:
    """Return median of ``values`` or ``None`` for empty input.

    Pure stdlib; we avoid ``statistics.median`` only to keep the import
    surface tiny and the behavior on empty input explicit.
    """
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


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


# ---------------------------------------------------------------------------
# Calibration suggestions (shared across review.py and home.py)
# ---------------------------------------------------------------------------


def _is_soft_score(score: float | int | None) -> bool:
    """A score is soft (LLM-judge-style) when strictly between 0 and 1."""
    if not isinstance(score, (int, float)):
        return False
    return 0.0 < float(score) < 1.0


def _verdict_metric_id(
    item_id: str,
    run_metric_results_by_item: dict[str, list[dict]],
) -> str | None:
    """Pick the most likely calibration metric_id for ``item_id`` in a run.

    Mirrors the review-queue cascade so a verdict is attributed to the
    same metric the cascade would have surfaced. Returns ``None`` when
    the item is missing from the run.
    """
    results = run_metric_results_by_item.get(item_id) or []
    if not results:
        return None
    for mr in results:
        details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
        jc = details.get("judge_confidence")
        if isinstance(jc, (int, float)) and PRIMARY_LO <= jc <= PRIMARY_HI:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    for mr in results:
        score = mr.get("score")
        if _is_soft_score(score) and SOFT_LO <= float(score) <= SOFT_HI:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    for mr in results:
        if mr.get("passed") is False:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    mid = results[0].get("metric_id")
    return str(mid) if mid else None


def _read_verdicts(path: Path, mtime: float | None = None) -> list[dict]:
    """Parse a reviews jsonl file. Bad lines are logged and skipped.

    Cached by ``(path, mtime)`` - reviews files only grow (verdicts are
    append-only) so the mtime check catches every new line.
    """
    if mtime is not None and not _caches_disabled():
        cached = _verdicts_file_cache.get((str(path), mtime))
        if cached is not None:
            return cached
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "verdicts jsonl line %d at %s: %s", line_num, path, exc
                    )
    except OSError as exc:
        logger.warning("verdicts jsonl read failed at %s: %s", path, exc)
    if mtime is not None and not _caches_disabled():
        _verdicts_file_cache[(str(path), mtime)] = out
    return out


def _walk_reviews_files(root: Path) -> list[tuple[Path, str]]:
    """Filesystem walk that ``_reviews_files_for_root`` memoizes."""
    out: list[tuple[Path, str]] = []
    for ds_dir in list_dataset_dirs(root):
        reviews_dir = ds_dir / "reviews"
        # ``glob`` returns ``[]`` silently when the dir is missing - one
        # syscall vs the explicit ``is_dir`` + ``iterdir`` pair.
        for path in sorted(reviews_dir.glob("*.jsonl")):
            out.append((path, ds_dir.name))
    return out


def _reviews_files_snapshot_dir() -> Path:
    """Directory holding on-disk reviews-files snapshots, one file per root."""
    return Path.cwd() / ".evalyn" / "cache" / "reviews_files"


def _reviews_files_snapshot_path(root: Path) -> Path:
    """Snapshot file for ``root``, hashed-name-keyed (mirrors run-dirs)."""
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return _reviews_files_snapshot_dir() / f"{digest}.json"


def _load_reviews_files_snapshot(
    root: Path, mtime: float,
) -> list[tuple[Path, str]] | None:
    """Read the persisted reviews-files list for ``root`` if mtime matches.

    Returns ``None`` on miss, mismatch, or any I/O/parse error - caller
    falls back to the live walk and rewrites the snapshot.

    Files that vanished since the snapshot was written are skipped
    silently (same defensive shape as run-dirs); poisoning the cache
    with stale paths would surface as confusing 404s in the
    calibration_suggestions hot path. The mtime bump on the next
    dataset add/remove rebuilds from scratch anyway.
    """
    path = _reviews_files_snapshot_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("reviews_files snapshot parse failed at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("root") != str(root):
        return None
    if payload.get("mtime") != mtime:
        return None
    files = payload.get("files")
    if not isinstance(files, list):
        return None
    out: list[tuple[Path, str]] = []
    for entry in files:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], str)
        ):
            return None
        candidate = Path(entry[0])
        if candidate.is_file():
            out.append((candidate, entry[1]))
    return out


def _save_reviews_files_snapshot(
    root: Path, mtime: float, files: list[tuple[Path, str]],
) -> None:
    """Atomically persist the reviews-files list for ``root`` keyed on mtime.

    Best-effort: I/O error -> WARN log + swallow. The in-memory cache
    is unaffected; we just take the slow walk next boot. Atomic via
    ``NamedTemporaryFile`` + ``os.replace`` (same shape as run-dirs).
    """
    snap_dir = _reviews_files_snapshot_dir()
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "reviews_files snapshot mkdir failed at %s: %s", snap_dir, exc
        )
        return
    payload = {
        "root": str(root),
        "mtime": mtime,
        "files": [[str(p), name] for p, name in files],
    }
    final = _reviews_files_snapshot_path(root)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(snap_dir),
            prefix=".tmp-",
            suffix=".json",
        ) as f:
            json.dump(payload, f)
            tmp_name = f.name
        os.replace(tmp_name, final)
    except OSError as exc:
        logger.warning("reviews_files snapshot write failed at %s: %s", final, exc)


def _clear_reviews_files_snapshots() -> None:
    """Remove every persisted reviews-files snapshot. Test-only seam."""
    snap_dir = _reviews_files_snapshot_dir()
    if not snap_dir.is_dir():
        return
    for entry in snap_dir.iterdir():
        try:
            entry.unlink()
        except OSError:
            pass


def _reviews_files_for_root(root: Path) -> list[tuple[Path, str]]:
    """Return ``[(jsonl_path, dataset_name), ...]`` under ``root``, mtime-cached.

    Cache layers (cheapest first), mirroring ``_list_run_dirs_for_root``:
      1. Process memory (``_reviews_dirs_cache``).
      2. On-disk snapshot under ``.evalyn/cache/reviews_files/`` keyed
         on the root's mtime - survives process restarts so the first
         request after a fresh dashboard launch skips the ~800ms WSL/
         NTFS glob walk over hundreds of placeholder ``reviews/`` dirs.
         calibration_suggestions sits on the /home and /review hot paths.
      3. Live filesystem walk via :func:`_walk_reviews_files`.
    """
    if not root.is_dir():
        return []
    if _caches_disabled():
        return _walk_reviews_files(root)
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return _walk_reviews_files(root)
    cached = _reviews_dirs_cache.get(root)
    if cached and cached[0] == mtime:
        return cached[1]
    snapshot = _load_reviews_files_snapshot(root, mtime)
    if snapshot is not None:
        _reviews_dirs_cache[root] = (mtime, snapshot)
        return snapshot
    out = _walk_reviews_files(root)
    _reviews_dirs_cache[root] = (mtime, out)
    _save_reviews_files_snapshot(root, mtime, out)
    return out


def _latest_calibration_mtime(dataset_dir: Path, metric_id: str) -> float | None:
    """Return the mtime of the metric's calibration file, or ``None`` if absent."""
    cal = dataset_dir / "calibrations" / metric_id / "calibration.json"
    try:
        return cal.stat().st_mtime
    except OSError:
        return None


def calibration_suggestions(runs: list[dict]) -> list[dict]:
    """Walk every dataset's reviews/*.jsonl and emit calibration suggestions.

    Aggregates verdicts by (dataset, metric_id). A suggestion fires when
    the count crosses :data:`CALIBRATION_VERDICT_THRESHOLD` AND no existing
    calibration file is newer than the most recent verdict for that metric.
    Capped at :data:`CALIBRATION_SUGGESTIONS_CAP`. Output shape matches
    what ``ReviewQueue.calibration_suggestions`` exposes.
    """
    # Build a per-run lookup (rid -> {item_id -> [metric_results]}) once
    # so we don't re-walk metric_results per verdict.
    run_lookup: dict[str, tuple[Path, dict[str, list[dict]]]] = {}
    for run in runs:
        rid = run_id(run)
        ds_dir = run_dataset_dir(run)
        per_item: dict[str, list[dict]] = defaultdict(list)
        for mr in run.get("metric_results") or []:
            iid = mr.get("item_id")
            if iid:
                per_item[iid].append(mr)
        run_lookup[rid] = (ds_dir, per_item)
        # Some runs also identify themselves by ``id`` (legacy).
        legacy_id = run.get("id")
        if isinstance(legacy_id, str) and legacy_id and legacy_id != rid:
            run_lookup.setdefault(legacy_id, (ds_dir, per_item))

    counts: dict[tuple[Path, str, str], int] = defaultdict(int)
    latest_verdict_mtime: dict[tuple[Path, str], float] = {}

    for root in dataset_roots():
        for path, ds_name in _reviews_files_for_root(root):
            try:
                f_mtime = path.stat().st_mtime
            except OSError:
                f_mtime = 0.0
            ds_dir = path.parent.parent
            for verdict in _read_verdicts(path, mtime=f_mtime):
                rid = verdict.get("source_run_id") or ""
                iid = verdict.get("item_id") or ""
                if not rid or not iid:
                    continue
                lookup = run_lookup.get(rid)
                if lookup is None:
                    continue
                _, per_item = lookup
                metric_id = _verdict_metric_id(iid, per_item)
                if not metric_id:
                    continue
                counts[(ds_dir, ds_name, metric_id)] += 1
                key = (ds_dir, metric_id)
                if f_mtime > latest_verdict_mtime.get(key, 0.0):
                    latest_verdict_mtime[key] = f_mtime

    suggestions: list[dict] = []
    for (ds_dir, ds_name, metric_id), count in counts.items():
        if count < CALIBRATION_VERDICT_THRESHOLD:
            continue
        cal_mtime = _latest_calibration_mtime(ds_dir, metric_id)
        v_mtime = latest_verdict_mtime.get((ds_dir, metric_id), 0.0)
        if cal_mtime is not None and cal_mtime >= v_mtime:
            # Already calibrated since the latest verdict landed.
            continue
        annotations_path = str(ds_dir / "reviews")
        suggestions.append({
            "metric_id": metric_id,
            "dataset": ds_name,
            "verdict_count": count,
            "threshold": CALIBRATION_VERDICT_THRESHOLD,
            "cli_args": {
                "metric_id": metric_id,
                "annotations": annotations_path,
            },
        })
    suggestions.sort(key=lambda s: (-s["verdict_count"], s["dataset"], s["metric_id"]))
    return suggestions[:CALIBRATION_SUGGESTIONS_CAP]


# ---------------------------------------------------------------------------
# Background watcher: poll dataset_roots() mtimes and broadcast invalidations.
# ---------------------------------------------------------------------------


def _drop_response_caches() -> None:
    """Clear in-memory v2 caches so the next request rebuilds.

    Mirrors ``_clear_caches_for_tests`` minus the test-only sibling
    invalidations: only the caches keyed on root mtimes are dropped
    (per-file mtime caches stay because they self-invalidate).
    """
    _response_snapshot_cache.clear()
    _run_dirs_cache.clear()
    _dataset_dirs_cache.clear()
    _all_runs_cache.clear()
    _reviews_dirs_cache.clear()


async def _watcher_loop() -> None:
    """Periodically diff ``roots_signature()`` and broadcast on change.

    Runs forever until cancelled. Imports :func:`broadcast` lazily to
    avoid a circular import at module load (``v2_ws`` doesn't import
    this module, but keeping the import lazy makes the dependency
    direction obvious).
    """
    from .v2_ws import broadcast  # local import: avoid load-time cycle

    global _LAST_ROOTS_SIG
    # Seed the signature so the first iteration doesn't fire spuriously
    # for a workspace that has been on disk all along.
    _LAST_ROOTS_SIG = roots_signature()
    while True:
        try:
            await asyncio.sleep(_WATCHER_INTERVAL_S)
            sig = roots_signature()
            if sig == _LAST_ROOTS_SIG:
                continue
            _LAST_ROOTS_SIG = sig
            _drop_response_caches()
            await broadcast(
                {"type": "cache_invalidate", "keys": list(V2_CACHE_KEYS)}
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never let the loop die
            logger.warning("v2 watcher loop error: %s", exc)


def start_watcher() -> None:
    """Start the watcher task on the current event loop.

    Idempotent: a second call while a task is alive is a no-op. Called
    from FastAPI's ``startup`` hook so the loop is the same one the WS
    handler runs on.
    """
    global _WATCHER_TASK
    if _WATCHER_TASK is not None and not _WATCHER_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet (e.g. called outside startup). Caller
        # should retry from a coroutine; bail quietly rather than crash.
        logger.debug("start_watcher called with no running loop; skipping")
        return
    _WATCHER_TASK = loop.create_task(_watcher_loop())


async def stop_watcher() -> None:
    """Cancel the watcher task and wait for it to exit (test seam)."""
    global _WATCHER_TASK
    if _WATCHER_TASK is None:
        return
    task = _WATCHER_TASK
    _WATCHER_TASK = None
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


_PREWARM_TASK: asyncio.Task | None = None


def _prewarm_blocking() -> None:
    """Synchronous worker that fills the run cache. Called via ``to_thread``.

    Walks every dataset root once via :func:`load_all_runs`, populating
    :data:`_run_dirs_cache`, :data:`_run_cache`, and :data:`_all_runs_cache`
    so the first inbound request hits warm caches instead of paying the
    multi-second cold walk + JSON parse. On a workspace with persisted
    snapshots from a previous boot the FS walk itself is also skipped.

    Also primes :data:`_reviews_dirs_cache` per root. The reviews-files
    walk is itself ~800ms cold on workspaces with hundreds of dataset
    placeholder dirs (one ``Path.glob`` per dir even when the
    ``reviews/`` subdir is absent), and it sits on the hot path for
    /home and /review first paint via :func:`calibration_suggestions`.
    Priming it here moves that cost off the user's first click.
    Logs the elapsed time on success so operators tuning cold-start
    have a metric: "v2 prewarm completed in 50ms" vs "...3500ms"
    distinguishes warm-cache-hit from cold-FS-walk; correlate with
    workspace size to budget appropriately.
    """
    import time as _time

    start = _time.monotonic()
    try:
        load_all_runs()
        for root in dataset_roots():
            _reviews_files_for_root(root)
    except Exception as exc:  # noqa: BLE001 - never block startup on a cache miss
        elapsed_ms = (_time.monotonic() - start) * 1000
        logger.warning("v2 prewarm aborted after %.0f ms: %s", elapsed_ms, exc)
        return
    elapsed_ms = (_time.monotonic() - start) * 1000
    logger.info("v2 prewarm completed in %.0f ms", elapsed_ms)


def prewarm() -> None:
    """Start the background prewarm task on the current event loop.

    Idempotent: a second call while a task is alive is a no-op. Called
    from the FastAPI ``startup`` hook alongside :func:`start_watcher`;
    the work happens in a thread so we don't block the event loop the
    WS handler uses. If no loop is running yet (e.g. called outside
    startup) the call is a quiet no-op so the caller can ship later.
    """
    global _PREWARM_TASK
    if _PREWARM_TASK is not None and not _PREWARM_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("prewarm called with no running loop; skipping")
        return
    _PREWARM_TASK = loop.create_task(asyncio.to_thread(_prewarm_blocking))


async def await_prewarm() -> None:
    """Wait for the prewarm task to finish if one is running. Test seam.

    Production code never awaits prewarm - it's deliberately fire-and-
    forget so a slow walk on a fresh workspace doesn't delay the first
    HTTP/WS handshake. Tests that want deterministic cache state after
    startup can ``await await_prewarm()`` to block until done.
    """
    task = _PREWARM_TASK
    if task is None:
        return
    try:
        await task
    except Exception:  # noqa: BLE001
        pass
