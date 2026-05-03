"""``/api/compare`` router.

GET ``/api/compare?run_a=<id>&run_b=<id>`` reads ``results.json`` for both
runs from disk under ``.evalyn/data/datasets/<dataset>/eval_runs/<run_id>/`` and
returns an aligned per-row diff payload that the frontend's CompareView
renders directly.

Approach choice (see plan): disk-read over subprocess. The dashboard
already locates run dirs the same way (``api/runs.py``); reusing that
helper keeps the request synchronous and side-effect free, and avoids
having to plumb a JobManager + stdout buffering for what is fundamentally
a pure transformation. Subprocess would only be necessary if the diff
required new code paths the SDK already implements but the dashboard
does not — and the per-row delta computation is simple enough to embed.

Schema (mirrors frontend ``CompareResponse`` in ``api.ts``):

.. code-block:: json

    {
      "run_a": {"id": "...", "dataset": "...", "pass": 0.84,
                 "metrics": {"helpfulness": 0.91}, "cost": 0.12},
      "run_b": {"id": "...", ...},
      "items": [
        {
          "input_hash": "abc",
          "input_preview": "first 200 chars of input",
          "a": {"pass": true,  "metrics": {"helpfulness": 1.0},
                 "output": "..."},
          "b": {"pass": false, "metrics": {"helpfulness": 0.0},
                 "output": "..."},
          "delta": -1.0
        }
      ]
    }

If either run id is not found, the endpoint returns 404. Empty items list
is returned (200) when both runs exist but neither has metric_results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Cap the input/output preview rendered in compare rows. Matches the spec.
_PREVIEW_CHARS = 200


def _datasets_root() -> Path:
    """Return ``.evalyn/data/datasets/`` resolved against ``cwd``.

    Per SDK convention (``sdk/evalyn_sdk/analysis/core.py``), runs live
    at ``<datasets_root>/<dataset>/eval_runs/<run>/``.
    """
    return (Path.cwd() / ".evalyn" / "data" / "datasets").resolve()


def _find_run_dir(run_id: str) -> Path | None:
    """Locate the run directory for ``run_id`` under any dataset.

    Mirrors ``api/runs.py:get_run`` so both endpoints agree on layout.
    Walks ``<datasets_root>/<dataset>/eval_runs/<run>/``. Returns
    ``None`` if not found.
    """
    root = _datasets_root()
    if not root.exists() or not root.is_dir():
        return None
    try:
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            eval_runs_dir = dataset_dir / "eval_runs"
            if not eval_runs_dir.is_dir():
                continue
            candidate = eval_runs_dir / run_id
            if (candidate / "results.json").exists():
                return candidate
    except OSError:
        return None
    return None


def _load_results(run_dir: Path) -> dict[str, Any] | None:
    """Parse ``results.json`` for one run dir; ``None`` on read/parse error."""
    path = run_dir / "results.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("compare: failed to parse %s: %s", path, exc)
        return None


def _load_dataset_index(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the dataset.jsonl as ``{item_id: {input, output, ...}}``.

    Empty mapping when the file is missing or unreadable. The dataset
    holds the human-readable ``input``/``output`` for items; the run's
    ``metric_results`` reference items by ``item_id`` only, so we need
    this side index to render previews.
    """
    path = dataset_dir / "dataset.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                rows = json.loads(f.read() or "[]")
                for row in rows:
                    iid = row.get("id")
                    if iid:
                        out[str(iid)] = row
                return out
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "compare: skipping malformed jsonl line %d in %s: %s",
                        line_num,
                        path,
                        exc,
                    )
                    continue
                iid = row.get("id")
                if iid:
                    out[str(iid)] = row
    except OSError as exc:
        logger.warning("compare: failed to read dataset %s: %s", path, exc)
    return out


def _safe_json_dump(raw: Any) -> str:
    """Stable JSON dump with a string fallback for non-serialisable values."""
    try:
        return json.dumps(raw, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(raw)


def _stringify_input(raw: Any) -> str:
    """Coerce a dataset input field into a single string for preview/hashing."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        kwargs = raw.get("kwargs")
        if isinstance(kwargs, dict):
            # Prefer common semantic keys; fall back to a stable JSON dump.
            for key in ("question", "prompt", "input", "query", "text"):
                v = kwargs.get(key)
                if isinstance(v, str) and v:
                    return v
    return _safe_json_dump(raw)


def _stringify_output(raw: Any) -> str:
    """Coerce a dataset output field into a single string for preview."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return _safe_json_dump(raw)


_WS_RE = re.compile(r"\s+")


def _hash_input(text: str) -> str:
    """16-char SHA256 prefix; deterministic alignment key across runs.

    Whitespace is normalised before hashing so trivial differences (CRLF
    vs LF, trailing spaces, indentation drift across an editor save) do
    not de-align rows that should match. Two inputs with the same
    whitespace-collapsed form produce the same hash; users almost
    always want this in compare.
    """
    canonical = _WS_RE.sub(" ", text).strip()
    return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()[:16]


def _truncate(text: str, n: int = _PREVIEW_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _summary_pass(summary: dict[str, Any]) -> float | None:
    """Best-effort pass-rate extraction from a summary block."""
    if not isinstance(summary, dict):
        return None
    raw = summary.get("pass_rate")
    if isinstance(raw, (int, float)):
        return float(raw)
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics:
        rates: list[float] = []
        for v in metrics.values():
            if isinstance(v, dict):
                pr = v.get("pass_rate")
                if isinstance(pr, (int, float)):
                    rates.append(float(pr))
        if rates:
            return sum(rates) / len(rates)
    return None


def _summary_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    """Per-metric mean score from the summary block; values may be null."""
    out: dict[str, float | None] = {}
    if not isinstance(summary, dict):
        return out
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return out
    for metric_id, payload in metrics.items():
        if not isinstance(payload, dict):
            out[str(metric_id)] = None
            continue
        avg = payload.get("avg_score")
        if isinstance(avg, (int, float)):
            out[str(metric_id)] = float(avg)
        else:
            pr = payload.get("pass_rate")
            out[str(metric_id)] = float(pr) if isinstance(pr, (int, float)) else None
    return out


def _total_cost(usage_summary: Any) -> float | None:
    """Pull a USD total from ``usage_summary`` if present."""
    if not isinstance(usage_summary, dict):
        return None
    for key in ("total_cost_usd", "total_cost", "cost_usd", "cost"):
        v = usage_summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _per_item_metrics(
    metric_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Group ``metric_results`` by ``item_id`` -> {metric_id: score}.

    When the same ``item_id``+``metric_id`` appears multiple times (judge
    retries; rare) the last entry wins — mirrors the SDK's last-write
    semantics for trace replay.
    """
    out: dict[str, dict[str, float]] = {}
    for r in metric_results:
        if not isinstance(r, dict):
            continue
        item_id = r.get("item_id")
        metric_id = r.get("metric_id")
        score = r.get("score")
        if not item_id or not metric_id:
            continue
        if not isinstance(score, (int, float)):
            continue
        out.setdefault(str(item_id), {})[str(metric_id)] = float(score)
    return out


def _per_item_pass(
    metric_results: list[dict[str, Any]],
) -> dict[str, list[bool]]:
    """Group per-item pass flags. An item passes overall iff every metric
    on that item passed (matches the CLI ``compare`` behavior)."""
    out: dict[str, list[bool]] = {}
    for r in metric_results:
        if not isinstance(r, dict):
            continue
        item_id = r.get("item_id")
        passed = r.get("passed")
        if not item_id or not isinstance(passed, bool):
            continue
        out.setdefault(str(item_id), []).append(passed)
    return out


def _build_run_side(
    run_dir: Path,
    run_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the per-run summary block + item -> {input_hash, input_preview,
    pass, metrics, output} side index."""
    summary = run_data.get("summary") or {}
    metrics_summary = _summary_metrics(summary)
    pass_rate = _summary_pass(summary)
    cost = _total_cost(run_data.get("usage_summary"))

    # Layout: ``<datasets_root>/<dataset>/eval_runs/<run>/``. The dataset
    # directory (where ``dataset.jsonl`` lives) is two levels up from the
    # run directory; ``run_dir.parent`` is the ``eval_runs`` folder.
    dataset_dir = run_dir.parent.parent
    summary_block = {
        "id": run_dir.name,
        "dataset": dataset_dir.name,
        "pass": pass_rate,
        "metrics": metrics_summary,
        "cost": cost,
    }

    metric_results = run_data.get("metric_results") or []
    if not isinstance(metric_results, list):
        metric_results = []

    per_item_metrics = _per_item_metrics(metric_results)
    per_item_pass = _per_item_pass(metric_results)
    dataset_index = _load_dataset_index(dataset_dir)

    items_side: dict[str, dict[str, Any]] = {}
    # We key items by input_hash so two runs that share inputs (verbatim)
    # align even when item_ids differ across regenerated datasets.
    seen_item_ids: set[str] = set()
    for item_id, metrics in per_item_metrics.items():
        seen_item_ids.add(item_id)
        ds_row = dataset_index.get(item_id) or {}
        input_text = _stringify_input(ds_row.get("input"))
        output_text = _stringify_output(ds_row.get("output"))
        flags = per_item_pass.get(item_id) or []
        passed: bool | None = all(flags) if flags else None
        ihash = _hash_input(input_text) if input_text else f"item:{item_id}"
        items_side[ihash] = {
            "input_preview": _truncate(input_text) if input_text else f"(item {item_id[:8]})",
            "input_hash": ihash,
            "side": {
                "pass": passed,
                "metrics": metrics,
                "output": _truncate(output_text) if output_text else None,
            },
        }
    # Items present in metric_results but missing from the dataset still
    # render with a synthetic "(item …)" preview; that's the branch above.
    return summary_block, items_side


def _align_rows(
    side_a: dict[str, dict[str, Any]],
    side_b: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join two per-side maps on input_hash. Rows present on only one side
    appear with a null on the missing side and ``delta = NaN`` (serialized
    as JSON null since NaN isn't valid JSON)."""
    keys = list(side_a.keys()) + [k for k in side_b.keys() if k not in side_a]
    rows: list[dict[str, Any]] = []
    for key in keys:
        a_entry = side_a.get(key)
        b_entry = side_b.get(key)
        # Prefer A's preview when both exist (deterministic; A is "before").
        meta = a_entry or b_entry
        assert meta is not None  # one of them must be present by construction
        a_side = a_entry["side"] if a_entry else None
        b_side = b_entry["side"] if b_entry else None
        delta: float | None
        if a_side is not None and b_side is not None:
            shared = set(a_side["metrics"].keys()) & set(b_side["metrics"].keys())
            if shared:
                delta = sum(b_side["metrics"][m] - a_side["metrics"][m] for m in shared)
            else:
                delta = None
        else:
            delta = None
        rows.append(
            {
                "input_hash": key,
                "input_preview": meta["input_preview"],
                "a": a_side,
                "b": b_side,
                # JSON null is the on-the-wire form for "incomparable"; the
                # frontend's `Number.isFinite` check handles it.
                "delta": delta if delta is not None else float("nan"),
            }
        )
    return rows


def _safe_floats(payload: Any) -> Any:
    """Recursively replace ``NaN``/``inf`` floats with ``None`` so the
    response is valid JSON. FastAPI/Starlette's JSON encoder accepts NaN
    by default (Python's permissive json), but strict clients (and our
    frontend's ``Number.isFinite`` checks) do not."""
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, dict):
        return {k: _safe_floats(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_safe_floats(v) for v in payload]
    return payload


@router.get("")
async def compare(
    run_a: str = Query(..., description="First run id"),
    run_b: str = Query(..., description="Second run id"),
) -> JSONResponse:
    """Return aligned per-row diff data for two run ids.

    Both run ids are required. 404 when either run is missing. Same id on
    both sides is allowed (the frontend renders a hint banner) and yields
    a row table where every delta is zero — useful as a sanity test.
    """
    a_id = (run_a or "").strip()
    b_id = (run_b or "").strip()
    if not a_id or not b_id:
        raise HTTPException(400, "run_a and run_b are required")
    # Anchor both ids to a single directory leaf — `.exists()` probes
    # below would otherwise let an attacker test arbitrary paths via
    # `../../...` traversal and infer their existence from 200/404.
    for label, val in (("run_a", a_id), ("run_b", b_id)):
        if "/" in val or "\\" in val or ".." in val:
            raise HTTPException(400, f"{label} contains illegal characters")

    a_dir = _find_run_dir(a_id)
    if a_dir is None:
        raise HTTPException(404, f"run {a_id} not found")
    b_dir = _find_run_dir(b_id)
    if b_dir is None:
        raise HTTPException(404, f"run {b_id} not found")

    a_data = _load_results(a_dir)
    b_data = _load_results(b_dir)
    if a_data is None or b_data is None:
        raise HTTPException(500, "failed to parse one or both runs")

    a_summary, a_side = _build_run_side(a_dir, a_data)
    b_summary, b_side = _build_run_side(b_dir, b_data)
    items = _align_rows(a_side, b_side)

    payload = {
        "run_a": a_summary,
        "run_b": b_summary,
        "items": items,
    }
    return JSONResponse(_safe_floats(payload))
