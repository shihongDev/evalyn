"""``/api/v2/rubrics`` router.

Endpoints:
- ``GET /``                 -> ``RubricList`` (augmented with kappa/drift/etc.)
- ``GET /trust``            -> ``TrustScoreboard``
- ``GET /{id}/calibration`` -> ``RubricDetail`` (augmented with weights/dimensions)
- ``POST /{id}``            -> save updated rubric (name, weights, dimensions)

Rubric ids are derived from metric definitions seen across runs. Cohen's
kappa is computed from the optional
``.evalyn/data/datasets/<dataset>/calibrations/<metric>/calibration.json``
file when present (shape: ``{annotations: [{item_id, judge_verdict,
human_verdict}]}``). Source-of-truth shapes live in
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ._shared import (
    _all_runs_cache_key,
    _caches_disabled,
    dataset_roots,
    list_dataset_dirs,
    load_all_runs,
    parse_iso,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Rubric files are saved here per dataset. The first existing dataset
# directory under any root is used; falls back to the dashboard's demo
# fixture path when nothing exists yet.
_RUBRIC_FILE_DIR = "rubrics"

# Mtime-keyed cache. The walk over ``data/prod/datasets/`` is ~1s on
# WSL even when zero calibrations exist (cost is in the dataset_dir
# stat()s, not file reads). Cleared via ``_clear_caches_for_tests``.
_calibration_index_cache: dict[tuple, dict[str, list[Path]]] = {}

# Mtime-keyed cache of persisted rubric overrides. The previous
# ``_load_saved_rubric`` did O(metrics x dataset_dirs) stat() probes;
# with ~40 metrics x ~495 dataset dirs on prod the warm path was 40s
# on WSL/NTFS just to render the metrics page. Inverting the walk
# (one pass, bucket by file name = metric id) mirrors the
# ``_calibration_index_cache`` fix already shipped for calibrations.
# Cleared on save (so the UI sees its own writes) and via
# ``_clear_caches_for_tests``.
_saved_rubric_index_cache: dict[tuple, dict[str, dict]] = {}


def _bool_verdict(v: object) -> bool | None:
    """Coerce a verdict (str/bool/int) to a Python bool, or ``None``."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"pass", "true", "1", "yes", "y"}:
            return True
        if s in {"fail", "false", "0", "no", "n"}:
            return False
    return None


def _kappa_from_annotations(annotations: list[dict]) -> tuple[float | None, int, int, int]:
    """Cohen's kappa on judge vs human verdicts.

    Returns ``(kappa, sample_size, false_positives, false_negatives)``.
    Uses the human verdict as ground truth for FP/FN labelling:
      * FP = judge passes, human fails
      * FN = judge fails, human passes
    """
    pairs: list[tuple[bool, bool]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        jv = _bool_verdict(ann.get("judge_verdict"))
        hv = _bool_verdict(ann.get("human_verdict"))
        if jv is None or hv is None:
            continue
        pairs.append((jv, hv))

    n = len(pairs)
    if n == 0:
        return None, 0, 0, 0

    tp = sum(1 for j, h in pairs if j and h)
    tn = sum(1 for j, h in pairs if (not j) and (not h))
    fp = sum(1 for j, h in pairs if j and (not h))
    fn = sum(1 for j, h in pairs if (not j) and h)

    po = (tp + tn) / n
    p_judge_yes = (tp + fp) / n
    p_human_yes = (tp + fn) / n
    pe = p_judge_yes * p_human_yes + (1 - p_judge_yes) * (1 - p_human_yes)
    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return kappa, n, fp, fn


def _calibration_index() -> dict[str, list[Path]]:
    """Walk dataset roots once and return ``{metric_id: [calibration.json, ...]}``.

    The previous per-metric loop did ``iterdir()`` + ``cal.exists()`` for
    every metric x every dataset dir. With 39 metrics and 491 dataset
    dirs that's ~19k stat calls per request - measured at 76s on prod.
    Inverting the walk drops it to one pass: list ``calibrations/``
    once, bucket by the subdirectory name (which is the metric id).

    Cached by every dataset root's mtime - new calibrations land via
    ``evalyn calibrate`` which writes a new directory under
    ``calibrations/`` (bumping the parent dataset's mtime, but not the
    dataset *root's* mtime). For now we accept that new calibrations on
    an existing dataset will not be visible until a server restart;
    that mirrors how the v2 endpoints handled run additions before this
    cache work and matches the read-mostly workload.
    """
    roots = dataset_roots()
    key = _all_runs_cache_key(roots) if not _caches_disabled() else None
    if key is not None and key in _calibration_index_cache:
        return _calibration_index_cache[key]
    index: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for dataset_dir in list_dataset_dirs(root):
            cal_root = dataset_dir / "calibrations"
            if not cal_root.is_dir():
                continue
            for metric_dir in cal_root.iterdir():
                if not metric_dir.is_dir():
                    continue
                cal = metric_dir / "calibration.json"
                if cal.is_file():
                    index[metric_dir.name].append(cal)
    if key is not None:
        _calibration_index_cache[key] = dict(index)
    return index


def _calibration_paths(metric_id: str) -> list[Path]:
    """Backwards-compat shim: cheap dict lookup against the prebuilt index."""
    return _calibration_index().get(metric_id, [])


def _load_calibration_from_paths(paths: list[Path]) -> tuple[float | None, int, int, int]:
    """Aggregate kappa across the given calibration files."""
    annotations: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("calibration.json read failed at %s: %s", p, exc)
            continue
        anns = data.get("annotations")
        if isinstance(anns, list):
            annotations.extend(a for a in anns if isinstance(a, dict))
    return _kappa_from_annotations(annotations)


def _load_calibration_annotations(paths: list[Path]) -> list[dict]:
    """Read every annotation across ``paths``; lossy invalid lines are skipped."""
    annotations: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("calibration.json read failed at %s: %s", p, exc)
            continue
        anns = data.get("annotations")
        if isinstance(anns, list):
            annotations.extend(a for a in anns if isinstance(a, dict))
    return annotations


def _ann_ts(ann: dict) -> datetime | None:
    """Parse ``timestamp`` / ``created_at`` from an annotation, tolerantly."""
    for key in ("timestamp", "created_at", "ts", "ts_iso"):
        v = ann.get(key)
        if isinstance(v, str):
            dt = parse_iso(v)
            if dt is not None:
                return dt
    return None


def _kappa_window_drift(annotations: list[dict]) -> float:
    """Δkappa over last 7d vs prior 7d.

    Returns ``0.0`` when either window has too few annotations to form a
    kappa. Annotations missing a timestamp are dropped (not silently
    bucketed) so the signal stays trustworthy.
    """
    if not annotations:
        return 0.0
    now = datetime.now(timezone.utc)
    week = now - timedelta(days=7)
    prior_start = now - timedelta(days=14)
    last7: list[dict] = []
    prior7: list[dict] = []
    for ann in annotations:
        dt = _ann_ts(ann)
        if dt is None:
            continue
        if dt >= week:
            last7.append(ann)
        elif dt >= prior_start:
            prior7.append(ann)
    if not last7 or not prior7:
        return 0.0
    k_last, n_last, _, _ = _kappa_from_annotations(last7)
    k_prior, n_prior, _, _ = _kappa_from_annotations(prior7)
    if k_last is None or k_prior is None or n_last < 5 or n_prior < 5:
        return 0.0
    return round(k_last - k_prior, 4)


def _confusion_matrix(
    annotations: list[dict],
) -> tuple[int, int, int, int] | None:
    """Return ``(tp, tn, fp, fn)`` or ``None`` when no usable pairs."""
    pairs: list[tuple[bool, bool]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        jv = _bool_verdict(ann.get("judge_verdict"))
        hv = _bool_verdict(ann.get("human_verdict"))
        if jv is None or hv is None:
            continue
        pairs.append((jv, hv))
    if not pairs:
        return None
    tp = sum(1 for j, h in pairs if j and h)
    tn = sum(1 for j, h in pairs if (not j) and (not h))
    fp = sum(1 for j, h in pairs if j and (not h))
    fn = sum(1 for j, h in pairs if (not j) and h)
    return tp, tn, fp, fn


def _load_calibration(metric_id: str) -> tuple[float | None, int, int, int]:
    """Aggregate kappa across all calibration files for a metric."""
    return _load_calibration_from_paths(_calibration_index().get(metric_id, []))


def _calibration_label(kappa: float | None) -> tuple[str, str]:
    """Return ``(label, kind)`` for the calibration chip."""
    if kappa is None:
        return "no calibration", "info"
    label = f"k={kappa:.2f}"
    if kappa >= 0.8:
        return label, "pass"
    if kappa >= 0.6:
        return label, "warn"
    return label, "fail"


def _metric_kinds_uses(runs: list[dict]) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Walk runs and return ``(kind, uses, name)`` keyed by metric id."""
    kind: dict[str, str] = {}
    uses: dict[str, int] = defaultdict(int)
    name: dict[str, str] = {}
    for run in runs:
        seen_in_run: set[str] = set()
        judge_metric_ids: set[str] = set()
        for jc in run.get("judge_configs") or []:
            if isinstance(jc, dict):
                jid = jc.get("metric_id") or jc.get("id")
                if jid:
                    judge_metric_ids.add(jid)
        for m in run.get("metrics") or []:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            seen_in_run.add(mid)
            if mid not in name:
                name[mid] = m.get("name") or mid
            if mid in judge_metric_ids:
                kind[mid] = "LLM judge"
            elif mid not in kind:
                kind[mid] = "Programmatic"
        for mid in seen_in_run:
            uses[mid] += 1
    return kind, dict(uses), name


def _saved_rubric_path(metric_id: str) -> Path | None:
    """Return the on-disk path for a saved rubric, or ``None`` if no root.

    Persists alongside the calibration files: ``<dataset_root>/<dataset>/
    rubrics/<metric_id>.json``. We pick the first writable dataset that
    has the metric in question. Falls back to the demo fixture root when
    no dataset is on disk yet.
    """
    for root in dataset_roots():
        for ds in list_dataset_dirs(root):
            target = ds / _RUBRIC_FILE_DIR
            try:
                target.mkdir(parents=True, exist_ok=True)
                return target / f"{metric_id}.json"
            except OSError:
                continue
    fallback = Path.cwd() / ".evalyn" / "data" / "rubrics"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / f"{metric_id}.json"
    except OSError:
        return None


def _saved_rubric_index() -> dict[str, dict]:
    """Walk dataset roots once and return ``{metric_id: rubric_dict}``.

    First match wins across ``dataset_roots()`` x ``list_dataset_dirs(root)``
    order, matching the prior ``_load_saved_rubric`` behaviour. Then
    falls back to ``.evalyn/data/rubrics/`` for metric ids not seen in
    any dataset. Cached against the same root-mtime key the calibration
    index uses; ``save_rubric`` clears it so the dashboard sees its own
    writes immediately.
    """
    roots = dataset_roots()
    key = _all_runs_cache_key(roots) if not _caches_disabled() else None
    if key is not None and key in _saved_rubric_index_cache:
        return _saved_rubric_index_cache[key]

    index: dict[str, dict] = {}
    for root in roots:
        for ds in list_dataset_dirs(root):
            rubric_dir = ds / _RUBRIC_FILE_DIR
            if not rubric_dir.is_dir():
                continue
            try:
                with os.scandir(rubric_dir) as it:
                    for entry in it:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if not entry.name.endswith(".json"):
                            continue
                        metric_id = entry.name[:-5]
                        if metric_id in index:
                            continue
                        try:
                            index[metric_id] = json.loads(
                                Path(entry.path).read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError) as exc:
                            logger.warning("rubric load failed at %s: %s", entry.path, exc)
            except OSError:
                continue

    fallback_dir = Path.cwd() / ".evalyn" / "data" / "rubrics"
    if fallback_dir.is_dir():
        try:
            with os.scandir(fallback_dir) as it:
                for entry in it:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if not entry.name.endswith(".json"):
                        continue
                    metric_id = entry.name[:-5]
                    if metric_id in index:
                        continue
                    try:
                        index[metric_id] = json.loads(
                            Path(entry.path).read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning("rubric load failed at %s: %s", entry.path, exc)
        except OSError:
            pass

    if key is not None:
        _saved_rubric_index_cache[key] = index
    return index


def _load_saved_rubric(metric_id: str) -> dict | None:
    """Read any persisted rubric override. O(1) lookup against the cached index."""
    return _saved_rubric_index().get(metric_id)


@router.get("")
async def list_rubrics() -> JSONResponse:
    """Return one row per discovered metric id (augmented).

    Augmented v2 fields: ``kappa``, ``drift_per_week``, ``sample_size``,
    ``weights``, ``dimensions`` (with FP/FN). Existing fields preserved.
    """
    runs = load_all_runs()
    if not runs:
        return JSONResponse([])
    kind, uses, name = _metric_kinds_uses(runs)
    cal_index = _calibration_index()
    rows: list[dict] = []
    for mid in sorted(uses):
        paths = cal_index.get(mid, [])
        annotations = _load_calibration_annotations(paths)
        kappa, sample_size, fp, fn = _kappa_from_annotations(annotations)
        label, label_kind = _calibration_label(kappa)
        drift = _kappa_window_drift(annotations)
        saved = _load_saved_rubric(mid) or {}
        weights = saved.get("weights") if isinstance(saved.get("weights"), dict) else None
        # Default to single-dim with overall FP/FN duplicated.
        dimensions = saved.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            dimensions = [
                {
                    "label": name.get(mid, mid),
                    "weight": 100,
                    "fp": int(fp),
                    "fn": int(fn),
                }
            ]
        rows.append(
            {
                "id": mid,
                "name": name.get(mid, mid),
                "kind": kind.get(mid, "Programmatic"),
                "dimensions": len(dimensions) if isinstance(dimensions, list) else 1,
                "calibration_label": label,
                "calibration_kind": label_kind,
                "uses": uses[mid],
                # Augmented fields:
                "kappa": round(kappa, 4) if kappa is not None else None,
                "drift_per_week": drift,
                "sample_size": sample_size,
                "weights": weights,
                "dimensions_detail": dimensions,
            }
        )
    return JSONResponse(rows)


@router.get("/trust")
async def get_trust_scoreboard() -> JSONResponse:
    """Return the trust scoreboard."""
    runs = load_all_runs()
    if not runs:
        return JSONResponse({
            "metrics": [],
            "thresholds": {"annotation": 0.7, "ship": 0.8},
        })
    kind, uses, name = _metric_kinds_uses(runs)
    cal_index = _calibration_index()
    metrics_rows: list[dict] = []
    for mid in sorted(uses):
        is_llm = kind.get(mid) == "LLM judge"
        paths = cal_index.get(mid, [])
        annotations = _load_calibration_annotations(paths)
        kappa, sample_size, _fp, _fn = _kappa_from_annotations(annotations)
        drift = _kappa_window_drift(annotations)
        # Score = kappa (clipped to 0..1) for LLM; deterministic 1.0 for prog.
        if is_llm:
            score = max(0.0, min(1.0, kappa)) if kappa is not None else 0.0
            kappa_out = round(kappa, 4) if kappa is not None else None
            sample_out: int | str = sample_size
        else:
            score = 1.0
            kappa_out = None
            sample_out = "-"
        needs_work = bool(score < 0.7 or drift < -0.05)
        metrics_rows.append(
            {
                "name": name.get(mid, mid),
                "metric_type": "llm" if is_llm else "programmatic",
                "score": round(score, 4),
                "kappa": kappa_out,
                "drift_per_week": drift,
                "sample_size": sample_out,
                "needs_work": needs_work,
            }
        )
    return JSONResponse(
        {
            "metrics": metrics_rows,
            "thresholds": {"annotation": 0.7, "ship": 0.8},
        }
    )


@router.get("/{rubric_id}/calibration")
async def get_rubric_calibration(rubric_id: str) -> JSONResponse:
    """Return ``RubricDetail`` for ``rubric_id`` or 404 (augmented v2)."""
    runs = load_all_runs()
    kind, _, name = _metric_kinds_uses(runs) if runs else ({}, {}, {})
    if rubric_id not in name:
        raise HTTPException(404, f"rubric {rubric_id} not found")

    kappa, sample_size, fp, fn = _load_calibration(rubric_id)
    label, label_kind = _calibration_label(kappa)

    cal_index = _calibration_index()
    annotations = _load_calibration_annotations(cal_index.get(rubric_id, []))
    drift = _kappa_window_drift(annotations)
    cm = _confusion_matrix(annotations)
    saved = _load_saved_rubric(rubric_id) or {}
    weights = saved.get("weights") if isinstance(saved.get("weights"), dict) else None
    dimensions_raw = saved.get("dimensions")
    if not isinstance(dimensions_raw, list) or not dimensions_raw:
        dimensions_aug = [
            {
                "label": name.get(rubric_id, rubric_id),
                "weight": 100,
                "fp": int(fp),
                "fn": int(fn),
            }
        ]
    else:
        dimensions_aug = []
        for d in dimensions_raw:
            if not isinstance(d, dict):
                continue
            dimensions_aug.append(
                {
                    "label": str(d.get("label", "")),
                    "weight": float(d.get("weight", 0)),
                    "fp": int(d.get("fp", fp)),
                    "fn": int(d.get("fn", fn)),
                }
            )

    confusion_matrix_payload = (
        {"tp": cm[0], "tn": cm[1], "fp": cm[2], "fn": cm[3]}
        if cm is not None
        else None
    )

    return JSONResponse(
        {
            "id": rubric_id,
            "name": saved.get("name") or name.get(rubric_id, rubric_id),
            "calibration": {
                "kappa": round(kappa, 4) if kappa is not None else None,
                "label": label,
                "kind": label_kind,
                "false_positives_pct": round(100.0 * fp / sample_size, 1) if sample_size else None,
                "false_negatives_pct": round(100.0 * fn / sample_size, 1) if sample_size else None,
                "sample_size": sample_size,
            },
            "dimensions": [
                {
                    "label": name.get(rubric_id, rubric_id),
                    "weight_pct": 100,
                    "example": "",
                    "kind": "judge" if kind.get(rubric_id) == "LLM judge" else "prog",
                }
            ],
            # Augmented fields:
            "kappa": round(kappa, 4) if kappa is not None else None,
            "drift_per_week": drift,
            "sample_size": sample_size,
            "weights": weights,
            "dimensions_detail": dimensions_aug,
            "confusion_matrix": confusion_matrix_payload,
        }
    )


def _validate_weights(weights: dict[str, float]) -> None:
    """Raise 422 if weights don't sum to 100 within +/- 1 percent."""
    if not isinstance(weights, dict) or not weights:
        raise HTTPException(422, "weights must be a non-empty object")
    total = 0.0
    for k, v in weights.items():
        if not isinstance(k, str) or not isinstance(v, (int, float)):
            raise HTTPException(422, "weights values must be numbers")
        total += float(v)
    if abs(total - 100.0) > 1.0:
        raise HTTPException(422, f"weights must sum to 100 (+/-1), got {total}")


@router.post("/{rubric_id}")
async def save_rubric(rubric_id: str, request: Request) -> JSONResponse:
    """Persist a rubric override (name, weights, dimensions)."""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a json object")

    name = body.get("name")
    if name is not None and not isinstance(name, str):
        raise HTTPException(422, "name must be a string")

    weights = body.get("weights")
    if weights is not None:
        _validate_weights(weights)

    dimensions = body.get("dimensions")
    if dimensions is not None:
        if not isinstance(dimensions, list):
            raise HTTPException(422, "dimensions must be a list")
        for d in dimensions:
            if not isinstance(d, dict):
                raise HTTPException(422, "each dimension must be an object")
            if "label" not in d or "weight" not in d:
                raise HTTPException(422, "dimension requires label and weight")

    payload = {
        "id": rubric_id,
        "name": name,
        "weights": weights,
        "dimensions": dimensions,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    p = _saved_rubric_path(rubric_id)
    if p is None:
        raise HTTPException(503, "no writable rubric path")
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        logger.warning("rubric save failed at %s: %s", p, exc)
        raise HTTPException(503, f"rubric save failed: {exc}") from exc

    # The cache is keyed on each dataset root's mtime, which does NOT
    # bump when a file is written deep inside (rubrics/<id>.json under a
    # dataset). Without an explicit invalidation, the user clicks Save
    # and ``GET /api/v2/rubrics`` still shows pre-save state until the
    # server restarts. Clear the index so the next read rebuilds.
    _saved_rubric_index_cache.clear()

    # Audit log: rubric persistence is a configuration change that
    # affects every future eval run (judge weights, dimensions).
    # Operators tracking "why did the trust score shift today?"
    # need to know when rubrics were edited. Log which fields the
    # user actually touched (None means "kept previous value")
    # so the trail captures intent precisely.
    fields_changed = [
        f for f, v in (
            ("name", name),
            ("weights", weights),
            ("dimensions", dimensions),
        ) if v is not None
    ]
    logger.info(
        "rubric saved: rubric_id=%s fields=%s",
        rubric_id,
        ",".join(fields_changed) or "<none>",
    )

    return JSONResponse({"ok": True, "saved_at": payload["saved_at"]})
