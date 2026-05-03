"""``/api/v2/rubrics`` router.

Endpoints:
- ``GET /``                 -> ``RubricList``
- ``GET /{id}/calibration`` -> ``RubricDetail``

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
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ._shared import datasets_root, load_all_runs

logger = logging.getLogger(__name__)

router = APIRouter()


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


def _calibration_paths(metric_id: str) -> list[Path]:
    """All ``calibration.json`` files for ``metric_id`` across datasets."""
    root = datasets_root()
    if not root.exists():
        return []
    out: list[Path] = []
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        cal = dataset_dir / "calibrations" / metric_id / "calibration.json"
        if cal.exists():
            out.append(cal)
    return out


def _load_calibration(metric_id: str) -> tuple[float | None, int, int, int]:
    """Aggregate kappa across all calibration files for a metric."""
    annotations: list[dict] = []
    for p in _calibration_paths(metric_id):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("calibration.json read failed at %s: %s", p, exc)
            continue
        anns = data.get("annotations")
        if isinstance(anns, list):
            annotations.extend(a for a in anns if isinstance(a, dict))
    return _kappa_from_annotations(annotations)


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


@router.get("")
async def list_rubrics() -> JSONResponse:
    """Return one row per discovered metric id."""
    runs = load_all_runs(datasets_root())
    if not runs:
        return JSONResponse([])
    kind, uses, name = _metric_kinds_uses(runs)
    rows: list[dict] = []
    for mid in sorted(uses):
        kappa, _, _, _ = _load_calibration(mid)
        label, label_kind = _calibration_label(kappa)
        rows.append(
            {
                "id": mid,
                "name": name.get(mid, mid),
                "kind": kind.get(mid, "Programmatic"),
                "dimensions": 1,
                "calibration_label": label,
                "calibration_kind": label_kind,
                "uses": uses[mid],
            }
        )
    return JSONResponse(rows)


@router.get("/{rubric_id}/calibration")
async def get_rubric_calibration(rubric_id: str) -> JSONResponse:
    """Return ``RubricDetail`` for ``rubric_id`` or 404."""
    runs = load_all_runs(datasets_root())
    kind, _, name = _metric_kinds_uses(runs) if runs else ({}, {}, {})
    # Always 404 unknown rubrics. Without the runs guard a fresh workspace
    # would echo the URL slug back as a fabricated rubric (silent-failure
    # hunter punch list, item 5).
    if rubric_id not in name:
        raise HTTPException(404, f"rubric {rubric_id} not found")

    kappa, sample_size, fp, fn = _load_calibration(rubric_id)
    label, label_kind = _calibration_label(kappa)

    return JSONResponse(
        {
            "id": rubric_id,
            "name": name.get(rubric_id, rubric_id),
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
        }
    )
