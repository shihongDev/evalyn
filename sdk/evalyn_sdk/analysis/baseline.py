"""Baseline pinning and drift computation.

Stores a small JSON catalog of named baseline runs under .evalyn/baselines.json
and computes per-metric drift between a baseline run and a current run. A
"baseline" is a pointer (run_id + path) to a previously executed eval run that
a team treats as the canonical reference; subsequent runs are compared against
it to detect regressions or improvements over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import now_utc
from .core import RunAnalysis

# Canonical project-state directory (matches doctor.py / report.py conventions).
DEFAULT_STATE_DIR = ".evalyn"
BASELINES_FILENAME = "baselines.json"
DEFAULT_BASELINE_NAME = "default"


@dataclass
class BaselinePointer:
    """Pointer to a pinned eval run used as a comparison reference."""

    name: str
    run_id: str
    run_path: str
    dataset_name: str
    pinned_at: str  # ISO8601 timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "run_path": self.run_path,
            "dataset_name": self.dataset_name,
            "pinned_at": self.pinned_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselinePointer:
        return cls(
            name=data["name"],
            run_id=data["run_id"],
            run_path=data["run_path"],
            dataset_name=data.get("dataset_name", ""),
            pinned_at=data.get("pinned_at", ""),
        )


@dataclass
class DriftEntry:
    """Per-metric drift between baseline and current run.

    delta_abs is current_mean - baseline_mean. delta_pct is the relative
    change expressed as a fraction (e.g. 0.1 = +10%); when the baseline
    mean is zero, delta_pct is None to avoid divide-by-zero noise.
    status is one of: "regression", "improvement", "stable", "missing".
    """

    metric_id: str
    baseline_mean: float | None
    current_mean: float | None
    delta_abs: float | None
    delta_pct: float | None
    status: str
    flagged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "delta_abs": self.delta_abs,
            "delta_pct": self.delta_pct,
            "status": self.status,
            "flagged": self.flagged,
        }


@dataclass
class DriftReport:
    """Aggregate drift report across all metrics in either run."""

    baseline_run_id: str
    current_run_id: str
    threshold: float
    entries: list[DriftEntry] = field(default_factory=list)

    @property
    def regressions(self) -> list[DriftEntry]:
        return [e for e in self.entries if e.status == "regression"]

    @property
    def improvements(self) -> list[DriftEntry]:
        return [e for e in self.entries if e.status == "improvement"]

    @property
    def flagged_regressions(self) -> list[DriftEntry]:
        return [e for e in self.entries if e.status == "regression" and e.flagged]

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "threshold": self.threshold,
            "entries": [e.as_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _baselines_path(state_dir: Path | str | None = None) -> Path:
    """Resolve the baselines JSON file path.

    Defaults to ``<cwd>/.evalyn/baselines.json`` so each project keeps its
    own catalog. Pass an explicit directory for tests or custom layouts.
    """
    root = Path(state_dir) if state_dir is not None else Path.cwd() / DEFAULT_STATE_DIR
    return root / BASELINES_FILENAME


def _read_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    # OSError is intentionally NOT swallowed here - a permissions or disk
    # issue should surface, not silently look like an empty catalog.
    # JSONDecodeError is treated as "corrupt, start fresh" because that is
    # recoverable; the next write will overwrite.
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_catalog(path: Path, catalog: dict[str, dict[str, Any]]) -> None:
    """Atomically write the baseline catalog.

    Uses temp-file + os.replace() so a concurrent write or mid-write crash
    cannot leave the catalog truncated or partially-serialized. os.replace
    is atomic on POSIX and on the same volume on Windows.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file if anything went wrong before the replace.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_baseline(
    pointer: BaselinePointer,
    *,
    state_dir: Path | str | None = None,
) -> Path:
    """Persist (or overwrite) a baseline pointer keyed by ``pointer.name``.

    Returns the path of the JSON catalog that was written.
    """
    path = _baselines_path(state_dir)
    catalog = _read_catalog(path)
    catalog[pointer.name] = pointer.as_dict()
    _write_catalog(path, catalog)
    return path


def load_baseline(
    name: str = DEFAULT_BASELINE_NAME,
    *,
    state_dir: Path | str | None = None,
) -> BaselinePointer | None:
    """Load a single baseline pointer by name, or None if missing."""
    entry = _read_catalog(_baselines_path(state_dir)).get(name)
    return BaselinePointer.from_dict(entry) if entry else None


def list_baselines(
    *,
    state_dir: Path | str | None = None,
) -> list[BaselinePointer]:
    """List all baseline pointers, sorted by pinned_at descending then name."""
    path = _baselines_path(state_dir)
    catalog = _read_catalog(path)
    pointers = [BaselinePointer.from_dict(v) for v in catalog.values()]
    pointers.sort(key=lambda p: (p.pinned_at, p.name), reverse=True)
    return pointers


def clear_baseline(
    name: str | None = None,
    *,
    state_dir: Path | str | None = None,
) -> bool:
    """Remove a baseline pointer.

    When ``name`` is None, clear every pointer. Returns True if any pointer
    was removed; False if nothing matched (or the catalog did not exist).
    """
    path = _baselines_path(state_dir)
    catalog = _read_catalog(path)
    if name is None:
        if not catalog:
            return False
        _write_catalog(path, {})
        return True
    if name not in catalog:
        return False
    del catalog[name]
    _write_catalog(path, catalog)
    return True


def now_iso() -> str:
    """Return current UTC time in ISO8601 format (used when pinning)."""
    return now_utc().isoformat()


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def _classify(delta_abs: float, delta_pct: float | None, threshold: float) -> tuple[str, bool]:
    """Classify a delta as regression/improvement/stable + flag against threshold.

    A positive delta is treated as an improvement (higher score = better).
    Threshold is compared against the absolute relative change when
    available; otherwise it falls back to absolute delta.
    """
    if delta_abs == 0:
        return "stable", False
    magnitude = abs(delta_pct) if delta_pct is not None else abs(delta_abs)
    flagged = magnitude >= threshold
    if delta_abs > 0:
        return "improvement", flagged
    return "regression", flagged


def compute_drift(
    baseline: RunAnalysis,
    current: RunAnalysis,
    threshold: float = 0.0,
) -> DriftReport:
    """Compare two RunAnalysis objects metric-by-metric.

    Args:
        baseline: The pinned reference RunAnalysis.
        current: The new RunAnalysis to evaluate.
        threshold: Relative change threshold (e.g. 0.05 = flag deltas >= 5%).
            Set to 0.0 to flag every non-zero change.

    Returns:
        DriftReport whose entries cover every metric present in either run.
        Metrics absent from one side get status "missing" and are not flagged.
    """
    metrics = set(baseline.metric_stats.keys()) | set(current.metric_stats.keys())
    entries: list[DriftEntry] = []
    for metric_id in sorted(metrics):
        b_stats = baseline.metric_stats.get(metric_id)
        c_stats = current.metric_stats.get(metric_id)

        if b_stats is None or c_stats is None:
            entries.append(
                DriftEntry(
                    metric_id=metric_id,
                    baseline_mean=b_stats.avg_score if b_stats else None,
                    current_mean=c_stats.avg_score if c_stats else None,
                    delta_abs=None,
                    delta_pct=None,
                    status="missing",
                    flagged=False,
                )
            )
            continue

        b_mean = b_stats.avg_score
        c_mean = c_stats.avg_score
        delta_abs = c_mean - b_mean
        delta_pct: float | None = delta_abs / b_mean if b_mean != 0 else None

        status, flagged = _classify(delta_abs, delta_pct, threshold)
        entries.append(
            DriftEntry(
                metric_id=metric_id,
                baseline_mean=b_mean,
                current_mean=c_mean,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                status=status,
                flagged=flagged,
            )
        )

    return DriftReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        threshold=threshold,
        entries=entries,
    )


__all__ = [
    "BaselinePointer",
    "DriftEntry",
    "DriftReport",
    "DEFAULT_BASELINE_NAME",
    "DEFAULT_STATE_DIR",
    "BASELINES_FILENAME",
    "save_baseline",
    "load_baseline",
    "list_baselines",
    "clear_baseline",
    "compute_drift",
    "now_iso",
]
