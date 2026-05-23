"""Tests for the baseline + drift commands.

Covers:
- Persistence roundtrips (BaselinePointer save / load / list / clear).
- compute_drift over identical, regressed, improved, and missing-metric runs.
- CLI dispatch for ``evalyn baseline {set,show,list,clear}`` and ``evalyn drift``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from evalyn_sdk.analysis.baseline import (
    BASELINES_FILENAME,
    DEFAULT_BASELINE_NAME,
    BaselinePointer,
    DriftReport,
    clear_baseline,
    compute_drift,
    list_baselines,
    load_baseline,
    now_iso,
    save_baseline,
)
from evalyn_sdk.analysis.core import RunAnalysis, analyze_run
from evalyn_sdk.cli.commands import baseline as baseline_cmd
from evalyn_sdk.cli.commands import drift as drift_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_dict(
    run_id: str,
    dataset_name: str = "demo-dataset",
    helpfulness: tuple[float, bool] = (0.9, True),
    safety: tuple[float, bool] = (0.95, True),
    n_items: int = 4,
    extra_metric: tuple[str, float, bool] | None = None,
) -> dict[str, Any]:
    """Build a synthetic eval-run dict for analyze_run."""
    metric_results: list[dict[str, Any]] = []
    for i in range(n_items):
        item_id = f"item-{i:03d}"
        metric_results.append(
            {
                "metric_id": "helpfulness",
                "item_id": item_id,
                "call_id": f"call-{item_id}",
                "score": helpfulness[0],
                "passed": helpfulness[1],
                "details": {},
            }
        )
        metric_results.append(
            {
                "metric_id": "safety",
                "item_id": item_id,
                "call_id": f"call-{item_id}",
                "score": safety[0],
                "passed": safety[1],
                "details": {},
            }
        )
        if extra_metric is not None:
            mid, mscore, mpassed = extra_metric
            metric_results.append(
                {
                    "metric_id": mid,
                    "item_id": item_id,
                    "call_id": f"call-{item_id}",
                    "score": mscore,
                    "passed": mpassed,
                    "details": {},
                }
            )

    metrics = [
        {"id": "helpfulness", "name": "Helpfulness", "type": "subjective", "description": ""},
        {"id": "safety", "name": "Safety", "type": "subjective", "description": ""},
    ]
    if extra_metric is not None:
        metrics.append(
            {"id": extra_metric[0], "name": extra_metric[0], "type": "objective", "description": ""}
        )

    return {
        "id": run_id,
        "dataset_name": dataset_name,
        "created_at": "2026-05-22T10:00:00+00:00",
        "metric_results": metric_results,
        "metrics": metrics,
        "judge_configs": [],
        "summary": {},
        "usage_summary": {},
    }


def _write_run_file(dataset_dir: Path, run_id: str, run_dict: dict[str, Any]) -> Path:
    """Write a run dict into dataset_dir/eval_runs/<run_id>/results.json."""
    runs_dir = dataset_dir / "eval_runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_dict, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestBaselinePersistence:
    """Roundtrips and catalog management for BaselinePointer."""

    def test_save_then_load_roundtrip(self, temp_dir: Path) -> None:
        pointer = BaselinePointer(
            name="default",
            run_id="run-abc-1234",
            run_path=str(temp_dir / "results.json"),
            dataset_name="demo",
            pinned_at=now_iso(),
        )
        catalog_path = save_baseline(pointer, state_dir=temp_dir)
        assert catalog_path == temp_dir / BASELINES_FILENAME
        assert catalog_path.exists()

        loaded = load_baseline("default", state_dir=temp_dir)
        assert loaded is not None
        assert loaded.run_id == pointer.run_id
        assert loaded.run_path == pointer.run_path
        assert loaded.dataset_name == "demo"

    def test_list_multiple_baselines(self, temp_dir: Path) -> None:
        save_baseline(
            BaselinePointer("default", "run-1", "", "ds", "2026-01-01T00:00:00+00:00"),
            state_dir=temp_dir,
        )
        save_baseline(
            BaselinePointer("nightly", "run-2", "", "ds", "2026-02-01T00:00:00+00:00"),
            state_dir=temp_dir,
        )
        save_baseline(
            BaselinePointer("prod", "run-3", "", "ds", "2026-03-01T00:00:00+00:00"),
            state_dir=temp_dir,
        )
        ptrs = list_baselines(state_dir=temp_dir)
        assert {p.name for p in ptrs} == {"default", "nightly", "prod"}
        # Newest pinned_at first.
        assert ptrs[0].name == "prod"
        assert ptrs[-1].name == "default"

    def test_clear_named_baseline(self, temp_dir: Path) -> None:
        save_baseline(
            BaselinePointer("default", "r1", "", "ds", now_iso()), state_dir=temp_dir
        )
        save_baseline(
            BaselinePointer("nightly", "r2", "", "ds", now_iso()), state_dir=temp_dir
        )
        assert clear_baseline("nightly", state_dir=temp_dir) is True
        assert load_baseline("nightly", state_dir=temp_dir) is None
        # Other entries untouched.
        assert load_baseline("default", state_dir=temp_dir) is not None

    def test_clear_default_when_only_default(self, temp_dir: Path) -> None:
        save_baseline(
            BaselinePointer("default", "r1", "", "ds", now_iso()), state_dir=temp_dir
        )
        assert clear_baseline(state_dir=temp_dir) is True  # name=None => clear all
        assert list_baselines(state_dir=temp_dir) == []

    def test_clear_missing_returns_false(self, temp_dir: Path) -> None:
        assert clear_baseline("nope", state_dir=temp_dir) is False

    def test_save_overwrites_same_name(self, temp_dir: Path) -> None:
        save_baseline(
            BaselinePointer("default", "old-run", "", "ds", "2026-01-01T00:00:00+00:00"),
            state_dir=temp_dir,
        )
        save_baseline(
            BaselinePointer("default", "new-run", "", "ds", "2026-02-01T00:00:00+00:00"),
            state_dir=temp_dir,
        )
        loaded = load_baseline("default", state_dir=temp_dir)
        assert loaded is not None
        assert loaded.run_id == "new-run"

    def test_load_missing_returns_none(self, temp_dir: Path) -> None:
        assert load_baseline("missing", state_dir=temp_dir) is None


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


class TestComputeDrift:
    """Pure function tests for compute_drift over RunAnalysis pairs."""

    def _analysis(self, **kw: Any) -> RunAnalysis:
        return analyze_run(_make_run_dict(**kw))

    def test_identical_runs_have_zero_delta(self) -> None:
        a = self._analysis(run_id="base")
        b = self._analysis(run_id="cur")
        report = compute_drift(a, b, threshold=0.05)
        assert isinstance(report, DriftReport)
        assert report.flagged_regressions == []
        for entry in report.entries:
            assert entry.status == "stable"
            assert entry.delta_abs == 0.0

    def test_regression_flagged_when_over_threshold(self) -> None:
        baseline = self._analysis(run_id="base", helpfulness=(0.90, True))
        current = self._analysis(run_id="cur", helpfulness=(0.50, False))
        report = compute_drift(baseline, current, threshold=0.05)
        help_entry = next(e for e in report.entries if e.metric_id == "helpfulness")
        assert help_entry.status == "regression"
        assert help_entry.flagged is True
        assert help_entry.delta_abs == pytest.approx(-0.40)
        assert help_entry.delta_pct == pytest.approx(-0.40 / 0.90)

    def test_regression_below_threshold_not_flagged(self) -> None:
        baseline = self._analysis(run_id="base", helpfulness=(0.90, True))
        current = self._analysis(run_id="cur", helpfulness=(0.89, True))
        # ~1.1% drop, threshold 5%.
        report = compute_drift(baseline, current, threshold=0.05)
        help_entry = next(e for e in report.entries if e.metric_id == "helpfulness")
        assert help_entry.status == "regression"
        assert help_entry.flagged is False

    def test_improvement_recorded(self) -> None:
        baseline = self._analysis(run_id="base", helpfulness=(0.50, False))
        current = self._analysis(run_id="cur", helpfulness=(0.90, True))
        report = compute_drift(baseline, current, threshold=0.05)
        help_entry = next(e for e in report.entries if e.metric_id == "helpfulness")
        assert help_entry.status == "improvement"
        assert help_entry.delta_abs == pytest.approx(0.40)
        assert report.flagged_regressions == []
        assert len(report.improvements) == 1

    def test_missing_metric_handled_gracefully(self) -> None:
        baseline = self._analysis(run_id="base")
        current = analyze_run(
            _make_run_dict(
                run_id="cur",
                extra_metric=("latency_ms", 250.0, None),
            )
        )
        report = compute_drift(baseline, current, threshold=0.05)
        lat_entry = next(e for e in report.entries if e.metric_id == "latency_ms")
        assert lat_entry.status == "missing"
        assert lat_entry.flagged is False
        assert lat_entry.baseline_mean is None
        assert lat_entry.current_mean == pytest.approx(250.0)

    def test_zero_baseline_avoids_divide_by_zero(self) -> None:
        baseline = self._analysis(run_id="base", helpfulness=(0.0, False))
        current = self._analysis(run_id="cur", helpfulness=(0.5, True))
        report = compute_drift(baseline, current, threshold=0.05)
        help_entry = next(e for e in report.entries if e.metric_id == "helpfulness")
        assert help_entry.delta_pct is None
        assert help_entry.delta_abs == pytest.approx(0.5)
        # delta_abs (0.5) >= threshold (0.05), so flagged True with status improvement.
        assert help_entry.status == "improvement"
        assert help_entry.flagged is True


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def _baseline_parser() -> argparse.ArgumentParser:
    """Build a parser that exposes only the baseline subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    baseline_cmd.register_commands(sub)
    return parser


def _drift_parser() -> argparse.ArgumentParser:
    """Build a parser that exposes only the drift subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    drift_cmd.register_commands(sub)
    return parser


class TestBaselineCLIDispatch:
    """Drive the baseline subcommands through the registered argparse parser."""

    def test_set_pins_baseline(self, temp_dir: Path, capsys: pytest.CaptureFixture) -> None:
        dataset_dir = temp_dir / "ds"
        dataset_dir.mkdir(parents=True)
        run_path = _write_run_file(dataset_dir, "run-abc", _make_run_dict("run-abc"))

        parser = _baseline_parser()
        args = parser.parse_args([
            "baseline", "set",
            "--run", str(run_path),
            "--name", "nightly",
            "--state-dir", str(temp_dir / "state"),
        ])
        args.func(args)

        out = capsys.readouterr().out
        assert "Pinned baseline 'nightly'" in out
        loaded = load_baseline("nightly", state_dir=temp_dir / "state")
        assert loaded is not None
        assert loaded.run_id == "run-abc"
        assert loaded.run_path == str(run_path)

    def test_list_baseline_outputs_rows(
        self, temp_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        state = temp_dir / "state"
        save_baseline(BaselinePointer("default", "r1", "", "ds", now_iso()), state_dir=state)
        save_baseline(BaselinePointer("prod", "r2", "", "ds", now_iso()), state_dir=state)

        parser = _baseline_parser()
        args = parser.parse_args(["baseline", "list", "--state-dir", str(state)])
        args.func(args)
        out = capsys.readouterr().out
        assert "default" in out
        assert "prod" in out

    def test_show_missing_exits_nonzero(self, temp_dir: Path) -> None:
        parser = _baseline_parser()
        args = parser.parse_args([
            "baseline", "show",
            "--name", "ghost",
            "--state-dir", str(temp_dir),
        ])
        with pytest.raises(SystemExit) as ei:
            args.func(args)
        assert ei.value.code == 1

    def test_clear_named(self, temp_dir: Path, capsys: pytest.CaptureFixture) -> None:
        state = temp_dir / "state"
        save_baseline(BaselinePointer("nightly", "r1", "", "ds", now_iso()), state_dir=state)
        parser = _baseline_parser()
        args = parser.parse_args([
            "baseline", "clear",
            "--name", "nightly",
            "--state-dir", str(state),
        ])
        args.func(args)
        out = capsys.readouterr().out
        assert "Cleared baseline 'nightly'" in out
        assert load_baseline("nightly", state_dir=state) is None

    def test_clear_all(self, temp_dir: Path, capsys: pytest.CaptureFixture) -> None:
        state = temp_dir / "state"
        save_baseline(BaselinePointer("a", "r1", "", "ds", now_iso()), state_dir=state)
        save_baseline(BaselinePointer("b", "r2", "", "ds", now_iso()), state_dir=state)
        parser = _baseline_parser()
        args = parser.parse_args(["baseline", "clear", "--all", "--state-dir", str(state)])
        args.func(args)
        out = capsys.readouterr().out
        assert "Cleared all baselines" in out
        assert list_baselines(state_dir=state) == []


class TestDriftCLIDispatch:
    """Drive ``evalyn drift`` end-to-end via argparse."""

    def _pin_baseline(self, state: Path, run_path: Path, run_id: str, dataset: str) -> None:
        save_baseline(
            BaselinePointer(
                name=DEFAULT_BASELINE_NAME,
                run_id=run_id,
                run_path=str(run_path),
                dataset_name=dataset,
                pinned_at=now_iso(),
            ),
            state_dir=state,
        )

    def test_drift_identical_runs_table(
        self, temp_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        dataset_dir = temp_dir / "ds"
        dataset_dir.mkdir(parents=True)
        baseline_path = _write_run_file(dataset_dir, "run-base", _make_run_dict("run-base"))
        current_path = _write_run_file(dataset_dir, "run-cur", _make_run_dict("run-cur"))

        state = temp_dir / "state"
        self._pin_baseline(state, baseline_path, "run-base", "ds")

        parser = _drift_parser()
        args = parser.parse_args([
            "drift",
            "--run", str(current_path),
            "--threshold", "0.05",
            "--state-dir", str(state),
        ])
        # Identical runs => no flagged regressions => no SystemExit.
        args.func(args)
        out = capsys.readouterr().out
        assert "DRIFT REPORT" in out
        assert "stable" in out

    def test_drift_regression_exits_nonzero(self, temp_dir: Path) -> None:
        dataset_dir = temp_dir / "ds"
        dataset_dir.mkdir(parents=True)
        baseline_path = _write_run_file(
            dataset_dir, "run-base", _make_run_dict("run-base", helpfulness=(0.90, True))
        )
        current_path = _write_run_file(
            dataset_dir, "run-cur", _make_run_dict("run-cur", helpfulness=(0.40, False))
        )

        state = temp_dir / "state"
        self._pin_baseline(state, baseline_path, "run-base", "ds")

        parser = _drift_parser()
        args = parser.parse_args([
            "drift",
            "--run", str(current_path),
            "--threshold", "0.05",
            "--state-dir", str(state),
        ])
        with pytest.raises(SystemExit) as ei:
            args.func(args)
        assert ei.value.code == 1

    def test_drift_json_format(
        self, temp_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        dataset_dir = temp_dir / "ds"
        dataset_dir.mkdir(parents=True)
        baseline_path = _write_run_file(
            dataset_dir, "run-base", _make_run_dict("run-base", helpfulness=(0.9, True))
        )
        current_path = _write_run_file(
            dataset_dir, "run-cur", _make_run_dict("run-cur", helpfulness=(0.95, True))
        )

        state = temp_dir / "state"
        self._pin_baseline(state, baseline_path, "run-base", "ds")

        parser = _drift_parser()
        args = parser.parse_args([
            "drift",
            "--run", str(current_path),
            "--format", "json",
            "--state-dir", str(state),
        ])
        args.func(args)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["baseline_name"] == DEFAULT_BASELINE_NAME
        assert payload["baseline_run_id"] == "run-base"
        assert payload["current_run_id"] == "run-cur"
        assert "summary" in payload
        # helpfulness should be an improvement.
        improvements = [e for e in payload["entries"] if e["status"] == "improvement"]
        assert any(e["metric_id"] == "helpfulness" for e in improvements)

    def test_drift_missing_baseline_errors(self, temp_dir: Path) -> None:
        parser = _drift_parser()
        args = parser.parse_args([
            "drift",
            "--run", str(temp_dir / "nope.json"),
            "--state-dir", str(temp_dir / "empty-state"),
        ])
        with pytest.raises(SystemExit) as ei:
            args.func(args)
        assert ei.value.code == 1
