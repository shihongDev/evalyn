"""Pipeline orchestration for one-click evaluation.

This module provides:
- PipelineStep: Base class for individual pipeline steps
- PipelineOrchestrator: Coordinates execution, state management, and resume
- Step implementations for the 7-step evaluation pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    """Result from a pipeline step."""

    status: str  # "success", "skipped", "failed", "interrupted"
    output: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineState:
    """Persistent state for pipeline execution."""

    started_at: str
    config: Dict[str, Any]
    steps: Dict[str, Dict[str, Any]]
    output_dir: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> Optional["PipelineState"]:
        """Load state from file."""
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                started_at=data.get("started_at", ""),
                config=data.get("config", {}),
                steps=data.get("steps", {}),
                output_dir=data.get("output_dir", ""),
                updated_at=data.get("updated_at"),
                completed_at=data.get("completed_at"),
            )
        except Exception:
            return None

    def save(self, path: Path) -> None:
        """Save state atomically."""
        self.updated_at = datetime.now().isoformat()
        data = {
            "started_at": self.started_at,
            "config": self.config,
            "steps": self.steps,
            "output_dir": self.output_dir,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=".state_",
                suffix=".tmp",
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        except Exception:
            pass  # Non-fatal

    def is_step_done(self, step_name: str) -> bool:
        """Check if a step completed successfully."""
        step_info = self.steps.get(step_name, {})
        return step_info.get("status") == "success"

    def mark_step(self, step_name: str, result: StepResult) -> None:
        """Record step result."""
        self.steps[step_name] = {
            "status": result.status,
            "output": result.output,
            **result.details,
        }
        if result.error:
            self.steps[step_name]["error"] = result.error


class PipelineStep(ABC):
    """Base class for pipeline steps."""

    name: str  # Step folder name (e.g. "dataset")
    display_name: str  # Human-readable name
    step_number: int  # 1-7

    def __init__(self, args: argparse.Namespace, config: Dict[str, Any]):
        self.args = args
        self.config = config

    @property
    def header(self) -> str:
        """Step header for display."""
        return f"[{self.step_number}/7] {self.display_name}"

    def should_skip(self) -> Optional[str]:
        """Return skip reason if step should be skipped, None otherwise."""
        return None

    @abstractmethod
    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> tuple[StepResult, Dict[str, Any]]:
        """Execute the step.

        Args:
            output_dir: Base output directory
            context: Shared context from previous steps (dataset_path, metrics_path, etc.)

        Returns:
            Tuple of (result, updated_context)
        """
        pass

    def dry_run_message(self, output_dir: Path) -> str:
        """Message to show in dry-run mode."""
        return f"  -> Would execute {self.display_name}"


class PipelineOrchestrator:
    """Coordinates pipeline execution with state management."""

    def __init__(
        self,
        steps: List[PipelineStep],
        output_dir: Path,
        args: argparse.Namespace,
    ):
        self.steps = steps
        self.output_dir = output_dir
        self.args = args
        self.state_path = output_dir / "pipeline_state.json"
        self.state: Optional[PipelineState] = None
        self.context: Dict[str, Any] = {}

    def run(self) -> None:
        """Run the pipeline."""
        self._init_state()
        self._print_header()

        try:
            for step in self.steps:
                self._run_step(step)

            self._finalize()

        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)

    def _init_state(self) -> None:
        """Initialize or load state."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resumed = False
        if self.state_path.exists() and getattr(self.args, "resume", False):
            self.state = PipelineState.load(self.state_path)
            if self.state:
                print("  Resuming from previous run...")
                print(f"  Completed steps: {', '.join(self.state.steps.keys())}\n")
                resumed = True
                # Restore context from completed steps
                self._restore_context()

        if not resumed or self.state is None:
            args_dict = {k: v for k, v in vars(self.args).items() if not callable(v)}
            self.state = PipelineState(
                started_at=datetime.now().isoformat(),
                config=args_dict,
                steps={},
                output_dir=str(self.output_dir),
            )

    def _restore_context(self) -> None:
        """Restore context from completed steps."""
        if not self.state:
            return

        # Restore dataset_path and dataset_dir
        if "dataset" in self.state.steps:
            output = self.state.steps["dataset"].get("output")
            if output:
                self.context["dataset_path"] = Path(output)
                self.context["dataset_dir"] = Path(output).parent

        # Restore metrics_path and reload metric_specs
        if "metrics" in self.state.steps:
            output = self.state.steps["metrics"].get("output")
            if output:
                metrics_path = Path(output)
                self.context["metrics_path"] = metrics_path
                # Reload metric_specs from saved file
                if metrics_path.exists():
                    self._reload_metric_specs(metrics_path)

        # Restore target_fn from args if available
        if self.args.target:
            from .loaders import _load_callable
            try:
                self.context["target_fn"] = _load_callable(self.args.target)
            except Exception:
                pass  # Will be None, steps handle this

        # Restore annotation_path
        if "annotation" in self.state.steps:
            output = self.state.steps["annotation"].get("output")
            if output:
                self.context["annotation_path"] = Path(output)

        # Restore calibrated_metrics list
        if "calibration" in self.state.steps:
            calibrated = self.state.steps["calibration"].get("calibrated_metrics", [])
            self.context["calibrated_metrics"] = calibrated

    def _reload_metric_specs(self, metrics_path: Path) -> None:
        """Reload metric specs from saved JSON file."""
        try:
            from ...models import MetricSpec
            with open(metrics_path, encoding="utf-8") as f:
                metrics_data = json.load(f)
            specs = []
            for data in metrics_data:
                spec = MetricSpec(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    type=data["type"],
                    description=data.get("description", ""),
                    config=data.get("config", {}),
                )
                specs.append(spec)
            self.context["metric_specs"] = specs
        except Exception:
            pass  # Non-fatal, steps will handle missing specs

    def _print_header(self) -> None:
        """Print pipeline header."""
        print("\n" + "=" * 70)
        print(" " * 15 + "EVALYN ONE-CLICK EVALUATION PIPELINE")
        print("=" * 70)
        print(f"\nProject:  {self.args.project}")
        if self.args.target:
            print(f"Target:   {self.args.target}")
        if self.args.version:
            print(f"Version:  {self.args.version}")
        mode = getattr(self.args, "metric_mode", "basic")
        model = getattr(self.args, "model", "")
        print(f"Mode:     {mode}" + (f" ({model})" if mode != "basic" else ""))
        print(f"Output:   {self.output_dir}")
        print("\n" + "-" * 70 + "\n")

        if self.args.dry_run:
            print("DRY RUN MODE - showing what would be done:\n")

    def _run_step(self, step: PipelineStep) -> None:
        """Run a single step with state management."""
        print(step.header)

        # Check if already done (resume mode)
        if self.state and self.state.is_step_done(step.name):
            print("  Already completed (skipping)\n")
            return

        # Check if should be skipped
        skip_reason = step.should_skip()
        if skip_reason:
            print(f"  SKIPPED ({skip_reason})\n")
            if self.state:
                self.state.mark_step(step.name, StepResult(status="skipped"))
                self.state.save(self.state_path)
            return

        # Dry run mode
        if self.args.dry_run:
            print(step.dry_run_message(self.output_dir))
            print()
            return

        # Execute step
        step_dir = self.output_dir / step.name
        step_dir.mkdir(exist_ok=True)

        try:
            result, new_context = step.execute(self.output_dir, self.context)
            self.context.update(new_context)

            if self.state:
                self.state.mark_step(step.name, result)
                self.state.save(self.state_path)

        except KeyboardInterrupt:
            print("\n  Step interrupted by user\n")
            if self.state:
                self.state.mark_step(step.name, StepResult(status="interrupted"))
                self.state.save(self.state_path)

    def _finalize(self) -> None:
        """Finalize pipeline execution."""
        if not self.state:
            return

        self.state.completed_at = datetime.now().isoformat()
        self.state.save(self.state_path)

        # Also save as pipeline_summary.json for backwards compat
        summary_path = self.output_dir / "pipeline_summary.json"
        self.state.save(summary_path)

        # Print summary
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nSummary:")

        status_icons = {"success": "[OK]", "skipped": "[SKIP]"}
        for step_name, step_info in self.state.steps.items():
            status = step_info.get("status", "unknown")
            icon = status_icons.get(status, "[FAIL]")
            print(f"  {icon} {step_name}: {status}")

        self._print_next_steps()

    def _print_next_steps(self) -> None:
        """Print suggested next steps."""
        if not self.state:
            return

        print("\nNext steps:")
        steps_info = self.state.steps

        if "calibrated_eval" in steps_info:
            if steps_info["calibrated_eval"].get("status") == "success":
                output = steps_info["calibrated_eval"].get("output", "")
                print(f"  1. Review results: cat {output}")
        elif "initial_eval" in steps_info:
            output = steps_info["initial_eval"].get("output", "")
            print(f"  1. Review results: cat {output}")

        summary_path = self.output_dir / "pipeline_summary.json"
        print(f"  2. View full summary: cat {summary_path}")
        print()

    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt."""
        print("\n\nPipeline interrupted by user")
        if self.state:
            self.state.save(self.state_path)
        print(f"Progress saved to: {self.state_path}")
        print(
            f"Resume with: evalyn one-click --project {self.args.project} "
            f"--output-dir {self.output_dir} --resume\n"
        )

    def _handle_error(self, error: Exception) -> None:
        """Handle pipeline error."""
        print(f"\n\nPipeline failed: {error}")
        if self.state:
            self.state.save(self.state_path)
        print(f"Progress saved to: {self.state_path}")
        print(
            f"Resume with: evalyn one-click --project {self.args.project} "
            f"--output-dir {self.output_dir} --resume\n"
        )
        if getattr(self.args, "verbose", False):
            import traceback

            traceback.print_exc()


__all__ = [
    "PipelineStep",
    "PipelineOrchestrator",
    "PipelineState",
    "StepResult",
]
