"""Infrastructure commands: init, one-click.

This module provides CLI commands for setup and running the full evaluation pipeline.

Commands:
- init: Initialize configuration file (evalyn.yaml) from template
- one-click: Run complete evaluation pipeline in one command

The one-click pipeline:
1. Build Dataset: Collect traces from storage by project/version
2. Suggest Metrics: Select metrics based on mode (basic/llm-registry/llm-brainstorm/bundle)
3. Run Initial Evaluation: Execute metrics on dataset items
4. Human Annotation: Interactive annotation session (optional)
5. Calibrate LLM Judges: Optimize prompts using annotations (optional)
6. Re-evaluate with Calibrated Prompts: Run again with improved judges (optional)
7. Generate Simulations: Create synthetic test data (optional)

Pipeline state:
- Progress is saved to pipeline_state.json for resume capability
- Use --resume to continue from where you left off
- Each step outputs to folders (dataset/, metrics/, etc.)

Typical workflow:
1. Initialize config: 'evalyn init'
2. Set API key: 'export GEMINI_API_KEY=...'
3. Run pipeline: 'evalyn one-click --project <name>'
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ...storage import SQLiteStorage
from ..utils.config import load_config, get_config_default, find_project_root
from ..utils.errors import fatal_error


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize configuration file by copying from evalyn.yaml.example."""
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        fatal_error(f"{output_path} already exists", "Use --force to overwrite")

    # Find the example file - check multiple locations
    example_paths = [
        Path("evalyn.yaml.example"),  # Current directory
        Path(__file__).parent.parent.parent.parent
        / "evalyn.yaml.example",  # Project root
    ]

    example_path = None
    for p in example_paths:
        if p.exists():
            example_path = p
            break

    if example_path:
        shutil.copy(example_path, output_path)
        print(f"Created {output_path} (from {example_path})")
    else:
        # Fallback: create minimal config if example not found
        minimal = """# Evalyn Configuration
# See evalyn.yaml.example for all available options

# API Keys - only set what you need
api_keys:
  gemini: "your-gemini-api-key-here"  # Required for example agent
  # openai: "your-openai-key"         # Optional

llm:
  model: "gemini-2.5-flash-lite"

defaults:
  project: null
  version: null
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(minimal)
        print(f"Created {output_path} (minimal config)")
        print("Note: evalyn.yaml.example not found for full template")

    print("\nSet your API key:")
    print("  export GEMINI_API_KEY='your-key'")
    print(f"  # or edit {output_path} directly")


def cmd_one_click(args: argparse.Namespace) -> None:
    """Run the complete evaluation pipeline from dataset building to calibrated evaluation.

    Pipeline steps:
    [1/7] Build Dataset: Collect traces by project, save to dataset/
    [2/7] Suggest Metrics: Select metrics based on mode, save to metrics/
    [3/7] Run Initial Eval: Execute metrics, save to initial_eval/
    [4/7] Human Annotation: Interactive session, save to annotations/ (optional)
    [5/7] Calibrate Judges: Optimize prompts, save to calibrations/ (optional)
    [6/7] Calibrated Eval: Re-run with improved prompts, save to calibrated_eval/
    [7/7] Simulate: Generate synthetic data, save to simulations/ (optional)

    State is persisted to pipeline_state.json for resume capability.
    """
    from ..utils.pipeline import PipelineOrchestrator
    from ..utils.pipeline_steps import create_pipeline_steps

    # Load config and apply defaults
    config = load_config()
    _apply_config_defaults(args, config)

    # Handle version selection
    _resolve_version(args)

    # Create output directory
    output_dir = _create_output_dir(args)

    # Create and run pipeline
    steps = create_pipeline_steps(args, config)
    orchestrator = PipelineOrchestrator(steps, output_dir, args)
    orchestrator.run()


def _apply_config_defaults(args: argparse.Namespace, config: dict) -> None:
    """Apply config file defaults to unset args."""
    defaults = [
        ("version", "defaults", "version", None),
        ("metric_mode", "metrics", "mode", "basic"),
        ("model", "llm", "model", "gemini-2.5-flash-lite"),
        ("llm_mode", "llm", "mode", "api"),
        ("bundle", "metrics", "bundle", None),
        ("skip_annotation", "annotation", "skip", False),
        ("per_metric", "annotation", "per_metric", False),
        ("skip_calibration", "calibration", "skip", False),
        ("optimizer", "calibration", "optimizer", "basic"),
        ("enable_simulation", "simulation", "enable", False),
        ("simulation_modes", "simulation", "modes", "similar"),
        ("auto_yes", "pipeline", "auto_yes", False),
        ("verbose", "pipeline", "verbose", False),
    ]

    for attr, section, key, fallback in defaults:
        if not getattr(args, attr, None):
            setattr(
                args, attr, get_config_default(config, section, key, default=fallback)
            )

    # Handle defaults with special conditions (default values from argparse)
    if args.dataset_limit == 100:
        args.dataset_limit = get_config_default(config, "dataset", "limit", default=100)
    if args.annotation_limit == 20:
        args.annotation_limit = get_config_default(
            config, "annotation", "limit", default=20
        )
    if args.num_similar == 3:
        args.num_similar = get_config_default(
            config, "simulation", "num_similar", default=3
        )
    if args.num_outlier == 2:
        args.num_outlier = get_config_default(
            config, "simulation", "num_outlier", default=1
        )
    if args.max_sim_seeds == 10:
        args.max_sim_seeds = get_config_default(
            config, "simulation", "max_seeds", default=50
        )


def _resolve_version(args: argparse.Namespace) -> None:
    """Prompt for version if not specified."""
    if args.version:
        return

    storage = SQLiteStorage()
    calls = storage.list_calls(limit=500)
    versions = {
        (call.metadata or {}).get("version")
        for call in calls
        if (call.metadata or {}).get("project_name") == args.project
        or (call.metadata or {}).get("project_id") == args.project
    }
    versions.discard(None)

    if len(versions) == 0:
        print(f"No versions found for project '{args.project}'")
        print("Proceeding without version filter (all traces)...")
    elif len(versions) == 1:
        args.version = list(versions)[0]
        print(f"Auto-selected version: {args.version}")
    else:
        # In auto-yes mode, select most recent (last in sorted list)
        if getattr(args, "auto_yes", False):
            args.version = sorted(versions)[-1]
            print(f"Auto-selected version (--auto-yes): {args.version}")
            return

        print(f"\nAvailable versions for project '{args.project}':")
        version_list = sorted(versions)
        for i, v in enumerate(version_list, 1):
            print(f"  {i}. {v}")
        print(f"  {len(version_list) + 1}. [all versions]")

        try:
            choice = input("\nSelect version (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(version_list):
                args.version = version_list[idx]
            elif idx == len(version_list):
                args.version = None
            else:
                print("Invalid selection, using all versions")
        except (ValueError, EOFError):
            print("Invalid input, using all versions")


def _create_output_dir(args: argparse.Namespace) -> Path:
    """Create and return output directory (always under project root/data/)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        version_str = f"-{args.version}" if args.version else ""
        project_root = find_project_root()
        output_dir = (
            project_root / "data" / f"{args.project}{version_str}-{timestamp}-oneclick"
        )
    return output_dir


def cmd_workflow(args: argparse.Namespace) -> None:
    """Show the evaluation workflow and next steps."""
    workflow = """
EVALYN WORKFLOW
===============

The evaluation pipeline has 3 phases:

PHASE 1: COLLECT
----------------
  1. Add @eval decorator to your agent function
  2. Run your agent to collect traces
  3. Build a dataset from traces

  Commands:
    evalyn list-calls              # View captured traces
    evalyn show-projects           # See project summary
    evalyn build-dataset --project <name>  # Create dataset

PHASE 2: EVALUATE
-----------------
  4. Select metrics for evaluation
  5. Run evaluation
  6. Analyze results

  Commands:
    evalyn suggest-metrics --dataset <path> --mode basic
    evalyn run-eval --dataset <path>
    evalyn analyze --dataset <path>

PHASE 3: CALIBRATE (optional)
-----------------------------
  7. Annotate results (human feedback)
  8. Calibrate LLM judges
  9. Re-evaluate with calibrated prompts

  Commands:
    evalyn annotate --dataset <path>
    evalyn calibrate --dataset <path> --metric-id <id>
    evalyn run-eval --dataset <path> --use-calibrated

ONE-CLICK OPTION
----------------
  Run the entire pipeline automatically:
    evalyn one-click --project <name>

NEXT STEPS
----------
"""
    print(workflow)

    # Show context-aware next steps
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    if tracer.storage:
        calls = tracer.storage.list_calls(limit=10)
        if calls:
            projects = set()
            for call in calls:
                if isinstance(call.metadata, dict):
                    proj = call.metadata.get("project_id") or call.metadata.get(
                        "project"
                    )
                    if proj:
                        projects.add(proj)
            if projects:
                print(f"  You have traces for: {', '.join(sorted(projects))}")
                print(f"  Try: evalyn build-dataset --project {sorted(projects)[0]}")
            else:
                print("  You have traces. Try: evalyn build-dataset")
        else:
            print("  No traces yet. Add @eval decorator to your agent and run it.")
    else:
        print("  No storage configured. Run your @eval-decorated agent first.")


def register_commands(subparsers) -> None:
    """Register infrastructure commands."""
    # workflow
    workflow_parser = subparsers.add_parser(
        "workflow", help="Show evaluation workflow and next steps"
    )
    workflow_parser.set_defaults(func=cmd_workflow)

    # init
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--output",
        default="evalyn.yaml",
        help="Output path for config file (default: evalyn.yaml)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file"
    )
    init_parser.set_defaults(func=cmd_init)

    # one-click
    oneclick_parser = subparsers.add_parser(
        "one-click",
        help="Run complete evaluation pipeline (dataset -> metrics -> eval -> annotate -> calibrate)",
    )
    oneclick_parser.add_argument(
        "--project", required=True, help="Project name to filter traces"
    )
    oneclick_parser.add_argument(
        "--target",
        help="Target function (file.py:func or module:func). Optional - if not provided, uses existing trace outputs",
    )
    oneclick_parser.add_argument(
        "--version", help="Version filter (default: all versions)"
    )
    oneclick_parser.add_argument(
        "--production-only", action="store_true", help="Use only production traces"
    )
    oneclick_parser.add_argument(
        "--simulation-only", action="store_true", help="Use only simulation traces"
    )
    oneclick_parser.add_argument(
        "--output-dir", help="Custom output directory (default: auto-generated)"
    )
    oneclick_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous incomplete run (auto-detects from output-dir)",
    )
    oneclick_parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="Max dataset items (default: 100)",
    )
    oneclick_parser.add_argument(
        "--since", help="Filter traces since date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--until", help="Filter traces until date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--metric-mode",
        choices=["basic", "llm-registry", "llm-brainstorm", "bundle", "all"],
        default="basic",
        help="Metric selection mode: basic (fast), llm-registry (LLM picks from templates), "
        "llm-brainstorm (LLM generates custom), bundle (preset), all (comprehensive - runs all modes)",
    )
    oneclick_parser.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        default="api",
        help="LLM mode for metric selection (default: api)",
    )
    oneclick_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model name (default: gemini-2.5-flash-lite)",
    )
    oneclick_parser.add_argument("--bundle", help="Bundle name (if metric-mode=bundle)")
    oneclick_parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--annotation-limit",
        type=int,
        default=20,
        help="Max items to annotate (default: 20)",
    )
    oneclick_parser.add_argument(
        "--per-metric", action="store_true", help="Use per-metric annotation mode"
    )
    oneclick_parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration step (default: false)",
    )
    oneclick_parser.add_argument(
        "--optimizer",
        choices=["basic", "gepa", "opro", "ape"],
        default="basic",
        help="Prompt optimization method: basic (single-shot, default), gepa (evolutionary), opro (trajectory), ape (search)",
    )
    oneclick_parser.add_argument(
        "--calibrate-all-metrics",
        action="store_true",
        help="Calibrate all subjective metrics (default: only poorly-aligned)",
    )
    oneclick_parser.add_argument(
        "--enable-simulation",
        action="store_true",
        help="Enable simulation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--simulation-modes",
        default="similar",
        help="Simulation modes: similar,outlier (default: similar)",
    )
    oneclick_parser.add_argument(
        "--num-similar",
        type=int,
        default=3,
        help="Similar queries per seed (default: 3)",
    )
    oneclick_parser.add_argument(
        "--num-outlier",
        type=int,
        default=2,
        help="Outlier queries per seed (default: 2)",
    )
    oneclick_parser.add_argument(
        "--max-sim-seeds",
        type=int,
        default=10,
        help="Max seeds for simulation (default: 10)",
    )
    oneclick_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Parallel workers for LLM evaluation (default: 4, max: 16)",
    )
    oneclick_parser.add_argument(
        "--auto-yes",
        action="store_true",
        help="Skip all confirmation prompts (default: false)",
    )
    oneclick_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed logs (default: false)"
    )
    oneclick_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing (default: false)",
    )
    oneclick_parser.set_defaults(func=cmd_one_click)


__all__ = ["cmd_init", "cmd_one_click", "register_commands"]
