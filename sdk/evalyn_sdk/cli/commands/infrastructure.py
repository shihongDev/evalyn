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

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Infrastructure"

# Non-required params worth exposing in the default dashboard form. one-click is
# the highest-traffic command in this module: --target lets users point at their
# agent, --dataset-limit caps the work, and --metric-mode picks the strategy.
# init's useful default-visible knobs are --output (path to write the YAML)
# and --dir (where to create it).
ESSENTIAL = {"target", "dataset_limit", "metric_mode", "output", "dir"}

# Slider hints for the common numeric knobs.
RANGES = {"dataset_limit": (1, 1000, 10), "annotation_limit": (0, 200, 1)}
UNITS = {"dataset_limit": "items", "annotation_limit": "items"}

import argparse
from datetime import datetime
from pathlib import Path

from ..utils.config import (
    create_evalyn_yaml,
    find_project_root,
    get_config_default,
    load_config,
)
from ..utils.errors import fatal_error
from ..utils.rich import banner, footer, icon, section

# Real defaults for one-click pipeline arguments.
# Stored as constants so help text stays accurate when argparse defaults are None.
_DEFAULT_DATASET_LIMIT = 100
_DEFAULT_ANNOTATION_LIMIT = 20
_DEFAULT_NUM_SIMILAR = 3
_DEFAULT_NUM_OUTLIER = 2
_DEFAULT_MAX_SIM_SEEDS = 10


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize configuration file by copying from evalyn.yaml.example."""
    target_dir = Path(args.dir) if getattr(args, "dir", None) else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / args.output

    if output_path.exists() and not args.force:
        fatal_error(f"{output_path} already exists", "Use --force to overwrite")

    _, from_example = create_evalyn_yaml(output_path, force=True)
    if from_example:
        print(f"{icon('pass')} Created {output_path} (from evalyn.yaml.example)")
    else:
        print(f"{icon('pass')} Created {output_path} (minimal config)")
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
    from ...defaults import DEFAULT_EVAL_MODEL

    defaults = [
        ("version", "defaults", "version", None),
        ("metric_mode", "metrics", "mode", "basic"),
        ("model", "llm", "model", DEFAULT_EVAL_MODEL),
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

    # Handle defaults with special conditions (argparse defaults are None)
    if args.dataset_limit is None:
        args.dataset_limit = get_config_default(
            config, "dataset", "limit", default=_DEFAULT_DATASET_LIMIT
        )
    if args.annotation_limit is None:
        args.annotation_limit = get_config_default(
            config, "annotation", "limit", default=_DEFAULT_ANNOTATION_LIMIT
        )
    if args.num_similar is None:
        args.num_similar = get_config_default(
            config, "simulation", "num_similar", default=_DEFAULT_NUM_SIMILAR
        )
    if args.num_outlier is None:
        args.num_outlier = get_config_default(
            config, "simulation", "num_outlier", default=_DEFAULT_NUM_OUTLIER
        )
    if args.max_sim_seeds is None:
        args.max_sim_seeds = get_config_default(
            config, "simulation", "max_seeds", default=_DEFAULT_MAX_SIM_SEEDS
        )


def _resolve_version(args: argparse.Namespace) -> None:
    """Prompt for version if not specified."""
    if args.version:
        return

    from ...decorators import get_default_tracer

    storage = get_default_tracer().storage
    calls = storage.list_calls(limit=500, lightweight=True)
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
    # NOTE: This workflow text is manually maintained. Update when adding/removing commands.
    print()
    print(banner("EVALYN WORKFLOW"))
    print("\n  The evaluation pipeline has 3 phases:\n")

    print(section("PHASE 1: COLLECT"))
    print("  1. Add @eval decorator to your agent function")
    print("  2. Run your agent to collect traces")
    print("  3. Build a dataset from traces")
    print()
    print(f"  {icon('next')} evalyn list-calls              # View captured traces")
    print(f"  {icon('next')} evalyn show-projects           # See project summary")
    print(f"  {icon('next')} evalyn build-dataset --project <name>  # Create dataset")
    print()

    print(section("PHASE 2: EVALUATE"))
    print("  4. Select metrics for evaluation")
    print("  5. Run evaluation")
    print("  6. Analyze results")
    print()
    print(f"  {icon('next')} evalyn suggest-metrics --dataset <path> --mode basic")
    print(f"  {icon('next')} evalyn run-eval --dataset <path>")
    print(f"  {icon('next')} evalyn analyze --dataset <path>")
    print()

    print(section("PHASE 3: CALIBRATE (OPTIONAL)"))
    print("  7. Annotate results (human feedback)")
    print("  8. Calibrate LLM judges")
    print("  9. Re-evaluate with calibrated prompts")
    print()
    print(f"  {icon('next')} evalyn annotate --dataset <path>")
    print(f"  {icon('next')} evalyn calibrate --dataset <path> --metric-id <id>")
    print(f"  {icon('next')} evalyn run-eval --dataset <path> --use-calibrated")
    print()

    print(section("ONE-CLICK OPTION"))
    print("  Run the entire pipeline automatically:")
    print(f"  {icon('next')} evalyn one-click --project <name>")
    print()

    # Show context-aware next steps
    from ...decorators import get_default_tracer

    next_steps = []
    tracer = get_default_tracer()
    if tracer.storage:
        calls = tracer.storage.list_calls(limit=10, lightweight=True)
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
                next_steps.append(
                    (f"evalyn build-dataset --project {sorted(projects)[0]}", "Build dataset from traces")
                )
            else:
                next_steps.append(("evalyn build-dataset", "Build dataset from traces"))
        else:
            print("  No traces yet. Add @eval decorator to your agent and run it.")
    else:
        print("  No storage configured. Run your @eval-decorated agent first.")

    if next_steps:
        print(footer(next_steps))


def register_commands(subparsers) -> None:
    """Register infrastructure commands."""
    from ...defaults import DEFAULT_EVAL_MODEL

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
        "--dir",
        default=None,
        help="Directory to create config in (default: current directory)",
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
        default=None,
        help=f"Max dataset items (default: {_DEFAULT_DATASET_LIMIT})",
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
        default=DEFAULT_EVAL_MODEL,
        help=f"LLM model name (default: {DEFAULT_EVAL_MODEL})",
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
        default=None,
        help=f"Max items to annotate (default: {_DEFAULT_ANNOTATION_LIMIT})",
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
        default=None,
        help=f"Similar queries per seed (default: {_DEFAULT_NUM_SIMILAR})",
    )
    oneclick_parser.add_argument(
        "--num-outlier",
        type=int,
        default=None,
        help=f"Outlier queries per seed (default: {_DEFAULT_NUM_OUTLIER})",
    )
    oneclick_parser.add_argument(
        "--max-sim-seeds",
        type=int,
        default=None,
        help=f"Max seeds for simulation (default: {_DEFAULT_MAX_SIM_SEEDS})",
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
