"""Simulation commands: simulate.

This module provides CLI commands for generating synthetic test data using
LLM-based user simulation. This helps expand test coverage beyond real traces.

Commands:
- simulate: Generate synthetic test data from seed dataset

Simulation modes:
- similar: Generate variations of existing queries (same intent, different phrasing)
- outlier: Generate edge cases and unusual inputs to test robustness

How it works:
1. Load seed dataset with real input/output pairs
2. LLM generates new queries based on patterns in seed data
3. If --target is provided, run queries through the agent to get outputs
4. Save results as new dataset for evaluation

Temperature settings:
- --temp-similar (default 0.3): Lower temperature for consistent variations
- --temp-outlier (default 0.8): Higher temperature for creative edge cases

Typical workflow:
1. Build initial dataset: 'evalyn build-dataset --project <name>'
2. Generate synthetic data: 'evalyn simulate --dataset <path> --target <fn>'
3. The simulated data appears in <dataset>/simulations/
4. Run evaluation on synthetic data to find edge case failures
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Simulation"

# Non-required params worth exposing in the default dashboard form. The seed
# dataset is required; --target wires the agent function and the two count
# knobs are how you scale generation up or down.
ESSENTIAL = {"target", "num_similar", "num_outlier"}

# Slider hints for the integer count and float temperature knobs.
RANGES = {
    "num_similar": (0, 20, 1),
    "num_outlier": (0, 20, 1),
    "max_seeds": (1, 500, 1),
    "temp_similar": (0.0, 2.0, 0.05),
    "temp_outlier": (0.0, 2.0, 0.05),
}
UNITS = {
    "num_similar": "per seed",
    "num_outlier": "per seed",
    "max_seeds": "seeds",
}

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..utils.errors import fatal_error
from ..utils.hints import HintCollector
from ..utils.loaders import _load_callable
from ..utils.ui import Spinner


def run_simulation(
    dataset: str,
    target: str,
    output: str,
    *,
    modes: str = "similar,outlier",
    num_similar: int = 3,
    num_outlier: int = 1,
    max_seeds: int = 50,
    model: str = "",
    temp_similar: float = 0.3,
    temp_outlier: float = 0.8,
) -> None:
    """Run simulation - typed API callable without argparse."""
    from ...defaults import DEFAULT_EVAL_MODEL

    if not model:
        model = DEFAULT_EVAL_MODEL
    args = argparse.Namespace(
        dataset=dataset,
        target=target,
        output=output,
        modes=modes,
        num_similar=num_similar,
        num_outlier=num_outlier,
        max_seeds=max_seeds,
        model=model,
        temp_similar=temp_similar,
        temp_outlier=temp_outlier,
    )
    cmd_simulate(args)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Generate synthetic test data using LLM-based user simulation."""
    try:
        _cmd_simulate_inner(args)
    except FileNotFoundError as e:
        fatal_error(str(e))
    except ValueError as e:
        fatal_error(str(e))


def _cmd_simulate_inner(args: argparse.Namespace) -> None:
    """Inner implementation of cmd_simulate."""
    from ...datasets import load_dataset
    from ...decorators import eval as eval_decorator
    from ...simulation import AgentSimulator, SimulationConfig, UserSimulator

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        dataset_file = dataset_path / "dataset.jsonl"
    else:
        dataset_file = dataset_path
        dataset_path = dataset_file.parent

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_file}")

    # Load seed dataset
    seed_items = load_dataset(dataset_file)
    if not seed_items:
        raise ValueError("No items found in seed dataset")

    print(f"Loaded {len(seed_items)} seed items from {dataset_file}")

    # Load target function if provided
    target_fn = None
    if args.target:
        try:
            target_fn = _load_callable(args.target)
            print(f"Loaded target function: {args.target}")
        except Exception as e:
            print(f"Warning: Could not load target function: {e}")
            print("Simulation will generate queries only (no agent execution)")

    # Parse modes
    raw_modes = [m.strip() for m in args.modes.split(",")]
    valid_modes = {"similar", "outlier"}
    invalid_modes = [m for m in raw_modes if m and m not in valid_modes]
    if invalid_modes:
        fatal_error(
            f"Invalid simulation mode(s): {', '.join(invalid_modes)}",
            "Valid modes are: similar, outlier",
        )
    modes = [m for m in raw_modes if m in valid_modes]
    if not modes:
        modes = ["similar", "outlier"]

    print(f"Simulation modes: {modes}")

    # Build config
    config = SimulationConfig(
        num_similar=args.num_similar,
        num_outlier=args.num_outlier,
        model=args.model,
        temperature_similar=args.temp_similar,
        temperature_outlier=args.temp_outlier,
        max_seed_items=args.max_seeds,
    )

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: create sim-<timestamp> folder inside dataset directory
        output_dir = dataset_path / "simulations"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if target_fn:
        # Wrap target function with is_simulation=True for all simulation traces
        target_fn_wrapped = eval_decorator(is_simulation=True)(target_fn)

        # Full simulation: generate + run agent
        simulator = AgentSimulator(
            target_fn=target_fn_wrapped,
            config=config,
            model=args.model,
        )

        with Spinner("Running agent simulation"):
            results = simulator.run(
                seed_dataset=seed_items,
                output_dir=output_dir,
                modes=modes,
            )

        print(f"\n{'=' * 60}")
        print("SIMULATION COMPLETE")
        print(f"{'=' * 60}")
        for mode, path in results.items():
            # Count items
            sim_dataset_file = path / "dataset.jsonl"
            if sim_dataset_file.exists():
                with open(sim_dataset_file, encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                print(f"  {mode}: {count} items -> {path}")
            else:
                print(f"  {mode}: -> {path}")

        # Show hints
        hints = HintCollector(quiet=getattr(args, "quiet", False))
        first_result_path = list(results.values())[0] if results else None
        if first_result_path:
            hints.add(f"evalyn run-eval --dataset {first_result_path}", "Evaluate simulated data")
        hints.render()
    else:
        # Query generation only (no target function)
        user_sim = UserSimulator(model=args.model)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for mode in modes:
            print(f"\nGenerating {mode} queries...")
            with Spinner(f"Generating {mode} queries"):
                if mode == "similar":
                    generated = user_sim.generate_similar(
                        seed_items[: config.max_seed_items],
                        num_per_seed=config.num_similar,
                    )
                else:
                    generated = user_sim.generate_outliers(
                        seed_items[: config.max_seed_items],
                        num_per_seed=config.num_outlier,
                    )

            if not generated:
                print(f"  No queries generated for mode={mode}")
                continue

            # Save generated queries
            mode_dir = output_dir / f"queries-{mode}-{timestamp}"
            mode_dir.mkdir(parents=True, exist_ok=True)

            queries_file = mode_dir / "queries.jsonl"
            with open(queries_file, "w", encoding="utf-8") as f:
                for gq in generated:
                    f.write(
                        json.dumps(
                            {
                                "query": gq.query,
                                "mode": gq.mode,
                                "seed_id": gq.seed_id,
                                "generation_reason": gq.generation_reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # Save meta
            meta = {
                "type": "generated_queries",
                "mode": mode,
                "created_at": datetime.now().isoformat(),
                "seed_dataset": str(dataset_file),
                "num_queries": len(generated),
                "config": {
                    "model": config.model,
                    "num_per_seed": config.num_similar
                    if mode == "similar"
                    else config.num_outlier,
                    "temperature": config.temperature_similar
                    if mode == "similar"
                    else config.temperature_outlier,
                },
            }
            with open(mode_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"  Generated {len(generated)} {mode} queries -> {mode_dir}")

        print(f"\n{'=' * 60}")
        print("QUERY GENERATION COMPLETE")
        print(f"{'=' * 60}")

        hints = HintCollector(quiet=getattr(args, "quiet", False))
        hints.add("evalyn simulate --target <module:func>", "Run queries through your agent")
        hints.render()


def register_commands(subparsers) -> None:
    """Register simulation commands."""
    from ...defaults import DEFAULT_EVAL_MODEL

    # simulate
    p = subparsers.add_parser(
        "simulate", help="Generate synthetic test data using LLM-based user simulation"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to seed dataset directory or dataset.jsonl",
    )
    p.add_argument(
        "--target",
        help="Target function to run queries against (module:func or path/to/file.py:func)",
    )
    p.add_argument(
        "--output",
        help="Output directory for simulated data (default: <dataset>/simulations/)",
    )
    p.add_argument(
        "--modes",
        default="similar,outlier",
        help="Simulation modes: similar,outlier (comma-separated)",
    )
    p.add_argument(
        "--num-similar",
        type=int,
        default=3,
        help="Number of similar variations per seed item (default: 3)",
    )
    p.add_argument(
        "--num-outlier",
        type=int,
        default=1,
        help="Number of outlier/edge cases per seed item (default: 1)",
    )
    p.add_argument(
        "--max-seeds",
        type=int,
        default=50,
        help="Maximum seed items to use (default: 50)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_EVAL_MODEL,
        help=f"LLM model for query generation (default: {DEFAULT_EVAL_MODEL})",
    )
    p.add_argument(
        "--temp-similar",
        type=float,
        default=0.3,
        help="Temperature for similar queries (default: 0.3)",
    )
    p.add_argument(
        "--temp-outlier",
        type=float,
        default=0.8,
        help="Temperature for outlier queries (default: 0.8)",
    )
    p.set_defaults(func=cmd_simulate)


__all__ = ["cmd_simulate", "register_commands"]
