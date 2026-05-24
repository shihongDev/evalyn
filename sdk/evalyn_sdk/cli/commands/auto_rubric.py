"""Auto-rubric command: synthesize an LLM-judge rubric from labeled examples.

Lowers the bar for first-time users: instead of hand-writing a subjective
metric rubric, label 5-10 examples (input/output/score/rationale) and let
the LLM draft a judge prompt for you.
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Calibration"

# Non-required params worth exposing in the default dashboard form.
ESSENTIAL = {"examples", "metric_id", "scale", "description"}

import argparse
import json
import re
from pathlib import Path

from ...calibration.rubric_generator import (
    SUPPORTED_SCALES,
    generate_rubric,
    load_examples,
)
from ..utils.errors import fatal_error
from ..utils.hints import HintCollector
from ..utils.rich import banner, icon, kv, section


def _safe_filename(name: str) -> str:
    """Sanitize a metric_id for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "metric"


def cmd_auto_rubric(args: argparse.Namespace) -> None:
    """Draft an LLM-judge rubric from labeled examples."""
    quiet = getattr(args, "quiet", False)
    examples_path = Path(args.examples)
    metric_id = args.metric_id
    description = getattr(args, "description", None)
    scale = getattr(args, "scale", "0-1")
    provider = getattr(args, "provider", "gemini")
    output_arg = getattr(args, "output", None)
    api_key = getattr(args, "api_key", None)

    if scale not in SUPPORTED_SCALES:
        fatal_error(
            f"Unsupported scale '{scale}'",
            f"Choose from: {', '.join(SUPPORTED_SCALES)}",
        )

    print()
    print(banner("EVALYN AUTO-RUBRIC"))

    # Step 1: load + validate examples
    print()
    print(section("STEP 1/3: LOAD EXAMPLES"))
    try:
        examples = load_examples(examples_path)
    except (FileNotFoundError, ValueError) as e:
        fatal_error(str(e))

    print(
        kv(
            [
                ("Examples file", str(examples_path)),
                ("Examples loaded", str(len(examples))),
                ("Scale", scale),
                ("Provider", provider),
            ]
        )
    )

    # Step 2: call LLM to draft rubric
    print()
    print(section("STEP 2/3: DRAFT RUBRIC"))
    print(f"  Sending {len(examples)} examples to {provider}...")
    try:
        rubric = generate_rubric(
            examples,
            scale=scale,
            metric_id=metric_id,
            description=description,
            provider=provider,
            api_key=api_key,
        )
    except ValueError as e:
        fatal_error(f"Rubric generation failed: {e}")
    except Exception as e:
        fatal_error(f"LLM call failed: {e}")

    rubric_criteria = rubric.get("config", {}).get("rubric", [])
    print(f"  {icon('pass')} Generated rubric with {len(rubric_criteria)} criteria")

    # Step 3: write to disk
    print()
    print(section("STEP 3/3: WRITE METRIC FILE"))
    output_path = (
        Path(output_arg)
        if output_arg
        else Path("data") / "metrics" / f"{_safe_filename(metric_id)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to silently clobber an existing rubric - a user re-running
    # this command may have hand-tuned the previous file.
    if output_path.exists() and not getattr(args, "force", False):
        fatal_error(
            f"Output file already exists: {output_path}",
            "Pass --force to overwrite, or --output <path> for a new location.",
        )
    # The file format used by --metrics is a JSON array of specs.
    output_path.write_text(
        json.dumps([rubric], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  {icon('pass')} Wrote rubric to {output_path}")

    # Final summary
    print()
    print(banner("DONE"))
    summary_rows = [
        ("Metric ID", str(rubric.get("id", metric_id))),
        ("Description", str(rubric.get("description", "")) or "(none)"),
        ("Category", str(rubric.get("category", "(unset)"))),
        ("Scope", str(rubric.get("scope", "(unset)"))),
        ("Scale", str(rubric.get("config", {}).get("scale", scale))),
        ("Criteria", str(len(rubric_criteria))),
        ("Examples used", str(len(examples))),
        ("Output file", str(output_path)),
    ]
    print(kv(summary_rows))

    hints = HintCollector(quiet=quiet)
    hints.add(
        f"evalyn run-eval --dataset <path> --metrics {output_path}",
        "Run evaluation with the new rubric",
    )
    hints.add(
        f"evalyn calibrate --metric-id {metric_id} --annotations <path>",
        "Calibrate the rubric prompt against more annotations",
    )
    hints.render()


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "auto-rubric",
        help="Draft an LLM-judge rubric from labeled examples (5+ required)",
    )
    parser.add_argument(
        "--examples",
        required=True,
        help="Path to JSONL file of labeled examples. Each line: "
        '{"input": ..., "output": ..., "label": <score>, "rationale": "..."}',
    )
    parser.add_argument(
        "--metric-id",
        required=True,
        help="Identifier for the new metric (used as both the rubric id and filename)",
    )
    parser.add_argument(
        "--description",
        help="One-line description of what the metric evaluates",
    )
    parser.add_argument(
        "--scale",
        choices=list(SUPPORTED_SCALES),
        default="0-1",
        help="Scoring scale for the rubric (default: 0-1)",
    )
    parser.add_argument(
        "--output",
        help="Path to write the rubric JSON file. "
        "Default: data/metrics/<metric_id>.json",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider for drafting the rubric (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the chosen provider (default: from env / evalyn.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing rubric file at the output path.",
    )
    parser.set_defaults(func=cmd_auto_rubric)


__all__ = ["cmd_auto_rubric", "register_commands"]
