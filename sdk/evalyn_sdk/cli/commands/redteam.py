"""Redteam command: produce an adversarial dataset variant for safety
and robustness testing.

Given an input evaluation dataset, this command generates adversarial
variants of each item using three well-known technique categories:

- injection: prompt-injection attempts (instruction overrides, indirect
  injection via embedded data, encoded payload hints). LLM-driven via the
  existing UserSimulator with deterministic seed-template fallback.
- jailbreak: jailbreak phrasing (role-play, hypothetical framing, fictional
  story framing). LLM-driven with seed-template fallback.
- edge: procedural edge cases (empty input, whitespace-only, extreme length,
  unicode/zero-width tricks, mixed languages, control characters). No LLM
  required; deterministic.

The resulting dataset is written as JSONL alongside the source, suitable for
``evalyn run-eval``. Each variant carries metadata.redteam_technique,
metadata.redteam_subtype, and metadata.redteam_source_id for downstream
attribution.

Usage:
    evalyn redteam --dataset path/to/dataset.jsonl
    evalyn redteam --dataset dir --techniques injection,jailbreak \
        --variants-per-item 3 --output adv.jsonl --limit 20
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Simulation"

# Non-required params worth exposing in the default dashboard form.
ESSENTIAL = {"techniques", "variants_per_item", "limit"}

# Slider hints for the integer knobs.
RANGES = {
    "variants_per_item": (1, 10, 1),
    "limit": (1, 1000, 1),
}
UNITS = {
    "variants_per_item": "per technique",
    "limit": "items",
}

import argparse
from pathlib import Path

from ..utils.errors import fatal_error
from ..utils.hints import HintCollector
from ..utils.rich import banner, kv

# Canonical technique keys.
_VALID_TECHNIQUES = ("injection", "jailbreak", "edge")
_DEFAULT_TECHNIQUES = "injection,jailbreak,edge"


def _resolve_techniques(raw: str) -> list[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    invalid = [p for p in parts if p not in _VALID_TECHNIQUES]
    if invalid:
        fatal_error(
            f"Invalid technique(s): {', '.join(invalid)}",
            f"Valid techniques: {', '.join(_VALID_TECHNIQUES)}",
        )
    if not parts:
        return list(_VALID_TECHNIQUES)
    # Preserve declared order but dedupe.
    return list(dict.fromkeys(parts))


def _resolve_dataset_file(dataset_arg: str) -> Path:
    """Resolve the dataset_arg (file or directory) to an actual dataset file."""
    path = Path(dataset_arg)
    dataset_file = path / "dataset.jsonl" if path.is_dir() else path
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_file}")
    return dataset_file


def _resolve_output_path(output_arg: str | None, dataset_file: Path) -> Path:
    if output_arg:
        return Path(output_arg)
    # Default: alongside source as <stem>_redteam.jsonl
    return dataset_file.with_name(f"{dataset_file.stem}_redteam.jsonl")


def cmd_redteam(args: argparse.Namespace) -> None:
    """Generate an adversarial variant dataset from a source dataset."""
    from ...datasets import load_dataset, save_dataset
    from ...simulation import UserSimulator
    from ...simulation.redteam import (
        generate_edge_case_variants,
        generate_injection_variants,
        generate_jailbreak_variants,
    )

    try:
        dataset_file = _resolve_dataset_file(args.dataset)
    except FileNotFoundError as e:
        fatal_error(str(e))
    seed_items = load_dataset(dataset_file)
    if not seed_items:
        fatal_error("No items found in source dataset")

    techniques = _resolve_techniques(getattr(args, "techniques", _DEFAULT_TECHNIQUES))
    variants_per_item = max(0, int(getattr(args, "variants_per_item", 2)))
    limit = getattr(args, "limit", None)
    if limit is not None:
        limit = max(0, int(limit))
        seed_items = seed_items[:limit]

    output_path = _resolve_output_path(getattr(args, "output", None), dataset_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(seed_items)} source items from {dataset_file}")
    print(f"Techniques: {', '.join(techniques)}")
    print(f"Variants per item per technique: {variants_per_item}")
    print(f"Output: {output_path}")

    # Lazy-construct the LLM only if an LLM-driven technique is requested.
    llm: UserSimulator | None = None
    if any(t in techniques for t in ("injection", "jailbreak")):
        try:
            llm = UserSimulator(model=getattr(args, "model", "gemini-2.5-flash-lite"))
        except Exception as e:
            print(f"Warning: could not initialize LLM simulator ({e}); falling back to seed templates only.")
            llm = None

    technique_counts: dict[str, int] = {t: 0 for t in techniques}
    all_variants = []
    for item in seed_items:
        for technique in techniques:
            if technique == "injection":
                variants = generate_injection_variants(item, variants_per_item, llm=llm)
            elif technique == "jailbreak":
                variants = generate_jailbreak_variants(item, variants_per_item, llm=llm)
            elif technique == "edge":
                variants = generate_edge_case_variants(item, variants_per_item)
            else:  # pragma: no cover - guarded above
                variants = []
            technique_counts[technique] += len(variants)
            all_variants.extend(variants)

    save_dataset(all_variants, output_path)

    print()
    print(banner("REDTEAM COMPLETE"))
    summary_pairs = [(t, str(c)) for t, c in technique_counts.items()]
    summary_pairs.append(("total variants", str(len(all_variants))))
    summary_pairs.append(("source items", str(len(seed_items))))
    summary_pairs.append(("output", str(output_path)))
    print(kv(summary_pairs))
    print(
        f"Generated {len(all_variants)} adversarial variants from "
        f"{len(seed_items)} source items across techniques: {', '.join(techniques)}"
    )

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add(
        f"evalyn run-eval --dataset {output_path}",
        "Evaluate the adversarial dataset against your agent",
        options=[
            ("--provider gemini|openai|ollama", "LLM provider for judges (default: gemini)"),
            ("--workers <N>", "Parallel workers (default: 4)"),
        ],
    )
    hints.render()


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the redteam command."""
    p = subparsers.add_parser(
        "redteam",
        help="Generate an adversarial dataset for safety/robustness testing",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to source dataset directory or dataset.jsonl",
    )
    p.add_argument(
        "--techniques",
        default=_DEFAULT_TECHNIQUES,
        help=f"Comma-separated techniques (default: {_DEFAULT_TECHNIQUES}). Valid: {', '.join(_VALID_TECHNIQUES)}",
    )
    p.add_argument(
        "--variants-per-item",
        type=int,
        default=2,
        help="Number of variants per source item per technique (default: 2)",
    )
    p.add_argument(
        "--output",
        help="Output JSONL path (default: <dataset_stem>_redteam.jsonl alongside source)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N source items (default: all)",
    )
    p.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for injection/jailbreak generation (default: gemini-2.5-flash-lite)",
    )
    p.set_defaults(func=cmd_redteam)


__all__ = ["cmd_redteam", "register_commands"]
