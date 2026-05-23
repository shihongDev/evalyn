"""Bundle commands: list-bundles, show-bundle.

Surfaces the curated metric bundles defined in `cli/constants.py` so users
don't have to read the source to discover them.
"""

from __future__ import annotations

import argparse
import json

from ..constants import BUNDLE_DESCRIPTIONS, BUNDLES

GROUP = "Metrics"
ESSENTIAL: set[str] = set()


def cmd_list_bundles(args: argparse.Namespace) -> None:
    """List all curated metric bundles."""
    if getattr(args, "format", "table") == "json":
        records = [
            {
                "name": name,
                "description": BUNDLE_DESCRIPTIONS.get(name, ""),
                "metric_count": len(metrics),
                "metrics": metrics,
            }
            for name, metrics in sorted(BUNDLES.items())
        ]
        print(json.dumps({"count": len(records), "bundles": records}, indent=2))
        return

    print(f"\nMetric Bundles ({len(BUNDLES)} total)")
    print("=" * 78)
    print(f"{'Name':<22}  {'# Metrics':>10}  Description")
    print("-" * 78)
    for bundle_name, metrics in sorted(BUNDLES.items()):
        desc = BUNDLE_DESCRIPTIONS.get(bundle_name, "")
        print(f"{bundle_name:<22}  {len(metrics):>10}  {desc}")
    print()
    print("Use a bundle:  evalyn suggest-metrics --mode bundle --bundle <name>")
    print("See details:   evalyn show-bundle <name>")


def cmd_show_bundle(args: argparse.Namespace) -> None:
    """Show all metrics in a specific bundle."""
    name = args.name
    if name not in BUNDLES:
        print(f"Error: bundle '{name}' not found.")
        print(f"\nAvailable bundles: {', '.join(sorted(BUNDLES.keys()))}")
        raise SystemExit(2)

    metrics = BUNDLES[name]
    desc = BUNDLE_DESCRIPTIONS.get(name, "")

    if getattr(args, "format", "table") == "json":
        print(json.dumps({
            "name": name,
            "description": desc,
            "metric_count": len(metrics),
            "metrics": metrics,
        }, indent=2))
        return

    print(f"\nBundle: {name}")
    if desc:
        print(f"  {desc}")
    print(f"  {len(metrics)} metrics")
    print()
    print("Metrics:")
    for i, metric_id in enumerate(metrics, 1):
        print(f"  {i:>3}. {metric_id}")
    print()
    print(f"Use this bundle:  evalyn suggest-metrics --mode bundle --bundle {name}")


def register_commands(subparsers) -> None:
    list_parser = subparsers.add_parser(
        "list-bundles",
        help="List curated metric bundles",
        description="Show all available metric bundles with metric counts.",
    )
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    list_parser.set_defaults(func=cmd_list_bundles)

    show_parser = subparsers.add_parser(
        "show-bundle",
        help="Show all metrics in a specific bundle",
        description="Show the full metric list for a curated bundle.",
    )
    show_parser.add_argument("name", help="Bundle name (e.g. 'rag-qa')")
    show_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    show_parser.set_defaults(func=cmd_show_bundle)
