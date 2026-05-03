"""Report command: generate and open a static HTML insights report.

Opens a visual insights report in the default browser, providing
a "local LangSmith" experience with one command.

Usage:
    evalyn report
    evalyn report --output report.html
    evalyn report --dataset data/myapp/
    evalyn report --latest
    evalyn report --run <id>

Note: Previously named ``evalyn dashboard``. The ``dashboard`` name now
refers to the new IDE shipped via the separate ``evalyn-dashboard``
package; the legacy alias still forwards here with a deprecation warning.
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Insights"

import argparse
import webbrowser
from pathlib import Path

from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.hints import HintCollector
from ..utils.rich import icon


def _open_in_browser(file_path: Path) -> bool:
    """Try to open file in the default browser. Returns True on success."""
    try:
        return webbrowser.open(file_path.as_uri())
    except Exception:
        return False


def cmd_report(args: argparse.Namespace) -> None:
    """Generate and open an HTML insights report."""
    from ...analysis.insights_dashboard import generate_insights_html

    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        config,
    )

    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run", None),
        dataset_path=dataset_path,
    )

    print(f"Analyzing run: {loaded.run.id[:12]}...")

    from .insights import build_insights_report

    analysis, report, dataset_items = build_insights_report(
        loaded.run, dataset_path, loaded.run_file_path
    )

    html_content = generate_insights_html(
        run_analysis=analysis,
        insights_report=report,
        panel_discussion=None,
        dataset_items=dataset_items or None,
    )

    output_path = Path(getattr(args, "output", None) or ".evalyn/report.html")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    print(f"{icon('pass')} Report saved to: {output_path}")

    if _open_in_browser(output_path):
        print(f"{icon('pass')} Opened report in default browser.")
    else:
        file_url = output_path.as_uri()
        print(f"{icon('warn')} Could not open browser automatically.")
        print(f"Open this file in your browser: {file_url}")

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add(
        "evalyn insights --deep",
        "Run LLM expert panel analysis",
        options=[
            ("--project <name>", "Analyze a specific project"),
            ("--format json", "Machine-readable output"),
        ],
    )
    hints.render()


def register_commands(subparsers) -> None:
    """Register the ``report`` subcommand."""
    p = subparsers.add_parser(
        "report",
        help="Generate and open an HTML insights report in the browser",
    )
    p.add_argument("--run", help="Eval run ID to analyze")
    p.add_argument("--dataset", help="Dataset path (uses latest run)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    p.add_argument(
        "--output",
        help="Output file path (default: .evalyn/report.html)",
    )
    p.set_defaults(func=cmd_report)


__all__ = ["cmd_report", "register_commands", "_open_in_browser"]
