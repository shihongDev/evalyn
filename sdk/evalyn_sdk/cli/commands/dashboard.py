"""Dashboard command: generate and open an HTML insights dashboard.

Opens a visual insights dashboard in the default browser, providing
a "local LangSmith" experience with one command.

Usage:
    evalyn dashboard
    evalyn dashboard --output report.html
    evalyn dashboard --dataset data/myapp/
    evalyn dashboard --latest
    evalyn dashboard --run <id>
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.hints import HintCollector


def _open_in_browser(file_path: Path) -> bool:
    """Try to open file in the default browser. Returns True on success."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(file_path)], check=True)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(file_path)], check=True)
        elif system == "Windows":
            subprocess.run(["start", str(file_path)], check=True, shell=True)
        else:
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate and open an HTML insights dashboard."""
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

    output_path = Path(getattr(args, "output", None) or ".evalyn/dashboard.html")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    print(f"Dashboard saved to: {output_path}")

    if _open_in_browser(output_path):
        print("Opened dashboard in default browser.")
    else:
        file_url = output_path.as_uri()
        print(f"Could not open browser automatically.")
        print(f"Open this file in your browser: {file_url}")

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add("evalyn insights --deep", "Run LLM expert panel analysis")
    hints.render()


def register_commands(subparsers) -> None:
    """Register dashboard command."""
    p = subparsers.add_parser(
        "dashboard",
        help="Generate and open an HTML insights dashboard in the browser",
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
        help="Output file path (default: .evalyn/dashboard.html)",
    )
    p.set_defaults(func=cmd_dashboard)


__all__ = ["cmd_dashboard", "register_commands"]
