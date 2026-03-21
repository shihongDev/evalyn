"""Quickstart command: guided first-run experience for new users.

Walks the user through:
1. Detecting the agent framework (OpenAI, LangChain, CrewAI, Anthropic, ADK)
2. Generating an instrumentation snippet to paste into their agent
3. Creating evalyn.yaml with sensible defaults
4. Optionally running the agent to capture traces
5. Suggesting metrics based on captured traces
6. Printing next-step instructions

Usage:
    evalyn quickstart
    evalyn quickstart --agent-file agent.py
    evalyn quickstart --run "python agent.py"
    evalyn quickstart --run "python agent.py" --timeout 60
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.errors import fatal_error
from ..utils.hints import print_hint


# Framework detection patterns: framework name -> (compiled regex patterns, display name)
FRAMEWORK_PATTERNS: Dict[str, Tuple[List[re.Pattern], str]] = {
    "openai": (
        [re.compile(r"^import openai\b", re.MULTILINE), re.compile(r"^from openai\b", re.MULTILINE)],
        "OpenAI",
    ),
    "langchain": (
        [re.compile(r"^from langchain\b", re.MULTILINE), re.compile(r"^import langchain\b", re.MULTILINE)],
        "LangChain",
    ),
    "crewai": (
        [re.compile(r"^from crewai\b", re.MULTILINE), re.compile(r"^import crewai\b", re.MULTILINE)],
        "CrewAI",
    ),
    "anthropic": (
        [re.compile(r"^import anthropic\b", re.MULTILINE), re.compile(r"^from anthropic\b", re.MULTILINE)],
        "Anthropic",
    ),
    "google_adk": (
        [re.compile(r"^from google\.adk\b", re.MULTILINE), re.compile(r"^import google\.adk\b", re.MULTILINE)],
        "Google ADK",
    ),
}

# Instrumentation snippets per framework
INSTRUMENTATION_SNIPPETS: Dict[str, str] = {
    "openai": (
        "# Add this BEFORE your OpenAI import:\n"
        "import evalyn_sdk  # must come before 'import openai'\n"
        "import openai"
    ),
    "langchain": (
        "# Add this BEFORE your LangChain import:\n"
        "import evalyn_sdk  # must come before 'from langchain ...'\n"
        "from langchain ..."
    ),
    "crewai": (
        "# Add this BEFORE your CrewAI import:\n"
        "import evalyn_sdk  # must come before 'from crewai ...'\n"
        "from crewai ..."
    ),
    "anthropic": (
        "# Add this BEFORE your Anthropic import:\n"
        "import evalyn_sdk  # must come before 'import anthropic'\n"
        "import anthropic"
    ),
    "google_adk": (
        "# Add this BEFORE your ADK import:\n"
        "import evalyn_sdk  # must come before 'from google.adk ...'\n"
        "from google.adk ..."
    ),
    "other": (
        "# Add the evalyn trace decorator to your agent function:\n"
        "from evalyn_sdk import trace\n"
        "\n"
        "@trace\n"
        "def my_agent_function(...):\n"
        "    ..."
    ),
}


def _scan_file_for_frameworks(filepath: Path) -> List[str]:
    """Scan a Python file for known framework imports.

    Returns list of framework keys found.
    """
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            # Only read first 50 lines — imports are always at the top
            lines = []
            for i, line in enumerate(f):
                if i >= 50:
                    break
                lines.append(line)
            content = "".join(lines)
    except OSError:
        return []

    found = []
    for framework, (patterns, _display_name) in FRAMEWORK_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(content):
                found.append(framework)
                break
    return found


def _scan_directory_for_frameworks() -> Dict[str, List[str]]:
    """Scan *.py files in cwd and one level of src/ or app/ subdirectories.

    Returns dict mapping framework key -> list of file paths where detected.
    """
    results: Dict[str, List[str]] = {}
    scan_dirs = [Path(".")]
    for subdir in ["src", "app"]:
        p = Path(subdir)
        if p.is_dir():
            scan_dirs.append(p)

    for scan_dir in scan_dirs:
        for py_file in scan_dir.glob("*.py"):
            frameworks = _scan_file_for_frameworks(py_file)
            for fw in frameworks:
                results.setdefault(fw, []).append(str(py_file))

    return results


def _detect_framework(agent_file: Optional[str]) -> Tuple[str, Optional[str]]:
    """Detect agent framework, returning (framework_key, agent_file_path).

    If agent_file is provided, scans only that file.
    Otherwise scans the directory. If multiple frameworks found, asks user to pick.
    If none found, asks user to specify.

    Returns:
        Tuple of (framework_key, agent_file_path or None)
    """
    if agent_file:
        filepath = Path(agent_file)
        if not filepath.exists():
            fatal_error(
                f"Agent file not found: {agent_file}",
                "Check the path and try again",
            )
        frameworks = _scan_file_for_frameworks(filepath)
        if len(frameworks) == 1:
            display = FRAMEWORK_PATTERNS[frameworks[0]][1]
            print(f"  Detected: {display} in {agent_file}")
            return frameworks[0], agent_file
        elif len(frameworks) > 1:
            return _ask_framework_selection(frameworks), agent_file
        else:
            return _ask_framework_selection(), agent_file

    # Scan directory
    detected = _scan_directory_for_frameworks()
    if len(detected) == 1:
        fw = list(detected.keys())[0]
        files = detected[fw]
        display = FRAMEWORK_PATTERNS[fw][1]
        print(f"  Detected: {display} client in {files[0]}")
        return fw, files[0]
    elif len(detected) > 1:
        print("\n  Multiple frameworks detected:")
        for fw, files in detected.items():
            display = FRAMEWORK_PATTERNS[fw][1]
            print(f"    - {display} in {', '.join(files)}")
        return _ask_framework_selection(list(detected.keys())), None
    else:
        print("\n  No known agent framework detected in *.py files.")
        return _ask_framework_selection(), None


def _ask_framework_selection(frameworks: Optional[List[str]] = None) -> str:
    """Ask the user to pick a framework. If frameworks is None, shows all known frameworks."""
    if frameworks is None:
        frameworks = list(FRAMEWORK_PATTERNS.keys())
    print("\n  Which framework is your agent using?")
    for i, fw in enumerate(frameworks, 1):
        display = FRAMEWORK_PATTERNS[fw][1]
        print(f"    {i}. {display}")
    print(f"    {len(frameworks) + 1}. Other (raw HTTP / custom)")

    try:
        choice = input("\n  Select (number): ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(frameworks):
            return frameworks[idx]
        elif idx == len(frameworks):
            return "other"
        else:
            print("  Invalid selection, defaulting to 'other'.")
            return "other"
    except (ValueError, EOFError):
        print("  Invalid input, defaulting to 'other'.")
        return "other"


def _check_instrumentation_present(agent_file: Optional[str]) -> bool:
    """Check if evalyn_sdk import is already present in the agent file."""
    if not agent_file:
        return False
    filepath = Path(agent_file)
    if not filepath.exists():
        return False
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(
        re.search(r"^import evalyn_sdk\b", content, re.MULTILINE)
        or re.search(r"^from evalyn_sdk\b", content, re.MULTILINE)
    )


def _print_instrumentation_snippet(framework: str) -> None:
    """Print the instrumentation snippet for the given framework."""
    snippet = INSTRUMENTATION_SNIPPETS.get(framework, INSTRUMENTATION_SNIPPETS["other"])
    print("\n  Paste this into your agent's entry point:\n")
    print("  " + "-" * 50)
    for line in snippet.split("\n"):
        print(f"    {line}")
    print("  " + "-" * 50)
    if framework != "other":
        print(
            "\n  Important: import evalyn_sdk MUST come before the framework import."
        )


def _create_evalyn_yaml() -> None:
    """Create evalyn.yaml using the same logic as cmd_init."""
    output_path = Path("evalyn.yaml")
    if output_path.exists():
        print(f"\n  evalyn.yaml already exists, skipping creation.")
        return

    # Find the example file - check multiple locations
    example_paths = [
        Path("evalyn.yaml.example"),
        Path(__file__).parent.parent.parent.parent / "evalyn.yaml.example",
    ]

    example_path = None
    for p in example_paths:
        if p.exists():
            example_path = p
            break

    if example_path:
        shutil.copy(example_path, output_path)
        print(f"\n  Created evalyn.yaml (from {example_path})")
    else:
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
        print(f"\n  Created evalyn.yaml (minimal config)")


def _run_agent(run_cmd: str, timeout: int) -> bool:
    """Run the user's agent as a subprocess.

    Returns True if the agent exited successfully.
    """
    print(f"\n  Running: {run_cmd}")
    print(f"  Timeout: {timeout}s\n")

    try:
        result = subprocess.run(
            run_cmd,
            shell=True,
            timeout=timeout,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"\n  Agent exited with code {result.returncode}.")
            print("  Check your agent independently before retrying.")
            print(
                "  Hint: Make sure 'import evalyn_sdk' is at the top of your agent file."
            )
            return False
        print("\n  Agent completed successfully.")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  Agent timed out after {timeout}s.")
        print("  You can increase the timeout with --timeout <seconds>.")
        return False
    except FileNotFoundError:
        print(f"\n  Command not found: {run_cmd}")
        print("  Check that the command is correct and Python is on your PATH.")
        return False


def _check_traces_captured() -> int:
    """Check how many traces were captured. Returns count."""
    try:
        from ...storage import SQLiteStorage

        storage = SQLiteStorage()
        calls = storage.list_calls(limit=10)
        return len(calls)
    except Exception:
        return 0


def _suggest_metrics_hint() -> None:
    """Print a hint about suggesting metrics."""
    print("\n  Suggested next steps:")
    print("    1. evalyn build-dataset --project <name>   # Build dataset from traces")
    print("    2. evalyn suggest-metrics --dataset <path>  # Pick evaluation metrics")
    print("    3. evalyn run-eval --dataset <path>         # Run evaluation")
    print(
        "\n  Or run the full pipeline at once:"
    )
    print("    evalyn one-click --project <name>")


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Interactive quickstart for first-time users.

    Steps:
    1. Detect agent framework
    2. Generate instrumentation snippet
    3. Create evalyn.yaml
    4. Optionally run the agent
    5. Suggest metrics
    6. Print next steps
    """
    quiet = getattr(args, "quiet", False)
    agent_file = getattr(args, "agent_file", None)
    run_cmd = getattr(args, "run", None)
    timeout = getattr(args, "timeout", 120)

    print("\n" + "=" * 60)
    print("  EVALYN QUICKSTART")
    print("=" * 60)

    # Check idempotency: if evalyn.yaml exists and instrumentation present, skip setup
    yaml_exists = Path("evalyn.yaml").exists()
    instrumentation_present = _check_instrumentation_present(agent_file)

    if yaml_exists and instrumentation_present:
        print("\n  Setup already complete (evalyn.yaml exists, instrumentation found).")
        print("  Skipping to run step...\n")
    else:
        # Step 1: Detect framework
        print("\n[1/3] Detecting agent framework...")
        framework, detected_file = _detect_framework(agent_file)

        # Step 2: Generate instrumentation snippet
        print("\n[2/3] Instrumentation snippet")
        _print_instrumentation_snippet(framework)

        # Step 3: Create evalyn.yaml
        print("\n[3/3] Setting up configuration...")
        _create_evalyn_yaml()

    # Step 4: Run the agent (optional)
    if run_cmd:
        print("\n" + "-" * 60)
        print("  RUNNING AGENT")
        print("-" * 60)

        traces_before = _check_traces_captured()
        success = _run_agent(run_cmd, timeout)

        if success:
            traces_after = _check_traces_captured()
            new_traces = traces_after - traces_before
            if new_traces > 0:
                print(f"\n  Captured {new_traces} new trace(s).")
                _suggest_metrics_hint()
            else:
                print("\n  Agent ran but no new traces were captured.")
                print("  Troubleshooting:")
                print("    - Ensure 'import evalyn_sdk' is at the TOP of your agent file")
                print("      (before the framework import)")
                print("    - Ensure your agent makes at least one LLM call")
                print("    - Check that evalyn_sdk is installed: pip show evalyn-sdk")
    else:
        print("\n" + "-" * 60)
        print("  NEXT: Run your agent to capture traces")
        print("-" * 60)
        print("\n  Run your agent with evalyn_sdk imported, then come back:")
        print("    python your_agent.py")
        print("\n  Or re-run quickstart with --run:")
        print('    evalyn quickstart --run "python your_agent.py"')

    # Final message
    print("\n" + "=" * 60)
    print("  Ready! Run `evalyn run-eval` to evaluate.")
    print("=" * 60 + "\n")

    if not quiet:
        print_hint(
            "Use `evalyn workflow` to see the full evaluation pipeline.",
            quiet=quiet,
        )


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the quickstart command."""
    quickstart_parser = subparsers.add_parser(
        "quickstart",
        help="Guided setup for first-time users (detect framework, instrument, configure)",
    )
    quickstart_parser.add_argument(
        "--agent-file",
        help="Path to your agent Python file (skips auto-detection scan)",
    )
    quickstart_parser.add_argument(
        "--run",
        help='Command to run your agent (e.g. "python agent.py")',
    )
    quickstart_parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for agent run (default: 120)",
    )
    quickstart_parser.set_defaults(func=cmd_quickstart)


__all__ = ["cmd_quickstart", "register_commands"]
