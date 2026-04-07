#!/usr/bin/env python3
"""
Test runner script for evalyn_sdk tests.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --fast       # Skip slow tests
    python tests/run_tests.py --imports    # Only import tests
    python tests/run_tests.py --cli        # Only CLI tests
    python tests/run_tests.py --verbose    # Verbose output
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Determine test path
    test_path = Path(__file__).parent

    if "--imports" in args:
        cmd.append(str(test_path / "test_imports.py"))
        args.remove("--imports")
    elif "--cli" in args:
        cmd.extend([
            str(test_path / "test_cli.py"),
            str(test_path / "test_cli_integration.py"),
        ])
        args.remove("--cli")
    elif "--tracing" in args:
        cmd.append(str(test_path / "test_tracing.py"))
        args.remove("--tracing")
    elif "--metrics" in args:
        cmd.append(str(test_path / "test_metrics.py"))
        args.remove("--metrics")
    else:
        cmd.append(str(test_path))

    # Handle fast mode
    if "--fast" in args:
        cmd.extend(["-m", "not slow"])
        args.remove("--fast")

    # Handle verbose
    if "--verbose" in args or "-v" in args:
        cmd.append("-v")
        if "--verbose" in args:
            args.remove("--verbose")

    # Add remaining args
    cmd.extend(args)

    # Add default options
    if "-v" not in cmd:
        cmd.append("-v")
    cmd.append("--tb=short")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
