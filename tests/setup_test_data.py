#!/usr/bin/env python3
"""
Setup script to generate test data from existing traces.

Usage:
    python tests/setup_test_data.py              # Generate from existing traces
    python tests/setup_test_data.py --clean      # Clean and regenerate
    python tests/setup_test_data.py --check      # Check if test data exists
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SDK_ROOT))

PROJECT_ROOT = SDK_ROOT.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test-output"
FIXTURES_DIR = TEST_DATA_DIR / "fixtures"


def check_test_data() -> bool:
    """Check if test data exists."""
    sample_dataset = FIXTURES_DIR / "sample-dataset"
    if sample_dataset.exists():
        dataset_file = sample_dataset / "dataset.jsonl"
        if dataset_file.exists():
            with open(dataset_file) as f:
                count = sum(1 for _ in f)
            print(f"Found sample dataset with {count} items")
            return True
    print("No sample dataset found")
    return False


def clean_test_data():
    """Clean all test data."""
    print("Cleaning test data...")

    # Clean temporary directories
    for path in TEST_DATA_DIR.glob("tmp_*"):
        if path.is_dir():
            shutil.rmtree(path)
            print(f"  Removed {path.name}")

    # Clean CLI integration data
    cli_int = TEST_DATA_DIR / "cli_integration"
    if cli_int.exists():
        shutil.rmtree(cli_int)
        print("  Removed cli_integration/")

    # Clean fixtures
    if FIXTURES_DIR.exists():
        shutil.rmtree(FIXTURES_DIR)
        print("  Removed fixtures/")

    print("Done")


def generate_test_data():
    """Generate test data from existing traces."""
    from evalyn_sdk.storage.sqlite import SQLiteStorage
    from evalyn_sdk.datasets import build_dataset_from_storage

    print("Generating test data from traces...")

    # Ensure directories exist
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Get storage
    storage = SQLiteStorage()

    # Check for existing traces
    calls = storage.list_calls(limit=100)
    if not calls:
        print("No traces found in database. Run the example agent first:")
        print("  export GEMINI_API_KEY='your-key'")
        print("  python -m example_agent.agent 'Your query here'")
        return False

    print(f"Found {len(calls)} traces")

    # Group by project
    projects = {}
    for call in calls:
        project = call.metadata.get("project", "unknown")
        if project not in projects:
            projects[project] = []
        projects[project].append(call)

    print(f"Projects: {list(projects.keys())}")

    # Generate sample dataset from the project with most traces
    best_project = max(projects.keys(), key=lambda p: len(projects[p]))
    best_calls = projects[best_project]

    print(f"Using project '{best_project}' with {len(best_calls)} traces")

    # Build dataset
    sample_dataset = FIXTURES_DIR / "sample-dataset"
    sample_dataset.mkdir(parents=True, exist_ok=True)

    # Create dataset.jsonl
    dataset_file = sample_dataset / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for i, call in enumerate(best_calls[:20]):  # Limit to 20 items
            item = {
                "id": f"test-item-{i}",
                "inputs": call.inputs,
                "output": call.output,
                "metadata": {
                    "call_id": call.id,
                    "function_name": call.function_name,
                    "duration_ms": call.duration_ms,
                    "project": call.metadata.get("project"),
                    "version": call.metadata.get("version"),
                },
            }
            f.write(json.dumps(item) + "\n")

    # Create meta.json
    meta = {
        "created_at": str(best_calls[0].started_at) if best_calls else None,
        "project": best_project,
        "version": best_calls[0].metadata.get("version", "v1") if best_calls else "v1",
        "item_count": min(len(best_calls), 20),
        "source": "generated from traces",
    }

    meta_file = sample_dataset / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # Create metrics directory
    metrics_dir = sample_dataset / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Create basic metrics
    basic_metrics = [
        {"id": "output_nonempty", "type": "objective", "name": "Non-Empty Output", "description": "Check output is not empty"},
        {"id": "latency_ms", "type": "objective", "name": "Latency", "description": "Measure execution latency"},
        {"id": "json_valid", "type": "objective", "name": "JSON Valid", "description": "Check if output is valid JSON"},
    ]

    with open(metrics_dir / "basic.json", "w") as f:
        json.dump(basic_metrics, f, indent=2)

    print(f"Created sample dataset at {sample_dataset}")
    print(f"  - {min(len(best_calls), 20)} items in dataset.jsonl")
    print(f"  - Basic metrics in metrics/basic.json")

    return True


def main():
    parser = argparse.ArgumentParser(description="Setup test data for evalyn tests")
    parser.add_argument("--clean", action="store_true", help="Clean all test data")
    parser.add_argument("--check", action="store_true", help="Check if test data exists")
    args = parser.parse_args()

    if args.clean:
        clean_test_data()
        if not args.check:
            generate_test_data()
    elif args.check:
        sys.exit(0 if check_test_data() else 1)
    else:
        if not check_test_data():
            generate_test_data()
        else:
            print("Test data already exists. Use --clean to regenerate.")


if __name__ == "__main__":
    main()
