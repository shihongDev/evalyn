#!/usr/bin/env python3
"""
Setup script to create test traces for one-click CLI testing.

Run this before running test_all_cli.sh to populate the test database
with sample traces.

Usage:
    python tests/fixtures/oneclick/setup_test_traces.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add SDK to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SDK_ROOT = PROJECT_ROOT / "sdk"
sys.path.insert(0, str(SDK_ROOT))

# Set test database
TEST_DB = PROJECT_ROOT / "data" / "test" / "traces.sqlite"
TEST_DB.parent.mkdir(parents=True, exist_ok=True)
os.environ["EVALYN_DB"] = str(TEST_DB)


def create_test_traces():
    """Create sample traces for one-click testing."""
    from evalyn_sdk import eval
    from evalyn_sdk.decorators import get_default_tracer

    # Test project: oneclick-test
    @eval(project="oneclick-test", version="v1")
    def sample_research_agent(query: str) -> str:
        """A sample research agent for testing.

        This function simulates a research agent that answers questions
        by searching for information and synthesizing a response.
        """
        # Simulate processing
        return f"Based on my research, {query.lower()} is an important topic. Here are the key findings: 1) First point 2) Second point 3) Third point."

    # Generate test traces
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the benefits of Python?",
        "Explain the concept of APIs",
        "What is cloud computing?",
    ]

    print(f"Creating {len(test_queries)} test traces for project 'oneclick-test'...")

    for query in test_queries:
        try:
            result = sample_research_agent(query)
            print(f"  [OK] {query[:40]}...")
        except Exception as e:
            print(f"  [FAIL] {query[:40]}... - {e}")

    # Flush traces
    tracer = get_default_tracer()
    if tracer.storage:
        print(f"\nTraces saved to: {tracer.storage.path}")

        # Verify
        calls = tracer.storage.list_calls(limit=10)
        oneclick_calls = [
            c for c in calls
            if (c.metadata or {}).get("project_name") == "oneclick-test"
            or (c.metadata or {}).get("project_id") == "oneclick-test"
        ]
        print(f"Verified: {len(oneclick_calls)} traces for 'oneclick-test'")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Setting up test traces for one-click CLI testing")
    print("=" * 60)
    print(f"\nTest database: {os.environ.get('EVALYN_DB')}")
    print()

    create_test_traces()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now run: ./tests/test_all_cli.sh")


if __name__ == "__main__":
    main()
