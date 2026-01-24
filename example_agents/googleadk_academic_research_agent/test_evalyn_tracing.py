"""
Test script for Evalyn automatic callback injection with Google ADK.

Requires GOOGLE_API_KEY environment variable set for Gemini API access.
Do NOT set GOOGLE_GENAI_USE_VERTEXAI - we use direct API.

Usage:
    GOOGLE_API_KEY=your_key python test_evalyn_tracing.py
"""

import asyncio
import os
import sys

# Ensure we use direct Gemini API, not Vertex AI
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

# Add SDK to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk"))

from evalyn_sdk import eval
from google.adk.runners import InMemoryRunner
from academic_research.agent import root_agent


@eval
async def run_research(query: str):
    """Run academic research agent with Evalyn tracing."""
    runner = InMemoryRunner(agent=root_agent, app_name="evalyn_test")
    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=query,
    ):
        events.append(event)
        # Print event type for debugging
        event_type = type(event).__name__
        content = getattr(event, "content", None)
        if content:
            from academic_research.agent import root_agent
            # Extract text if available
            parts = getattr(content, "parts", []) or []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    print(f"[{event_type}] {part.text[:200]}...")
                    break
    return events[-1] if events else None


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Usage: GOOGLE_API_KEY=your_key python test_evalyn_tracing.py")
        sys.exit(1)

    print("Running academic research agent with Evalyn tracing...")
    print("-" * 60)

    result = await run_research("What is attention in transformers? Give a brief answer.")

    print("-" * 60)
    print("Done. Check Evalyn trace output for span hierarchy.")


if __name__ == "__main__":
    asyncio.run(main())
