import argparse
import os
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

from evalyn_sdk import eval, get_default_tracer, configure_otel
from example_agent.graph import graph


# Optional: enable OpenTelemetry spans if OTEL_CONSOLE=true is set.
if os.getenv("OTEL_CONSOLE", "").lower() in {"1", "true", "yes"}:
    tracer = get_default_tracer()
    tracer.attach_otel_tracer(
        configure_otel(service_name="example-agent", exporter="console")
    )


@eval(project="gemini-deep-research-agent", version="v1", name="research_agent")
def run_agent(
    question: str,
    initial_queries: int = 3,
    max_loops: int = 2,
    reasoning_model: str = "gemini-2.5-flash-lite",
) -> str:
    """
    Main LLM-facing entrypoint for the research agent. Wrapped with @eval so calls are traced/stored.
    """
    state = {
        "messages": [HumanMessage(content=question)],
        "initial_search_query_count": initial_queries,
        "max_research_loops": max_loops,
        "reasoning_model": reasoning_model,
    }
    result = graph.invoke(state)
    messages = result.get("messages", [])
    return messages[-1].content if messages else ""


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--reasoning-model",
        default="gemini-2.5-flash-lite",
        help="Model for the final answer",
    )
    args = parser.parse_args()

    answer = run_agent(
        question=args.question,
        initial_queries=args.initial_queries,
        max_loops=args.max_loops,
        reasoning_model=args.reasoning_model,
    )
    if answer:
        print(answer)


if __name__ == "__main__":
    main()
