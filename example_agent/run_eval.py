"""
End-to-end Evalyn pipeline for the LangGraph research agent in this folder.
- Uses @eval-wrapped `run_agent` to trace calls.
- Curates a dataset by actually running the agent on example prompts.
- Runs EvalRunner with basic metrics and prints a summary.

Requirements:
- GEMINI_API_KEY (and any other env vars your agent/graph needs).
- pip install -e ".[llm]" from sdk/ for Gemini/OpenAI clients if required by graph config.
"""

import json
from pathlib import Path

from evalyn_sdk import EvalRunner, curate_dataset, latency_metric
from evalyn_sdk.metrics.objective import substring_metric, bleu_metric

from example_agent.agent import run_agent


PROMPTS = [
    "What are effective defenses against prompt injection in LLM agents?",
    "How to improve retrieval quality for a vector database under latency constraints?",
    "Design safety rails for an autonomous research agent.",
]

DATASET_PATH = Path(__file__).parent / "agent_dataset.jsonl"


def main() -> None:
    items = curate_dataset(
        target_fn=run_agent,
        prompts=PROMPTS,
        expected_strategy="as_output",  # use current output as baseline
        build_metadata=lambda prompt, _: {"expected_substring": prompt.split()[0]},
        session_id="agent-eval",
        store_path=DATASET_PATH,
    )
    print(f"Curated {len(items)} items and wrote to {DATASET_PATH}")

    runner = EvalRunner(
        target_fn=run_agent,
        metrics=[latency_metric(), substring_metric(), bleu_metric()],
        dataset_name="agent-eval",
        instrument=False,  # already wrapped with @eval
    )
    run = runner.run_dataset(items)
    print("Eval summary:", json.dumps(run.summary, indent=2))


if __name__ == "__main__":
    main()
