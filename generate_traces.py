"""Generate 100 traces by running the example agent with different questions in parallel."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add sdk to path
sys.path.insert(0, str(Path(__file__).parent / "sdk"))

# Sample questions for the research agent
QUESTIONS = [
    "What is quantum computing and how does it differ from classical computing?",
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of climate change?",
    "How does machine learning work?",
    "What is the history of the internet?",
    "Explain how vaccines work.",
    "What are black holes and how are they formed?",
    "How does the human brain process information?",
    "What is blockchain technology?",
    "Explain the water cycle.",
    "What are renewable energy sources?",
    "How do electric vehicles work?",
    "What is artificial intelligence?",
    "Explain photosynthesis.",
    "What is the Big Bang theory?",
    "How does DNA replication work?",
    "What are the effects of deforestation?",
    "Explain how airplanes fly.",
    "What is dark matter?",
    "How do antibiotics work?",
    "What is the greenhouse effect?",
    "Explain the concept of infinity.",
    "How does the stock market work?",
    "What is CRISPR gene editing?",
    "Explain the Doppler effect.",
    "What causes earthquakes?",
    "How do computers store data?",
    "What is nuclear fusion?",
    "Explain how the immune system works.",
    "What is string theory?",
    "How do solar panels work?",
    "What is cryptocurrency mining?",
    "Explain the scientific method.",
    "How do neural networks learn?",
    "What is the ozone layer?",
    "Explain how GPS works.",
    "What are stem cells?",
    "How does 5G technology work?",
    "What is quantum entanglement?",
    "Explain the carbon cycle.",
    "How do submarines work?",
    "What is machine vision?",
    "Explain how batteries work.",
    "What is the Higgs boson?",
    "How does radar work?",
    "What are gravitational waves?",
    "Explain how memory works in computers.",
    "What is gene therapy?",
    "How do touchscreens work?",
    "What is antimatter?",
]


def run_single_query(idx: int, question: str) -> tuple[int, str, bool]:
    """Run a single query and return result."""
    from example_agent.agent import run_agent

    try:
        # Use minimal settings for faster execution
        result = run_agent(
            question=question,
            initial_queries=1,
            max_loops=1,
            reasoning_model="gemini-2.5-flash-lite",
        )
        print(f"[{idx+1:03d}] OK: {question[:50]}...", flush=True)
        return (idx, question, True)
    except Exception as e:
        print(f"[{idx+1:03d}] ERROR: {question[:50]}... - {e}", flush=True)
        return (idx, question, False)


def main():
    num_traces = 100
    max_workers = 10  # Parallel workers

    # Expand questions to 100 by cycling
    questions = []
    for i in range(num_traces):
        questions.append(QUESTIONS[i % len(QUESTIONS)])

    print(f"Generating {num_traces} traces with {max_workers} parallel workers...", flush=True)
    print("-" * 60, flush=True)

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_query, i, q): i for i, q in enumerate(questions)
        }

        for future in as_completed(futures):
            idx, question, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1

    print("-" * 60, flush=True)
    print(f"Done. Success: {success}, Failed: {failed}", flush=True)
    print(f"Check traces with: evalyn list-calls --limit 100", flush=True)


if __name__ == "__main__":
    main()
