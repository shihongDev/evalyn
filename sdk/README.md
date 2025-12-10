# Evalyn Python SDK (work in progress)

This package provides the Python building blocks for Evalyn's automatic evaluation framework:
- `@eval` decorator to trace LLM-facing functions (sync + async)
- pluggable storage (SQLite included)
- metric registry with objective/subjective metrics (latency, exact match, substring, cost, BLEU, pass@k, tone/toxicity via judges)
- eval runner over datasets with optional caching
- metric suggester (heuristic + LLM pluggable)
- hooks for LLM judges, human annotation, and calibration
- automatic capture of function signature/doc/source (hashed) via `@eval` for registry-aware LLM metric selection

## Local development
1) Create a virtual environment in `sdk/`:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2) Install the package in editable mode:
   ```
   pip install -e ".[dev]"
   ```
3) Run the CLI help:
   ```
   evalyn --help
   ```

Optional LLM judge dependencies:
```
pip install -e ".[llm]"   # includes OpenAI + Google Gemini clients
```
OpenTelemetry spans are enabled by default if the dependency is installed:
```
pip install -e ".[otel]"
# env knobs:
#   EVALYN_OTEL=off            # disable spans
#   EVALYN_OTEL_EXPORTER=otlp  # use OTLP exporter instead of console
#   EVALYN_OTEL_ENDPOINT=...   # OTLP endpoint if needed
```

## Quick usage

```python
from evalyn_sdk import eval, EvalRunner, DatasetItem, latency_metric, exact_match_metric

@eval  # auto-traces inputs/outputs/errors
def handle(user_input: str) -> str:
    return f"echo:{user_input}"

dataset = [
    DatasetItem(id="1", inputs={"user_input": "hi"}, expected="echo:hi"),
    DatasetItem(id="2", inputs={"user_input": "hello"}, expected="echo:hello"),
]

runner = EvalRunner(
    target_fn=handle,
    metrics=[latency_metric(), exact_match_metric()],
    instrument=False,  # already decorated
)
run = runner.run_dataset(dataset)
print(run.summary)
```

### CLI example
Run the sentiment demo agent against the sample dataset:
```
evalyn run-dataset --target examples.agent:classify_sentiment --dataset examples/dataset.jsonl --dataset-name sentiment-demo
```

Run the deep-research mock agent (multi-step, richer traces):
```
evalyn run-dataset --target examples.research_agent:run_research --dataset examples/research_dataset.jsonl --dataset-name research-demo
```

Run the live Gemini-backed research agent (requires `GEMINI_API_KEY` and optional `GEMINI_MODEL`):
```
python examples/generate_research_live.py   # generates baseline dataset and runs eval
# or directly:
evalyn run-dataset --target examples.research_agent_live:run_research_live --dataset examples/research_live_dataset.jsonl
python examples/run_research_live_eval.py    # one-shot pipeline: curate dataset + eval summary
python -m example_agent.run_eval             # uses LangGraph agent in example_agent/, curates data via SDK + evals
```

Suggest metrics for a target (heuristic mode):
```
evalyn suggest-metrics --target examples.agent:classify_sentiment
```

Select metrics from the registry using an LLM (needs a caller that accepts a prompt and returns metric IDs):
```
evalyn select-metrics --target examples.agent:classify_sentiment --llm-caller mymodule:llm_call
```

Import human annotations and calibrate a judge metric:
```
evalyn import-annotations --path annotations.jsonl
evalyn calibrate --metric-id llm_judge --annotations annotations.jsonl --run-id <run-id-from-list>
```
