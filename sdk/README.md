# Evalyn Python SDK (work in progress)

This package provides the Python building blocks for Evalyn's automatic evaluation framework:
- `@eval` decorator to trace LLM-facing functions (sync + async)
- pluggable storage (SQLite included)
- metric registry with objective/subjective metrics
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
pip install -e ".[llm]"
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
Run the demo agent against the sample dataset:
```
evalyn run-dataset --target examples.agent:classify_sentiment --dataset examples/dataset.jsonl --dataset-name sentiment-demo
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
