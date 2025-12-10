# Evalyn SDK Quick Start (Current State)

This repo has two parts:
- **Frontend demo** (React) that simulates the experience.
- **Python SDK** (`sdk/`) that provides `@eval`, tracing, storage, metrics, judges, runner, suggester, and CLI.

## What’s implemented now
- `@eval` decorator (sync/async) with trace capture: inputs, outputs/errors, duration, session id, trace events.
- Automatic capture of function metadata: signature, docstring, source (when available), hash, module/file path.
- Pluggable storage with SQLite backend (calls, runs, annotations).
- Metric system: registry with objective metrics (latency, exact match, substring, cost) and subjective judge metrics.
- LLM judges: OpenAI-backed judge (optional extra) + debug EchoJudge.
- Dataset runner with optional caching (hash by inputs) and summary stats.
- Metric suggestion/selection:
  - Heuristic suggester.
  - LLM suggester for new specs.
  - LLM registry selector that picks from available metrics using code + traces.
- Human loop: import/export annotations; calibration engine to suggest judge thresholds.
- CLI commands for dataset runs, listing calls/runs, metric suggestion/selection, annotation import, and calibration.

## Pipeline (end-to-end)
1) **Instrument**: add `@eval` to your LLM-facing function. Calls are traced to SQLite (`evalyn.sqlite`) by default, with code metadata captured for metric selection.
2) **Collect**: run your agent; traces are stored (`evalyn list-calls`).
3) **Suggest/Select Metrics**:
   - Heuristic/LLM suggestions: `evalyn suggest-metrics --target module:function`
   - LLM registry selection (uses code + traces): `evalyn select-metrics --target module:function --llm-caller mymodule:llm_call`
4) **Run Eval**: prepare JSON/JSONL dataset, then `evalyn run-dataset --target module:function --dataset path --dataset-name name` (uses built-in metrics; extend via registry).
5) **Annotate**: import human labels `evalyn import-annotations --path annotations.jsonl` (target_id should match call_id).
6) **Calibrate**: `evalyn calibrate --metric-id <judge-metric> --annotations annotations.jsonl --run-id <eval-run>` to get suggested threshold adjustments.

## Minimal code example
```python
from evalyn_sdk import eval, EvalRunner, DatasetItem, latency_metric, exact_match_metric

@eval
def handle(user_input: str) -> str:
    return f"echo:{user_input}"

dataset = [
    DatasetItem(id="1", inputs={"user_input": "hi"}, expected="echo:hi"),
]

runner = EvalRunner(target_fn=handle, metrics=[latency_metric(), exact_match_metric()], instrument=False)
run = runner.run_dataset(dataset)
print(run.summary)
```

## CLI cheatsheet
- `evalyn run-dataset --target examples.agent:classify_sentiment --dataset examples/dataset.jsonl`
- `evalyn list-calls`
- `evalyn list-runs`
- `evalyn suggest-metrics --target module:function`
- `evalyn select-metrics --target module:function --llm-caller mymodule:llm_call`
- `evalyn import-annotations --path annotations.jsonl`
- `evalyn calibrate --metric-id llm_judge --annotations annotations.jsonl --run-id <run>`

## Notes on cleanliness
- Git ignore covers node_modules/dist/__pycache__/venv/etc.
- Default SQLite file (`evalyn.sqlite`) is created lazily; delete it if you want a fresh slate.
