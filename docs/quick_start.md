# Evalyn SDK Quick Start

This repo focuses on the **Python SDK** (`sdk/`). The former frontend demo has been removed.

## What’s implemented now
- `@eval` decorator (sync/async) with trace capture: inputs, outputs/errors, duration, session id, trace events.
- Automatic capture of function metadata: signature, docstring, source (when available), hash, module/file path.
- Pluggable storage with SQLite backend (calls, runs, annotations).
- Metric system: registry with objective metrics (latency, exact match, substring, cost, BLEU, pass@k) and subjective judge metrics (tone/toxicity templates).
- LLM judges: OpenAI-backed judge (optional extra) + debug EchoJudge.
- Dataset runner with optional caching (hash by inputs) and summary stats.
- OpenTelemetry spans: enabled by default if the otel dependency is installed; disable with `EVALYN_OTEL=off` or configure exporter via env.
- Metric suggestion/selection:
  - Heuristic suggester.
  - LLM suggester for new specs.
  - LLM registry selector that picks from available metrics using code + traces.
- Human loop: import/export annotations; calibration engine to suggest judge thresholds.
- CLI commands for dataset runs, listing calls/runs, metric suggestion/selection, annotation import, and calibration.

## Pipeline (end-to-end)
1) **Instrument**: add `@eval` to your LLM-facing function. Calls are traced to SQLite (`evalyn.sqlite`) by default, with code metadata captured for metric selection.
2) **Collect**: run your agent; traces are stored (`evalyn list-calls`).
3) **Curate dataset with tracing**: use `evalyn_sdk.curate_dataset` to run prompts through the agent, capture traces, and optionally write JSONL for regression.
4) **Suggest/Select Metrics**:
   - Heuristic/LLM suggestions: `evalyn suggest-metrics --target module:function`
   - LLM registry selection (uses code + traces): `evalyn select-metrics --target module:function --llm-caller mymodule:llm_call`
5) **Run Eval**: prepare JSON/JSONL dataset, then `evalyn run-dataset --target module:function --dataset path --dataset-name name` (uses built-in metrics; extend via registry).
6) **Annotate**: import human labels `evalyn import-annotations --path annotations.jsonl` (target_id should match call_id).
7) **Calibrate**: `evalyn calibrate --metric-id <judge-metric> --annotations annotations.jsonl --run-id <eval-run>` to get suggested threshold adjustments.

### Visual flow
```
@eval-wrapped function
      |
      v
 Traces (inputs/outputs/errors/code metadata, events)
      |
      +--> Build dataset
      |     - from calls (dataset_from_calls)
      |     - or curate_dataset over prompts
      |
      +--> Metric selection
      |     - suggest-metrics / select-metrics
      |
      +--> Eval setup & run
      |     - run-dataset (EvalRunner/CLI)
      |
      +--> Inspect results
      |     - show-call / show-run
      |
      +--> Annotate & calibrate
            - import annotations / calibrate judges
```

## Minimal code example
```python
import json
from pathlib import Path
from evalyn_sdk import eval, EvalRunner, dataset_from_calls, get_default_tracer
from evalyn_sdk.metrics.objective import latency_metric, exact_match_metric

@eval
def handle(user_input: str) -> str:
    return f"echo:{user_input}"

# 1) Run the function a few times (traces are captured automatically)
handle("hi")
handle("hello")

# 2) Build a dataset from successful past calls (no pre-created dataset needed)
tr = get_default_tracer()
dataset = dataset_from_calls(tr.storage.list_calls(), use_only_success=True)
# Optionally save to JSONL for reuse
Path("trace_dataset.jsonl").write_text("\n".join(json.dumps(item.__dict__) for item in dataset))

runner = EvalRunner(target_fn=handle, metrics=[latency_metric(), exact_match_metric()], instrument=False)
run = runner.run_dataset(dataset)
print(run.summary)
```

## CLI cheatsheet (pipeline order)
- Instrument & run agent: `python -m example_agent.agent "your question"` (requires `GEMINI_API_KEY`)
- Inspect traces: `evalyn list-calls --limit 20`, then `evalyn show-call --id <call_id>`
- Build dataset from past calls (successful only): use `dataset_from_calls(tracer.storage.list_calls())` in a short script and save as JSONL
- Run eval on a dataset: `evalyn run-dataset --target module:function --dataset path --dataset-name name`
- Suggest metrics: `evalyn suggest-metrics --target module:function`
- LLM registry selection: `evalyn select-metrics --target module:function --llm-caller mymodule:llm_call`
- View eval runs/results: `evalyn list-runs`, `evalyn show-run --id <run_id>`
- Import annotations: `evalyn import-annotations --path annotations.jsonl`
- Calibrate judges: `evalyn calibrate --metric-id llm_judge --annotations annotations.jsonl --run-id <run>`

## Notes on cleanliness
- `.gitignore` covers node_modules/dist/__pycache__/venv/etc.
- Default SQLite file (`evalyn.sqlite`) is created lazily; delete it if you want a fresh slate.
