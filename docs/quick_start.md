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
- `evalyn run-dataset --target examples.research_agent:run_research --dataset examples/research_dataset.jsonl`
- `python examples/generate_research_live.py` (requires `GEMINI_API_KEY`; runs live Gemini-backed agent and creates baseline dataset)
- `python examples/run_research_live_eval.py` (requires `GEMINI_API_KEY`; curates live dataset via SDK + runs eval with summary)
- `python -m example_agent.run_eval` (uses the LangGraph agent in `example_agent/`, curates data via SDK + runs eval)
- `python -m example_agent.agent "your question"` (requires `GEMINI_API_KEY`; @eval instrumented agent)
- `evalyn list-calls --limit 20` (recent traces)
- `evalyn show-call --id <call_id>` (full trace details, inputs/outputs/events)
- `evalyn show-run --id <run_id>` (per-metric results for an eval run)
- `evalyn list-calls`
- `evalyn list-runs`
- `evalyn suggest-metrics --target module:function`
- `evalyn select-metrics --target module:function --llm-caller mymodule:llm_call`
- `evalyn import-annotations --path annotations.jsonl`
- `evalyn calibrate --metric-id llm_judge --annotations annotations.jsonl --run-id <run>`

## Notes on cleanliness
- `.gitignore` covers node_modules/dist/__pycache__/venv/etc.
- Default SQLite file (`evalyn.sqlite`) is created lazily; delete it if you want a fresh slate.
