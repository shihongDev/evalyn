# Evalyn SDK Design Overview

## Goals
- Make evaluating LLM agents easy: add `@eval`, capture traces (with code metadata), run metrics, support subjective judges, human annotations, calibration, and simulations.
- Provide observability via OpenTelemetry (OTel) spans so you can send traces to external backends while keeping Evalyn's own eval storage.

## Layers
- **Instrumentation (`@eval`)**: wraps sync/async functions, captures inputs/outputs/errors/duration, code metadata, session id, and trace events. Writes to storage (SQLite). Emits OTel spans if OTel is installed/enabled.
- **Storage**: pluggable; default SQLite keeps `function_calls`, `eval_runs`, `annotations`. CLI: `list-calls`, `show-call`, `list-runs`, `show-run`.
- **Metrics**: registry with objective (latency, exact match, substring, cost, BLEU, pass@k) and subjective (judge-driven) metrics. Suggester/selector can propose metrics using code/traces.
- **Runner**: executes datasets, applies metrics, caches outputs, summarizes results.
- **Judges & Calibration**: LLM judges for subjective metrics, calibration against human annotations.
- **Curation**: `curate_dataset` to build datasets by running instrumented functions over prompts, storing traces.
- **CLI**: manage traces/runs, suggest/select metrics, run datasets, import annotations, calibrate judges, inspect details.
- **Observability (OTel)**: Evalyn auto-attaches an OTel tracer (if dependency installed) and can propagate spans/events. You can also instrument SDK clients (e.g., Gemini) to emit child spans/events.

## Default Workflow
1) **Instrument**: add `@eval` to your LLM-facing function. OTel is required (install extra) so spans are emitted by default.
2) **Run & Capture**: calls are stored in SQLite with metadata; OTel spans go to console/OTLP based on env (`EVALYN_OTEL_EXPORTER`, `EVALYN_OTEL_ENDPOINT`, `EVALYN_OTEL_SERVICE`).
3) **Curate**: use `curate_dataset` to run prompts, build dataset items, and write JSONL if needed.
4) **Suggest/Select Metrics**: `evalyn suggest-metrics`, or LLM-based selection.
5) **Eval**: `evalyn run-dataset ...` or programmatic `EvalRunner`.
6) **Inspect**: `evalyn show-call --id ...`, `evalyn show-run --id ...` for detailed per-call/per-metric views.
7) **Annotate/Calibrate**: import annotations, calibrate judges.

## OTel Integration
- Evalyn uses OTel to emit spans for any `@eval`-wrapped function. If OTel isn't installed, `get_default_tracer` will raise and instruct to install `pip install -e "sdk[otel]"`.
- Exporters: console, OTLP, or sqlite (`EVALYN_OTEL_EXPORTER=sqlite`) to persist spans locally (table `otel_spans` keyed by `evalyn.call_id`).
- You can instrument SDK clients (e.g., Gemini) to log events and create child spans; see `example_agent/graph.py` for Gemini wrappers.
- `evalyn show-call` displays the Evalyn trace (inputs/outputs/metadata/events). OTel spans contain the same call context for observability, but Evalyn remains the source of eval data.

## Example Agent Notes
- `example_agent/agent.py` is wrapped with `@eval`, runs the LangGraph research agent, and logs Gemini calls via events/spans.
- `python -m example_agent.agent "your question"` (requires `GEMINI_API_KEY`) to run and capture traces.
- `python -m example_agent.run_eval` curates a dataset and shows a dashboard.
- Inspect traces/runs: `evalyn list-calls`, `evalyn show-call --id <call_id>`, `evalyn list-runs`, `evalyn show-run --id <run_id>`.

## Diagrams

### Eval Workflow (high level)
```
User function @eval
      |
      v
 EvalTracer captures inputs/outputs/errors + code metadata
      |
      +--> Storage (SQLite: function_calls, eval_runs, annotations)
      |
      +--> Metrics/Runner apply metrics over datasets
      |
      +--> CLI (list-calls/show-call, list-runs/show-run)
      |
      v
 Reports (dashboards, summaries, calibration)
```

### OTel Integration
```
@eval-wrapped function
      |
      +--> Evalyn trace + events (stored in SQLite)
      +--> OTel span (via configured tracer provider)   <-- built into @eval, no extra decorator
                |
                +--> Exporter (console or OTLP) -> observability backend (e.g., Arize/Grafana)

Instrumented SDKs/tools (e.g., Gemini generate_content)
      |
      +--> tracer.log_event(...) + child OTel spans
```
