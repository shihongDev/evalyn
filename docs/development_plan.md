# Evalyn Python SDK — Development Plan

## Scope
- Python-only SDK (frontend removed): `@eval` decorator, tracer, storage, metrics (objective/subjective), judges, runner, suggester/selector, calibration, CLI.
- Goal: auto-instrument LLM functions, capture traces (including code metadata), suggest/select metrics, run evals, incorporate human labels, calibrate LLM judges, and support simulations later.

## Architecture
- **Instrumentation**: `@eval` (sync/async), session context, trace events, code metadata capture (source/signature/doc/hash/path).
- hooks/context managers for tool calls (future).
- **Storage**: pluggable; SQLite shipped; Parquet/export planned. Schemas for calls, runs, annotations.
- **Metrics**: registry with built-in objective metrics; subjective via judge pipeline; plugin API.
- **Runner**: dataset execution, caching, retries (future), aggregation, reporting.
- **Curator**: `curate_dataset` helper to run agents over prompts and build datasets with traces and optional JSONL export.
- **Suggester/Selector**: heuristic & LLM-based metric suggestion; LLM registry selector uses code + traces to pick from registry.
- **Judges**: abstraction; OpenAI judge adapter; multi-judge voting planned.
- **Human loop**: annotation import/export; calibration engine for thresholds; prompt tuning later.
- **Simulation**: synthetic sessions/datasets; adapters for agent frameworks (later).
- **CLI**: manage traces, runs, metric suggestion/selection, run datasets, import annotations, calibrate.

## Data Model (shipped)
- `FunctionCall`: id, function_name, inputs, output/error, timing, session_id, trace events, metadata (incl. code).
- `DatasetItem`: id, inputs, expected?, metadata.
- `MetricSpec` / `MetricResult`: id, type (objective/subjective), config; scores/pass/details/judge raw.
- `EvalRun`: id, dataset name, metric results, metrics, judge configs, summary.
- `Annotation`: id, target_id, label, rationale, annotator, source, confidence, created_at.
- `CalibrationRecord`: judge_config_id, gold set, adjustments, created_at.

## Execution Flow (current)
1) Instrument function with `@eval`; traces stored to SQLite.
2) Collect traces; list via CLI.
3) Suggest/select metrics (heuristic or LLM; registry selection uses code + traces).
4) Run dataset through target; apply metrics; store eval run.
5) Import human annotations; attach by call_id.
6) Calibrate subjective metric thresholds using annotations; prompt tuning later.

## Milestones (next)
- **M4** (enhance runner): retries, parallelism controls, richer caching policy.
- **M5** (subjective/judges): multi-judge voting, prompt templates, response parsers, cost tracking.
- **M6** (storage/export): Parquet/exporter, filtered queries (by session/function/time), CLI reports.
- **M7** (human+calibration): prompt-tuning loop using gold labels; versioned judge configs.
- **M8** (simulation/adapters): agent/tool adapters; synthetic session generator.
- **M9** (optional UI): small web viewer for runs/annotations.

## Testing Strategy
- Unit: decorator/tracer (sync/async), code metadata capture, storage round-trips, metric registry, objective metrics, selector logic.
- Integration: end-to-end dataset run with mocked judge; calibration over small gold set.
- Perf: stress SQLite with 10k traces; assess caching impact.

## Immediate Next Steps
- Add multi-judge voting and prompt templates for subjective metrics.
- Implement Parquet/export backend and query filters.
- Extend runner with retries/parallelism knobs and richer CLI reporting.
- Optional: lightweight UI for browsing runs/annotations once backend stabilizes.
