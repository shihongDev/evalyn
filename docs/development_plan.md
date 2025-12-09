# Evalyn Python SDK — Development Plan

## Purpose
- Ship a Python SDK that auto-instruments LLM-facing functions with `@eval`, captures traces at multiple levels (call/session/tool), and runs end-to-end evaluations with LLM judges, objective metrics, human annotations, calibration, and agent simulations.
- Deliver composable building blocks (decorators, tracer, storage, metric registry, runner, judges, calibration loop) plus a CLI for operating the system.

## High-Level Architecture
- **Instrumentation Layer**: `@eval` decorator (sync + async), context-aware tracer, trace event model, event bus hooks for tool calls, optional manual context manager for frameworks we cannot patch.
- **Storage Abstraction**: pluggable backends (ship SQLite + Parquet later); schema for calls, sessions, metrics, eval runs, annotations, calibration artifacts.
- **Metrics Engine**: registry for objective and subjective metrics; built-in objective templates; LLM-judge pipeline for subjective scoring; aggregation utilities.
- **Eval Runner**: dataset orchestrator (parallelism, retries, caching), metric application, summary reports.
- **Human-in-the-Loop**: export/import for annotations; mark gold labels; feed calibration loop.
- **Calibration**: compare judge outputs vs. gold, auto prompt-tune + hyperparam search (temp/top_p), track prompt versions.
- **Simulation Harness**: generate synthetic sessions/tasks, stress scenarios for tools/agents; can populate datasets.
- **CLI**: commands to trace, suggest metrics, run evals, manage annotations, calibrate judges, and launch simulations.

## Data Model (initial cut)
- `FunctionCall`: id, function_name, inputs, output/error, started_at/ended_at, duration_ms, session_id, trace events (tool calls/logs), metadata.
- `Session`: session_id, created_at, labels/metadata.
- `DatasetItem`: id, inputs (dict), expected (optional), tags/metadata.
- `MetricSpec`: id, name, type (`objective|subjective`), config, prompt/code template, owners.
- `MetricResult`: metric_id, item_id, score/value, passed flag, explanation/details, raw_judge (if subjective).
- `EvalRun`: id, dataset ref, metrics applied, judge configs, item-level results, summary stats.
- `Annotation`: id, target (call/result), label, rationale, annotator/source, confidence, created_at.
- `JudgeConfig`: model, prompt, parameters, version, calibration metadata.
- `CalibrationRecord`: judge_config_id, gold set used, adjustments, before/after metrics.

## Module Breakdown (planned layout)
- `evalyn_sdk/__init__.py`: exports decorator, tracer, runner, registry, judges.
- `evalyn_sdk/models.py`: dataclasses for the data model + JSON (de)serialization helpers.
- `evalyn_sdk/tracing.py`: `EvalTracer`, contextvars-based session management, trace event recording, helpers to log tool calls/events.
- `evalyn_sdk/decorators.py`: `@eval` decorator handling sync/async, correlation ids, error capture, storage flush.
- `evalyn_sdk/storage/base.py`: `StorageBackend` protocol.
- `evalyn_sdk/storage/sqlite.py`: default local backend; JSON columns; simple indexes; list/query helpers.
- `evalyn_sdk/storage/parquet.py` (later): fast columnar export/import.
- `evalyn_sdk/metrics/registry.py`: registry for metric specs + runtime functions; plugin API.
- `evalyn_sdk/metrics/objective.py`: built-ins (exact match, regex, JSON schema validation, latency, cost).
- `evalyn_sdk/metrics/subjective.py`: LLM-judge orchestration, multi-judge voting, prompt templates.
- `evalyn_sdk/judges.py`: judge client abstraction, adapters for OpenAI/Anthropic/Gemini.
- `evalyn_sdk/runner.py`: dataset executor, caching, retries, aggregation, reporting.
- `evalyn_sdk/annotations.py`: import/export of human labels, attachment to results.
- `evalyn_sdk/calibration.py`: prompt tuning/calibration loop using human gold.
- `evalyn_sdk/simulation.py`: agent simulation utilities for generating traces/datasets.
- `sdk/cli.py`: argparse/typer CLI entrypoint mapping to runner/calibration/simulation.

## Execution Flow (target)
1) **Instrument**: user adds `@eval` to LLM/tool entry functions; tracer records calls + events into storage with session ids.
2) **Collect**: traces accumulate; optionally grouped by `eval_session(...)` context.
3) **Suggest Metrics**: LLM suggester reviews sample traces/function signature → returns metric specs (objective vs subjective).
4) **Configure Metrics**: objective metrics bind to code templates; subjective metrics bind to judge prompts/models.
5) **Run Eval**: dataset runner replays dataset through target function (or uses cached outputs), applies metrics, stores `EvalRun`.
6) **Annotate**: export results; collect human labels; mark as gold.
7) **Calibrate**: compare judge vs. gold; auto-adjust prompts/params; version judge configs.
8) **Simulate**: generate synthetic sessions/tasks to enrich datasets and probe edge cases.

## Milestones
- **M1 (Tracer + Storage)**: decorator (sync/async), context/session handling, SQLite backend, trace models.
- **M2 (Metrics Core)**: registry, objective built-ins, metric application over recorded calls, report summary.
- **M3 (Subjective + Judges)**: judge abstraction, prompt templates, multi-judge voting, caching of LLM calls.
- **M4 (Runner + CLI)**: dataset runner with retries/cache/parallelism; CLI commands for run/view/export.
- **M5 (Human/Calibration)**: annotation import/export, calibration loop with prompt tuning, prompt versioning.
- **M6 (Simulation + Adapters)**: agent/tool adapters, simulation harness, Parquet exporter, documentation + examples.

## Testing Strategy
- Unit: decorator/tracer correctness (args/kwargs capture, errors, async), storage round-trips, metric registry, built-in metrics.
- Integration: end-to-end eval run over toy dataset with mocked LLM judge; calibration loop over small gold set.
- Load/Perf: stress SQLite backend with 10k traces; parallel dataset run.
- Tooling: provide mocks/fakes for judge clients to keep tests offline and deterministic.

## Immediate Next Steps
- Scaffold SDK package + pyproject.
- Implement tracer/decorator + SQLite storage minimum viable path.
- Add metric registry with initial objective metrics and a thin eval runner.
- Wire a simple CLI stub for running a dataset with built-in metrics.
