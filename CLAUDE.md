# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evalyn is a Python SDK for instrumenting, tracing, and evaluating LLM agents. It captures function calls automatically via the `@eval` decorator, stores traces in SQLite, and runs metrics (both objective and subjective/LLM-judge) against datasets.

## Development Setup

### Installation

```bash
# Install SDK in development mode with all dependencies
cd sdk
pip install -e ".[dev,llm,otel,agent]"
```

### Running the Example Agent

```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Run agent (traces are captured automatically)
python -m example_agent.agent "What are the latest developments in AI?"
```

## Key Commands

### CLI Usage

The SDK provides an `evalyn` CLI command. All commands support file paths (e.g., `example_agent/agent.py:run_agent`) or module imports (e.g., `example_agent.agent:run_agent`).

```bash
# View captured traces
evalyn list-calls --limit 20
evalyn show-call --id <call_id>

# View project summaries
evalyn show-projects

# Build dataset from traces
evalyn build-dataset --project myproj --version v1 --limit 200

# Suggest metrics (basic heuristic mode - fast, no LLM)
evalyn suggest-metrics --target example_agent/agent.py:run_agent --num-traces 5 --mode basic

# Suggest metrics (LLM-powered selection from 50+ metric registry using Gemini API)
evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode llm-registry --llm-mode api --model gemini-2.0-flash-exp

# Suggest metrics (LLM brainstorm - generates custom metrics)
evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode llm-brainstorm --llm-mode api --model gpt-4

# Suggest metrics using local Ollama
evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode llm-registry --llm-mode local --model llama3.1

# Use pre-configured bundle (no LLM needed)
evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode bundle --bundle research-agent

# Save suggested metrics to dataset folder
evalyn suggest-metrics --dataset data/myproj-v1-20250101 --target example_agent/agent.py:run_agent --mode llm-registry --llm-mode api --model gemini-2.0-flash-exp --metrics-name llm-selected

# Run evaluation on dataset using specified metrics (no re-running needed!)
evalyn run-eval --dataset data/myproj-v1/dataset.jsonl --metrics data/myproj-v1/metrics/llm-registry.json

# List available metric templates
evalyn list-metrics

# View eval runs
evalyn list-runs
evalyn show-run --id <run_id>

# Import human annotations
evalyn import-annotations --path annotations.jsonl

# Calibrate LLM judges
evalyn calibrate --metric-id <metric_id> --annotations annotations.jsonl --run-id <run_id>
```

### Testing

```bash
# Run tests (when available)
cd sdk
pytest
pytest -v  # verbose mode
```

## Architecture

### Core Components

**Tracing Layer** (`tracing.py`, `decorators.py`):
- `@eval` decorator instruments functions for automatic tracing
- Uses `contextvars` for session management and nested call tracking
- `EvalTracer` orchestrates call capture and storage
- Captures: inputs, outputs, errors, duration, trace events, function metadata (signature, docstring, source code, hash)
- Optional OpenTelemetry integration via `otel_tracer` attachment

**Storage** (`storage/`):
- Abstract `StorageBackend` interface
- `SQLiteStorage` implementation (default: `evalyn.sqlite` in project root)
- Stores: `FunctionCall` objects, `EvalRun` results, `Annotation` records
- No migrations needed; schema is created lazily

**Metrics System** (`metrics/`):
- **Registry** (`registry.py`): `MetricRegistry` holds `Metric` instances, each binding a `MetricSpec` to a handler function
- **Templates** (`templates.py`): 50+ pre-defined metric templates in `OBJECTIVE_TEMPLATES` and `SUBJECTIVE_TEMPLATES`
- **Objective Metrics** (`objective.py`): Deterministic metrics (latency, BLEU, token overlap, JSON validation, tool counts)
- **Subjective Metrics** (`subjective.py`): LLM-judge metrics (toxicity, tone, hallucination, clarity, instruction following)
- **Judges** (`judges.py`): `GeminiJudge` (default), `OpenAIJudge`, `EchoJudge` (debug)
- **Factory** (`factory.py`): `build_objective_metric()` and `build_subjective_metric()` construct metrics from template IDs
- **Suggester** (`suggester.py`):
  - `HeuristicSuggester`: Fast, offline metric suggestions based on function signature and traces
  - `LLMSuggester`: Free-form metric brainstorming via LLM
  - `LLMRegistrySelector`: LLM picks metrics from the registry using function code + traces

**Dataset Management** (`datasets.py`):
- `load_dataset()` / `save_dataset()`: JSON/JSONL I/O
- `dataset_from_calls()`: Convert traced calls into regression datasets
- `build_dataset_from_storage()`: CLI-friendly builder with filtering (project, version, time range)
- Default dataset layout: `data/<project>-<version>-<timestamp>/` with `dataset.jsonl`, `meta.json`, and `metrics/` subdirectory

**Evaluation Runner** (`runner.py`):
- `EvalRunner`: Executes datasets against target functions
- Automatic caching by input hash (optional)
- Applies all metrics to each call
- Generates summary statistics (avg, min, max, pass rates)
- Stores results as `EvalRun` objects

**CLI** (`cli.py`):
- Unified command-line interface for all SDK operations
- Target loading: Supports both file paths (`path/to/file.py:func`) and module imports (`module:func`)
- Metric bundles: Pre-configured metric sets (`summarization`, `orchestrator`, `research-agent`)

### Data Models (`models.py`)

Core dataclasses:
- `FunctionCall`: Captured trace with inputs, output, error, duration, session_id, trace events, metadata
- `TraceEvent`: Timestamped event logged during execution
- `DatasetItem`: Test case with inputs, expected output, metadata
- `MetricSpec`: Metric definition (id, name, type, description, config)
- `MetricResult`: Evaluation result (metric_id, score, passed, details)
- `EvalRun`: Collection of metric results with summary stats
- `Annotation`: Human labels for calibration
- `CalibrationRecord`: Judge threshold adjustments

### Evaluation Pipeline Flow

1. **Instrument**: Add `@eval` decorator to function → captures traces to SQLite
2. **Collect**: Run function(s) → traces accumulate in storage
3. **Build Dataset**: Export traces to JSONL (`evalyn build-dataset`)
4. **Select Metrics**: Use heuristic/LLM/bundle modes (`evalyn suggest-metrics`)
5. **Run Evaluation**: Execute dataset with metrics (`evalyn run-dataset`)
6. **Annotate**: Import human labels (`evalyn import-annotations`)
7. **Calibrate**: Adjust judge thresholds (`evalyn calibrate`)

### Metric Selection Strategies

The `evalyn suggest-metrics` command has two separate configuration dimensions:

**1. Selection Strategy (`--mode`)**:
- `basic` (or `auto`): Fast heuristic suggestions based on function signature and trace inspection (no LLM)
- `llm-registry`: LLM chooses from 50+ metric templates using function code + sample traces
- `llm-brainstorm`: LLM generates custom metric specs (not constrained to registry)
- `bundle`: Pre-configured sets (summarization, orchestrator, research-agent)

**2. LLM Caller Mode (`--llm-mode`)** - only needed when using `llm-registry` or `llm-brainstorm`:
- `api`: Use OpenAI or Gemini API (requires `OPENAI_API_KEY` or `GEMINI_API_KEY` env var)
- `local`: Use local Ollama (requires `ollama` binary installed)

**Model Selection (`--model`)** - only needed with LLM modes:
- Gemini: `gemini-2.0-flash-exp`, `gemini-2.5-flash-lite`, `gemini-1.5-pro`
- OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Ollama: `llama3.1`, `mistral`, etc.

**Example combinations**:
```bash
# Fast heuristic (no API key needed)
evalyn suggest-metrics --target app.py:func --mode basic

# LLM selection with Gemini
evalyn suggest-metrics --target app.py:func --mode llm-registry --llm-mode api --model gemini-2.0-flash-exp

# LLM brainstorm with OpenAI
evalyn suggest-metrics --target app.py:func --mode llm-brainstorm --llm-mode api --model gpt-4

# Local Ollama
evalyn suggest-metrics --target app.py:func --mode llm-registry --llm-mode local --model llama3.1

# Pre-configured bundle
evalyn suggest-metrics --target app.py:func --mode bundle --bundle research-agent
```

Users can annotate functions with preferences:
```python
@eval(metric_mode="llm-registry", metric_bundle="summarization", project="myproj", version="v1")
def my_agent(...):
    ...
```

### Storage Schema

SQLite tables (implicit schema, created lazily):
- `calls`: FunctionCall records (JSON blobs)
- `eval_runs`: EvalRun records (JSON blobs)
- `annotations`: Annotation records (JSON blobs)

The schema is flexible; all complex fields are stored as JSON.

## Important Patterns

### Using the `@eval` Decorator

```python
from evalyn_sdk import eval

@eval(project="myproj", version="v1", name="custom_name")
def my_function(input_text: str) -> str:
    return process(input_text)
```

- Supports both sync and async functions
- Automatically captures function metadata (signature, docstring, source, hash)
- Sessions: Use `eval_session()` context manager to group calls

### Loading Functions in CLI

The CLI accepts two target formats:
1. **File path**: `example_agent/agent.py:run_agent` (preferred, no PYTHONPATH needed)
2. **Module import**: `example_agent.agent:run_agent` (fallback)

Both work on Bash/WSL and PowerShell (use backslashes on Windows).

### Metric Template Structure

Templates define:
- `id`: Unique identifier
- `type`: "objective" or "subjective"
- `description`: Human-readable explanation
- `category`: efficiency, correctness, structure, robustness, quality
- `inputs`: Required fields (e.g., `["output", "expected"]`, `["call.duration_ms"]`, `["trace"]`)
- `config`: Default configuration (overridable)

Objective metrics compute scores deterministically. Subjective metrics call LLM judges and return `passed` (bool) + `reason` (str).

### Dataset Folder Structure

```
data/<project>-<version>-<timestamp>/
  dataset.jsonl           # Dataset items
  meta.json               # Metadata (filters, schema, counts, metric_sets)
  metrics/
    llm-registry-<timestamp>.json   # LLM-selected metrics
    bundle-summarization.json       # Bundle metrics
```

`meta.json` includes `active_metric_set` pointing to the most recent selection.

### OpenTelemetry Integration

OTel is optional and controlled by environment variables:
- `EVALYN_OTEL=off`: Disable OTel spans
- `EVALYN_OTEL_SERVICE=myservice`: Set service name
- `EVALYN_OTEL_EXPORTER=console|sqlite`: Set exporter type
- `EVALYN_OTEL_ENDPOINT=http://...`: Set OTLP endpoint

Default: SQLite exporter if OTel dependencies installed.

## Files to Ignore

- `evalyn.sqlite`: Local trace database (auto-generated, not committed)
- `data/`: Dataset folder (gitignored)
- `.venv/`: Virtual environment
- `*.egg-info/`: Package metadata
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`

## Example Agent Structure

The `example_agent/` directory demonstrates LangGraph integration:
- `agent.py`: Entry point with `@eval`-decorated `run_agent()` function
- `graph.py`: LangGraph state machine definition
- `state.py`: Agent state schema
- `tools_and_schemas.py`: Tool definitions
- `prompts.py`: System prompts
- `configuration.py`: Configuration objects

This is a reference implementation showing how to instrument a multi-step agent.
