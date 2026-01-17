# Evalyn Technical Manual

Internal technical reference for Evalyn SDK architecture, design decisions, and implementation details.

## Table of Contents

1. [Auto-Instrumentation](#auto-instrumentation)
2. [Tracing Architecture](#tracing-architecture)
3. [Storage Schema](#storage-schema)
4. [Metrics System](#metrics-system)
5. [Calibration Pipeline](#calibration-pipeline)
6. [Data Models](#data-models)
7. [Analysis & Visualization](#analysis--visualization)
8. [Environment Variables](#environment-variables)
9. [CLI Conveniences](#cli-conveniences)

---

## Auto-Instrumentation

### Overview

Evalyn automatically captures LLM calls by monkey-patching client libraries at import time.

```python
import evalyn_sdk  # Patches happen here
```

### Patched Libraries

| Library | Method Patched | Captured Data |
|---------|----------------|---------------|
| OpenAI | `openai.chat.completions.create` | tokens, cost, duration, request/response |
| Anthropic | `anthropic.messages.create` | tokens, cost, duration, request/response |
| Google Gemini | `genai.models.generate_content` | tokens, cost, duration, request/response |
| LangChain | Callback handler injection | LLM calls, tool calls |

### How Patching Works

```
┌─────────────────────────────────────────────────────────────┐
│  import evalyn_sdk                                          │
│                                                             │
│    1. Check if library is installed                         │
│    2. Store original method reference                       │
│    3. Replace with wrapper that:                            │
│       - Records start time                                  │
│       - Calls original method                               │
│       - Captures response (tokens, cost)                    │
│       - Logs to current trace session                       │
│       - Returns original response                           │
└─────────────────────────────────────────────────────────────┘
```

### What Gets Captured vs Not Captured

| Captured Automatically | NOT Captured (needs @trace) |
|------------------------|----------------------------|
| All LLM API calls | Custom functions |
| Token usage & cost | Tool execution logic |
| Request/response content | Agent loop structure |
| Call duration | Business logic between calls |
| Errors | LangGraph node transitions |

### LangChain Callback Handler

For LangChain, we inject `EvalynCallbackHandler` which implements:

```python
class EvalynCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs): ...
    def on_llm_end(self, response, **kwargs): ...
    def on_tool_start(self, serialized, input_str, **kwargs): ...
    def on_tool_end(self, output, **kwargs): ...
```

This captures LLM and tool calls but NOT chain/agent structure (would need `on_chain_start/end`).

### Disabling Auto-Instrumentation

```bash
export EVALYN_AUTO_INSTRUMENT=off
```

---

## Tracing Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                      @eval Decorator                         │
│  - Wraps function                                            │
│  - Creates session context                                   │
│  - Captures inputs/outputs/errors                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      EvalTracer                              │
│  - Manages trace sessions (contextvars)                      │
│  - Collects trace events                                     │
│  - Handles nested calls                                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    StorageBackend                            │
│  - SQLiteStorage (default)                                   │
│  - Persists FunctionCall objects                             │
└──────────────────────────────────────────────────────────────┘
```

### Session Management

Uses Python's `contextvars` for thread-safe session tracking:

```python
_current_session: ContextVar[Optional[TraceSession]] = ContextVar('evalyn_session')
```

This allows:
- Nested `@eval` calls to share context
- Auto-instrumented LLM calls to attach to parent session
- Thread-safe concurrent tracing

### Trace Event Types

| Event Type | Source | Data |
|------------|--------|------|
| `llm_call` | Auto-instrumentation | model, tokens, cost, duration |
| `tool_call` | LangChain callback | tool name, input, output |
| `trace` | `@trace` decorator | function name, args, result |
| `error` | Exception handler | error type, message, traceback |

### Function Metadata Captured

The `@eval` decorator captures:
- Function signature (parameters, types, return type)
- Docstring
- Source code (first 500 lines)
- Code hash (for change detection)

---

## Storage Schema

### SQLite Tables

```sql
-- Main trace storage
CREATE TABLE calls (
    id TEXT PRIMARY KEY,
    data JSON  -- FunctionCall serialized
);

-- Evaluation run results
CREATE TABLE eval_runs (
    id TEXT PRIMARY KEY,
    data JSON  -- EvalRun serialized
);

-- Human annotations
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    data JSON  -- Annotation serialized
);
```

All complex fields stored as JSON blobs for flexibility. No migrations needed.

### Default Location

```
./evalyn.sqlite  # Project root
```

Override with:
```python
from evalyn_sdk import configure
configure(storage_path="/custom/path/evalyn.sqlite")
```

---

## Metrics System

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MetricRegistry                           │
│  - Holds all Metric instances                               │
│  - Lookup by ID                                             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Objective Metrics     │     │   Subjective Metrics    │
│   (Deterministic)       │     │   (LLM Judge)           │
│                         │     │                         │
│   - latency_ms          │     │   - helpfulness         │
│   - token_count         │     │   - toxicity            │
│   - json_valid          │     │   - hallucination       │
│   - bleu, rouge         │     │   - coherence           │
└─────────────────────────┘     └─────────────────────────┘
```

### Metric Types

| Type | Execution | Examples |
|------|-----------|----------|
| **Objective** | Deterministic code | `latency_ms`, `bleu`, `json_valid` |
| **Subjective** | LLM judge call | `helpfulness_accuracy`, `toxicity_safety` |

### Metric Suggestion Modes

| Mode | Description | Output |
|------|-------------|--------|
| `basic` | Heuristic based on function signature | Objective + Subjective |
| `llm-registry` | LLM selects from 50+ templates | Objective + Subjective |
| `llm-brainstorm` | LLM generates custom metrics | **Subjective only** |
| `bundle` | Pre-configured sets | Objective + Subjective |

### Why Brainstorm is Subjective-Only

Custom objective metrics require code implementation (handlers). Custom subjective metrics work because:
1. LLM generates custom rubric
2. At eval time, generic LLM judge uses that rubric
3. No code needed - just prompt engineering

### Metric Scopes

| Scope | What It Evaluates |
|-------|-------------------|
| `overall` | Final output only |
| `llm_call` | Individual LLM API calls |
| `tool_call` | Tool executions |
| `trace` | Aggregates across trace (counts, ratios) |

### Reference-Based Metrics

These require `expected` field in dataset:
- `bleu`, `rouge_l`, `rouge_1`, `rouge_2`
- `exact_match`, `token_overlap_f1`, `jaccard_similarity`
- `numeric_mae`, `numeric_rmse`

Auto-excluded if dataset has no expected values.

---

## Calibration Pipeline

### Purpose

LLM judges aren't perfect. Calibration aligns them with human judgment.

### Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Run Eval  │ ──▶ │   Annotate  │ ──▶ │  Calibrate  │
│  (LLM judge)│     │   (Human)   │     │  (Optimize) │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  Improved Rubric    │
                                    │  + Alignment Metrics│
                                    └─────────────────────┘
```

### Alignment Metrics

| Metric | Description |
|--------|-------------|
| Precision | Of LLM's PASS predictions, how many human agreed |
| Recall | Of human PASS labels, how many LLM caught |
| F1 | Harmonic mean of precision/recall |
| Cohen's Kappa | Agreement accounting for chance |
| Accuracy | Overall agreement rate |

### Annotation Modes

| Mode | Description |
|------|-------------|
| Default | Overall pass/fail for entire output |
| `--per-metric` | Agree/disagree with each metric's LLM judgment |

### Prompt Optimization

`PromptOptimizer` analyzes disagreements and suggests rubric improvements:

```
Input: Original rubric + disagreement examples
Output: Improved rubric with clarified criteria
```

---

## Data Models

### Core Dataclasses

```python
@dataclass
class FunctionCall:
    id: str
    function_name: str
    inputs: Dict[str, Any]
    output: Any
    error: Optional[str]
    duration_ms: float
    timestamp: datetime
    session_id: str
    project: str
    version: str
    trace_events: List[TraceEvent]
    metadata: Dict[str, Any]  # signature, docstring, source, hash

@dataclass
class TraceEvent:
    type: str  # llm_call, tool_call, trace, error
    timestamp: datetime
    data: Dict[str, Any]

@dataclass
class DatasetItem:
    id: str
    inputs: Dict[str, Any]
    expected: Optional[Any]
    metadata: Dict[str, Any]

@dataclass
class MetricSpec:
    id: str
    name: str
    type: Literal["objective", "subjective"]
    description: str
    config: Dict[str, Any]  # rubric, thresholds, etc.

@dataclass
class MetricResult:
    metric_id: str
    score: float
    passed: bool
    details: Dict[str, Any]  # reason, raw_response, etc.

@dataclass
class EvalRun:
    id: str
    dataset_path: str
    results: List[MetricResult]
    summary: Dict[str, Any]  # per-metric aggregates
    timestamp: datetime

@dataclass
class Annotation:
    id: str
    item_id: str
    label: bool  # pass/fail
    confidence: int  # 1-5
    notes: str
    metric_labels: Dict[str, MetricLabel]  # per-metric mode

@dataclass
class MetricLabel:
    agree_with_llm: bool
    human_label: bool
    notes: str
```

---

## Analysis & Visualization

### Overview

The `evalyn analyze` command provides comprehensive analysis and visualization of eval run results.

### Features

| Feature | Description |
|---------|-------------|
| **Pass Rate Charts** | ASCII bar charts showing per-metric pass rates |
| **Score Statistics** | Avg, min, max, std deviation per metric |
| **Score Distributions** | Mini histograms showing score spread |
| **Failed Item Breakdown** | List of failed items with failing metrics |
| **Run Comparison** | Compare pass rates across multiple runs |
| **HTML Reports** | Interactive charts with Chart.js |

### ASCII Visualizations

The analyzer includes ASCII visualization helpers:

```
Pass Rate Bar:
  helpfulness_accuracy     ████████████████████░░░░░  80.0% (n=5)

Score Distribution (0.0 → 1.0):
  helpfulness_accuracy     [▂▁▁▁▆] avg=0.80
```

### Analysis Data Model

```python
@dataclass
class MetricStats:
    metric_id: str
    metric_type: str
    count: int
    passed: int
    failed: int
    scores: List[float]
    # Computed: pass_rate, avg_score, min_score, max_score, std_dev

@dataclass
class ItemStats:
    item_id: str
    metrics_passed: int
    metrics_failed: int
    metric_results: Dict[str, Tuple[bool, float]]

@dataclass
class RunAnalysis:
    run_id: str
    dataset_name: str
    created_at: str
    total_items: int
    total_metrics: int
    metric_stats: Dict[str, MetricStats]
    item_stats: Dict[str, ItemStats]
    failed_items: List[str]
    # Computed: overall_pass_rate
```

### CLI Usage

```bash
# Basic analysis (latest run)
evalyn analyze --latest

# Verbose with failed items
evalyn analyze --dataset data/myapp --verbose

# Compare multiple runs
evalyn analyze --dataset data/myapp --compare --num-runs 5

# Generate HTML report
evalyn analyze --dataset data/myapp --format html --output report.html
```

### HTML Report

The HTML report includes:
- Summary statistics cards
- Interactive bar chart (Chart.js)
- Detailed metrics table
- Color-coded pass/fail indicators

---

## File Structure

```
evalyn/
├── sdk/
│   └── evalyn_sdk/
│       ├── __init__.py          # Public API exports
│       ├── decorators.py        # @eval, @trace
│       ├── models.py            # Dataclasses
│       ├── datasets.py          # Dataset I/O
│       ├── runner.py            # EvalRunner
│       ├── cli_impl.py          # CLI command implementations
│       ├── analysis/            # Analysis & visualization module
│       │   ├── core.py          # RunAnalysis, MetricStats classes
│       │   ├── reports.py       # Text/ASCII reports
│       │   ├── html_report.py   # HTML dashboard generation
│       │   └── trends.py        # Trend analysis over time
│       ├── trace/
│       │   ├── tracer.py        # EvalTracer, session management
│       │   ├── context.py       # Context management
│       │   ├── auto_instrument.py # Monkey-patching logic
│       │   ├── langgraph.py     # LangGraph integration
│       │   └── otel.py          # OpenTelemetry support
│       ├── storage/
│       │   ├── base.py          # StorageBackend interface
│       │   └── sqlite.py        # SQLiteStorage
│       ├── metrics/
│       │   ├── objective.py     # 30 objective metric templates + handlers
│       │   ├── subjective.py    # 22 subjective metric definitions
│       │   ├── judges.py        # LLM judge implementations
│       │   ├── factory.py       # Metric builders
│       │   └── suggester.py     # Metric suggestion logic
│       ├── annotation/
│       │   ├── annotations.py   # Annotation models
│       │   ├── calibration.py   # Calibration engine
│       │   └── span_annotation.py # Span-level annotation
│       ├── simulation/
│       │   ├── simulator.py     # Synthetic data generation
│       │   └── simulation.py    # Simulation models
│       ├── cli/
│       │   ├── main.py          # CLI entry point
│       │   ├── commands/        # CLI command modules
│       │   └── utils/           # CLI utilities
│       └── utils/
│           └── api_client.py    # API client utilities
├── docs/
│   ├── technical-manual.md      # This file
│   └── clis/                    # CLI command documentation
│       ├── README.md
│       ├── one-click.md
│       ├── run-eval.md
│       └── ...                  # Other CLI docs
└── example_agent/               # Reference implementation
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVALYN_AUTO_INSTRUMENT` | `on` | Enable/disable auto-patching |
| `EVALYN_NO_HINTS` | `off` | Set to `1` or `true` to suppress CLI hint messages |
| `GEMINI_API_KEY` | - | Gemini API key for LLM judges |
| `OPENAI_API_KEY` | - | OpenAI API key (alternative) |
| `EVALYN_OTEL` | `off` | Enable OpenTelemetry spans |
| `EVALYN_OTEL_SERVICE` | `evalyn` | OTel service name |
| `EVALYN_OTEL_EXPORTER` | `sqlite` | OTel exporter type |

---

## CLI Conveniences

### Short IDs

All IDs in Evalyn are UUIDs, but commands accept 8-character prefixes for convenience:

```bash
# Full UUID
evalyn show-call --id fde2d07e-1234-5678-90ab-cdef12345678

# Short ID (first 8 chars) - works the same
evalyn show-call --id fde2d07e
```

The `list-calls` and `list-runs` commands display short IDs by default. If a short ID matches multiple records, you'll be prompted to use more characters.

### Quick Access Flags

Several commands support `--last` to quickly access the most recent record:

```bash
evalyn show-call --last    # Most recent trace
evalyn show-trace --last   # Most recent trace (span tree)
evalyn show-run --last     # Most recent eval run
```

### Suppressing Hints

Hint messages appear after commands to guide next steps. To suppress them:

```bash
# Per-command
evalyn list-calls --quiet

# Globally (environment variable)
export EVALYN_NO_HINTS=1
```

---

*Last updated: 2026-01-16*
