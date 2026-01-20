# Evalyn Technical Manual

Internal technical reference for Evalyn SDK architecture, design decisions, and implementation details.

## Table of Contents

1. [Auto-Instrumentation](#auto-instrumentation)
2. [Tracing Architecture](#tracing-architecture)
3. [Storage Schema](#storage-schema)
4. [Metrics System](#metrics-system)
5. [Calibration Pipeline](#calibration-pipeline)
6. [Data Models](#data-models)
7. [Execution Strategies](#execution-strategies)
8. [Pipeline Orchestration](#pipeline-orchestration)
9. [Analysis & Visualization](#analysis--visualization)
10. [File Structure](#file-structure)
11. [Environment Variables](#environment-variables)
12. [CLI Conveniences](#cli-conveniences)

---

## Auto-Instrumentation

### Overview

Evalyn automatically captures LLM calls by instrumenting client libraries **lazily** when the first trace starts (not at import time). This keeps CLI commands fast.

```python
import evalyn_sdk
# Instrumentation happens when the first @eval function is called, not here
```

### Supported SDKs

| SDK | Instrumentation Type | Captured Data |
|-----|---------------------|---------------|
| OpenAI | Monkey-patch | tokens, cost, duration, request/response |
| Anthropic Client | Monkey-patch | tokens, cost, duration, request/response |
| Anthropic Agent SDK | Hook-based | agent turns, tool calls, handoffs |
| Google Gemini | Monkey-patch | tokens, cost, duration, request/response |
| Google ADK | OTEL-native | agent spans via SpanProcessor |
| LangChain | Callback handler | LLM calls, tool calls |
| LangGraph | Monkey-patch | graph/node execution spans |

### Instrumentation Types

The instrumentation registry supports three strategies:

| Type | Description | SDKs |
|------|-------------|------|
| `MONKEY_PATCH` | Wrap SDK methods directly | OpenAI, Anthropic, Gemini |
| `OTEL_NATIVE` | Use SDK's built-in OTEL with custom SpanProcessor | Google ADK |
| `HOOK_BASED` | Use SDK's hook/callback system | Anthropic Agent SDK |

### How Instrumentation Works

```
┌─────────────────────────────────────────────────────────────┐
│  First @eval function call                                  │
│                                                             │
│    1. InstrumentorRegistry.ensure_instrumented()            │
│    2. For each registered instrumentor:                     │
│       a. Check if SDK is installed (is_available)           │
│       b. Apply instrumentation strategy:                    │
│          - MONKEY_PATCH: Wrap methods                       │
│          - OTEL_NATIVE: Add SpanProcessor                   │
│          - HOOK_BASED: Register callbacks                   │
│    3. Instrumented calls log to current trace session       │
└─────────────────────────────────────────────────────────────┘
```

### What Gets Captured vs Not Captured

| Captured Automatically | NOT Captured (needs @trace) |
|------------------------|----------------------------|
| All LLM API calls | Custom functions |
| Token usage & cost | Tool execution logic |
| Request/response content | Agent loop structure |
| Call duration | Business logic between calls |
| Errors | - |

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
data/prod/traces.sqlite   # Production traces
data/test/traces.sqlite   # Test traces (when EVALYN_ENV=test)
```

Override with:
```python
from evalyn_sdk import configure
configure(storage_path="/custom/path/traces.sqlite")
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
│   - bleu, rouge, etc.   │     │   - coherence, etc.     │
│   (73 metrics total)    │     │   (60 metrics total)    │
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
| `llm-registry` | LLM selects from 130+ templates | Objective + Subjective |
| `llm-brainstorm` | LLM generates custom metrics | **Subjective only** |
| `bundle` | Pre-configured sets (17 bundles) | Objective + Subjective |

### Metric Bundles

17 curated bundles for common GenAI use cases:

| Category | Bundles |
|----------|---------|
| **Conversational AI** | `chatbot`, `customer-support` |
| **Content Generation** | `content-writer`, `summarization`, `creative-writer` |
| **Knowledge & Research** | `rag-qa`, `research-agent`, `tutor` |
| **Code & Technical** | `code-assistant`, `data-extraction` |
| **Agents & Orchestration** | `orchestrator`, `multi-step-agent` |
| **High-Stakes Domains** | `medical-advisor`, `legal-assistant`, `financial-advisor` |
| **Safety & Translation** | `moderator`, `translator` |

Bundle design principles:
1. Start with safety metrics for user-facing applications
2. Include efficiency metrics (latency) for production monitoring
3. Add domain-specific quality metrics based on use case
4. Keep bundles focused (8-12 metrics) to balance coverage and evaluation cost

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

### Multi-Provider Support

LLM judges can use different providers via the `--provider` flag:

| Provider | Client Class | Default Model | Logprobs Support |
|----------|--------------|---------------|------------------|
| `gemini` | `GeminiClient` | gemini-2.5-flash-lite | No |
| `openai` | `OpenAIClient` | gpt-4o-mini | Yes |
| `ollama` | `OllamaClient` | llama3.2 | Limited |

Provider selection happens at judge creation time. The `LLMJudge` class uses lazy initialization for the API client.

### Confidence Estimation

Evalyn supports multiple methods for estimating judge confidence:

| Method | Implementation | Accuracy | Cost |
|--------|----------------|----------|------|
| `consistency` | Run judge N times with temp=0.7, measure agreement | Medium | N API calls |
| `logprobs` | Use token log probabilities | High | 1 API call |

**Logprobs Confidence Calculation**:
```
confidence = exp(mean(token_logprobs))
```
Higher logprobs = more confident the model is about each token.

**Self-Consistency Confidence Calculation**:
```
confidence = max(pass_count, fail_count) / total_samples
```
Higher agreement across samples = higher confidence.

The confidence module (`evalyn_sdk/confidence/`) provides:
- `LogprobsConfidence`: Token probability-based (OpenAI/Ollama only)
- `DeepConfConfidence`: Meta AI's DeepConf with bottom-10%/tail strategies (OpenAI only)
- `SelfConsistencyConfidence`: Multi-sample agreement
- `MajorityVoteConfidence`: Weighted voting
- `PerplexityConfidence`: Perplexity-based
- `EntropyConfidence`: Entropy from top-k logprobs

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

## Execution Strategies

### Overview

The evaluation runner uses a Strategy pattern to support different execution modes. Strategies are pluggable and handle how metric evaluation is parallelized and checkpointed.

### Available Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SequentialStrategy` | Simple for-loop with per-item checkpointing | Debugging, small datasets |
| `ParallelStrategy` | ThreadPoolExecutor with batch checkpointing | Production, large datasets |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EvalRunner                              │
│  - Prepares (item, call) tuples                             │
│  - Delegates execution to strategy                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ExecutionStrategy (ABC)                     │
│  - execute(prepared, metrics, progress_cb, run_id, done)    │
│  - Handles checkpointing via checkpoint_fn                  │
└─────────────────────────────────────────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   SequentialStrategy    │   │      ParallelStrategy       │
│   - For-loop            │   │   - ThreadPoolExecutor      │
│   - Per-item checkpoint │   │   - Batch checkpoint        │
└─────────────────────────┘   └─────────────────────────────┘
```

### Checkpointing

Both strategies support automatic checkpointing for resume capability:

- **Sequential**: Checkpoints after every N items (configurable)
- **Parallel**: Checkpoints after each batch completes

Checkpoint data includes: completed item IDs, partial results, run metadata.

---

## Pipeline Orchestration

### Overview

The `one-click` command uses a pipeline orchestrator to coordinate multi-step evaluation workflows. The pipeline supports resume, state persistence, and step-level error handling.

### Pipeline Steps

| Step | Description |
|------|-------------|
| `create-dataset` | Initialize dataset from traces or existing data |
| `suggest-metrics` | Generate metrics using selected mode (basic, llm, bundle) |
| `review-metrics` | Interactive metric review and editing |
| `run-eval` | Execute evaluation with selected metrics |
| `analyze` | Generate analysis and visualizations |
| `annotate` | Human annotation interface (optional) |
| `calibrate` | Prompt optimization from annotations (optional) |

### State Management

Pipeline state is persisted to `{output_dir}/pipeline_state.json`:

```python
@dataclass
class PipelineState:
    started_at: str
    config: Dict[str, Any]      # Original CLI args
    steps: Dict[str, Dict]      # Per-step status and outputs
    output_dir: str
    updated_at: Optional[str]
    completed_at: Optional[str]
```

### Resume Capability

```bash
# Initial run (interrupted)
evalyn one-click --dataset data/myapp

# Resume from last successful step
evalyn one-click --dataset data/myapp --resume
```

The orchestrator:
1. Loads existing state from `pipeline_state.json`
2. Skips completed steps
3. Resumes from the first incomplete/failed step

### Step Results

Each step returns a `StepResult`:

```python
@dataclass
class StepResult:
    status: str      # "success", "skipped", "failed", "interrupted"
    output: Optional[str]
    details: Dict[str, Any]
    error: Optional[str]
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
│       ├── runner.py            # EvalRunner (uses ExecutionStrategy)
│       ├── execution.py         # Execution strategies (Sequential/Parallel)
│       ├── analysis/            # Analysis & visualization module
│       │   ├── core.py          # RunAnalysis, MetricStats classes
│       │   ├── reports.py       # Text/ASCII reports
│       │   ├── html_report.py   # HTML dashboard generation
│       │   └── trends.py        # Trend analysis over time
│       ├── trace/
│       │   ├── tracer.py        # EvalTracer, session management
│       │   ├── context.py       # Context management
│       │   ├── auto_instrument.py # Backward-compat wrapper
│       │   ├── otel.py          # OpenTelemetry support
│       │   └── instrumentation/ # SDK instrumentation
│       │       ├── registry.py  # InstrumentorRegistry
│       │       ├── base.py      # Instrumentor base class
│       │       └── providers/   # Per-SDK instrumentors
│       │           ├── openai.py
│       │           ├── anthropic.py
│       │           ├── anthropic_agents.py
│       │           ├── gemini.py
│       │           ├── google_adk.py
│       │           ├── langchain.py
│       │           └── langgraph.py
│       ├── storage/
│       │   ├── base.py          # StorageBackend interface
│       │   └── sqlite.py        # SQLiteStorage
│       ├── metrics/
│       │   ├── objective.py     # 73 objective metric templates + handlers
│       │   ├── subjective.py    # 60 subjective metric definitions
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
│       ├── confidence/
│       │   ├── base.py          # ConfidenceEstimator ABC
│       │   ├── logprobs.py      # Logprobs, perplexity, entropy
│       │   ├── consistency.py   # Self-consistency, majority vote
│       │   └── verbalized.py    # Extract self-reported confidence
│       ├── cli/
│       │   ├── main.py          # CLI entry point
│       │   ├── commands/        # CLI command modules
│       │   │   ├── analysis.py
│       │   │   ├── annotation.py
│       │   │   ├── calibration.py
│       │   │   ├── dataset.py
│       │   │   ├── evaluation.py
│       │   │   ├── export.py
│       │   │   ├── infrastructure.py  # one-click command
│       │   │   ├── runs.py
│       │   │   ├── simulate.py
│       │   │   └── traces.py
│       │   └── utils/           # CLI utilities
│       │       ├── config.py         # Config file handling
│       │       ├── dataset_resolver.py # Dataset path resolution
│       │       ├── dataset_utils.py  # Dataset loading helpers
│       │       ├── errors.py         # CLI error handling
│       │       ├── formatters.py     # Output formatters
│       │       ├── hints.py          # Post-command hints
│       │       ├── input_helpers.py  # User input helpers
│       │       ├── loaders.py        # File loaders
│       │       ├── pipeline.py       # Pipeline orchestration
│       │       ├── pipeline_steps.py # Pipeline step implementations
│       │       ├── ui.py             # UI helpers
│       │       └── validation.py     # Input validation
│       └── utils/
│           └── api_client.py    # API clients (Gemini, OpenAI, Ollama)
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

*Last updated: 2026-01-19*
