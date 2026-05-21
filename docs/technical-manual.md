# Evalyn Technical Manual

Internal technical reference for Evalyn SDK architecture, design decisions, and implementation details.

## Table of Contents

1. [Terminology](#terminology)
2. [Auto-Instrumentation](#auto-instrumentation)
3. [Tracing Architecture](#tracing-architecture)
4. [Storage Schema](#storage-schema)
5. [Metrics System](#metrics-system)
6. [Calibration Pipeline](#calibration-pipeline)
7. [Data Models](#data-models)
8. [Execution Strategies](#execution-strategies)
9. [Pipeline Orchestration](#pipeline-orchestration)
10. [Analysis & Visualization](#analysis--visualization)
11. [File Structure](#file-structure)
12. [Extension Module Inventory](#extension-module-inventory)
13. [Environment Variables](#environment-variables)
14. [CLI Conveniences](#cli-conveniences)

---

## Terminology

This section defines key terms and their relationships. Understanding this hierarchy is essential for working with Evalyn.

### Hierarchy Diagram

```
Session (optional grouping)
 │
 └── FunctionCall (aka "Call" or "Trace")     <-- Root: @eval decorated function
      │
      ├── Span: agent                          <-- Hierarchical tree of operations
      │    ├── Span: llm_call
      │    │    ├── Span: input_message
      │    │    └── Span: output_message
      │    └── Span: tool_call
      │         ├── Span: tool_use
      │         └── Span: tool_result
      │
      └── Span: llm_call
           └── ...
```

### Core Terms

| Term | Definition | Data Model |
|------|------------|------------|
| **Session** | Optional grouping of related function calls (e.g., same user session) | `session_id` field |
| **FunctionCall** | Root-level capture of an `@eval`-decorated function execution. Contains all spans. Synonymous with "Call" or "Trace" in CLI output. | `FunctionCall` dataclass |
| **Span** | A timed operation within a FunctionCall. Forms a tree via `parent_id`. Each span has a `span_type`. | `Span` dataclass |
| **Trace** | Informal term for FunctionCall. "Show trace" means "show the span tree of a call". | - |

### Span Types

| SpanType | Description | Created By |
|----------|-------------|------------|
| `session` | Root session span | Manual |
| `agent` | Agent execution (ADK, Claude Agent SDK) | Instrumentation |
| `llm_call` | LLM API call (OpenAI, Gemini, Anthropic, etc.) | Instrumentation |
| `tool_call` | Tool/function invocation | Instrumentation |
| `retrieval` | RAG retrieval operation | Instrumentation |
| `graph` | LangGraph execution | Instrumentation |
| `node` | LangGraph node | Instrumentation |
| `scorer` | Metric evaluation span | EvalRunner |
| `custom` | User-defined span | Manual |

**Semantic span kinds** (for fine-grained evaluation):

| SpanType | Description |
|----------|-------------|
| `input_message` | User/system message input to LLM |
| `output_message` | Assistant message output from LLM |
| `tool_use` | Tool invocation request (what the LLM asked for) |
| `tool_result` | Tool execution result (what the tool returned) |

### Evaluation Terms

| Term | Definition | Data Model |
|------|------------|------------|
| **EvalUnit** | An evaluatable unit discovered from trace structure. Can be the full outcome, a single LLM turn, a tool use, etc. | `EvalUnit` dataclass |
| **EvalUnitType** | Category of eval unit: `outcome`, `single_turn`, `tool_use`, `multi_turn`, `custom` | `EvalUnitType` literal |
| **EvalView** | Normalized projection of an EvalUnit with `input` and `output` fields. Decouples metrics from trace structure. | `EvalView` dataclass |
| **Metric** | Evaluation function that scores a call/unit. Can be objective (code) or subjective (LLM judge). | `Metric` class |
| **MetricResult** | Output of metric evaluation: score, passed, details, and optional unit info. | `MetricResult` dataclass |

### Relationship Summary

```
FunctionCall 1:N Span           # A call contains many spans
Span 1:N Span                   # Spans form parent-child tree (via parent_id)
FunctionCall 1:N EvalUnit       # Builders discover units from call's spans
EvalUnit 1:1 EvalView           # Each unit projects to one view
Metric 1:N MetricResult         # One metric produces results for each item/unit
```

### CLI Command Mapping

| CLI Command | What It Shows |
|-------------|---------------|
| `list-calls` | List of FunctionCalls (shows id, function, duration) |
| `show-call --id X` | Single FunctionCall details |
| `show-trace --call-id X` | Span tree of a FunctionCall |
| `show-span --call-id X --span Y` | Single Span details |

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
| OpenAI | Monkey-patch | tokens, cost, duration, request/response, streaming |
| Anthropic Client | Monkey-patch | tokens, cost, duration, request/response, streaming |
| Claude Agent SDK | Hook-based | tool calls, subagent hierarchy, token usage, thinking blocks, GenAI attributes |
| Google Gemini | Monkey-patch | tokens, cost, duration, request/response, streaming |
| xAI (Grok) | Monkey-patch | tokens, cost, duration, request/response |
| Google ADK | Hybrid (OTEL + Callbacks) | agent/LLM/tool spans, token usage, request/response |
| LangChain | Callback handler | LLM calls, tool calls |
| LangGraph | Monkey-patch | graph/node execution spans |
| CrewAI | Monkey-patch | agent/task/tool spans |
| AutoGen | Monkey-patch | agent/message spans |
| DSPy | Monkey-patch | module/predict spans |
| Haystack | Monkey-patch | pipeline/component spans |
| LlamaIndex | Monkey-patch | query/retrieval spans |
| Semantic Kernel | Monkey-patch | function/plugin spans |

### Instrumentation Types

The instrumentation registry supports three strategies:

| Type | Description | SDKs |
|------|-------------|------|
| `MONKEY_PATCH` | Wrap SDK methods directly | OpenAI, Anthropic, Gemini, xAI, CrewAI, AutoGen, DSPy, Haystack, LlamaIndex, Semantic Kernel |
| `OTEL_NATIVE` | Use SDK's built-in OTEL with custom SpanProcessor + callback injection | Google ADK |
| `HOOK_BASED` | Use SDK's hook/callback system | Claude Agent SDK |

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

### Streaming Capture

Provider instrumentors capture streaming LLM responses via `StreamingSpanWrapper` (`_streaming.py`). When a provider returns a streaming iterator, the wrapper intercepts each chunk, accumulates the full response, extracts token usage from the final chunk, and creates a complete span. Transparent to callers. Supported for OpenAI, Anthropic, and Gemini.

### GenAI Semantic Convention Attributes

Provider instrumentors attach OpenTelemetry GenAI semantic convention attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) for interoperability. Currently applied to Claude Agent SDK spans.

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

### Claude Agent SDK Integration

The Claude Agent SDK (claude_agent_sdk) uses a hook-based instrumentation approach. Unlike monkey-patching, hooks must be explicitly passed to the agent.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    @eval Decorator                          │
│  Creates root span, collects all child spans at end         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ query() Patch │   │ Hook Handlers   │   │ Stream Adapter  │
│               │   │                 │   │                 │
│ Captures user │   │ PreToolUse:     │   │ Captures:       │
│ input message │   │  - tool name    │   │  - LLM turns    │
│               │   │  - tool input   │   │  - model name   │
│               │   │  - session_id   │   │  - output text  │
│               │   │                 │   │  - thinking     │
│               │   │ PostToolUse:    │   │  - subagent ctx │
│               │   │  - tool output  │   │  - final metrics│
│               │   │  - duration     │   │                 │
└───────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Span Collector  │
                    │                 │
                    │ Gathers spans   │
                    │ from all layers │
                    │ into call.spans │
                    └─────────────────┘
```

#### Key Components

| Component | Purpose |
|-----------|---------|
| `EvalynAgentHooks` | Hook adapter that captures tool calls as spans |
| `MessageStreamAdapter` | Wraps message stream to capture LLM turns, subagent context, and metrics |
| `create_agent_hooks()` | Factory function to create hooks |
| `create_stream_adapter()` | Factory function to create stream adapter |

#### Three-Layer Instrumentation

**Layer 1: query() Patching**

The instrumentor patches `ClaudeSDKClient.query()` to capture user input:

```python
# Automatic - happens when instrumentation is enabled
async def patched_query(self_client, prompt, **kwargs):
    hooks.capture_user_input(prompt)  # Creates user_message span
    return await original(self_client, prompt, **kwargs)
```

**Layer 2: Hook Handlers**

PreToolUse/PostToolUse hooks capture every tool execution:

```python
# PreToolUse - called before tool runs
async def pre_tool_use_hook(self, hook_input, tool_use_id, context):
    span = Span.new(name=tool_name, span_type="tool_call", ...)
    self._tool_spans[tool_use_id] = SpanState(span, time.time())

# PostToolUse - called after tool completes
async def post_tool_use_hook(self, hook_input, tool_use_id, context):
    state = self._tool_spans.pop(tool_use_id)
    state.span.attributes["output"] = str(tool_response)[:4000]
    state.span.finish(status="ok")
```

**Layer 3: Stream Adapter**

Wraps the message stream to capture LLM turns and final metrics:

```python
async def wrap_stream(self, stream):
    async for msg in stream:
        if type(msg).__name__ == "AssistantMessage":
            self._hooks.log_llm_turn(turn=self._turn_count, model=model, ...)
        elif type(msg).__name__ == "ResultMessage":
            self._hooks.finalize_run(msg)  # Capture tokens, cost, duration
        yield msg
```

#### What Gets Captured

| Data | Source | Limit |
|------|--------|-------|
| User input (query text) | Patched query() method | 4000 chars |
| Tool calls (name, input, output, duration) | PreToolUse/PostToolUse hooks | 4000 chars |
| LLM turns (model, output) | MessageStreamAdapter | Full |
| Subagent spawns (Task tool) | Hook + stream processing | - |
| Parent-child hierarchy | parent_tool_use_id tracking | - |
| Extended thinking blocks (with signature) | ThinkingBlock in stream | - |
| Session ID | All hooks/messages | - |
| Token usage with cache metrics | ResultMessage | - |
| Total cost and duration | ResultMessage | - |
| is_error, result, structured_output | ResultMessage | - |

#### Span Types Created

| Span Type | Name Pattern | Key Attributes |
|-----------|--------------|----------------|
| `user_message` | user_input | content, content_length |
| `tool_call` | WebSearch, Task, Read, Write, Bash, etc. | input, output, session_id, executing_subagent |
| `llm_call` | llm_turn_1, llm_turn_2, ... | model, output, provider |
| `session` | (function name) | call_id, is_error, total tokens |

#### Span Collection Flow

```
1. @eval decorator calls start_call()
   └── Creates root span, initializes span collector

2. User code runs:
   └── client.query(prompt)
       └── Patched query() creates user_message span

   └── async for msg in adapter.wrap_stream(...):
       └── PreToolUse hook -> creates tool_call span (not finished)
       └── PostToolUse hook -> finishes tool_call span, adds to collector
       └── AssistantMessage -> creates llm_call span
       └── ResultMessage -> captures final metrics

3. @eval decorator calls finish_call()
   └── Collects all spans from:
       - Context-local collector (normal spans)
       - Global collector (thread/async spans)
       - Orphan collector (hooks without @eval)
   └── Attaches spans to FunctionCall
   └── Stores to SQLite
```

#### Integration Pattern

```python
from evalyn_sdk import eval
from evalyn_sdk.trace.instrumentation import create_agent_hooks, create_stream_adapter
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher

# Create evalyn hooks
evalyn_hooks = create_agent_hooks()

# Configure hooks with HookMatcher (matcher=None matches all tools)
hooks = {
    'PreToolUse': [
        HookMatcher(matcher=None, hooks=[evalyn_hooks.pre_tool_use_hook])
    ],
    'PostToolUse': [
        HookMatcher(matcher=None, hooks=[evalyn_hooks.post_tool_use_hook])
    ]
}

options = ClaudeAgentOptions(hooks=hooks, ...)

@eval(project="my-agent")
async def chat():
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt=user_input)

        # Wrap stream for additional instrumentation
        adapter = create_stream_adapter(evalyn_hooks)
        async for msg in adapter.wrap_stream(client.receive_response()):
            # Process messages - instrumentation happens automatically
            ...
```

#### Composing with Existing Hooks

If you have existing hooks (e.g., for logging), evalyn hooks can compose with them:

```python
hooks = {
    'PreToolUse': [
        HookMatcher(
            matcher=None,
            hooks=[evalyn_hooks.pre_tool_use_hook, my_logger.pre_tool_use_hook]
        )
    ],
    'PostToolUse': [
        HookMatcher(
            matcher=None,
            hooks=[evalyn_hooks.post_tool_use_hook, my_logger.post_tool_use_hook]
        )
    ]
}
```

#### Viewing Captured Data

```bash
# See counts and span timeline
evalyn show-call --last

# See hierarchical tree with details
evalyn show-trace --last --verbose

# Full output without truncation
evalyn show-trace --last --verbose --full

# Inspect single span fully
evalyn show-span --call-id xxx --span "WebSearch"
```

#### Backwards Compatibility

The old name `AnthropicAgentsInstrumentor` is aliased to `ClaudeAgentSDKInstrumentor` for backwards compatibility.

### Google ADK Integration

Google ADK (Agent Development Kit) uses a hybrid instrumentation approach combining OTEL spans with automatic callback injection for rich content capture.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    @eval Decorator                          │
│  Creates root span, collects all child spans at end         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Runner Patch  │   │ Callback Inject │   │ OTEL Spans      │
│               │   │                 │   │                 │
│ Captures user │   │ before/after:   │   │ Optional base   │
│ input from    │   │  - model_cb     │   │ span structure  │
│ run_async()   │   │  - tool_cb      │   │ via openinfer-  │
│               │   │  - agent_cb     │   │ ence library    │
└───────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Span Collector  │
                    │                 │
                    │ Gathers spans   │
                    │ from all layers │
                    │ into call.spans │
                    └─────────────────┘
```

#### Automatic Callback Injection

The instrumentor automatically injects Evalyn callbacks into all `LlmAgent` and `BaseAgent` instances by patching the `canonical_*_callbacks` properties:

| Property | Class | Evalyn Callback |
|----------|-------|-----------------|
| `canonical_before_model_callbacks` | LlmAgent | `before_model_callback` |
| `canonical_after_model_callbacks` | LlmAgent | `after_model_callback` |
| `canonical_before_tool_callbacks` | LlmAgent | `before_tool_callback` |
| `canonical_after_tool_callbacks` | LlmAgent | `after_tool_callback` |
| `canonical_before_agent_callbacks` | BaseAgent | `before_agent_callback` |
| `canonical_after_agent_callbacks` | BaseAgent | `after_agent_callback` |

**Key behavior:**
- Evalyn callbacks are prepended to user callbacks (run first)
- Evalyn callbacks return `None` to not interfere with user callbacks
- Works automatically for all agent instances - no manual wiring needed

#### What Gets Captured

| Data | Source | Limit |
|------|--------|-------|
| User input | Runner patch (run_async) | Full |
| Agent execution (name, duration) | before/after_agent_callback | - |
| LLM calls (model, tokens, request/response) | before/after_model_callback | 3000 chars prompt, 4000 chars response |
| Tool calls (name, args, result) | before/after_tool_callback | 4000 chars |
| Token usage with cache metrics | LlmResponse.usage_metadata | - |
| Sub-agent hierarchy | AgentTool detection | - |

#### Span Types Created

| Span Type | Name Pattern | Key Attributes |
|-----------|--------------|----------------|
| `user_message` | user_input | content, session_id |
| `agent` | agent:{name} | agent_name, invocation_id, parent_agent |
| `llm_call` | llm:{model} | model, provider, prompt_tokens, completion_tokens |
| `tool_call` | {tool_name} | input, output, is_agent_tool, sub_agent_name |

#### Usage

No manual setup required - just use `@eval`:

```python
from evalyn_sdk import eval
from google.adk.runners import InMemoryRunner
from my_agent import root_agent

@eval(project="my-adk-agent")
async def run_agent(query: str):
    runner = InMemoryRunner(agent=root_agent, app_name="test")
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=query,
    ):
        pass  # All spans captured automatically
```

#### Manual Callback Integration (Optional)

If you need direct access to callbacks (e.g., for custom processing):

```python
from evalyn_sdk.trace.instrumentation.providers.google_adk import (
    create_adk_callbacks,
    create_stream_adapter,
)

callbacks = create_adk_callbacks()

# Use callbacks directly if needed
agent = LlmAgent(
    name="my_agent",
    before_model_callback=callbacks.before_model_callback,
    after_model_callback=callbacks.after_model_callback,
    # ... other callbacks
)
```

#### Environment Setup

For direct Gemini API (recommended for testing):
```bash
export GOOGLE_API_KEY=your_gemini_api_key
# Do NOT set GOOGLE_GENAI_USE_VERTEXAI
```

For Vertex AI:
```bash
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your_project
export GOOGLE_CLOUD_LOCATION=your_location
```

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
CREATE TABLE function_calls (
    id TEXT PRIMARY KEY,
    function_name TEXT,
    session_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    duration_ms REAL,
    inputs TEXT,      -- JSON
    output TEXT,      -- JSON
    error TEXT,
    trace TEXT,       -- JSON array of TraceEvents
    metadata TEXT     -- JSON
);

-- OpenTelemetry span storage
CREATE TABLE otel_spans (
    trace_id TEXT,
    span_id TEXT PRIMARY KEY,
    parent_span_id TEXT,
    call_id TEXT,
    name TEXT,
    start_time TEXT,
    end_time TEXT,
    status TEXT,
    attributes TEXT,  -- JSON
    events TEXT       -- JSON
);

-- Evaluation run results
CREATE TABLE eval_runs (
    id TEXT PRIMARY KEY,
    dataset_name TEXT,
    created_at TEXT,
    metric_results TEXT,  -- JSON array of MetricResults
    metrics TEXT,         -- JSON array of MetricSpecs
    judge_configs TEXT,   -- JSON array of JudgeConfigs
    summary TEXT,         -- JSON
    usage_summary TEXT    -- JSON
);

-- Human annotations
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    target_id TEXT,
    label TEXT,
    rationale TEXT,
    annotator TEXT,
    source TEXT,
    confidence REAL,
    created_at TEXT,
    metric_labels TEXT   -- JSON
);

-- Span-metric attribution links
CREATE TABLE span_metric_links (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    metric_result_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    relevance REAL NOT NULL,
    reason TEXT DEFAULT '',
    UNIQUE(run_id, metric_result_id, span_id)
);
```

Tables use individual columns (not JSON blobs) for efficient querying. JSON is used for nested/variable-length fields (inputs, metadata, metric_results, etc.).

### Default Location

```
data/prod/traces.sqlite   # Production traces
data/test/traces.sqlite   # Test traces (when EVALYN_ENV=test)
```

Override with:
```python
from evalyn_sdk import configure_tracer
from evalyn_sdk.trace.tracer import EvalTracer
from evalyn_sdk.storage.sqlite import SQLiteStorage

configure_tracer(EvalTracer(SQLiteStorage("/custom/path/traces.sqlite")))
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
│   (76 metrics total)    │     │   (60 metrics total)    │
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
| `llm-registry` | LLM selects from 136 built-in templates | Objective + Subjective |
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

The confidence module (`evalyn_sdk/judges/confidence/`) provides:
- `LogprobsConfidence`: Token probability-based (OpenAI/Ollama only)
- `DeepConfConfidence`: Meta AI's DeepConf with bottom-10%/tail strategies (OpenAI only)
- `SelfConsistencyConfidence`: Multi-sample agreement
- `MajorityVoteConfidence`: Weighted voting
- `PerplexityConfidence`: Perplexity-based
- `EntropyConfidence`: Entropy from top-k logprobs

---

## Evaluation Units (Span-Level Evaluation)

### Overview

By default, Evalyn evaluates each dataset item as a single "outcome" unit representing the full trace. The EvalUnit system enables fine-grained span-level evaluation, allowing metrics to be applied to individual LLM calls, tool invocations, or conversation turns within a trace.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EvalRunner                              │
│  1. Load items from dataset                                  │
│  2. For each item, get FunctionCall                          │
│  3. Discover units using EvalUnitBuilders                    │
│  4. Project units to EvalViews                               │
│  5. Apply metrics to views                                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    EvalUnitBuilders     │     │       EvalViews         │
│    (Unit Discovery)     │     │    (Projection)         │
│                         │     │                         │
│   - OutcomeBuilder      │     │   Normalizes units      │
│   - SingleTurnBuilder   │     │   into input/output     │
│   - ToolUseBuilder      │     │   pairs for metrics     │
│   - MultiTurnBuilder    │     │                         │
│   - CustomBuilder       │     │                         │
└─────────────────────────┘     └─────────────────────────┘
```

### Unit Types

| Type | Description | Created From |
|------|-------------|--------------|
| `outcome` | Full trace (default) | Entire FunctionCall |
| `single_turn` | Individual LLM call | Each `llm_call` span |
| `tool_use` | Tool invocation | Each `tool_call` span |
| `multi_turn` | Conversation group | Consecutive `llm_call` spans sharing parent |
| `custom` | User-defined | Spans with `eval_boundary` attribute |

### Data Flow

```
FunctionCall (with spans)
        │
        ▼
┌─────────────────────────┐
│   EvalUnitBuilder       │
│   .discover(call)       │
│                         │
│   Returns: List[EvalUnit]
│   - id, unit_type       │
│   - call_id, span_ids   │
│   - context             │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   project_unit()        │
│                         │
│   Returns: EvalView     │
│   - unit_id, unit_type  │
│   - input, output       │
│   - context             │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Metric.evaluate_unit()│
│                         │
│   Returns: MetricResult │
│   + unit_id, unit_type  │
│   + span_ids            │
└─────────────────────────┘
```

### EvalUnit Dataclass

```python
@dataclass
class EvalUnit:
    id: str              # Unique unit identifier
    unit_type: str       # outcome, single_turn, tool_use, etc.
    call_id: str         # Parent FunctionCall ID
    span_ids: List[str]  # Spans comprising this unit
    context: Dict        # Type-specific metadata
```

### EvalView Dataclass

```python
@dataclass
class EvalView:
    unit_id: str         # From EvalUnit
    unit_type: str       # From EvalUnit
    input: Any           # Projected input (varies by type)
    output: Any          # Projected output (varies by type)
    context: Dict        # Merged context
```

### Builder Implementations

**OutcomeBuilder** (default):
- Creates exactly 1 unit per FunctionCall
- `input` = call.inputs, `output` = call.output
- Backward-compatible with existing evaluations

**SingleTurnBuilder**:
- Creates 1 unit per `llm_call` span
- `input` = span.attributes["input"] or ["messages"]
- `output` = span.attributes["output"] or ["response"]

**ToolUseBuilder**:
- Creates 1 unit per `tool_call` span
- `input` = {tool_name, arguments}
- `output` = tool result from associated `tool_result` span

**MultiTurnBuilder**:
- Groups consecutive `llm_call` spans sharing a parent
- `input` = list of all turn inputs
- `output` = final turn output
- `context.turns` = full conversation history

**CustomBuilder**:
- Finds spans with `eval_boundary=True` attribute
- Uses `eval_input`/`eval_output` attributes if present

### Metric Unit Type Support

Metrics declare supported unit types via `MetricSpec.unit_types`:

```python
MetricSpec(
    id="helpfulness",
    name="Helpfulness",
    type="subjective",
    unit_types=["outcome", "single_turn"],  # Supports both
    ...
)
```

Default is `["outcome"]` for backward compatibility.

### Runner Mode Switching

The EvalRunner automatically switches between modes:

1. **Outcome Mode** (default): When only `OutcomeBuilder` is active
   - Uses parallel execution strategy
   - Cardinality: N items x M metrics = N*M results

2. **Unit Mode**: When non-default builders are active
   - Discovers units per call
   - Filters metrics by supported unit types
   - Cardinality: N items x U units x M compatible metrics

### CLI Usage

```bash
# Evaluate each LLM call individually
evalyn run-eval --dataset data/my-dataset --unit-types single_turn

# Evaluate both full outcome and individual LLM calls
evalyn run-eval --dataset data/my-dataset --unit-types "outcome,single_turn"

# Evaluate tool usage
evalyn run-eval --dataset data/my-dataset --unit-types tool_use
```

### MetricResult Extensions

Unit-based evaluation adds optional fields to MetricResult:

```python
@dataclass
class MetricResult:
    # ... existing fields ...
    unit_id: Optional[str] = None      # EvalUnit.id
    unit_type: Optional[str] = None    # EvalUnit.unit_type
    span_ids: Optional[List[str]] = None  # Spans evaluated
```

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

### Optimizer Methods

| Optimizer | Algorithm | Key Idea |
|-----------|-----------|----------|
| `basic` | Single-shot LLM | Analyze disagreements in one pass |
| `ape` | UCB bandit search | Generate candidates, select via exploration/exploitation |
| `opro` | Trajectory-based | Use history of prompt/score pairs to guide improvement |
| `gepa` | Evolutionary (external) | LLM-based reflection and evolution |
| `gepa-native` | Evolutionary (native) | Pareto-front evolution with token tracking |
| `evoprompt` | Evolutionary | Population-based mutation/crossover with LLM operators |
| `textgrad` | Critique-revise | Iterative LLM critique of failures, revise preamble |
| `miprov2` | Instruction+demo joint | Optimize preamble and few-shot examples together |
| `promptbreeder` | Self-referential | Evolve both prompts and the mutation operators themselves |

All share a common interface: `optimize() -> PromptOptimizationResult`. New optimizers inherit from `BaseOptimizer` (in `calibration/base_optimizer.py`), which provides train/val split, candidate scoring, and token tracking utilities. A factory function in `engine.py` dispatches to the correct optimizer.

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
    started_at: datetime
    ended_at: Optional[datetime]
    duration_ms: Optional[float]
    session_id: Optional[str]
    trace: List[TraceEvent]          # Formerly trace_events
    metadata: Dict[str, Any]         # signature, docstring, source, hash
    parent_call_id: Optional[str]    # For nested @eval calls
    spans: List[Span]                # Hierarchical span tree

@dataclass
class TraceEvent:
    kind: str                        # llm_call, tool_call, trace, error
    timestamp: datetime
    detail: Dict[str, Any]
    span_id: Optional[str]           # Link to associated Span
    parent_span_id: Optional[str]    # Parent span for hierarchy

@dataclass
class DatasetItem:
    id: str
    input: Dict[str, Any]            # User input
    output: Optional[Any]            # Agent output
    human_label: Optional[Dict]      # Human judgement (for calibration)
    metadata: Dict[str, Any]
    # Backward-compat aliases: inputs, expected

@dataclass
class MetricResult:
    metric_id: str
    item_id: str
    call_id: str
    score: Optional[float]
    passed: Optional[bool]
    details: Dict[str, Any]
    raw_judge: Optional[Dict]        # Raw LLM judge response
    input_tokens: Optional[int]      # Token usage for subjective metrics
    output_tokens: Optional[int]
    model: Optional[str]
    unit_id: Optional[str]           # For span-level evaluation
    unit_type: Optional[str]         # EvalUnitType
    span_ids: Optional[List[str]]

@dataclass
class EvalRun:
    id: str
    dataset_name: str
    created_at: datetime
    metric_results: List[MetricResult]
    metrics: List[MetricSpec]
    judge_configs: List[JudgeConfig]
    summary: Dict[str, Any]          # Per-metric aggregates
    usage_summary: Dict[str, Any]    # Token usage totals

@dataclass
class Annotation:
    id: str
    target_id: str                   # Item being annotated
    label: Any                       # Overall pass/fail
    rationale: Optional[str]
    annotator: str
    source: str                      # "human" default
    confidence: Optional[int]        # 1-5 scale
    metric_labels: Dict[str, MetricLabel]
    created_at: datetime

@dataclass
class MetricLabel:
    metric_id: str
    agree_with_llm: bool
    human_label: bool
    notes: str

@dataclass
class SpanMetricLink:
    id: str
    metric_result_id: str            # composite: metric_id:item_id:call_id
    span_id: str
    relevance: float                 # 0.0-1.0
    reason: str
    run_id: str
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

### Insights Engine

`evalyn insights` (`analysis/insights.py`) provides deeper analysis: metric correlations (Pearson r), regression detection vs previous run, input feature analysis, score distribution shape detection (bimodal, cliff, skewed), and prioritized recommendations.

### LLM Expert Panel

`--deep` activates an LLM expert panel (`analysis/panel.py`): quality_analyst -> metric_critic -> data_scientist -> strategist -> moderator. Each expert sees prior opinions for progressive deepening. Moderator synthesizes into action plan with dissenting views.

### Insights HTML Dashboard

`--format html` generates an interactive Chart.js dashboard (`analysis/insights_dashboard.py`) with pass rate bars, radar chart, score histograms, item-metric heatmap, scatter plots, correlation matrix, regression waterfall, recommendation cards, and collapsible expert panel section.

### Span-Metric Attribution

`attribution.py` links metric results to specific spans: `extract_span_metric_links()` produces `SpanMetricLink` records with relevance scores (0.0-1.0) and textual reasons, enabling span-level drill-down.

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
│       ├── defaults.py          # Default model constants
│       ├── parsing.py           # JSON and response parsing
│       ├── sampling.py          # Dataset sampling strategies
│       ├── attribution.py       # Span-metric attribution extraction
│       ├── evaluation/          # Evaluation engine
│       │   ├── runner.py        # EvalRunner orchestrator
│       │   ├── execution.py     # Sequential/Parallel strategies
│       │   ├── units/
│       │   │   ├── builders.py  # EvalUnit discovery from traces
│       │   │   └── views.py     # EvalView projections
│       │   └── batch/
│       │       ├── evaluator.py # Large-scale batch evaluation
│       │       └── providers.py # Batch API providers (OpenAI)
│       ├── analysis/            # Analysis & visualization module
│       │   ├── core.py          # RunAnalysis, MetricStats classes
│       │   ├── reports.py       # Text/ASCII reports
│       │   ├── html_report.py   # HTML report generation
│       │   ├── trends.py        # Trend analysis over time
│       │   ├── insights.py      # Correlations, regressions, distributions
│       │   ├── insights_dashboard.py  # Interactive HTML insights dashboard
│       │   ├── panel.py         # LLM expert panel analysis
│       │   └── clustering.py    # Failure/misalignment clustering
│       ├── trace/
│       │   ├── tracer.py        # EvalTracer, session management
│       │   ├── context.py       # Context management
│       │   ├── auto_instrument.py # Backward-compat wrapper
│       │   ├── otel.py          # OpenTelemetry support
│       │   └── instrumentation/ # SDK instrumentation
│       │       ├── registry.py  # InstrumentorRegistry
│       │       ├── base.py      # Instrumentor base class
│       │       ├── conventions.py # Naming conventions
│       │       ├── span_converter.py # OTEL span conversion
│       │       ├── span_processor.py # Span processing
│       │       └── providers/   # Per-SDK instrumentors
│       │           ├── _shared.py       # Shared utilities
│       │           ├── _streaming.py    # StreamingSpanWrapper base
│       │           ├── openai.py
│       │           ├── anthropic.py
│       │           ├── claude_agent_sdk.py
│       │           ├── gemini.py
│       │           ├── xai.py
│       │           ├── google_adk.py
│       │           ├── langchain.py
│       │           ├── langgraph.py
│       │           ├── crewai.py
│       │           ├── autogen.py
│       │           ├── dspy.py
│       │           ├── haystack.py
│       │           ├── llamaindex.py
│       │           └── semantic_kernel.py
│       ├── storage/
│       │   ├── base.py          # StorageBackend interface
│       │   ├── sqlite.py        # SQLiteStorage
│       │   └── migrations.py    # Schema version upgrades
│       ├── metrics/
│       │   ├── objective.py     # 76 objective metric templates + handlers
│       │   ├── subjective.py    # 60 subjective metric definitions
│       │   ├── factory.py       # Metric builders
│       │   └── suggester.py     # Metric suggestion logic
│       ├── judges/
│       │   ├── llm_judge.py     # LLM judge implementation
│       │   └── confidence/      # Confidence estimation
│       │       ├── base.py      # ConfidenceEstimator ABC
│       │       ├── logprobs.py  # Logprobs, perplexity, entropy
│       │       ├── consistency.py # Self-consistency, majority vote
│       │       └── verbalized.py # Extract self-reported confidence
│       ├── calibration/         # Prompt calibration optimizers
│       │   ├── engine.py        # Calibration execution engine
│       │   ├── base_optimizer.py # BaseOptimizer protocol
│       │   ├── factory.py       # Optimizer factory
│       │   ├── models.py        # Optimization models/results
│       │   ├── utils.py         # Shared utilities
│       │   ├── basic.py         # Basic random search
│       │   ├── ape.py           # Automatic Prompt Engineer
│       │   ├── opro.py          # In-context optimization
│       │   ├── gepa.py          # Genetic evolutionary alignment
│       │   ├── gepa_native.py   # Native GEPA implementation
│       │   ├── evoprompt.py     # Evolutionary optimization
│       │   ├── textgrad.py      # Text-based gradient optimization
│       │   ├── miprov2.py       # Multi-stage instruction optimization
│       │   └── promptbreeder.py # Self-referential evolution
│       ├── annotation/
│       │   ├── annotations.py   # Annotation models
│       │   └── span_annotation.py # Span-level annotation
│       ├── simulation/
│       │   └── simulator.py     # Synthetic data generation
│       ├── cli/
│       │   ├── main.py          # CLI entry point
│       │   ├── commands/        # CLI command modules
│       │   │   ├── analysis.py
│       │   │   ├── annotation.py
│       │   │   ├── calibration.py
│       │   │   ├── clustering.py
│       │   │   ├── dataset.py
│       │   │   ├── evaluation.py
│       │   │   ├── export.py
│       │   │   ├── infrastructure.py  # one-click command
│       │   │   ├── insights.py        # evalyn insights command
│       │   │   ├── report.py          # static HTML insights report
│       │   │   ├── dashboard_alias.py # `dashboard` alias (forwards to report when evalyn-dashboard not installed)
│       │   │   ├── quickstart.py     # Onboarding workflow
│       │   │   ├── runs.py
│       │   │   ├── simulate.py
│       │   │   └── traces.py
│       │   └── utils/           # CLI utilities
│       │       ├── command_common.py  # Shared command utilities
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
│   ├── clis/                    # CLI command documentation
│   │   ├── README.md
│   │   ├── one-click.md
│   │   ├── run-eval.md
│   │   └── ...                  # Other CLI docs
│   ├── dev/                     # Developer docs
│   └── optimizers/              # Optimizer design docs
└── example_agents/              # SDK integration examples
```

---

## Extension Module Inventory

Beyond the core packages above, `sdk/evalyn_sdk/` contains 190 top-level Python modules (as of 2026-03-29). The extension modules are grouped by domain below.

### Sampling (23 modules)

Strategies for selecting subsets of evaluation data.

| Module | Purpose |
|--------|---------|
| `adversarial_sampling` | Adversarial example selection |
| `balanced_sampling` | Class-balanced subset selection |
| `bootstrap_resampling` | Bootstrap resampling for confidence intervals |
| `coreset_sampling` | Coreset construction for representative subsets |
| `cost_aware_sampling` | Cost-weighted sample selection |
| `coverage_sampling` | Coverage-maximizing sampling |
| `curriculum_sampling` | Curriculum-ordered sampling (easy to hard) |
| `disagreement_sampling` | Samples where judges disagree |
| `drift_sampling` | Distribution-drift-aware sampling |
| `error_pattern_sampling` | Sampling focused on error patterns |
| `importance_sampling` | Importance-weighted sampling |
| `locale_sampling` | Locale/language-aware sampling |
| `metadata_sampling` | Metadata-driven subset selection |
| `novelty_sampling` | Novel/unseen-pattern sampling |
| `progressive_sampling` | Incrementally growing sample sizes |
| `reservoir_sampling` | Online reservoir sampling |
| `sampling_impact` | Impact analysis of sampling choices |
| `sampling_pipeline` | Composable sampling pipeline |
| `sampling_reproducibility` | Reproducibility guarantees for sampling |
| `seed_selection` | Seed item selection heuristics |
| `similarity_sampling` | Similarity-based sampling |
| `stratified_sampling` | Stratified random sampling |
| `time_weighted_sampling` | Recency-weighted sampling |

### Simulation (22 modules)

Synthetic data generation and simulation-based evaluation.

| Module | Purpose |
|--------|---------|
| `adversarial_simulation` | Adversarial input generation |
| `behavior_test_gen` | Behavioral test case generation |
| `budget_optimizer` | Simulation budget optimization |
| `conditional_simulation` | Conditional data generation |
| `constraint_simulation` | Constraint-satisfying generation |
| `diversity_metrics` | Diversity measurement for generated data |
| `domain_transfer` | Cross-domain simulation |
| `eval_loop` | Simulation-evaluation loop |
| `evol_instruct` | Evol-Instruct style complexity scaling |
| `feedback_injection` | Injecting feedback into simulations |
| `multiturn_simulation` | Multi-turn conversation simulation |
| `parallel_simulation` | Parallel simulation execution |
| `persona_simulation` | Persona-driven simulation |
| `quality_score` | Quality scoring for generated data |
| `reference_simulation` | Reference answer generation |
| `regression_simulation` | Regression-focused test generation |
| `reproducibility_seed` | Seed management for reproducible simulations |
| `seed_clustering` | Clustering-based seed selection |
| `simulation_templates` | Reusable simulation templates |
| `simulation_validation` | Validation of simulation outputs |
| `structured_simulation` | Structured output simulation |
| `tool_schema_simulation` | Tool/function-call simulation |

### CLI (26 modules)

CLI extensions, UX improvements, and output formatting.

| Module | Purpose |
|--------|---------|
| `batch_script` | Batch script execution |
| `cli_aliases` | User-defined command aliases |
| `cli_plugins` | Plugin loading for CLI extensions |
| `color_theme` | Terminal color theme configuration |
| `command_chaining` | Pipe-style command chaining |
| `command_history` | Command history tracking |
| `compare_shorthand` | Shorthand syntax for comparisons |
| `completion_notify` | Desktop notifications on completion |
| `config_show` | Display current configuration |
| `config_validation` | Config file validation |
| `execution_audit` | Audit log of CLI executions |
| `garbage_collect` | Storage cleanup and compaction |
| `json_output` | Machine-readable JSON output mode |
| `openai_evals_export` | Export to OpenAI Evals format |
| `output_pagination` | Paged terminal output |
| `output_width` | Dynamic output width detection |
| `pipeline_visualization` | Visual pipeline diagrams |
| `playground` | Interactive evaluation playground |
| `profile_command` | Performance profiling for commands |
| `progress_dashboard` | Real-time progress dashboard |
| `quick_rerun` | One-command re-run of last evaluation |
| `shell_completion` | Shell tab-completion generation |
| `side_by_side` | Side-by-side output comparison |
| `time_tracking` | Wall-clock time tracking per command |
| `tui_mode` | Terminal UI mode |
| `watch_mode` | File-watch triggered re-evaluation |

### Analysis and Reporting (13 modules)

Extended analysis, reporting, and visualization.

| Module | Purpose |
|--------|---------|
| `benchbuilder` | Custom benchmark construction |
| `clustering_report` | Cluster analysis reports |
| `comparison_overlay` | Overlay comparison charts |
| `compliance_report` | Compliance/governance reports |
| `coverage_report` | Evaluation coverage reports |
| `curation_suggestions` | Dataset curation recommendations |
| `dashboard_export` | Export dashboards to static files |
| `embeddable_widget` | Embeddable HTML report widgets |
| `eval_diff` | Diff between evaluation runs |
| `irt_benchmarks` | Item Response Theory benchmarks |
| `metric_catalog` | Browsable metric catalog |
| `nl_summary` | Natural language summary generation |
| `trace_summary` | Trace-level summary extraction |

### Metrics and Evaluation (11 modules)

Metric extensions, routing, and optimization.

| Module | Purpose |
|--------|---------|
| `adaptive_metrics` | Dynamically adjusted metrics |
| `capo_optimizer` | CAPO-based metric optimization |
| `cascade_routing` | Cascading model routing for evaluation |
| `judge_routing` | Judge selection and routing |
| `metric_debug` | Metric debugging and introspection |
| `metric_namespacing` | Namespaced metric organization |
| `model_baselines` | Baseline model comparisons |
| `offline_eval` | Offline evaluation from stored traces |
| `prompt_optimization` | Prompt optimization for metrics |
| `score_binning` | Score discretization and binning |
| `user_bundles` | User-defined metric bundles |

### Security and Governance (6 modules)

Data protection, audit, and compliance.

| Module | Purpose |
|--------|---------|
| `audit_trail` | Immutable audit trail logging |
| `data_governance` | Data governance policy enforcement |
| `embedding_pii_check` | PII detection in embeddings |
| `key_rotation` | API key rotation management |
| `secrets_backend` | Secrets storage abstraction |
| `trace_redaction` | PII/sensitive data redaction in traces |

### Infrastructure (10 modules)

Packaging, storage, search, and deployment.

| Module | Purpose |
|--------|---------|
| `binary_packaging` | Single-binary packaging |
| `docker_config` | Docker configuration generation |
| `embedding_index` | Embedding vector index |
| `embedding_selection` | Embedding model selection |
| `experiment_tracker` | Experiment tracking integration |
| `fts_search` | Full-text search over traces |
| `large_dataset` | Large dataset streaming support |
| `parquet_export` | Parquet file export |
| `provider_diversity` | Multi-provider diversity enforcement |
| `web_dashboard` | Web-based dashboard server |

### Annotation (7 modules)

Human annotation workflow and agreement.

| Module | Purpose |
|--------|---------|
| `annotation_delegation` | Annotation task delegation |
| `annotation_session` | Annotation session management |
| `annotation_ux` | Annotation UX helpers |
| `annotator_agreement` | Inter-annotator agreement metrics |
| `conflict_resolution` | Annotation conflict resolution |
| `guidelines_generator` | Annotation guidelines generation |
| `pre_annotation` | Pre-annotation with model predictions |

### Rubrics (4 modules)

Rubric management and testing.

| Module | Purpose |
|--------|---------|
| `rubric_i18n` | Rubric internationalization |
| `rubric_library` | Shared rubric library |
| `rubric_packs` | Bundled rubric packs |
| `rubric_testing` | Rubric validation and testing |

### Other (9 modules)

Configuration references, versioning, tutorials, and miscellaneous.

| Module | Purpose |
|--------|---------|
| `breaking_changes` | Breaking change detection |
| `cli_reference` | Auto-generated CLI reference |
| `config_reference` | Configuration reference docs |
| `cost_estimation` | Evaluation cost estimation |
| `deprecation` | Deprecation warnings and migration |
| `example_gallery` | Example gallery generation |
| `persona_hub` | Persona library for simulations |
| `tutorial` | Interactive tutorial system |
| `version_check` | SDK version checking |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVALYN_AUTO_INSTRUMENT` | `on` | Enable/disable auto-patching |
| `EVALYN_NO_HINTS` | `off` | Set to `1` or `true` to suppress CLI hint messages |
| `GEMINI_API_KEY` | - | Gemini API key for LLM judges |
| `GOOGLE_API_KEY` | - | Fallback for GEMINI_API_KEY |
| `OPENAI_API_KEY` | - | OpenAI API key (alternative) |
| `EVALYN_DB` | - | Override database path |
| `EVALYN_OTEL_ENDPOINT` | - | OpenTelemetry endpoint URL |
| `EVALYN_OTEL` | `on` | Enable OpenTelemetry spans |
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

*Last updated: 2026-03-24*
