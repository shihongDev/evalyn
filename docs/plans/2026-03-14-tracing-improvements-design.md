# Tracing Improvements Design

**Goal:** Enhance evalyn's tracing infrastructure with span-level evaluation attribution, richer Agent SDK spans, streaming capture, and context window tracking.

**Approach:** Layered additions - each feature extends the existing architecture independently.

**Scope:** 4 features. W3C traceparent propagation and multi-agent correlation deferred to future design.

---

## Feature 1: Span-Metric Mapping (Many-to-Many)

### Problem
MetricResult links to item_id (a dataset item / FunctionCall). When a metric fails, you know which trace failed but not which span caused it. Credit assignment is the #1 developer pain point (45% of developers).

### Model

```python
@dataclass
class SpanMetricLink:
    id: str
    metric_result_id: str   # composite key: metric_id + item_id + call_id
    span_id: str             # -> Span.id
    relevance: float         # 0.0-1.0
    reason: str              # LLM judge explanation
    run_id: str              # -> EvalRun.id
```

### Storage

New SQLite table:

```sql
CREATE TABLE span_metric_links (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    metric_result_id TEXT NOT NULL,  -- metric_id:item_id:call_id
    span_id TEXT NOT NULL,
    relevance REAL NOT NULL,
    reason TEXT DEFAULT '',
    UNIQUE(run_id, metric_result_id, span_id)
);
CREATE INDEX idx_sml_run_metric ON span_metric_links(run_id, metric_result_id);
CREATE INDEX idx_sml_run_span ON span_metric_links(run_id, span_id);
```

### Population

LLM judge prompt extended with span attribution request:

```
After your verdict, identify which spans in the trace most influenced your decision.
Return a "span_attribution" array:
[{"span_id": "...", "relevance": 0.0-1.0, "reason": "..."}]
```

Judge response parser extracts `span_attribution` and creates SpanMetricLink rows. Objective metrics skip this.

### CLI

- `evalyn show-run --id X --verbose`: shows span attribution per metric result
- `evalyn analyze`: "most implicated spans" summary when failures cluster on specific spans

### Data Flow

```
LLM Judge evaluates item
  -> returns verdict + span_attribution[]
  -> MetricResult created (existing)
  -> SpanMetricLink rows created (new)
  -> stored in span_metric_links table
  -> queryable via CLI
```

---

## Feature 2: Agent SDK Native Span Export

### Problem
Anthropic Agent SDK has no native trace/span export (closed as "Not Planned"). Evalyn's existing claude_agent_sdk.py instrumentor captures tool events but lacks GenAI semantic convention attributes.

### Changes

Enhance existing `instrumentation/providers/claude_agent_sdk.py`:

| Hook Event | Span Type | Key Attributes |
|-----------|-----------|---------------|
| Session start | `agent` | `gen_ai.agent.name`, `gen_ai.agent.id` |
| PreToolUse/PostToolUse | `tool_call` | `gen_ai.tool.name`, `gen_ai.tool.type`, duration, success |
| SubagentStart/Stop | `agent` | `gen_ai.agent.name`, parent_id links to parent agent |
| Message stream chunks | `llm_call` | `gen_ai.request.model`, tokens, cost |

### Storage
Same SQLite otel_spans table and Evalyn span collector. No new tables.

### Conventions
Follows OpenTelemetry GenAI semantic conventions:
- `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.agent.description`
- `gen_ai.request.model`, `gen_ai.response.model`
- `gen_ai.tool.name`, `gen_ai.tool.type` (extension/function)

---

## Feature 3: Streaming Span Support

### Problem
Monkey-patch instrumentors only capture the final assembled response. Missing: time-to-first-token, chunk count, streaming duration, and sometimes inaccurate token counts from streaming.

### Scope
OpenAI, Anthropic, Gemini only. Other providers added on demand.

### Implementation

Wrap streaming return values with a thin proxy that yields chunks unchanged while recording timestamps:

```python
class StreamingSpanWrapper:
    # Wraps streaming iterator
    # Records: first_chunk_time, chunk_count, last_chunk_time
    # On exhaustion: calls log_llm_call with streaming attributes
    # Accumulates token counts from chunk deltas
```

### Span Attributes Added

- `streaming: true` - marks this as a streaming response
- `time_to_first_token_ms: float` - request start to first chunk
- `chunk_count: int` - total chunks received
- `streaming_duration_ms: float` - first chunk to last chunk

### Per-Provider Details

- **OpenAI**: Wrap `Stream[ChatCompletionChunk]`. Token counts from `usage` field on final chunk (if `stream_options.include_usage=True`) or accumulated from chunk deltas.
- **Anthropic**: Wrap `MessageStream`. Token counts from `message_start` and `message_delta` events. Cache tokens from `usage` object.
- **Gemini**: Wrap `GenerateContentResponse` iterator. Token counts from `usage_metadata` on chunks.

---

## Feature 4: Context Window Utilization

### Problem
When an LLM approaches its context limit, output quality degrades. This is invisible in current traces.

### Implementation

Add `MODEL_CONTEXT_WINDOWS` dict to `_shared.py` (parallel to existing `COST_PER_1M_TOKENS`):

```python
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    # ...
}
```

In `log_llm_call`, after computing token counts:

```python
max_ctx = get_model_context_window(model)
if max_ctx:
    span.attributes["context_utilization_pct"] = round(
        total_input_tokens / max_ctx * 100, 1
    )
```

Unknown models: skip attribute (no guessing).

### Behavior
Passive recording only. Visible in `evalyn show-trace` span attributes. No warnings or alerts.

---

## Decisions Record

| Question | Decision |
|----------|----------|
| Platform export | Local only (SQLite + eval loop) |
| Span-eval binding | Many-to-many SpanMetricLink table |
| Relevance scoring | LLM judge populates attribution |
| Streaming scope | OpenAI, Anthropic, Gemini only |
| Streaming focus | Both latency debugging and cost accuracy |
| Context utilization | Passive attribute, no warnings |
| W3C traceparent | Deferred to future design |
| Multi-agent correlation | Deferred to future design |
| Architecture | Layered additions, no refactor |

---

## Files to Modify/Create

### New Files
- `sdk/evalyn_sdk/models.py` - add SpanMetricLink dataclass
- `sdk/evalyn_sdk/storage/sqlite.py` - add span_metric_links table + CRUD

### Modified Files
- `sdk/evalyn_sdk/trace/instrumentation/providers/_shared.py` - MODEL_CONTEXT_WINDOWS, context_utilization_pct, streaming attributes
- `sdk/evalyn_sdk/trace/instrumentation/providers/openai.py` - streaming wrapper
- `sdk/evalyn_sdk/trace/instrumentation/providers/anthropic.py` - streaming wrapper
- `sdk/evalyn_sdk/trace/instrumentation/providers/gemini.py` - streaming wrapper
- `sdk/evalyn_sdk/trace/instrumentation/providers/claude_agent_sdk.py` - GenAI convention attributes
- `sdk/evalyn_sdk/metrics/llm_judge.py` - span attribution in judge prompt + response parsing
- `sdk/evalyn_sdk/cli/commands/traces.py` - show span attribution in show-run --verbose
- `sdk/evalyn_sdk/cli/commands/analysis.py` - "most implicated spans" in analyze output
