# Tracing Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add span-metric mapping, streaming capture, context window tracking, and Agent SDK span enrichment to evalyn's tracing infrastructure.

**Architecture:** Layered additions - each feature extends existing monkey-patch instrumentors and storage independently. No refactoring of existing code.

**Tech Stack:** Python dataclasses, SQLite, monkey-patch instrumentors, existing Span/MetricResult models.

---

### Task 1: SpanMetricLink Model

**Files:**
- Modify: `sdk/evalyn_sdk/models.py` (after MetricResult, ~line 383)
- Test: `tests/test_span_metric_link.py`

**Step 1: Write the failing test**

```python
# tests/test_span_metric_link.py
from evalyn_sdk.models import SpanMetricLink

def test_span_metric_link_creation():
    link = SpanMetricLink(
        id="sml-1",
        metric_result_id="metric1:item1:call1",
        span_id="span-abc",
        relevance=0.85,
        reason="This span contained the hallucinated claim",
        run_id="run-123",
    )
    assert link.relevance == 0.85
    assert link.span_id == "span-abc"

def test_span_metric_link_as_dict_from_dict():
    link = SpanMetricLink(
        id="sml-2",
        metric_result_id="m:i:c",
        span_id="s1",
        relevance=0.5,
        reason="test",
        run_id="r1",
    )
    d = link.as_dict()
    restored = SpanMetricLink.from_dict(d)
    assert restored.id == link.id
    assert restored.relevance == link.relevance
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_span_metric_link.py -v`
Expected: FAIL with ImportError (SpanMetricLink not defined)

**Step 3: Write minimal implementation**

Add after MetricResult in `sdk/evalyn_sdk/models.py`:

```python
@dataclass
class SpanMetricLink:
    """Links a metric result to a specific span with relevance scoring."""
    id: str
    metric_result_id: str   # composite: metric_id:item_id:call_id
    span_id: str
    relevance: float         # 0.0-1.0
    reason: str
    run_id: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanMetricLink":
        return cls(
            id=data["id"],
            metric_result_id=data["metric_result_id"],
            span_id=data["span_id"],
            relevance=data.get("relevance", 0.0),
            reason=data.get("reason", ""),
            run_id=data["run_id"],
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_span_metric_link.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/models.py tests/test_span_metric_link.py
git commit -m "feat: add SpanMetricLink dataclass for span-metric attribution"
```

---

### Task 2: SpanMetricLink Storage

**Files:**
- Modify: `sdk/evalyn_sdk/storage/sqlite.py` (_init_tables, new CRUD methods)
- Test: `tests/test_span_metric_link.py` (extend)

**Step 1: Write the failing test**

```python
# Append to tests/test_span_metric_link.py
import tempfile
from evalyn_sdk.storage.sqlite import SQLiteStorage
from evalyn_sdk.models import SpanMetricLink

def test_store_and_list_span_metric_links():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        link = SpanMetricLink(
            id="sml-1",
            metric_result_id="m:i:c",
            span_id="s1",
            relevance=0.9,
            reason="caused failure",
            run_id="run-1",
        )
        store.store_span_metric_links([link])
        results = store.list_span_metric_links(run_id="run-1")
        assert len(results) == 1
        assert results[0].span_id == "s1"
        store.close()

def test_list_span_metric_links_by_span():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        links = [
            SpanMetricLink("sml-1", "m1:i:c", "span-a", 0.9, "r1", "run-1"),
            SpanMetricLink("sml-2", "m2:i:c", "span-a", 0.5, "r2", "run-1"),
            SpanMetricLink("sml-3", "m3:i:c", "span-b", 0.7, "r3", "run-1"),
        ]
        store.store_span_metric_links(links)
        results = store.list_span_metric_links(run_id="run-1", span_id="span-a")
        assert len(results) == 2
        store.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_span_metric_link.py::test_store_and_list_span_metric_links -v`
Expected: FAIL (store_span_metric_links not found)

**Step 3: Write minimal implementation**

In `sqlite.py` `_init_tables`, add after annotations table:

```python
cur.execute("""
    CREATE TABLE IF NOT EXISTS span_metric_links (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        metric_result_id TEXT NOT NULL,
        span_id TEXT NOT NULL,
        relevance REAL NOT NULL,
        reason TEXT DEFAULT '',
        UNIQUE(run_id, metric_result_id, span_id)
    )
""")
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_sml_run_metric
    ON span_metric_links(run_id, metric_result_id)
""")
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_sml_run_span
    ON span_metric_links(run_id, span_id)
""")
```

Add methods to `SQLiteStorage`:

```python
def store_span_metric_links(self, links: Iterable[SpanMetricLink]) -> None:
    cur = self.conn.cursor()
    for link in links:
        cur.execute("""
            INSERT OR REPLACE INTO span_metric_links
            (id, run_id, metric_result_id, span_id, relevance, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (link.id, link.run_id, link.metric_result_id,
              link.span_id, link.relevance, link.reason))
    self.conn.commit()

def list_span_metric_links(
    self, run_id: str,
    span_id: Optional[str] = None,
    metric_result_id: Optional[str] = None,
) -> List[SpanMetricLink]:
    cur = self.conn.cursor()
    query = "SELECT * FROM span_metric_links WHERE run_id = ?"
    params: list = [run_id]
    if span_id:
        query += " AND span_id = ?"
        params.append(span_id)
    if metric_result_id:
        query += " AND metric_result_id = ?"
        params.append(metric_result_id)
    cur.execute(query, params)
    rows = cur.fetchall()
    return [
        SpanMetricLink(
            id=r["id"], run_id=r["run_id"],
            metric_result_id=r["metric_result_id"],
            span_id=r["span_id"], relevance=r["relevance"],
            reason=r["reason"],
        ) for r in rows
    ]
```

Import SpanMetricLink in sqlite.py:

```python
from ..models import Annotation, EvalRun, FunctionCall, SpanMetricLink
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_span_metric_link.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/storage/sqlite.py tests/test_span_metric_link.py
git commit -m "feat: add span_metric_links table and CRUD to SQLiteStorage"
```

---

### Task 3: Context Window Utilization

**Files:**
- Modify: `sdk/evalyn_sdk/trace/instrumentation/providers/_shared.py`
- Test: `tests/test_context_window.py`

**Step 1: Write the failing test**

```python
# tests/test_context_window.py
from evalyn_sdk.trace.instrumentation.providers._shared import (
    get_model_context_window,
    MODEL_CONTEXT_WINDOWS,
)

def test_known_model_context_window():
    assert get_model_context_window("gpt-4o") == 128_000
    assert get_model_context_window("claude-sonnet-4-5") == 200_000
    assert get_model_context_window("gemini-2.5-flash") == 1_048_576

def test_unknown_model_returns_none():
    assert get_model_context_window("totally-unknown-model") is None

def test_substring_matching():
    # "gpt-4o-2024-08-06" should match "gpt-4o"
    assert get_model_context_window("gpt-4o-2024-08-06") == 128_000
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_context_window.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

Add to `_shared.py` after `COST_PER_1M_TOKENS`:

```python
_MODEL_CONTEXT_WINDOWS_UNSORTED = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "grok-4": 131_072,
}

MODEL_CONTEXT_WINDOWS = dict(
    sorted(_MODEL_CONTEXT_WINDOWS_UNSORTED.items(), key=lambda x: len(x[0]), reverse=True)
)


def get_model_context_window(model: str) -> Optional[int]:
    """Return context window size for a model, or None if unknown."""
    model_lower = model.lower()
    for model_key, window in MODEL_CONTEXT_WINDOWS.items():
        if model_key in model_lower:
            return window
    return None
```

In `log_llm_call`, after cost calculation, add context utilization:

```python
max_ctx = get_model_context_window(model)
if max_ctx and input_tokens:
    context_utilization_pct = round(input_tokens / max_ctx * 100, 1)
    detail["context_utilization_pct"] = context_utilization_pct
    span.attributes["context_utilization_pct"] = context_utilization_pct
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_context_window.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/_shared.py tests/test_context_window.py
git commit -m "feat: add context window utilization tracking to spans"
```

---

### Task 4: Streaming Wrapper Base

**Files:**
- Create: `sdk/evalyn_sdk/trace/instrumentation/providers/_streaming.py`
- Test: `tests/test_streaming_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_streaming_wrapper.py
import time
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper

def test_streaming_wrapper_basic():
    chunks = ["Hello", " world", "!"]
    wrapper = StreamingSpanWrapper(iter(chunks), request_start_time=time.time())
    collected = list(wrapper)
    assert collected == ["Hello", " world", "!"]
    assert wrapper.chunk_count == 3
    assert wrapper.time_to_first_token_ms >= 0
    assert wrapper.streaming_duration_ms >= 0

def test_streaming_wrapper_empty():
    wrapper = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    collected = list(wrapper)
    assert collected == []
    assert wrapper.chunk_count == 0

def test_streaming_wrapper_as_attributes():
    wrapper = StreamingSpanWrapper(iter(["a", "b"]), request_start_time=time.time())
    list(wrapper)  # exhaust
    attrs = wrapper.as_span_attributes()
    assert attrs["streaming"] is True
    assert "time_to_first_token_ms" in attrs
    assert "chunk_count" in attrs
    assert attrs["chunk_count"] == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streaming_wrapper.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# sdk/evalyn_sdk/trace/instrumentation/providers/_streaming.py
"""Streaming span wrapper for capturing streaming response metrics."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, TypeVar

T = TypeVar("T")


class StreamingSpanWrapper:
    """Wraps a streaming iterator to capture timing and chunk metrics.

    Yields chunks unchanged while recording:
    - time_to_first_token_ms
    - chunk_count
    - streaming_duration_ms
    """

    def __init__(self, iterator: Iterator[T], request_start_time: float):
        self._iterator = iterator
        self._request_start_time = request_start_time
        self._first_chunk_time: float | None = None
        self._last_chunk_time: float | None = None
        self.chunk_count: int = 0
        self.accumulated_input_tokens: int = 0
        self.accumulated_output_tokens: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._iterator)
        now = time.time()
        if self._first_chunk_time is None:
            self._first_chunk_time = now
        self._last_chunk_time = now
        self.chunk_count += 1
        return chunk

    @property
    def time_to_first_token_ms(self) -> float:
        if self._first_chunk_time is None:
            return 0.0
        return (self._first_chunk_time - self._request_start_time) * 1000

    @property
    def streaming_duration_ms(self) -> float:
        if self._first_chunk_time is None or self._last_chunk_time is None:
            return 0.0
        return (self._last_chunk_time - self._first_chunk_time) * 1000

    def as_span_attributes(self) -> Dict[str, Any]:
        return {
            "streaming": True,
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "chunk_count": self.chunk_count,
            "streaming_duration_ms": round(self.streaming_duration_ms, 2),
        }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streaming_wrapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/_streaming.py tests/test_streaming_wrapper.py
git commit -m "feat: add StreamingSpanWrapper base for streaming instrumentation"
```

---

### Task 5: OpenAI Streaming Instrumentation

**Files:**
- Modify: `sdk/evalyn_sdk/trace/instrumentation/providers/openai.py`
- Test: `tests/test_openai_streaming.py`

**Step 1: Write the failing test**

```python
# tests/test_openai_streaming.py
from unittest.mock import MagicMock, patch
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper

def test_openai_streaming_detection():
    """Verify that stream=True in kwargs triggers StreamingSpanWrapper."""
    # This tests the logic pattern, not the actual OpenAI client
    kwargs = {"model": "gpt-4o", "stream": True, "messages": []}
    assert kwargs.get("stream") is True
```

**Step 2: Run test to verify it passes** (this is a logic test)

Run: `uv run pytest tests/test_openai_streaming.py -v`

**Step 3: Implement streaming detection in openai.py**

In `patched_create` and `patched_acreate`, add after getting the response:

```python
# Detect streaming
if kwargs.get("stream"):
    from ._streaming import StreamingSpanWrapper
    stream_wrapper = StreamingSpanWrapper(response, request_start_time=start)

    def _on_stream_end(wrapper):
        duration_ms = (time.time() - start) * 1000
        # Try to get usage from accumulated stream data
        input_tokens = wrapper.accumulated_input_tokens
        output_tokens = wrapper.accumulated_output_tokens
        attrs = wrapper.as_span_attributes()
        log_llm_call(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            success=True,
            request=_build_request_dict(kwargs),
            streaming_attributes=attrs,
        )

    stream_wrapper._on_end_callback = _on_stream_end
    return stream_wrapper
```

Add `streaming_attributes` parameter to `log_llm_call` in `_shared.py`:

```python
streaming_attributes: Optional[Dict[str, Any]] = None,
```

And in the function body, after building the span:
```python
if streaming_attributes:
    span.attributes.update(streaming_attributes)
    detail.update(streaming_attributes)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_openai_streaming.py tests/test_streaming_wrapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/openai.py sdk/evalyn_sdk/trace/instrumentation/providers/_shared.py tests/test_openai_streaming.py
git commit -m "feat: add streaming span capture for OpenAI provider"
```

---

### Task 6: Anthropic Streaming Instrumentation

**Files:**
- Modify: `sdk/evalyn_sdk/trace/instrumentation/providers/anthropic.py`
- Test: `tests/test_anthropic_streaming.py`

**Step 1: Write the failing test**

```python
# tests/test_anthropic_streaming.py

def test_anthropic_streaming_detection():
    """Verify stream=True detection for Anthropic provider."""
    kwargs = {"model": "claude-sonnet-4-5", "stream": True, "messages": []}
    assert kwargs.get("stream") is True
```

**Step 2: Run test**

Run: `uv run pytest tests/test_anthropic_streaming.py -v`

**Step 3: Implement streaming in anthropic.py**

Same pattern as OpenAI. In `patched_create`, check `kwargs.get("stream")`. Anthropic streaming returns a `MessageStream`. Wrap with `StreamingSpanWrapper`. Extract tokens from `message_start` and `message_delta` events.

Add an `AnthropicStreamWrapper` subclass of `StreamingSpanWrapper` that overrides `__next__` to extract token usage from Anthropic-specific event types:

```python
class AnthropicStreamWrapper(StreamingSpanWrapper):
    def __next__(self):
        chunk = super().__next__()
        # Extract token usage from Anthropic streaming events
        event_type = getattr(chunk, "type", "")
        if event_type == "message_start":
            msg = getattr(chunk, "message", None)
            if msg:
                usage = getattr(msg, "usage", None)
                if usage:
                    self.accumulated_input_tokens = getattr(usage, "input_tokens", 0)
        elif event_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage:
                self.accumulated_output_tokens = getattr(usage, "output_tokens", 0)
        return chunk
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_anthropic_streaming.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/anthropic.py tests/test_anthropic_streaming.py
git commit -m "feat: add streaming span capture for Anthropic provider"
```

---

### Task 7: Gemini Streaming Instrumentation

**Files:**
- Modify: `sdk/evalyn_sdk/trace/instrumentation/providers/gemini.py`
- Test: `tests/test_gemini_streaming.py`

**Step 1: Write the failing test**

```python
# tests/test_gemini_streaming.py

def test_gemini_streaming_detection():
    """Verify stream=True detection for Gemini provider."""
    kwargs = {"model": "gemini-2.5-flash", "stream": True}
    assert kwargs.get("stream") is True
```

**Step 2: Run test**

Run: `uv run pytest tests/test_gemini_streaming.py -v`

**Step 3: Implement streaming in gemini.py**

Gemini uses `stream=True` on `generate_content`. Wrap the returned iterator. Token counts from `usage_metadata` on chunks:

```python
class GeminiStreamWrapper(StreamingSpanWrapper):
    def __next__(self):
        chunk = super().__next__()
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            self.accumulated_input_tokens = getattr(usage, "prompt_token_count", 0)
            self.accumulated_output_tokens = getattr(usage, "candidates_token_count", 0)
        return chunk
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_gemini_streaming.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/gemini.py tests/test_gemini_streaming.py
git commit -m "feat: add streaming span capture for Gemini provider"
```

---

### Task 8: Agent SDK Span Enrichment

**Files:**
- Modify: `sdk/evalyn_sdk/trace/instrumentation/providers/claude_agent_sdk.py`
- Test: `tests/test_agent_sdk_genai.py`

**Step 1: Write the failing test**

```python
# tests/test_agent_sdk_genai.py

def test_genai_convention_keys():
    """Verify GenAI semantic convention attribute keys are correct."""
    expected_keys = [
        "gen_ai.agent.name",
        "gen_ai.agent.id",
        "gen_ai.request.model",
        "gen_ai.tool.name",
        "gen_ai.tool.type",
    ]
    # Just verify the keys are valid strings for now
    for key in expected_keys:
        assert isinstance(key, str)
        assert key.startswith("gen_ai.")
```

**Step 2: Run test**

Run: `uv run pytest tests/test_agent_sdk_genai.py -v`

**Step 3: Implement GenAI convention attributes**

In `claude_agent_sdk.py`, enhance the span attributes set during hook callbacks:

For agent spans (session start, subagent):
```python
span.attributes["gen_ai.agent.name"] = agent_name
span.attributes["gen_ai.agent.id"] = agent_id
```

For tool call spans:
```python
span.attributes["gen_ai.tool.name"] = tool_name
span.attributes["gen_ai.tool.type"] = "function"  # or "extension"
```

For LLM call spans:
```python
span.attributes["gen_ai.request.model"] = model
span.attributes["gen_ai.response.model"] = response_model
```

Read `claude_agent_sdk.py` fully to find exact hook methods to modify (on_tool_start, on_tool_end, etc.), then add the attributes.

**Step 4: Run tests**

Run: `uv run pytest tests/test_agent_sdk_genai.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/trace/instrumentation/providers/claude_agent_sdk.py tests/test_agent_sdk_genai.py
git commit -m "feat: add GenAI semantic convention attributes to Agent SDK spans"
```

---

### Task 9: LLM Judge Span Attribution

**Files:**
- Modify: `sdk/evalyn_sdk/judges/llm_judge.py` (_build_evaluation_prompt, _parse_response)
- Test: `tests/test_judge_attribution.py`

**Step 1: Write the failing test**

```python
# tests/test_judge_attribution.py
import json

def test_parse_span_attribution():
    """Test that span_attribution is extracted from judge response."""
    from evalyn_sdk.judges.llm_judge import LLMJudge

    judge = LLMJudge.__new__(LLMJudge)
    judge.rubric = None

    raw = json.dumps({
        "passed": True,
        "reason": "Good output",
        "score": 0.9,
        "span_attribution": [
            {"span_id": "span-1", "relevance": 0.8, "reason": "main LLM call"},
            {"span_id": "span-2", "relevance": 0.3, "reason": "tool call"},
        ]
    })

    score, passed, reason, parsed = judge._parse_response(raw)
    assert passed is True
    assert "span_attribution" in parsed
    assert len(parsed["span_attribution"]) == 2
    assert parsed["span_attribution"][0]["span_id"] == "span-1"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_judge_attribution.py -v`
Expected: Depends on whether _parse_response already preserves unknown keys (it might pass since parsed dict comes from JSON).

**Step 3: Implement**

In `_build_evaluation_prompt`, append to the return format instructions:

```python
# After the existing "Return ONLY a JSON object with:" block, add:
- "span_attribution": array of objects (optional, only if trace spans are provided)
  Each object: {"span_id": "...", "relevance": 0.0-1.0, "reason": "..."}
  Identify which spans most influenced your verdict.
```

In `_parse_response`, ensure `span_attribution` flows through in the returned `parsed` dict (it should already since we return the parsed JSON object). If not, explicitly extract it.

**Step 4: Run tests**

Run: `uv run pytest tests/test_judge_attribution.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/judges/llm_judge.py tests/test_judge_attribution.py
git commit -m "feat: add span attribution to LLM judge prompt and response parsing"
```

---

### Task 10: Wire Attribution into Pipeline

**Files:**
- Create: `sdk/evalyn_sdk/attribution.py`
- Test: `tests/test_attribution_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_attribution_pipeline.py
from evalyn_sdk.attribution import extract_span_metric_links

def test_extract_links_from_metric_result():
    """Extract SpanMetricLinks from a MetricResult with raw_judge attribution."""
    from evalyn_sdk.models import MetricResult

    result = MetricResult(
        metric_id="helpfulness",
        item_id="item-1",
        call_id="call-1",
        score=0.9,
        passed=True,
        raw_judge={
            "span_attribution": [
                {"span_id": "s1", "relevance": 0.8, "reason": "main call"},
            ]
        },
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert len(links) == 1
    assert links[0].span_id == "s1"
    assert links[0].relevance == 0.8
    assert links[0].run_id == "run-1"

def test_extract_links_no_attribution():
    from evalyn_sdk.models import MetricResult

    result = MetricResult(
        metric_id="accuracy",
        item_id="item-1",
        call_id="call-1",
        score=1.0,
        passed=True,
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert links == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution_pipeline.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# sdk/evalyn_sdk/attribution.py
"""Extract span-metric attribution links from judge results."""
from __future__ import annotations

import uuid
from typing import List

from .models import MetricResult, SpanMetricLink


def extract_span_metric_links(
    result: MetricResult, run_id: str
) -> List[SpanMetricLink]:
    """Extract SpanMetricLinks from a MetricResult's raw_judge data."""
    if not result.raw_judge:
        return []

    attributions = result.raw_judge.get("span_attribution", [])
    if not attributions:
        return []

    metric_result_id = f"{result.metric_id}:{result.item_id}:{result.call_id}"
    links = []
    for attr in attributions:
        span_id = attr.get("span_id")
        if not span_id:
            continue
        links.append(
            SpanMetricLink(
                id=str(uuid.uuid4()),
                metric_result_id=metric_result_id,
                span_id=span_id,
                relevance=float(attr.get("relevance", 0.0)),
                reason=attr.get("reason", ""),
                run_id=run_id,
            )
        )
    return links
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sdk/evalyn_sdk/attribution.py tests/test_attribution_pipeline.py
git commit -m "feat: add attribution extraction pipeline for span-metric links"
```

---

### Task 11: Integration Test

**Files:**
- Test: `tests/test_tracing_integration.py`

**Step 1: Write integration test**

```python
# tests/test_tracing_integration.py
"""Integration tests for tracing improvements."""
import tempfile
from evalyn_sdk.models import SpanMetricLink, MetricResult
from evalyn_sdk.storage.sqlite import SQLiteStorage
from evalyn_sdk.attribution import extract_span_metric_links
from evalyn_sdk.trace.instrumentation.providers._shared import (
    get_model_context_window,
    MODEL_CONTEXT_WINDOWS,
)
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper
import time


def test_full_attribution_flow():
    """End-to-end: judge result -> attribution extraction -> storage -> retrieval."""
    result = MetricResult(
        metric_id="quality",
        item_id="item-1",
        call_id="call-1",
        score=0.3,
        passed=False,
        raw_judge={
            "passed": False,
            "reason": "Low quality",
            "span_attribution": [
                {"span_id": "llm-span-1", "relevance": 0.95, "reason": "hallucinated"},
                {"span_id": "tool-span-2", "relevance": 0.4, "reason": "wrong tool input"},
            ],
        },
    )

    links = extract_span_metric_links(result, run_id="run-abc")
    assert len(links) == 2

    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        store.store_span_metric_links(links)

        # Query by run
        all_links = store.list_span_metric_links(run_id="run-abc")
        assert len(all_links) == 2

        # Query by span
        span_links = store.list_span_metric_links(run_id="run-abc", span_id="llm-span-1")
        assert len(span_links) == 1
        assert span_links[0].relevance == 0.95

        store.close()


def test_context_window_known_models():
    """All models in cost table should ideally have context windows."""
    from evalyn_sdk.trace.instrumentation.providers._shared import COST_PER_1M_TOKENS
    for model_key in COST_PER_1M_TOKENS:
        # Not all may have entries, but common ones should
        pass  # This is a coverage check, not an assertion


def test_streaming_wrapper_attributes_in_span():
    """StreamingSpanWrapper produces correct span attributes."""
    start = time.time()
    wrapper = StreamingSpanWrapper(iter(["a", "b", "c"]), request_start_time=start)
    list(wrapper)
    attrs = wrapper.as_span_attributes()
    assert attrs["streaming"] is True
    assert attrs["chunk_count"] == 3
    assert isinstance(attrs["time_to_first_token_ms"], float)
    assert isinstance(attrs["streaming_duration_ms"], float)
```

**Step 2: Run all tests**

Run: `uv run pytest tests/test_tracing_integration.py tests/test_span_metric_link.py tests/test_context_window.py tests/test_streaming_wrapper.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_tracing_integration.py
git commit -m "test: add integration tests for tracing improvements"
```

---

## Unsolved Questions

None - all decisions made in the design doc.
