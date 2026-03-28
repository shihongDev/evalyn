# Technical Deep Dive: Tracing Platform Internals

## Phoenix Storage Architecture

**Schema:** SQLAlchemy ORM with spans table containing: trace_rowid (indexed FK), span_id (unique), parent_id (indexed), name, span_kind, start_time (indexed), end_time, attributes (JSON blob), events (JSON array), status_code, cumulative token counts.

**Indexes:** trace_rowid, parent_id, start_time, latency (expression index), cumulative token count, session_id (partial index on attributes).

**BulkInserter:** Async context manager, background coroutine, 1000 ops/transaction, 100ms cycle, 20K span OTLP queue capacity (HTTP 429 when exceeded). Recursive CTE for updating ancestor cumulative values on out-of-order span arrival.

## Langfuse v3 SDK Architecture

**@observe overhead:** 106-266us per operation. LangChain integration: ~16.8% latency increase. OpenAI direct: ~1.7%.

**Batching:** BatchSpanProcessor with 512 flush_at, 5.0s flush interval, 100K score queue, 100K media queue.

**OTEL mapping:** Spans with `model` attribute become "generations", otherwise "spans". Full attribute mapping for gen_ai.*, langfuse.*, llm.* namespaces.

**Architecture shift (March 2026):** Single immutable observations table (denormalized) replacing separate traces/observations tables. 10x dashboard performance improvement.

## Braintrust DAG Model

**Multi-parent spans:** spanParents array (not single parent_id). Cross-process propagation via export()/import pattern (SpanComponentsV3/V4 serialization).

**Online scoring:** Production logs scored automatically. Configurable sampling rate, metadata filters, span targeting. Low-scoring calls auto-populate "needs improvement" datasets.

**Proxy caching:** Cloudflare Workers, AES-GCM encryption, auto-cache when temperature=0 or seed set, sub-100ms cached responses, 1-week default TTL.

## OTel GenAI Conventions (Full Attribute List)

40+ attributes across: gen_ai.agent.*, gen_ai.request.*, gen_ai.response.*, gen_ai.usage.*, gen_ai.tool.*, gen_ai.retrieval.*, gen_ai.evaluation.*. All still experimental (not stable). OpenInference is more expressive for AI-specific operations.

## Trace Replay State of Art

No production tool offers full deterministic replay. Closest: Temporal (event-sourced workflow recovery), CrewAI (task checkpoint replay), AgentRR (academic JSONL record-replay with stub clients). Practical approach: capture prompts + contexts at trace time, re-run as evaluation (not replay).

*Sources: Phoenix GitHub/models.py, Langfuse SDK benchmarks, Braintrust docs, OTel GenAI spec, AgentRR paper*
