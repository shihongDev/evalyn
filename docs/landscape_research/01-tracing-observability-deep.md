# Deep Research: LLM Tracing & Observability

## Phoenix Architecture

**Storage:** SQLite (local dev) or PostgreSQL (production) via SQLAlchemy ORM. Queue-based bulk insertion (20K capacity, protobuf decode, BulkInserter). Traces stored as hierarchical span trees.

**Trace viewer:** Hierarchical span tree with color-coded span types. Per-span: input/output, token counts, duration, cost. Search by span attributes. Filter by time range, span type, error status.

**Prompt Playground:** Pull production traces, modify prompts, re-execute against same inputs. Side-by-side comparison of original vs modified outputs. Supports multiple provider backends.

**Clustering:** UMAP dimensionality reduction on span embeddings. HDBSCAN clustering for anomaly detection. Drift detection: compare embedding distributions across time windows.

**Built-in evaluators:** Hallucination, QA correctness, relevance, toxicity, code readability, SQL generation correctness. Plus Ragas and DeepEval integrations.

## Langfuse Architecture (March 2026)

**Major architecture shift:** Moved from separate traces/observations tables to single immutable observations table. Denormalized trace-level attributes onto every observation. Result: 10x dashboard performance.

**@observe decorator:** v3 migrated to OpenTelemetry foundation. ContextVars for async safety. Automatic parent-child hierarchy. Overhead: <1ms per span.

**Prompt versioning:** Prompts stored as versioned objects linked to traces. Before/after performance comparison when prompt changes. Non-technical UI for prompt editing.

**New OTEL-based SDK v3:** Full OpenTelemetry compatibility. Export to any OTLP backend. Backward compatible with v2 decorator API.

## Braintrust Architecture

**DAG trace model:** Spans can have multiple parents (vs tree model where each span has one parent). Enables modeling complex agent workflows where a span's output feeds multiple downstream spans. In practice, "most executions are a tree" but DAG handles edge cases.

**Production-to-test pipeline:** One-click: select a production trace, it becomes a test case in a dataset. Online scoring API continuously evaluates live requests. Low-scoring calls auto-populate "needs improvement" dataset. Tightest production-to-eval loop of any platform.

**Online scoring:** API endpoint that scores production requests in real-time. Configurable scorers run asynchronously. Results feed back into monitoring dashboards and datasets.

## OpenTelemetry GenAI Conventions (v1.38.0)

**Status:** Still experimental (not stable). 40+ attributes defined in `gen_ai.*` namespace.

**Key attributes:**
- `gen_ai.system` - provider name
- `gen_ai.request.model` / `gen_ai.response.model`
- `gen_ai.request.temperature`, `gen_ai.request.max_tokens`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.response.finish_reasons`
- Agent spans: `gen_ai.agent.name`, `gen_ai.agent.description`

**Gap vs OpenInference:** OpenInference (Phoenix) is more expressive with document/retrieval/embedding attributes. OTel GenAI is catching up but still lacks RAG-specific conventions.

## Emerging Patterns (2025-2026)

**Deterministic replay:** Record all nondeterministic inputs as append-only JSONL events. Replay with stub clients. Maps directly to trace-to-evaluation workflow.

**Streaming trace capture:** Real-time span updates (not just start/end). Token-by-token timing data flowing into trace viewer.

**Multi-modal trace content:** Images and audio as span attachments. Phoenix supports image thumbnails in trace viewer.

## Key Insights for Evalyn

1. **Langfuse's denormalized storage** (single observations table) dramatically improved performance - evalyn should consider similar optimization
2. **Braintrust's production-to-test pipeline** is the gold standard for closing the eval loop
3. **OTel GenAI conventions are still experimental** - evalyn can safely build on OpenInference while OTel catches up
4. **Deterministic replay** is an emerging pattern evalyn should implement (trace -> swap model -> replay -> compare)
5. **Phoenix uses SQLite for local dev** (like evalyn) but PostgreSQL for production - validates evalyn's storage strategy

*Sources: Phoenix GitHub/docs, Langfuse docs/blog, Braintrust docs, OpenTelemetry GenAI spec*
