# Competitive Landscape: LLM Evaluation Ecosystem

Research date: 2026-03-27

---

## 1. CI/CD Integration for LLM Evaluation

### Braintrust

**GitHub Action**: `braintrustdata/eval-action` (GitHub Marketplace)

Inputs: `api_key`, `runtime` (node/python), `root`, `paths`, `package_manager` (npm/pnpm/yarn/pip/uv), `use_proxy` (default true - routes through Braintrust proxy for caching), `terminate_on_failure`.

Workflow requires `permissions: pull-requests: write, contents: read`.

Behavior: Runs `braintrust eval`, collects experiment results, posts formatted PR comments with score comparisons (improvements/regressions with percentage changes). Links to Braintrust dashboard for detailed drill-down.

Key differentiator: Built-in experiment comparison - shows which eval cases regressed, by how much, versus previous deployments. Supports both Python and Node runtimes with automatic dependency management.

```yaml
- uses: braintrustdata/eval-action@v1
  with:
    api_key: ${{ secrets.BRAINTRUST_API_KEY }}
    runtime: node
    root: ./evals
```

### DeepEval

**No dedicated GitHub Action** - uses raw pytest in CI.

Integration pattern: Write test files with `@pytest.mark.parametrize`, use `assert_test(test_case, metrics)`. Run via `deepeval test run test_llm_app.py` (not plain `pytest` - the wrapper adds LLM-specific features like concurrent metric evaluation, 8+ optional flags).

Pytest plugin entry point: `[tool.poetry.plugins."pytest11"] deepeval = "deepeval.plugins.plugin"`. Plugin loads automatically on any pytest invocation. Disable with `pytest -p no:deepeval`.

CLI entry point: `deepeval = 'deepeval.cli.main:app'` (Typer-based).

Typical CI YAML:
```yaml
- run: poetry run deepeval test run test_llm_app.py
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
```

### PromptFoo

**GitHub Action**: `promptfoo/promptfoo-action@v1` (GitHub Marketplace)

Inputs: `openai-api-key`, `github-token`, `prompts` (glob pattern), `config` (path to promptfooconfig.yaml), `cache-path`.

Behavior: Produces before/after view of edited prompts on PRs. Triggers on `pull_request` when `paths: 'prompts/**'` change. Posts results as PR comments with links to interactive web viewer.

Caching: Uses `actions/cache@v4` with `~/.cache/promptfoo`. Stores LLM requests/outputs for reuse. Cache keys include content hashes for automatic invalidation.

Quality gates: Defined in `promptfooconfig.yaml`, not in the action itself. Supports minimum score thresholds.

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/promptfoo
    key: ${{ runner.os }}-promptfoo-v1
- uses: promptfoo/promptfoo-action@v1
  with:
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    github-token: ${{ secrets.GITHUB_TOKEN }}
    prompts: 'prompts/**/*.json'
    config: 'prompts/promptfooconfig.yaml'
```

### LangSmith

**No dedicated GitHub Action** - uses custom Python/JS scripts in CI.

Integration: `langsmith.evaluate()` or `aevaluate()` for async. Integrates with pytest and Vitest. Runs evals on every PR or nightly build. Uses `@traceable` decorator for instrumentation.

CI pattern: Write evaluation scripts, run them as CI steps, fail on threshold breach. Comparison between experiments done via LangSmith web UI or API.

### Evidently AI

**GitHub Action** recently released.

Behavior: Downloads predefined test prompt dataset, runs agent against inputs, evaluates responses using LLM judges / Python functions / metrics, generates Test Suite report with pass/fail based on thresholds, fails CI if any test fails, produces artifacts (report + scored dataset).

Optionally saves to Evidently Cloud for trend tracking.

### Summary of GitHub Actions for LLM Evaluation

| Tool | Action | PR Comments | Caching | Quality Gates |
|------|--------|-------------|---------|---------------|
| Braintrust | `braintrustdata/eval-action@v1` | Yes (score diffs) | Via proxy | Yes |
| PromptFoo | `promptfoo/promptfoo-action@v1` | Yes (web viewer link) | `actions/cache` | Via config |
| Evidently | Custom action | Artifacts only | No | Yes (pass/fail) |
| DeepEval | None (raw pytest) | No | No | Via assertions |
| LangSmith | None (custom scripts) | No | No | Via API |

---

## 2. Dashboard and Visualization

### Phoenix (Arize)

**Frontend tech stack:**
- React 19
- Recharts 3.8+ (primary charting)
- D3 utilities (d3-format, d3-scale-chromatic for formatting/colors only)
- React Aria Components (accessibility)
- Emotion (CSS-in-JS)
- Apollo Client (GraphQL)
- TanStack Table + Virtual (tables/virtualization)
- React Router 7

**Key trace viewer features:**
- Timeline view of every step, prompt, response
- Error detection with retries visibility
- Latency and cost metrics per span
- Pre-defined metrics dashboards per project (auto-indexed)
- Time series plots from span data
- Span attribute filtering directly on page
- In-trace evaluation: ask questions and run evals from trace view
- Span-level detail: functions, inputs, outputs display

**Deployment**: Self-hosted (Docker, Kubernetes/Helm), or cloud at app.phoenix.arize.com. No telemetry collected from self-hosted instances.

### Langfuse

**Frontend tech stack:**
- React / Next.js
- Recharts (charting - confirmed in PR #6322 adding widget creator)
- shadcn/ui components
- Magic UI (animations)
- Tailwind CSS

**Dashboard features:**
- Two dashboard types: Prebuilt (auto-generated per project) and Custom (fully configurable)
- Widget types: line charts, bar charts, time series, pie charts
- Multi-level aggregations across tracing data
- Group by: user, model, time, trace name, other dimensions
- Live data reflection
- Cross-project sharing
- Tool call filtering and visualization (added 2025-12)
- Prebuilt covers: trace count, error rates, token usage, cost breakdowns, latency distributions

**Architecture (v3)**: PostgreSQL + ClickHouse + MinIO + Redis (all via single Docker Compose).

### LangSmith

**Dashboard features:**
- Prebuilt dashboards: auto-generated per tracing project (trace count, error rates, tokens)
- Custom dashboards: fully configurable chart collections
- Side-by-side comparison views: same dataset against different prompt versions/models/agents
- Visual tradeoff display: accuracy vs latency (P50/P99) vs token efficiency
- Collaborative prompt building interface
- Step-by-step trace inspection with metadata and intermediate outputs

**Noted weakness**: UI can overwhelm non-technical collaborators.

### PromptFoo

**Frontend tech stack:**
- React 19 (confirmed)
- TypeScript (96.9% of codebase)
- Vite (build tool)
- Drizzle ORM (structured data)
- Socket.io (real-time)
- Ink (terminal UI)
- CLI-first with web viewer for interactive exploration

Specific charting library not confirmed in root package.json - likely in app subdirectory or uses lightweight custom rendering.

### Opik (Comet)

- Open-source, self-contained
- Enhanced charts for experiments
- Large trace handling
- Real-time online evaluation scoring
- Kubernetes Helm deployment support
- Prompt optimization suite (Few-shot Bayesian, MIPRO, evolutionary, MetaPrompt)

### Visualization Library Summary

| Platform | Primary Chart Lib | CSS/UI | Data Layer |
|----------|-------------------|--------|------------|
| Phoenix | Recharts 3 + D3 utils | Emotion + React Aria | Apollo/GraphQL |
| Langfuse | Recharts | shadcn/ui + Tailwind | ClickHouse |
| LangSmith | Unknown (proprietary) | Custom | Proprietary |
| PromptFoo | Unknown | Vite + React | Drizzle/SQLite |
| Opik | Unknown | Unknown | Unknown |

**Industry standard**: Recharts (built on D3, React-native) is the dominant choice for open-source LLM dashboards. Both Phoenix and Langfuse use it.

---

## 3. Packaging and Distribution

### Distribution Matrix

| Tool | pip | npm/npx | Docker | Homebrew | Conda | pipx/uvx |
|------|-----|---------|--------|----------|-------|----------|
| DeepEval | Yes (PyPI) | - | - | - | - | Yes |
| Ragas | Yes (PyPI) | - | - | - | Yes (conda-forge) | Yes |
| Phoenix | Yes (PyPI) | - | Yes (Docker Hub) | - | - | Yes |
| Langfuse | Yes (SDK only) | Yes (SDK) | Yes (platform) | - | - | - |
| PromptFoo | Yes (PyPI) | Yes (primary) | Yes | Yes (brew) | - | - |
| Braintrust | Yes (PyPI) | Yes (npm) | - | - | - | - |
| LangSmith | Yes (PyPI) | Yes (npm) | - | - | - | - |
| Opik | Yes (PyPI) | - | Yes | - | - | Yes |

### DeepEval Packaging Details

- Build system: Poetry (`poetry-core` backend)
- Python: >=3.9
- Extras: `[integrations]` (CrewAI, Pydantic AI, LlamaIndex - requires 3.10+), `[langchain]` (LangChain + LangGraph)
- Core deps: pytest, openai, pydantic >=2.11, requests
- CLI: Typer-based (`deepeval` command)
- Pytest plugin: Auto-registered via `pytest11` entry point

### Ragas Packaging Details

- PyPI primary distribution
- Conda-forge available
- Extras: `[all]`, `[git]`, `[tracing]`, `[gdrive]`, `[ai-frameworks]`, `[oci]`, `[ag-ui]`, `[dspy]`, `[dev-minimal]`, `[test]`
- Also has `ragas-experimental` as separate package

### Phoenix Packaging Details

- PyPI: `arize-phoenix` (includes entire platform)
- Docker: `arizephoenix/phoenix:latest` (>= v4.0)
- Ports: 6006 (HTTP), 4317 (gRPC/OTLP), 9090 (Prometheus)
- Kubernetes: Helm chart on Docker Hub (`arizephoenix/phoenix-helm`)
- Lightweight sub-packages available for SDK-only usage

### PromptFoo Packaging Details

- Primary distribution: npm (`promptfoo` package)
- Also on: PyPI, Homebrew, Docker Hub, GHCR
- Self-hosting: Docker Compose or Helm
- CLI: `npx promptfoo@latest` for zero-install usage
- Monorepo: pnpm workspaces

### CLI Distribution Best Practices (2025-2026)

**Modern Python CLI distribution**:
1. `uv tool install <package>` - fastest, isolated environments (Rust-based)
2. `pipx install <package>` - established, slower (Python-based)
3. `pip install <package>` - direct install, no isolation
4. `uvx <package>` - ephemeral execution without install

**uv** has emerged as the unified tool replacing pip + pip-tools + pipx + poetry + pyenv + virtualenv + twine. Written in Rust, significantly faster than Python-based alternatives.

**Multi-platform pattern** (PromptFoo model): Publish to npm + PyPI + Homebrew + Docker simultaneously. npm/npx for primary, pip as bridge for Python users, brew for macOS convenience, Docker for self-hosting.

---

## 4. Developer Experience and SDK Design

### LangSmith @traceable Decorator

**Context propagation**: Uses `_PARENT_RUN_TREE` ContextVar. When a traceable function executes, it sets itself as parent in context. Nested traceable calls auto-detect and become children. Also maintains `_PROJECT_NAME`, `_TAGS`, `_METADATA` ContextVars that child runs inherit.

**Async handling**: Uses `asyncio.create_task()` with context propagation (Python 3.11+) or manual context managers for older versions. Wrapper preserves `_PARENT_RUN_TREE` through async boundaries.

**Background batching architecture**:
1. Client creates `PriorityQueue` with `auto_batch_tracing=True` (default)
2. `create_run()` and `update_run()` enqueued, not sent directly
3. Dedicated background thread processes queue
4. Payloads compressed before transmission
5. Optional Rust-based ingestion via `USE_PYO3_CLIENT` for better performance
6. Must call `wait_for_all_tracers()` before exit to ensure delivery

**RunTree data structure**: id (UUID), name, run_type ("llm"/"chain"/"tool"), start/end time, parent_run ref, child_runs list, session_name, events list, inputs/outputs.

**Special LLM handling**: When `run_type="llm"`, yielded items become "new_token" events for streaming token display.

**Config injection**: Automatically adds `config` parameter to function signatures for LangChain `RunnableConfig` integration.

### Langfuse @observe Decorator

**v2 (decorator-based)**:
- Uses Python `contextvars` for thread-safe, async-aware context
- Parameters: `name`, `as_type` ("span"/"generation"/"embedding"), `capture_input`, `capture_output`, `transform_to_string`
- Trace vs span: Creates new trace when no parent exists, child span otherwise
- Override with: `langfuse_trace_id`, `langfuse_parent_observation_id`, `langfuse_public_key`
- Batching: Reuses low-level SDK, derived from PostHog SDKs
- Generator handling: `_ContextPreservedSyncGeneratorWrapper` stores original generator + context + span, runs each `__next__()` in preserved context
- Streaming: Detects Starlette `StreamingResponse` and wraps internal generator
- Environment: `LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED` for global I/O toggle

**v3 (OpenTelemetry-based, GA since 2025-06)**:
- Built on OpenTelemetry standard
- `@observe()` now creates root span (not trace)
- OTEL automatic context propagation replaces custom contextvars
- Third-party OTEL-instrumented libraries integrate automatically
- `from langfuse import observe, get_client`
- v2 still supported (critical fixes only, no new features)

### W&B Weave @weave.op()

**Versioning mechanism**: Captures source code representation of the op, including inline comments and recursively captures variable values / non-Op function sources. Creates new op version if code has changed since last call. Falls back to hash if code capture is disabled.

**Key parameters**: `tracing_sample_rate` (control trace frequency for high-volume ops), `postprocess_inputs` (transform input dict), `postprocess_output` (transform return value).

**Activation**: Requires `weave.init('project-name')` call. Without it, decorated functions behave normally (no tracking).

**Async**: Works for both sync and async. Auto-detects iterator functions.

### DeepEval Pytest Plugin

**Mechanism**: Registers via `pytest11` entry point as `deepeval.plugins.plugin`. Loads on every pytest invocation (controversial - see GitHub issue #1419 about telemetry).

**Test structure**: Standard pytest files with `@pytest.mark.parametrize` looping through test cases/goldens. Uses `assert_test(test_case, metrics, run_async=True)` instead of standard assertions.

**`deepeval test run`** wraps pytest with 8+ additional flags: concurrent metric evaluation, enhanced reporting, Confident AI integration. Regular `pytest` works but loses LLM-specific features.

**Test case**: `LLMTestCase` requires `input`, `actual_output`. Optional: `expected_output`, `context`, `retrieval_context`.

### Config and API Key Management Patterns

**DeepEval**: Loads .env files in order: `.env` -> `.env.{APP_ENV}` -> `.env.local` (highest). Process env vars never overwritten. Explicit constructor args always win. CLI manages env vars: `deepeval set-api-key`.

**LangSmith**: Requires `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`. Optional: `LANGCHAIN_PROJECT`, `LANGCHAIN_ENDPOINT`.

**Langfuse**: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`. Constructor args override env vars.

**PromptFoo**: Config in `promptfooconfig.yaml`. API keys via env vars or `--env` flag. Supports `.env` files.

**Common pattern**: All tools follow `ENV_VAR -> constructor arg -> .env file` precedence, with constructor args as highest priority. Provider-specific key names are standardized (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

---

## 5. Interoperability and Export

### OpenTelemetry for LLM - Current Standard (2026)

**GenAI Semantic Conventions** (experimental, v1.38.0):
- Namespace: `gen_ai.*`
- Maintained by OTel Generative AI Instrumentation SIG (started April 2024)
- Status: Experimental (requires `OTEL_SEMCONV_STABILITY_OPT_IN` for opt-in)

**Required span attributes**:
- `gen_ai.operation.name` (chat, embeddings, retrieval, execute_tool, create_agent, invoke_agent)
- `gen_ai.provider.name` (openai, anthropic, aws.bedrock, gcp.vertex_ai, etc.)

**Recommended attributes**:
- `gen_ai.request.model`, `gen_ai.response.model`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens`
- `gen_ai.response.finish_reasons`, `gen_ai.response.id`

**Opt-in content attributes** (sensitive):
- `gen_ai.input.messages`, `gen_ai.output.messages`
- `gen_ai.system_instructions`, `gen_ai.tool.definitions`

**Agent spans** (development status):
- `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.agent.version`
- `gen_ai.conversation.id`
- Span types: create_agent, invoke_agent, execute_tool

**Deprecated (v1.38.0)**: `gen_ai.prompt`, `gen_ai.completion` replaced by structured message attributes.

**Adoption**: Datadog (v1.37+), Langfuse (v3 SDK built on OTEL), Phoenix (native OTEL), Traceloop/OpenLLMetry.

### Tool-to-Tool Export Patterns

**OpenTelemetry as the bridge**: The dominant interop mechanism. Tools export OTLP data that any compliant backend consumes.

**OpenLLMetry (Traceloop)**: Open-source library that instruments LLM calls using OpenTelemetry. Supports exporting to: Langfuse, Datadog, Honeycomb, Grafana, New Relic, and others. Captures data from LLM providers, vector DBs, and frameworks.

**Langfuse as OTLP backend**: Accepts traces from any OTEL-instrumented application. Compatible with OpenLLMetry, giving Java and Go language support.

**Multi-destination export**: Teams increasingly need multiple backends simultaneously (e.g., Langfuse for GenAI analytics + Instana/Datadog for infrastructure). Proposed via multiple OTLP endpoints in OpenLLMetry.

**Phoenix**: Built on OpenTelemetry and OpenInference. Traces stay with your stack, not a vendor.

### Data Format Landscape

**JSONL**: Most common for evaluation datasets and results. Used by:
- EleutherAI lm-evaluation-harness (results + per-sample logs)
- DeepEval (test case import/export)
- PromptFoo (dataset format)
- Braintrust (dataset format)

**Parquet**: Growing adoption for large-scale evaluation data:
- Arize (Iceberg + Parquet as source of truth)
- Hyperparam (supports Parquet, JSONL, CSV)
- HuggingFace datasets (default Parquet storage)

**JSON**: Structured results output:
- EleutherAI harness: per-task metrics, standard errors, task versions, seed, commit hash
- PromptFoo: `promptfooconfig.yaml` for config, JSON for results
- Braintrust: JSON experiment results via API

**No universal standard format** for LLM evaluation results. Each tool defines its own schema. The closest to standardization:
1. OpenTelemetry GenAI semantic conventions (for traces/spans)
2. JSONL with tool-specific schemas (for eval datasets)
3. Parquet for bulk data exchange

### Practical Interop Summary

| From | To | Mechanism |
|------|----|-----------|
| Any OTEL app | Langfuse | OTLP endpoint |
| Any OTEL app | Phoenix | OTLP endpoint |
| OpenLLMetry | Langfuse/Datadog/etc | OTLP exporters |
| Langfuse | External | OTEL-compatible export |
| Phoenix | External | OpenInference + OTLP |
| DeepEval | Confident AI | Proprietary API |
| PromptFoo | External | JSON/CSV export |
| Braintrust | External | API + JSON export |

---

## Key Takeaways for Evalyn

1. **CI/CD**: The bar is a GitHub Action that posts PR comments with eval diffs. Braintrust and PromptFoo set this standard. DeepEval's pytest integration is the simplest entry point for Python-first teams.

2. **Dashboards**: Recharts on React is the industry standard for open-source LLM dashboards. Both Phoenix and Langfuse use it. A self-hosted dashboard with trace viewing and metric charts is table stakes.

3. **Packaging**: Multi-channel (PyPI + Docker minimum). `uv tool install` and `pipx` for CLI tools. Docker for self-hosted dashboards. PromptFoo's npm+pip+brew+docker approach is the gold standard for reach.

4. **SDK Design**: Decorator-based tracing (@observe, @traceable, @weave.op) is the universal pattern. Key requirements: contextvars for async safety, background batching for zero-overhead, automatic parent-child nesting, generator/streaming support. Langfuse v3's move to OTEL as the foundation signals where the industry is heading.

5. **Interop**: OpenTelemetry GenAI semantic conventions are the emerging standard but still experimental. JSONL is the practical default for eval data exchange. No universal eval result schema exists - this is an opportunity.
