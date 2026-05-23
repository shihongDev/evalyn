# Evalyn Roadmap

Last verified: 2026-05-23.

This roadmap is grounded in the actual code. Every checked item below
corresponds to a CLI command, source file, or function that exists in
`sdk/evalyn_sdk/` today. Items that were previously listed but never
implemented have been moved to the "Not Yet Implemented" section at the
bottom.

If you ship something new, add it to the right section and check the box.
If you add a planned-but-unbuilt feature, use `- [ ]` and put it under
"Not Yet Implemented" until the code lands.

---

## Shipped

### CLI Surface (50 commands)

Run `evalyn help` to see the live list. Source of truth:
`sdk/evalyn_sdk/cli/main.py:_COMMAND_MODULE_MAP`.

- [x] **Quick start** - `quickstart`, `workflow`, `one-click`, `init`,
      `doctor`
- [x] **Tracing** - `list-calls`, `show-call`, `show-trace`, `show-span`,
      `show-projects`, `delete-traces`, `watch`, `search`, `import-traces`
- [x] **Dataset** - `build-dataset`, `validate`, `status`, `dataset-diff`
- [x] **Metrics** - `suggest-metrics`, `select-metrics`, `list-metrics`
- [x] **Evaluation** - `run-eval` (with `--max-cost` budget guard and
      `--fail-on` quality gate flags), `list-runs`, `show-run`
- [x] **Analysis** - `analyze`, `compare`, `trend`, `insights`, `report`
- [x] **Annotation** - `annotate`, `annotation-stats`, `import-annotations`
- [x] **Calibration** - `calibrate`, `list-calibrations`
- [x] **Clustering** - `cluster-failures`, `cluster-misalignments`
- [x] **Export & simulation** - `export`, `export-for-annotation`,
      `simulate`, `redteam`
- [x] **Dashboard plugin alias** - `dashboard` (forwards to `report` when the
      `evalyn-dashboard` plugin is not installed)

### Production & CI Integration

- [x] **`evalyn doctor`** - environment, dependency, API-key, config, and
      storage diagnostics; exit 1 on any FAIL
      (`sdk/evalyn_sdk/cli/commands/doctor.py`)
- [x] **`evalyn watch`** - real-time trace tail (`tail -f` for evalyn);
      polls SQLite for new calls and prints them as they arrive
      (`sdk/evalyn_sdk/cli/commands/watch.py`)
- [x] **`evalyn search`** - query DSL over traces; reuses the
      `datasets/filter.py` parser; supports nested `metadata.X` paths,
      `and`/`or` compound expressions, all standard operators
      (`sdk/evalyn_sdk/cli/commands/search.py`)
- [x] **`evalyn import-traces`** - bring external OpenAI / Anthropic /
      generic-JSONL logs into evalyn storage so `list-calls`, `analyze`,
      etc. see them as native calls
      (`sdk/evalyn_sdk/cli/commands/import_traces.py`,
      `sdk/evalyn_sdk/integration/trace_importers.py`)
- [x] **`evalyn dataset-diff`** - CLI front-end for the existing
      `diff_datasets()` library function; reports added / removed /
      modified items with table or JSON output
      (`sdk/evalyn_sdk/cli/commands/dataset_diff.py`)
- [x] **`evalyn redteam`** - generate adversarial dataset variants
      (prompt injection, jailbreak phrasing, edge cases); LLM-driven
      where useful, procedural for edge cases
      (`sdk/evalyn_sdk/cli/commands/redteam.py`,
      `sdk/evalyn_sdk/simulation/redteam.py`)
- [x] **`run-eval` cost budget** - `--max-cost USD`; warns at 80%,
      aborts cleanly with exit 2 at 100%, persists partial results
      (`sdk/evalyn_sdk/evaluation/guards.py:BudgetTracker`)
- [x] **`run-eval` quality gates** - `--fail-on metric<threshold`
      (repeatable); exits 3 if any threshold is breached; catches
      typoed metric IDs preflight where possible
      (`sdk/evalyn_sdk/evaluation/guards.py:ThresholdExpression`)
- [x] **pytest plugin** - `@pytest.mark.evalyn(dataset=..., metrics=...,
      thresholds={...})` runs an eval as part of pytest and fails the
      test with a per-metric breach breakdown; registered via the
      `pytest11` entry point
      (`sdk/evalyn_sdk/pytest_plugin.py`)

### Production Monitoring & Workflow (shipped 2026-05-23 round 2)

- [x] **`evalyn replay`** - re-run a captured trace's prompts against
      a different model; reports cost / latency / output deltas
      (`sdk/evalyn_sdk/cli/commands/replay.py`,
      `sdk/evalyn_sdk/evaluation/replay.py`)
- [x] **`evalyn baseline`** - pin an eval run as the canonical
      reference; supports multiple named baselines
      (`sdk/evalyn_sdk/cli/commands/baseline.py`,
      `sdk/evalyn_sdk/analysis/baseline.py`)
- [x] **`evalyn drift`** - compare current run to baseline; per-metric
      delta with regression flagging; exit 1 on threshold breach
      (`sdk/evalyn_sdk/cli/commands/drift.py`)
- [x] **`evalyn notify`** - Slack / Discord / generic webhook alerts
      on `--fail-on`-style threshold breaches; auto-detects format from
      URL host; retries on 5xx; reads from `notify:` block in
      `evalyn.yaml`
      (`sdk/evalyn_sdk/cli/commands/notify.py`,
      `sdk/evalyn_sdk/integration/notifiers.py`)
- [x] **`evalyn auto-rubric`** - synthesize an LLM-judge rubric from
      labeled examples; writes a JSON spec immediately consumable by
      `run-eval --metrics`
      (`sdk/evalyn_sdk/cli/commands/auto_rubric.py`,
      `sdk/evalyn_sdk/calibration/rubric_generator.py`)
- [x] **`evalyn export-otlp`** - verify an external OTLP collector
      and emit the env-var / `evalyn.yaml` snippet to route traces to
      it (`sdk/evalyn_sdk/cli/commands/export_otlp.py`)
- [x] **`evalyn list-bundles` / `evalyn show-bundle`** - discover and
      inspect the 17 curated metric bundles (previously only
      discoverable by reading the source)
      (`sdk/evalyn_sdk/cli/commands/bundles.py`)
- [x] **`evalyn dataset-stats`** - quick summary over a dataset
      (item count, input/output length distribution, label coverage,
      metadata-key frequency)
      (`sdk/evalyn_sdk/cli/commands/dataset_stats.py`)

### Tracing & Instrumentation

- [x] **`@evalyn.eval` decorator** - `sdk/evalyn_sdk/decorators.py:eval`
- [x] **OpenTelemetry backbone** - SQLite span exporter +
      `BatchSpanProcessor`; `sdk/evalyn_sdk/trace/otel.py`
- [x] **OTLP exporter** - gRPC OTLP exporter for sending traces to external
      OTel collectors; falls back to in-tree exporter when the optional
      `opentelemetry-exporter-otlp` package is missing
      (`trace/otel.py:14-15,183-191`)
- [x] **Streaming response capture** - `StreamingSpanWrapper` used by
      Anthropic and Gemini providers
      (`trace/instrumentation/providers/_streaming.py`)
- [x] **ContextVar-based span context** - async-friendly span hierarchy
      (`trace/context.py:_span_stack`)
- [x] **Auto-instrumentation** - `trace/auto_instrument.py`

### Provider & Framework Instrumentors (14)

Source: `sdk/evalyn_sdk/trace/instrumentation/providers/`.

- [x] OpenAI (`openai.py`)
- [x] Anthropic (`anthropic.py`)
- [x] Gemini / Google GenAI (`gemini.py`)
- [x] Google ADK (`google_adk.py`)
- [x] xAI (`xai.py`)
- [x] Claude Agent SDK (`claude_agent_sdk.py`)
- [x] LangChain (`langchain.py`)
- [x] LangGraph (`langgraph.py`)
- [x] CrewAI (`crewai.py`)
- [x] AutoGen (`autogen.py`)
- [x] DSPy (`dspy.py`)
- [x] Haystack (`haystack.py`)
- [x] LlamaIndex (`llamaindex.py`)
- [x] Semantic Kernel (`semantic_kernel.py`)

### Datasets

- [x] **Build from traces** - `cli/commands/dataset.py` (`build-dataset`)
- [x] **Versioning & snapshots** - `datasets/versioning.py`
      (`create_snapshot`, `list_versions`, `rollback_to_version`)
- [x] **Split** - random + stratified; `datasets/split.py`
      (`split_dataset`, `split_two_way`)
- [x] **Filter (DSL)** - parser + compound filter;
      `datasets/filter.py` (`parse_filter`, `filter_items`)
- [x] **Merge & diff** - `datasets/merge.py`
      (`diff_datasets`, `merge_datasets`)
- [x] **Incremental build** - `datasets/incremental.py`
      (`incremental_build`, build-state persistence)
- [x] **Pinning** - content-hash pinning; `datasets/pin.py`
      (`create_pin`, `verify_pin`, `load_pin`, `remove_pin`)
- [x] **Sampling** - random, diverse, stratified, clustered, dedupe;
      `sampling.py`

### Metrics

- [x] **76 objective metrics** - `metrics/objective.py`
      (lexical, structural, latency, token, code complexity, etc.)
- [x] **60 subjective (LLM-judge) metrics** - `metrics/subjective.py`
      (`SUBJECTIVE_REGISTRY`, `JUDGE_TEMPLATES`)
- [x] **17 curated bundles** - `cli/constants.py:BUNDLES`
      (`chatbot`, `customer-support`, `content-writer`, `summarization`,
      `creative-writer`, `rag-qa`, `research-agent`, `tutor`,
      `code-assistant`, `data-extraction`, `orchestrator`,
      `multi-step-agent`, `medical-advisor`, `legal-assistant`,
      `financial-advisor`, `moderator`, `translator`)
- [x] **Metric factory & suggester** - `metrics/factory.py`,
      `metrics/suggester.py`
- [x] **LLM-guided metric selection** - `select-metrics` command (LLM
      picks from registry given a target signature)
- [x] **Metric suggestion modes** - `basic`, `bundle`, `llm-registry`,
      `llm-brainstorm` (`decorators.py:ALLOWED_METRIC_MODES`)

### Evaluation Engine

- [x] **`evalyn.evaluate(...)` programmatic API** - `api.py:evaluate`
      returns `EvalResult`
- [x] **`run-eval` CLI** - end-to-end runner over a dataset + metric set
- [x] **Batch evaluation** - `evaluation/batch/evaluator.py`,
      `evaluation/batch/providers.py`
- [x] **Evaluation units & views** - `evaluation/units/builders.py`,
      `evaluation/units/views.py`

### LLM Judges

- [x] **LLM judge** - `judges/llm_judge.py`
- [x] **Confidence estimation** - logprobs, verbalized, consistency
      methods (`judges/confidence/{logprobs,verbalized,consistency}.py`)

### Calibration (Judge Optimization)

- [x] **Calibration engine** - `calibration/engine.py`,
      `calibration/factory.py`, `calibration/utils.py`
- [x] **9 optimizers** - `calibration/optimizers/`
  - [x] Basic (`basic.py`)
  - [x] APE (`ape.py`)
  - [x] EvoPrompt (`evoprompt.py`)
  - [x] GEPA + native variant (`gepa.py`, `gepa_native.py`)
  - [x] MIPROv2 (`miprov2.py`)
  - [x] OPRO (`opro.py`)
  - [x] PromptBreeder (`promptbreeder.py`)
  - [x] TextGrad (`textgrad.py`)
- [x] **Calibration record store** - `list-calibrations` CLI

### Annotation

- [x] **Interactive annotate** - `annotate` CLI
      (item-level, per-metric, span-level modes)
- [x] **Annotation persistence & merge** -
      `annotation/annotations.py` (`export_annotations`,
      `import_annotations`, `merge_annotations_into_dataset`)
- [x] **Span-level annotation** - `annotation/span_annotation.py`
- [x] **Compatibility shim for legacy schema** - `annotation/compat.py`
- [x] **Annotation stats** - `annotation-stats` CLI

### Simulation (Synthetic Data)

- [x] **`UserSimulator`** - generates similar / outlier queries;
      `simulation/simulator.py`
- [x] **`AgentSimulator`** - re-runs an agent over generated inputs
- [x] **`synthetic_dataset(prompts)`** - helper to wrap a prompt iterable
      into a `DatasetItem` list
- [x] **`simulate` CLI** - versioned dataset directory output

### Analysis & Insights

- [x] **Core run analysis** - `analysis/core.py`
      (`find_eval_runs`, `RunAnalysis`, `MetricStats`, `ItemStats`)
- [x] **Stats utilities** - `analysis/stats.py`
- [x] **Insights engine** - `analysis/insights.py`
      (correlation, regression detection, feature insights,
      score-distribution insights, recommendations)
- [x] **`insights` CLI** - human-readable and JSON/HTML output
- [x] **`analyze` enhancements** - KEY FINDINGS surfaced from insights
- [x] **`compare` enhancements** - REGRESSION ALERTS surfaced from
      insights
- [x] **`trend` CLI** - `analysis/trends.py`
- [x] **Clustering** - failure-reason clusters and judge-misalignment
      clusters (`analysis/clustering.py`)
- [x] **HTML insights report** - `analysis/html_report.py`,
      `analysis/insights_dashboard.py`, `analysis/reports.py`,
      `analysis/panel.py`; surfaced via `report` CLI

### Storage

- [x] **SQLite storage backend** - `storage/sqlite.py`
- [x] **Schema migrations** - `storage/migrations.py`
- [x] **Storage base interface** - `storage/base.py`

### Integrations & Extensions

- [x] **CI/CD integration helpers** - `integration/cicd.py`
- [x] **Plugin discovery via entry points** -
      `cli/main.py:_discover_plugin_commands` reads the
      `evalyn.commands` entry-point group; lets external packages such
      as `evalyn-dashboard` register subcommands
- [x] **Lazy command loading** - only the invoked subcommand's module
      is imported; CLI startup stays fast (`cli/main.py`)
- [x] **OTel-off fast path for CLI** - CLI invocations short-circuit
      the OTel stack since they only read storage (`cli/main.py:294`)

### Export

- [x] **`export`** - JSON / CSV / Markdown / HTML output formats
- [x] **`export-for-annotation`** - structured export for external
      annotation tools

---

## Not Yet Implemented

These items appeared in the previous roadmap as "shipped" but no code,
CLI command, or test references them. They are kept here as future
candidates. Re-add to the shipped list (with file path) once built.

### Tracing & Observability

- [ ] Multi-modal tracing (image/audio/video capture)
- [ ] PII redaction for stored span payloads
- [ ] Trace sampling rate / priority sampling
- [ ] Trace payload compression (gzip / zstd) in SQLite
- [ ] W3C trace-context propagation across services
- [ ] Trace anonymization export
- [ ] Trace flame-graph rendering
- [ ] Trace dependency / lineage graphs
- [ ] Trace template / pattern matching
- [ ] Hot-path detection across traces
- [ ] Trace density heatmap
- [ ] Trace complexity score
- [ ] Provider SDK version tracking + compatibility report
- [ ] Trace correlation with external events (deploys, incidents)
- [ ] Distributed parallel-span aggregation
- [ ] Orphan span recovery
- [ ] Context propagation diagnostics
- [ ] Instrumentation toggle API
- [ ] Span collector statistics
- [ ] Instrumentation dry-run

### Additional Provider Instrumentors

- [ ] Cohere
- [ ] Mistral
- [ ] AWS Bedrock
- [ ] Azure OpenAI
- [ ] Groq
- [ ] Together AI
- [ ] Replicate

### Cost Intelligence

- [x] **Cost budget alerts** - `--max-cost` flag with 80% warning / 100%
      abort, partial results persisted (shipped 2026-05-23)
- [ ] Per-session budget limits in `evalyn.yaml`
- [ ] Per-phase cost attribution
- [ ] Cost comparison for trace replay

### Trace Lifecycle

- [ ] `archive-traces` (cold storage)
- [ ] `restore-traces`
- [ ] `purge` / GC commands
- [ ] `compact` storage maintenance
- [ ] `storage-stats`, `storage-tune`, `storage-verify`, etc.

### Dataset Expansions

- [x] **`dataset-diff` CLI** - wraps existing `diff_datasets()`
      (shipped 2026-05-23)
- [ ] `dataset-drift`, `dataset-decontaminate`, `dataset-xcontam`
- [ ] `dataset-health`, `dataset-audit`, `dataset-changelog`
- [ ] Cross-run A/B split, interleave, golden-set tooling

### Run Management

- [ ] `pin-run`, `bookmark`, `list-bookmarks`
- [ ] `bisect` regression bisection
- [ ] `verify-run`, `verify-manifest`
- [ ] `cleanup-runs`

### Reporting & Analytics

- [ ] `summarize-trace` (LLM natural-language trace summary)
- [ ] `mark-event` + event overlays on trend charts
- [ ] `classify-traces` / pattern coverage report
- [ ] `metric-history`, `compare-snapshots`

### Calibration & Judge Lifecycle

- [ ] `freeze-calibration`, `unfreeze-calibration`
- [ ] `compare-calibrations`
- [ ] `benchmark-judges`
- [ ] `tune-confidence`
- [ ] `pre-annotate`

### Rubric Tooling

- [ ] `test-rubric`, `rubric-import`, `rubric-export`
- [ ] `install-rubric-pack`

### Pipeline / Replay / What-If

- [ ] `replay` (re-run captured prompts against a different model)
- [ ] `simulate-and-eval`
- [ ] `what-if`, `sample-impact`, `compare-pipelines`

### Output, Diff & Snapshots

- [ ] `analysis-diff`, `diff-outputs`, `code-diff`
- [ ] `snapshot`, `compare-snapshots`, `dataset-snapshot-diff`

### Audit & Governance

- [ ] `audit-log`
- [ ] `rotate-key`
- [ ] `freeze-metrics` / `unfreeze-metrics`

### Distribution & Updates

- [ ] `self-update`, `update-pricing`
- [ ] `migrate-config`, `config-check`, `config-show`
- [ ] `new-project`, `tutorial`, `playground`, `docs`

### Diagnostics

- [x] **`doctor`** (shipped 2026-05-23)
- [ ] `timing-stats`, `check-context`,
      `check-instrumentation`, `check-compat`

---

## Known Bugs

- [x] **CrewAI `_SpanTracker` not thread-safe** (FIXED 2026-05-23) -
      `trace/instrumentation/providers/crewai.py` was reading the
      `_span_stack` ContextVar (default `None`) and doing
      `None + [span.id]`, raising `TypeError` in any fresh thread.
      Patched with `or []`. Verified via
      `tests/test_framework_instrumentors.py::TestCrewAIInstrumentor::test_span_tracker_thread_safety`.

- [ ] **Same `None + [...]` pattern is unguarded in 8 other provider
      instrumentors** - `autogen.py`, `dspy.py`, `google_adk.py`,
      `haystack.py`, `langgraph.py`, `llamaindex.py`,
      `semantic_kernel.py` all do `stack + [span.id]` without
      `or []`. No tests catch it because none exercise fresh threads
      without inherited context. Either fix each site or change the
      ContextVar default (audit needed first: see comment at
      `trace/context.py:30`).

- [ ] **LangChain test suite doesn't skip when `langchain_core` is
      missing** - 50+ tests in `tests/test_langchain_instrumentors.py`
      `assert False` instead of `pytest.skip` when the optional
      `langchain_core` dependency is not installed.

- [ ] **Sampling tests do bare `import numpy` inside test bodies** -
      4 tests in `tests/test_sampling.py` fail with
      `ModuleNotFoundError` instead of skipping when `numpy` is absent.

- [ ] **`test_list_calls_with_limit` doesn't account for boxed CLI
      banner** - `tests/test_cli_integration.py` filter logic predates
      the ASCII-banner rendering used by `list-calls`.

- [ ] **`test_evalyn_dashboard_runs_placeholder` hard-codes port 7401**
      - collides under `pytest-xdist`. Should bind to an ephemeral
      port.
