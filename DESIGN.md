# Evalyn System Design

This document describes the current system architecture and proposes designs for planned features from ROADMAP.md. It is a living design document - not implementation specs.

---

## 1. Current Architecture

### Layer Diagram

```
                           CLI Layer
                    (argparse + formatters)
                              |
              Pipeline Orchestrator (7 steps)
                              |
         +----------+---------+---------+----------+
         |          |         |         |          |
     Tracing   Evaluation  Metrics  Calibration  Analysis
         |          |         |         |          |
         +----------+---------+---------+----------+
                              |
                      Storage Layer
                     (StorageBackend)
                              |
                       SQLiteStorage
```

### Data Model Core

```
FunctionCall (root trace unit)
  |-- spans: List[Span]           # hierarchical tree
  |-- trace: List[TraceEvent]     # legacy flat log
       |
DatasetItem (evaluation unit)
  |-- input / output / metadata
       |
EvalUnit (discovered from spans)
  |-- EvalView (projected for metric evaluation)
       |
MetricResult (one per item-metric pair)
  |-- SpanMetricLink (attribution to specific spans)
       |
EvalRun (aggregated results)
  |-- CalibrationRecord (optimized prompts)
```

### Extension Points

| What | How | Where |
|---|---|---|
| Storage backend | `StorageBackend` Protocol | `storage/base.py` |
| LLM provider | `Instrumentor` ABC + `create_llm_client()` | `trace/instrumentation/base.py`, `utils/api_client.py` |
| Eval unit type | `EvalUnitBuilder` ABC | `evaluation/units/builders.py` |
| Metric | Function returning `Metric` | `metrics/factory.py` |
| Confidence method | `ConfidenceEstimator` ABC | `judges/confidence/base.py` |
| Prompt optimizer | `BaseOptimizer` ABC + registry entry | `calibration/base_optimizer.py` |
| Execution strategy | `ExecutionStrategy` ABC | `evaluation/execution.py` |
| Pipeline step | `PipelineStep` ABC | `cli/utils/pipeline.py` |

---

## 2. Tracing & Instrumentation Design

### Current: Three instrumentation strategies

```
InstrumentorType.MONKEY_PATCH   -- wraps SDK methods (OpenAI, Anthropic, Gemini)
InstrumentorType.OTEL_NATIVE    -- custom SpanProcessor (Google ADK)
InstrumentorType.HOOK_BASED     -- SDK callback adapter (Claude Agent SDK)
```

### Design: Multi-modal Tracing

**Problem:** Current spans only capture text. Images, audio, and video in LLM inputs/outputs are lost.

**Proposed approach:**
- Add `attachments: List[Attachment]` field to `Span` model
- `Attachment` dataclass: `{type: "image"|"audio"|"video", format: str, data: Optional[str], url: Optional[str], size_bytes: int}`
- Storage strategy: below 1MB inline as base64 in span attributes; above 1MB store as files in `data/attachments/` with URL reference
- Thumbnails: generate on capture for images (PIL lazy import), store as separate attachment
- Provider instrumentors extract multimodal content from provider-specific response formats

**Impact on storage:** New `attachments` column in `otel_spans` table (JSON blob). Migration: additive column.

### Design: Streaming Capture Enhancement

**Problem:** StreamingSpanWrapper captures final text but loses token-by-token timing.

**Proposed approach:**
- Add `token_timings: Optional[List[float]]` to Span attributes (relative ms from stream start per token)
- First-token latency (TTFT): `token_timings[0]` if available
- Streaming interruption: detect when stream ends before `finish_reason="stop"` (provider-specific)
- Keep backward compat: existing spans without timings continue to work

### Design: New Provider Instrumentors

**Pattern:** Each new provider follows the existing `Instrumentor` ABC:
1. Create `providers/<name>.py` implementing `Instrumentor`
2. Register in `InstrumentorRegistry` via `auto_instrument.py`
3. Add cost table entries to `_shared.py:COST_PER_1M_TOKENS`

**Specific considerations:**
- **AWS Bedrock / Azure OpenAI:** Wrapper APIs around underlying models. Detect via `base_url` pattern (like xAI detection). Map to underlying model for cost tracking.
- **Cohere / Mistral / Groq / Together AI / Replicate:** Standard REST APIs. Follow OpenAI instrumentor pattern.

### Design: Memory/RAG Tracing

**Problem:** Retrieval operations are captured as generic spans. Document content, relevance scores, and retrieval-to-generation links are lost.

**Proposed approach:**
- New span types: `retrieval` and `memory_read`/`memory_write` (already partially defined in `SpanType`)
- Retrieval spans capture: query, documents (with scores), vector store name, latency
- Link retrieval spans to downstream LLM calls via `parent_id` in span tree
- Convention: retrieval results stored in `span.attributes["retrieval.documents"]` following OpenInference

### Design: Trace Compression, Anonymization, Cost Breakdown

**Trace Compression:**
- Gzip compress `input`/`output` fields in SQLiteStorage when size exceeds threshold (default 10KB)
- Transparent decompress on read via `_loads()` helper
- Store compression flag in span metadata

**Trace Anonymization Export:**
- `evalyn export-traces --anonymize` pipeline:
  1. Replace text content with length-preserving lorem ipsum
  2. Preserve: span structure, timing, token counts, cost, model names
  3. Deterministic (seeded by trace ID) so same trace always anonymizes the same way

**Trace Cost Breakdown by Phase:**
- Classify spans into phases: `reasoning` (llm_call without tool results), `tool_use` (tool_call + tool_result pairs), `output` (final generation)
- Aggregate cost per phase using existing `calculate_cost()` from `_shared.py`
- Add to `show-trace` output as summary line

---

## 3. Evaluation Design

### Current: Two execution modes

```
OutcomeBuilder (default)              Non-default unit builders
  |                                      |
  v                                      v
ExecutionStrategy                    _run_unit_evaluation
(Sequential or Parallel)             (sequential only, no checkpoint)
  |                                      |
  v                                      v
metric.evaluate(call, item)          metric.evaluate_unit(view, item)
```

### Design: Span-Level Evaluation

**Problem:** Only `OutcomeBuilder` supports parallel execution and checkpointing. Unit-based evaluation is second-class.

**Proposed approach:**
- Unify: make `_run_unit_evaluation` use `ExecutionStrategy` by wrapping the unit discovery + projection into the `evaluate_fn` callback
- Evaluation task = `(item, unit, metric)` triple instead of `(item, metric)` pair
- Checkpoint granularity: per `(item_id, unit_id, metric_id)` triple

### Design: Pairwise Comparison (A vs B)

**Problem:** No built-in way to compare two model outputs on the same input.

**Proposed approach:**
- New `PairwiseEvalRunner` that takes two datasets (A and B) with shared item IDs
- Judge prompt template: "Given INPUT and two outputs A and B, which is better?"
- Result: `PairwiseResult(winner: "A"|"B"|"tie", confidence: float, reason: str)`
- Aggregation: Elo rating system computed from pairwise results
- Visualization: win/loss matrix, Elo ranking table

### Design: Evaluation Caching

**Problem:** Re-running unchanged metrics on unchanged data wastes tokens.

**Proposed approach:**
- Cache key: `sha256(item_hash + metric_id + prompt_hash + provider + model)`
- Cache storage: `eval_cache` table in SQLiteStorage
- Hit: return cached `MetricResult` directly
- Miss: evaluate, then store result
- `--no-cache` flag to force re-evaluation
- Cache invalidation: when metric spec changes (detected via prompt_hash mismatch)

### Design: Cross-Validation Evaluation

**Proposed approach:**
- `--cv-folds N` flag: split dataset into N stratified folds
- Run N evaluations, each holding out one fold
- Report per-fold metrics + aggregate mean/std
- Identify high-variance items (inconsistent across folds)

### Design: Batch Evaluation Enhancements

**Batch Progress Polling:**
- Poll loop with configurable interval (default 30s)
- Progress bar showing completion percentage + ETA
- `BatchJob` persistence to `.evalyn/batch_jobs/` for crash recovery

**Streaming Partial Results:**
- Process completed items as they arrive during batch wait
- Live-updating `RunAnalysis` from partial data
- Early termination: stop if N items show clear statistical signal

---

## 4. Metrics Design

### Current: 133 metrics (73 objective + 60 subjective)

```
Objective: pure function (FunctionCall, DatasetItem) -> MetricResult
Subjective: LLMJudge.score() -> parsed JSON -> MetricResult
```

### Design: Custom Metric DSL

**Problem:** Adding custom metrics requires Python code.

**Proposed approach:**
- YAML metric definitions in `evalyn.yaml`:
  ```yaml
  custom_metrics:
    - id: my_check
      type: objective
      expression: "len(output) < 500"
    - id: my_judge
      type: subjective
      prompt: "Evaluate whether {{output}} addresses {{input}}"
      rubric:
        - "Response must be factually accurate"
        - "Response must be complete"
  ```
- Objective: safe expression evaluation (restricted to string/math ops, no exec)
- Subjective: template variable interpolation ({{input}}, {{output}}, {{expected}}, custom vars)
- Hot-reload: detect YAML changes on `run-eval`, no code restart needed

### Design: Metric Composition

**Problem:** No way to combine metrics into weighted composites.

**Proposed approach:**
- `CompositeMetric` class wrapping a list of child metrics with weights
- Aggregation strategies: weighted_average, min, max, all_pass
- Pass threshold on composite score
- Drill-down: `show-run` displays child metric contributions

### Design: Multi-Modal Evaluation Metrics

**Image metrics:**
- CLIP score (requires `transformers` optional dep): measure image-text alignment
- Visual quality (via multimodal LLM judge): rate image quality 1-5
- OCR accuracy: extract text from generated images, compare to expected

**Audio metrics:**
- WER (word error rate) for transcription accuracy
- Speech clarity score via LLM judge on transcript

### Design: Agent-Specific Evaluation

**Tool Use Evaluation:**
- New metrics: `tool_selection_accuracy`, `tool_param_correctness`, `tool_error_recovery`
- Require `tool_call`/`tool_result` span pairs in trace
- Evaluated via `ToolUseBuilder` unit type

**Planning Evaluation:**
- New metrics: `plan_completeness`, `step_ordering`, `resource_efficiency`
- Require multi-step agent traces with planning spans
- Compare planned steps vs executed steps

**Reasoning Evaluation:**
- New metrics: `cot_faithfulness`, `logical_consistency`, `evidence_usage`
- Capture thinking/reasoning content from Anthropic thinking blocks, OpenAI reasoning tokens

---

## 5. Calibration Design

### Current: 9 optimizers, preamble-only optimization

```
CalibrationEngine
  1. compute_alignment(results, annotations) -> AlignmentMetrics
  2. analyze_disagreements() -> DisagreementAnalysis
  3. optimize(optimizer) -> PromptOptimizationResult
  4. validate(held_out) -> ValidationResult
```

### Design: Rubric Optimization

**Problem:** Currently only the preamble is optimized. The rubric is fixed.

**Proposed approach:**
- New optimization mode: `--optimize-rubric`
- LLM generates rubric candidates from pass/fail examples
- Rubric clarity scoring: can a different LLM interpret the rubric consistently?
- Constraint: human approval required before rubric changes take effect
- A/B test: compare original vs optimized rubric on held-out data

### Design: Few-Shot Example Selection

**Problem:** No systematic way to choose few-shot examples for judge prompts.

**Proposed approach:**
- Select from annotation pool using diversity-based selection (embedding distance)
- Leave-one-out evaluation: measure each example's contribution to alignment
- Dynamic k: find optimal number of examples (diminishing returns analysis)
- Store selected examples in `CalibrationRecord`

### Design: Judge Ensemble

**Problem:** Single judge model can be unreliable.

**Proposed approach:**
- `JudgeEnsemble` class wrapping N `LLMJudge` instances (same or different models)
- Aggregation: majority vote, weighted by calibration accuracy, or confidence-weighted
- Disagreement flagging: items where judges disagree go to human review queue
- Cost-aware: use cheap judge first, expensive only for uncertain items

---

## 6. Storage Design

### Current: Single SQLite file

```
SQLiteStorage
  |-- function_calls (JSON blobs for inputs/output/trace/spans)
  |-- eval_runs (JSON blob for metric_results, or relational table)
  |-- metric_results_rows (relational, batch-inserted)
  |-- annotations
  |-- otel_spans
  |-- span_metric_links
```

### Design: Cloud Storage Backend

**Problem:** SQLite doesn't scale for teams or production monitoring.

**Proposed approach:**
- New `PostgresStorage` implementing `StorageBackend`
- Same schema, adapted for Postgres types (JSONB instead of TEXT for JSON fields)
- Connection pooling via `psycopg2.pool`
- Migration path: `evalyn export-db --format postgres` for one-time migration
- Hybrid mode: SQLite for traces (local, fast), Postgres for eval_runs (shared, queryable)

### Design: Storage Compaction and Retention

**Compaction:**
- `evalyn compact` running `VACUUM` + `ANALYZE`
- Orphan cleanup: delete spans not linked to any function_call

**Retention:**
- `retention_days` in evalyn.yaml
- `evalyn purge --older-than 30d`
- Exempt pinned runs from auto-deletion

### Design: Encrypted Storage

**Proposed approach:**
- SQLCipher integration (drop-in SQLite replacement with AES-256)
- Key management: `EVALYN_DB_KEY` env var or system keyring
- Selective encryption: encrypt input/output payloads only, keep metadata queryable

---

## 7. Analysis Design

### Current: RunAnalysis + InsightsReport

```
analyze_run(run_dict) -> RunAnalysis
  |-- MetricStats per metric
  |-- ItemStats per item
  |-- overall_pass_rate, cost tracking

Insights functions:
  |-- compute_correlations()
  |-- detect_regressions()
  |-- analyze_features()
  |-- analyze_distributions()
```

### Design: Statistical Significance Testing

**Problem:** Run-to-run comparisons show deltas but no statistical confidence.

**Proposed approach:**
- Two-proportion z-test for pass rate differences
- Bootstrap confidence intervals (1000 resamples) for score means
- Effect size (Cohen's d) alongside p-values
- Auto-flag significant changes in `compare` output
- Require minimum sample size warning for underpowered comparisons

### Design: Cohort Analysis

**Proposed approach:**
- `--cohort-by metadata.field` flag on `analyze`
- Split RunAnalysis by metadata field values
- Per-cohort MetricStats + cross-cohort comparison table
- Identify worst-performing cohort with root cause suggestions

### Design: Failure Root Cause Analysis

**Proposed approach:**
- LLM-powered analysis of common patterns in failed items
- Feature attribution: which input features correlate with failure (using existing FeatureInsight)
- Failure clustering by root cause category: prompt, data, model, tool
- Actionable fix suggestions per failure cluster
- Integration with `cluster-failures` output

### Design: Web Dashboard

**Problem:** HTML reports are static files. No interactive exploration.

**Proposed approach:**
- Lightweight FastAPI server bundled with evalyn
- Routes: `/traces`, `/runs`, `/datasets`, `/metrics`, `/compare`
- Real-time: WebSocket for live eval progress
- Frontend: minimal Jinja2 templates + Chart.js (no heavy JS framework)
- Launch: `evalyn dashboard` command starting local server

---

## 8. Simulation Design

### Current: UserSimulator (LLM-based query generation)

```
UserSimulator
  |-- generate_queries(seeds, mode="similar"|"outlier")
  |-- configurable temperatures per mode
```

### Design: Persona-Based Simulation

**Proposed approach:**
- Built-in personas: novice_user, power_user, adversarial_attacker, non_native_speaker
- Custom persona definitions in evalyn.yaml:
  ```yaml
  personas:
    impatient_customer:
      description: "A frustrated customer with a billing issue"
      style: "short sentences, demanding tone, may include typos"
  ```
- Persona tag stored in generated item metadata for cohort analysis
- Mix multiple personas in a single simulation run

### Design: Multi-Turn Simulation

**Problem:** Current simulation generates single queries only.

**Proposed approach:**
- Configurable conversation length (2-10 turns)
- Follow-up generation: LLM reads agent response and generates next user message
- Conversation flow patterns: clarification, topic_shift, error_recovery, escalation
- Output: multi-turn traces stored as session-linked FunctionCalls

### Design: Adversarial Simulation

**Proposed approach:**
- Prompt injection attempts (jailbreak patterns from known datasets)
- Boundary inputs: empty, max_length, special characters, unicode edge cases
- Contradiction inputs: conflict with system prompt instructions
- Built-in adversarial templates with configurable intensity (mild/moderate/aggressive)

---

## 9. Sampling Design

### Current: 5 modes

```
apply_sampling(items, mode, limit, seed, dedup_threshold)
  |-- all: identity
  |-- random: seeded random.sample
  |-- diverse: farthest-point on embeddings
  |-- stratified: proportional by metadata groups
  |-- clustered: KMeans on embeddings
```

### Design: New Sampling Strategies

**Importance Sampling:**
- Weight by inverse pass rate from previous eval run
- Over-sample hard items, under-sample easy items
- Useful for calibration datasets

**Curriculum Sampling:**
- Order items easy-to-hard based on difficulty estimate
- Progressive evaluation: stop early if easy items fail
- Difficulty estimate: cross-run fail rate or input complexity heuristic

**Coverage-Aware Sampling:**
- Embedding-based coverage maximization
- Greedy algorithm: at each step, pick item most unlike current sample
- Report coverage metric: % of embedding space represented

---

## 10. CLI & Pipeline Design

### Current: 7-step pipeline

```
1. BuildDatasetStep     -> dataset.jsonl
2. SuggestMetricsStep   -> metrics.json
3. InitialEvalStep      -> eval_runs/
4. AnnotationStep       -> annotations.jsonl (interactive)
5. CalibrationStep      -> calibrations/
6. CalibratedEvalStep   -> eval_runs/ (re-eval)
7. SimulationStep       -> simulations/
```

### Design: Custom Pipeline Definitions

**Proposed approach:**
- Pipeline definition in evalyn.yaml:
  ```yaml
  pipelines:
    quick-check:
      steps: [dataset, metrics, eval, analyze]
    full-audit:
      steps: [dataset, metrics, eval, annotate, calibrate, re-eval, simulate, analyze, insights]
    ci-gate:
      steps: [dataset, metrics, eval]
      abort_on: "overall_pass_rate < 0.8"
  ```
- `evalyn one-click --pipeline quick-check`
- Each step references a registered `PipelineStep` by name

### Design: Interactive TUI Mode

**Proposed approach:**
- Optional `textual` dependency for rich terminal UI
- Views: trace list, run list, metric dashboard, item detail
- Keyboard navigation: j/k scroll, enter drill-down, q quit
- Real-time eval progress with per-metric status bars

### Design: Shell Completion

**Proposed approach:**
- `argcomplete` integration
- Complete: command names, flag names, run IDs (from storage), dataset paths (from filesystem)
- Install: `evalyn --install-completion`

---

## 11. Interoperability Design

### Design: Phoenix/Langfuse Export

**Proposed approach:**
- `evalyn export-traces --format phoenix` producing Phoenix-compatible JSONL
- Map Evalyn span types to OpenInference conventions
- Preserve hierarchy via `parent_id` linkage
- Include eval scores as span annotations

### Design: Trace Import

**Proposed approach:**
- `evalyn import-traces --format phoenix/langfuse/otel`
- `SpanConverter.from_external(format, data)` extension
- Map external span types to Evalyn types via conventions.py
- Dedup against existing traces by span ID hash

---

## 12. Security & Governance Design

### Design: PII Redaction

**Proposed approach:**
- Regex patterns for: emails, phone numbers, SSNs, credit cards
- Named entity recognition (optional `spacy` dependency) for names, addresses
- Configurable strategy: mask (`***`), hash (SHA256 prefix), or remove
- Pre-storage hook in `SQLiteStorage.store_call()` and `SQLiteSpanExporter`
- Redaction audit: track what was redacted per trace

### Design: Audit Trail

**Proposed approach:**
- Append-only `audit.jsonl` in `.evalyn/`
- Record: user, timestamp, command, args, config hash, result summary
- `evalyn audit-log` command for viewing with filters
- Tamper detection: SHA256 chain linking each entry to previous

### Design: API Key Rotation

**Proposed approach:**
- Accept multiple keys per provider in evalyn.yaml (primary + fallback)
- Automatic fallback: on 401/403, try next key
- `evalyn rotate-key --provider gemini` to update and verify connectivity

---

## 13. Scale & Performance Design

### Design: Large Dataset Optimization

**Proposed approach:**
- Streaming evaluation: process items without loading full dataset into memory
- Chunked writes: batch metric result storage (already done: 1000/batch)
- Progress checkpointing every N items (currently only on interrupt)
- Memory monitoring: warn when approaching system limits (psutil optional dep)

### Design: SQLite Full-Text Search

**Proposed approach:**
- FTS5 index on `function_calls.inputs` and `function_calls.output`
- `evalyn search "user asked about refund"` finding matching traces
- Integration with `build-dataset` for content-based curation

### Design: Connection Pooling

**Proposed approach:**
- Already uses WAL mode + thread-local connections
- Add connection pool with configurable max size
- Health check: verify connection before reuse
- Auto-enable `PRAGMA journal_mode=WAL` on first connection

---

## 14. Programmatic SDK Design

### Design: Python API

**Problem:** All functionality requires CLI. No way to call from Python code or test suites.

**Proposed approach:**
```python
import evalyn

# Run evaluation
run = evalyn.run(dataset="path/to/dataset", metrics="path/to/metrics.json")

# Analyze
analysis = evalyn.analyze(run)
print(analysis.overall_pass_rate)

# Compare
comparison = evalyn.compare(run_a, run_b)

# Async
run = await evalyn.run_async(dataset, metrics)
```

- Thin wrappers around existing engine classes (EvalRunner, analyze_run)
- Return typed objects (EvalRun, RunAnalysis), not dicts
- No `fatal_error` / `sys.exit` - raise exceptions
- pytest plugin: `@pytest.mark.evalyn(metrics=["helpfulness"])`

---

## Design Principles

1. **Local-first:** SQLite by default, no cloud dependency for core functionality
2. **Incremental adoption:** `@eval` decorator is the only required entry point; everything else is optional
3. **Provider-agnostic:** Judge model is configurable; defaults to cheapest (Gemini Flash Lite)
4. **Backward-compatible:** Old datasets, old APIs, old configs continue to work via aliases and migration
5. **Lazy loading:** Heavy dependencies (sentence-transformers, spacy, gepa) imported only when needed
6. **Pure analysis:** Analysis functions are stateless and testable; no side effects
7. **Extension over modification:** New metrics, providers, optimizers added via registries, not by editing core

*Last updated: 2026-03-27*
