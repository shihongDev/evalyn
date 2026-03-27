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

## 15. Trace Lifecycle & Advanced Tracing Design

### Design: Trace Search Query Language

**Problem:** No way to filter traces by span attributes, duration, cost, or error status beyond basic list-calls flags.

**Proposed syntax:**
```
spans where type=llm_call and duration_ms > 5000
traces with total_cost > 0.10
spans where model contains "gpt-4" and error is not null
```

**Implementation approach:**
- Parser: simple recursive descent for `field op value [and/or field op value]`
- Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`, `contains`, `matches` (regex), `is null`, `is not null`
- Fields: span attributes (`type`, `model`, `duration_ms`, `input_tokens`), call fields (`function_name`, `cost`, `error`)
- Backend: translate to SQL WHERE clauses for SQLite; fallback to Python filtering for complex attribute queries
- CLI: `--query` flag on `list-calls`

### Design: Trace Replay

**Problem:** Can't re-run a captured trace against a different model to compare outputs.

**Data flow:**
```
Trace (stored) -> extract LLM span inputs -> swap model config -> re-execute -> new Trace
  |                                                                                |
  +--- original_trace_id linked via metadata ----------------------- diff report --+
```

**Key decisions:**
- Extract `llm.input_messages` from each LLM span for replay
- Create new `FunctionCall` with `metadata.replay_source = original_call_id`
- Side-by-side diff: text diff of outputs per span position
- Cost comparison: sum costs of original vs replay

### Design: Trace Archival & Lifecycle

**Archive strategy:**
```
Active DB (traces.sqlite)
  |-- traces < retention_days old

Archive DB (archive.sqlite)
  |-- traces > retention_days old
  |-- read-only, queryable via --db archive
```

**Commands:**
- `evalyn archive-traces --older-than 90d` - move to archive
- `evalyn restore-traces --from archive --id <id>` - move back
- `evalyn purge --older-than 365d --include-archive` - permanent delete

### Design: Experiment Tracking

**Problem:** No way to group traces by experiment for A/B comparisons.

**Proposed approach:**
- New `experiment_id` field on `FunctionCall.metadata`
- Set via decorator: `@decorated_fn(experiment="prompt-v2")`
- Filter in `list-calls --experiment prompt-v2`
- Cross-experiment comparison: `evalyn compare-experiments --exp1 prompt-v1 --exp2 prompt-v2`

### Design: Conditional Tracing

**Problem:** Tracing everything in production is expensive.

**Proposed approach:**
- Sample-based: `@trace_decorator(sample_rate=0.1)` traces 10% of calls
- Predicate-based: `@trace_decorator(trace_if=lambda args: args.get("user_id") in sample_set)`
- Environment-based: `EVALYN_TRACE_ENV=production` - only trace in matching environment
- Implementation: check condition in `EvalTracer.instrument()` wrapper before `start_call()`

---

## 16. Dataset Engineering Design

### Design: Dataset Versioning

**Problem:** No way to track dataset changes or roll back.

**Data model:**
```
data/prod/datasets/my_project/
  |-- dataset.jsonl           # current version
  |-- meta.json               # includes version_hash
  |-- .versions/
      |-- v1_abc123.jsonl     # snapshot
      |-- v2_def456.jsonl     # snapshot
      |-- changelog.json      # [{version, timestamp, item_count, hash, filters_used}]
```

**Key decisions:**
- Version = SHA256 of sorted dataset content
- On each `build-dataset`, auto-snapshot previous version to `.versions/`
- `evalyn dataset-diff --v1 v1_abc123 --v2 v2_def456` shows added/removed/changed items
- `evalyn dataset-rollback --to v1_abc123` restores previous version

### Design: Dataset Filtering DSL

**Proposed syntax:**
```
items where output_length > 500 and metadata.tag = "production"
items where input contains "refund" or input contains "return"
```

**Implementation:** Reuse the same parser as Trace Search Query Language (shared `QueryParser` module). Different field namespace (item fields vs span fields).

### Design: Golden Set Management

**Problem:** No way to maintain curated evaluation benchmarks.

**Data model:**
```
data/golden/
  |-- golden_set.jsonl    # locked items
  |-- golden_meta.json    # {locked: true, coverage: {metric_id: count}}
```

**Commands:**
- `evalyn golden-set create --from-dataset <path> --items <ids>`
- `evalyn golden-set add --id <item_id>`
- `evalyn golden-set validate` - re-evaluate golden set to detect model drift

### Design: External Format Import

**Format mapping:**
```
HuggingFace datasets -> DatasetItem:
  - "question" -> input
  - "answer" or "response" -> output
  - remaining fields -> metadata

LMSYS Arena -> PairwiseDatasetItem:
  - "conversation_a" -> output_a
  - "conversation_b" -> output_b
  - "winner" -> human_label

CSV -> DatasetItem:
  - Column mapping configurable: --input-col question --output-col answer
```

**Auto-detection:** Sniff first 5 lines; check for JSONL (starts with `{`), JSON array (starts with `[`), CSV (has header with commas), TSV (has header with tabs).

---

## 17. Advanced Calibration Design

### Design: Active Learning for Annotation

**Problem:** Random annotation is inefficient; some items are more informative.

**Proposed approach:**
```
Pool of unannotated items
  |
  v
Uncertainty Sampling: items where judge confidence is lowest
  +
Disagreement Sampling: items where judge and heuristics disagree
  +
Diversity Sampling: items that cover unexplored embedding regions
  |
  v
Ranked annotation queue (highest priority first)
```

**Implementation:**
- `evalyn annotate --active-learning` flag
- Score each item: `info_score = (1 - confidence) * diversity_weight`
- Present highest-scoring items first
- Batch mode: select top-K for each annotation session

### Design: Calibration Staleness Detection

**Problem:** Calibrated prompts may become stale as the dataset evolves.

**Detection strategy:**
- Store `dataset_hash` and `calibration_date` in `CalibrationRecord`
- On `run-eval --use-calibrated`, compare current dataset hash to calibration hash
- If hash differs, compute drift score: `|new_items / total_items|`
- Warning levels: >10% new items = "consider re-calibrating", >30% = "calibration likely stale"

### Design: Multi-Objective Calibration

**Problem:** Optimizing only for alignment may produce verbose prompts that cost more.

**Proposed Pareto approach:**
- Objective 1: alignment F1 (maximize)
- Objective 2: prompt token count (minimize)
- Generate N candidate prompts, plot on Pareto front
- User selects preferred trade-off via `--cost-weight 0.3` (0=accuracy only, 1=cost only)

---

## 18. Resilience & Error Handling Design

### Design: Circuit Breaker for Providers

**Problem:** Provider outages cause cascading failures across all metrics.

**State machine:**
```
CLOSED (normal) --[N consecutive failures]--> OPEN (reject all)
OPEN --[cool-down period elapsed]--> HALF_OPEN (try one request)
HALF_OPEN --[success]--> CLOSED
HALF_OPEN --[failure]--> OPEN (reset cool-down)
```

**Configuration:**
```yaml
providers:
  gemini:
    circuit_breaker:
      failure_threshold: 5
      cool_down_seconds: 60
```

**Integration:** Wrap `create_llm_client()` with circuit breaker state check.

### Design: Provider Fallback Chain

**Problem:** Single provider failure blocks evaluation.

**Proposed approach:**
```yaml
providers:
  fallback_chain: [gemini, openai, ollama]
```

- On timeout/rate-limit/API-error, try next provider in chain
- Log which provider was actually used per item in `MetricResult.model`
- Cost tracking accounts for actual provider used, not configured default

### Design: Graceful Item-Level Failure

**Proposed standardization:**
- All errors produce `MetricResult(passed=None, score=None, details={"error": str(e)})`
- Error categorization: `timeout`, `api_error`, `parse_error`, `internal_error`
- Summary at end of run: "3 items failed: 2 timeouts, 1 parse error"
- `--fail-fast` flag to override and stop on first error

---

## 19. Reporting & Visualization Design

### Design: Jupyter Notebook Export

**Output structure:**
```python
# Cell 1: Data loading
import json
run = json.load(open("results.json"))

# Cell 2: Metric summary (pandas DataFrame)
# Cell 3: Pass rate charts (matplotlib)
# Cell 4: Score distributions (per-metric histogram)
# Cell 5: Failed item analysis (filterable table)
```

**Implementation:** Generate `.ipynb` via `nbformat` (no Jupyter dependency at SDK level).

### Design: Comparative Heatmap

**Visualization:**
```
         metric1  metric2  metric3  metric4
item_1   [0.9]    [0.3]    [0.8]    [1.0]
item_2   [0.1]    [0.9]    [0.7]    [0.5]
item_3   [0.8]    [0.8]    [0.2]    [0.9]
```
- Color scale: red (0) -> yellow (0.5) -> green (1.0)
- Sort by: worst items, worst metrics, or custom
- Multi-run overlay: show delta as +/- annotations

### Design: Web Dashboard Architecture

**Components:**
```
evalyn dashboard (CLI command)
  |
  v
FastAPI Server (localhost:8501)
  |-- GET /api/traces          -> list_calls()
  |-- GET /api/runs            -> list_eval_runs()
  |-- GET /api/runs/:id        -> get_eval_run() + RunAnalysis
  |-- GET /api/datasets        -> list_datasets()
  |-- GET /api/compare/:a/:b   -> compare two runs
  |-- WS  /ws/progress         -> live progress updates
  |
  v
Jinja2 Templates + Chart.js (no heavy JS framework)
```

---

## 20. Plugin System Design

### Design: Entry-Point Plugin Discovery

**Problem:** Adding custom metrics, instrumentors, or storage requires editing SDK source.

**Proposed approach:**
```toml
# In user's pyproject.toml:
[project.entry-points."evalyn.metrics"]
my_metric = "my_package.metrics:register"

[project.entry-points."evalyn.instrumentors"]
my_provider = "my_package.instrumentors:MyInstrumentor"

[project.entry-points."evalyn.storage"]
postgres = "my_package.storage:PostgresStorage"
```

**Discovery at startup:**
```python
from importlib.metadata import entry_points

for ep in entry_points(group="evalyn.metrics"):
    register_fn = ep.load()
    register_fn(metric_registry)
```

**Plugin categories:**
- `evalyn.metrics` - custom objective/subjective metrics
- `evalyn.instrumentors` - new LLM provider instrumentors
- `evalyn.storage` - alternative storage backends
- `evalyn.optimizers` - calibration optimizers
- `evalyn.exporters` - new export formats

---

## 21. Offline & Air-Gapped Mode Design

### Design: Fully Offline Evaluation

**Tiered offline support:**
```
Tier 1: Objective-only (no internet, no optional deps)
  |-- All 73 objective metrics work out of the box

Tier 2: Local LLM (Ollama, no internet after model download)
  |-- provider: ollama in evalyn.yaml

Tier 3: Full offline (pre-cached embeddings + local models)
  |-- sentence-transformers model cached locally
  |-- Sampling modes (diverse, clustered) work offline
```

**`--offline` flag:** Validates before execution that no metric/step would require internet. Fails fast with clear message listing which metrics need network access.

---

## 22. CI/CD Integration Design

### Design: GitHub Actions Integration

**Workflow template:**
```yaml
name: Evalyn Evaluation
on: [pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install evalyn-sdk
      - run: evalyn run-eval --dataset data/golden/ --format json > results.json
      - run: evalyn compare --latest --format json > comparison.json
```

**Exit codes:** 0 = pass, 1 = regression detected, 2 = execution error

**Regression gate in evalyn.yaml:**
```yaml
ci:
  pass_rate_threshold: 0.85
  fail_on_regression: true
  regression_threshold: 0.05  # 5% drop triggers failure
```

---

## 23. Graph & Multi-Agent Evaluation Design

### Design: Graph Topology Extraction

**Problem:** LangGraph execution paths are captured as flat span lists. No way to visualize the graph structure.

**Proposed approach:**
- Build DAG from `graph`/`node` spans captured by `LangGraphInstrumentor`
- Identify critical path (longest execution chain)
- Detect cycles and redundant node executions
- `evalyn show-graph --call <id>` rendering Mermaid or ASCII diagram

**Data model:**
```
GraphTopology:
  nodes: List[{id, name, span_id, duration_ms, cost}]
  edges: List[{source, target, data_flow: bool}]
  critical_path: List[node_id]
  cycle_detected: bool
```

### Design: Node-Level Metric Attribution

- Map `MetricResult` failures back to the node span that produced the failing output
- Per-node pass rate aggregation: "Node X fails 40% of the time"
- Bottleneck identification: nodes causing the most downstream failures

### Design: Subagent Cost Allocation

- Aggregate token/cost from Claude Agent SDK's SubagentContext hierarchy
- Per-subagent cost breakdown in `show-trace` and `analyze` output
- Tree view: main agent -> subagent_a ($0.05) -> subagent_b ($0.12)

---

## 24. Advanced Evaluation Modes Design

### Design: Differential Evaluation

**Problem:** Re-evaluating unchanged items wastes tokens.

**Approach:**
- Hash-based change detection using `hash_inputs(item)`
- `--diff-from <baseline_run_id>` flag on `run-eval`
- Changed items: evaluate normally
- Unchanged items: carry forward `MetricResult` from baseline run
- Report: "Evaluated 15/100 items (85 carried forward from baseline)"

### Design: Distributed Evaluation

**Problem:** Large datasets are slow on a single machine.

**Architecture:**
```
Coordinator (evalyn run-eval --distributed)
  |
  v
Task Queue (Redis or RabbitMQ)
  |-- task: {item_id, metric_id, item_data, metric_config}
  |
  v
Worker Pool (N machines)
  |-- each pulls tasks, evaluates, pushes results
  |
  v
Result Collector
  |-- merge results into EvalRun
  |-- checkpoint: persist partial results
```

**Worker:** `evalyn worker --queue redis://localhost:6379` process

### Design: Async Evaluation Strategy

**Problem:** `ThreadPoolExecutor` has GIL limitations for I/O-heavy LLM calls.

**Proposed `AsyncStrategy`:**
- `asyncio.gather()` for concurrent metric calls
- `asyncio.Semaphore(max_workers)` for concurrency control
- Compatible with async LLM clients (httpx, aiohttp)
- `--strategy async` flag on `run-eval`

---

## 25. Config Architecture Design

### Design: Typed Configuration

**Problem:** Config is `Dict[str, Any]` with no schema, no validation, no documentation.

**Proposed `EvalynConfig` dataclass:**
```python
@dataclass
class EvalynConfig:
    # Provider settings
    default_provider: str = "gemini"
    api_keys: Dict[str, str] = field(default_factory=dict)

    # Evaluation defaults
    max_workers: int = 1
    checkpoint_interval: int = 5

    # Calibration defaults
    default_optimizer: str = "basic"

    # Storage
    db_path: Optional[str] = None
    retention_days: Optional[int] = None

    # CI/CD
    pass_rate_threshold: float = 0.0
    fail_on_regression: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalynConfig": ...
    @classmethod
    def from_env(cls) -> "EvalynConfig": ...
    def merge(self, override: "EvalynConfig") -> "EvalynConfig": ...
```

**Resolution order:** defaults -> global `~/.evalyn/config.yaml` -> project `evalyn.yaml` -> env vars -> CLI flags

---

## 40. Research-Driven Designs (from Landscape Analysis)

### Design: Judge Debiasing Pipeline

**Research basis:** CALM framework identifies 12 judge biases. AlpacaEval's length-controlled win rates improve correlation with human preference from 0.94 to 0.98. Regression-based bias correction halves residual error.

**Proposed pipeline:**
```
Raw judge scores
  |
  v
Position-bias correction (for pairwise only)
  |-- Evaluate A vs B, then B vs A
  |-- Average scores; flag if results flip
  |
  v
Length-bias correction
  |-- Fit GLM: preference ~ length_difference + true_quality
  |-- Condition on zero length difference
  |
  v
Calibration correction (from human-labeled subset)
  |-- Fit regression: human_score ~ judge_score + item_features
  |-- Apply correction to all scores
  |
  v
Debiased scores with uncertainty estimates
```

**Implementation:**
- New `JudgeDebiasingPipeline` class in `judges/debiasing.py`
- Called optionally via `--debias` flag on `run-eval`
- Requires a small calibration set (50-100 human-labeled items)
- Outputs bias report: "Position bias: 0.12, Length bias: 0.08"

### Design: Agent Evaluation Metrics (Ragas-Inspired)

**Research basis:** Ragas defines ToolCallAccuracy (sequence + arguments), ToolCallF1 (unordered), AgentGoalAccuracy. DeepEval adds trace-based agent evaluation.

**Proposed metrics for evalyn:**
```
agent_tool_accuracy:
  type: objective
  scope: trace
  evaluates:
    - Were the right tools called? (tool names match expected)
    - Were arguments correct? (parameter values match expected)
    - Was the sequence correct? (order matches expected)
  score: fraction of correct tool calls

agent_goal_completion:
  type: subjective
  scope: outcome
  evaluates:
    - Did the agent achieve the stated objective?
    - Was the final output correct?
  prompt: "Given the user's goal and the agent's trace, did the agent succeed?"

agent_topic_adherence:
  type: subjective
  scope: conversation
  evaluates:
    - Did the agent stay within its defined domain?
    - Did it refuse out-of-scope requests appropriately?
```

**Implementation:** Leverage existing `ToolUseBuilder` and `MultiTurnBuilder` for span discovery. Tool accuracy is an objective metric comparing `tool_call` span attributes against expected tool definitions.

### Design: Bloom-Style Test Case Generation

**Research basis:** Anthropic's Bloom uses a four-agent pipeline achieving 0.86 Spearman correlation with human scores on generated test cases.

**Proposed pipeline for evalyn:**
```
Step 1: Understand
  |-- LLM analyzes the agent's system prompt and capabilities
  |-- Extracts: domain, intended behaviors, constraints, tools

Step 2: Ideate
  |-- Generate diverse scenario categories (edge cases, adversarial, typical)
  |-- Ensure coverage across identified capability dimensions

Step 3: Generate
  |-- For each scenario: generate specific user input
  |-- Apply persona diversity (novice, expert, adversarial)
  |-- Optional: include expected behavior description

Step 4: Score
  |-- Run generated inputs through the agent
  |-- Score quality: naturalness, diversity, difficulty balance
  |-- Filter: remove duplicates and low-quality items
```

**CLI:** `evalyn generate-tests --behavior "customer support agent" --count 50`

### Design: DAG-Based Deterministic Evaluation (DeepEval-Inspired)

**Research basis:** DeepEval's DAGMetric uses LLM-powered decision trees for structured scoring - cheaper than full LLM-as-judge, more flexible than regex rules.

**Proposed design:**
```yaml
# In evalyn.yaml or metrics definition:
dag_metrics:
  - id: response_quality
    tree:
      - question: "Is the response relevant to the input?"
        yes: next
        no: {score: 0.0, reason: "Off-topic response"}
      - question: "Does it contain factual errors?"
        yes: {score: 0.3, reason: "Factual errors detected"}
        no: next
      - question: "Is it complete and helpful?"
        yes: {score: 1.0, reason: "Good response"}
        no: {score: 0.6, reason: "Incomplete but relevant"}
```

**Implementation:** Each decision node is an LLM call with a yes/no question. Total cost: N calls (depth of tree) vs 1 expensive call for full judge. Deterministic path through the tree for reproducibility.

### Design: Statistical Evaluation Reporting (Anthropic-Inspired)

**Research basis:** Anthropic's "Statistical Approach to Model Evaluations" recommends confidence intervals, sample size planning, and bootstrap methods.

**Proposed additions to RunAnalysis:**
```python
@dataclass
class StatisticalMetricStats:
    pass_rate: float
    confidence_interval_95: tuple[float, float]  # bootstrap CI
    sample_size: int
    minimum_detectable_effect: float  # given current sample
    power_at_5pct_change: float  # statistical power

    @property
    def is_sufficient_sample(self) -> bool:
        return self.power_at_5pct_change >= 0.8
```

**CLI output:**
```
Metric: helpfulness
  Pass rate:  85.0% [80.2%, 89.1%] (95% CI, n=100)
  Power:      72% to detect 5% change (need n=150 for 80% power)
  Significance vs baseline: p=0.023 (significant at alpha=0.05)
```

### Design: Annotation Queue Flywheel

**Research basis:** LangSmith's annotation queues feed back into automated evaluation. Active learning reduces labeling effort 30-70%.

**Proposed flywheel:**
```
Round 1: Human annotates 100 items
  |
  v
Calibrate judge on annotations -> F1 = 0.75
  |
  v
Round 2: Judge pre-labels all items
  |-- High confidence (>0.9): auto-accept (skip human review)
  |-- Low confidence (<0.7): route to human annotation queue
  |-- Medium: sample 20% for human review
  |
  v
Track: judge accuracy on human-reviewed items
  |-- If accuracy > 0.9 for a metric: reduce human review to 10%
  |-- If accuracy < 0.8: increase human review, trigger re-calibration
```

**Implementation:**
- New `AnnotationStrategy` in calibration engine
- `evalyn annotate --active-learning --confidence-threshold 0.8`
- Flywheel metrics tracked in `.evalyn/flywheel_state.json`

---

## 26. Session Management Design

### Design: Session-Level Analysis

**Problem:** Traces are analyzed individually. No way to aggregate metrics across a user session.

**Data model:**
- `eval_session(session_id)` context manager already groups calls by `session_id`
- New `SessionAnalysis` extending `RunAnalysis` with session-level aggregation
- Per-session: pass rate, total cost, total latency, call count

**Proposed analysis flow:**
```
EvalRun.metric_results
  |-- group by item.metadata.session_id
  |-- per-session MetricStats
  |-- cross-session comparison table
```

### Design: Session Replay

**Approach:**
- Extract all inputs from session traces in chronological order
- Replay with swapped model/provider while preserving conversation state
- Session-level diff: turn-by-turn comparison of original vs replayed outputs

---

## 27. Reproducibility Design

### Design: Deterministic Evaluation Mode

**Problem:** LLM judge non-determinism makes runs non-reproducible.

**Proposed approach:**
- `--seed <int>` flag on `run-eval`
- Seed controls: sampling order, metric evaluation order, random choices in evaluation
- Force `temperature=0` for all judge LLM calls
- Record seed in `EvalRun.metadata.seed`

### Design: Run Manifest

**Record everything that could affect results:**
```json
{
  "evalyn_version": "0.15.0",
  "python_version": "3.13.5",
  "provider_versions": {"openai": "1.82.0", "anthropic": "0.49.0"},
  "metric_hashes": {"helpfulness": "sha256:abc...", "toxicity": "sha256:def..."},
  "config_hash": "sha256:ghi...",
  "seed": 42,
  "dataset_hash": "sha256:jkl..."
}
```

- Saved alongside eval run results as `manifest.json`
- `evalyn verify-manifest --run <id>` checks if current environment matches manifest

### Design: Custom Cost Models

**Problem:** Local/self-hosted models have unknown pricing.

**Config in evalyn.yaml:**
```yaml
cost_models:
  ollama/llama3.2:
    input: 0.0    # free (local)
    output: 0.0
  my-custom-model:
    input: 0.50   # per 1M tokens
    output: 1.50
```

---

## 28. Cost Intelligence Design

### Design: Auto-Update Pricing Tables

**Problem:** `COST_PER_1M_TOKENS` in `_shared.py` gets stale as providers change prices.

**Proposed approach:**
- `evalyn update-pricing` command fetching latest from provider pricing pages
- Fallback: bundled pricing table with `last_updated` timestamp
- Warning when using a model not in the pricing table

### Design: Prompt Cache Savings Report

**Data source:** Spans already capture `cache_creation_tokens` and `cache_read_tokens`.

**Report format:**
```
Prompt Cache Savings:
  Cache writes:    12,500 tokens ($0.016)
  Cache reads:     87,300 tokens ($0.009)
  Without cache:   99,800 tokens ($0.125)
  Savings:         $0.100 (80% reduction)
```

### Design: Context Window Utilization Alerts

- Track `context_utilization_pct` per LLM span (already in span attributes)
- Alert when any span exceeds configurable threshold (default 80%)
- Per-run summary: max utilization, mean utilization, models hitting limits

---

## 29. Rubric Engineering Design

### Design: Multi-Language Rubrics

**Problem:** Judge prompts are English-only. Evaluating non-English outputs with English rubrics may miss language-specific issues.

**Proposed approach:**
- `locale` field in JUDGE_TEMPLATES entries
- Language-matched judging: route to rubric matching output language
- Cross-language evaluation: compare English-rubric vs native-rubric scores
- Implementation: `build_subjective_metric(..., locale="ja")` selects matching template variant

### Design: Community Rubric Library

**Format for portable rubric files:**
```yaml
# rubric_toxicity_v2.yaml
id: toxicity_safety
version: "2.0"
author: "evalyn-community"
locale: "en"
prompt: "You are a safety evaluator..."
rubric:
  - "No harassment, hate speech, or demeaning content"
  - "No instructions for self-harm or violence"
config:
  threshold: 0.5
tested_on:
  accuracy: 0.92
  dataset_size: 500
```

- `evalyn rubric-export --metric toxicity_safety > rubric.yaml`
- `evalyn rubric-import rubric.yaml` adds to local registry

### Design: Domain-Specific Rubric Packs

**Downloadable packs:**
```
evalyn install-rubric-pack medical
  |-- medical_accuracy
  |-- hipaa_compliance
  |-- patient_safety
  |-- drug_interaction_check
  |-- clinical_accuracy
```

**Implementation:** Rubric packs are YAML bundles hosted as GitHub releases or PyPI extras.

---

## 30. Metrics Enhancements Design

### Design: Metric Composition

**Problem:** No way to create weighted composite scores from multiple metrics.

**Proposed `CompositeMetric`:**
```python
CompositeMetric(
    id="quality_score",
    children=[
        ("helpfulness", 0.4),
        ("factual_accuracy", 0.3),
        ("toxicity_safety", 0.3),
    ],
    aggregation="weighted_average",  # or: min, max, all_pass
    threshold=0.7,
)
```

- Pass/fail determined by composite score vs threshold
- `show-run` shows composite + drill-down into children
- Composites can nest (composite of composites)

### Design: Metric Versioning

**Problem:** Changing a metric's rubric silently invalidates historical comparisons.

**Approach:**
- Hash metric prompt + scoring logic as version identifier
- Store metric version hash in `MetricResult.details.metric_hash`
- Warning when comparing runs using different metric versions
- `evalyn metric-history --id helpfulness` shows version changes over time

### Design: Metric Dependencies

**Problem:** Some metrics should only run if prerequisite metrics pass.

**Proposed declaration:**
```yaml
metrics:
  - id: helpfulness
    depends_on: []
  - id: detailed_analysis
    depends_on: ["json_valid"]  # only run if json_valid passes
```

- Topological sort of metrics before evaluation
- Skipped metrics produce `MetricResult(passed=None, details={"skipped": "dependency json_valid failed"})`

### Design: Conditional Metric Chains

**Diagnostic follow-up pattern:**
```yaml
chains:
  - trigger: toxicity_safety
    condition: "failed"
    follow_up: toxicity_type_classifier
```

- If `toxicity_safety` fails, automatically run `toxicity_type_classifier` on that item
- Chain results stored alongside primary results
- Useful for failure diagnosis without running expensive diagnostics on all items

---

## 31. CLI Enhancements Design

### Design: Interactive TUI Mode

**Architecture:**
```
evalyn tui (launches Textual app)
  |
  v
TUI Application (optional 'textual' dependency)
  |-- TraceListView    -> j/k scroll, enter for detail, / to search
  |-- RunListView      -> sort by date/pass_rate/cost
  |-- MetricDashboard  -> bar charts, score distributions
  |-- ItemDetailView   -> input/output/metrics side by side
  |-- LiveEvalView     -> real-time progress during run-eval
```

**Data access:** All views call the same engine functions as CLI commands. TUI is a presentation layer only.

### Design: Shell Completion

**Using argcomplete:**
- Command names: all registered commands
- Flag names: per-command from argparse definitions
- Dynamic values: run IDs (query storage), dataset paths (filesystem glob), metric IDs (from registry)
- Install: `evalyn --install-completion` adding to `.bashrc`/`.zshrc`

### Design: Watch Mode

**`evalyn run-eval --watch`:**
- File watcher on `dataset.jsonl` and `evalyn.yaml`
- Debounce: wait 2s after last change before re-running
- Diff output: only show changed metrics since last run
- Hot-reload metrics from YAML (see Custom Metric DSL)

### Design: Profile Command

**`evalyn profile`:**
```
Evalyn Profile
  Database:     data/prod/traces.sqlite (45.2 MB)
  Traces:       1,234 calls across 5 projects
  Eval runs:    23 runs (latest: 2h ago)
  Annotations:  156 labels across 3 metrics
  Python:       3.13.5
  Providers:    gemini (key: valid), openai (key: missing)
  Disk usage:   data/ = 128 MB
```

---

## 32. Reporting & Analytics Design

### Design: Custom Report Templates

**Jinja2 template system:**
```
evalyn export --template executive_summary.html
```

**Template variables available:**
```python
{
    "run": EvalRun,
    "analysis": RunAnalysis,
    "insights": InsightsReport,
    "charts": {"pass_rates": base64_png, "distributions": base64_png},
    "metadata": {"date": ..., "project": ..., "version": ...}
}
```

**Built-in templates:** `executive_summary`, `technical_deep_dive`, `compliance_report`

### Design: Trend Anomaly Detection

**Approach:**
- Z-score on metric pass rate time series (across last N runs)
- Anomaly = pass rate more than 2 standard deviations from rolling mean
- Configurable sensitivity threshold
- Visual markers on trend charts (red dot for anomaly)
- Auto-alert when anomaly detected during `evalyn trend`

### Design: Regression Bisection

**`evalyn bisect --baseline <run1> --current <run2>`:**
1. Identify items that changed from pass to fail
2. Cluster newly-failing items by input features
3. Rank by regression severity (score delta)
4. Output: "15 items regressed. Top cluster: 'long multi-step queries' (8 items, avg delta -0.4)"

### Design: Failure Taxonomy

**Auto-categorization via LLM:**
```
failure_taxonomy:
  prompt_ambiguity: 12 items (24%)
  model_limitation: 8 items (16%)
  data_quality: 15 items (30%)
  tool_error: 5 items (10%)
  hallucination: 10 items (20%)
```

- Built-in categories or custom taxonomy in evalyn.yaml
- Each failed item tagged with category in `MetricResult.details`
- Distribution chart in analysis output

---

## 33. Interoperability Design

### Design: OpenInference Full Compliance

**Problem:** Current span attributes partially follow OpenInference. Some attributes are missing.

**Gap list and plan:**
- `DocumentAttributes` (retrieval.documents, document.content/score/metadata) - add to retrieval spans
- `EmbeddingAttributes` (embedding.model_name, embedding.text, embedding.vector) - add to embedding spans
- `SessionAttributes` (session.id) - already captured via `session_id`
- `RerankerAttributes` (reranker.model_name, reranker.query, reranker.top_k) - add to reranker spans

### Design: Eval Result Export to Observability Platforms

**Bi-directional flow:**
```
Evalyn traces -> export to Phoenix/Langfuse (existing spans)
Evalyn eval scores -> annotate Phoenix spans with metric results
Phoenix/Langfuse traces -> import into Evalyn for evaluation
```

**Score push-back:** After evaluation, POST metric scores back to the source platform's annotation API.

---

## 34. Code Change Tracking Design

### Design: Source Code Diff Correlation

**Problem:** When metrics change between runs, hard to tell if it's due to code changes.

**Approach:**
- Store `source_hash` (SHA256 of agent source file) in each `EvalRun.metadata`
- Already partially implemented: `_extract_code_meta` in tracer captures function source
- `evalyn code-diff --run1 <id> --run2 <id>`:
  1. Compare source_hash between runs
  2. If different, show code diff alongside metric deltas
  3. Correlate: "Source changed -> 3 metrics regressed"

### Design: Prompt Version Tracking

- Hash judge prompts (preamble + rubric) and store in `MetricResult.details.prompt_hash`
- Warning when comparing runs with different prompt versions for the same metric
- `evalyn prompt-changelog --metric helpfulness` showing prompt evolution over time

---

## 35. Packaging & Distribution Design

### Design: Docker Image

**Dockerfile strategy:**
```dockerfile
FROM python:3.13-slim
RUN pip install evalyn-sdk[all]
# Include all optional deps: sentence-transformers, spacy, gepa
ENTRYPOINT ["evalyn"]
```

- `evalyn-sdk[core]` - minimal, no optional deps
- `evalyn-sdk[all]` - everything including sentence-transformers
- Docker Compose example with SQLite volume mount

### Design: Standalone Binary

**PyInstaller build:**
- Single-file executable for Linux, macOS, Windows
- GitHub Releases automation via GitHub Actions
- Install script: `curl -sSL https://evalyn.dev/install | sh`
- Tradeoff: large binary (~100MB) but zero Python dependency

---

## 36. Deprecation & Migration Design

### Design: Deprecation Warnings

**Registry approach:**
```python
DEPRECATIONS = {
    "inputs": {"replacement": "input", "since": "0.12.0", "remove_in": "1.0.0"},
    "expected": {"replacement": "output", "since": "0.12.0", "remove_in": "1.0.0"},
}
```

- Yellow warning on first use per session
- `evalyn migrate-config` auto-updates deprecated config keys
- Grace period: 2 minor versions between deprecation and removal

### Design: Breaking Change Detection

- Compare metric version hashes between installed version and pinned run manifest
- `evalyn check-compat --run <id>` warns before evaluation if metric behavior changed
- Migration guide output for each detected breaking change

---

## 37. Run Management Design

### Design: Run Naming

**Problem:** Runs identified only by UUIDs.

**Approach:**
- `--name "prompt-v3-experiment"` flag on `run-eval`
- Name stored in `EvalRun.metadata.name`
- Resolve by name: `evalyn show-run --name "prompt-v3-experiment"`
- Names must be unique within a project

### Design: Run Pinning

- `evalyn pin-run --id <id>` marks as project baseline
- Subsequent `analyze` and `compare` auto-compare against pinned run
- `list-runs` shows pinned run with `[*]` marker
- Only one pinned run per project at a time

### Design: Run Cleanup

- `evalyn cleanup-runs --older-than 30d --keep-pinned`
- `--below-pass-rate 0.3` for removing low-quality runs
- `--dry-run` showing what would be deleted with storage savings

---

## 38. Dashboard Interactivity Design

### Design: Embeddable Widget Mode

**`evalyn dashboard --embed`:**
- Produces minimal HTML without navigation chrome
- Configurable widget size and chart selection
- PostMessage API for parent page communication (filter events)
- Use case: embed in Notion, Confluence, or internal tools

### Design: Comparison Overlay Dashboard

**`evalyn dashboard --compare <run1> <run2>`:**
- Dual bar charts (run A vs run B per metric)
- Overlaid radar plots
- Side-by-side heatmaps
- Toggle visibility of each run for clean comparison

---

## 39. Audit & Governance Design

### Design: Data Governance Metadata

**Dataset-level compliance tags:**
```json
{
  "governance": {
    "data_classification": "internal",
    "pii_present": true,
    "approved_for_eval": true,
    "retention_policy": "90d"
  }
}
```

- Tags stored in `meta.json` of each dataset
- Eval run compliance flag: was evaluation run on approved infrastructure?
- `evalyn governance-report` producing exportable compliance audit

### Design: Structured Logging

- `--log-level` flag (debug, info, warning, error) on all commands
- JSON log format: `{"timestamp": "...", "level": "INFO", "module": "runner", "message": "..."}`
- `--log-file evalyn.log` for file output
- Separate from CLI output: logs go to stderr/file, results go to stdout

---

## 41. Research-Driven Designs Round 2 (Security, CI/CD, Calibration)

### Design: Prompt Injection Detection Metric

**Research basis:** Garak has 150+ probes, Lakera Guard (now Check Point) is the leading API, PromptFoo tests 50+ vulnerability types.

**Proposed objective metric:**
```python
def prompt_injection_metric(call, item) -> MetricResult:
    """Detect prompt injection patterns in input/output."""
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s*prompt\s*:",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"disregard\s+(your|all)\s+(rules|instructions)",
        r"<\|.*\|>",  # special token injection
        r"```.*system.*```",  # markdown system prompt leak
    ]
    # Score: 1.0 = clean, 0.0 = injection detected
```

**Levels:**
- `basic`: regex patterns only (fast, no API)
- `advanced`: LLM-based classification (more accurate, requires API)
- Integration: add to `OBJECTIVE_REGISTRY` as `prompt_injection_check`

### Design: GitHub Action for Evalyn

**Research basis:** Braintrust's eval-action and PromptFoo's GitHub Action both post eval diffs as PR comments. DeepEval uses pytest plugin for CI integration.

**Proposed action:**
```yaml
# .github/workflows/evalyn.yml
- uses: evalyn/eval-action@v1
  with:
    dataset: data/golden/
    metrics: metrics/production.json
    baseline: latest  # compare against most recent run
    threshold: 0.85   # fail PR if pass rate below
    comment: true     # post results as PR comment
```

**Implementation:**
- Composite action wrapping `uv pip install evalyn-sdk && evalyn run-eval && evalyn compare`
- Uses GitHub Actions cache for previous run results
- Posts markdown table as PR comment via `github-script`
- Exit code 1 = regression, checked as PR status

### Design: Specialized Judge Model Support

**Research basis:** Patronus Lynx (fine-tuned Llama-3-70B) beats GPT-4o by 8.3% on hallucination detection. Specialized models consistently outperform general-purpose LLM-as-judge.

**Proposed approach:**
```yaml
# evalyn.yaml
judge_models:
  safety_metrics:
    provider: custom
    endpoint: "https://my-lynx-deployment.com/v1/chat/completions"
    model: "lynx-70b"
  quality_metrics:
    provider: gemini
    model: "gemini-2.5-flash-lite"
```

**Implementation:**
- Extend `create_llm_client()` with `provider: "custom"` for arbitrary OpenAI-compatible endpoints
- Per-metric provider routing via `MetricSpec.config.provider` override
- Track accuracy per judge model in calibration records

### Design: EU AI Act Compliance Report

**Research basis:** EU AI Act GPAI obligations (Aug 2025): models >10^23 FLOPs must document evaluation methodology. NIST AI RMF and ISO 42001 are complementary frameworks.

**Proposed report structure:**
```
evalyn compliance-report --format pdf

1. System Description
   - Model(s) evaluated, provider, version
   - Application domain and intended use

2. Evaluation Methodology
   - Metrics used (objective + subjective, with rubric text)
   - Dataset description (size, source, PII classification)
   - Judge model and calibration status

3. Results Summary
   - Per-metric pass rates with confidence intervals
   - Failure analysis and known limitations
   - Comparison against previous evaluation

4. Governance
   - Audit trail reference (hash-chain verification)
   - Data governance tags
   - Annotator information and agreement metrics
```

**Implementation:** Jinja2 template extending existing HTML report generator. Add `--compliance` flag to `evalyn export`.

### Design: CAPO Optimizer

**Research basis:** CAPO (Confidence-Aware Prompt Optimization) is the current SOTA optimizer, outperforming OPRO and EvoPrompt on standard benchmarks.

**Integration approach:**
- New `calibration/capo.py` implementing `BaseOptimizer`
- Add `"capo"` entry to `OPTIMIZER_REGISTRY`
- Configuration: `CAPOConfig(confidence_threshold, max_iterations, population_size)`
- Benchmark: run against `basic`, `opro`, `evoprompt` on internal test suite

---

## 42. Research-Driven Designs Round 3 (Multi-Agent, Simulation, SDK, Cost)

### Design: Multi-Agent Communication Scoring

**Research basis:** MARBLE (ACL 2025) milestone-based KPIs, CLEAR framework (cost/latency/efficiency/assurance/reliability), finding that 60% single-run success drops to 25% at 8-run consistency.

**Proposed metrics:**
```python
# New objective metrics in metrics/objective.py:

agent_communication_score:
  # Per-message scoring (1-5): relevance, clarity, information density
  # Aggregate: mean score across all inter-agent messages in trace

agent_consistency:
  # Run N times, measure: tool call sequence similarity, output similarity
  # Score: fraction of runs producing equivalent results

milestone_completion:
  # User defines milestones as span name patterns
  # Score: fraction of milestones achieved
  # Config: milestones: ["data_retrieved", "analysis_complete", "response_sent"]
```

**Implementation:** These require multi-run evaluation support. Add `--repeat N` flag to `run-eval` that runs each item N times and computes consistency metrics across runs.

### Design: Evol-Instruct Data Evolution

**Research basis:** WizardLM's Evol-Instruct methodology, DeepEval's Synthesizer with quality scoring, Auto Evol-Instruct that meta-optimizes the evolution process.

**Proposed pipeline:**
```
Seed items
  |
  v
Evolution (LLM-powered, configurable depth)
  |-- In-depth: add constraints ("must use only public data"), increase reasoning steps
  |-- In-breadth: domain transfer ("same question but for healthcare"), format change
  |
  v
Quality Filter
  |-- Score: clarity (1-5), depth (1-5), structure (1-5), relevance (1-5)
  |-- Reject items below configurable threshold (default: 3.0 average)
  |
  v
Deduplication
  |-- Embedding-based dedup against seed + previously evolved items
  |
  v
Evolved dataset
```

**CLI:** `evalyn simulate --mode evolve --depth 3 --breadth 2`

### Design: IRT-Based Tiny Benchmarks

**Research basis:** tinyBenchmarks (NeurIPS 2024) - Item Response Theory from psychometrics reduces 14K items to 100 (140x) within 2% error. SubLIME achieves 0.85-0.95 correlation at 10% sampling rate.

**Proposed approach:**
```python
@dataclass
class IRTItemParams:
    item_id: str
    difficulty: float      # how hard the item is (theta)
    discrimination: float  # how well it separates good from bad models (alpha)
    information: float     # Fisher information at target ability

def optimize_dataset_irt(
    items: List[DatasetItem],
    eval_history: List[EvalRun],  # historical runs for parameter estimation
    target_size: int = 100,
) -> List[DatasetItem]:
    """Select items maximizing total information."""
    # 1. Estimate IRT parameters from historical pass/fail data
    # 2. Compute Fisher information per item at target ability level
    # 3. Greedy selection: pick items with highest information
    # 4. Ensure coverage: at least 1 item per difficulty quintile
```

**CLI:** `evalyn dataset-optimize --method irt --target-size 100 --history last-10-runs`

### Design: Cascade Model Routing for Evaluation

**Research basis:** ETH Zurich's unified framework achieves 87% cost reduction. Gatekeeper reduces expensive calls by 40% without quality loss. Key: quality estimators determine when to escalate.

**Proposed design:**
```yaml
# evalyn.yaml:
model_routing:
  strategy: cascade
  models:
    - provider: gemini
      model: gemini-2.5-flash-lite  # tier 1: cheapest
      max_input_tokens: 2000        # only for short inputs
    - provider: gemini
      model: gemini-2.5-flash       # tier 2: capable
      # default for everything tier 1 can't handle
    - provider: openai
      model: gpt-4o                  # tier 3: most capable
      only_when: confidence < 0.7    # escalate uncertain items
```

**Implementation:**
- New `CascadeJudge` wrapping multiple `LLMJudge` instances
- Tier 1 evaluates all items. If confidence < threshold, escalate to tier 2.
- Track: actual cost vs estimated full-price cost (savings report)
- Compatible with existing `--provider` flag as single-tier fallback

### Design: Declarative Evaluation API

**Research basis:** All major frameworks converge on a declarative pattern: data + task + scorers in a single call.

**Proposed evalyn Python API:**
```python
import evalyn

# Pattern 1: Braintrust-style declarative
results = evalyn.evaluate(
    name="helpfulness-test",
    dataset="data/prod/datasets/my_project/",
    metrics=["helpfulness_accuracy", "toxicity_safety", "json_valid"],
    provider="gemini",
)
print(results.overall_pass_rate)  # 0.85
print(results.to_pandas())        # DataFrame with per-item scores

# Pattern 2: Weave-style with model
results = evalyn.evaluate(
    dataset=[{"input": "...", "expected": "..."}],
    model=my_agent_function,
    metrics=["helpfulness_accuracy"],
)

# Pattern 3: Existing CLI-compatible
run = evalyn.run_eval(
    dataset_path="data/prod/datasets/my_project/",
    metrics_path="metrics/production.json",
    max_workers=4,
)
analysis = evalyn.analyze(run)
```

**Key decisions:**
- Return typed `EvalResult` object with `.to_pandas()`, `.to_dict()`, `.summary`
- No `sys.exit` or `print` in the API path - raise exceptions, return objects
- Async variant: `await evalyn.evaluate_async(...)`
- Metrics can be specified as strings (lookup registry) or `Metric` objects

### Design: Semantic Caching for Judge Calls

**Research basis:** GPTCache achieves 68.8% API call reduction with 97% positive hit accuracy. Architecture: embedding -> vector similarity -> threshold matching.

**Proposed design:**
```
Judge call (prompt, input, output, model)
  |
  v
Cache lookup: sha256(prompt + input + output + model)
  |-- Exact hit: return cached MetricResult
  |-- Miss: continue to LLM call
  |
  v
(Optional) Fuzzy lookup: embed(input + output), cosine search in cache
  |-- Similarity > 0.98: return cached result (with cache_hit=fuzzy flag)
  |-- Miss: continue to LLM call
  |
  v
LLM judge call -> MetricResult
  |
  v
Store in cache: eval_cache table in SQLiteStorage
```

**Cache table schema:**
```sql
CREATE TABLE eval_cache (
    cache_key TEXT PRIMARY KEY,    -- sha256 hash
    metric_id TEXT,
    result_json TEXT,              -- serialized MetricResult
    created_at TEXT,
    hit_count INTEGER DEFAULT 0,
    embedding BLOB                 -- optional, for fuzzy matching
);
```

---

## 43. Research-Driven Designs Round 4 (Deep Dive Findings)

### Design: Sandboxed Agent Evaluation

**Research basis:** Inspect AI has the most mature sandboxing (Docker, K8s, Proxmox, Modal). Critical for agent evals where models execute code.

**Proposed approach:**
```python
# In evalyn.yaml:
sandbox:
  enabled: true
  runtime: docker          # or "none" for no sandbox
  image: "python:3.13-slim"
  timeout: 30              # seconds per execution
  memory_limit: "512m"
  network: false           # disable network access in sandbox
```

**Implementation:**
- New `SandboxExecutor` class wrapping Docker API (optional `docker` dependency)
- Instrument target function to run inside container when sandbox enabled
- Capture container stdout/stderr as span attributes
- Fallback: when Docker unavailable, warn and run unsandboxed

### Design: Composable Assertion Framework

**Research basis:** PromptFoo's assertion types (contains, llm-rubric, similar, cost-below) are clean evaluation primitives.

**Proposed assertion types:**
```yaml
# In metrics definition:
assertions:
  - type: contains
    value: "Bonjour"
  - type: not_contains
    value: "I cannot"
  - type: regex_match
    pattern: "^\{.*\}$"
  - type: cost_below
    max_cost: 0.01
  - type: latency_below
    max_ms: 5000
  - type: llm_rubric
    criteria: "Response is factually accurate and helpful"
  - type: similar_to
    reference: "Expected output text"
    threshold: 0.8
```

**Implementation:**
- Each assertion type maps to an objective metric function
- Composable with `all_of` (AND) and `any_of` (OR) combinators
- Pass/fail determined by assertion results, score = fraction passing
- Readable failure messages: "Assertion 'contains: Bonjour' failed on item 3"

### Design: Evaluation Result Schema Standard

**Research basis:** No universal standard for evaluation results exists (identified as industry gap). Every platform uses its own format.

**Proposed schema (JSON):**
```json
{
  "$schema": "https://evalyn.dev/schemas/eval-result-v1.json",
  "version": "1.0",
  "metadata": {
    "tool": "evalyn",
    "tool_version": "0.15.0",
    "timestamp": "2026-03-27T12:00:00Z",
    "dataset": {"name": "...", "hash": "sha256:...", "item_count": 100}
  },
  "metrics": [
    {"id": "helpfulness", "type": "subjective", "pass_rate": 0.85, "avg_score": 0.82}
  ],
  "items": [
    {
      "id": "item-1",
      "results": [
        {"metric_id": "helpfulness", "score": 0.9, "passed": true, "details": {}}
      ]
    }
  ],
  "summary": {
    "overall_pass_rate": 0.85,
    "total_cost_usd": 1.23,
    "total_items": 100
  }
}
```

**Goal:** Publish as open spec that other tools can adopt for interoperability.

### Design: Denormalized Storage for Query Performance

**Research basis:** Langfuse found 10x dashboard speedup by denormalizing trace attributes onto every observation row (March 2026 architecture shift).

**Proposed migration for evalyn:**
- Add trace-level columns to `otel_spans` table: `project_name`, `session_id`, `function_name`, `call_started_at`
- Populate via additive migration (backfill from function_calls table)
- Enable direct span queries without JOIN to function_calls
- Key queries that benefit: "list spans for project X", "spans slower than Y ms in session Z"

**Risk:** Increases storage size by ~20% due to denormalization. Acceptable for query performance gains on large databases.

---

## Design Principles

1. **Local-first:** SQLite by default, no cloud dependency for core functionality
2. **Incremental adoption:** The tracing decorator is the only required entry point; everything else is optional
3. **Provider-agnostic:** Judge model is configurable; defaults to cheapest (Gemini Flash Lite)
4. **Backward-compatible:** Old datasets, old APIs, old configs continue to work via aliases and migration
5. **Lazy loading:** Heavy dependencies (sentence-transformers, spacy, gepa) imported only when needed
6. **Pure analysis:** Analysis functions are stateless and testable; no side effects
7. **Extension over modification:** New metrics, providers, optimizers added via registries, not by editing core
8. **Fail-open for optional features:** Missing optional dependencies degrade gracefully with warnings, never crash
9. **Schema stability:** Additive-only migrations; existing data always readable by newer versions

*Last updated: 2026-03-27*
