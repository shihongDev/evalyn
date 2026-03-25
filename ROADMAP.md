# Evalyn Roadmap

This document tracks planned features and completed work. Future roadmap items are listed first, followed by completed features.

---

## Roadmap (Planned Features)

### Tracing & Instrumentation

- [ ] **Multi-modal Tracing** - Capture images, audio, video in traces
  - [ ] Image input/output capture with thumbnails
  - [ ] Audio transcription logging
  - [ ] Video frame sampling
  - [ ] Base64/URL reference storage options
- [ ] **Streaming Support** - Capture streaming LLM responses
  - [x] Streaming response capture (OpenAI, Anthropic, Gemini via StreamingSpanWrapper)
  - [ ] Token-by-token capture with timing
  - [ ] First-token latency (TTFT) metric
  - [ ] Streaming interruption detection
- [ ] **More LLM Provider Instrumentors**
  - [ ] Cohere
  - [ ] Mistral
  - [ ] AWS Bedrock
  - [ ] Azure OpenAI
  - [ ] Groq
  - [ ] Together AI
  - [ ] Replicate
- [ ] **Framework Instrumentors**
  - [x] CrewAI
  - [x] AutoGen
  - [x] DSPy
  - [x] Haystack
  - [x] LlamaIndex
  - [x] Semantic Kernel
- [ ] **Memory/RAG Tracing** - Capture retrieval context and memory operations
  - [ ] Capture retrieved documents with relevance scores per query
  - [ ] Track vector store lookup latency and result count
  - [ ] Link retrieval spans to downstream LLM calls that consume them
  - [ ] Memory read/write operation logging for stateful agents
- [ ] **Async/Parallel Call Tracking** - Better support for concurrent LLM calls
  - [ ] Detect concurrent spans and render as parallel branches in show-trace
  - [ ] Measure total wall-clock vs sum of individual span durations
  - [ ] asyncio-native context propagation (ContextVar across await boundaries)
  - [ ] Thread-pool executor span grouping
- [ ] **Trace Export to OTel Backends** - Export traces to Jaeger, Zipkin, or any OpenTelemetry collector
  - [ ] OTLP gRPC exporter alongside existing SQLiteSpanExporter
  - [ ] OTLP HTTP/JSON exporter for firewall-friendly environments
  - [ ] Configurable export filters (only export errors, only export slow spans)
  - [ ] Dual-write mode: SQLite for evalyn + OTLP for observability platform
- [ ] **Trace Replay** - Re-run a captured trace against a different model to compare outputs
  - [ ] Extract input messages from each LLM span for replay
  - [ ] Swap model name and re-execute captured prompts
  - [ ] Generate side-by-side diff of original vs replayed outputs
  - [ ] Cost comparison report between original and replayed model
- [ ] **Cost Budget Alerts** - Warn or stop when cumulative LLM cost exceeds a configurable threshold
  - [ ] Per-session budget limit in evalyn.yaml
  - [ ] Per-run budget limit as --max-cost flag
  - [ ] Warning at 80% threshold, hard stop at 100%
  - [ ] Budget tracking across multiple eval runs in a session
- [ ] **Trace Diff** - Side-by-side comparison of two traces showing divergent spans
  - [ ] Align spans by name/type and highlight added/removed/changed spans
  - [ ] Show output text diff for matching spans
  - [ ] Cost and latency delta per span
  - [ ] ASCII and HTML diff output formats
- [ ] **Trace Search Query Language** - Filter traces by span attributes, duration, cost, or error status
  - [ ] SQL-like syntax: "spans where type=llm_call and duration_ms > 5000"
  - [ ] Attribute filtering: model name, token count, error status
  - [ ] Aggregate queries: "traces with total_cost > $0.10"
  - [ ] Integration with list-calls command via --query flag
- [ ] **PII Redaction** - Scrub sensitive data from inputs/outputs before storage
  - [ ] Regex-based patterns for emails, phone numbers, SSNs, credit cards
  - [ ] Named entity recognition for names and addresses
  - [ ] Configurable redaction strategy: mask, hash, or remove
  - [ ] Pre-storage hook in SQLiteSpanExporter and SQLiteStorage
- [ ] **Trace Sampling Rate** - Capture only N% of traces in production to reduce storage overhead
  - [ ] Configurable sample rate in evalyn.yaml (0.0 to 1.0)
  - [ ] Priority-based sampling: always capture errors and slow traces
  - [ ] Per-project sampling rate override
- [ ] **Distributed Trace Propagation** - Pass trace context across service boundaries via HTTP headers
  - [ ] W3C Trace Context (traceparent/tracestate) header injection
  - [ ] HTTP client instrumentation to propagate headers on outbound calls
  - [ ] Incoming header extraction to attach child spans to external parent
- [ ] **Trace Size Limits** - Cap span payload size with configurable truncation for large inputs/outputs
  - [ ] Max input/output size in bytes with tail truncation
  - [ ] Configurable per span type (larger limit for llm_call, smaller for tool_call)
  - [ ] Truncation marker in span metadata when content is clipped
- [ ] **Custom Span Types** - Register user-defined span types beyond the built-in set (llm_call, tool_call, etc.)
  - [ ] Registration API: register_span_type(name, icon, color)
  - [ ] Custom span type validation in span creation
  - [ ] Custom types rendered in show-trace with user-defined icons
- [ ] **Span Tagging at Trace Time** - Add custom key-value tags to spans during execution for later filtering
  - [ ] API: tag_current_span(key, value) callable inside traced functions
  - [ ] Tags stored in span metadata and queryable via list-calls
  - [ ] Standard tags: environment, user_id, experiment_id, variant
- [ ] **Native Embedding and Reranker Span Types** - First-class span types for embedding and reranking operations
  - [ ] "embedding" span type capturing model name, input text, vector dimensions
  - [ ] "reranker" span type capturing query, documents, and re-ranked scores
  - [ ] "guardrail" span type capturing check name, pass/fail, and blocked content
  - [ ] Update SPAN_KIND_TO_TYPE mapping in conventions.py (currently mapped to "custom")
- [ ] **Span Attribute Extraction Plugins** - Pluggable attribute extractors for SpanConverter
  - [ ] Plugin interface for extracting custom attributes from OTEL spans
  - [ ] Provider-specific extractors (e.g. extract function_call from OpenAI tool use spans)
  - [ ] Configurable truncation limits per attribute (currently hardcoded 1000 chars)

### Instrumentation & Decorator Enhancements

- [ ] **Selective Instrumentation** - Only instrument specific methods or classes, not entire SDK
  - [ ] Allowlist/blocklist of method names to instrument per provider
  - [ ] Config in evalyn.yaml: instrument.openai.methods: ["chat.completions.create"]
  - [ ] Reduce overhead by skipping low-value calls (e.g. embeddings, moderation)
- [ ] **Instrumentation Health Check** - Verify instrumentation is capturing spans correctly
  - [ ] evalyn check-instrumentation that runs a test call and verifies span capture
  - [ ] Report which providers are instrumented, which failed, and why
  - [ ] Warning when instrumented SDK is imported before evalyn_sdk
- [ ] **Instrumentation Overhead Measurement** - Measure performance impact of tracing
  - [ ] Benchmark: instrumented vs uninstrumented call latency
  - [ ] Report added overhead in ms and % per provider
  - [ ] Auto-disable instrumentation if overhead exceeds threshold
- [ ] **Experiment Tracking** - Group traces by experiment ID for A/B comparisons
  - [ ] @eval(experiment="prompt-v2") decorator parameter
  - [ ] Filter traces by experiment in list-calls and build-dataset
  - [ ] Cross-experiment metric comparison in analyze command
- [ ] **Conditional Tracing** - Only trace when runtime conditions are met
  - [ ] Sample-based: trace 10% of calls via @eval(sample_rate=0.1)
  - [ ] Predicate-based: @eval(trace_if=lambda args: args["user_id"] in sample_set)
  - [ ] Environment-based: only trace in production, skip in unit tests

### Onboarding & Templates

- [ ] **Quickstart Templates** - Framework-specific guided templates beyond generic quickstart
  - [ ] evalyn quickstart --template rag for RAG pipeline setup
  - [ ] evalyn quickstart --template chatbot for conversational agent setup
  - [ ] evalyn quickstart --template multi-agent for multi-agent orchestration
  - [ ] Each template pre-selects relevant metric bundles
- [ ] **Interactive Tutorial Mode** - Step-by-step in-terminal tutorial for learning evalyn
  - [ ] evalyn tutorial that walks through trace/build/eval/analyze cycle
  - [ ] Bundled sample traces so tutorial works without API keys
  - [ ] Progressive disclosure: each step explains what happened and why
- [ ] **Example Agent Gallery** - Bundled working example agents for each supported framework
  - [ ] example_agents/ directory with one example per framework
  - [ ] Each example includes: agent code, pre-built dataset, expected results
  - [ ] evalyn example --framework openai to scaffold from template

### Config & Project Management

- [ ] **Config Inheritance** - Base config with per-project overrides
  - [ ] Global ~/.evalyn/config.yaml for shared settings (API keys, provider defaults)
  - [ ] Project-level evalyn.yaml inherits and overrides global config
  - [ ] Per-dataset config override via meta.json
- [ ] **Project Scaffolding** - evalyn new-project to create standard project structure
  - [ ] Create data/ directory, evalyn.yaml, and .gitignore entries
  - [ ] Optional: create example agent file for chosen framework
  - [ ] Optional: create GitHub Actions workflow for CI evaluation
- [ ] **Multi-Project Dashboard** - View and compare metrics across multiple projects
  - [ ] evalyn projects showing all projects with latest run status
  - [ ] Cross-project regression detection
  - [ ] Unified cost tracking across projects

### Confidence & Judge Robustness

- [ ] **Confidence Method Comparison** - Run all confidence methods on same data and compare calibration
  - [ ] Side-by-side comparison of logprobs, deepconf, consistency, verbalized methods
  - [ ] Calibration curve: confidence score vs actual correctness
  - [ ] Recommend best method per metric/provider combination
- [ ] **Hybrid Confidence** - Combine multiple confidence methods into a single robust score
  - [ ] Weighted ensemble of available methods
  - [ ] Fall back gracefully when a method is unavailable (e.g. no logprobs)
  - [ ] Bayesian combination with learned weights
- [ ] **Structured Output Enforcement** - Force JSON mode on judge LLM calls for reliable parsing
  - [ ] Use provider-native JSON mode (Gemini response_mime_type, OpenAI response_format)
  - [ ] Schema enforcement via provider-specific structured output features
  - [ ] Fallback to regex extraction when JSON mode unavailable
- [ ] **Judge Output Retry** - Automatically retry judge calls when output fails to parse
  - [ ] Configurable max retries (default 2)
  - [ ] Append "respond with valid JSON" on retry attempts
  - [ ] Track parse failure rate per metric for diagnostics
- [ ] **Judge Latency Optimization** - Reduce judge call overhead for large-scale evaluation
  - [ ] Prompt caching: reuse system prompt prefix across items
  - [ ] Batch multiple items into single judge call where possible
  - [ ] Model-specific prompt length optimization

### Evaluation Units & Views

- [ ] **Custom Unit Builder Plugins** - User-defined evaluation boundaries via pluggable builders
  - [ ] Register custom EvalUnitBuilder subclasses via entry points
  - [ ] Builder configuration in evalyn.yaml per metric
  - [ ] Example builders: per-paragraph, per-code-block, per-citation
- [ ] **Unit Type Auto-Detection** - Infer best EvalUnit type from trace structure
  - [ ] Detect multi-turn patterns from sequential LLM spans
  - [ ] Detect tool-use patterns from tool_call/tool_result span pairs
  - [ ] Default to outcome when trace structure is flat
- [ ] **Unit-Level Reporting** - Per-unit-type metric breakdowns in analysis
  - [ ] Separate pass rates for outcome vs single_turn vs tool_use units
  - [ ] Unit type distribution chart in analysis output
  - [ ] Filter analysis by unit type: --unit-type single_turn

### Batch Evaluation Enhancements

- [ ] **Batch Job Persistence** - Save batch job state to disk for recovery after crash or restart
  - [ ] Write BatchJob to .evalyn/batch_jobs/ as JSON on submit
  - [ ] evalyn batch-status to list pending/completed batch jobs
  - [ ] evalyn batch-resume to collect results from a previously submitted batch
- [ ] **Mixed-Mode Evaluation** - Use batch API for large runs, real-time for small runs
  - [ ] Auto-select mode based on item count threshold (e.g. batch if > 50 items)
  - [ ] --mode auto/batch/realtime flag on run-eval
  - [ ] Cost/speed comparison in dry-run output
- [ ] **Batch Progress Polling** - Live progress updates while batch job is processing
  - [ ] Poll provider API for completion percentage
  - [ ] Display progress bar with ETA during batch wait
  - [ ] Configurable poll interval (default 30s)
- [ ] **Multi-Provider Batch Splitting** - Split a single evaluation batch across multiple providers
  - [ ] Route N% of items to gemini, M% to openai for cost/latency comparison
  - [ ] Provider-aware retry: re-route failed items to alternate provider
  - [ ] Unified result merging regardless of which provider evaluated each item
- [ ] **Streaming Partial Results** - Start analyzing results before the full batch completes
  - [ ] Process completed items as they arrive from batch polling
  - [ ] Live-updating analysis dashboard during batch wait
  - [ ] Early termination: stop batch if enough results show clear pass/fail

### Session Management

- [ ] **Session-Level Analysis** - Aggregate metrics across all calls within an eval_session
  - [ ] Group traces by session_id in analysis output
  - [ ] Per-session pass rate, cost, and latency summaries
  - [ ] Cross-session comparison for the same user journey
- [ ] **Session Replay** - Re-execute a full session against a different model or prompt version
  - [ ] Extract all inputs from session traces in order
  - [ ] Replay with swapped model/provider
  - [ ] Session-level diff: compare original vs replayed outputs turn by turn

### Reproducibility

- [ ] **Deterministic Evaluation Mode** - Ensure runs produce identical results given identical inputs
  - [ ] Fixed random seed for all sampling operations
  - [ ] Temperature 0 enforcement for judge LLM calls
  - [ ] --seed flag on run-eval for reproducible runs
- [ ] **Run Manifest** - Record every parameter that could affect evaluation results
  - [ ] Store: evalyn version, Python version, provider versions, metric hashes, config hash
  - [ ] Manifest file alongside eval run results
  - [ ] evalyn verify-manifest to check reproducibility of a past run
- [ ] **Custom Cost Models** - User-defined pricing for custom or self-hosted models
  - [ ] Per-model cost-per-token config in evalyn.yaml
  - [ ] Override default pricing for Ollama and other local models
  - [ ] Cost model versioning for tracking price changes over time

### Cost Intelligence

- [ ] **Auto-Update Pricing Tables** - Fetch latest model pricing from provider APIs
  - [ ] Scrape/fetch pricing from OpenAI, Anthropic, Google pricing pages
  - [ ] evalyn update-pricing command to refresh COST_PER_1M_TOKENS in _shared.py
  - [ ] Warn when using a model not in the pricing table
- [ ] **Prompt Cache Savings Report** - Show how much prompt caching saved per run
  - [ ] Aggregate cache_creation_tokens and cache_read_tokens from spans
  - [ ] Calculate: actual cost vs hypothetical cost without caching
  - [ ] Recommend caching strategy based on prompt repetition patterns
- [ ] **Context Window Utilization Alerts** - Warn when spans approach context limits
  - [ ] Alert when context_utilization_pct exceeds configurable threshold (default 80%)
  - [ ] Per-run summary: max utilization, mean utilization, models hitting limits
  - [ ] Suggest model upgrade when context is consistently near capacity

### Confidence Enhancements

- [ ] **Adaptive Consistency Sampling** - Stop early when judge agreement is already clear
  - [ ] Sequential sampling: stop after 3 samples if all agree (skip remaining 2)
  - [ ] Configurable early-stop threshold (e.g. 100% agreement after 3 of 5 samples)
  - [ ] Cost savings report: samples skipped vs full sampling
- [ ] **Confidence-Based Re-Evaluation** - Re-evaluate uncertain items with a stronger model
  - [ ] Identify items where confidence score < threshold after initial eval
  - [ ] Automatically re-run those items with a more capable model (e.g. flash -> pro)
  - [ ] Merge re-evaluated scores back into the run results
- [ ] **Confidence Threshold Tuning** - Find optimal confidence cutoff per metric
  - [ ] Binary search for threshold that maximizes alignment with human annotations
  - [ ] Per-metric optimal threshold stored in calibration record
  - [ ] evalyn tune-confidence command

### Config Enhancements

- [ ] **Config Profiles** - Named environment profiles (dev/staging/prod) in evalyn.yaml
  - [ ] profiles: section with per-profile overrides
  - [ ] --profile flag on all commands to select active profile
  - [ ] Profiles inherit from base config, override specific keys
- [ ] **Environment Variable Validation** - Check all required env vars at command startup
  - [ ] Required vars per command (e.g. run-eval needs GEMINI_API_KEY)
  - [ ] Validate key format and basic connectivity before starting long operations
  - [ ] Clear error messages: "GEMINI_API_KEY is set but invalid (HTTP 401)"

### Evaluation Enhancements

- [ ] **Span-Level Evaluation** - Evaluate individual spans within a trace
  - [ ] Per-LLM-call quality metrics
  - [ ] Tool call success/failure analysis
  - [ ] Node-level evaluation for graph agents
  - [ ] Span-specific rubrics
- [ ] **Multi-Turn Evaluation** - Specialized evaluation for conversations
  - [ ] Turn-by-turn quality assessment
  - [ ] Conversation flow metrics
  - [ ] Context carryover evaluation
  - [ ] Memory consistency across turns
  - [ ] Topic drift detection
  - [ ] Response latency patterns
- [ ] **Pairwise Comparison** - A vs B evaluation mode
  - [ ] Side-by-side LLM judge comparison
  - [ ] Elo rating system for models
  - [ ] Win/loss/tie statistics
- [ ] **Reference-Free Evaluation** - Metrics that don't need ground truth
  - [x] Self-consistency checking (via --confidence consistency)
  - [x] Uncertainty quantification (via confidence module)
- [ ] **Evaluation Budget Control** - Stop early if token or cost budget is exceeded mid-run
  - [ ] --max-tokens and --max-cost flags on run-eval
  - [ ] Real-time budget tracking in ProgressCallback
  - [ ] Graceful stop: finish current item, checkpoint, report partial results
  - [ ] Budget summary in EvalRun metadata
- [ ] **Differential Evaluation** - Only re-evaluate items that changed between dataset versions
  - [ ] Hash-based change detection using datasets.hash_inputs
  - [ ] Carry forward unchanged MetricResults from previous run
  - [ ] --diff-from flag to specify baseline run ID
  - [ ] Report showing only changed items and their score deltas
- [ ] **Evaluation Caching** - Skip re-computing unchanged metric/item pairs across runs
  - [ ] Content-addressable cache keyed by (item_hash, metric_id, prompt_hash)
  - [ ] Cache stored in SQLite alongside eval runs
  - [ ] --no-cache flag to force re-evaluation
  - [ ] Cache hit/miss statistics in run summary
- [ ] **Evaluation Dry-Run** - Estimate token cost and wall-clock time before executing
  - [ ] Count items x metrics, estimate tokens per metric type
  - [ ] Cost estimate by provider (Gemini, OpenAI pricing)
  - [ ] --dry-run flag that prints estimate and exits
  - [ ] Wall-clock estimate based on historical run data
- [ ] **Cross-Validation Evaluation** - K-fold scoring for statistically robust metric estimates
  - [ ] --cv-folds N flag to split dataset into N folds
  - [ ] Stratified splitting by metadata or score
  - [ ] Per-fold and aggregate metric statistics with std deviation
  - [ ] Identify items with high variance across folds
- [ ] **Evaluation Replay** - Re-run a past evaluation with different judge prompts or providers
  - [ ] --replay-run flag to reuse items/metrics from a previous run
  - [ ] Override provider, model, or calibrated prompts
  - [ ] Automatic comparison report between original and replayed run
- [ ] **Conditional Metrics** - Run expensive subjective metrics only if cheap objective metrics pass first
  - [ ] Metric dependency declaration: "run helpfulness only if json_valid passes"
  - [ ] Gate conditions: pass/fail, score threshold, or custom predicate
  - [ ] Skip tracking: report which items had metrics skipped and why
- [ ] **Evaluation Profiles** - Named configs (fast/thorough/cost-optimized) bundling workers, providers, and metric sets
  - [ ] Profile definitions in evalyn.yaml (fast: 8 workers, objective only; thorough: all metrics, 2 workers)
  - [ ] --profile flag on run-eval
  - [ ] Built-in profiles: smoke-test, standard, comprehensive
- [ ] **Evaluation Tagging** - Tag runs with custom labels for filtering and organization
  - [ ] --tag flag on run-eval (multiple tags allowed)
  - [ ] Tags stored in EvalRun metadata and queryable via list-runs
  - [ ] Filter list-runs by tag: --filter-tag experiment-v2
- [ ] **Async Evaluation Strategy** - Native asyncio execution strategy alongside sequential and parallel
  - [ ] AsyncStrategy using asyncio.gather for concurrent metric calls
  - [ ] Semaphore-based concurrency control (replaces ThreadPoolExecutor)
  - [ ] Compatible with async LLM client libraries (httpx, aiohttp)
  - [ ] --strategy flag: sequential, parallel, async
- [ ] **Distributed Evaluation** - Fan out metric evaluation across multiple machines via task queue
  - [ ] Redis/RabbitMQ task queue for distributing metric evaluations
  - [ ] Worker process that pulls and evaluates metric tasks
  - [ ] Centralized result collection and checkpoint merging
  - [ ] --distributed flag with queue URL configuration

### Calibration & Optimization

- [x] **More Optimizers**
  - [x] DSPy MIPROv2 - Multi-stage instruction optimization
  - [x] TextGrad - Gradient-based prompt optimization
  - [x] EvoPrompt - Evolutionary prompt optimization
  - [x] PromptBreeder - Self-referential prompt evolution
- [ ] **Rubric Optimization** - Auto-generate and refine evaluation rubrics
  - [ ] LLM-generated rubric from example pass/fail items
  - [ ] Iterative rubric refinement based on disagreement analysis
  - [ ] Rubric clarity scoring (can a different LLM interpret it consistently?)
  - [ ] A/B test rubric variants for inter-judge agreement
- [ ] **Few-Shot Example Selection** - Optimize which examples to include in prompts
  - [ ] Select maximally informative examples from annotation pool
  - [ ] Diversity-based selection: cover different failure modes
  - [ ] Leave-one-out evaluation to measure example contribution
  - [ ] Dynamic example count optimization (find optimal k)
- [ ] **Judge Ensemble** - Combine multiple judges for robust evaluation
  - [ ] Majority vote across N judges (same or different models)
  - [ ] Weighted ensemble based on per-judge calibration accuracy
  - [ ] Disagreement flagging: items where judges disagree go to human review
  - [ ] Cost-aware ensemble: use cheap judge first, expensive only on uncertain items
- [ ] **Active Learning** - Smart sample selection for annotation
  - [ ] Uncertainty sampling: prioritize items where judge confidence is lowest
  - [ ] Disagreement sampling: prioritize items where judge and heuristics disagree
  - [ ] Diversity sampling: ensure coverage of input space
  - [ ] Batch-mode active learning with configurable batch size
- [ ] **Transfer Calibration** - Apply calibration learned on one metric to similar metrics
  - [ ] Metric similarity detection based on rubric text embedding
  - [ ] Shared preamble transfer with metric-specific rubric
  - [ ] Transfer effectiveness validation on held-out samples
- [ ] **Calibration Staleness Detection** - Warn when calibration age or dataset drift exceeds threshold
  - [ ] Track calibration date and dataset hash at calibration time
  - [ ] Alert when dataset changes exceed drift threshold (new items, distribution shift)
  - [ ] Re-calibration recommendation with estimated alignment degradation
- [ ] **Cross-Provider Calibration** - Calibrate for consistency when switching judge providers
  - [ ] Run same calibration set across providers (Gemini, OpenAI, Ollama)
  - [ ] Provider-specific preamble adjustments
  - [ ] Cross-provider agreement metrics
- [ ] **Calibration A/B Testing** - Compare calibrated vs uncalibrated prompts on the same dataset
  - [ ] Side-by-side evaluation run with original and calibrated prompts
  - [ ] Per-item comparison showing score changes
  - [ ] Statistical significance test for improvement
- [ ] **Calibration Rollback** - Revert to a previous calibration if the new one degrades alignment
  - [ ] Calibration history stored in CalibrationRecord
  - [ ] --rollback flag on calibrate command
  - [ ] Automatic rollback suggestion when validation metrics drop
- [ ] **Multi-Objective Calibration** - Optimize jointly for accuracy and cost (fewer tokens per judgment)
  - [ ] Pareto front of accuracy vs token count
  - [ ] Prompt compression as optimization objective
  - [ ] Configurable accuracy/cost trade-off weight
- [ ] **Calibration Cost Tracking** - Report total LLM cost of the calibration process itself
  - [ ] Per-optimizer token usage tracking (extend TokenAccumulator)
  - [ ] Cost breakdown by calibration phase (alignment, optimization, validation)
  - [ ] Historical cost trends across calibration runs
- [ ] **Calibration Curriculum** - Start optimization on easy examples, progressively add harder ones
  - [ ] Sort calibration examples by judge confidence (easy = high confidence)
  - [ ] Progressive expansion: start with top-50% easiest, add harder items
  - [ ] Early stopping if optimizer plateaus before reaching hard examples

### Multi-Modal Evaluation

- [ ] **Image Evaluation Metrics**
  - [ ] Image-text alignment (CLIP score)
  - [ ] Visual quality assessment
  - [ ] OCR accuracy for generated images
  - [ ] Style consistency
- [ ] **Audio Evaluation Metrics**
  - [ ] Speech clarity
  - [ ] Transcription accuracy (WER)
  - [ ] Prosody and tone
- [ ] **Video Evaluation Metrics**
  - [ ] Frame consistency
  - [ ] Temporal coherence
  - [ ] Action recognition accuracy

### Agent-Specific Evaluation

- [ ] **Tool Use Evaluation**
  - [ ] Tool selection appropriateness
  - [ ] Parameter correctness
  - [ ] Error recovery patterns
  - [ ] Tool chain efficiency
- [ ] **Planning Evaluation**
  - [ ] Plan completeness
  - [ ] Step ordering correctness
  - [ ] Resource efficiency
  - [ ] Replanning quality
- [ ] **Reasoning Evaluation**
  - [ ] Chain-of-thought faithfulness
  - [ ] Logical consistency
  - [ ] Evidence usage
  - [ ] Conclusion validity

### Graph & Multi-Agent Evaluation

- [ ] **Graph Topology Extraction** - Extract and visualize LangGraph execution topology from traces
  - [ ] Build DAG from graph/node spans captured by LangGraphInstrumentor
  - [ ] Identify critical path (longest execution chain through nodes)
  - [ ] Detect cycles and redundant node executions
  - [ ] evalyn show-graph --call <id> rendering ASCII or Mermaid diagram
- [ ] **Node-Level Metric Attribution** - Attribute eval failures to specific graph nodes
  - [ ] Map MetricResult failures back to the node span that produced the failing output
  - [ ] Per-node pass rate aggregation across dataset items
  - [ ] Identify "bottleneck nodes" that cause the most failures
- [ ] **Subagent Cost Allocation** - Track cost per subagent in multi-agent traces
  - [ ] Aggregate token/cost from Claude Agent SDK's SubagentContext hierarchy
  - [ ] Per-subagent cost breakdown in show-trace and analyze output
  - [ ] Identify most expensive subagent paths for optimization
- [ ] **Agent Decision Tree Visualization** - Render agent's tool selection choices as a tree
  - [ ] Build decision tree from tool_call/tool_result span sequences
  - [ ] Highlight decision points where agent chose between tools
  - [ ] Compare decision trees across different runs or models

### Pipeline Customization

- [ ] **Custom Pipeline Definitions** - User-defined step sequences beyond the fixed 7-step pipeline
  - [ ] Pipeline definition in evalyn.yaml with ordered step list
  - [ ] Skip/include steps declaratively (instead of --skip-annotation flags)
  - [ ] Custom step plugins: user-defined Python functions as pipeline steps
- [ ] **Pipeline Templates** - Preset pipelines for different evaluation goals
  - [ ] "quick-check" template: build-dataset -> objective metrics only -> analyze
  - [ ] "full-audit" template: all 7 steps + simulation + deep insights
  - [ ] "ci-gate" template: objective metrics + threshold check + exit code
  - [ ] evalyn one-click --template quick-check
- [ ] **Pipeline Comparison** - Compare results of two one-click pipeline runs
  - [ ] evalyn compare-pipelines <dir1> <dir2>
  - [ ] Step-by-step comparison: dataset size, metric count, scores, cost
  - [ ] Identify which pipeline changes improved or degraded results

### Infrastructure & Platform

- [ ] **Web Dashboard** - Browser-based UI for viewing traces, datasets, and results
  - [ ] Trace viewer with span tree navigation (like Phoenix/LangSmith)
  - [ ] Dataset browser with item search, sort, and filter
  - [ ] Eval run comparison view with metric charts
  - [ ] Real-time run progress monitoring
  - [ ] Lightweight server (Flask/FastAPI) bundled with evalyn
- [ ] **CI/CD Integration** - GitHub Actions for automated testing and evaluation on PR
  - [ ] GitHub Action YAML template for evalyn run-eval
  - [ ] PR comment bot posting eval results as markdown table
  - [ ] Regression gate: fail CI if metrics drop below threshold
  - [ ] Artifact upload of HTML reports and datasets
  - [ ] GitLab CI and Jenkins pipeline examples
- [x] **Regression Detection** - Automatic alerts when metrics drop below threshold
- [ ] **Multi-model Comparison** - Compare same prompts across different LLM providers
  - [ ] --models flag to run same eval across multiple providers in one command
  - [ ] Cross-model comparison table (rows=items, columns=models)
  - [ ] Cost/latency/quality trade-off analysis per model
  - [ ] Best-model-per-item analysis
- [ ] **Cost Tracking Dashboard** - Visualize LLM API costs over time
  - [ ] Per-run cost breakdown by metric and provider
  - [ ] Cumulative cost chart across all runs
  - [ ] Cost-per-item and cost-per-metric averages
  - [ ] Budget forecast based on historical usage
- [ ] **API Server Mode** - REST API for programmatic access
  - [ ] REST endpoints: /runs, /traces, /datasets, /metrics
  - [ ] Trigger eval runs via POST /runs with JSON config
  - [ ] WebSocket endpoint for real-time run progress
  - [ ] API key authentication for multi-user access
- [ ] **Team Collaboration** - Multi-user annotation with conflict resolution
  - [ ] User identity tracking on annotations
  - [ ] Assignment queue: distribute items across annotators
  - [ ] Conflict detection when multiple users annotate same item
  - [ ] Resolution strategies: majority vote, senior override, discussion
- [ ] **Cloud Storage Backend** - Optional S3/GCS storage for large datasets
  - [ ] S3-compatible backend implementing StorageBackend protocol
  - [ ] GCS backend with service account authentication
  - [ ] Hybrid mode: SQLite for metadata, cloud for large payloads
  - [ ] Configurable via evalyn.yaml storage section
- [ ] **Storage Compaction** - Vacuum and optimize SQLite database on demand
  - [ ] evalyn compact command to VACUUM and ANALYZE
  - [ ] Auto-compaction trigger when DB exceeds size threshold
  - [ ] Orphan cleanup: remove spans not linked to any function_call
- [ ] **Data Retention Policies** - Auto-delete traces and runs older than a configurable threshold
  - [ ] retention_days setting in evalyn.yaml
  - [ ] evalyn purge --older-than 30d command
  - [ ] Exempt pinned/starred runs from auto-deletion
  - [ ] Dry-run mode showing what would be deleted
- [ ] **Storage Migration** - Export/import data between different storage backends
  - [ ] evalyn export-db --format sqlite/json/parquet
  - [ ] evalyn import-db to load from another backend
  - [ ] Schema version validation on import
  - [ ] Incremental export: only new data since last export
- [ ] **Encrypted Storage** - At-rest encryption for sensitive trace and evaluation data
  - [ ] SQLCipher integration for encrypted SQLite
  - [ ] Key management via environment variable or keyring
  - [ ] Selective encryption: encrypt input/output payloads, keep metadata queryable
- [ ] **Storage Statistics** - Show database size, row counts, and growth rate over time
  - [ ] evalyn storage-stats command
  - [ ] Row counts per table (function_calls, eval_runs, annotations, otel_spans)
  - [ ] Size breakdown: data vs index vs free space
  - [ ] Growth rate: new rows per day/week
- [ ] **Plugin System** - Third-party metric, instrumentor, and storage backend plugins via entry points
  - [ ] Python entry_points discovery for evalyn.metrics, evalyn.instrumentors, evalyn.storage
  - [ ] Plugin manifest with version compatibility declaration
  - [ ] evalyn list-plugins command
  - [ ] Plugin isolation: plugins cannot modify core behavior
- [ ] **Webhook Notifications** - Trigger HTTP webhooks on eval completion, failure, or regression
  - [ ] Configurable webhook URLs in evalyn.yaml
  - [ ] Event types: run_complete, regression_detected, annotation_needed
  - [ ] Payload includes run summary, metric scores, and delta from previous
  - [ ] Retry with exponential backoff on delivery failure
- [ ] **Rate Limit Awareness** - Respect LLM provider rate limits with automatic throttling during evaluation
  - [ ] Per-provider rate limit config (RPM, TPM) in evalyn.yaml
  - [ ] Adaptive backoff when 429 errors received
  - [ ] Token bucket rate limiter shared across parallel workers
  - [ ] Rate limit status in progress callback output
- [ ] **Connection Pooling** - Reuse SQLite connections for high-throughput multi-threaded evaluation
  - [ ] Thread-local connection pool with configurable max size
  - [ ] Connection health checking and recycling
  - [ ] WAL mode auto-enable for concurrent readers
- [ ] **Incremental Backup** - Periodic automatic backup of database to a secondary location
  - [ ] SQLite online backup API integration
  - [ ] Configurable backup schedule and destination path
  - [ ] Backup rotation: keep last N backups
- [ ] **Auto Model Selection** - Choose judge model based on task complexity (fast model for easy items, smart model for hard ones)
  - [ ] Complexity heuristic based on input length, output length, and metric type
  - [ ] Model routing: flash-lite for simple items, flash for complex items
  - [ ] Cost savings report showing how much auto-selection saved vs always-smart

### Data & Dataset

- [ ] **Dataset Versioning** - Track dataset changes over time with diff view
  - [ ] Content-hash versioning on each build-dataset invocation
  - [ ] Diff view: items added, removed, and modified between versions
  - [ ] Version log stored alongside dataset.jsonl
  - [ ] Rollback to previous version via evalyn dataset-rollback
- [ ] **Synthetic Data Generation**
  - [ ] Adversarial example generation
  - [ ] Edge case mining
  - [ ] Demographic variation
  - [ ] Domain-specific generators
- [ ] **Data Augmentation** - Automatically expand datasets
  - [ ] Paraphrase generation: rephrase inputs preserving semantics
  - [ ] Input perturbation: typos, casing, formatting variations
  - [ ] Language translation: generate multilingual variants
  - [ ] Context expansion: add/remove context to test robustness
- [ ] **Golden Set Management** - Curate and maintain evaluation benchmarks
  - [ ] evalyn golden-set create/add/remove commands
  - [ ] Lock golden set items from modification
  - [ ] Track golden set coverage: % of metrics with golden examples
  - [ ] Periodic validation: re-evaluate golden set to detect model drift
- [ ] **Dataset Splitting** - Train/test/validation splits with stratification by metadata fields
  - [ ] evalyn split-dataset --ratio 0.7/0.15/0.15
  - [ ] Stratification by metadata keys (tag, source, difficulty)
  - [ ] Deterministic splitting with configurable random seed
  - [ ] Output as separate JSONL files in split/ subdirectory
- [ ] **Dataset Statistics** - Auto-compute input/output length distributions, token counts, label balance
  - [ ] evalyn dataset-stats command
  - [ ] Input/output token count histograms
  - [ ] Metadata field value distributions
  - [ ] Expected reference coverage (% items with ground truth)
  - [ ] Duplicate detection report
- [ ] **Dataset Merge and Diff** - Combine two datasets or show item-level differences between them
  - [ ] evalyn dataset-merge --deduplicate
  - [ ] evalyn dataset-diff showing added/removed/changed items
  - [ ] Conflict resolution for items with same ID but different content
- [ ] **External Format Import** - Import from HuggingFace datasets, LMSYS Arena, or custom CSV schemas
  - [ ] evalyn import --format huggingface --dataset-name <name>
  - [ ] CSV import with column mapping config
  - [ ] LMSYS Arena format (conversation pairs with human preference)
  - [ ] Auto-detect format from file extension and content
- [ ] **Schema Evolution** - Handle format changes across dataset versions with automatic migration
  - [ ] Version field in dataset header line
  - [ ] Automatic migration on load (old format to current)
  - [ ] Migration log showing which transformations were applied
- [ ] **Dataset Sampling Preview** - Show sample items and summary stats before building full dataset
  - [ ] --preview flag on build-dataset showing 5 sample items
  - [ ] Summary: item count, avg input/output length, metadata distribution
  - [ ] Confirmation prompt before writing full dataset
- [ ] **Dataset Pinning** - Lock a dataset version hash for reproducible evaluations across environments
  - [ ] SHA-256 hash stored in dataset metadata
  - [ ] --pinned flag on run-eval to verify hash before evaluation
  - [ ] Pin file (.evalyn-pin) for CI/CD reproducibility
- [ ] **Dataset Lineage** - Track which traces and runs produced each dataset item
  - [ ] Source trace ID and function_call ID in item metadata
  - [ ] Lineage query: "which traces contributed to this dataset?"
  - [ ] Reverse lineage: "which datasets use this trace?"
- [ ] **Dataset Filtering DSL** - Query-based item filtering (e.g. "items where output_length > 500 and tag=production")
  - [ ] --filter flag on build-dataset and run-eval
  - [ ] Operators: =, !=, >, <, contains, matches (regex)
  - [ ] Compound filters with AND/OR
  - [ ] Filter on metadata fields, input/output length, and item ID patterns
- [ ] **Incremental Dataset Build** - Append new traces to an existing dataset without full rebuild
  - [ ] --append flag on build-dataset
  - [ ] Track last-build timestamp to only process new traces
  - [ ] Deduplication against existing items using hash_inputs
- [ ] **Dataset Health Check** - Validate dataset quality before evaluation
  - [ ] Reference coverage: % of items with ground truth (uses _dataset_has_reference logic)
  - [ ] Empty/null field detection in input, output, and metadata
  - [ ] Duplicate input detection via hash_inputs
  - [ ] evalyn dataset-health command with pass/warn/fail summary

### Reporting & Analytics

- [ ] **Custom Report Templates** - User-defined HTML report layouts
  - [ ] Jinja2 template engine for HTML report customization
  - [ ] Template variables: run data, analysis, insights, charts
  - [ ] Built-in templates: executive summary, technical deep-dive, compliance
  - [ ] evalyn export --template custom_template.html
- [ ] **Slack/Discord Notifications** - Alert on evaluation completion or failures
  - [ ] Slack webhook integration with rich message formatting
  - [ ] Discord webhook with embedded metric summary
  - [ ] Configurable alert thresholds: only notify on regression or failure
  - [ ] Channel routing: different alerts to different channels
- [x] **Metric Correlation Analysis** - Understand relationships between metrics
- [ ] **Failure Root Cause Analysis** - Automated diagnosis of failures
  - [ ] LLM-powered analysis of common patterns in failed items
  - [ ] Feature attribution: which input features correlate with failure
  - [ ] Failure clustering by root cause category (prompt, data, model, tool)
  - [ ] Actionable fix suggestions per failure cluster
- [ ] **Trend Anomaly Detection** - Alert on unusual metric patterns
  - [ ] Z-score based anomaly detection on metric time series
  - [ ] Configurable sensitivity threshold
  - [ ] Automatic alert when anomaly detected during trend analysis
  - [ ] Visual anomaly markers in trend charts
- [ ] **Cohort Analysis** - Compare metrics across user-defined item groups (by metadata, input length, etc.)
  - [ ] --cohort-by flag on analyze command (split by metadata field)
  - [ ] Per-cohort metric statistics and pass rates
  - [ ] Cross-cohort comparison table
  - [ ] Identify worst-performing cohort with improvement suggestions
- [ ] **Statistical Significance Testing** - P-values and confidence intervals for run-to-run comparisons
  - [ ] Two-proportion z-test for pass rate differences
  - [ ] Bootstrap confidence intervals for score means
  - [ ] Effect size (Cohen's d) alongside p-values
  - [ ] Automatic significance flag in compare output
- [ ] **Judge Confusion Matrix** - Visualize agreement/disagreement patterns between judge and human
  - [ ] 2x2 matrix: TP/FP/TN/FN per metric
  - [ ] ASCII table and HTML heatmap renderers
  - [ ] Per-metric confusion matrix in annotation-stats
  - [ ] Aggregate confusion matrix across all metrics
- [ ] **Jupyter Notebook Export** - Generate .ipynb with pre-built charts and analysis from eval runs
  - [ ] evalyn export --format notebook
  - [ ] Pre-built cells: data loading, metric charts, distribution plots, correlations
  - [ ] Interactive widgets for filtering by metric, item, or cohort
  - [ ] nbformat-based generation (no Jupyter dependency required)
- [ ] **Metric Budget Analysis** - Estimate cost savings from dropping low-signal metrics
  - [ ] Compute information gain of each metric (redundancy with others)
  - [ ] Cost attribution: how much each metric costs per run
  - [ ] Recommended metric subset that preserves N% of signal at minimum cost
- [ ] **Regression Bisection** - Binary search across dataset items to pinpoint exact cause of a regression
  - [ ] evalyn bisect --baseline <run1> --current <run2>
  - [ ] Identify items that changed from pass to fail
  - [ ] Cluster newly-failing items by input features
  - [ ] Rank items by regression severity (score delta)
- [ ] **Comparative Heatmap** - Visual heatmap of metric scores across items and runs
  - [ ] Items on Y-axis, metrics on X-axis, color = score
  - [ ] Multi-run heatmap: side-by-side comparison
  - [ ] ASCII heatmap for terminal, HTML/SVG for reports
  - [ ] Sort by worst-performing items or metrics
- [ ] **Failure Taxonomy** - Auto-categorize failures into a structured taxonomy (prompt, model, data, tool)
  - [ ] LLM-powered categorization of failure reasons
  - [ ] Built-in taxonomy: prompt_ambiguity, model_limitation, data_quality, tool_error, hallucination
  - [ ] Custom taxonomy definition in evalyn.yaml
  - [ ] Taxonomy distribution chart in analysis output
- [ ] **Analysis Snapshots** - Save analysis state at a point in time for later comparison
  - [ ] evalyn snapshot --name "pre-refactor" saves RunAnalysis + InsightsReport
  - [ ] evalyn compare-snapshots for before/after comparison
  - [ ] Snapshots stored in .evalyn/ directory as JSON

### Interoperability

- [ ] **Phoenix/Langfuse Trace Export** - Native export to popular LLM observability platforms
  - [ ] evalyn export-traces --format phoenix to produce Phoenix-compatible JSONL
  - [ ] evalyn export-traces --format langfuse for Langfuse import format
  - [ ] Preserve span hierarchy and OpenInference attributes in export
- [ ] **Trace Import from External Platforms** - Bring existing traces into evalyn for evaluation
  - [ ] evalyn import-traces --format phoenix/langfuse/otel
  - [ ] Map external span types to Evalyn span types via conventions.py
  - [ ] Deduplicate against existing traces by span ID
- [ ] **OpenInference Full Compliance** - Complete implementation of OpenInference semantic conventions
  - [ ] Full document/retrieval attribute capture (DocumentAttributes, RetrievalAttributes)
  - [ ] Embedding attribute capture (EmbeddingAttributes.EMBEDDINGS, TEXT)
  - [ ] Session and user attribute propagation (SessionAttributes)
  - [ ] Reranker score capture and display in show-trace
- [ ] **Eval Result Export to Observability Platforms** - Push evaluation scores back to trace viewers
  - [ ] Annotate Phoenix spans with evalyn metric scores
  - [ ] Push eval results as Langfuse scores
  - [ ] Bi-directional sync: traces in, scores out

### Resilience & Error Handling

- [ ] **Circuit Breaker for Providers** - Stop calling a provider after N consecutive failures
  - [ ] Configurable failure threshold (default: 5 consecutive errors)
  - [ ] Cool-down period before retrying (exponential backoff)
  - [ ] Automatic fallback to alternative provider when circuit opens
  - [ ] Circuit state visible in progress output
- [ ] **Graceful Item-Level Failure** - Continue evaluation when individual items fail
  - [ ] Catch and log per-item errors without stopping the run
  - [ ] Record failure reason in MetricResult.details
  - [ ] Summary of failed items at end of run with error categories
  - [ ] --fail-fast flag to override and stop on first error
- [ ] **Provider Fallback Chain** - Automatically try alternative providers on failure
  - [ ] Ordered provider list: [gemini, openai, ollama]
  - [ ] Fall back to next provider on timeout, rate limit, or API error
  - [ ] Log which provider was actually used per item
- [ ] **Evaluation Timeout Per Item** - Prevent single slow items from blocking the entire run
  - [ ] --item-timeout flag (default: 120s per item)
  - [ ] Timeout recorded as failure with reason "timeout"
  - [ ] Separate timeout for objective vs subjective metrics

### Output & Formatting

- [ ] **Color-Coded Terminal Output** - ANSI colors for pass/fail/warning states
  - [ ] Green for pass, red for fail, yellow for warning across all commands
  - [ ] Respect NO_COLOR env var and --no-color flag for CI environments
  - [ ] Color-coded score ranges in analyze and compare output
- [ ] **Compact Output Mode** - Minimal output for CI logs and scripting
  - [ ] --compact flag producing single-line summaries per command
  - [ ] Summary format: "RUN <id> PASS 85% (17/20) COST $0.12 TIME 45s"
  - [ ] Pair with exit codes for CI gate integration (exit 1 if pass rate < threshold)
- [ ] **PDF Report Export** - Generate PDF reports from HTML dashboards
  - [ ] evalyn export --format pdf using headless browser or weasyprint
  - [ ] Page breaks between sections, print-friendly layout
  - [ ] Cover page with run metadata, date, project name
- [ ] **HTML Report Dark Mode** - Dark theme option for HTML dashboards and insights
  - [ ] CSS dark mode support via prefers-color-scheme media query
  - [ ] Manual toggle button in report header
  - [ ] Dark-friendly Chart.js color palette

### Code Change Tracking

- [ ] **Source Code Diff Correlation** - Track agent code changes alongside metric changes
  - [ ] Store source_hash from _extract_code_meta in each eval run
  - [ ] Detect when source code changed between consecutive runs
  - [ ] Correlate code diffs with metric deltas in compare output
  - [ ] evalyn code-diff --run1 <id> --run2 <id> showing code changes alongside score changes
- [ ] **Prompt Version Tracking** - Track judge prompt changes across calibration rounds
  - [ ] Hash judge prompts and store in MetricResult metadata
  - [ ] Warn when comparing runs that used different prompt versions
  - [ ] Prompt changelog: show how each metric's prompt evolved over time

### Programmatic SDK

- [ ] **Python API for Running Evaluations** - Run evaluations from Python code without CLI
  - [ ] evalyn.run(dataset, metrics, provider) returning EvalRun object
  - [ ] evalyn.analyze(run) returning RunAnalysis directly
  - [ ] evalyn.compare(run_a, run_b) returning comparison dict
  - [ ] Async variants: await evalyn.run_async(...)
- [ ] **Event Callback Hooks** - Register functions that fire on evaluation events
  - [ ] on_item_complete(callback) for per-item processing
  - [ ] on_metric_complete(callback) for per-metric processing
  - [ ] on_run_complete(callback) for post-run triggers
  - [ ] Hook registration via evalyn.yaml or Python API
- [ ] **Context Manager Tracing** - Manual span creation with `with` syntax
  - [ ] with evalyn.span("name", "type") as s: for explicit span boundaries
  - [ ] Automatic parent-child linking via context propagation
  - [ ] Span attribute setting: s.set_attribute("key", "value")
- [ ] **Embedding as Library** - Use evalyn as imported library in test suites
  - [ ] pytest plugin: @pytest.mark.evalyn(metrics=["helpfulness"])
  - [ ] Assert on metric scores: assert result.metrics["helpfulness"].passed
  - [ ] Integration with pytest-xdist for parallel testing

### Testing & Quality Enhancements

- [ ] **Snapshot Testing for Metrics** - Detect unintended changes to metric scoring behavior
  - [ ] Record expected scores for a golden dataset
  - [ ] Flag when metric output changes (new code, model update)
  - [ ] evalyn test-metrics --update-snapshots to accept changes
- [ ] **Performance Benchmark Suite** - Track and prevent performance regressions in evalyn itself
  - [ ] Benchmarks for: dataset loading, metric scoring, analysis, export
  - [ ] Baseline timings stored in repo
  - [ ] CI check: fail if any benchmark regresses > 20%
- [ ] **Fuzz Testing for Parsers** - Stress-test JSON/judge output parsing with malformed inputs
  - [ ] Fuzz _extract_json_object and extract_json_list with random strings
  - [ ] Fuzz _parse_passed with edge case values
  - [ ] Ensure no unhandled exceptions on any input

### Rubric Engineering

- [ ] **Multi-Language Rubrics** - Judge prompts and rubrics in languages other than English
  - [ ] Rubric translation support in JUDGE_TEMPLATES (locale field per template)
  - [ ] Language-matched judging: use rubric language matching the output language
  - [ ] Cross-language evaluation: judge non-English outputs with English rubrics vs native rubrics
- [ ] **Community Rubric Library** - Import and export rubrics from a shared repository
  - [ ] evalyn rubric-export --metric <id> producing a portable YAML rubric file
  - [ ] evalyn rubric-import from URL or local file
  - [ ] Rubric metadata: author, version, tested-on, accuracy stats
- [ ] **Rubric Testing** - Validate that a rubric produces consistent scores on test cases
  - [ ] evalyn test-rubric --metric <id> running rubric against a set of known pass/fail items
  - [ ] Consistency score: same rubric, same item, N runs, measure agreement
  - [ ] Edge case detection: find items where rubric is ambiguous (close to threshold)
- [ ] **Domain-Specific Rubric Packs** - Downloadable rubric sets for specialized domains
  - [ ] Medical: HIPAA compliance, clinical accuracy, patient safety, drug interaction checks
  - [ ] Legal: jurisdictional accuracy, precedent citation, privilege preservation
  - [ ] Finance: SEC compliance, fiduciary duty, risk disclosure completeness
  - [ ] evalyn install-rubric-pack medical

### Dashboard Interactivity

- [ ] **Embeddable Widget Mode** - Iframe-friendly dashboard for embedding in other tools
  - [ ] evalyn dashboard --embed producing minimal HTML without navigation chrome
  - [ ] Configurable widget size and chart selection
  - [ ] PostMessage API for parent page communication (filter events, score updates)
- [ ] **In-Dashboard Data Export** - CSV/JSON export buttons on each chart in HTML reports
  - [ ] Download button per chart exporting underlying data as CSV
  - [ ] Full dataset export button in failed items section
  - [ ] Copy-to-clipboard for individual metric summaries
- [ ] **Comparison Overlay Dashboard** - Overlay two runs on same charts for visual comparison
  - [ ] evalyn dashboard --compare <run1> <run2>
  - [ ] Dual bar charts, overlaid radar plots, side-by-side heatmaps
  - [ ] Toggle visibility of each run for clean comparison

### Audit & Governance

- [ ] **Evaluation Audit Trail** - Immutable log of who ran what and when
  - [ ] Record: user, timestamp, command, args, config hash, result summary
  - [ ] Append-only audit log in .evalyn/audit.jsonl
  - [ ] evalyn audit-log showing evaluation history with filters
- [ ] **Data Governance Metadata** - Track data provenance and compliance attributes
  - [ ] Dataset-level tags: PII-present, internal-only, customer-data, synthetic
  - [ ] Eval run compliance flag: was evaluation run on approved infrastructure?
  - [ ] Exportable governance report for compliance audits
- [ ] **Structured Logging** - JSON-formatted logs with configurable verbosity
  - [ ] --log-level flag (debug, info, warning, error) on all commands
  - [ ] JSON log format for machine parsing in production environments
  - [ ] Log file output: --log-file evalyn.log

---

## Completed Features

### Setup & Configuration

- [x] **evalyn init** - Initialize evalyn.yaml config file
- [x] **evalyn one-click** - Run complete pipeline in one command
- [x] **evalyn help** - Show available commands with examples
- [x] **Environment Variables** - GEMINI_API_KEY, OPENAI_API_KEY, EVALYN_NO_HINTS, EVALYN_AUTO_INSTRUMENT

### Tracing & Instrumentation

- [x] **@eval decorator** - Automatic function call tracing
- [x] **Auto-instrumentation** - Automatic LLM SDK patching (OpenAI, Anthropic, Gemini, LangChain, LangGraph)
- [x] **Span tree capture** - Hierarchical trace of LLM calls, tool calls, graph nodes
- [x] **Token & cost tracking** - Automatic token counting and cost estimation
- [x] **evalyn list-calls** - List captured traces with filtering and sorting
- [x] **evalyn show-call** - View detailed call information
- [x] **evalyn show-trace** - Phoenix-style span tree visualization
- [x] **evalyn show-projects** - Project summary with trace counts
- [x] **Streaming response capture** - StreamingSpanWrapper for OpenAI, Anthropic, Gemini
- [x] **GenAI semantic convention attributes** - OpenTelemetry gen_ai.* attributes on spans
- [x] **Span-metric attribution** - Link metric results to specific spans with relevance scoring
- [x] **Context window utilization tracking** - Track context usage in spans
- [x] **--db flag** - Switch between prod/test databases
- [x] **Short ID support** - 8-character ID prefixes for convenience

### Dataset Management

- [x] **evalyn build-dataset** - Build dataset.jsonl from traces
- [x] **evalyn validate** - Validate dataset format
- [x] **evalyn status** - Show comprehensive dataset status
- [x] **--latest flag** - Auto-resolve most recent dataset
- [x] **Production/simulation filtering** - Separate real vs synthetic traces
- [x] **Date range filtering** - --since and --until options

### Metrics System

- [x] **73 Objective Metrics** - Deterministic code-based evaluation
  - [x] Efficiency: latency_ms, cost, token_length, compression_ratio
  - [x] Structure: json_valid, json_schema_keys, regex_match, xml_valid, syntax_valid
  - [x] Correctness: bleu, rouge_l, rouge_1, rouge_2, exact_match, levenshtein_similarity
  - [x] Robustness: tool_call_count, llm_call_count, tool_success_ratio, retry_count
  - [x] Grounding: url_count, citation_count, source_diversity
  - [x] Style: word_count, sentence_count, avg_sentence_length, vocabulary_diversity
  - [x] Diversity: unique_ngrams, type_token_ratio
- [x] **60 Subjective Metrics** - LLM judge evaluation
  - [x] Safety: toxicity_safety, pii_safety, manipulation_resistance, bias_detection
  - [x] Correctness: helpfulness_accuracy, factual_accuracy, technical_accuracy
  - [x] Style: tone_alignment, formality_match, brand_voice_consistency
  - [x] Instruction: instruction_following, constraint_adherence, format_compliance
  - [x] Grounding: hallucination_risk, source_attribution, claim_verification
  - [x] Agent: reasoning_quality, tool_use_appropriateness, planning_quality
  - [x] Domain: medical_accuracy, legal_compliance, financial_prudence
  - [x] Conversation: context_retention, memory_consistency, empathy, patience
- [x] **evalyn list-metrics** - List all available metrics
- [x] **evalyn suggest-metrics** - Suggest metrics for a function
  - [x] basic mode - Fast heuristic-based
  - [x] bundle mode - Pre-configured metric sets
  - [x] llm-registry mode - LLM picks from registry
  - [x] llm-brainstorm mode - LLM generates custom metrics
  - [x] auto mode - Uses function hints or defaults
- [x] **evalyn select-metrics** - Interactive LLM-guided selection

### Metric Bundles (17 Curated Sets)

- [x] **Conversational AI**
  - [x] chatbot - Safety, helpfulness, multi-turn memory
  - [x] customer-support - Empathy, patience, escalation handling
- [x] **Content Generation**
  - [x] content-writer - Style, engagement, readability
  - [x] summarization - Compression, reference overlap, grounding
  - [x] creative-writer - Originality, engagement, vocabulary diversity
- [x] **Knowledge & Research**
  - [x] rag-qa - Grounding, citations, factual accuracy
  - [x] research-agent - Citations, grounding, tool use
  - [x] tutor - Pedagogical clarity, examples, patience
- [x] **Code & Technical**
  - [x] code-assistant - Syntax validity, complexity, technical accuracy
  - [x] data-extraction - JSON validity, schema compliance
- [x] **Agents & Orchestration**
  - [x] orchestrator - Tool success, planning, error handling
  - [x] multi-step-agent - Planning, context retention, memory
- [x] **High-Stakes Domains**
  - [x] medical-advisor - Medical accuracy, safety, ethics
  - [x] legal-assistant - Legal compliance, citations, accuracy
  - [x] financial-advisor - Financial prudence, safety, ethics
- [x] **Safety & Translation**
  - [x] moderator - Toxicity, bias, PII, manipulation
  - [x] translator - BLEU, Levenshtein, cultural sensitivity

### Evaluation Engine

- [x] **evalyn run-eval** - Run evaluation on dataset
- [x] **Parallel execution** - Multi-threaded metric evaluation (--workers)
- [x] **Batch API mode** - 50% cost savings for large-scale evaluation (--batch)
  - [x] Gemini batch provider
  - [x] OpenAI batch provider
  - [x] Anthropic batch provider
- [x] **Confidence estimation** - Confidence scores for LLM judgments (--confidence)
  - [x] Logprobs-based confidence (OpenAI/Ollama)
  - [x] DeepConf confidence (Meta AI's bottom-10% strategy)
  - [x] Self-consistency confidence (multi-sample agreement)
  - [x] Perplexity and entropy methods
- [x] **Multi-provider support** - Choose judge provider (--provider)
  - [x] Gemini (default)
  - [x] OpenAI
  - [x] Ollama (local)
- [x] **Token usage tracking** - Track LLM API token consumption per eval run
  - [x] Per-metric input/output token counts
  - [x] Aggregated usage summary in EvalRun
  - [x] Display in run-eval output and show-run command
- [x] **Checkpoint & resume** - Save progress on interrupt, resume later
- [x] **HTML reports** - Interactive visualization with Chart.js
- [x] **evalyn list-runs** - List past evaluation runs
- [x] **evalyn show-run** - View run details
- [x] **--use-calibrated** - Apply calibrated prompts

### Analysis & Insights

- [x] **evalyn analyze** - Analyze evaluation results
- [x] **evalyn compare** - Compare two runs side-by-side
- [x] **evalyn trend** - View metric trends over time
- [x] **evalyn cluster-failures** - Cluster failed items by failure reason
- [x] **evalyn cluster-misalignments** - Cluster judge vs human disagreements
- [x] **Pass rate charts** - ASCII bar charts in terminal
- [x] **Score distributions** - Mini histograms
- [x] **Failed item breakdown** - List items with failure reasons
- [x] **evalyn insights** - Comprehensive diagnostic, prescriptive, and proactive analysis
  - [x] Metric correlations, regressions, distributions, feature analysis
  - [x] Prioritized recommendations
  - [x] LLM expert panel (--deep) with 4 expert roles + moderator synthesis
  - [x] Interactive HTML dashboard (--format html) with Chart.js charts

### Annotation Enhancements

- [ ] **Inter-Annotator Agreement** - Track and visualize consistency between multiple annotators
  - [ ] Cohen's Kappa and Krippendorff's Alpha per metric
  - [ ] Pairwise agreement matrix across annotators
  - [ ] Identify items with highest disagreement for re-annotation
  - [ ] Agreement trend over time as annotators calibrate
- [ ] **Annotation Delegation** - Assign specific items to specific annotators by expertise
  - [ ] Annotator profiles with domain expertise tags
  - [ ] Auto-assignment based on item metadata and annotator expertise match
  - [ ] Workload balancing across annotators
  - [ ] Progress dashboard per annotator
- [ ] **Bulk Pre-Annotation via LLM** - Use LLM to pre-fill annotations for human review and correction
  - [ ] evalyn pre-annotate --provider gemini to generate draft annotations
  - [ ] Confidence-based triage: auto-accept high-confidence, human-review low-confidence
  - [ ] Track pre-annotation accuracy vs human corrections
  - [ ] Use corrections to improve pre-annotation prompts
- [ ] **Annotation Guidelines Generator** - Auto-generate annotation guidelines from metric definitions
  - [ ] Convert metric rubrics to annotator-friendly instructions
  - [ ] Include concrete pass/fail examples from existing annotations
  - [ ] Export as markdown document or HTML with examples
- [ ] **Annotation Conflict Resolution UI** - Side-by-side view when annotators disagree, with tiebreaker workflow
  - [ ] Display both annotators' labels with their confidence and reasoning
  - [ ] Third-party tiebreaker annotation with full context
  - [ ] Resolution policies: majority vote, senior override, discussion required

### Human Annotation

- [x] **evalyn annotate** - Interactive annotation interface
  - [x] Simple mode - Overall pass/fail
  - [x] Per-metric mode - Agree/disagree with each metric
  - [x] Span mode - Annotate individual LLM/tool calls
- [x] **evalyn annotation-stats** - Show annotation coverage
- [x] **evalyn import-annotations** - Import from JSONL
- [x] **evalyn export-for-annotation** - Export for external tools
- [x] **Confidence scores** - 1-5 scale for annotation certainty
- [x] **Immediate save** - Each annotation saved instantly

### Calibration (LLM Judge Optimization)

- [x] **evalyn calibrate** - Optimize judge prompts
  - [x] Basic method - Single-shot LLM analysis of disagreements
  - [x] APE method - Search-based optimization with UCB selection
  - [x] OPRO method - Trajectory-based optimization
  - [x] GEPA method - Evolutionary prompt optimization (external library)
  - [x] GEPA-Native method - Evolutionary optimization with token tracking
  - [x] EvoPrompt method - Population-based mutation/crossover
  - [x] TextGrad method - Iterative critique-revise refinement
  - [x] MIPROv2 method - Joint instruction + few-shot demo optimization
  - [x] PromptBreeder method - Self-referential prompt evolution
  - [x] BaseOptimizer base class + factory dispatch
- [x] **evalyn list-calibrations** - List calibration records
- [x] **Alignment metrics** - Accuracy, precision, recall, F1, Cohen's Kappa
- [x] **Validation split** - Test calibration on held-out samples

### Simulation (Synthetic Data)

- [x] **evalyn simulate** - Generate synthetic test data
  - [x] similar mode - Variations of existing queries
  - [x] outlier mode - Edge cases and unusual inputs
- [x] **Temperature control** - Separate temps for similar/outlier
- [x] **Seed sampling** - Control number of seed examples
- [ ] **Persona-Based Simulation** - Generate inputs as specific user personas (novice, expert, adversarial)
  - [ ] Built-in personas: novice user, power user, adversarial attacker, non-native speaker
  - [ ] Custom persona definitions in evalyn.yaml
  - [ ] Persona tag in generated item metadata for cohort analysis
- [ ] **Multi-Turn Simulation** - Generate full multi-turn conversations, not just single queries
  - [ ] Configurable conversation length (2-10 turns)
  - [ ] Follow-up generation based on agent response
  - [ ] Conversation flow patterns: clarification, topic shift, error recovery
- [ ] **Adversarial Simulation** - Deliberately craft inputs targeting known failure modes
  - [ ] Prompt injection attempts
  - [ ] Boundary inputs: empty, max length, special characters, unicode edge cases
  - [ ] Contradiction inputs that conflict with system prompt
  - [ ] Jailbreak pattern variations
- [ ] **Domain Transfer Simulation** - Adapt seed inputs from one domain to another (e.g. medical to legal)
  - [ ] LLM-powered domain rewriting preserving query structure
  - [ ] Domain vocabulary substitution
  - [ ] Complexity preservation across domain transfer
- [ ] **Regression Simulation** - Re-generate past failure inputs to verify they no longer fail
  - [ ] Extract failure patterns from cluster-failures output
  - [ ] Generate new inputs matching each failure pattern
  - [ ] Track fix rate: % of previously-failing patterns now passing
- [ ] **Conditional Simulation** - Generate inputs that specifically test edge conditions (empty input, max length, unicode)
  - [ ] Edge condition library: empty, null, max_length, unicode, mixed_language
  - [ ] Combinatorial generation across edge conditions
  - [ ] Configurable via --conditions flag
- [ ] **Simulation Validation** - Auto-verify that generated items match expected statistical distributions
  - [ ] Input length distribution comparison (generated vs seed)
  - [ ] Vocabulary overlap check between generated and seed
  - [ ] Deduplication against both seed and existing dataset
- [ ] **Parallel Simulation** - Generate synthetic data with configurable concurrency for large-scale runs
  - [ ] --workers flag on simulate command
  - [ ] Batch LLM calls for generation efficiency
  - [ ] Progress bar with items generated / total target

### Sampling

- [ ] **Importance Sampling** - Weight sample selection by item difficulty or model uncertainty
  - [ ] Weight by inverse pass rate from previous eval run
  - [ ] Weight by judge confidence (low confidence = high importance)
  - [ ] Configurable weight function via Python callable
- [ ] **Curriculum Sampling** - Order samples from easy to hard for progressive evaluation
  - [ ] Difficulty estimation from input length, complexity heuristics, or past scores
  - [ ] Progressive disclosure: evaluate easy items first, add harder ones
  - [ ] Early stopping if easy items already fail
- [ ] **Time-Weighted Sampling** - Prefer recent traces over older ones during dataset construction
  - [ ] Exponential decay weighting by trace timestamp
  - [ ] Configurable half-life parameter (e.g. 7 days, 30 days)
  - [ ] Minimum representation guarantee for older traces
- [ ] **Coverage-Aware Sampling** - Maximize coverage of the input feature space
  - [ ] Embedding-based coverage using existing SentenceTransformer infrastructure
  - [ ] Greedy maximal-diversity selection
  - [ ] Coverage report: % of embedding space represented
- [ ] **Balanced Sampling** - Ensure equal representation across metadata categories or labels
  - [ ] Balance by any metadata field (tag, source, difficulty)
  - [ ] Undersample majority or oversample minority categories
  - [ ] Report sampling ratio adjustments applied
- [ ] **Adversarial Sampling** - Select items most likely to trigger model failures based on past results
  - [ ] Prioritize items that failed in previous runs
  - [ ] Select items near decision boundaries (scores close to threshold)
  - [ ] Include items from underperforming cohorts
- [ ] **Score-Stratified Sampling** - Ensure representation across the full metric score range
  - [ ] Bin items by score range (0-0.2, 0.2-0.4, ..., 0.8-1.0)
  - [ ] Equal sampling from each bin
  - [ ] Useful for calibration datasets needing score diversity

### Export & Reporting

- [x] **evalyn export** - Export results in multiple formats
  - [x] JSON - Full structured data
  - [x] CSV - Spreadsheet-compatible
  - [x] Markdown - Human-readable report
  - [x] HTML - Standalone interactive report
- [x] **evalyn export-for-annotation** - Export for external annotation tools

### Developer Experience

- [x] **Context-aware hints** - Suggests next steps after each command
- [x] **--quiet flag** - Suppress hints
- [x] **--format flag** - table/json output for all commands
- [x] **--last flag** - Quick access to most recent item
- [x] **Short IDs** - 8-character prefixes for easier use
- [x] **Error messages with hints** - Helpful troubleshooting suggestions

### CLI Enhancements

- [ ] **Interactive TUI Mode** - Rich terminal UI with navigation, filtering, and drill-down
  - [ ] Textual or Rich-based TUI framework
  - [ ] Views: trace list, run list, metric dashboard, item detail
  - [ ] Keyboard navigation: j/k scroll, enter drill-down, q quit
  - [ ] Real-time eval progress view with per-metric status
- [ ] **Shell Completion** - Bash/zsh/fish tab completion for all commands and flags
  - [ ] argcomplete integration for automatic completion generation
  - [ ] Complete command names, flag names, and flag values (run IDs, dataset paths)
  - [ ] Installation helper: evalyn --install-completion
- [ ] **Watch Mode** - Auto-rerun evaluation when dataset or config file changes
  - [ ] File watcher on dataset.jsonl and evalyn.yaml
  - [ ] Debounce: wait 2s after last change before re-running
  - [ ] Diff output: only show changed metrics since last run
  - [ ] --watch flag on run-eval command
- [ ] **Profile Command** - Show storage size, run counts, disk usage, and system health
  - [ ] Database file size and table row counts
  - [ ] Total eval runs, traces, and annotations
  - [ ] Disk usage by data directory
  - [ ] Python environment info: version, installed providers, API key status
- [ ] **Config Validation Command** - Check evalyn.yaml for errors, missing fields, and deprecations
  - [ ] Schema validation against expected evalyn.yaml structure
  - [ ] Warn on unknown keys, deprecated fields, and type mismatches
  - [ ] Suggest fixes for common misconfigurations
  - [ ] evalyn config-check command
- [ ] **evalyn doctor** - Diagnose common setup issues (missing API keys, stale data, broken config)
  - [ ] Check API key validity for each configured provider
  - [ ] Verify database accessibility and schema version
  - [ ] Check disk space and write permissions
  - [ ] Verify Python dependencies are installed (sentence-transformers, etc.)
  - [ ] Generate diagnostic report for bug reports
- [ ] **evalyn playground** - Interactive prompt testing with live metric scoring in the terminal
  - [ ] Enter input, see agent output, instantly score with selected metrics
  - [ ] Side-by-side: original prompt vs modified prompt
  - [ ] Score history across playground iterations
  - [ ] Save good examples to dataset
- [ ] **evalyn diff** - Diff two evaluation runs showing changed scores per item
  - [ ] Per-item score delta table sorted by largest regression
  - [ ] Metric-level summary: improved/regressed/unchanged counts
  - [ ] --threshold flag to only show items with delta > N
  - [ ] ASCII color coding: green for improvement, red for regression
- [ ] **evalyn gc** - Garbage collect orphaned data (stale checkpoints, runs without datasets)
  - [ ] Identify orphaned checkpoint files without matching runs
  - [ ] Find runs referencing deleted datasets
  - [ ] Remove temporary files in .evalyn/ directory
  - [ ] --dry-run mode showing what would be cleaned
- [ ] **Piped JSON Mode** - Machine-readable JSON output for scripting and CI pipeline integration
  - [ ] --output json on all commands producing structured JSON to stdout
  - [ ] JSONL streaming for long-running operations (progress events)
  - [ ] Exit codes: 0=pass, 1=fail, 2=error for CI gate integration
  - [ ] jq-friendly output structure

### Metrics Enhancements

- [ ] **Custom Metric DSL** - Define metrics via YAML config without writing Python code
  - [ ] YAML metric definition: name, type, prompt template, threshold, scoring rubric
  - [ ] Variable interpolation: {{input}}, {{output}}, {{expected}} in prompt templates
  - [ ] Custom objective metrics via Python expressions (e.g. "len(output) < 500")
  - [ ] Hot-reload: modify YAML, re-run eval without code changes
- [ ] **Metric Composition** - Combine multiple metrics into weighted composite scores
  - [ ] Composite metric definition: weighted average of child metrics
  - [ ] Min/max/mean aggregation strategies
  - [ ] Pass threshold on composite score
  - [ ] Drill-down: see child metric contributions to composite
- [ ] **Metric Weighting Profiles** - Named weight sets for different evaluation use cases
  - [ ] Profile definitions in evalyn.yaml (e.g. "safety-first": safety=3x, quality=1x)
  - [ ] --weight-profile flag on analyze and compare commands
  - [ ] Weighted pass rate and weighted overall score
- [ ] **Metric Versioning** - Track when metric implementations change and flag affected runs
  - [ ] Hash metric prompt + scoring logic as version identifier
  - [ ] Store metric version in MetricResult metadata
  - [ ] Warn when comparing runs with different metric versions
  - [ ] evalyn metric-history showing version changes over time
- [ ] **Metric Benchmarking** - Measure computation cost and latency per metric
  - [ ] Per-metric timing in evaluation runner
  - [ ] Token usage and cost per metric type
  - [ ] Benchmark report: slowest metrics, most expensive metrics
  - [ ] Optimization suggestions for costly metrics
- [ ] **Inter-Rater Reliability** - Compute agreement stats when multiple judges score the same items
  - [ ] Run same metric with N different judges (models or prompts)
  - [ ] Fleiss' Kappa for multi-rater agreement
  - [ ] Identify items with lowest agreement for human review
  - [ ] Recommend judge selection based on reliability
- [ ] **Metric Sensitivity Analysis** - Measure score stability across small input perturbations
  - [ ] Perturb inputs (typos, rephrasing) and measure score variance
  - [ ] Flag metrics with high sensitivity to minor input changes
  - [ ] Robustness score per metric
- [ ] **Metric Correlation Pruning** - Auto-suggest removing redundant metrics that track the same signal
  - [ ] Pearson/Spearman correlation matrix across all metrics
  - [ ] Flag pairs with r > 0.95 as candidates for pruning
  - [ ] Recommend minimal metric set preserving signal coverage
- [ ] **Metric Dependencies** - Declare that metric B requires metric A to run first (dependency graph)
  - [ ] Dependency declaration in MetricSpec
  - [ ] Topological sort of metrics before evaluation
  - [ ] Pass metric A results as context to metric B prompt
- [ ] **Conditional Metric Chains** - If metric A fails, automatically run a diagnostic follow-up metric B
  - [ ] Chain definition: "if toxicity_safety fails, run toxicity_type_classifier"
  - [ ] Diagnostic metrics produce detailed failure categorization
  - [ ] Chain results stored alongside primary metric results
- [ ] **Metric Namespacing** - Organize metrics by project/team namespace to avoid collisions
  - [ ] Namespace prefix: "team-safety/toxicity" vs "team-quality/toxicity"
  - [ ] Namespace-scoped metric search in list-metrics
  - [ ] Cross-namespace metric comparison

### LLM Provider Support

- [x] **Gemini** - Full support with auto-instrumentation
- [x] **OpenAI** - Full support with auto-instrumentation
- [x] **Anthropic** - Full support with auto-instrumentation
- [x] **xAI (Grok)** - Full support with auto-instrumentation
- [x] **Ollama** - Local model support (--provider ollama)

### Framework Support

- [x] **LangChain** - Automatic instrumentation
- [x] **LangGraph** - Automatic instrumentation with node tracking
- [x] **Google ADK** - Automatic instrumentation
- [x] **Claude Agent SDK** - Automatic instrumentation

### Storage & Data

- [x] **SQLite storage** - Local-first, no cloud dependencies
- [x] **Prod/test separation** - Separate databases for environments
- [x] **JSONL datasets** - Human-readable, git-friendly format
- [x] **Checkpoint system** - Resume interrupted evaluations

### Testing & Quality

- [x] **Test coverage improvement** - 1,063 tests across 30 test files
  - [x] Analysis engine: trends, reports, core properties, insights
  - [x] Model roundtrips: Span, FunctionCall, DatasetItem, Annotation, SpanMetricLink
  - [x] SQLiteStorage: CRUD, ID resolution, annotations
  - [x] CLI utilities: formatters, validation, config
  - [x] CLI commands: analyze, compare, trend, list-runs, show-run, insights
  - [x] Export formats: markdown, HTML, CSV builders
  - [x] Metrics: HeuristicSuggester, subjective template validation, objective metrics
  - [x] Tracing: instrumentation, streaming, provider instrumentors
- [x] **Realistic test fixtures** - 10+ items, 3 metrics, mixed scores, failure reasons
- [x] **pytest-cov integration** - Coverage reporting via `--cov=evalyn_sdk`
- [x] **Integration test unskip** - Fixed 2 skipped integration tests

*Last updated: 2026-03-25*
