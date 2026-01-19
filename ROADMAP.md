# Evalyn Roadmap

This document tracks planned features and completed work. Future roadmap items are listed first, followed by completed features.

---

## Roadmap (Planned Features)

### High Priority

- [ ] **Web Dashboard** - Browser-based UI for viewing traces, datasets, and evaluation results
- [ ] **CI/CD Integration** - GitHub Actions workflow for automated evaluation on PR
- [ ] **Regression Detection** - Automatic alerts when metrics drop below threshold
- [ ] **Multi-model Comparison** - Compare same prompts across different LLM providers
- [ ] **Cost Tracking Dashboard** - Visualize LLM API costs over time by project/version

### Medium Priority

- [ ] **Custom Objective Metrics** - Plugin system for user-defined objective metrics
- [ ] **Metric Weights** - Configurable importance weights for composite scoring
- [ ] **Slack/Discord Notifications** - Alert on evaluation completion or failures
- [ ] **Dataset Versioning** - Track dataset changes over time with diff view
- [ ] **A/B Test Framework** - Statistical significance testing for model comparisons
- [ ] **Prompt Versioning** - Track and compare prompt changes with evaluations

### Low Priority

- [ ] **Team Collaboration** - Multi-user annotation with conflict resolution
- [ ] **Cloud Storage Backend** - Optional S3/GCS storage for large datasets
- [ ] **Evaluation Scheduling** - Cron-based automatic re-evaluation
- [ ] **Custom Report Templates** - User-defined HTML report layouts
- [ ] **API Server Mode** - REST API for programmatic access

### SDK Enhancements

- [ ] **More LLM Provider Instrumentors** - Cohere, Mistral, AWS Bedrock
- [ ] **Framework Instrumentors** - CrewAI, AutoGen, DSPy
- [ ] **Streaming Support** - Capture streaming LLM responses
- [ ] **Multi-turn Conversation Metrics** - Specialized metrics for chat history

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
  - [x] LLM method - Analyze disagreements, suggest improvements
  - [x] GEPA method - Evolutionary prompt optimization
- [x] **evalyn list-calibrations** - List calibration records
- [x] **Alignment metrics** - Accuracy, precision, recall, F1, Cohen's Kappa
- [x] **Validation split** - Test calibration on held-out samples

### Simulation (Synthetic Data)

- [x] **evalyn simulate** - Generate synthetic test data
  - [x] similar mode - Variations of existing queries
  - [x] outlier mode - Edge cases and unusual inputs
- [x] **Temperature control** - Separate temps for similar/outlier
- [x] **Seed sampling** - Control number of seed examples

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

### LLM Provider Support

- [x] **Gemini** - Full support with auto-instrumentation
- [x] **OpenAI** - Full support with auto-instrumentation
- [x] **Anthropic** - Full support with auto-instrumentation
- [x] **Ollama** - Local model support (--llm-mode local)

### Framework Support

- [x] **LangChain** - Automatic instrumentation
- [x] **LangGraph** - Automatic instrumentation with node tracking
- [x] **Google ADK** - Automatic instrumentation

### Storage & Data

- [x] **SQLite storage** - Local-first, no cloud dependencies
- [x] **Prod/test separation** - Separate databases for environments
- [x] **JSONL datasets** - Human-readable, git-friendly format
- [x] **Checkpoint system** - Resume interrupted evaluations

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.1.0 | 2025-12 | Initial release with core tracing and evaluation |
| v0.2.0 | 2026-01 | Added 130+ metrics, 17 bundles, calibration, simulation |
| v0.3.0 | 2026-01 | Added clustering, checkpoint/resume, --db flag |

---

## Contributing

We welcome contributions! See [README.md](README.md#contribution) for how to add new metrics.

**Priority areas for contribution:**
1. New domain-specific metrics (healthcare, finance, legal)
2. Additional LLM provider instrumentors
3. Framework integrations (CrewAI, AutoGen, DSPy)
4. Documentation improvements

---

*Last updated: 2026-01-18*
