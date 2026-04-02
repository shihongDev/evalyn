# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-29

### Added
- **Sampling strategies** (24 modules) - adversarial, stratified, drift, cost-aware, curriculum, time-weighted, coverage, balanced, importance, bootstrap, boundary, disagreement, similarity, coreset, reservoir, progressive, error-pattern, metadata-conditional, novelty, locale, pipeline composition, reproducibility, impact analysis
- **Simulation** (14 modules) - multi-turn, adversarial, regression, domain-transfer, conditional, structured, parallel, constraint, persona, reference, tool-schema, validation, templates, eval-loop
- **Analysis and reporting** (10 modules) - clustering report, comparison overlay, dashboard export, coverage report, progress dashboard, OpenAI evals export, parquet export, trace summary, NL summary, web dashboard
- **Annotation and calibration** (12 modules) - rubric library, rubric packs, rubric testing, rubric i18n, annotation sessions, annotation UX, annotation delegation, annotator agreement, conflict resolution, guidelines generator, pre-annotation, CAPO optimizer
- **CLI tools** (18 modules) - shell completion, watch mode, profile, config validation, doctor diagnostics, eval diff, JSON output, batch scripting, config show, time tracking, metric namespacing, garbage collection, plugins, aliases, command history, command chaining, output pagination, color themes
- **Security and governance** (8 modules) - PII safety check, trace redaction, compliance report, data governance, execution audit, audit trail, secrets backend, key rotation
- **Infrastructure** (29 modules) - evol-instruct, cascade/judge routing, persona hub, provider diversity, cost estimation, reproducibility seed, feedback injection, seed clustering, difficulty grading, quality score, diversity metrics, budget optimizer, IRT benchmarks, benchbuilder, embedding selection, experiment tracker, sandbox eval, embedding index, semantic search, knowledge-graph test gen, behavior test gen, curation suggestions, adaptive metrics, metric debug, binary packaging, docker config, TUI mode, playground, web dashboard

### Stats
- 579 Python modules (up from ~75 in 0.1.0)
- 242 files changed, 68K+ lines added
- ROADMAP.md: all 559 items implemented (100% complete)

## [0.1.0] - 2026-03-21

### Added
- Initial public release on PyPI
- Auto-instrumentation for OpenAI, Anthropic, Google Gemini, xAI, LangChain, LangGraph, Google ADK, Claude Agent SDK, CrewAI, AutoGen, DSPy, Haystack, LlamaIndex, Semantic Kernel
- 136 built-in evaluation metrics (76 objective, 60 LLM judges)
- Tracing with SQLite storage (fully local, no cloud dependencies)
- CLI with 32 commands covering the full evaluation pipeline
- `one-click` command for running the entire pipeline in a single step
- Calibration system with 8 optimizers: Basic, APE, GEPA, OPRO, EvoPrompt, TextGrad, MIPROv2, PromptBreeder
- Human-in-the-loop annotation workflow
- Insights engine with deterministic analysis and LLM expert panel
- HTML dashboard with interactive Chart.js visualizations
- Failure and misalignment clustering
- Synthetic data simulation
- Evaluation trend tracking and run comparison
