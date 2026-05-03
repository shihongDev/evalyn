# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CLI rich output** - unified visual system across all 34 CLI commands using box-drawing primitives in `cli/utils/rich.py` (`banner`, `section`, `table`, `kv`, `footer`, `progress_bar`, semantic icons). All list, detail, analysis, status, and action commands migrated. Falls back to plain text when stdout is not a TTY or `EVALYN_NO_COLOR` is set; `--format json` and `--compact` modes unchanged. See `docs/superpowers/specs/2026-04-07-cli-rich-output-design.md`.
- **Hints system overhaul** - new `HintCollector` aggregates multiple hints per command for organized display; suggested commands are enriched with their key options.
- **Entry-point plugin discovery** for the `evalyn` CLI. At startup, `evalyn_sdk.cli.main` reads the `evalyn.commands` entry-point group via `importlib.metadata` and merges discovered modules into the lazy command map. Third-party packages can register subcommands without modifying core. Used by the new `evalyn-dashboard` package to register the `dashboard` subcommand.

### Changed
- CLI startup sets `EVALYN_OTEL=off` by default for read-only commands, removing a ~45ms import cost from the otel stack.
- 23 medium-severity and 4 high-severity code issues resolved across the CLI and storage layers.
- **`evalyn dashboard` renamed to `evalyn report`.** The static HTML insights report previously exposed as `evalyn dashboard` is now `evalyn report`. The implementation moved from `cli/commands/dashboard.py` to `cli/commands/report.py` (function renamed `cmd_dashboard` -> `cmd_report`).

### Deprecated
- **`evalyn dashboard` (static report alias).** When the `evalyn-dashboard` plugin is **not** installed, the `dashboard` subcommand prints a stderr deprecation warning and forwards to `cmd_report`. When `evalyn-dashboard` **is** installed, the entry-point plugin takes precedence and `evalyn dashboard` launches the new localhost IDE. The deprecation alias will be removed in evalyn v3.0. Migrate to either `evalyn report` (static HTML) or `pip install evalyn-dashboard` (interactive IDE).

### Fixed
- `quickstart --run` preserves Windows backslash paths.
- `show-projects` restores the Version column.
- Calibration optimizer args preserve explicit zero values instead of dropping them.
- `annotation-stats` uses the newest stored annotation per target, not an arbitrary one.
- `get_calls_batch` chunks query parameters to stay under SQLite's parameter limit.
- `list-calls --function` escapes SQL LIKE wildcards in the filter argument.
- Clustering and trend hint output no longer leaks into JSON mode or uses stale args; `compare` now shows a hint when results are equal.

## [0.2.0] - 2026-03-29

### Added
- **Sampling strategies** (24 modules) - adversarial, stratified, drift, cost-aware, curriculum, time-weighted, coverage, balanced, importance, bootstrap, boundary, disagreement, similarity, coreset, reservoir, progressive, error-pattern, metadata-conditional, novelty, locale, pipeline composition, reproducibility, impact analysis
- **Simulation** (14 modules) - multi-turn, adversarial, regression, domain-transfer, conditional, structured, parallel, constraint, persona, reference, tool-schema, validation, templates, eval-loop
- **Analysis and reporting** (10 modules) - clustering report, comparison overlay, dashboard export, coverage report, progress dashboard, OpenAI evals export, parquet export, trace summary, NL summary, web dashboard
- **Annotation and calibration** (12 modules) - rubric library, rubric packs, rubric testing, rubric i18n, annotation sessions, annotation UX, annotation delegation, annotator agreement, conflict resolution, guidelines generator, pre-annotation, CAPO optimizer (note: CAPO lives at package root as `capo_optimizer.py`, not in `calibration/`)
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
- 133 built-in evaluation metrics (73 objective, 60 LLM judges)
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
