# TODOS

Technical debt and improvements identified during architecture review (2026-03-23).
Updated 2026-03-29: ROADMAP 100% complete (559/559 items, 2151 sub-items). Focus shifts to tech debt and stabilization.

---

## High Priority

### Write tests for new modules
- **What:** The SDK grew from ~75 to 579 modules. The original 11 untested modules (parsing.py, execution.py, batch/evaluator.py, llm_judge.py, confidence/*, simulator.py, calibration/factory.py, calibration/engine.py, datasets.py, pipeline_steps.py) still need dedicated tests. Additionally, the ~500 new modules (sampling, simulation, security, CLI tools) have no unit tests yet.
- **Why:** New modules are covered only through commit-time smoke tests. Regressions in sampling strategies or simulation generators would go undetected until user-facing failures.
- **Depends on:** Nothing.

---

## Medium Priority

### Co-pilot tour effectiveness telemetry
- **What:** Log which co-pilot tours fired, were dismissed, or completed; attribute downstream feature adoption to tour exposure. Wire into existing analytics path (or add minimal local events store).
- **Why:** Without measurement we cannot validate that UI-guidance tours actually help users learn the dashboard. Per CLAUDE.md epistemology rule ("Assumptions are the enemy. Never guess - benchmark instead of estimating"), shipping more tours without this is shipping by hunch. Should land before a second wave of tours is authored.
- **Depends on:** Co-pilot UI guidance core (CEO plan 2026-05-03).

### Co-pilot stuck-detection trigger
- **What:** Idle timer (>90s with no clicks) on content-rich routes triggers a copilot proactive offer ("anything I can help with?"). Respects the global UI-guidance toggle.
- **Why:** This is the AI-only differentiator no static tour library can replicate. Defer until the core tour engine is stable so we can iterate on heuristics without destabilizing the foundation.
- **Depends on:** Co-pilot UI guidance core, Settings 3-state migration (so users have "first-time-only" semantics to opt out of stuck-detection without disabling all guidance).

### Co-pilot guidance Settings: migrate boolean to 3-state
- **What:** Replace boolean "Co-pilot UI guidance on/off" toggle with 3-state "Always / First-time-only / Off". First-time-only becomes the default for new installs.
- **Why:** "Default on" as a binary annoys returning users. "First-time-only" is the actually-correct default and addresses the original feature ask cleanly. Migration cost grows with usage, so address before tour count grows.
- **Depends on:** Co-pilot UI guidance core landed.

### Glossary-on-hover via co-pilot
- **What:** Hover unfamiliar terms (rubric, BLEU, judge, calibration, cluster) -> small popover with copilot-curated explanation. Distinct from the existing static `Glossary.tsx`; uses the copilot for personalized phrasing.
- **Why:** Sticky differentiation; complements the sequenced-tour mechanism with always-available term-level help. Lower priority than tours themselves.
- **Depends on:** Nothing strict; can ship independently of the tour engine.

### Extract shared CLI storage helpers
- **What:** Create `cli/utils/storage_helpers.py` with `get_storage(args)` and `load_run(args)`. Wire all 12+ `SQLiteStorage()` instantiations through shared helpers.
- **Why:** SQLiteStorage is instantiated independently in analysis.py, dashboard.py, export.py, infrastructure.py, insights.py, quickstart.py, traces.py, and pipeline_steps.py. The run-loading pattern (resolve from --run ID or --dataset path) is reimplemented in 3 places.
- **Depends on:** Nothing.

### Remove deprecated DatasetItem fields
- **What:** Drop `inputs` and `expected` fields from DatasetItem. Keep only `input` and `output`. Update all internal references.
- **Why:** The property aliases (`inputs`/`expected`) add indirection. Dual fields confuse contributors. Pre-1.0 with no external users is the right time.
- **Depends on:** Nothing.

### Consolidate sampling module organization
- **What:** 24 sampling modules live as top-level files in evalyn_sdk/. Consider moving to a `sampling/` subpackage.
- **Why:** Top-level directory has 200+ files. Grouping by domain (sampling/, simulation/, security/) would improve navigability.
- **Depends on:** Nothing.

---

## Completed

### Commit storage/migrations.py to git
- **Completed:** fix/test-parallel-and-migrations (2026-03-24)

### Consolidate run-loading into shared helper (partial: Extract shared CLI storage helpers)
- **Completed:** fix/test-parallel-and-migrations (2026-03-24) - run-loading pattern consolidated in `load_eval_run_for_command`. Storage instantiation helpers remain TODO.

### Enable SQLite WAL mode + thread-local connections + relational metric results
- **Completed:** feat/efficiency-at-scale (2026-03-24) - WAL mode, busy_timeout=5000, thread-local connections for worker threads, relational metric_results_rows table with batch inserts and performance indexes. Old JSON blob data still loads via fallback.

### Add stream_dataset() for memory-efficient dataset loading
- **Completed:** feat/efficiency-at-scale (2026-03-24) - Generator-based JSONL streaming with malformed line skipping. load_dataset() now wraps stream_dataset().

### Restructure __init__.py to lazy imports
- **Completed:** feat/efficiency-at-scale (2026-03-24) - Calibration, simulation, annotation, judges, metrics, suggesters moved behind __getattr__. Core (models, eval, trace, datasets) stays eager. Broke circular import in metrics/factory.py.

### Add retry with exponential backoff to HTTP client
- **Completed:** feat/efficiency-at-scale (2026-03-24) - 3 retries with jitter on 429/5xx, 30s total timeout, stdlib-only.
