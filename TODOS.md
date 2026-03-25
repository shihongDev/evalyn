# TODOS

Technical debt and improvements identified during architecture review (2026-03-23).

---

## High Priority

### Write tests for 11 untested modules
- **What:** Add dedicated test files for: parsing.py, evaluation/execution.py, evaluation/batch/evaluator.py, judges/llm_judge.py, judges/confidence/* (3 files), simulation/simulator.py, calibration/factory.py, calibration/engine.py, datasets.py, cli/utils/pipeline_steps.py.
- **Why:** These orchestrator modules are currently covered only indirectly through CLI integration tests. A failure in JSON parsing surfaces as "run-eval failed" in the CLI test - debugging is 5-10x slower. The test suite has 236 tests across 18.5K LOC but is concentrated on instrumentation (8.2K LOC) while the evaluation core has gaps.
- **Pros:** Faster debugging, better regression detection, confidence in refactoring.
- **Cons:** ~300 tests to write (though CC can do this in ~30min).
- **Context:** Most critical gaps: parsing.py (used everywhere for LLM response extraction), execution.py (parallel strategy), confidence/* (3 estimation methods). The parsing module likely accounts for a significant chunk of production bugs.
- **Depends on:** Nothing.


---

## Medium Priority

### Extract shared CLI storage helpers
- **What:** Create `cli/utils/storage_helpers.py` with `get_storage(args)` and `load_run(args)`. Wire all 12+ `SQLiteStorage()` instantiations through shared helpers.
- **Why:** SQLiteStorage is instantiated independently in analysis.py, dashboard.py, export.py, infrastructure.py, insights.py, quickstart.py, traces.py, and pipeline_steps.py. The run-loading pattern (resolve from --run ID or --dataset path) is reimplemented in 3 places. When the storage API changes, every callsite must be updated.
- **Pros:** DRY, single point of change for storage initialization, consistent --db flag handling.
- **Cons:** Minor refactor touching many files (but each change is mechanical).
- **Context:** traces.py already has a `_storage()` helper that other commands don't use. Standardize on one approach.
- **Depends on:** Nothing.

### Remove deprecated DatasetItem fields
- **What:** Drop `inputs` and `expected` fields from DatasetItem. Keep only `input` and `output`. Update all internal references.
- **Why:** The bidirectional sync in `__post_init__` has edge cases (empty dict `{}` is falsy, so `if self.inputs and not self.input` fails). Dual fields confuse contributors - some code uses `item.input`, some uses `item.inputs`. Pre-1.0 with no external users is the right time to clean this up.
- **Pros:** Simpler model, no sync bugs, clearer API.
- **Cons:** Breaks any local scripts using the old field names. Need to update `from_payload()` to still accept old-format JSONL files (read compat without write compat).
- **Context:** `as_dict()` already only serializes new fields. The cleanup is mostly mechanical: grep for `.inputs` and `.expected` across the codebase.
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
