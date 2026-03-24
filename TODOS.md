# TODOS

Technical debt and improvements identified during architecture review (2026-03-23).

---

## High Priority

### Restructure __init__.py to lazy imports
- **What:** Move calibration, simulation, annotation behind `__getattr__` so `import evalyn_sdk` only loads core (models, eval, trace).
- **Why:** Current eager loading pulls in all 9 calibration optimizers, all metric templates, simulation, annotation, storage, tracing, and the full evaluation engine on any import. CLI startup is slower than needed, and optional dependencies would fail the import even if users only want tracing.
- **Pros:** Faster import, smaller memory footprint, optional deps don't block core functionality.
- **Cons:** Slightly less discoverable (IDE autocomplete may not show lazy-loaded names without type stubs).
- **Context:** The SDK has ~80 exported names in `__all__`. Most users will only need `@eval`, `EvalRunner`, and a few metrics. Pre-1.0 is the right time to restructure.
- **Depends on:** Nothing.

### Write tests for 11 untested modules
- **What:** Add dedicated test files for: parsing.py, evaluation/execution.py, evaluation/batch/evaluator.py, judges/llm_judge.py, judges/confidence/* (3 files), simulation/simulator.py, calibration/factory.py, calibration/engine.py, datasets.py, cli/utils/pipeline_steps.py.
- **Why:** These orchestrator modules are currently covered only indirectly through CLI integration tests. A failure in JSON parsing surfaces as "run-eval failed" in the CLI test - debugging is 5-10x slower. The test suite has 236 tests across 18.5K LOC but is concentrated on instrumentation (8.2K LOC) while the evaluation core has gaps.
- **Pros:** Faster debugging, better regression detection, confidence in refactoring.
- **Cons:** ~300 tests to write (though CC can do this in ~30min).
- **Context:** Most critical gaps: parsing.py (used everywhere for LLM response extraction), execution.py (parallel strategy), confidence/* (3 estimation methods). The parsing module likely accounts for a significant chunk of production bugs.
- **Depends on:** Nothing.

### Add retry with exponential backoff to HTTP client
- **What:** Add 3 retries with jitter (1s, 2s, 4s) to `api_client._http_post()` for 429/503/5xx responses.
- **Why:** Eval runs make hundreds of LLM calls. A single transient 429 rate-limit or 503 from Gemini crashes the entire eval run. Checkpoint/resume helps recover but doesn't prevent the crash.
- **Pros:** Eval runs survive transient API failures without manual intervention.
- **Cons:** Adds ~30 lines to api_client.py. Needs a max timeout to avoid hanging forever.
- **Context:** The urllib-based HTTP client is deliberately dependency-free (core deps are just opentelemetry-sdk + tqdm). The retry logic should use stdlib only (time.sleep + random.uniform for jitter).
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

### Enable SQLite WAL mode
- **What:** Add `self.conn.execute('PRAGMA journal_mode=WAL')` to `SQLiteStorage.__init__()`.
- **Why:** Eval runs with 500+ items do 500 individual commits (`self.conn.commit()` per store_call). WAL mode allows concurrent reads during writes and improves write performance 2-5x, especially on WSL2 or networked filesystems.
- **Pros:** One-line change, significant performance improvement on slow filesystems.
- **Cons:** WAL creates additional -wal and -shm files alongside the DB. Some very old SQLite versions don't support it (but Python 3.10+ bundles SQLite 3.35+).
- **Context:** The project root is on /mnt/c (WSL2), which has known slow I/O. WAL mode is the standard recommendation for SQLite write-heavy workloads.
- **Depends on:** Nothing.

---

## Completed

### Commit storage/migrations.py to git
- **Completed:** fix/test-parallel-and-migrations (2026-03-24)

### Consolidate run-loading into shared helper (partial: Extract shared CLI storage helpers)
- **Completed:** fix/test-parallel-and-migrations (2026-03-24) - run-loading pattern consolidated in `load_eval_run_for_command`. Storage instantiation helpers remain TODO.
