# Code Audit Log

Recurring audit driven by `/loop 15m` (cron `*/15 * * * *`, session-scoped, auto-expires 7d).
Each pass dispatches 4 parallel Explore agents (deadcode, security, perf, extensibility)
and appends to this file. Findings carry a stable `id` so future iterations mark them
`resolved` / `still-present` / `stale` rather than re-listing.

## Status legend

- `[open]` — found, not yet addressed
- `[wip]` — in progress (referenced by a branch / commit)
- `[resolved]` — verified fixed; next audit should re-confirm
- `[stale]` — finding no longer reflects reality (false positive on re-check)
- `[wontfix]` — acknowledged and intentionally kept

Severity: `[crit]` `[high]` `[med]` `[low]`.

---

## Iteration log

| # | Timestamp (local) | Pass type | New | Re-confirmed | Resolved since last |
|---|-------------------|-----------|-----|--------------|---------------------|
| 1 | 2026-05-12 (seed)    | Full sweep         | 33  | -            | -                   |
| 2 | 2026-05-12 01:17 PDT | Diff-style recheck | 1   | 33           | 0                   |
| 3 | 2026-05-12 01:32 PDT | Targeted perf scan (v2 API handlers) | 5   | 3 spot-checked | 0                   |

---

## Findings

### Dead code / unreferenced surface

- `[open] [high] DC-001` — 60+ analysis submodules in `sdk/evalyn_sdk/analysis/` are never imported by any production caller. Only 8 are wired (`core`, `reports`, `trends`, `insights`, `clustering`, `html_report`, `panel`, `insights_dashboard`). Representative offenders (not exhaustive): `analysis_snapshots.py`, `cache_savings.py`, `category_stats.py`, `change_attribution.py`, `changelog.py`, `cohort_analysis.py`, `cold_start.py`, `comparative_heatmap.py`, `confusion_matrix.py`, `context_utilization.py`, `correlation_pruning.py`, `cost_dashboard.py`, `cost_phase.py`, `cross_run_stability.py`, `curve_fitting.py`, `dark_mode.py`, `dashboard_theming.py`, `data_export.py`, `dataset_stats.py`, `decision_tree_viz.py`, `density_heatmap.py`, `failure_taxonomy.py`, `forecast.py`, `graph_topology.py`, `hot_path.py`, `improvement_priority.py`, `inter_rater.py`, `item_cost.py`, `item_difficulty.py`, `jupyter_export.py`. Next-iteration task: enumerate the full list and decide per-module (delete vs wire-up vs move to `research/`).
- `[open] [med] DC-002` — `sdk/evalyn_sdk/analysis/clustering.py:1124` `_build_html_page` — internal helper with no callers in the module or elsewhere.
- `[open] [med] DC-003` — `sdk/evalyn_sdk/analysis/clustering.py:1150` `_get_base_plotly_layout` — orphaned helper.
- `[open] [med] DC-004` — `sdk/evalyn_sdk/analysis/clustering.py:1354` `_generate_fallback_html` — unreachable fallback path.
- `[open] [med] DC-005` — `sdk/evalyn_sdk/analysis/clustering.py:1612` `_generate_failure_fallback_html` — orphaned twin of DC-004.

### Security

- `[open] [low] SEC-001` — `dashboard/evalyn_dashboard/jobs_persistence.py:337` (and 347) — f-string interpolates `column` derived from `kind` parameter into SQL. Safe today because `kind` is enum-constrained to `"stdout"` / `"stderr"`, but static analyzers will flag. Suggested fix: map `kind` → column via a literal dict and look up, no f-string in SQL.

Ruled out on this pass (good baseline — re-confirm next iterations and flag if a NEW occurrence appears):

- No shell-string subprocess invocations; all process spawns pass list argv.
- No dynamic code-evaluation calls on non-constant strings. `importlib.util.spec_from_file_location()` in `_load_callable()` is the intentional user-code loader (CLI only, not web-exposed).
- No use of Python's unsafe binary-deserialization stdlib module on untrusted bytes. YAML calls use the safe-loader variant only.
- No real secrets committed; test fixtures use `sk-test` / `sk-secret` prefixes; no `.env` checked in.
- `/api/files/read` uses `Path.resolve().relative_to(root)` for traversal containment.
- All SQL uses `?` placeholders; `list_recent()` builds WHERE clauses with proper param separation.
- No user-controlled URL fetching (SSRF). All outbound HTTP targets hardcoded provider endpoints.
- No raw-HTML React injection escape hatches in the dashboard frontend; components render text content, not unsanitized markup.
- Credentials stored mode `0600`; `public_view()` excludes `api_key`; audit log does NOT include CLI args.
- Dashboard binds loopback by default; `--unsafe-bind` required for non-local. CSRF middleware on all mutating `/api/*` routes.

### Performance / inefficiency

- `[open] [high] PERF-001` — `sdk/evalyn_sdk/cli/utils/dataset_resolver.py:70` — `.stat().st_mtime` called inside the sort comparator (O(n log n) stat syscalls). Fix: build `[(path, path.stat().st_mtime) for path in candidates]` once, then sort.
- `[open] [high] PERF-002` — `sdk/evalyn_sdk/annotation_delegation.py:299` — nested double loop computing the Gini coefficient is O(n²). Fix: sort once and use the linear formula `2·Σ(i·counts[i]) / (n·Σ counts) - (n+1)/n`.
- `[open] [med] PERF-003` — `sdk/evalyn_sdk/cli/commands/analysis.py:120` — `list(d.glob("*.json"))` used only for truthiness check. Fix: `any(d.glob("*.json"))` short-circuits at first match.
- `[open] [med] PERF-004` — `sdk/evalyn_sdk/cli/commands/analysis.py:143-145` — per-metric glob inside outer loop over metric dirs. Fix: glob once into a dict keyed by metric dir.
- `[open] [med] PERF-005` — `sdk/evalyn_sdk/cli/commands/analysis.py:194` — `list(prompts_dir.glob("*_full.txt"))` for existence. Fix: `next(prompts_dir.glob("*_full.txt"), None)`.
- `[open] [med] PERF-006` — `sdk/evalyn_sdk/calibration/ape.py:309` — `ucb_scores.index(max(ucb_scores))` traverses twice. Fix: `max(range(len(ucb_scores)), key=ucb_scores.__getitem__)`.
- `[open] [med] PERF-007` — `sdk/evalyn_sdk/analysis/graph_topology.py:315` — `path.index(neighbor)` inside a DFS loop. Fix: maintain a `dict[node, index_in_path]` alongside `path`.
- `[open] [med] PERF-008` — `sdk/evalyn_sdk/analysis/core.py:147-148` — `metric_types` dict rebuilt per run inside hot path. Fix: hoist construction above the loop.
- `[open] [low] PERF-009` — `sdk/evalyn_sdk/cli/commands/analysis.py:108-110` — `calibrations_dir.exists()` evaluated twice consecutively. Fix: cache to local.
- `[open] [low] PERF-010` — `sdk/evalyn_sdk/calibration/ape.py:324-326` — `mean_scores` comprehension rebuilt at line 329 identically. Fix: compute once.
- `[open] [low] PERF-011` — `sdk/evalyn_sdk/calibration/data_augmentation.py:132` — `word.index(stripped[0])` after `if stripped[0] in word`. Fix: capture index from the membership test.

Note vs prior memory snapshot: the `find_eval_runs` 484-empty-dir cold-start regression was not re-detected on this pass; treat as already-mitigated unless a future audit re-finds it.

### Extensibility / architecture

- `[open] [high] EXT-001` — `sdk/evalyn_sdk/defaults.py:12-25` — provider/model map hardcoded (gemini, openai, anthropic, ollama). Adding a provider requires editing core. Suggested shape: registry + entry-point discovery, mirroring the existing CLI plugin pattern.
- `[open] [high] EXT-002` — `sdk/evalyn_sdk/metrics/objective.py:61` — `OBJECTIVE_REGISTRY` is a single 4000+ line literal list. No discovery hook. New metrics require a core PR. Suggested shape: `register_objective()` decorator + entry-point group `evalyn.metrics`.
- `[open] [high] EXT-003` — `sdk/evalyn_sdk/metrics/subjective.py` — `JUDGE_TEMPLATES` static; no registry for third-party judges. Mirror EXT-002.
- `[open] [high] EXT-004` — `dashboard/frontend/src/v2/V2App.tsx:119-137` — routes are explicit `<Route>` children (copilot, experiments, datasets, metrics, review, reports). No injection point for plugin pages. Suggested shape: route registry that plugins push into at mount time.
- `[open] [med] EXT-005` — `sdk/evalyn_sdk/cli/main.py:68-121` — `_COMMAND_MODULE_MAP` hardcoded; CLI plugin support exists via entry points but core commands cannot be overridden. Document override precedence or expose merge hook.
- `[open] [med] EXT-006` — `sdk/evalyn_sdk/analysis/insights.py:85` — `REDUNDANT_THRESHOLD=0.7`, `CRITICAL_THRESHOLD=0.15` are module constants. Promote to config / CLI flags.
- `[open] [med] EXT-007` — `sdk/evalyn_sdk/analysis/clustering.py:539-551` — clustering prompt + cluster count hardcoded in `_build_failure_clustering_prompt`. Promote to parameters.
- `[open] [med] EXT-008` — `sdk/evalyn_sdk/analysis/html_report.py:42` — failed-item list truncated to `[:30]`; line ~17000 has a 10000-char body truncation. Make pagination / cap configurable.
- `[open] [med] EXT-009` — `sdk/evalyn_sdk/cli/utils/dataset_resolver.py:42-43` — layout hardcoded: `eval_runs/`, `metrics/`, `dataset.jsonl`. Alternative storage layouts require code edits. Centralize as `paths.py` constants and accept overrides via config.
- `[open] [med] EXT-010` — `sdk/evalyn_sdk/cli/constants.py:28-311` — metric bundles (`chatbot`, `rag-qa`, `code-assistant`, ...) static dict. Same plugin pattern as EXT-002.
- `[open] [med] EXT-011` — `sdk/evalyn_sdk/metrics/factory.py:5-101` — 100+ direct imports of metric functions. After EXT-002 lands, replace with discovery loop.
- `[open] [med] EXT-012` — `sdk/evalyn_sdk/cli/commands/evaluation.py:128` — validates metric names against `OBJECTIVE_REGISTRY` only, so plugin metrics are filtered out. Fix in lockstep with EXT-002.
- `[open] [med] EXT-013` — `sdk/evalyn_sdk/evaluation/runner.py:106-112` — unit builders selected via `get_default_builders()` / `get_builders_for_types()`; no clear hook for plugin builders.
- `[open] [med] EXT-014` — `sdk/evalyn_sdk/models.py:32-47` — `SpanType` is a closed `Literal` union. New span types require dataclass edits; consider `str` with documented well-known values.
- `[open] [med] EXT-015` — `sdk/evalyn_sdk/storage/base.py` — `StorageBackend` protocol exists but `sqlite.py` is the only implementation and CLI does not advertise the protocol as a plugin point.
- `[open] [med] EXT-016` — `sdk/evalyn_sdk/analysis/html_report.py:1-50` — report generator tightly coupled to `RunAnalysis`. No hook for custom chart types or report formats.
- `[open] [med] EXT-017` — `dashboard/evalyn_dashboard/agent.py` — dashboard agent state and command execution tightly coupled; no plugin contract for alternative agent backends.
- `[open] [low] EXT-018` — `sdk/evalyn_sdk/analysis/core.py:99-137` — `eval_runs_dir` derived as `dataset_path / "eval_runs"`. Plumb through config.
- `[open] [low] EXT-019` — `sdk/evalyn_sdk/config_show.py:107` — env-var prefix hardcoded `EVALYN_`. Expose for multi-tenant setups.
- `[open] [low] EXT-020` — `sdk/evalyn_sdk/cli/commands/analysis.py:49-55` — health-score thresholds (`_PROBLEM_METRIC_THRESHOLD=0.2`, `_HEALTH_GOOD=90`, `_HEALTH_MODERATE=70`) hardcoded.
- `[open] [low] EXT-021` — `sdk/evalyn_sdk/__init__.py:137-235` — `__all__` does not re-export `evaluation`, `trace.instrumentation`; third-party code forced into deep imports. Decide what is public API.
- `[open] [low] EXT-022` — `sdk/evalyn_sdk/calibration/__init__.py` — GEPA + APE optimizers hardcoded as the only options; add a registry for custom calibrators.

---

## Iteration 2 delta (2026-05-12 01:17 PDT)

Commits inspected since seed (`b138042a`):

- `9ae576f2` — retro(annotation): perf fix for `_replay_log` quadratic + CLI/dashboard compat layer
- `d2d329aa` — merge: retro /plan-eng-review on /annotate feature

### Re-confirmation summary

- PERF-001..PERF-011, DC-001..DC-005, SEC-001, EXT-001..EXT-022: not touched by the new commits. All **still `[open]`**.
- Specifically PERF-002 (`sdk/evalyn_sdk/annotation_delegation.py:299` Gini O(n²)) is verified still present at line 299 — the `9ae576f2` perf fix is in `dashboard/evalyn_dashboard/api/v2/annotation.py` and targets `_replay_log`, a distinct hot path. PERF-002 was NOT addressed.

### Audit miss (worth noting)

- The seed pass did not flag the `_replay_log` quadratic pattern in `dashboard/evalyn_dashboard/api/v2/annotation.py` — the team's `/plan-eng-review` retro found it independently. Future iterations: extend the perf agent's scope to include `dashboard/evalyn_dashboard/api/v2/*.py` request handlers, with attention to per-request replay/scan patterns. Add a `[low]` self-improvement note rather than a finding ID — the issue is resolved.

### New findings

- `[open] [low] EXT-023` — `sdk/evalyn_sdk/annotation/compat.py:49-69` `detect_shape` — closed 3-shape registry (dashboard, cli_annotation, cli_annotation_item). A fourth on-disk annotation shape (e.g. from a third-party tool that exports to `annotations.jsonl`) requires editing this file. Suggested fix: register-by-decorator pattern with `(predicate, normalizer)` pairs, mirroring the family of EXT-002 / EXT-010 fixes. Small surface, low priority — flagged for visibility, not urgency.

### Re-confirmed clean (security)

- The new `annotation/compat.py` and the modified `dashboard/evalyn_dashboard/api/v2/annotation.py` both pass the SEC baseline: defensive `isinstance` guards in `detect_shape`, no shell/eval/yaml-unsafe/SQL-format, no raw HTML render. No new SEC findings.

---

## Iteration 3 delta (2026-05-12 01:32 PDT)

No commits since iteration 2. This iteration paid down iteration 2's "audit miss" debt by extending the perf scan to `dashboard/evalyn_dashboard/api/v2/*.py` request handlers (12 files). One focused Explore agent, not the 4-parallel sweep.

### Spot-check of prior high-severity items (still `[open]`)

- `PERF-002` — Gini at `sdk/evalyn_sdk/annotation_delegation.py:298-301` still O(n²) double-sum (`abs_diffs = sum(abs(counts[i] - counts[j]) for i in range(n) for j in range(n))`).
- `DC-001` — Sample submodules (`forecast.py`, `dataset_stats.py`) confirmed still have zero importers in production code.
- `EXT-001` — `sdk/evalyn_sdk/defaults.py:12-25` `DEFAULT_MODELS_BY_PROVIDER` still hardcodes `gemini`/`openai`/`anthropic`/`ollama`.

### New findings (paying down iter-2 audit-miss debt)

All in `dashboard/evalyn_dashboard/api/v2/`. The team's `_replay_log` fix in `9ae576f2` showed the *shape* of perf bug to look for — these are the analogous ones in sibling handlers.

- `[open] [high] PERF-012` — `rubrics.py:312-320` `list_rubrics()` calls `_load_saved_rubric(metric_id)` once per metric, and each call does a nested `for root in dataset_roots(): for ds in list_dataset_dirs(root):` walk. With ~30 metrics × ~491 dataset dirs that's ~15k dataset-dir traversals per request, none cached. Fix: hoist the `(root, ds)` enumeration outside the metric loop OR cache `_saved_rubric_path` lookups by metric.
- `[open] [high] PERF-013` — `annotation.py:1010-1031` `smart_queue()` per-request rebuild: for every dataset it `reviews_dir.glob("*.jsonl")`, opens each file, line-reads to count verdicts, and rebuilds `coverage_counter` from scratch — no cache. Fix: cache `coverage_counter` keyed on `(dataset, mtime of reviews_dir)`, invalidate on write.
- `[open] [med] PERF-014` — `annotation.py:484-492` `_find_session()` uses `root.iterdir()` directly instead of the cached `list_dataset_dirs()` helper used elsewhere in the file. Per-request session lookups defeat the mtime cache. Fix: route through `list_dataset_dirs()`.
- `[open] [med] PERF-015` — `annotation.py:501-513` `_list_sessions()` same bypass pattern as PERF-014 for the `/sessions` listing endpoint. Fix: same as PERF-014.
- `[open] [med] PERF-016` — `rubrics.py:294-301` `_saved_rubric_path(metric_id)` does the full nested traversal for a path that doesn't change between requests for the same metric. Fix: memoize on `metric_id` (`functools.lru_cache` with size matching the metric count).

### Confirmed-good patterns worth preserving (regression watch)

- `_shared.py` `list_dataset_dirs()` correctly caches by directory mtime — keep this pattern; the new PERF-014/015 findings are exactly violations of it.
- `load_all_runs()` is whole-list mtime-cached — fine.
- `annotation.py` lines 769-792: the `_meta_has_progress_index` / `_recompute_progress_from_log` migration helpers are the FIX for the prior `_replay_log` quadratic. Confirmed working.

---

## Notes for future passes

- Re-running deadcode every 15 min on a 1590-file tree is wasteful. Future iterations should *narrow*: diff against the last pass's "live module" set and only re-audit modules whose mtime changed.
- The security baseline above is the bigger asset than the single low-severity finding. Treat any NEW occurrence of the ruled-out patterns as a regression worth a `[crit]` flag.
- Extensibility findings cluster around a single design choice: closed registries for providers, metrics, judges, dashboard pages. Resolving EXT-002 (entry-point discovery) likely unlocks EXT-001, EXT-003, EXT-010, EXT-011, EXT-012 in one move.
