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
| 4 | 2026-05-12 01:47 PDT | Frontend perf+ext audit (dashboard/frontend/src) | 10  | EXT-004 expanded | 0                   |
| 5 | 2026-05-12 02:05 PDT | DC-001 exhaustive enumeration (transitive closure) | 0  | DC-001 made exact | 0                   |
| 6 | 2026-05-12 02:17 PDT | Dashboard backend non-v2 (sec+perf+ext) **[1st crit found]** | 7  | PERF-012/013 still open | 0                   |
| 7 | 2026-05-12 02:32 PDT | example_agents/ audit (24 files, 3 demos) | 2  | SEC-002 still open (+15 min) | 0                   |

---

## Findings

### Dead code / unreferenced surface

- `[open] [high] DC-001` — **69 of 77 modules** in `sdk/evalyn_sdk/analysis/` are orphan: not referenced from `__init__.py`'s `_LAZY_IMPORTS` map, not imported anywhere outside the package, and not reached transitively from any live module. **Total dead lines: 15,964.** Detected by parsing the lazy-import map + a static `git ls-files` import scan + transitive closure (see Iteration 5 delta below for the full list and the reproducer). The 8 live modules are `core`, `clustering`, `html_report`, `insights`, `insights_dashboard`, `panel`, `reports`, `trends` — and they do NOT import any orphan module, so the orphan set can be deleted/moved in one operation without code changes elsewhere. Next-step decision: delete vs move to `research/` (per-module judgment — some, like `confusion_matrix.py`, `cohort_analysis.py`, `forecast.py`, are large enough features that they may deserve wiring rather than deletion).
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
- `[open] [high] EXT-004` — `dashboard/frontend/src/v2/V2App.tsx:119-137` AND `dashboard/frontend/src/v2/routes/NotFound.tsx:18-29` — route registry hardcoded as explicit `<Route>` children in V2App AND duplicated as a path list in NotFound's 404-fallback logic. No injection point for plugin pages, and the duplication means adding a route requires two synchronized edits (merge-conflict risk). Suggested shape: single `const ROUTES = [{path, element, preload}, ...]` module that both V2App and NotFound import.
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

## Iteration 4 delta (2026-05-12 01:47 PDT)

No commits since iteration 3. Spent this iteration extending coverage to the **dashboard frontend** (`dashboard/frontend/src/`, ~110 .ts/.tsx files), which the seed only audited at a single spot (EXT-004). One focused Explore agent, perf + extensibility only (no SEC / deadcode this pass).

Scope scanned: `v2/routes/*`, `v2/api/*`, `v2/hooks/*`, `V2App.tsx`, `AppShell.tsx`, `CommandPalette.tsx`, `CliRunner.tsx`, `RecentJobsDrawer.tsx`. Excluded `__tests__/`, `*.test.ts(x)`.

### Updates to existing findings

- `EXT-004` expanded to include `NotFound.tsx:18-29` — the route list is *duplicated* across V2App and NotFound. The fix is unchanged (single ROUTES module), but the duplication adds merge-conflict risk on top of the plugin-injection block.

### New findings — performance

- `[open] [high] PERF-017` — `dashboard/frontend/src/v2/routes/AnnotateSession.tsx:4091` — evidence list rendered with `key={index}` on a reorderable/filterable collection. Reordering and filter changes break cursor/focus tracking and waste DOM diff work. Fix: use stable IDs (`key={ev.snippet}` or a hash).
- `[open] [med] PERF-018` — `AnnotateSession.tsx:237-271` — diff ops (`eq`/`del`/`ins` spans) rendered with index keys. Fragment reordering breaks highlight/animation state. Fix: stable hash of `(op, text, start)`.
- `[open] [med] PERF-019` — `RunDetail.tsx` `ItemsTab` / `ItemsCompareTab` sub-components have no `React.memo` wrapper despite deep trees and frequent parent re-renders on tab switch. Inline handlers (`jumpToFailedItems`, `handleRerun`) compound the problem. Fix: `React.memo()` on both subs + `useCallback` on the parent's handlers.
- `[open] [med] PERF-020` — `RunDetail.tsx:141-149` — `runs.sort()` and `passes.filter().sort()` executed in the render body rather than memoized. Fix: `useMemo(() => [...runs].sort(...), [runs])`.

### New findings — extensibility

- `[open] [med] EXT-024` — `AnnotateSession.tsx:132-135, 4100, 2530` — hardcoded UI-state colors not in the design-token object `E` (e.g. `#fff8c8` evidence highlight, `#ffe89c` evidence flash, `#fcefe2` filter button, `#fff0d5` flash state). Blocks dark-mode + theme overrides. Fix: add `evidenceHighlight`, `evidenceFlash`, `filterPill` etc. to `E`.
- `[open] [med] EXT-025` — `AnnotateSession.tsx:245, 263` — diff highlight colors hardcoded (`#e8f5e9` pass-bg, `#fde2e2` fail-bg). Fix: tokens `diffInsertBg`, `diffDeleteBg`.
- `[open] [med] EXT-026` — `dashboard/frontend/src/v2/routes/Settings.tsx:42-46` — `KNOWN_PROVIDERS` hardcoded in the frontend. Coupled to backend `EXT-001` (provider map in `defaults.py`). Single-source fix: serve providers from `/api/settings/providers` and consume in both places.
- `[open] [med] EXT-027` — `dashboard/frontend/src/v2/CommandPalette.tsx:479` — `const order: EntryKind[] = ['command', 'run', 'dataset', 'rubric']` literal bounds the entry-kind extension surface. Plugin entry types can't influence ordering without editing core. Fix: derive from a registered `KIND_ORDER` table.
- `[open] [low] EXT-028` — `dashboard/frontend/src/v2/routes/Commands.tsx` — shell colors hardcoded (`SHELL_BG #2c281f`, `SHELL_TEXT #f6e4d2`, `SHELL_FLAG #a8caa8`). Fix: move into tokens or a shell-theme sub-object.
- `[open] [low] EXT-029` — `dashboard/frontend/src/v2/routes/CoPilotThread.tsx:682, 761` — shadow / `rgba()` overlay values inlined. Fix: centralize as `shadowMd`, `shadowLg`, `overlayDim` in tokens.

### Notes on this pass

- 10 new findings, all in the frontend. Skewed toward extensibility (6) vs perf (4) — that matches the project's plug-in-your-own-judge thesis: the frontend needs to be as extensible as the SDK.
- Performance findings here are **measurable but not user-visible lag** on typical dataset sizes (~100s of items). They'll bite when datasets grow or when a plugin renders thousands of evidence rows. Treat as `[med]` debt, not paging incidents.
- EXT-026 cross-references EXT-001 (backend providers map). Fixing one without the other leaves drift.

---

## Iteration 5 delta (2026-05-12 02:05 PDT)

No commits since iteration 4. Spent this iteration making DC-001 **exact**: replaced the seed's "60+ representative offenders" estimate with a rigorous enumeration via `__init__.py` lazy-import map parsing + static import scan + transitive closure.

### Method (reproducible)

1. Parse `sdk/evalyn_sdk/analysis/__init__.py` for `_LAZY_IMPORTS` entries → 8 lazy-wired modules.
2. `git ls-files '*.py'`, exclude `sdk/evalyn_sdk/analysis/*` and `tests/*` → 588 candidate caller files.
3. Static-pattern search for `from evalyn_sdk.analysis.{mod}` and `from .analysis.{mod}` → 2 statically-wired (`clustering`, `core`).
4. Union (lazy ∪ static) = 8 wired entries.
5. Compute transitive closure inside `analysis/` via `from .{mod}` and `from evalyn_sdk.analysis.{mod}` regex on each live module's source → closure adds 0 (the live 8 are isolated from each other AND from the orphan set).
6. Orphan = all_modules - live = **69 modules**.

### Correction to seed audit

- Seed said "60+ orphans, only 8 wired" — direction right, count off. Exact figures:
  - **77** total `.py` modules in `analysis/` (seed said 75)
  - **8** live (matches seed)
  - **69** orphan (seed said "60+")
  - **15,964** total dead lines across orphans
- More importantly: seed's "8 wired" was sourced from naming-pattern intuition. This pass confirmed rigorously via `__init__.py` parsing — the lazy-import map is what makes them reachable. A naive `grep` for direct submodule imports finds only 2 of them.

### Complete orphan inventory (sorted by line count, descending)

For per-module deletion decisions. Numbers are SLOC including blanks.

| Lines | Module |
|------:|--------|
|   399 | root_cause.py |
|   361 | what_if_simulator.py |
|   351 | comparative_heatmap.py |
|   350 | significance_testing.py |
|   336 | stats.py |
|   326 | graph_topology.py |
|   326 | metric_budget.py |
|   325 | forecast.py |
|   317 | failure_taxonomy.py |
|   314 | cohort_analysis.py |
|   299 | dashboard_theming.py |
|   296 | change_attribution.py |
|   292 | pdf_export.py |
|   285 | run_quality_score.py |
|   284 | trend_anomaly.py |
|   278 | confusion_matrix.py |
|   276 | metric_contribution.py |
|   273 | report_diff.py |
|   269 | decision_tree_viz.py |
|   268 | span_dependency.py |
|   260 | multimodel_comparison.py |
|   260 | time_to_fix.py |
|   253 | multi_project.py |
|   253 | trace_template.py |
|   252 | node_attribution.py |
|   250 | report_templates.py |
|   248 | cross_run_stability.py |
|   248 | regression_bisection.py |
|   246 | improvement_priority.py |
|   243 | cost_dashboard.py |
|   243 | language_detect.py |
|   240 | dataset_stats.py |
|   230 | multi_project.py |
|   229 | comparison_template.py |
|   228 | jupyter_export.py |
|   224 | dark_mode.py |
|   223 | metric_interaction.py |
|   220 | analysis_snapshots.py |
|   218 | worst_case_items.py |
|   214 | curve_fitting.py |
|   213 | correlation_pruning.py |
|   212 | item_difficulty.py |
|   210 | time_series.py |
|   207 | density_heatmap.py |
|   204 | subagent_cost.py |
|   200 | context_utilization.py |
|   199 | metric_volatility.py |
|   197 | cache_savings.py |
|   190 | inter_rater.py |
|   185 | trace_complexity.py |
|   184 | unit_reporting.py |
|   179 | output_diff.py |
|   171 | cold_start.py |
|   171 | weighting.py |
|   169 | sensitivity.py |
|   167 | changelog.py |
|   166 | normalization.py |
|   159 | metric_benchmark.py |
|   156 | hot_path.py |
|   154 | what_if.py |
|   149 | cost_phase.py |
|   149 | span_distribution.py |
|   141 | category_stats.py |
|   141 | runtime_estimation.py |
|   135 | item_cost.py |
|   134 | quality_score.py |
|   123 | data_export.py |
|   121 | versioning.py |

### Cleanup triage suggestions

- **Delete-now candidates** (small, narrowly scoped, no obvious feature): `cache_savings.py`, `cost_phase.py`, `data_export.py`, `versioning.py`, `quality_score.py`, `item_cost.py`, `hot_path.py`, `runtime_estimation.py`, `category_stats.py` — under 200 lines, unlikely to deserve wiring as standalone features.
- **Move-to-research candidates** (substantial unfinished features): `root_cause.py` (399), `what_if_simulator.py` (361), `forecast.py` (325), `metric_budget.py` (326), `failure_taxonomy.py` (317), `cohort_analysis.py` (314), `confusion_matrix.py` (278), `decision_tree_viz.py` (269), `significance_testing.py` (350) — large enough that they may represent abandoned experiments worth preserving for later.
- **Decision-needed**: `stats.py` (336) — generic name suggests it may be a forgotten helper module other code SHOULD be using; verify what's in it before deleting.

### Confirmed-good pattern worth noting

- The lazy-import gate in `analysis/__init__.py` is good engineering — it prevents `from evalyn_sdk.analysis import ...` from triggering imports of heavyweight modules (clustering, panel). Preserve this pattern when cleaning up: deleted modules should be removed from `_LAZY_IMPORTS` first to avoid `AttributeError` at import time.

---

## Iteration 6 delta (2026-05-12 02:17 PDT)

No commits since iteration 5. Extended coverage to the dashboard backend **outside `api/v2/`** (8 top-level + ~10 non-v2 `api/*.py` handlers). One Explore agent, three dimensions (sec + perf + ext). This pass surfaced the **first `[crit]` finding** across all 6 iterations.

### CRITICAL — must fix before next deploy

- `[open] [crit] SEC-002` — `dashboard/evalyn_dashboard/api/promote.py:365-374` — the audit-log `logger.info(...)` references bare names `run_id` and `row_hashes`, but the function scope only binds `req.run_id`, `req.row_hashes`, and `safe_run_id = _safe_run_id(req.run_id)` at line 266. Result: every successful promote raises `NameError: name 'run_id' is not defined` at the end of `promote_run_failures`, which FastAPI surfaces as a 500. The dataset is written to disk first, so the user sees a 500 but the side effect persisted; a retry hits the already-created directory and is handled by the `shutil.rmtree(target_dir, ignore_errors=True)` cleanup path — meaning retries DESTROY the just-promoted dataset. Audit log entries are also lost. Verified by reading function scope and confirming no module-level binding. **Fix: replace `run_id` -> `safe_run_id`, `row_hashes` -> `req.row_hashes` in the logger args; add a regression test that calls the endpoint end-to-end.**

This is exactly the class of bug the recurring audit is designed to surface — it survived the seed and 5 subsequent iterations because it lives in a single non-v2 handler the prior passes hadn't scanned. Justifies the loop.

### New findings — security

- `[open] [high] SEC-003` — `dashboard/evalyn_dashboard/api/agent_ws.py` AND `api/jobs_ws.py` — WebSocket endpoints (`/ws/agent/{thread_id}`, `/ws/jobs/{job_id}`) accept connections with NO session-bound auth. Security model is "the IDs are opaque hex/uuid, plus loopback binding by default." If `--unsafe-bind` is ever used (or on a shared dev machine), an attacker who can enumerate `/api/jobs/recent` can subscribe to any job's stream. Fix: require a short-lived signed token (HMAC over the ID + a server secret + expiry) as a query-string param, validate at handshake.
- `[open] [med] SEC-004` — `dashboard/evalyn_dashboard/credentials.py` — credentials are written to `~/.evalyn/credentials.json` and `chmod 0600` is applied AFTER the write. There's a brief window where the file exists with default umask permissions. Race window is small but real. Fix: open the file with `os.open(path, O_WRONLY|O_CREAT|O_EXCL, 0o600)` so the restrictive mode is applied at create time, then write; or write to a tempfile with mode 0600 and rename.

### New findings — performance

- `[open] [med] PERF-021` — `dashboard/evalyn_dashboard/jobs.py:458-461` — event-stream replay computes `replay_events = [evt for evt in job.events if cursor < evt["event_id"] <= snapshot_id]` on every WebSocket subscribe. Long-running jobs with 10k+ events pay an O(N) scan per connection. Fix: keep `job.events` as a sorted-by-event_id list and use `bisect_left` / `bisect_right` for O(log N) windowing; or maintain an `events_by_id` dict alongside.
- `[open] [med] PERF-022` — `dashboard/evalyn_dashboard/api/cli.py:323, 362` — `get_command_history` calls `persistence.list_recent(limit=500)` twice in one request to compute total count and `used_count_this_week`. Two full row scans per request. Fix: single SQL query with conditional aggregation (`COUNT(*) FILTER (WHERE created_at >= week_start)` alongside `COUNT(*)`), or pull rows once and bucket in Python.

### New findings — extensibility

- `[open] [low] EXT-030` — `dashboard/evalyn_dashboard/server.py` (and scattered across other backend modules) — env-var names (`EVALYN_LOG_LEVEL`, `EVALYN_MAX_CONCURRENT_JOBS`, `EVALYN_AGENT_CONFIRM_TIMEOUT_S`, `EVALYN_AGENT_PURGE_INTERVAL_S`, `EVALYN_AGENT_THREAD_TTL_S`) are read at use-site in several files. No central registry — adding a tunable requires editing multiple files, and there's no single place to enumerate or document them. Fix: introduce a `config.py` module with an `@dataclass`-style `BackendConfig` populated from `os.environ` at startup.
- `[open] [low] EXT-031` — `dashboard/evalyn_dashboard/api/settings.py:34-51` `HARDCODED_MODELS` — same plugin-registry smell as EXT-001 / EXT-026 but for model lists. Model catalogs drift fast (the seed audit was authored in 2026-03; this is now 2026-05 and the list already pre-dates several public releases). Fix: fetch from provider APIs on demand (with valid key) and cache; or move to a versioned JSON file refreshed by a build step.

### Spot-checks of prior high-severity findings (all still `[open]`)

- `PERF-012` (rubrics N+1 ~15k traversals/req) — verified still present at `api/v2/rubrics.py:310-322` (`for root in dataset_roots(): for ds in list_dataset_dirs(root):` nested in `_load_saved_rubric` called once per metric in `list_rubrics`).
- `PERF-013` (smart_queue full-file scan) — verified still present at `api/v2/annotation.py:1008-1018` (`coverage_counter: dict[str, int] = defaultdict(int)` rebuilt from per-dataset `reviews_dir.glob("*.jsonl")` walk on every request).

### Security baseline re-confirmation (regression watch)

The seed audit's "ruled out" baseline holds in the non-v2 backend surface too:

- No shell-string subprocess invocations; all spawns are `asyncio.create_subprocess_exec()` with argv lists.
- No dynamic code-evaluation calls on non-constant strings anywhere in the scanned set.
- No unsafe binary deserialization on inbound data; only `json.loads`.
- No SQL injection: all SQL uses `?` placeholders (matches SEC-001's "single low-sev case" for f-string column names being the lone exception).
- `/api/files/read` `Path.resolve().relative_to(root)` containment confirmed at the endpoints.

### Loop value note

After 6 iterations the recurring audit has materially exceeded the seed: 23 additional findings (5 v2 perf, 10 frontend, 1 compat, 7 backend non-v2 including 1 crit) plus a precision-upgraded DC-001. The crit in SEC-002 alone justifies the cost of every iteration so far combined.

---

## Iteration 7 delta (2026-05-12 02:32 PDT)

No commits since iteration 6. Extended coverage to `example_agents/` (24 Python files across 3 demos: `anthropic_research_agent/`, `googleadk_academic_research_agent/`, `langchain_deep_research_agent/`). These ship with the SDK as reference implementations — new users encounter them first, so the severity bar is "should be exemplary," not "works on my machine."

### SEC-002 status (15 min elapsed since flag)

- `[open] [crit] SEC-002` — STILL OPEN. The `logger.info(...)` call at `api/promote.py:365-374` still references bare `run_id` / `row_hashes`. Verified at iteration timestamp (re-read lines 363-374). No commits since iter 6, so no fix has landed. The recurring audit's job here is to keep the visibility level high until the fix lands.

### New findings

- `[open] [low] DC-006` — `example_agents/langchain_deep_research_agent/app.py` — 45-line FastAPI entry point (`app = FastAPI()`, `create_frontend_router(build_dir="../frontend/dist")`) that no other file in the demo imports, and which has no Dockerfile / pyproject / langgraph.json / compose config referencing it as a `uvicorn` target either. Orphan scaffolding from the original LangChain template that assumed a colocated `frontend/dist`. Fix: delete `app.py` and the `from fastapi.staticfiles` dependency note in `README` if any, or wire it up if you actually want users to launch a UI.
- `[open] [med] EXT-032` — model defaults hardcoded across all 3 example agents without an env-var override path:
  - `example_agents/anthropic_research_agent/agent.py:90, 102, 115, 146` — 4 subagents pinned to `model="haiku"`.
  - `example_agents/googleadk_academic_research_agent/academic_research/agent.py:24` and `sub_agents/*/agent.py:21-22` — 3 occurrences pinned to `model="gemini-2.5-pro"`.
  - `example_agents/langchain_deep_research_agent/configuration.py:12, 19, 26` and `agent.py:43` — 3 model fields all defaulting to `"gemini-2.5-flash-lite"` with no per-stage differentiation.
  - Cross-cuts EXT-001 (backend providers map) and EXT-026 (frontend `KNOWN_PROVIDERS`). Fix: each demo should read its model from an env var (`EVALYN_DEMO_MODEL` or per-demo equivalent) with a sane default, and document the env var in each demo's README. The langchain demo's `Configuration` dataclass is the right pattern to lift into the other two.

### Confirmed clean across example_agents/ (regression watch)

- **No committed secrets** — all API key references go through `os.environ[...]`; no `sk-` / `Bearer ` / `ANTHROPIC_API_KEY=...` literals in source.
- **No shell injection vectors** — tool execution paths don't pass LLM output to a shell.
- **No unsafe deserialization** — no use of Python's binary-deserialization stdlib module, no `yaml.load`-without-safe-loader on tool/state files.
- **No SSRF amplifiers** — URL fetching is constrained to a known set of search providers; no demo lets an LLM-generated URL drive a `requests.get`.
- **Evalyn instrumentation correctly wired** — all 3 demos use the `@eval` decorator and `create_agent_hooks()` for tracing. Good pattern; preserve if these demos are ever refactored.
- **No async/sync hazards** — no blocking I/O inside async loops.
- **No dead imports or TODO/FIXME rot** in production paths.

### Loop value note

Iteration 7 added 2 findings (small), but the example_agents/ scope was the last untouched corner of "code users will read." Coverage is now: SDK CLI + analysis + annotation + metrics + storage + trace + calibration (iters 1, 2, 5); dashboard backend api/v2/ (iter 3) and non-v2 (iter 6); dashboard frontend (iter 4); shipped demos (this iter). Remaining unaudited corners are `local_scripts/` (1 file), `research/` (intentionally excluded), and the tests themselves.

---

## Notes for future passes

- Re-running deadcode every 15 min on a 1590-file tree is wasteful. Future iterations should *narrow*: diff against the last pass's "live module" set and only re-audit modules whose mtime changed.
- The security baseline above is the bigger asset than the single low-severity finding. Treat any NEW occurrence of the ruled-out patterns as a regression worth a `[crit]` flag.
- Extensibility findings cluster around a single design choice: closed registries for providers, metrics, judges, dashboard pages. Resolving EXT-002 (entry-point discovery) likely unlocks EXT-001, EXT-003, EXT-010, EXT-011, EXT-012 in one move.
