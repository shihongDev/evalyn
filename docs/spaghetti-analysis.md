# Spaghetti Code Analysis: Evalyn Codebase

## Executive Summary

Found **significant architectural debt** across 3 areas:
- **18 functions** over 100 lines (4 over 300 lines)
- **11-level nesting** in `main()` (worst case)
- **47 repetitions** of MetricResult boilerplate
- **Duplicate implementations** between cli_impl.py and cli/commands/

---

## Critical Issues (Priority 1)

### 1. God Functions (300+ lines)

| Function | File | Lines | Nesting |
|----------|------|-------|---------|
| `cmd_one_click()` | cli_impl.py:5043 | 799 | 8 |
| `main()` | cli_impl.py:5842 | 761 | 11 |
| `cmd_show_call()` | cli_impl.py:681 | 434 | 8 |
| `cmd_annotate()` | cli_impl.py:2923 | 461 | 7 |

### 2. Massive Code Duplication

**cli_impl.py vs cli/commands/*.py** - Same functions implemented twice:
- `cmd_run_eval()` - 325 lines duplicated
- `cmd_annotate()` - 461 lines duplicated
- `cmd_one_click()` - 818 lines duplicated
- `cmd_suggest_metrics()` - 327 lines duplicated

**MetricResult boilerplate** - `objective.py` has 47 identical patterns:
```python
MetricResult(metric_id=spec.id, item_id=item.id, call_id=call.id, ...)
```

### 3. Duplicated Patterns (4 major)

| Pattern | Occurrences | Files |
|---------|-------------|-------|
| Metadata extraction | 13+ | traces.py, cli_impl.py |
| Dataset path resolution | 15+ | evaluation.py, annotation.py |
| JSON output format | 40+ | All cmd_* functions |
| API key validation | 8+ | evaluation.py, cli_impl.py |

---

## High Priority Issues (Priority 2)

### 4. Functions with Too Many Parameters (>6)

| Function | Params | File |
|----------|--------|------|
| `EvalRunner.__init__()` | 12 | runner.py:75 |
| `build_dataset_from_storage()` | 12 | datasets.py:96 |
| `LLMJudge.__init__()` | 7 | judges.py:128 |

### 5. Complex Conditionals

- `build_dataset_from_storage()` - 18 sequential if-conditions
- `run_dataset()` - 4-level fallback chain for call resolution
- `_type_ok()` - 7-way if-chain (should be dict lookup)

### 6. Mixed Concerns (I/O + Business Logic)

| File | Function | Issue |
|------|----------|-------|
| calibration.py:753 | `_validate_optimization()` | 162 lines mixing judge calls, fake data creation, metrics |
| html_report.py:108 | `generate_html_report()` | 155 lines mixing data prep, template, stats |
| tracer.py:147 | `finish_call()` | Storage I/O mixed with business logic |

---

## Medium Priority Issues (Priority 3)

### 7. Async/Sync Wrapper Duplication

- `tracer.py:211-275` - 64 lines of nearly identical code
- `auto_instrument.py:187-276` - Same duplication

### 8. Hard-coded Configuration Values

| Location | Value |
|----------|-------|
| calibration.py:232 | `"gemini-2.5-flash-lite"` |
| calibration.py:890 | `improvement_delta > 0.02` |
| html_report.py:42 | `failed_items[:30]` |
| storage/sqlite.py:22 | `"data/evalyn.sqlite"` |

### 9. Missing Abstractions

- No unified SpanContextManager (fragmented across context.py, tracer.py)
- No StorageDTO layer (direct model->SQL in sqlite.py)
- No output formatter abstraction (JSON/Table logic in every command)

---

## Recommended Refactoring Plan

### Phase 1: Quick Wins (Low Risk)
1. Extract `format_metadata()` helper - fixes 13+ duplications
2. Extract `validate_api_keys()` helper - fixes 8+ duplications
3. Extract `_make_metric_result()` factory - fixes 47 duplications
4. Replace 7-way if-chain with dict lookup in `_type_ok()`

### Phase 2: Consolidation (Medium Risk)
5. Remove duplicate cli/commands/*.py implementations - use cli_impl.py only OR vice versa
6. Extract `DatasetResolver` class - fixes 15+ path resolution duplications
7. Create output formatter classes (JSONFormatter, TableFormatter)

### Phase 3: Major Refactoring (Higher Risk)
8. Break `cmd_one_click()` into PipelineOrchestrator + StepExecutors
9. Break `cmd_show_call()` into TraceFormatter + SpanTreeRenderer
10. Create ExecutionStrategy pattern for parallel vs sequential in runner.py
11. Extract CalibrationEngine responsibilities into smaller classes

---

## Files to Modify

**Priority 1 (Critical):**
- `sdk/evalyn_sdk/cli_impl.py` - Or delete if using cli/commands/
- `sdk/evalyn_sdk/metrics/objective.py` - MetricResult factory
- `sdk/evalyn_sdk/cli/commands/*.py` - Or delete if using cli_impl.py

**Priority 2 (High):**
- `sdk/evalyn_sdk/runner.py` - Execution strategy
- `sdk/evalyn_sdk/datasets.py` - Filter builder
- `sdk/evalyn_sdk/annotation/calibration.py` - Split classes

**Priority 3 (Medium):**
- `sdk/evalyn_sdk/trace/tracer.py` - Wrapper factory
- `sdk/evalyn_sdk/analysis/html_report.py` - Split modules

---

## Metrics Summary

| Category | Count |
|----------|-------|
| Functions > 100 lines | 18 |
| Functions > 300 lines | 4 |
| Max nesting depth | 11 levels |
| Duplicated functions | 4 pairs |
| MetricResult boilerplate | 47 |
| Hard-coded values | 8 |
| Total actionable issues | 61+ |
