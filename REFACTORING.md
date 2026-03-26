# Evalyn Refactoring Guide

Comprehensive architectural analysis and code quality findings for `sdk/evalyn_sdk/`.
Covers dependency structure, data flow, interface design, state management, extensibility,
CLI-engine coupling, type safety, and serialization consistency.

---

## Part 1: Architecture

### Dependency Graph

```
models.py                    (pure data, zero internal imports)
    ^
    |-- storage/base.py      (protocol, imports models)
    |-- storage/sqlite.py    (imports models, migrations)
    |-- trace/context.py     (imports models via TYPE_CHECKING only)
    |-- trace/tracer.py      (imports models, storage/base, trace/context)
    |-- decorators.py        (imports storage/sqlite, trace/otel, trace/tracer)  <-- VIOLATION
    |-- datasets.py          (imports models)
    |-- parsing.py           (imports models.FunctionCall)
    |-- attribution.py       (imports models)
    |-- evaluation/
    |    |-- execution.py    (imports models only)
    |    |-- units/*         (imports models only)
    |    |-- runner.py       (imports decorators, datasets, units, execution, models)
    |-- metrics/             (imports models, judges, parsing, defaults)
    |-- judges/              (imports models, parsing, defaults)
    |-- calibration/         (imports models, judges, metrics, evaluation/runner)
    |-- analysis/            (imports models only)
    |-- cli/                 (imports decorators, storage, datasets, analysis, models)
```

**Layering violations:**

1. `decorators.py` imports `SQLiteStorage` directly (concrete class, not protocol). The user-facing facade depends on infrastructure. Swapping storage requires editing the decorator.

2. `evaluation/runner.py` imports `calculate_cost` from `trace/instrumentation/providers/_shared.py` inside a method body. The evaluation layer reaches into provider-specific pricing code.

3. `trace/tracer.py` -> `auto_instrument.py` -> `decorators.py` (deferred circular loop, broken only by function-level import).

### Data Flow

```
@decorated function call
  -> EvalTracer.start_call()          creates FunctionCall, sets ContextVars
  -> LLM SDK call (monkey-patched)    creates Spans via context.span()
  -> EvalTracer.finish_call()         collects spans, stores call via storage.store_call()
                                      (manual field-by-field extraction, NOT call.as_dict())

build-dataset
  -> build_dataset_from_storage()     fetches up to fetch_limit*5 calls, filters IN PYTHON
  -> DatasetItem constructed          manually (not using any factory method)

run-eval
  -> EvalRunner._prepare_item_call()  lookup call from storage OR create synthetic FunctionCall
  -> strategy.execute()               SequentialStrategy or ParallelStrategy
  -> metric.evaluate(call, item)      handler(call, item) -> MetricResult
  -> store_eval_run()                 EvalRun serialized and stored

analyze / insights / compare
  -> analyze_run(run.as_dict())       builds RunAnalysis from dict (not from EvalRun directly)
```

**Inefficiencies:**

- `store_call` in sqlite.py extracts fields manually instead of calling `call.as_dict()`. Adding a field requires edits in 3+ places.
- `build_dataset_from_storage` fetches 5x more calls than needed and filters in Python. SQL-pushable filters (`function_name`, `since`, `until`) are applied post-fetch.
- `analyze_run` takes a dict (from `run.as_dict()`), not an `EvalRun` object. Every caller calls `.as_dict()` first, then the function reconstructs structure from the dict.

### State Management

| Location | Variable | Problem |
|---|---|---|
| `context.py:43-47` | `_global_call_id`, `_global_collectors`, `_orphan_spans` | Module-level mutables, not ContextVar. Data races under concurrency. |
| `context.py:29-30` + `tracer.py:19` | Two `_active_call` ContextVars | Same concept, two variables. Manually synchronized - fragile. |
| `decorators.py:12` | `_default_tracer` singleton | Hardcodes SQLiteStorage. No reset path for tests. |
| `config.py:14` | `_project_root_cache` | Never invalidated. Breaks tests that change cwd. |
| `tracer.py:76` | `_function_meta_cache` keyed by `id(func)` | Python reuses object IDs after GC. Rare but real stale-cache bug. |
| `registry.py:26` | `InstrumentorRegistry._instance` singleton | Has `.reset()` for tests. Correct pattern. |

### Configuration Architecture

No typed config object exists. Config flows as `Dict[str, Any]` everywhere:

```
evalyn.yaml -> load_config() -> Dict[str, Any]
  -> each CLI command calls get_config_default(config, "key1", "key2")
  -> ad-hoc parameter extraction, no validation, no schema
```

Environment variables are read in 6+ scattered locations with no central registry documenting them.

---

## Part 2: CLI-Engine Coupling

### The Core Problem

CLI commands contain 60-75% business logic. The pipeline calls CLI functions via fake `argparse.Namespace` objects. There is no service layer.

**Business logic ratios by command:**

| Command file | Lines | Business logic % | Worst pattern |
|---|---|---|---|
| evaluation.py | ~1,450 | 62% | 6 helper functions that are pure factory/orchestration logic |
| analysis.py | ~1,000 | 70% | `_aggregate_analysis_stats` duplicates `analyze_run` from core |
| annotation.py | ~800 | 75% | Storage access, analytics, file I/O all inline |
| calibration.py | ~750 | 53% | Optimizer config factory embedded in CLI |
| traces.py | ~750 | 67% | In-Python query filtering that should be SQL |
| insights.py | ~345 | 64% | 9-step orchestration workflow inline |
| simulate.py | ~282 | 64% | Query persistence in CLI handler |
| export.py | ~367 | 68% | Full report generators with embedded HTML/CSS |

### The argparse.Namespace Anti-Pattern

`pipeline_steps.py` calls CLI commands by constructing fake Namespace objects:

- **AnnotationStep** (line 468): 10-field Namespace to call `cmd_annotate`
- **CalibrationStep** (line 558): **30+ field Namespace** duplicating all optimizer arg defaults
- **SimulationStep** (line 769): 10-field Namespace to call `cmd_simulate`

When CLI arguments change (new optimizer params, renamed flags), these Namespace constructions silently break.

### Direct SQLiteStorage Construction (7 locations)

CLI commands construct `SQLiteStorage()` directly instead of using dependency injection:
- `pipeline_steps.py:98, 183`
- `command_common.py:54`
- `analysis.py:754, 954`
- `quickstart.py:325`
- `infrastructure.py:174`

### `fatal_error()` in Business Logic

Business logic helpers call `fatal_error()` (a `sys.exit()` wrapper), making them unusable as library functions. Any Python API caller would need to catch `SystemExit`.

### Repeated Patterns Across Commands

**Metrics building** - 3 independent implementations of "read specs, create MetricSpec, build Metric objects":
- `evaluation.py:302-425`
- `pipeline_steps.py:26-73`
- `pipeline_steps.py:686-699`

**Dataset path resolution** - 3 implementations:
- `command_common.py:90-144`
- `dataset_utils.py` reimplements it
- `clustering.py` adds a third

**Run loading** - 11 commands load eval runs, 4 bypass the shared helper and call `tracer.storage.get_eval_run` directly.

**Metric summary table** - 4 commands format metric summary tables independently.

### Proposed Architecture

```
Current:
  Pipeline -> CLI cmd_*() -> Engine (via fake Namespace)

Proposed:
  CLI cmd_*()  ----\
                    +---> Service Layer ---> Engine Layer
  Pipeline --------/      (typed args)      (raises exceptions)
                                            (no print/sys.exit)
```

Service layer would provide:
- `EvalService.run(dataset_path, metrics_path, config) -> EvalRun`
- `InsightsService.analyze(run, dataset_path, prev_run) -> InsightsReport`
- `CalibrationService.calibrate(metric_id, annotations, config) -> CalibrationRecord`
- `DatasetService.build(storage, filters, sampling) -> (items, meta)`
- `SimulationService.simulate(seed_items, target_fn, config) -> SimulationResult`

---

## Part 3: Type Safety and Protocols

### Protocol Gaps

**StorageBackend** is missing 4 methods that `SQLiteStorage` exposes publicly and that evaluation/CLI code calls:
- `batch_insert_metric_results`
- `load_metric_results`
- `store_span_metric_links`
- `list_span_metric_links`

Any alternative backend implementing only the protocol is silently incompatible.

**`list_spans`** returns `List[Dict[str, Any]]` instead of `List[Span]`.

### ABC Completeness

| ABC | Complete? | Issue |
|---|---|---|
| `Instrumentor` | Yes | Clean 5-method contract |
| `EvalUnitBuilder` | Yes | 2 abstract methods, clean |
| `ExecutionStrategy` | Yes | Cleanest ABC in codebase |
| `BatchProvider` | Yes | Private abstract `_get_api_key()` is unusual but intentional |
| `ConfidenceEstimator` | Weak | `estimate(**kwargs)` - callers can't discover required args |
| `BaseOptimizer` | Not an ABC | `optimize()` raises `NotImplementedError` but isn't `@abstractmethod`. Bad subclasses fail at runtime, not instantiation. |

### Serialization Inconsistency

| Approach | Classes using it |
|---|---|
| Manual dict | Span, TraceEvent, FunctionCall, DatasetItem, JudgeConfig, EvalRun, HumanLabel, Annotation, MetricSpec, CalibrationRecord |
| `dataclasses.asdict()` | EvalUnit, EvalView, MetricResult, SpanMetricLink |

`asdict()` does NOT convert `datetime` to ISO strings. Currently safe because those 4 classes have no datetime fields - but adding one would silently break serialization.

`_dumps()` in `sqlite.py` uses `default=lambda o: repr(o)` - unserializable objects become strings like `"datetime.datetime(2026, ...)"` that can never be deserialized back. This is a permanent data corruption path.

### Missing `from_dict()` Round-Trip

- `CalibrationRecord`: has `as_dict()`, no `from_dict()`. Cannot reconstruct from serialized form.
- `EvalView`: has `as_dict()`, no `from_dict()`. Write-only serialization.
- `DatasetItem`: uses `from_payload()` instead of `from_dict()`. Breaks naming convention.

### Untyped Dicts That Should Be Dataclasses

- `MetricResult.raw_judge: Dict[str, Any]` - always has `score, passed, reason, raw, input_tokens, output_tokens, model`
- `DatasetItem.metadata: Dict[str, Any]` - always has `call_id, function, duration_ms, error, session_id` when from FunctionCall
- `EvalRun.summary: Dict[str, Any]` - known schema from `runner.py:408-418`
- `EvalRun.usage_summary: Dict[str, Any]` - known schema from `runner.py:464-474`
- `ItemStats.metric_results: Dict[str, Dict[str, Any]]` - inner dict always has `passed, score, reason, details`

### Surprising Defaults

- `HumanLabel.from_dict()` defaults `passed=True` when key is missing. Absent data = pass.
- `MetricSpec.unit_types` has double-defaulting: `field(default_factory=lambda: ["outcome"])` AND defensive `or ["outcome"]` in `as_dict()`.

---

## Part 4: Extension Point Costs

| Extension | Files to touch | Effort | Main friction |
|---|---|---|---|
| New objective metric | 1-2 (objective.py + factory.py) | Low | Registry in factory is a plain dict |
| New subjective metric | 1 (subjective.py) | Low | Just add to SUBJECTIVE_REGISTRY |
| New instrumentor | 2 (new file + auto_instrument.py) | Medium | 5 abstract methods |
| New eval unit builder | 1-2 (builders.py + views.py) | Low | Clean pattern |
| New execution strategy | 1-2 (execution.py) | Low | Single abstract method |
| New calibration optimizer | 2 (new file + factory.py) | Medium | BaseOptimizer is not a real ABC |
| New LLM judge provider | 3 (api_client.py + llm_judge.py + factory.py) | Medium | Provider dispatch duplicated in 3 places |
| New storage backend | 1 file + integration | **High** | Protocol missing 4 methods, decorators.py hardcodes SQLiteStorage |
| New batch provider | 1-2 | Medium | 5 abstract methods |
| New confidence estimator | 2-3 | Medium | kwargs-based estimate() hides contract |

### Extractable Generic Patterns

**Registry[T]**: Three distinct registry implementations (MetricRegistry, InstrumentorRegistry, OPTIMIZER_REGISTRY as a plain dict, _BUILDERS as a plain dict). A generic `Registry[T]` with `register/get/list/keys` could unify them.

**Provider dispatch**: LLM provider selection (gemini/openai/ollama model defaults, client construction) is repeated in `LLMJudge.__init__`, `build_subjective_metric()`, and `_wrap_with_*_confidence()`. A single `ProviderFactory` would centralize this.

---

## Part 5: Specific Code Issues

### Critical (data integrity / correctness)

| # | Issue | File:Line | Fix |
|---|---|---|---|
| 1 | Global mutable state data race in span collection | `context.py:43-47` | Use thread-keyed dict or `threading.local` |
| 2 | Composite PK truncation (64 chars) causes silent result overwrite | `sqlite.py:336` | Hash the composite key |
| 3 | Bare except swallows checkpoint errors | `runner.py:126-128, 163-165` | Log warning, check return value |
| 4 | ContextVar token discarded, wrong reset in async | `context.py:180-190` | Use `cv.reset(token)` |
| 5 | JSON brace-matching breaks on `}` inside strings | `parsing.py:36-50` | Use `rfind("}")` approach |
| 6 | Unit eval bypasses checkpoint + parallel strategy | `runner.py:269-306` | Extend to use ExecutionStrategy |
| 7 | Ollama confidence uses generation speed as probability | `api_client.py:462-473` | Return 0.5, warn |
| 8 | Silent metric-build failures (bare `except: pass`) | `pipeline_steps.py:63-71` | Log warning |

### Important (code quality / maintainability)

| # | Issue | File:Line | Fix |
|---|---|---|---|
| 9 | StorageBackend protocol missing 4 public methods | `storage/base.py:9-31` | Add to protocol |
| 10 | N+1 query in list_eval_runs | `sqlite.py:656-682` | Batch load with `IN (...)` |
| 11 | DatasetItem dual-field sync breaks after mutation | `models.py:519-529` | Remove dual fields, use properties |
| 12 | MetricSpec constructed manually in 4 places | Multiple | Use `MetricSpec.from_dict()` |
| 13 | MultiTurnBuilder non-deterministic unit IDs | `builders.py:165` | Derive from span IDs |
| 14 | `sample_diverse` ignores caller seed | `sampling.py:114-116` | Add seed parameter |
| 15 | Consistency tie-break inconsistent score/passed | `factory.py:180-188` | Set `score = 1.0 if passed else 0.0` |
| 16 | Process-level cache breaks tests | `config.py:13-45` | Accept optional `cwd` argument |
| 17 | `_aggregate_analysis_stats` duplicates `analyze_run` | `analysis.py:357-409` | Delete, use `analyze_run()` |
| 18 | Hardcoded `/7` step count | `pipeline.py:117` | Compute from `len(self.steps)` |
| 19 | `list_datasets` misses `data/prod/datasets/` | `dataset_resolver.py:140-150` | Mirror 3-location logic |
| 20 | CalibrationStep 30+ field Namespace | `pipeline_steps.py:558-601` | Extract typed service function |
| 21 | Duplicated scorer client init | `base_optimizer.py:105-112, 152-161` | Extract `_get_scorer_client()` |
| 22 | Default model table duplicated | `llm_judge.py:75-80` + `factory.py:607-611` | Move to `defaults.py` |
| 23 | Substring model matching brittle | `_shared.py:183-190` | Exact match first, then prefix |
| 24 | Dataset path normalization duplicated 3x | `command_common.py:90-144` | Extract shared function |
| 25 | CalibrationEngine silently discards deprecated fields | `engine.py:102-126` | Warn when both provided |

---

## Part 6: Refactoring Priorities

### Phase 1 - Data Integrity (do first, low risk)
- ~~Fix composite PK truncation (#2)~~ DONE
- ~~Fix checkpoint error handling (#3)~~ DONE
- ~~Fix JSON parsing (#5)~~ DONE
- ~~Use MetricSpec.from_dict everywhere (#12)~~ DONE
- ~~Fix consistency tie-break (#15)~~ DONE

### Phase 2 - Concurrency Correctness
- ~~Fix global state data race (#1)~~ DONE
- ~~Fix ContextVar token reset (#4)~~ DONE
- ~~Fix unit eval strategy bypass (#6)~~ DONE
- ~~Fix non-deterministic unit IDs (#13)~~ DONE
- ~~Fix sample_diverse seed (#14)~~ DONE

### Phase 3 - Code Quality
- ~~Fix silent metric build failures (#8)~~ DONE
- ~~Fix StorageBackend protocol (#9)~~ DONE
- ~~Fix N+1 query (#10)~~ DONE
- ~~Fix DatasetItem dual fields (#11)~~ DONE
- ~~Fix hardcoded /7 step count (#18)~~ DONE

### Phase 4 - Maintainability
- ~~Consolidate default model table (#22)~~ DONE
- ~~Extract _get_scorer_client (#21)~~ DONE
- ~~Extract dataset path normalization (#24)~~ DONE
- ~~Delete _aggregate_analysis_stats, use analyze_run (#17)~~ DONE
- ~~Fix list_datasets missing data/prod/datasets (#19)~~ DONE
- ~~Extract CalibrationStep typed service (#20)~~ DONE
- ~~Fix CalibrationEngine deprecated field warnings (#25)~~ DONE
- ~~Fix substring model matching (#23)~~ DONE
- ~~Fix process-level cache (#16)~~ DONE

### Phase 5 - Service Layer Extraction
- Create `services/` directory with typed service classes
- Migrate business logic from CLI commands to services
- ~~Replace argparse.Namespace pipeline calls with service calls~~ DONE (all 3: Calibration, Annotation, Simulation)
- Replace `fatal_error()` with exceptions in business logic
- ~~Replace direct `SQLiteStorage()` construction with injection~~ DONE (7 sites eliminated)

### Phase 6 - Protocol and Type Completeness (future)
- ~~Complete StorageBackend protocol (add 4 missing methods)~~ DONE
- ~~Make BaseOptimizer a proper ABC~~ DONE
- ~~Type the `**kwargs` on ConfidenceEstimator.estimate()~~ DONE (documented per-subclass args, fixed details field)
- Replace untyped Dict fields with dataclasses (raw_judge, summary, usage_summary)
- Unify serialization (mixin or consistent manual approach, handle datetime everywhere)

### Phase 7 - Generic Abstractions (future)
- Extract generic Registry[T] from 4 registry implementations
- Extract ProviderFactory from 3 provider dispatch locations
- Extract MetricsLoader from 3 metric-building implementations

*Last updated: 2026-03-26*
