# evalyn_sdk Structure Cleanup - Design

Date: 2026-05-20
Status: Approved for implementation

## Problem

`sdk/evalyn_sdk/` has 189 .py files at the top level. User finds it confusing and cannot map files to modules. Existing semantic subdirs (`evaluation/`, `judges/`, `metrics/`, etc.) already exist but the top-level clutter drowns them.

## Root cause

Of 189 top-level files, only 6 are live (referenced anywhere in the SDK beyond their paired test). The other 183 are scaffolded feature implementations added in batched "feat:" commits. Each file is real Python (working dataclasses, logic) but was never wired into `__init__.py`, the CLI, or any production caller. Only paired unit tests import them.

The structure isn't badly designed - it's drowning in parked inventory.

## Goals

- Remove the top-level clutter (drop from 189 files to 7 + the existing 13 semantic subdirs).
- Preserve the existing semantic layering (`evaluation/`, `judges/`, `metrics/`, ...).
- Zero impact on the public API surface (`evalyn_sdk.EvalRun`, `evalyn_sdk.load_dataset`, `evalyn.api.evaluate(...)`, etc.).

## Non-goals

- Restructuring the live layer beyond cleanup. The existing subdir layout already matches the user's mental model. Adding a `core/` package or moving live files would churn 312 importers of `evalyn_sdk.models` with no legibility win.
- Renaming any module that has external importers.
- Refactoring code inside the live modules.

## Live layer (preserved as-is)

Top-level files that stay:

| File | Purpose | Importers |
|---|---|---|
| `__init__.py` | PEP 562 lazy public-API map | n/a |
| `models.py` | Core dataclasses (EvalRun, MetricResult, ...) | 312 |
| `decorators.py` | `@eval` public decorator + tracer config | 3 |
| `datasets.py` | Dataset load/save/hash | 12 |
| `api.py` | Programmatic Python API (`evalyn.api.evaluate(...)`) | 3 |
| `parsing.py` | LLM output parsing (JSON + verdict extraction) | 3 |
| `defaults.py` | Default model name constants | 3 |

Subdirs that stay (each is the home for one bounded responsibility):

- `analysis/` - post-run analysis, reports, insights
- `annotation/` - span annotations
- `calibration/` - judge calibration / prompt optimization
- `cli/` - command line interface
- `evaluation/` - eval runner + execution
- `integration/` - CI/CD, GitHub Action, webhooks
- `judges/` - LLM judges
- `metrics/` - objective + subjective metrics
- `simulation/` - user/agent simulation
- `testing/` - test utilities
- `trace/` - tracing + OTel
- `utils/` - shared helpers
- `storage/` - persistent storage backends (after orphan cleanup)

## Cleanup rule

**Orphan rule**: a module is an orphan if no file outside of itself and its paired test under `tests/` imports it.

Detection: `python3 /tmp/orphan_scan.py` style scan - walk `sdk/evalyn_sdk/`, walk the whole repo, count `from evalyn_sdk.<X>` and `import evalyn_sdk.<X>` references per module. Module is orphan if only its paired test imports it.

## Execution phases

### Phase 1 - Delete top-level orphans

Scope: 183 top-level `.py` files identified by the orphan rule + their paired tests.

```
sdk/evalyn_sdk/<orphan>.py  +  tests/test_<orphan>.py
```

Files kept at top-level after Phase 1: `__init__.py`, `models.py`, `decorators.py`, `datasets.py`, `api.py`, `parsing.py`, `defaults.py`.

### Phase 2 - Delete in-subdir orphans

Apply the same orphan rule to files inside each subdir. Known suspects from prior memory notes:

- `storage/` - prior memory notes flag most files as scaffolding; verify with orphan scan before deleting any
- `analysis/` - large dir; likely many orphans alongside live `core.py`, `reports.py`, `insights.py`, `clustering.py`, `trends.py`, `html_report.py`
- `evaluation/` - has ~80 files; only `runner.py` confirmed live via `__init__.py`
- `trace/` - large; only entries listed in `__init__.py` `_LAZY_IMPORTS` are guaranteed live
- `integration/`, `testing/` - small; verify each file

Per-file verification before delete: re-run orphan scan after each subdir pass to catch transitive references.

### Phase 3 - Verify nothing broke

- Run `uv run pytest tests/` (the surviving tests).
- Run `uv run evalyn --help` and a few CLI smoke commands.
- Run `python -c "import evalyn_sdk; print(dir(evalyn_sdk))"` and confirm public symbols still resolve via lazy import.

## Public API stability

`__init__.py` uses PEP 562 lazy imports (`_LAZY_IMPORTS` dict mapping public name -> module path). Phase 1 deletes only modules absent from this map, so external users importing `from evalyn_sdk import EvalRun`, `LLMJudge`, etc. are unaffected by design.

## Branch strategy

Per `<branching>` in CLAUDE.md: feature branch `cleanup/sdk-structure`. Test on branch, merge to main, delete branch.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| A "live" module is misclassified orphan | Run pytest after each phase. Any failing import surfaces the missed reference. |
| Orphan scan misses dynamic imports (`importlib.import_module(...)`) | Manual grep for `importlib`/`__import__` in live files before delete. |
| Tests reference deleted modules | Paired-test deletion is part of the rule. Any other test that imports an orphan was already broken or testing nothing. |
| User wants the scaffolding back later | The deletion is recoverable from git history (each orphan was added in a discoverable `feat:` commit). |

## Out of scope (not in this spec)

- Renaming or splitting any live module.
- Introducing new subdirs (`core/`, `pipeline/`, etc.).
- Touching `dashboard/`, `cli` shell scripts, or docs beyond updating any references that point to deleted files.

## Open questions

None. All decisions are locked in. If the per-subdir orphan scan in Phase 2 surfaces an ambiguous case (a module that looks orphan but is wired through an unusual mechanism), flag it back to the user before deleting that specific file.
