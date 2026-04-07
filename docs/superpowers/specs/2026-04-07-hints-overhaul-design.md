# Hints System Overhaul

## Problem

Current hints system prints individual `Hint: ...` lines one at a time. Issues:
- Some commands show 2 hints back-to-back (run-eval) - messy
- Some hints missing `format=` param - leak into JSON output
- "Keep up the good work!" is non-actionable filler
- No tests
- Priority chains hide useful hints (analyze shows only top-priority)

## Design

### Core: Collect-and-Render Pattern

Replace inline `print_hint()` calls with `HintCollector` that gathers hints, then renders once.

```python
class HintCollector:
    def __init__(self, quiet=False, format="table"):
        self._hints: list[tuple[str, str]] = []  # (command, description)
        self._quiet = quiet
        self._format = format

    def add(self, command: str, description: str) -> None:
        self._hints.append((command, description))

    def render(self, max_hints: int = 3) -> None:
        # suppression: quiet, EVALYN_NO_HINTS, json format
        # print "Next steps:" header
        # two-column aligned output, capped at max_hints
```

Output format:
```
Next steps:
  evalyn cluster-failures --metric-id accuracy   Cluster failures by pattern
  evalyn analyze --run a3f8b2c1                  Analyze results in detail
```

### Backward Compat

Keep `print_hint()` working - internally wraps a single-hint HintCollector. Allows incremental migration.

### Command Changes

#### traces.py (7 call sites)
- Migrate all `print_hint()` to collector pattern
- No logic changes - same conditions, same messages

#### evaluation.py (3 call sites)
- `run-eval`: collect both cluster-failures hint AND analyze hint, render together
- `suggest-metrics`: migrate to collector

#### analysis.py (5 call sites)
- `analyze`: show ALL applicable hints up to 3 (not just top priority)
  - Subjective calibration hint
  - Annotation hint
  - Trend hint
- `compare`: fix missing `format=`, show regression + improvement hints
- `trend`: fix missing `format=`, replace "Keep up the good work!" with `evalyn insights --project <name>`

#### annotation.py (3 call sites)
- Migrate to collector, no logic changes

#### calibration.py (2 call sites)
- Migrate to collector, no logic changes

#### clustering.py (2 call sites)
- Migrate to collector, no logic changes

#### dataset.py, export.py, runs.py, insights.py, dashboard.py, quickstart.py (1 each)
- Migrate to collector, no logic changes

### Bug Fixes (included)
1. `compare` hints: add `format=output_format` param
2. `trend` hints: add `format=output_format` param
3. `run-eval`: remove unconditional second hint, collect both contextually
4. Replace "Keep up the good work!" with `evalyn insights --project <name>`

### Tests (new file: tests/test_hints.py)

Suppression:
- `quiet=True` suppresses all hints
- `EVALYN_NO_HINTS=1` suppresses all hints
- `format="json"` suppresses all hints

Collector:
- `.add()` stores hints
- `.render()` prints formatted block
- Cap at max_hints (default 3)
- Empty collector prints nothing

Integration:
- Key commands produce expected hint commands under test conditions

## Files

| File | Change |
|------|--------|
| `sdk/evalyn_sdk/cli/utils/hints.py` | Add HintCollector, keep print_hint |
| `sdk/evalyn_sdk/cli/commands/traces.py` | Migrate 7 call sites |
| `sdk/evalyn_sdk/cli/commands/evaluation.py` | Migrate 3, fix double-hint |
| `sdk/evalyn_sdk/cli/commands/analysis.py` | Migrate 5, show all hints, fix format bugs |
| `sdk/evalyn_sdk/cli/commands/annotation.py` | Migrate 3 |
| `sdk/evalyn_sdk/cli/commands/calibration.py` | Migrate 2 |
| `sdk/evalyn_sdk/cli/commands/clustering.py` | Migrate 2 |
| `sdk/evalyn_sdk/cli/commands/dataset.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/export.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/runs.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/insights.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/dashboard.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/quickstart.py` | Migrate 1 |
| `sdk/evalyn_sdk/cli/commands/simulate.py` | Migrate 2 |
| `tests/test_hints.py` | New - all hint tests |
