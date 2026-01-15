# CLI UX Improvements Plan

Reviewed all 27 CLI commands from user perspective. Found 50+ issues across categories.

---

## CRITICAL (Breaking/Blocking)

### 1. Missing `--format json` for `show-call`
- **File:** traces.py:883-886
- **Impact:** Can't script/automate with show-call output
- **Fix:** Add `--format` arg, refactor output to support JSON

### 2. Error exit codes not used
- **Files:** traces.py:202,644, analysis.py:various
- **Impact:** Scripts can't detect failures (returns 0 on error)
- **Fix:** Return sys.exit(1) on errors, not just print

### 3. Dead code - flags registered but unused
- **`--quiet`:** Checked in evaluation.py:555 but never registered in argparser
- **`--only-disagreements`:** Registered annotation.py:986 but never used
- **Fix:** Either implement or remove

### 4. Duplicate utility functions
- **Files:** evaluation.py vs dataset_utils.py
- **Functions:** `_resolve_dataset_and_metrics`, `_dataset_has_reference`, `_extract_code_meta`
- **Fix:** Consolidate to single location

### 5. `cmd_one_click` is 820+ lines
- **File:** infrastructure.py:96-912
- **Impact:** Unmaintainable, hard to test
- **Fix:** Break into smaller functions/classes

---

## HIGH (Confusing UX)

### 6. Inconsistent `--format` support
| Command | Has --format? |
|---------|---------------|
| list-calls | Yes |
| show-call | No |
| list-runs | Yes |
| show-run | No |
| suggest-metrics | No |
| list-metrics | No |
| analyze | No |
| trend | Yes |

**Fix:** Add --format to all list/show commands

### 7. Inconsistent `--latest` support
| Command | Has --latest? |
|---------|---------------|
| run-eval | Yes |
| suggest-metrics | No |
| status | Yes |
| validate | Yes |
| analyze | Yes |
| compare | No |
| trend | No |

**Fix:** Add --latest to all dataset-consuming commands

### 8. Inconsistent `--quiet` support
- Main parser has `-q/--quiet` (main.py:73-74)
- But individual commands don't respect it consistently
- **Fix:** Pass quiet flag to all hint/progress functions

### 9. Silent metric deduplication
- evaluation.py:267-291 - first metric wins, no warning
- **Fix:** Warn user when duplicate metric IDs found

### 10. Error message quality varies
- Good: evaluation.py:579 (shows usage examples)
- Bad: analysis.py:419 (vague alternatives)
- **Fix:** Standardize error message format with suggestions

---

## MEDIUM (Usability)

### 11. No filtering in list commands
- list-calls: only --project, --simulation/--production
- Missing: --error-only, --date-range, --duration-min/max, --function
- **Fix:** Add common filters

### 12. No sorting/pagination
- list-calls: default 20, no indication of more
- **Fix:** Add --sort, --offset, show "X more available"

### 13. Progress indicators inconsistent
- ProgressBar in evaluation.py:402
- No progress in annotation, calibration
- **Fix:** Add progress to long operations

### 14. Hint system incomplete
- Some commands show next steps, others don't
- compare, trend have no hints
- **Fix:** Add hints to all commands

### 15. Time formatting inconsistent
- list-calls: raw datetime
- show-trace: smart formatting (123ms, 1.2s)
- **Fix:** Use consistent smart formatting

### 16. Truncation inconsistent
- Different max lengths: 300, 120, 400 chars
- Different indicators: "...", "..." prefix
- **Fix:** Standardize truncation with --full flag

---

## LOW (Polish)

### 17. Emoji usage
- show-trace uses emoji (search icon, book icon)
- Violates CLAUDE.md style guide
- **Fix:** Replace with text indicators

### 18. Column names unclear
- `sim?` instead of `simulation`
- **Fix:** Use clear column names

### 19. No --depth for span trees
- Large trees overwhelm terminal
- **Fix:** Add --max-depth option

### 20. No call comparison
- Can't diff two calls
- **Fix:** Add compare-calls command (low priority)

---

## Implementation Priority

### Phase 1: Critical fixes (immediate value)
1. Add `--format json` to show-call
2. Fix exit codes for errors
3. Implement or remove dead flags
4. Consolidate duplicate functions

### Phase 2: Consistency (user trust)
5. Add `--format` to remaining commands
6. Add `--latest` to remaining commands
7. Fix `--quiet` propagation
8. Standardize error messages

### Phase 3: Usability (power users)
9. Add filtering to list commands
10. Add sorting/pagination
11. Add progress indicators
12. Complete hint system

### Phase 4: Polish
13. Fix emoji usage
14. Improve column names
15. Add --depth for trees

---

## Files to Modify

| File | Changes |
|------|---------|
| traces.py | --format for show-call, exit codes, column names |
| evaluation.py | Remove duplicates, --format for suggest/list-metrics |
| analysis.py | --format for analyze/compare, --latest for compare/trend |
| annotation.py | Implement --only-disagreements or remove |
| calibration.py | Progress indicator |
| infrastructure.py | Break down cmd_one_click |
| runs.py | --format for show-run |
| main.py | Ensure --quiet passed to subcommands |
| utils/dataset_utils.py | Keep consolidated functions here |

---

## Verification

1. Run `evalyn <cmd> --help` for each command - check consistent flags
2. Test error cases return non-zero exit codes
3. Test `--format json` produces valid JSON
4. Run full test suite: `uv run pytest`
