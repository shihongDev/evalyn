# Bug Report: Evalyn Codebase

Last updated: 2026-01-15

## Summary

Found **3 critical**, **3 high**, and **5 moderate** potential bugs across the codebase.

---

## Critical Bugs

### 1. IndexError in traces.py - Unsafe List Access After Filtering
**File:** `sdk/evalyn_sdk/cli/commands/traces.py`
**Status:** Fixed

```python
first_id = calls[0].id  # crashes if filtering removed all calls
```

**Issue:** After filtering by --project/--simulation/--production, the `calls` list can become empty, but code accesses `calls[0]` without re-checking.

**Fix:** Add `if calls:` guard before accessing `calls[0]`

### 2. Race Condition in runner.py - Non-thread-safe Counter
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

```python
with progress_lock:
    current_eval += 1  # increment is NOT atomic
```

**Issue:** `current_eval` is declared in outer scope. The lock protects the block but `+=` is two operations (read + write) which can be interrupted.

**Fix:** Use `itertools.count()` for thread-safe counting

### 3. Missing Exception Handling in Parallel Execution
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

```python
for future in as_completed(futures):
    item_id, result = future.result()  # Can throw if worker failed
```

**Issue:** If a worker raises an unexpected exception, entire evaluation crashes without cleanup.

**Fix:** Wrap in try-except to gracefully handle failures

---

## High Priority Bugs

### 4. Inconsistent Progress Callback Parameters
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

- Parallel mode: `callback(..., "parallel")`
- Sequential mode: `callback(..., metric.spec.type)`

**Issue:** 4th parameter differs between modes, can break progress UI

**Fix:** Use consistent parameter (metric type) in both modes

### 5. Data Loss in Checkpoint Serialization
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

```python
json.dump(data, f, indent=2, ensure_ascii=False, default=str)
```

**Issue:** Non-serializable objects (datetime, etc.) converted to strings, losing type info on restore

**Fix:** Use custom JSON encoder that preserves datetime as ISO strings with type markers

### 6. Missing Encoding on File Opens (Windows Issue)
**File:** Multiple files in `sdk/evalyn_sdk/cli/commands/`
**Status:** Fixed

```python
with open(dataset_file) as f:  # Missing encoding="utf-8"
```

**Issue:** Windows may use cp1252 instead of UTF-8, causing UnicodeDecodeError

**Fix:** Add `encoding="utf-8"` to all file opens

---

## Moderate Bugs

### 7. Unsafe EvalRun Construction
**File:** `sdk/evalyn_sdk/cli/commands/analysis.py`
**Status:** Fixed

```python
run1 = EvalRun(**data)  # No validation or error handling
```

**Fix:** Add try-except with meaningful error messages

### 8. Broad Exception Handling
**Files:** evaluation.py, infrastructure.py
**Status:** Acknowledged (intentional in some cases)

```python
except Exception:
    pass  # Silently ignores all errors
```

**Note:** Some broad exception handling is intentional for non-critical operations (e.g., extracting source code metadata). Critical paths have proper error handling.

### 9. JSON Serialization Loses Type Info
**File:** `sdk/evalyn_sdk/metrics/judges.py`
**Status:** Acknowledged (by design)

```python
detail_str = json.dumps(detail, default=str)  # datetime/numpy lost
```

**Note:** This is intentional - LLM judges receive string representations for simplicity.

### 10. Checkpoint Validation Missing
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

No validation of checkpoint data integrity before restoring.

**Fix:** Add basic validation and graceful fallback if checkpoint is corrupted

### 11. Implicit Type Conversions
**File:** `sdk/evalyn_sdk/runner.py`
**Status:** Fixed

```python
completed = set(data.get("completed_items", []))  # Types may not match
```

**Fix:** Ensure item IDs are consistently stored and restored as strings

---

## Files Modified

1. `sdk/evalyn_sdk/cli/commands/traces.py` - Fix IndexError
2. `sdk/evalyn_sdk/runner.py` - Fix race condition, exception handling, progress callback, checkpoint
3. `sdk/evalyn_sdk/cli/commands/analysis.py` - Add encoding, error handling
4. Multiple CLI files - Add `encoding="utf-8"` to file opens
