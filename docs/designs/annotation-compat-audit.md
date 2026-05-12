# Audit: CLI ↔ dashboard annotation compatibility

Date: 2026-05-12
Trigger: retro `/plan-eng-review` of the May 4 `/annotate` ship.
Question: can a user hop between the CLI annotation flow and the dashboard
`/annotate` UI without losing work or hitting silent failures?

Short answer: **no.** Both surfaces write to `<dataset_dir>/annotations.jsonl`
but with incompatible record shapes; neither reader handles the other's shape.

## Two annotation surfaces, two record shapes

### CLI shape (`evalyn annotate ...`)

Producer: `sdk/evalyn_sdk/cli/commands/annotation.py::cmd_annotate`
Record class: `sdk.evalyn_sdk.models.Annotation` (+ `MetricLabel`)
Written to: `<dataset_dir>/annotations.jsonl`

```json
{
  "id": "ann-abc12345-3",
  "target_id": "call_abc12345",
  "label": true,
  "rationale": "looks correct",
  "annotator": "human",
  "source": "human",
  "confidence": 4,
  "metric_labels": {
    "helpfulness": {
      "metric_id": "helpfulness",
      "agree_with_llm": true,
      "human_label": true,
      "notes": ""
    }
  },
  "created_at": "2026-05-04T14:30:00+00:00"
}
```

Per-metric verdict carries `agree_with_llm` (a tri-derivable from
`human_label vs llm_label`) plus the `human_label: bool`.

### Dashboard shape (`/annotate` web UI)

Producer: `dashboard/evalyn_dashboard/api/v2/annotation.py::post_verdict` +
`finalize_session`
Written to:
- `<dataset_dir>/annotation_sessions/<sid>.jsonl` (event log)
- `<dataset_dir>/reviews/<run>.jsonl` (when source kind = run)
- `<dataset_dir>/annotations.jsonl` (on finalize)

```json
{
  "item_id": "call_abc12345",
  "labels": [
    {
      "metric_id": "helpfulness",
      "label": "pass",
      "used_ai_verdict": true,
      "note": null
    }
  ],
  "skipped_metrics": [],
  "note": null,
  "evidence": [
    {"snippet": "ripples in spacetime", "metric_id": "helpfulness", "note": null}
  ],
  "annotator_id": "shiho",
  "session_id": "ann-20260504-073930_a1b2c3",
  "ts_iso": "2026-05-04T07:39:30+00:00"
}
```

## Field-level diff

| Concept | CLI | Dashboard | Compat? |
|---|---|---|---|
| Item identifier | `target_id` | `item_id` | NO - different key |
| Overall label | `label: bool` (derived) | not stored | NO |
| Per-metric labels | `metric_labels: {metric_id -> {agree_with_llm, human_label, notes}}` | `labels: [{metric_id, label: "pass"\|"fail"\|"skip", used_ai_verdict, note?}]` | NO - dict vs list, bool vs string enum |
| Skipped metrics | implicit (absent from `metric_labels`) | explicit `skipped_metrics: [str]` | partial |
| Annotator | `annotator: str` | `annotator_id: str` | NO - different key |
| Created timestamp | `created_at: ISO` | `ts_iso: ISO` | NO - different key |
| Annotation ID | `id: "ann-<call_id-prefix>-N"` | implicit (record is keyed by `(annotator_id, item_id, session_id)`) | NO |
| Provenance | `source: "human"` | implicit | NO |
| Confidence | `confidence: int (1-5)` | per-label `confidence: float` (validated but not persisted) | NO - scale + scope |
| Free-text rationale | `rationale: str?` | item-level `note?` + per-label `note?` + `evidence[]` | partial |
| Evidence snippets | not supported | `evidence: [{snippet, metric_id?, note?}]` | dashboard-only |
| Session grouping | not supported | `session_id` | dashboard-only |
| AI verdict anchoring | not tracked | `used_ai_verdict: bool` per label | dashboard-only |

## Observable consequences

### 1. Calibration pipeline blind to dashboard annotations

`evalyn calibrate` reads `<dataset>/annotations.jsonl` via the CLI
`Annotation.from_dict` path. When it hits a dashboard-written line, it
parses for `target_id` and `metric_labels` and finds neither. Result:
the dashboard's annotations are invisible to calibration unless the user
manually translates them.

The dashboard's design doc claimed:

> Verdicts on 'run' sources also append to reviews/<run>.jsonl so the
> existing calibration_suggestions detector picks them up automatically.

True for the `calibration_suggestions` detector (which reads
`reviews/<run>.jsonl` loosely). False for `evalyn calibrate` (which reads
`annotations.jsonl` strictly). The detector and the calibrator look at
different files; the dashboard fixed the easier surface but left the
harder one broken.

### 2. `evalyn annotation-stats` crashes / silently skips dashboard rows

`cmd_annotation_stats` (`sdk/evalyn_sdk/cli/commands/annotation.py:86`)
calls `AnnotationItem.from_dict(data)` on every line of `annotations.jsonl`.
Dashboard-shaped rows don't match `AnnotationItem`'s schema:
they'll either raise `KeyError` (no `id`/`target_id`) or instantiate with
missing fields and silently corrupt the stats output.

Test it: `evalyn annotation-stats --dataset <ds-with-dashboard-annotations>`.

### 3. Dashboard's own test acknowledges the divergence

```python
# dashboard/tests/test_api_v2_annotation.py:215-220
# The fixture ships a pre-existing annotations.jsonl with a different
# shape (target_id instead of item_id). Finalize appends our records
# without touching legacy ones; find ours by session_id.
lines = [json.loads(l) for l in canonical.read_text().splitlines() if l.strip()]
ours = [r for r in lines if r.get("session_id") == sid]
```

The test workaround proves the test author knew the file would have
mixed shapes. No alarm was raised at the time.

### 4. Cannot resume a CLI session in the dashboard (and vice versa)

CLI annotation is stateless: each invocation reads the existing
`annotations.jsonl`, lists unannotated items, appends as the user works.
Dashboard introduces `annotation_sessions/<id>.json` to track in-flight
progress.

A user who annotates 10 items via `evalyn annotate`, then visits
`/annotate` in the browser, will see zero in-progress sessions even
though `annotations.jsonl` has 10 records. The dashboard only lists
sessions it created itself.

## What "compatibility" should mean

Two practical levels:

**Read-side compat** (small, high-value): any reader of `annotations.jsonl`
handles both shapes. Calibration sees all annotations regardless of which
surface produced them. `annotation-stats` doesn't crash on dashboard rows.
The dashboard's finalize dedup correctly identifies CLI-written records.

**Write-side compat** (larger, debatable value): both surfaces write a
canonical superset shape so future readers don't need normalization.
Requires schema migration of existing `annotations.jsonl` files.

**Full bidirectional symmetry** (out of scope): CLI annotations show up
as resumable sessions in the dashboard, and vice versa. Requires
retrofitting session semantics onto the CLI surface. Not worth doing
unless multi-surface annotation becomes a real user request.

## Recommended approach: read-side compat first

1. **Canonical record** module: `sdk/evalyn_sdk/annotation/_records.py`
   exports `detect_shape(rec: dict) -> Literal["cli", "dashboard", "unknown"]`
   and `normalize_to_canonical(rec: dict) -> CanonicalAnnotationRecord`.

2. **`cmd_annotation_stats`** (and the calibration loader, separately)
   route every JSONL line through `normalize_to_canonical`. The CLI's
   `Annotation.from_dict` becomes the fallback for "cli"-shaped records.

3. **Dashboard's `finalize_session` dedup**: extend the dedup key to use
   normalized `item_id` so CLI records added later don't get re-merged.

4. **Docstring update** in both surfaces pointing at the canonical module
   as the single source of truth for the on-disk shape.

5. **Deferred** to TODO: write-side canonical shape + migration of legacy
   `annotations.jsonl` files. Revisit when multi-surface annotation
   becomes a measured pain.

## Not in scope

- Session resume across surfaces
- Multi-annotator UI (separately deferred in original design)
- Schema migration of existing on-disk records
- Evidence/snippet support in the CLI surface

## Out of this audit, a perf finding worth noting

`post_verdict` calls `_replay_log` on every request: appends one line,
then re-reads the entire log to recompute progress. Quadratic in session
size. Worth fixing alongside the compat work since both touch the same
module. See eng-review test plan dated 2026-05-12 for details.
