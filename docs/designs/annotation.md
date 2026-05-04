# Design: /annotate route

Status: APPROVED — ready for implementation
Mode: HOLD SCOPE + Approach B (dedicated /annotate route)
Reviewed via: `/plan-ceo-review` (2026-05-04, 0 critical gaps, 0 unresolved)

## Goal

A user can annotate items from the web UI, batched and keyboard-driven,
with progress that survives refresh and feeds the existing
`calibration_suggestions` signal on /home and /review.

## Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Approach | B: dedicated `/annotate` route | Matches user's stated intuition; smallest IA delta |
| Pre-labeling | ON by default, toggle to disable | Faster (~3s/item vs ~8s); fits SDK `annotation_flywheel` pattern |
| Multi-metric | All metrics per item | Denser data per session; matches how humans read an item |

## NOT in scope (explicitly deferred)

- Judge lifecycle reframe (Approach C: rename /metrics → /judges) — bigger product story; revisit when adding a 2nd new judge becomes a chore
- Multi-annotator UI (data model supports `annotator_id`; UI not built)
- External imports (Label Studio, Argilla, Prodigy)
- Gold-set pinning + drift detection
- Anchoring-bias telemetry (`used_ai_verdict %` dashboard)

## Architecture

### Page IA

```
                                   /annotate (NEW)
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
       ┌──────┴──────┐          ┌──────┴──────┐          ┌──────┴──────┐
       │ Pick source │  ─────▶  │ Annotate    │  ─────▶  │ Done        │
       │ (campaign)  │          │ (item N/M)  │          │ summary     │
       └──────┬──────┘          └──────┬──────┘          └─────────────┘
              │                        │
              │                        ├─ keyboard: 1/2/3 per metric
              │                        ├─ A = accept all AI verdicts
              │                        ├─ N = next, ← / → nav
              │                        ├─ progress bar + saved state
              │                        ├─ judge pre-label visible
       ┌──────┴───────┐                ├─ skip / undo / quit
       │ Sources:     │                └─ on each verdict: POST /verdict
       │ - Run        │                          + WS bumps cache
       │ - Dataset    │
       │ - Cluster    │
       │ - Custom IDs │
       └──────────────┘
```

### Routes

- `/annotate` — landing. Pick a source (run / dataset / failure cluster / paste IDs). Resume any in-flight session.
- `/annotate/:sessionId` — active session. Item N of M, keyboard nav, pre-label, save+exit any time.

### Item view (one item, all metrics)

```
  ┌─────────────────────────────────────────────────────┐
  │ Item #abc-123  (12 of 30)                           │
  ├─────────────────────────────────────────────────────┤
  │ INPUT:    What are gravitational waves?             │
  │ OUTPUT:   Gravitational waves are ripples in...     │
  ├─────────────────────────────────────────────────────┤
  │ [1]  helpfulness        [AI: pass]   ✓ ☓ -        │
  │ [2]  factuality         [AI: pass]   ✓ ☓ -        │
  │ [3]  source_attribution [AI: fail]   ✓ ☓ -        │
  ├─────────────────────────────────────────────────────┤
  │   A = accept all AI verdicts  •  N = next  •  ← →  │
  └─────────────────────────────────────────────────────┘
```

Per-metric keys: `1` cycles helpfulness through pass→fail→skip, `2` cycles factuality, etc. `A` accepts all AI pre-labels and advances. `N` advances if all metrics labeled. Pre-label ON pre-fills user's verdict to AI verdict; one keystroke to override.

### Storage layout

```
.evalyn/data/datasets/<ds>/
  ├── reviews/<run>.jsonl             # existing - per-run quick verdicts
  ├── annotations.jsonl               # NEW - canonical merged set
  └── annotation_sessions/
      ├── <session_id>.json           # NEW - session metadata + progress
      └── <session_id>.jsonl          # NEW - per-verdict event log (resumable)
```

Session JSON shape:
```json
{
  "id": "ann-20260503-1230_a1b2c3",
  "annotator_id": "user@host",
  "source_kind": "run|dataset|cluster|custom",
  "source_id": "20260330-012500_cd347c59",
  "metric_ids": ["helpfulness", "factuality", "source_attribution"],
  "items_total": 30,
  "items_done": 12,
  "items_skipped": 1,
  "started_at_iso": "...",
  "last_active_iso": "...",
  "status": "in_progress|completed|abandoned"
}
```

### Verdict POST shape (multi-metric)

```json
POST /api/v2/annotation/sessions/{id}/verdict
{
  "item_id": "abc-123",
  "labels": [
    { "metric_id": "helpfulness", "label": "pass", "used_ai_verdict": true },
    { "metric_id": "factuality", "label": "pass", "used_ai_verdict": true },
    { "metric_id": "source_attribution", "label": "fail", "used_ai_verdict": false, "note": "URL is real but doesn't say what the agent claims" }
  ],
  "skipped_metrics": []
}
```

`used_ai_verdict: bool` per metric is the anchoring-bias telemetry hook. Future iter could surface it as a per-session metric.

### Data flow

```
SOURCE PICKER ─▶ POST /api/v2/annotation/sessions
                  body: { source_kind, source_id, metric_ids?, annotator_id }
                  ─▶ creates session.json on disk
                  ─▶ returns session_id + first batch of items
                                │
                                ▼
ITEM N ─▶ user verdicts ─▶ POST /api/v2/annotation/sessions/{id}/verdict
                          body: { item_id, labels: [{metric_id, label, used_ai_verdict, note?}] }
                          ─▶ append to session.jsonl + reviews/<run>.jsonl
                          ─▶ session.json bumps progress
                          ─▶ broadcast cache_invalidate (review/queue + home)
                                │
                                ▼
SESSION COMPLETE ─▶ POST /api/v2/annotation/sessions/{id}/finalize
                    ─▶ merges into <ds>/annotations.jsonl (canonical)
                    ─▶ if any metric_id count >= 10 threshold:
                       calibration suggestion fires on /home and /review
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v2/annotation/sessions` | Create session (validates source + metric_ids) |
| GET | `/api/v2/annotation/sessions` | List in-progress sessions (for /annotate landing) |
| GET | `/api/v2/annotation/sessions/{id}` | Session metadata + progress |
| GET | `/api/v2/annotation/sessions/{id}/items?cursor=` | Paginated item batch with pre-labels |
| POST | `/api/v2/annotation/sessions/{id}/verdict` | Append verdict; idempotent on (session, item_id) |
| POST | `/api/v2/annotation/sessions/{id}/finalize` | Merge to annotations.jsonl; idempotent |
| DELETE | `/api/v2/annotation/sessions/{id}` | Abandon session (status flip; jsonl preserved) |

## Error & Rescue Map (no critical gaps)

| Codepath | Failure | Exception | Rescued | Action | User sees |
|---|---|---|---|---|---|
| POST /sessions | Source items missing | FileNotFoundError | Y | 404 | "Source <id> not found" inline |
| POST /sessions | metric_id not in any run | (logical guard) | Y | 422 | "Metric must be one of: ..." |
| POST /verdict | Stale session_id | (logical guard) | Y | 404 | redirect to /annotate |
| POST /verdict | Disk write fails | OSError | Y | 503 + log WARN | "Could not save - retry" |
| Resume | session jsonl corrupted line | json.JSONDecodeError | Y | log WARN, skip, continue | banner "N events skipped" |
| POST /finalize | annotations.jsonl write fails | OSError | Y | 503; session stays in_progress | "Retry - your verdicts are still saved" |
| WS broadcast | dead subscriber | Exception | Y (existing) | drop + log WARN | nothing |
| Frontend keyboard | repeated key fire | (logical guard) | Y | 250ms debounce | nothing |
| Frontend navigate-away | unsaved verdicts | beforeunload | Y | confirm prompt | "You have N unsaved verdicts" |

No `except Exception`. Every IO failure logged at WARN with path; client gets 4xx/5xx with detail.

## Edge cases

| Scenario | Handled |
|---|---|
| User mashes verdict key 10/sec | 250ms debounce; UI advances optimistically |
| Crash mid-write (jsonl > json) | replay events from jsonl, recompute progress on load |
| Source items changed since session start | banner "Source has changed - 2 items missing"; offer to abandon |
| Session abandoned mid-flight | status=abandoned after 24h idle; surfaces in /annotate as resumable |
| Custom IDs picker - one ID invalid | session creates with valid_count; banner lists invalid IDs |
| Two annotators, same source | both write to reviews/<run>.jsonl with annotator_id; conflicts surface in /review's existing disagreement detection |
| 1000 items, fatigue mid-flight | progress + save+resume; jsonl is append-only |
| User overrides AI pre-label | label captured + diff vs ai_label preserved (existing flywheel pattern) |

## Test plan

```
NEW BACKEND CODEPATHS:
  POST /api/v2/annotation/sessions          → tests: happy, source-missing, metric-missing
  POST /api/v2/annotation/sessions/{id}/verdict  → tests: happy, stale-session, disk-fail, idempotent
  POST /api/v2/annotation/sessions/{id}/finalize → tests: happy, finalize-twice, partial
  GET  /api/v2/annotation/sessions          → tests: list resumable, filter by source, sort by recency
  GET  /api/v2/annotation/sessions/{id}     → tests: 200 + 404
  GET  /api/v2/annotation/sessions/{id}/items?cursor → tests: pagination, last item

NEW FRONTEND CODEPATHS:
  Source picker form              → tests: validate + submit
  Keyboard handler (1/2/3/A/N/← →) → tests: each key, debounce, focus-in-textarea bypass
  Resume detection on /annotate   → tests: lists in-progress sessions, sorted
  Pre-label render                → tests: shows ai_label per metric, mismatch highlight
  Progress bar + saved state      → tests: persists across refresh
  Cache invalidate on verdict     → tests: review/queue + home refetch

NEW DATA FLOWS:
  verdict → session.jsonl (append)        → tested via 503 simulation
  verdict → reviews/<run>.jsonl (append)  → existing path, no new test
  finalize → annotations.jsonl (merge)    → idempotency + dedup tests
```

Estimated: **+15 backend tests, +8 frontend tests** → suite goes 260 → ~283.

## Observability

- `TimingMiddleware` already logs every `/api/*` request (iter 1).
- New per-session counter logged on verdict POST: `INFO ann.verdict session=X done=N/M metric=Y label=Z used_ai=bool`.
- New v2 cache key: `annotation/sessions` — broadcast on session create/finalize so source picker reflects in-progress state across tabs.
- Reviews jsonl format unchanged → existing `calibration_suggestions` detector picks up annotation verdicts automatically.

## Deployment & rollback

- Backend: 7 new endpoints under `/api/v2/annotation/*`. Net-new files; no migration.
- Frontend: 1 new route + nav item ("Annotate" between "Human review" and "Reports"). No existing route logic changes.
- Storage: NEW `annotation_sessions/` dir under each dataset. Created lazily on first session.
- Rollback: `git revert` the commit. Sessions on disk become orphaned (still readable; no impact on existing /review).

## SDK code to leverage (don't rebuild)

- `sdk/evalyn_sdk/annotation_ux.py` — `AnnotationItem`, `KeyAction` (keyboard model)
- `sdk/evalyn_sdk/annotation_session.py` — `AnnotationRecord` with annotator_id+confidence+skipped, save/resume
- `sdk/evalyn_sdk/calibration/annotation_flywheel.py` — `AnnotationItem(ai_label, human_label, agreement)`, `FlywheelState`
- `sdk/evalyn_sdk/pre_annotation.py` — pre-label with judge so humans confirm/correct
- `dashboard/evalyn_dashboard/api/v2/_shared.py::calibration_suggestions()` — existing closing-loop signal

## Implementation sequence (when picked up)

1. **Backend** (~2 hours CC time): new `dashboard/evalyn_dashboard/api/v2/annotation.py` module with 7 endpoints + sqlite-mirrored sessions in `.evalyn/data/datasets/<ds>/annotation_sessions/`. Wire into `server.py::_register_v2_routers`. Tests in `dashboard/tests/test_api_v2_annotation.py`.
2. **Frontend** (~2 hours CC time): new `dashboard/frontend/src/v2/routes/Annotate.tsx` (landing) + `AnnotateSession.tsx` (active session). New `dashboard/frontend/src/v2/api/annotation.ts` typed client. Add nav item between Human review and Reports. Wire WS cache-invalidate.
3. **Polish** (~30 min CC time): keyboard handler debounce; beforeunload confirm; banner for stale source / corrupted lines; mismatch highlight on pre-label.
4. **Tests** (~1 hour CC time): 15 BE + 8 FE per the test plan above.

Total estimate: **~5-6 hours focused CC time** for the full ship including tests + polish.

## Reviewer concerns

None. Zero critical gaps. Zero unresolved decisions. The design is ready for `/plan-eng-review` (when subagent capacity returns) or direct implementation.

## Resume instructions

When picking this back up:
1. Re-read this doc.
2. Verify SDK annotation infra hasn't changed: `find sdk -name "*annotation*" -newer docs/designs/annotation.md`.
3. Branch off main: `git checkout main && git pull && git checkout -b feat/dashboard-v3-annotate`.
4. Implement per the sequence above. Commit each step.
5. Optionally run `/plan-eng-review` first for an architecture pass.
