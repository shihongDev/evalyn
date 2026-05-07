/**
 * RecentJobsDrawer - right-side slide-over listing the user's recent CLI
 * jobs. Backed by the localStorage ledger in :mod:`./jobsHistory` (the
 * backend's `/api/jobs/recent` is in-memory only and currently empty,
 * so we keep a client-side cache to make navigating-away-and-back from a
 * still-running job a recoverable action).
 *
 * Click handling:
 * - Click a still-running row: re-opens the global CliRunner in resume
 *   mode (`{resumeJobId}`), which reattaches the WS to the same job_id.
 * - Click a finished row: same behavior - the runner shows the final
 *   state pulled from history; user can hit Re-run to launch again.
 *
 * The drawer also performs a one-shot status refresh per visit for any
 * row whose stored status is `queued` or `running`. If the backend says
 * the job no longer exists (404), we patch the row to `unknown` and dim
 * it in the list. Polling stops there - we trust the user to re-open
 * the drawer if they want a fresh check.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import { E } from './tokens';
import { Btn, Eyebrow, Pill, StatusDot } from './ui';
import {
  cancelJob,
  fetchJobStatus,
  fetchJobsCapacity,
  restartJob,
  type JobsCapacity,
} from './api/jobs';
import { CapacityError } from './api/errors';
import {
  notificationPermission,
  requestNotificationPermission,
  type NotificationPermissionState,
} from './notifications';
import {
  clearJobsHistory,
  getJobsDrawerFailureFilter,
  loadJobsHistory,
  patchJob,
  pruneStaleUnknown,
  setFailureAckTime,
  setJobPinned,
  setJobsDrawerFailureFilter,
  subscribeJobsHistory,
  upsertJob,
  type JobHistoryEntry,
  type JobHistoryStatus,
} from './jobsHistory';
import { listCli, previewCommand, type CliSchema } from './api/cli';
import { useArmedConfirm } from './hooks/useArmedConfirm';
import { useLiveDuration } from './hooks/useLiveDuration';
import { openCliRunner } from './cliRunnerBridge';

interface RecentJobsDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function RecentJobsDrawer({ open, onClose }: RecentJobsDrawerProps): ReactElement | null {
  const [entries, setEntries] = useState<JobHistoryEntry[]>(() => loadJobsHistory());
  const [cliCatalog, setCliCatalog] = useState<CliSchema[] | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  // Capacity status (running / max_concurrent) for a small chip in the
  // header. Polled every 5s while the drawer is open; suppressed when
  // closed (no point spending bandwidth on a UI nobody is looking at).
  // Null until the first fetch returns or if the fetch ever fails -
  // we never block the UI on a stats fetch.
  const [capacity, setCapacity] = useState<JobsCapacity | null>(null);
  // "Show failed only" filter, toggled via a clickable badge in the
  // header. Lets a user spot recent regressions in a long history
  // (the local cap is 30 entries) without scanning each pill by eye.
  // Persisted to localStorage so users tracking regressions across
  // sessions stay in "failed only" mode after a page refresh.
  const [failureFilter, setFailureFilterState] = useState<boolean>(() =>
    getJobsDrawerFailureFilter(),
  );
  const setFailureFilter: typeof setFailureFilterState = (next) => {
    setFailureFilterState((prev) => {
      const resolved = typeof next === 'function' ? next(prev) : next;
      setJobsDrawerFailureFilter(resolved);
      return resolved;
    });
  };

  // Browser notifications permission. Read once on mount; update
  // after the user clicks "Enable notifications". When 'default',
  // the drawer header shows the enable affordance; once granted or
  // denied, the link disappears (re-prompting is browser-blocked
  // anyway, so showing it would be misleading).
  const [notifPerm, setNotifPerm] = useState<NotificationPermissionState>(
    () => notificationPermission(),
  );
  const onEnableNotifications = async () => {
    const result = await requestNotificationPermission();
    setNotifPerm(result);
  };

  // Re-render whenever history mutates (within this tab or another).
  useEffect(() => {
    return subscribeJobsHistory(() => setEntries(loadJobsHistory()));
  }, []);

  // Acknowledge currently-known failures whenever the drawer opens.
  // The AppShell tab title's "!N" prefix counts only failures whose
  // failed_at_iso is newer than this ack timestamp, so opening the
  // drawer clears the badge in this tab AND, via the ACK_KEY storage
  // event, in any other tab open on the same dashboard.
  useEffect(() => {
    if (!open) return;
    setFailureAckTime(Date.now());
    // Prune `unknown`-status entries older than 1 hour. They cannot be
    // recovered or resumed (the backend evicted the record - usually
    // a server restart), so they are pure clutter. The grace window
    // is generous enough that a quick dev-server restart won't wipe
    // a row the user might still want to inspect.
    pruneStaleUnknown(60 * 60 * 1000);
  }, [open]);

  const failedCount = useMemo(
    () => entries.reduce((n, e) => (e.status === 'failed' ? n + 1 : n), 0),
    [entries],
  );

  // Search filter state. The raw input is captured immediately so
  // typing feels responsive; the value used for filtering is debounced
  // 120ms so a long history isn't refiltered on every keystroke.
  // Substring match (case-insensitive) against cli_id + a JSON dump of
  // cli_args so "model=gpt-4o" and "compare" both find what you'd
  // expect. Empty / whitespace-only query treated as no filter.
  const [searchInput, setSearchInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  useEffect(() => {
    const handle = window.setTimeout(() => {
      setSearchQuery(searchInput.trim().toLowerCase());
    }, 120);
    return () => window.clearTimeout(handle);
  }, [searchInput]);
  const searchRef = useRef<HTMLInputElement | null>(null);

  // Filtered view used by the row list. Failure filter and search
  // filter AND-combine. We deliberately exclude cancelled and unknown
  // from the failure filter: cancelled is user-driven (not an
  // unexpected failure), unknown means the backend evicted the record
  // (server restart). Surface only true failures.
  const visibleEntries = useMemo(() => {
    let out = entries;
    if (failureFilter) out = out.filter((e) => e.status === 'failed');
    if (searchQuery) {
      out = out.filter((e) => {
        if (e.cli_id.toLowerCase().includes(searchQuery)) return true;
        // JSON.stringify is fine here - cli_args is a small object
        // serialized once per pass; the alternative (deep-walking
        // values) is more code with no real win.
        try {
          const dump = JSON.stringify(e.cli_args).toLowerCase();
          return dump.includes(searchQuery);
        } catch {
          return false;
        }
      });
    }
    return out;
  }, [entries, failureFilter, searchQuery]);

  // Auto-clear the filter when no failures remain so an accidental
  // toggle does not strand the user with an empty list once they
  // clear history or the failed jobs scroll out of the local cache.
  useEffect(() => {
    if (failureFilter && failedCount === 0) {
      setFailureFilter(false);
    }
  }, [failureFilter, failedCount]);

  // Fetch the CLI catalog once (cached) so we can resolve cli_id -> CliSchema
  // when the user clicks a row. We do it lazily on first open to avoid a
  // network call for users who never use the drawer.
  useEffect(() => {
    if (!open || cliCatalog || catalogError) return;
    let cancelled = false;
    void (async () => {
      try {
        const cat = await listCli();
        if (!cancelled) setCliCatalog(cat);
      } catch (e) {
        if (!cancelled) {
          setCatalogError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, cliCatalog, catalogError]);

  // Capacity poll: fetch /api/jobs/stats every 5s while the drawer is
  // open AND the tab is visible. Fires immediately on open so the chip
  // lights up without a visible delay; subsequent ticks pick up natural
  // changes (a job finishing, a fresh spawn). Closed drawer = no
  // polling. Backgrounded tab = no polling either - the chip is
  // invisible, so the user gains nothing from the fetch and the
  // dashboard burns network/CPU on always-open tabs. When the tab
  // returns to visible we fire an immediate refetch so the chip
  // reflects state-of-the-world without waiting up to 5s.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    let interval: number | null = null;

    const tick = async () => {
      const cap = await fetchJobsCapacity();
      if (!cancelled) setCapacity(cap);
    };

    const start = () => {
      if (interval !== null) return;
      void tick();
      interval = window.setInterval(() => void tick(), 5000);
    };
    const stop = () => {
      if (interval !== null) {
        window.clearInterval(interval);
        interval = null;
      }
    };

    if (typeof document === 'undefined' || document.visibilityState === 'visible') {
      start();
    }
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        start();
      } else {
        stop();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [open]);

  // Ref so the visibility refresh below sees the latest entries
  // without a re-binding closure. We can't put `entries` in the
  // useEffect deps - patching a row would re-run the effect, which
  // would patch again, etc. The ref breaks that loop.
  const entriesRef = useRef(entries);
  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  // Extracted refresh: fetch the canonical status for any row that
  // looks queued/running and patch the local mirror. Used by both
  // the per-open one-shot AND the visibility-change refresh below.
  // Cancellation flag is owned by the caller; we read it on every
  // iteration so a fast tab-switch doesn't spam patches into a
  // stale closure.
  const refreshActiveRows = async (
    isCancelled: () => boolean,
  ): Promise<void> => {
    const active = entriesRef.current.filter(
      (e) => e.status === 'queued' || e.status === 'running',
    );
    for (const e of active) {
      if (isCancelled()) return;
      const snap = await fetchJobStatus(e.job_id);
      if (isCancelled()) return;
      if (snap.kind === 'notFound') {
        patchJob(e.job_id, { status: 'unknown' });
      } else if (snap.kind === 'found') {
        patchJob(e.job_id, {
          status: snap.status,
          exit_code: snap.exit_code ?? undefined,
          duration: snap.duration != null ? String(snap.duration) : undefined,
        });
      }
    }
  };

  // One-shot per-open refresh of any row that LOOKS like it's still active.
  // Avoids polling - the visibility hook below picks up tab returns;
  // for foreground browsing we trust the user to re-open the drawer.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void refreshActiveRows(() => cancelled);
    return () => {
      cancelled = true;
    };
    // We intentionally re-run only when `open` flips. The entries list
    // is read via entriesRef so the effect doesn't re-fire per patch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Visibility-aware refresh: when the user comes back to the tab
  // after backgrounding it (e.g. for an hour while a long eval runs),
  // refresh any rows still in queued/running so the drawer reflects
  // reality. Without this, rows would stay stale until the user
  // re-opens the drawer or clicks somewhere. Drawer-closed = no work
  // (the per-open useEffect above will refresh on the next open).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const onVisibility = () => {
      if (
        typeof document !== 'undefined' &&
        document.visibilityState === 'visible'
      ) {
        void refreshActiveRows(() => cancelled);
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Close on Escape. Move focus into the drawer on open and restore
  // it on close. aria-modal alone doesn't move focus - keyboard
  // users were left with focus still on the trigger button behind
  // the drawer, so Tab walked them into the underlying page rather
  // than the drawer's own controls. The search input is the most
  // likely first action (matches CommandPalette's pattern of
  // focusing its query field on open).
  useEffect(() => {
    if (!open) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    // Defer one tick so the drawer is mounted before we focus into
    // it; without this, browsers occasionally skip the focus call
    // when the element hasn't been laid out yet.
    const focusTimer = window.setTimeout(() => {
      searchRef.current?.focus();
    }, 0);
    function onKey(ev: KeyboardEvent) {
      if (ev.key === 'Escape') {
        // Two-step Escape: clear search first, close drawer second.
        // Lets a user back out of a filter without losing the drawer
        // entirely - matches the cmd-palette / spotlight idiom.
        if (searchInput) {
          ev.preventDefault();
          setSearchInput('');
          return;
        }
        onClose();
        return;
      }
      // "/" focuses the search input when the drawer is open and the
      // user is not already in another input. Common power-user
      // shortcut from GitHub / Slack search.
      if (
        ev.key === '/' &&
        !ev.metaKey &&
        !ev.ctrlKey &&
        !ev.altKey
      ) {
        const tag = (ev.target as HTMLElement | null)?.tagName?.toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;
        ev.preventDefault();
        searchRef.current?.focus();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => {
      window.clearTimeout(focusTimer);
      window.removeEventListener('keydown', onKey);
      if (
        prevFocus &&
        prevFocus !== document.body &&
        document.contains(prevFocus)
      ) {
        prevFocus.focus();
      }
    };
  }, [open, onClose, searchInput]);

  const onRowClick = (entry: JobHistoryEntry) => {
    const cli = cliCatalog?.find((c) => c.id === entry.cli_id);
    if (!cli) {
      // Catalog hasn't loaded yet, or this cli_id is no longer in the
      // catalog. Build a minimal stub so the runner can still open and
      // show the cached output - the form path is locked behind hasJob
      // anyway in resume mode.
      const stub: CliSchema = {
        id: entry.cli_id,
        name: entry.cli_id,
        params: [],
      };
      openCliRunner(stub, {
        resumeJobId: entry.job_id,
        initialValues: entry.cli_args,
      });
    } else {
      openCliRunner(cli, {
        resumeJobId: entry.job_id,
        initialValues: entry.cli_args,
      });
    }
    onClose();
  };

  // Re-run path: one-click via the server's POST /api/jobs/{id}/restart
  // (added in tick 35). The server looks up the source's stored cli_id
  // + args, rebuilds argv from the canonical args dict (NOT the lossy
  // space-joined cmd string), and spawns a fresh subprocess. We
  // optimistically upsert a 'running' history entry so the new row
  // appears immediately; the runner / WS will fill in real status as
  // events arrive.
  //
  // On API failure (404 because cli_id no longer in catalog, 409
  // because source has no cli_id, network blip, etc.) we fall back to
  // the original behavior: open the runner pre-filled so the user can
  // either click Run as-is or tweak args first. This keeps the rare
  // "edit before rerun" affordance reachable without cluttering the row
  // with a second button.
  const onRerun = async (entry: JobHistoryEntry) => {
    try {
      const { job_id: newId } = await restartJob(entry.job_id);
      upsertJob({
        job_id: newId,
        cli_id: entry.cli_id,
        cli_args: entry.cli_args,
        started_at_iso: new Date().toISOString(),
        status: 'running',
        // Inherit pinned-ness from the source. A user who pinned a
        // reference run and then re-runs it almost certainly wants the
        // new spawn pinned too - the chain of "the run I keep around"
        // shouldn't break on every re-run.
        ...(entry.pinned ? { pinned: true } : {}),
      });
      onClose();
    } catch (err) {
      // Capacity 503: the right action is to retry, not to fall back
      // to the prefill form. Surface a clear banner with the cap and
      // the server's Retry-After hint.
      if (err instanceof CapacityError) {
        setActionMessage(
          `Job queue full (${err.running} / ${err.maxConcurrent} running). Try again in a few seconds.`,
        );
        return;
      }
      console.warn('restartJob failed; falling back to prefill', err);
      const cli = cliCatalog?.find((c) => c.id === entry.cli_id);
      if (!cli) {
        onRowClick(entry);
        return;
      }
      openCliRunner(cli, {
        initialValues: entry.cli_args,
      });
      onClose();
    }
  };

  // Bulk Re-run path: visible only when the failure filter is on AND
  // there is at least one failed entry to act on. Iterates the
  // currently-visible failures sequentially (NOT parallel) for two
  // reasons:
  //   1. The server's max_concurrent cap will 503 once we saturate;
  //      sequential keeps each request inside the previous one's
  //      capacity slot and lets the cap throttle naturally rather
  //      than firing N parallel 503s.
  //   2. The drawer history is a localStorage list - rapid parallel
  //      upserts can race on the SAME tab if React batches them in
  //      one tick. Sequential awaits sidestep that.
  // Any single failure (catalog drift, 409, network) is logged and
  // skipped; the loop continues so a partial success is better than
  // an all-or-nothing cliff.
  const [bulkRerunPending, setBulkRerunPending] = useState(false);
  // Transient banner for action feedback - currently only set when a
  // restart/bulk action hits the server's capacity cap (HTTP 503). We
  // don't fall back to the "open prefill form" branch in that case
  // because the user should retry the SAME action in a few seconds,
  // not edit args.
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  // Auto-clear the banner after a few seconds so it doesn't linger.
  useEffect(() => {
    if (!actionMessage) return;
    const t = window.setTimeout(() => setActionMessage(null), 6000);
    return () => window.clearTimeout(t);
  }, [actionMessage]);
  const onBulkRerunFailures = async () => {
    if (bulkRerunPending) return;
    const targets = visibleEntries.filter((e) => e.status === 'failed');
    if (targets.length === 0) return;
    setBulkRerunPending(true);
    let succeeded = 0;
    let hitCap = false;
    let capInfo: { running: number; max: number } | null = null;
    try {
      for (const entry of targets) {
        try {
          const { job_id: newId } = await restartJob(entry.job_id);
          upsertJob({
            job_id: newId,
            cli_id: entry.cli_id,
            cli_args: entry.cli_args,
            started_at_iso: new Date().toISOString(),
            status: 'running',
            // Inherit pinned-ness, same rationale as the single-row
            // onRerun above.
            ...(entry.pinned ? { pinned: true } : {}),
          });
          succeeded += 1;
        } catch (err) {
          // Sequential bulk: if we hit the capacity cap, the remaining
          // entries will hit the same 503. Stop early and surface the
          // partial-success count rather than spamming N identical
          // errors and burning the cap's Retry-After window.
          if (err instanceof CapacityError) {
            hitCap = true;
            capInfo = {
              running: err.running,
              max: err.maxConcurrent,
            };
            break;
          }
          console.warn('bulk restartJob failed for', entry.job_id, err);
        }
      }
    } finally {
      setBulkRerunPending(false);
    }
    if (hitCap && capInfo) {
      const remaining = targets.length - succeeded;
      setActionMessage(
        `Re-ran ${succeeded} / ${targets.length}; queue full (${capInfo.running} / ${capInfo.max} running). Retry the remaining ${remaining} in a few seconds.`,
      );
    }
    // After a successful bulk run, drop the failure filter so the
    // user lands back on the unfiltered view and can watch the new
    // running entries appear at the top. If nothing succeeded we
    // keep the filter so they can retry with full context.
    if (succeeded > 0) {
      setFailureFilter(false);
    }
  };

  // Cancel path: send SIGTERM via the existing /api/jobs/{id}/cancel
  // endpoint (with grace + SIGKILL escalation server-side) and
  // optimistically patch the local entry to 'cancelled'. If the CliRunner
  // is also open for this job, its WS will deliver an exit event - the
  // existing CliRunner onStatus handler will overwrite to 'failed'
  // (pre-existing behavior because SIGTERM produces a non-zero exit code
  // and the WS does not distinguish cancel from generic failure). When
  // the runner is NOT open, our optimistic patch is the only signal and
  // the row stays correctly labelled 'cancelled'.
  const onCancelRow = async (entry: JobHistoryEntry) => {
    try {
      await cancelJob(entry.job_id);
      patchJob(entry.job_id, { status: 'cancelled' });
    } catch (err) {
      // The request failed (network down, server gone, 404 because the
      // job already finished between click and HTTP). Trigger a snapshot
      // refresh so the row reflects whatever the server actually thinks.
      console.warn('cancelJob from drawer failed', err);
      const snap = await fetchJobStatus(entry.job_id);
      if (snap.kind === 'found') {
        patchJob(entry.job_id, {
          status: snap.status,
          exit_code: snap.exit_code ?? undefined,
          duration: snap.duration != null ? String(snap.duration) : undefined,
        });
      } else if (snap.kind === 'notFound') {
        patchJob(entry.job_id, { status: 'unknown' });
      }
    }
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Recent jobs"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(26,24,18,0.35)',
        zIndex: 880,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 360,
          maxWidth: 'calc(100vw - 24px)',
          height: '100%',
          background: E.panel,
          borderLeft: `1px solid ${E.hair2}`,
          boxShadow: '0 -10px 40px rgba(0,0,0,0.18)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <DrawerHeader
          onClose={onClose}
          count={entries.length}
          failedCount={failedCount}
          failureFilter={failureFilter}
          onToggleFailureFilter={() => setFailureFilter((v) => !v)}
          onBulkRerunFailures={onBulkRerunFailures}
          bulkRerunPending={bulkRerunPending}
          capacity={capacity}
          notifPerm={notifPerm}
          onEnableNotifications={onEnableNotifications}
        />
        {/* Thin search row. Tucked under the header rather than packed
            into the chip line so a long history with active filters
            stays readable. "/" focuses, Escape clears (or closes the
            drawer if the search is already empty). */}
        {entries.length > 0 && (
          <div
            style={{
              padding: '6px 14px 8px',
              borderBottom: `1px solid ${E.hair}`,
              flexShrink: 0,
            }}
          >
            <input
              ref={searchRef}
              type="search"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Filter by cli or args... (press / to focus)"
              aria-label="Filter recent jobs by cli or arg substring"
              style={{
                width: '100%',
                padding: '6px 10px',
                fontSize: 12,
                fontFamily: E.fMono,
                color: E.text0,
                background: E.panel2,
                border: `1px solid ${E.hair}`,
                borderRadius: 4,
                outline: 'none',
              }}
            />
            {searchQuery && (
              <div
                aria-live="polite"
                style={{
                  marginTop: 4,
                  fontSize: 10.5,
                  fontFamily: E.fMono,
                  color: E.text3,
                }}
              >
                {visibleEntries.length}{' '}
                {visibleEntries.length === 1 ? 'match' : 'matches'}
              </div>
            )}
          </div>
        )}
        {actionMessage && (
          <div
            role="status"
            aria-live="polite"
            style={{
              padding: '8px 18px',
              fontSize: 12,
              fontFamily: E.fMono,
              color: E.text1,
              background: 'rgba(217, 132, 51, 0.10)',
              borderBottom: `1px solid rgba(217, 132, 51, 0.3)`,
            }}
          >
            {actionMessage}
          </div>
        )}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            padding: '8px 0',
          }}
        >
          {entries.length === 0 ? (
            <EmptyState />
          ) : visibleEntries.length === 0 && (failureFilter || searchQuery) ? (
            <FilterEmptyState />
          ) : (
            renderRowsWithDayHeaders(visibleEntries, {
              onRowClick,
              onRerun,
              onCancelRow,
            })
          )}
        </div>
        <DrawerFooter
          hasEntries={entries.length > 0}
          activeCount={entries.filter(
            (e) => e.status === 'queued' || e.status === 'running',
          ).length}
          onExport={() => exportEntriesAsCsv(entries)}
          onClear={() => {
            clearJobsHistory();
          }}
          onCancelAll={async () => {
            const active = entries.filter(
              (e) => e.status === 'queued' || e.status === 'running',
            );
            // Fire all cancels in parallel; allSettled so one stale row's
            // 404 does not abort cancelling the rest. patch only on
            // fulfilled to avoid optimistic-cancelled-then-actually-failed
            // states.
            const results = await Promise.allSettled(
              active.map((e) => cancelJob(e.job_id)),
            );
            results.forEach((r, i) => {
              if (r.status === 'fulfilled') {
                patchJob(active[i].job_id, { status: 'cancelled' });
              }
            });
          }}
        />
      </div>
    </div>
  );
}

function DrawerHeader({
  onClose,
  count,
  failedCount,
  failureFilter,
  onToggleFailureFilter,
  onBulkRerunFailures,
  bulkRerunPending,
  capacity,
  notifPerm,
  onEnableNotifications,
}: {
  onClose: () => void;
  count: number;
  failedCount: number;
  failureFilter: boolean;
  onToggleFailureFilter: () => void;
  onBulkRerunFailures: () => void;
  bulkRerunPending: boolean;
  capacity: JobsCapacity | null;
  notifPerm: NotificationPermissionState;
  onEnableNotifications: () => void;
}) {
  return (
    <div
      style={{
        padding: '14px 18px',
        borderBottom: `1px solid ${E.hair}`,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexShrink: 0,
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <Eyebrow>Jobs</Eyebrow>
        <div
          style={{
            fontSize: 14,
            color: E.text0,
            marginTop: 2,
            fontWeight: 500,
            display: 'flex',
            alignItems: 'baseline',
            gap: 8,
            flexWrap: 'wrap',
          }}
        >
          Recent jobs
          {count > 0 && (
            <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
              {count}
            </span>
          )}
          {capacity &&
            capacity.max_concurrent > 0 &&
            capacity.running > 0 && (
              <span
                aria-label={`${capacity.running} of ${capacity.max_concurrent} concurrent slots in use`}
                title={
                  capacity.running >= capacity.max_concurrent
                    ? `At capacity: ${capacity.running} / ${capacity.max_concurrent} running. New runs will queue.`
                    : `${capacity.running} / ${capacity.max_concurrent} running`
                }
                style={{
                  fontSize: 10.5,
                  fontFamily: E.fMono,
                  padding: '0 8px',
                  borderRadius: 4,
                  lineHeight: 1.6,
                  whiteSpace: 'nowrap',
                  // Highlight when at the cap so the user notices the
                  // throttling reason for any "queue full" banner that
                  // surfaces in this drawer.
                  color:
                    capacity.running >= capacity.max_concurrent
                      ? E.ember
                      : E.text2,
                  background:
                    capacity.running >= capacity.max_concurrent
                      ? 'rgba(217, 132, 51, 0.16)'
                      : 'rgba(255, 255, 255, 0.04)',
                  border: `1px solid ${
                    capacity.running >= capacity.max_concurrent
                      ? 'rgba(217, 132, 51, 0.45)'
                      : E.hair
                  }`,
                }}
              >
                {capacity.running} / {capacity.max_concurrent} running
              </span>
            )}
          {failedCount > 0 && (
            <button
              type="button"
              onClick={onToggleFailureFilter}
              aria-pressed={failureFilter}
              aria-label={
                failureFilter
                  ? `Showing only ${failedCount} failed job${failedCount === 1 ? '' : 's'}; click to show all`
                  : `${failedCount} failed job${failedCount === 1 ? '' : 's'}; click to filter`
              }
              title={
                failureFilter
                  ? 'Showing failed only - click to show all'
                  : `Filter to ${failedCount} failed job${failedCount === 1 ? '' : 's'}`
              }
              style={{
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.fail,
                background: failureFilter
                  ? 'rgba(231, 102, 83, 0.28)'
                  : 'rgba(231, 102, 83, 0.12)',
                border: `1px solid rgba(231, 102, 83, ${failureFilter ? 0.7 : 0.3})`,
                borderRadius: 4,
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: failureFilter ? 600 : 500,
                cursor: 'pointer',
              }}
            >
              {failureFilter ? '✓ ' : ''}
              {failedCount} failed
            </button>
          )}
          {failureFilter && failedCount > 0 && (
            <button
              type="button"
              onClick={onBulkRerunFailures}
              disabled={bulkRerunPending}
              aria-label={`Re-run all ${failedCount} visible failed job${failedCount === 1 ? '' : 's'}`}
              title={
                bulkRerunPending
                  ? 'Re-running'
                  : `Re-run all ${failedCount} visible failures`
              }
              style={{
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: bulkRerunPending ? E.text3 : E.ember,
                background: 'transparent',
                border: `1px solid ${bulkRerunPending ? E.hair : 'rgba(217, 132, 51, 0.4)'}`,
                borderRadius: 4,
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: 500,
                cursor: bulkRerunPending ? 'progress' : 'pointer',
              }}
            >
              {bulkRerunPending
                ? `Re-running ${failedCount}...`
                : `Re-run all ${failedCount}`}
            </button>
          )}
        </div>
      </div>
      {notifPerm === 'default' && (
        <button
          type="button"
          onClick={onEnableNotifications}
          aria-label="Enable browser notifications for backgrounded jobs"
          title="Get an OS notification when a job finishes while you are on another tab"
          style={{
            flexShrink: 0,
            padding: '0 8px',
            fontFamily: E.fMono,
            fontSize: 10.5,
            color: E.text2,
            background: 'transparent',
            border: `1px solid ${E.hair2}`,
            borderRadius: 4,
            cursor: 'pointer',
            lineHeight: 1.6,
            whiteSpace: 'nowrap',
            marginRight: 6,
          }}
        >
          Enable notifications
        </button>
      )}
      <button
        type="button"
        onClick={onClose}
        aria-label="Close recent jobs"
        title="Close"
        style={{
          width: 28,
          height: 28,
          borderRadius: 6,
          background: 'transparent',
          border: `1px solid ${E.hair2}`,
          color: E.text2,
          cursor: 'pointer',
          fontSize: 14,
          lineHeight: 1,
        }}
      >
        <span aria-hidden="true">×</span>
      </button>
    </div>
  );
}

function DrawerFooter({
  hasEntries,
  activeCount,
  onExport,
  onClear,
  onCancelAll,
}: {
  hasEntries: boolean;
  /** Count of rows whose status is still queued or running. We surface
   * this in the footer copy and gate clear behind a stronger confirm
   * when non-zero, since the localStorage entry is the only handle the
   * user has to re-attach mid-stream. */
  activeCount: number;
  /** Export the local drawer history as a CSV download. */
  onExport: () => void;
  onClear: () => void;
  /** Bulk cancel handler. Iterates active rows and sends SIGTERM to
   * each via /api/jobs/{id}/cancel. Wrapped in two-click confirm. */
  onCancelAll: () => void | Promise<void>;
}) {
  // Two independent two-click confirm flows - one for clear, one for
  // cancel-all. Both share the same 4s arm window. Migrated from a
  // hand-rolled state pair + cleanup effect to the shared
  // useArmedConfirm hook (the hook's docstring even named this site
  // as a known caller, but the actual implementation hadn't migrated).
  const clearArm = useArmedConfirm();
  const cancelArm = useArmedConfirm();
  const [cancellingAll, setCancellingAll] = useState(false);

  const handleClick = () => {
    if (!clearArm.armed) {
      clearArm.arm();
      return;
    }
    clearArm.reset();
    onClear();
  };

  const handleCancelAllClick = async () => {
    if (cancellingAll) return;
    if (!cancelArm.armed) {
      cancelArm.arm();
      return;
    }
    cancelArm.reset();
    setCancellingAll(true);
    try {
      await onCancelAll();
    } finally {
      setCancellingAll(false);
    }
  };

  const footerText =
    activeCount > 0
      ? `${activeCount} job${activeCount === 1 ? '' : 's'} still running - clearing won't cancel them`
      : 'Local history only - clearing does not cancel running jobs';

  return (
    <div
      style={{
        padding: '10px 18px',
        borderTop: `1px solid ${E.hair}`,
        background: E.panel2,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <span
        style={{
          flex: 1,
          fontSize: 11,
          color: activeCount > 0 ? E.warn : E.text3,
          lineHeight: 1.4,
        }}
      >
        {footerText}
      </span>
      {activeCount > 0 && (
        <Btn
          kind={cancelArm.armed ? 'primary' : 'ghost'}
          size="sm"
          onClick={() => void handleCancelAllClick()}
          disabled={cancellingAll}
          title={
            cancelArm.armed
              ? 'Click again to send SIGTERM to all active jobs'
              : `Cancel all ${activeCount} active job${activeCount === 1 ? '' : 's'}`
          }
        >
          {cancellingAll
            ? 'Cancelling...'
            : cancelArm.armed
              ? 'Confirm cancel all?'
              : `Cancel ${activeCount}`}
        </Btn>
      )}
      <Btn
        kind="ghost"
        size="sm"
        onClick={onExport}
        disabled={!hasEntries}
        title="Download the local drawer history as CSV"
      >
        Export CSV
      </Btn>
      <Btn
        kind={clearArm.armed ? 'primary' : 'ghost'}
        size="sm"
        onClick={handleClick}
        disabled={!hasEntries}
        title={
          clearArm.armed
            ? 'Click again to clear (auto-cancels in 4s)'
            : activeCount > 0
              ? `Clear ${activeCount} active + finished jobs from local history`
              : 'Clear all jobs from local history'
        }
      >
        {clearArm.armed ? 'Confirm clear?' : 'Clear history'}
      </Btn>
    </div>
  );
}

/** Serialize a single CSV cell. Wraps in quotes when needed (commas,
 * quotes, newlines) and doubles internal quotes per RFC 4180. Empty /
 * undefined values become empty cells (no surrounding quotes). */
function csvCell(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (s === '') return '';
  if (/[",\r\n]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

/** Build a CSV text body from the drawer's history entries.
 * Newest-first to match the drawer's visual order. */
function buildEntriesCsv(entries: JobHistoryEntry[]): string {
  const headers = [
    'job_id',
    'cli_id',
    'cli_args_json',
    'status',
    'exit_code',
    'started_at_iso',
    'failed_at_iso',
    'duration',
    'stderr_count',
    'pinned',
  ];
  const lines: string[] = [headers.join(',')];
  for (const e of entries) {
    lines.push(
      [
        csvCell(e.job_id),
        csvCell(e.cli_id),
        csvCell(JSON.stringify(e.cli_args ?? {})),
        csvCell(e.status),
        csvCell(e.exit_code ?? ''),
        csvCell(e.started_at_iso),
        csvCell(e.failed_at_iso ?? ''),
        csvCell(e.duration ?? ''),
        csvCell(e.stderr_count ?? ''),
        csvCell(e.pinned ? 'true' : 'false'),
      ].join(','),
    );
  }
  return lines.join('\n') + '\n';
}

/** Trigger a CSV download for the local drawer history. Browser
 * blob-URL pattern matches what CliRunner's "Download .log" uses. */
function exportEntriesAsCsv(entries: JobHistoryEntry[]): void {
  if (entries.length === 0) return;
  const csv = buildEntriesCsv(entries);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const a = document.createElement('a');
  a.href = url;
  a.download = `evalyn-jobs-${ts}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer revoke to give the browser a tick to start the download;
  // matches the pattern used in CliRunner's handleDownload.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function EmptyState() {
  return (
    <div
      style={{
        padding: '32px 18px',
        textAlign: 'center',
        color: E.text3,
        fontSize: 12.5,
        lineHeight: 1.5,
      }}
    >
      No recent jobs yet. Jobs you launch from the CLI runner will appear here.
    </div>
  );
}

function FilterEmptyState() {
  // Reachable only transiently: the auto-clear effect resets the filter
  // when failedCount drops to 0. This state is shown briefly during the
  // render between the toggle and the effect firing if the user races
  // ahead of React. Kept defensive so we never render an empty list
  // beneath an active filter chip.
  return (
    <div
      style={{
        padding: '32px 18px',
        textAlign: 'center',
        color: E.text3,
        fontSize: 12.5,
        lineHeight: 1.5,
      }}
    >
      No failed jobs match. Toggle the chip above to show all jobs.
    </div>
  );
}

/** Compute the group-header label for an entry. Pinned entries all
 * share a single "Pinned" group regardless of when they ran; unpinned
 * entries fall into "Today", "Yesterday", or a date label.
 *
 * The date formatting drops the year when it matches the current
 * year so the common case is short. A run from a previous year shows
 * "Mar 5, 2025" so cross-year ambiguity is impossible. */
function dayHeaderLabel(entry: JobHistoryEntry, now: Date): string {
  if (entry.pinned) return '★ Pinned';
  const startedAt = new Date(entry.started_at_iso);
  if (!Number.isFinite(startedAt.getTime())) return 'Earlier';
  const startedDay = new Date(
    startedAt.getFullYear(),
    startedAt.getMonth(),
    startedAt.getDate(),
  );
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const dayDiff = Math.round(
    (today.getTime() - startedDay.getTime()) / 86_400_000,
  );
  if (dayDiff === 0) return 'Today';
  if (dayDiff === 1) return 'Yesterday';
  const sameYear = startedAt.getFullYear() === now.getFullYear();
  return new Intl.DateTimeFormat(
    'en',
    sameYear
      ? { month: 'short', day: 'numeric' }
      : { month: 'short', day: 'numeric', year: 'numeric' },
  ).format(startedAt);
}

/** Render the entry list with section dividers when the day-group
 * label changes. Sections respect the existing pinned-first sort -
 * "Pinned" is first when there are any pinned entries, then date
 * groups in newest-first order. */
function renderRowsWithDayHeaders(
  entries: JobHistoryEntry[],
  handlers: {
    onRowClick: (e: JobHistoryEntry) => void;
    onRerun: (e: JobHistoryEntry) => void;
    onCancelRow: (e: JobHistoryEntry) => void | Promise<void>;
  },
): ReactElement[] {
  const out: ReactElement[] = [];
  const now = new Date();
  let lastLabel: string | null = null;
  for (const e of entries) {
    const label = dayHeaderLabel(e, now);
    if (label !== lastLabel) {
      out.push(<DayHeader key={`h-${label}-${e.job_id}`} label={label} />);
      lastLabel = label;
    }
    out.push(
      <JobRow
        key={e.job_id}
        entry={e}
        onClick={() => handlers.onRowClick(e)}
        onRerun={() => handlers.onRerun(e)}
        onCancel={() => handlers.onCancelRow(e)}
        onTogglePin={() => setJobPinned(e.job_id, !e.pinned)}
      />,
    );
  }
  return out;
}

function DayHeader({ label }: { label: string }): ReactElement {
  return (
    <div
      style={{
        padding: '8px 18px 4px',
        fontFamily: E.fMono,
        fontSize: 10,
        textTransform: 'uppercase',
        letterSpacing: 0.4,
        color: E.text3,
        background: 'transparent',
        borderTop: `1px solid ${E.hair}`,
      }}
    >
      {label}
    </div>
  );
}

interface JobRowProps {
  entry: JobHistoryEntry;
  onClick: () => void;
  /** "Re-run" handler: open the runner with this row's args pre-filled
   * but no resumeJobId, so the user gets the form ready for a fresh
   * launch. Hidden for running/queued rows (they already have a live
   * job) and for the unknown-eviction state. */
  onRerun?: () => void;
  /** "Cancel" handler: send SIGTERM via the existing /api/jobs/{id}/cancel
   * endpoint and optimistically patch the local entry. Hidden for
   * terminal rows (no live job to kill). */
  onCancel?: () => void | Promise<void>;
  /** Toggle the pinned state. Pinned entries survive the history cap
   * and sort to the top of the drawer. Always available (unlike
   * rerun/cancel which are status-gated). */
  onTogglePin?: () => void;
}

function JobRow({ entry, onClick, onRerun, onCancel, onTogglePin }: JobRowProps) {
  const dim = entry.status === 'unknown';
  const pill = statusPillFor(entry.status);
  const rel = useMemo(() => relativeTime(entry.started_at_iso), [entry.started_at_iso]);
  // Build the full argv preview ("evalyn cli-id --flag value ...") for the
  // row tooltip so users can disambiguate similar runs without opening
  // them. Mirrors the preview command CliRunner shows in its header.
  const argvPreview = useMemo(
    () => previewCommand(entry.cli_id, entry.cli_args),
    [entry.cli_id, entry.cli_args],
  );
  // Args-only summary for the row's secondary line. Strips the
  // "evalyn <cli_id> " prefix the preview command always emits, so a
  // row already showing the cli_id as its primary line doesn't repeat
  // it. Empty when the user ran the command with no flags - we just
  // hide the line in that case rather than render a useless empty
  // placeholder. This closes the visibility gap where 5 "compare" or
  // "run-eval" rows looked identical without hovering each.
  const argsLine = useMemo(() => {
    const prefix = `evalyn ${entry.cli_id} `;
    return argvPreview.startsWith(prefix)
      ? argvPreview.slice(prefix.length).trim()
      : '';
  }, [argvPreview, entry.cli_id]);
  // Re-run only makes sense for terminal rows the user can rerun. We
  // hide it for queued/running (live job already exists) and for
  // unknown (backend lost the record - the cli_args we cached should
  // still work, but the status is suspect, so skip rather than risk
  // surprising the user with an unexpected fresh launch).
  const canRerun =
    onRerun != null &&
    (entry.status === 'complete' ||
      entry.status === 'failed' ||
      entry.status === 'cancelled');
  const canCancel =
    onCancel != null && (entry.status === 'queued' || entry.status === 'running');
  // Live "running for Ns" counter for queued/running rows. We tick a
  // local state every second so the metadata visibly counts up, which
  // makes a long-running eval feel responsive even though no other UI
  // updates fire while the user is parked on the drawer. Hides on
  // terminal status (the real duration field takes over).
  const isLive = entry.status === 'queued' || entry.status === 'running';
  const liveDuration = useLiveDuration(entry.started_at_iso, isLive);
  // Single-click cancel matches the CliRunner's own Cancel button for
  // consistency. Disable while in-flight so a double-click does not
  // fire two cancel HTTPs (the second would 404 once the first
  // succeeded; harmless but noisy in console).
  const [cancelling, setCancelling] = useState(false);
  const handleCancel = async () => {
    if (!onCancel || cancelling) return;
    setCancelling(true);
    try {
      await onCancel();
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'stretch',
        width: '100%',
        borderBottom: `1px solid ${E.hair}`,
        opacity: dim ? 0.6 : 1,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = E.panel2;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
      }}
    >
      <button
        type="button"
        onClick={onClick}
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 10,
          flex: 1,
          minWidth: 0,
          padding: '10px 18px',
          background: 'transparent',
          border: 'none',
          textAlign: 'left',
          cursor: 'pointer',
          color: 'inherit',
        }}
        title={
          dim
            ? 'Backend evicted this job (server restart?)'
            : argvPreview
        }
      >
        <div style={{ paddingTop: 4, flexShrink: 0 }}>
          <StatusDot
            status={dotStatusFor(entry.status)}
            animated={entry.status === 'running'}
            label={`Job status: ${entry.status}`}
          />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 12.5,
              color: E.text0,
              fontWeight: 500,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {entry.cli_id}
          </div>
          {argsLine && (
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 11,
                color: E.text3,
                marginTop: 2,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {argsLine}
            </div>
          )}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginTop: 3,
              fontSize: 11,
              color: E.text3,
            }}
          >
            <span>{rel}</span>
            {entry.duration && (
              <>
                <span style={{ color: E.text4 }}>·</span>
                <span style={{ fontFamily: E.fMono }}>
                  {formatDuration(entry.duration)}
                </span>
              </>
            )}
            {liveDuration != null && (
              <>
                <span style={{ color: E.text4 }}>·</span>
                <span style={{ fontFamily: E.fMono, color: E.ember }}>
                  running {liveDuration}
                </span>
              </>
            )}
            {entry.exit_code !== undefined &&
              entry.exit_code !== null &&
              entry.status !== 'running' && (
                <>
                  <span style={{ color: E.text4 }}>·</span>
                  <span style={{ fontFamily: E.fMono }}>exit {entry.exit_code}</span>
                </>
              )}
            {entry.stderr_count != null && entry.stderr_count > 0 && (
              <>
                <span style={{ color: E.text4 }}>·</span>
                <span style={{ fontFamily: E.fMono, color: '#c47766' }}>
                  {entry.stderr_count} stderr
                </span>
              </>
            )}
          </div>
        </div>
        <Pill mono color={pill.color} bg={pill.bg}>
          {pill.label}
        </Pill>
      </button>
      {onTogglePin && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onTogglePin();
          }}
          aria-label={
            entry.pinned
              ? `Unpin ${entry.cli_id} from history`
              : `Pin ${entry.cli_id} to keep it in history`
          }
          aria-pressed={Boolean(entry.pinned)}
          title={
            entry.pinned
              ? 'Pinned: kept in history. Click to unpin.'
              : 'Pin to history (survives the cap)'
          }
          style={{
            flexShrink: 0,
            width: 32,
            border: 'none',
            background: 'transparent',
            // Filled star = pinned; dim outline = unpinned. We use the
            // SAME glyph slot in both states (different unicode chars)
            // so the row layout never shifts when the user toggles.
            color: entry.pinned ? E.ember : E.text4,
            cursor: 'pointer',
            fontSize: 13,
            lineHeight: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = entry.pinned ? E.ember : E.text1;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = entry.pinned ? E.ember : E.text4;
          }}
        >
          {entry.pinned ? '★' : '☆'}
        </button>
      )}
      {canRerun && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRerun!();
          }}
          aria-label={`Re-run ${entry.cli_id} with the same arguments`}
          title="Re-run with the same args (instant)"
          style={{
            flexShrink: 0,
            width: 36,
            border: 'none',
            background: 'transparent',
            color: E.text3,
            cursor: 'pointer',
            fontSize: 14,
            lineHeight: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            // Brighten on hover so it reads as actionable, distinct
            // from the row's own (panel2) hover background.
            e.currentTarget.style.color = E.text0;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = E.text3;
          }}
        >
          ↻
        </button>
      )}
      {canCancel && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            void handleCancel();
          }}
          disabled={cancelling}
          aria-label={`Cancel ${entry.cli_id}`}
          title={cancelling ? 'Cancelling...' : 'Cancel this job (SIGTERM)'}
          style={{
            flexShrink: 0,
            width: 36,
            border: 'none',
            background: 'transparent',
            color: cancelling ? E.text4 : E.fail,
            cursor: cancelling ? 'wait' : 'pointer',
            fontSize: 14,
            lineHeight: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: cancelling ? 0.5 : 1,
          }}
          onMouseEnter={(e) => {
            if (!cancelling) e.currentTarget.style.opacity = '0.85';
          }}
          onMouseLeave={(e) => {
            if (!cancelling) e.currentTarget.style.opacity = '1';
          }}
        >
          ✕
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusPillFor(status: JobHistoryStatus): {
  color: string;
  bg: string;
  label: string;
} {
  switch (status) {
    case 'queued':
      return { color: E.text2, bg: E.panel3, label: 'queued' };
    case 'running':
      return { color: E.ember, bg: E.emberDim, label: 'running' };
    case 'complete':
      return { color: E.pass, bg: E.passDim, label: 'done' };
    case 'failed':
      return { color: E.fail, bg: E.failDim, label: 'failed' };
    case 'cancelled':
      return { color: E.text2, bg: E.panel3, label: 'cancelled' };
    case 'unknown':
      return { color: E.text3, bg: E.panel3, label: 'unknown' };
  }
}

function dotStatusFor(status: JobHistoryStatus): string {
  switch (status) {
    case 'complete':
      return 'pass';
    case 'failed':
      return 'fail';
    case 'running':
      return 'running';
    case 'queued':
      return 'info';
    case 'cancelled':
    case 'unknown':
    default:
      return 'idle';
  }
}

/** Compact relative-time string for row metadata. */
function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const dSec = (Date.now() - t) / 1000;
  if (dSec < 5) return 'just now';
  if (dSec < 60) return `${Math.floor(dSec)}s ago`;
  const min = dSec / 60;
  if (min < 60) return `${Math.floor(min)}m ago`;
  const hr = min / 60;
  if (hr < 24) return `${Math.floor(hr)}h ago`;
  const day = hr / 24;
  if (day < 7) return `${Math.floor(day)}d ago`;
  return new Date(t).toLocaleDateString();
}

/** Format the stored duration ("12.4" -> "12.4s"). Pass-through for other shapes. */
function formatDuration(d: string): string {
  const n = Number(d);
  if (Number.isFinite(n)) {
    if (n < 60) return `${n.toFixed(1)}s`;
    const m = Math.floor(n / 60);
    const s = Math.floor(n % 60);
    return `${m}m${s.toString().padStart(2, '0')}s`;
  }
  return d;
}

