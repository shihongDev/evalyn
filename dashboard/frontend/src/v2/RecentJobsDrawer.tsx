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
import { cancelJob, fetchJobStatus } from './api/jobs';
import {
  clearJobsHistory,
  getJobsDrawerFailureFilter,
  loadJobsHistory,
  patchJob,
  setFailureAckTime,
  setJobsDrawerFailureFilter,
  subscribeJobsHistory,
  type JobHistoryEntry,
  type JobHistoryStatus,
} from './jobsHistory';
import { listCli, previewCommand, type CliSchema } from './api/cli';
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
  }, [open]);

  const failedCount = useMemo(
    () => entries.reduce((n, e) => (e.status === 'failed' ? n + 1 : n), 0),
    [entries],
  );

  // Filtered view used by the row list. We deliberately exclude
  // cancelled and unknown from the failure filter: cancelled is
  // user-driven (not an unexpected failure), unknown means the backend
  // evicted the record (server restart). Surface only true failures.
  const visibleEntries = useMemo(
    () => (failureFilter ? entries.filter((e) => e.status === 'failed') : entries),
    [entries, failureFilter],
  );

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

  // One-shot per-open refresh of any row that LOOKS like it's still active.
  // Avoids polling - if the user wants newer data, they re-open the drawer.
  useEffect(() => {
    if (!open) return;
    const active = entries.filter(
      (e) => e.status === 'queued' || e.status === 'running',
    );
    if (active.length === 0) return;
    let cancelled = false;
    void (async () => {
      for (const e of active) {
        if (cancelled) return;
        const snap = await fetchJobStatus(e.job_id);
        if (cancelled) return;
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
    })();
    return () => {
      cancelled = true;
    };
    // We intentionally re-run only when `open` flips. The entries list changes
    // as we patch rows, but we don't want to re-poll on every patch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Close on Escape. Also restore focus to whatever element opened the
  // drawer so keyboard users land back on the trigger button instead
  // of <body>.
  useEffect(() => {
    if (!open) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    function onKey(ev: KeyboardEvent) {
      if (ev.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      if (
        prevFocus &&
        prevFocus !== document.body &&
        document.contains(prevFocus)
      ) {
        prevFocus.focus();
      }
    };
  }, [open, onClose]);

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

  // Re-run path: open the CliRunner pre-filled with the entry's args
  // but WITHOUT a resumeJobId, so the user lands on the form ready for
  // a fresh launch (one Run click away). Skipping the catalog stub
  // path here because re-run requires a real schema to render the
  // form - if the cli_id is gone from the catalog we just fall back
  // to the resume row click (which can show the cached output).
  const onRerun = (entry: JobHistoryEntry) => {
    const cli = cliCatalog?.find((c) => c.id === entry.cli_id);
    if (!cli) {
      onRowClick(entry);
      return;
    }
    openCliRunner(cli, {
      initialValues: entry.cli_args,
    });
    onClose();
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
        />
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
          ) : visibleEntries.length === 0 && failureFilter ? (
            <FilterEmptyState />
          ) : (
            visibleEntries.map((e) => (
              <JobRow
                key={e.job_id}
                entry={e}
                onClick={() => onRowClick(e)}
                onRerun={() => onRerun(e)}
                onCancel={() => onCancelRow(e)}
              />
            ))
          )}
        </div>
        <DrawerFooter
          hasEntries={entries.length > 0}
          activeCount={entries.filter(
            (e) => e.status === 'queued' || e.status === 'running',
          ).length}
          onClear={() => {
            clearJobsHistory();
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
}: {
  onClose: () => void;
  count: number;
  failedCount: number;
  failureFilter: boolean;
  onToggleFailureFilter: () => void;
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
        </div>
      </div>
      <button
        type="button"
        onClick={onClose}
        aria-label="Close"
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
        ×
      </button>
    </div>
  );
}

function DrawerFooter({
  hasEntries,
  activeCount,
  onClear,
}: {
  hasEntries: boolean;
  /** Count of rows whose status is still queued or running. We surface
   * this in the footer copy and gate clear behind a stronger confirm
   * when non-zero, since the localStorage entry is the only handle the
   * user has to re-attach mid-stream. */
  activeCount: number;
  onClear: () => void;
}) {
  // Two-click confirm. First click flips to "Confirm" for 4s; second
  // click within that window actually clears. Avoids a destructive
  // single-click action while keeping the button compact (no modal).
  const [armed, setArmed] = useState(false);
  const armedTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (armedTimerRef.current != null) {
        window.clearTimeout(armedTimerRef.current);
      }
    };
  }, []);

  const handleClick = () => {
    if (!armed) {
      setArmed(true);
      armedTimerRef.current = window.setTimeout(() => {
        setArmed(false);
        armedTimerRef.current = null;
      }, 4000);
      return;
    }
    if (armedTimerRef.current != null) {
      window.clearTimeout(armedTimerRef.current);
      armedTimerRef.current = null;
    }
    setArmed(false);
    onClear();
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
      <Btn
        kind={armed ? 'primary' : 'ghost'}
        size="sm"
        onClick={handleClick}
        disabled={!hasEntries}
        title={
          armed
            ? 'Click again to clear (auto-cancels in 4s)'
            : activeCount > 0
              ? `Clear ${activeCount} active + finished jobs from local history`
              : 'Clear all jobs from local history'
        }
      >
        {armed ? 'Confirm clear?' : 'Clear history'}
      </Btn>
    </div>
  );
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
}

function JobRow({ entry, onClick, onRerun, onCancel }: JobRowProps) {
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
      {canRerun && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRerun!();
          }}
          aria-label={`Re-run ${entry.cli_id} with the same arguments`}
          title="Re-run with the same args"
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

