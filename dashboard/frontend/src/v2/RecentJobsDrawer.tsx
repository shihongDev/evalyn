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
import { fetchJobStatus } from './api/jobs';
import {
  clearJobsHistory,
  loadJobsHistory,
  patchJob,
  subscribeJobsHistory,
  type JobHistoryEntry,
  type JobHistoryStatus,
} from './jobsHistory';
import { listCli, type CliSchema } from './api/cli';
import { openCliRunner } from './cliRunnerBridge';

interface RecentJobsDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function RecentJobsDrawer({ open, onClose }: RecentJobsDrawerProps): ReactElement | null {
  const [entries, setEntries] = useState<JobHistoryEntry[]>(() => loadJobsHistory());
  const [cliCatalog, setCliCatalog] = useState<CliSchema[] | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  // Re-render whenever history mutates (within this tab or another).
  useEffect(() => {
    return subscribeJobsHistory(() => setEntries(loadJobsHistory()));
  }, []);

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

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    function onKey(ev: KeyboardEvent) {
      if (ev.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
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
        <DrawerHeader onClose={onClose} count={entries.length} />
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
          ) : (
            entries.map((e) => (
              <JobRow key={e.job_id} entry={e} onClick={() => onRowClick(e)} />
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

function DrawerHeader({ onClose, count }: { onClose: () => void; count: number }) {
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
          }}
        >
          Recent jobs
          {count > 0 && (
            <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
              {count}
            </span>
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

interface JobRowProps {
  entry: JobHistoryEntry;
  onClick: () => void;
}

function JobRow({ entry, onClick }: JobRowProps) {
  const dim = entry.status === 'unknown';
  const pill = statusPillFor(entry.status);
  const rel = useMemo(() => relativeTime(entry.started_at_iso), [entry.started_at_iso]);

  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 10,
        width: '100%',
        padding: '10px 18px',
        background: 'transparent',
        border: 'none',
        borderBottom: `1px solid ${E.hair}`,
        textAlign: 'left',
        cursor: 'pointer',
        opacity: dim ? 0.6 : 1,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = E.panel2;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
      }}
      title={dim ? 'Backend evicted this job (server restart?)' : entry.job_id}
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
          {entry.exit_code !== undefined &&
            entry.exit_code !== null &&
            entry.status !== 'running' && (
              <>
                <span style={{ color: E.text4 }}>·</span>
                <span style={{ fontFamily: E.fMono }}>exit {entry.exit_code}</span>
              </>
            )}
        </div>
      </div>
      <Pill mono color={pill.color} bg={pill.bg}>
        {pill.label}
      </Pill>
    </button>
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
