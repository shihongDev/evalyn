/**
 * JobsList — bottom-panel "Jobs" tab content.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx:468-497 (`JobsView`).
 *
 * Reads `store.jobs` (a Map keyed by job id), sorts most-recent first,
 * and renders one row per job with:
 *   id · command preview · status pill · elapsed/duration · actions
 *
 * For running jobs:
 *   - elapsed time is computed from `startedAt` and ticks every second
 *   - a Cancel button POSTs /api/jobs/{id}/cancel (api.cancelJob)
 *   - if `progress` is set we draw a thin accent bar underneath
 *
 * Click the ↗ button on any row to open the job's tab so the Terminal
 * shows just that job. The store's `addTab` is idempotent so repeat
 * clicks just activate the existing tab.
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import type { Job } from '../types/jobs';
import { api } from '../api';

const STATUS_LABEL: Record<Job['status'], string> = {
  pending: '… pending',
  running: '▸ running',
  complete: '✓ complete',
  failed: '✗ failed',
  cancelled: '⊘ cancelled',
};

const STATUS_CLASS: Record<Job['status'], string> = {
  pending: 'text-2',
  running: 'accent',
  complete: 'pass',
  failed: 'fail',
  cancelled: 'warn',
};

/** Truncate a long command for the table cell. */
const truncate = (s: string, max = 60): string =>
  s.length > max ? `${s.slice(0, max - 1)}…` : s;

/** Format milliseconds as M:SS (or H:MM:SS for very long jobs). */
const formatElapsed = (ms: number): string => {
  if (ms < 0) ms = 0;
  const total = Math.floor(ms / 1000);
  const s = total % 60;
  const m = Math.floor(total / 60) % 60;
  const h = Math.floor(total / 3600);
  const pad = (n: number) => n.toString().padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
};

/**
 * Per-second tick: returns Date.now() at most once per `intervalMs`.
 * The state update is what causes re-renders; the value itself is just
 * a "now" reference that downstream cells can subtract from `startedAt`.
 */
const useNow = (active: boolean, intervalMs = 1000): number => {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(t);
  }, [active, intervalMs]);
  return now;
};

interface JobRowProps {
  job: Job;
  /** "Now" reference for elapsed-time calc — keeps render pure. */
  now: number;
}

const JobRow = ({ job, now }: JobRowProps) => {
  const addTab = useStore((s) => s.addTab);
  const upsertJob = useStore((s) => s.upsertJob);

  const elapsed = (() => {
    if (job.duration) return job.duration;
    if (job.status === 'running' && job.startedAt) {
      return formatElapsed(now - job.startedAt);
    }
    return job.eta ?? '—';
  })();

  const onOpen = () => {
    addTab({
      id: `job:${job.id}`,
      title: job.cliId ?? job.id,
      kind: 'job',
      tone: job.status === 'failed' ? 'var(--fail)' : 'var(--accent)',
    });
  };

  const onCancel = async () => {
    try {
      await api.cancelJob(job.id);
      // Optimistic state update — the WS exit event will overwrite.
      upsertJob({ ...job, status: 'cancelled' });
    } catch (err) {
      // Surface the failure as a status note; backend WS will land truth.
      console.error('cancel failed', err);
    }
  };

  const cmdParts = job.cmd.split(' ');
  const head = cmdParts.slice(0, 2).join(' ');
  const tail = cmdParts.slice(2).join(' ');

  return (
    <div
      data-testid={`job-row-${job.id}`}
      style={{
        display: 'grid',
        gridTemplateColumns: '70px 1fr 110px 80px 80px',
        gap: 12,
        padding: '8px 0',
        borderBottom: '1px solid var(--line)',
        alignItems: 'center',
      }}
    >
      <span className="text-1 mono">{job.id}</span>
      <span className="text-2 mono" title={job.cmd} style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        <span className="accent">{head}</span>
        {tail && <span> {truncate(tail, 80)}</span>}
      </span>
      <span className={STATUS_CLASS[job.status]}>{STATUS_LABEL[job.status]}</span>
      <span className="text-3 mono">{elapsed}</span>
      <span style={{ display: 'flex', justifyContent: 'flex-end', gap: 4 }}>
        {job.status === 'running' && (
          <button
            type="button"
            className="btn ghost icon"
            title="Cancel job"
            aria-label={`Cancel job ${job.id}`}
            onClick={onCancel}
            style={{ width: 22, height: 22 }}
          >
            {'⊘'}
          </button>
        )}
        <button
          type="button"
          className="btn ghost icon"
          title="Open job tab"
          aria-label={`Open job ${job.id}`}
          onClick={onOpen}
          style={{ width: 22, height: 22 }}
        >
          {'↗'}
        </button>
      </span>
      {job.status === 'running' && job.progress != null && (
        <div style={{ gridColumn: '1 / -1', padding: '2px 0 4px' }}>
          <div
            className="bar"
            style={{
              height: 2,
              background: 'var(--bg-3)',
              borderRadius: 1,
              overflow: 'hidden',
            }}
          >
            <i
              style={{
                display: 'block',
                width: `${Math.max(0, Math.min(1, job.progress)) * 100}%`,
                height: '100%',
                background: 'var(--accent)',
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

const JobsList = () => {
  const jobs = useStore((s) => s.jobs);
  const hasRunning = Array.from(jobs.values()).some((j) => j.status === 'running');
  // Tick every second while any job is running so elapsed-time labels update.
  const now = useNow(hasRunning, 1000);

  // Most-recent first: by `startedAt`, falling back to insertion order.
  const rows = Array.from(jobs.values()).sort((a, b) => {
    const at = a.startedAt ?? 0;
    const bt = b.startedAt ?? 0;
    return bt - at;
  });

  return (
    <div className="mono" style={{ fontSize: 11, height: '100%', overflow: 'auto' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '70px 1fr 110px 80px 80px',
          gap: 12,
          padding: '4px 0',
          borderBottom: '1px solid var(--line)',
          color: 'var(--text-3)',
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}
      >
        <span>id</span>
        <span>command</span>
        <span>status</span>
        <span>elapsed</span>
        <span />
      </div>
      {rows.length === 0 ? (
        <div className="text-3" style={{ padding: '12px 0', fontSize: 11 }}>
          {'// no jobs yet — run a CLI from the catalog to start one'}
        </div>
      ) : (
        rows.map((job) => <JobRow key={job.id} job={job} now={now} />)
      )}
    </div>
  );
};

export default JobsList;
