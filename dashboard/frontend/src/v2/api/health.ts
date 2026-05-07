/**
 * `/api/health` client. Returns process + capacity + persistence
 * snapshot. Used by Settings to render the SystemStatus card so
 * operators can spot DB bloat before it bites and verify uptime
 * after a hot-restart.
 *
 * Like `fetchJobsCapacity`, this is a nice-to-have indicator and
 * NEVER load-bearing - failures resolve to null and the caller
 * hides the surface rather than surfacing an error toast. The
 * health endpoint must always 200 (best-effort fields fall through
 * to 0 server-side), so a null here means a network failure or a
 * malformed body, both of which the user shouldn't be alerted to
 * just because they opened Settings.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';

export interface SystemHealth {
  ok: boolean;
  version: string;
  started_at: number;
  uptime_seconds: number;
  running: number;
  max_concurrent: number;
  agent_threads: number;
  agent_open_threads: number;
  jobs_persisted: number;
  jobs_db_bytes: number;
  /** Unix epoch float of the last successful VACUUM, or null when
   * the server hasn't vacuumed yet in this process (vacuum is
   * triggered by the shutdown hook, not steady-state). */
  last_vacuum_at: number | null;
  /** Failed jobs in the last 24h. Read from the persistence
   * `stats()` call (cheap due to the TTL cache shared with the
   * `jobs_persisted` count). Useful as an "are things going
   * wrong?" glance metric on the SystemStatusCard. */
  recent_failures_24h: number;
}

const NUMBER_FIELDS = [
  'started_at',
  'uptime_seconds',
  'running',
  'max_concurrent',
  'agent_threads',
  'agent_open_threads',
  'jobs_persisted',
  'jobs_db_bytes',
  'recent_failures_24h',
] as const;

function isSystemHealth(body: unknown): body is SystemHealth {
  if (typeof body !== 'object' || body === null) return false;
  const b = body as Record<string, unknown>;
  if (typeof b.ok !== 'boolean') return false;
  if (typeof b.version !== 'string') return false;
  for (const k of NUMBER_FIELDS) {
    if (typeof b[k] !== 'number') return false;
  }
  // last_vacuum_at is `number | null`. Reject anything else (e.g.
  // a stringified date) so a malformed body hides the card via
  // `null` rather than rendering "-" through the formatter.
  const lv = b.last_vacuum_at;
  if (lv !== null && typeof lv !== 'number') return false;
  return true;
}

export async function fetchSystemHealth(): Promise<SystemHealth | null> {
  try {
    const res = await fetch('/api/health', {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) return null;
    const body = (await res.json()) as unknown;
    return isSystemHealth(body) ? body : null;
  } catch {
    return null;
  }
}

/** Result of POST /api/jobs/admin/vacuum. ``bytes_saved`` is
 * before - after; can be 0 or negative when nothing was reclaimed
 * (already-vacuumed db). The FE renders "(no change)" for <=0
 * rather than showing a misleading positive. */
export interface VacuumResult {
  ok: boolean;
  before: number;
  after: number;
  bytes_saved: number;
}

/** Trigger a manual VACUUM on the persistence db. Throws on
 * non-2xx. Self-heals stale CSRF tokens after a server restart
 * (matches the cancelJob retry pattern in api/jobs.ts). */
export async function vacuumPersistence(): Promise<VacuumResult> {
  const send = async (token: string | null): Promise<Response> =>
    fetch('/api/jobs/admin/vacuum', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        ...(token ? { 'X-CSRF-Token': token } : {}),
      },
    });
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /api/jobs/admin/vacuum ${res.status}: ${text}`);
  }
  return (await res.json()) as VacuumResult;
}

/** Result of POST /api/jobs/admin/prune. ``deleted`` is the
 * rowcount that got pruned; ``kept`` is the post-prune total
 * from stats() so the UI can show the new persisted count
 * without a separate health refetch. */
export interface PruneResult {
  ok: boolean;
  deleted: number;
  kept: number;
}

/** Prune persisted job rows older than the `keep` most recent.
 * Destructive; callers wrap in a two-click confirmation.
 * Self-heals stale CSRF tokens. */
export async function pruneOldJobs(keep: number): Promise<PruneResult> {
  const url = `/api/jobs/admin/prune?keep=${encodeURIComponent(String(keep))}`;
  const send = async (token: string | null): Promise<Response> =>
    fetch(url, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        ...(token ? { 'X-CSRF-Token': token } : {}),
      },
    });
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /api/jobs/admin/prune ${res.status}: ${text}`);
  }
  return (await res.json()) as PruneResult;
}
