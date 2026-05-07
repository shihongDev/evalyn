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
