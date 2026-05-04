/**
 * jobsHistory - localStorage-backed cache of jobs the user has launched
 * from this browser. The backend's `/api/jobs/recent` is in-memory only
 * (gone after server restart) and currently returns `[]`, so we keep a
 * client-side ledger so the Recent Jobs drawer survives page reloads
 * and gives the user a way back to a still-running job after navigating
 * away from the CliRunner.
 *
 * This is a best-effort cache, not a contract. All localStorage calls
 * are wrapped in try/catch (quota / SSR / private-mode / parse failure
 * silently degrade to "empty history"). The drawer can also dim entries
 * whose backend job-record has already been evicted.
 *
 * Cap: keep at most `MAX_ENTRIES` rows (drop oldest by `started_at_iso`).
 * The cap is intentionally small - this is a "where did my job go?"
 * recovery aid, not a full job-history product.
 */

const STORAGE_KEY = 'evalyn:v2:jobsHistory';
const MAX_ENTRIES = 30;

export type JobHistoryStatus =
  | 'queued'
  | 'running'
  | 'complete'
  | 'failed'
  | 'cancelled'
  | 'unknown';

export interface JobHistoryEntry {
  job_id: string;
  cli_id: string;
  /** Frozen snapshot of the args the runner sent to POST /api/cli/run. */
  cli_args: Record<string, unknown>;
  /** ISO 8601 timestamp at which the runner kicked off the job. */
  started_at_iso: string;
  status: JobHistoryStatus;
  exit_code?: number | null;
  /** Human-friendly duration string ("12.4s") or seconds-as-string. */
  duration?: string;
}

function safeStorage(): Storage | null {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Load the history list, sorted newest-first. Returns [] on any error. */
export function loadJobsHistory(): JobHistoryEntry[] {
  const ls = safeStorage();
  if (!ls) return [];
  try {
    const raw = ls.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    const out: JobHistoryEntry[] = [];
    for (const item of parsed) {
      if (!isJobHistoryEntry(item)) continue;
      out.push(item);
    }
    return sortNewestFirst(out);
  } catch {
    return [];
  }
}

/** Persist `entries` to localStorage. Caps at MAX_ENTRIES (drop oldest). */
export function saveJobsHistory(entries: JobHistoryEntry[]): void {
  const ls = safeStorage();
  if (!ls) return;
  try {
    const sorted = sortNewestFirst(entries).slice(0, MAX_ENTRIES);
    ls.setItem(STORAGE_KEY, JSON.stringify(sorted));
    notify();
  } catch {
    // Quota exceeded or storage disabled - silently drop.
  }
}

/**
 * Insert or replace an entry by `job_id`. New entries are added; existing
 * entries are merged (incoming fields win) so re-emitting the same entry
 * during start-up doesn't clobber later status updates.
 */
export function upsertJob(entry: JobHistoryEntry): void {
  const list = loadJobsHistory();
  const idx = list.findIndex((e) => e.job_id === entry.job_id);
  if (idx >= 0) {
    list[idx] = { ...list[idx], ...entry };
  } else {
    list.unshift(entry);
  }
  saveJobsHistory(list);
}

/** Patch fields on an existing entry. No-op if the job is not in history. */
export function patchJob(jobId: string, patch: Partial<JobHistoryEntry>): void {
  const list = loadJobsHistory();
  const idx = list.findIndex((e) => e.job_id === jobId);
  if (idx < 0) return;
  list[idx] = { ...list[idx], ...patch, job_id: jobId };
  saveJobsHistory(list);
}

/** Drop a single entry. */
export function removeJob(jobId: string): void {
  const list = loadJobsHistory().filter((e) => e.job_id !== jobId);
  saveJobsHistory(list);
}

/** Drop every entry. Does NOT cancel any actually-running jobs. */
export function clearJobsHistory(): void {
  const ls = safeStorage();
  if (!ls) return;
  try {
    ls.removeItem(STORAGE_KEY);
    notify();
  } catch {
    // ignore
  }
}

// ---------------------------------------------------------------------------
// Subscription - lightweight pub/sub so the drawer + topbar badge re-render
// when history changes. No external dep; same shape as cliRunnerBridge.
// ---------------------------------------------------------------------------

type Listener = () => void;
const listeners = new Set<Listener>();

export function subscribeJobsHistory(fn: Listener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function notify(): void {
  for (const fn of listeners) {
    try {
      fn();
    } catch {
      // ignore listener errors
    }
  }
}

// Cross-tab updates: storage events fire in OTHER tabs only, so a second
// open tab's drawer stays in sync after history mutations here.
if (typeof window !== 'undefined') {
  try {
    window.addEventListener('storage', (ev) => {
      if (ev.key === STORAGE_KEY) notify();
    });
  } catch {
    // ignore
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isJobHistoryEntry(v: unknown): v is JobHistoryEntry {
  if (!v || typeof v !== 'object') return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.job_id === 'string' &&
    typeof o.cli_id === 'string' &&
    typeof o.started_at_iso === 'string' &&
    typeof o.status === 'string' &&
    typeof o.cli_args === 'object' &&
    o.cli_args !== null
  );
}

function sortNewestFirst(entries: JobHistoryEntry[]): JobHistoryEntry[] {
  return [...entries].sort((a, b) =>
    a.started_at_iso < b.started_at_iso ? 1 : a.started_at_iso > b.started_at_iso ? -1 : 0,
  );
}

/** Convenience: count entries that are still queued/running. */
export function activeJobCount(entries?: JobHistoryEntry[]): number {
  const list = entries ?? loadJobsHistory();
  let n = 0;
  for (const e of list) {
    if (e.status === 'queued' || e.status === 'running') n++;
  }
  return n;
}
