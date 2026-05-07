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
const ACK_KEY = 'evalyn:v2:jobsFailureAckTs';
const FILTER_KEY = 'evalyn:v2:jobsDrawer:failureFilter';
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
  /** ISO 8601 timestamp when the status first transitioned to 'failed'.
   * Distinct from started_at_iso because a job queued before the user
   * last opened the drawer can still produce a failure AFTER that ack.
   * Stamped automatically by patchJob. */
  failed_at_iso?: string;
  /** Cumulative count of stderr lines observed for this job by the
   * client. Updated by CliRunner as it streams output. Persisted so
   * the Recent Jobs drawer can surface "5 stderr" inline next to the
   * exit code - useful for spotting noisy jobs even when exit code 0. */
  stderr_count?: number;
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
  // Stamp failed_at_iso on first transition to 'failed' so the
  // unacknowledged-failures badge in the AppShell title knows when the
  // failure became visible (started_at_iso is when the job was queued,
  // which can be much earlier).
  let merged: Partial<JobHistoryEntry> = patch;
  if (
    patch.status === 'failed' &&
    list[idx].status !== 'failed' &&
    !patch.failed_at_iso
  ) {
    merged = { ...patch, failed_at_iso: new Date().toISOString() };
  }
  list[idx] = { ...list[idx], ...merged, job_id: jobId };
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
// open tab's drawer stays in sync after history mutations here. We also
// listen for ACK_KEY changes so a sibling tab opening the drawer
// triggers a title-prefix re-compute in this tab.
if (typeof window !== 'undefined') {
  try {
    window.addEventListener('storage', (ev) => {
      if (ev.key === STORAGE_KEY || ev.key === ACK_KEY) notify();
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

/** Read the timestamp at which the user last viewed the Recent Jobs
 * drawer. Used as the "ack" boundary for the unacknowledged-failures
 * count surfaced in the tab title. Returns 0 when unset (so every
 * recorded failure is treated as unacknowledged on first session). */
export function getFailureAckTime(): number {
  const ls = safeStorage();
  if (!ls) return 0;
  try {
    const raw = ls.getItem(ACK_KEY);
    if (!raw) return 0;
    const n = Number(raw);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  } catch {
    return 0;
  }
}

/** Persist the "ack" timestamp; called by the drawer on open to
 * acknowledge currently-known failures. Notifies subscribers so the
 * tab-title prefix recomputes in the same tab. */
export function setFailureAckTime(ms: number): void {
  const ls = safeStorage();
  if (!ls) return;
  try {
    ls.setItem(ACK_KEY, String(ms));
    notify();
  } catch {
    // ignore
  }
}

/** Read the persisted "show failed only" preference for the Recent
 * Jobs drawer. Defaults to false on missing/unset/parse error so a
 * fresh session shows all jobs by default. */
export function getJobsDrawerFailureFilter(): boolean {
  const ls = safeStorage();
  if (!ls) return false;
  try {
    return ls.getItem(FILTER_KEY) === '1';
  } catch {
    return false;
  }
}

/** Persist the "show failed only" preference. Called by the drawer
 * whenever the user toggles the chip so the choice survives page
 * reloads - useful for users tracking regressions over time who do
 * not want to re-arm the filter on every refresh. */
export function setJobsDrawerFailureFilter(on: boolean): void {
  const ls = safeStorage();
  if (!ls) return;
  try {
    if (on) {
      ls.setItem(FILTER_KEY, '1');
    } else {
      ls.removeItem(FILTER_KEY);
    }
  } catch {
    // Quota / private mode - silently degrade.
  }
}

/** Convenience: count failed entries whose failure timestamp is newer
 * than the user's last drawer-open ack. Falls back to started_at_iso
 * for legacy entries without failed_at_iso. */
export function unacknowledgedFailureCount(entries?: JobHistoryEntry[]): number {
  const list = entries ?? loadJobsHistory();
  const ack = getFailureAckTime();
  let n = 0;
  for (const e of list) {
    if (e.status !== 'failed') continue;
    const ts = e.failed_at_iso ?? e.started_at_iso;
    const t = Date.parse(ts);
    if (Number.isFinite(t) && t > ack) n++;
  }
  return n;
}
