/**
 * CliRunner - global slide-over panel that turns any ``CliSchema`` into a
 * fillable form, spawns ``POST /api/cli/run``, then live-tails the job's
 * output. Mounted once in :mod:`AppShell` so any route can call
 * :func:`openCliRunner` (from ``./cliRunnerBridge``) without prop-drilling.
 *
 * Read-vs-write warning: we mirror the backend's :data:`READ_ONLY_ALLOWLIST`
 * (``dashboard/evalyn_dashboard/agent.py``) as a frontend constant and show
 * a "this command writes" warning chip when a command isn't in that list.
 * The actual auth gate is the CSRF middleware on /api/cli/run; this is a
 * UX nudge so users don't fire writes by accident.
 */

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from 'react';
import { E } from './tokens';
import { Btn, Eyebrow, Pill, StatusDot } from './ui';
import { useStickToBottom } from './hooks/useStickToBottom';
import { useLiveDuration } from './hooks/useLiveDuration';
import { useSearchFilter } from './hooks/useSearchFilter';
import { copyToClipboard } from './clipboard';
import type { CliParam, CliParamKind, CliSchema } from './api/cli';
import { commandSummary, previewCommand } from './api/cli';
import {
  cancelJob,
  fetchJobStatus,
  startJob,
  subscribeJob,
  type JobLine,
  type JobStatus,
  type JobStatusKind,
} from './api/jobs';
import { CapacityError, errorMessage } from './api/errors';
import { clearDraft, loadDraft, saveDraft } from './cliRunnerDrafts';
import { notifyJobTerminal } from './notifications';
import { closeCliRunner, subscribeRunner } from './cliRunnerBridge';
import {
  loadJobsHistory,
  patchJob,
  upsertJob,
  type JobHistoryEntry,
} from './jobsHistory';

// ---------------------------------------------------------------------------
// Read-only allowlist (frontend mirror).
// ---------------------------------------------------------------------------
//
// Pasted from ``dashboard/evalyn_dashboard/agent.py::READ_ONLY_ALLOWLIST``.
// If the backend list changes, update this set so the warning chip stays
// in sync. The backend is the source of truth for actual authorization.
const READ_ONLY_ALLOWLIST: ReadonlySet<string> = new Set([
  'list-calls',
  'list-runs',
  'list-metrics',
  'list-calibrations',
  'show-call',
  'show-trace',
  'show-span',
  'show-projects',
  'analyze',
  'compare',
  'trend',
  'annotation-stats',
  'validate',
  'status',
  'workflow',
  'cluster-failures',
  'cluster-misalignments',
  'insights',
  'select-metrics',
]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MAX_OUTPUT_LINES = 1000;

/** Marker prefix for the client-side overflow indicator. Tested as
 * a startsWith check so the marker line can include a count that
 * grows over time without breaking detection. Kept narrow so we
 * don't accidentally classify a server-dropped marker (different
 * prefix) as a client one. */
const CLIENT_OVERFLOW_PREFIX = '[client buffer rolled over';

function isClientOverflowMarker(line: { kind: string; text: string }): boolean {
  return line.kind === 'system' && line.text.startsWith(CLIENT_OVERFLOW_PREFIX);
}

/** Pure helper: merge a previous lines buffer with newly arrived
 * lines, applying client-side overflow handling. Extracted so the
 * logic is unit-testable without rendering the runner.
 *
 * If `prev` starts with our own overflow marker it is stripped
 * before measuring overflow (otherwise we'd double-count it as a
 * real line). Server "[server dropped ...]" markers ARE real lines
 * and stay in the buffer.
 *
 * Returns the new lines array AND the cumulative dropped count so
 * the caller can persist the count across calls. */
export function mergeLinesWithOverflow(
  prev: JobLine[],
  buf: JobLine[],
  prevDropped: number,
  maxLines: number,
  now: number = Date.now() / 1000,
): { lines: JobLine[]; dropped: number } {
  const stripped =
    prev.length > 0 && isClientOverflowMarker(prev[0]) ? prev.slice(1) : prev;
  const merged = stripped.concat(buf);
  const trimmed =
    merged.length > maxLines
      ? merged.slice(merged.length - maxLines)
      : merged;
  const overflow = merged.length - trimmed.length;
  const dropped = prevDropped + overflow;
  if (dropped > 0) {
    return {
      lines: [
        {
          kind: 'system',
          text: `[client buffer rolled over: ${dropped.toLocaleString()} earlier lines hidden]`,
          ts: now,
        },
        ...trimmed,
      ],
      dropped,
    };
  }
  return { lines: trimmed, dropped };
}

/** Coerce a CliParam value through its kind. Empty strings -> undefined. */
function coerce(kind: CliParamKind, raw: unknown): unknown {
  if (raw === undefined || raw === null) return undefined;
  if (kind === 'number') {
    if (raw === '') return undefined;
    const n = Number(raw);
    return Number.isFinite(n) ? n : undefined;
  }
  if (kind === 'bool') return Boolean(raw);
  if (kind === 'multiselect') {
    return Array.isArray(raw) ? raw : [];
  }
  if (typeof raw === 'string' && raw === '') return undefined;
  return raw;
}

/** Names of required params still missing a value. Empty when the
 * form is valid; used both for the boolean validity check (length === 0)
 * and for the Run button's tooltip ("Missing required: foo, bar") so
 * the user sees exactly which fields are blocking submit. */
function missingRequired(
  params: CliParam[],
  values: Record<string, unknown>,
): string[] {
  const out: string[] = [];
  for (const p of params) {
    if (!p.required) continue;
    const v = coerce(p.kind, values[p.name]);
    if (v === undefined) {
      out.push(p.name);
      continue;
    }
    if (Array.isArray(v) && v.length === 0) out.push(p.name);
    else if (typeof v === 'string' && v.trim() === '') out.push(p.name);
  }
  return out;
}

/** Map a status to a pill color + label. */
function statusPill(status: JobStatusKind): {
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
      return { color: E.pass, bg: E.passDim, label: 'complete' };
    case 'failed':
      return { color: E.fail, bg: E.failDim, label: 'failed' };
    case 'cancelled':
      return { color: E.text2, bg: E.panel3, label: 'cancelled' };
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CliRunner(): ReactElement | null {
  const [cli, setCli] = useState<CliSchema | null>(null);
  const [seed, setSeed] = useState<Record<string, unknown> | undefined>(undefined);
  const [resumeJobId, setResumeJobId] = useState<string | undefined>(undefined);
  const [nonce, setNonce] = useState(0);

  useEffect(
    () =>
      subscribeRunner((s) => {
        setCli(s.cli);
        setSeed(s.initialValues);
        setResumeJobId(s.resumeJobId);
        setNonce(s.nonce);
      }),
    [],
  );

  if (!cli) return null;
  // Re-keying on (cli.id, nonce) resets all internal state (form values,
  // output) when the user closes one runner and opens another - and also
  // when the SAME command is re-opened with a different initialValues
  // payload (deep-link refire from another route).
  return (
    <RunnerBody
      key={`${cli.id}:${nonce}`}
      cli={cli}
      seed={seed}
      resumeJobId={resumeJobId}
      onClose={closeCliRunner}
    />
  );
}

interface RunnerBodyProps {
  cli: CliSchema;
  seed?: Record<string, unknown>;
  resumeJobId?: string;
  onClose: () => void;
}

function RunnerBody({ cli, seed, resumeJobId, onClose }: RunnerBodyProps): ReactElement {
  // --- form state ---
  // Order: schema default -> caller-provided seed -> user edits.
  // We don't coerce the seed values here; the per-kind input components
  // render whatever raw shape we hand them (string for selects, boolean
  // for checkboxes, array for multiselects). Callers building deep links
  // are responsible for handing in compatible types.
  // Priority order when computing initial form values:
  //   schema default -> persisted draft -> caller-provided seed -> user
  // edits (state changes via setValues).
  //
  // The draft layer is a customer-cared addition: a partially-filled
  // form survives a refresh / nav-away. We DON'T let the draft
  // override the seed because seed is more specific intent (deep
  // link, "Re-run with same args" prefill from the drawer) and a
  // stale draft from a prior session shouldn't silently win over
  // what the caller explicitly handed in.
  //
  // Only loaded for fresh form opens (no resumeJobId) - in resume
  // mode the form isn't shown anyway.
  const initialValues = useMemo<Record<string, unknown>>(() => {
    const draft = !resumeJobId && !seed ? loadDraft(cli.id) : null;
    const out: Record<string, unknown> = {};
    for (const p of cli.params) {
      let v: unknown;
      if (p.default !== undefined) {
        v = p.default;
      } else if (p.kind === 'bool') {
        v = false;
      } else if (p.kind === 'multiselect') {
        v = [];
      } else {
        v = '';
      }
      if (draft && draft[p.name] !== undefined) v = draft[p.name];
      const seeded = seed?.[p.name];
      if (seeded !== undefined) v = seeded;
      out[p.name] = v;
    }
    return out;
  }, [cli, seed, resumeJobId]);
  const [values, setValues] = useState<Record<string, unknown>>(initialValues);

  // Auto-save the in-progress form to localStorage so a refresh /
  // nav-away does NOT lose the user's typing. Debounced to 300ms so
  // each keystroke isn't a JSON.stringify + setItem pair. Suppressed
  // for resume mode (form not visible) and for seed-driven opens
  // (the seed itself is the authoritative state - we don't want to
  // shadow it with a draft mid-session).
  useEffect(() => {
    if (resumeJobId || seed) return;
    const handle = window.setTimeout(() => {
      saveDraft(cli.id, values);
    }, 300);
    return () => window.clearTimeout(handle);
  }, [cli.id, values, resumeJobId, seed]);
  const [previewOpen, setPreviewOpen] = useState(false);

  // --- job state ---
  // When `resumeJobId` is set, we boot in the OUTPUT view immediately and
  // attach the WS to the existing job (no POST /api/cli/run). The form is
  // never shown for resumed jobs - this is a "rejoin" flow, not "re-run"
  // (the existing Re-run button handles re-runs once the job finishes).
  // The lazy initializers below seed status / exitInfo from the cached
  // history entry so a completed-job resume opens with its final pill in
  // place rather than blinking through "queued".
  const resumeEntry = useMemo(
    () => (resumeJobId ? loadJobsHistory().find((e) => e.job_id === resumeJobId) : undefined),
    [resumeJobId],
  );
  const [jobId, setJobId] = useState<string | null>(resumeJobId ?? null);
  const [status, setStatus] = useState<JobStatusKind>(() => {
    if (!resumeEntry) return 'queued';
    return resumeEntry.status === 'unknown' ? 'failed' : resumeEntry.status;
  });
  const [lines, setLines] = useState<JobLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [exitInfo, setExitInfo] = useState<{
    code?: number;
    duration?: number;
  } | null>(() => {
    if (!resumeEntry || resumeEntry.exit_code === undefined || resumeEntry.exit_code === null) {
      return null;
    }
    return {
      code: resumeEntry.exit_code,
      duration: resumeEntry.duration ? Number(resumeEntry.duration) : undefined,
    };
  });
  // Wall-clock start of the active job, used by useLiveDuration to
  // render "running 12s" in the output header. Seeded from resumeEntry
  // for resume mode; reset to "now" on each fresh Run; cleared on
  // Re-run before the next launch.
  const [jobStartedAtIso, setJobStartedAtIso] = useState<string | null>(
    () => resumeEntry?.started_at_iso ?? null,
  );

  const subRef = useRef<{ close: () => void } | null>(null);

  // Debounced "Reconnecting" indicator. The WS subscriber retries
  // transparently with exp backoff (1s -> 2s -> 4s -> 8s -> 8s); we
  // surface that to the user only when the disconnect lasts >1.5s,
  // hiding the cosmetic flash for blips we recover from quickly.
  const [reconnecting, setReconnecting] = useState(false);
  const reconnectArmTimerRef = useRef<number | null>(null);

  const handleReconnecting = useCallback(() => {
    if (reconnectArmTimerRef.current != null) return;
    reconnectArmTimerRef.current = window.setTimeout(() => {
      reconnectArmTimerRef.current = null;
      setReconnecting(true);
    }, 1500);
  }, []);

  const handleReconnected = useCallback(() => {
    if (reconnectArmTimerRef.current != null) {
      window.clearTimeout(reconnectArmTimerRef.current);
      reconnectArmTimerRef.current = null;
    }
    setReconnecting(false);
  }, []);

  // Line throughput buffer. WS messages can arrive at hundreds of lines per
  // second on chatty eval runs; one setLines per message means one React
  // render per line which causes visible jank. We coalesce arrivals into
  // ref-buffered batches flushed once per animation frame (60Hz max),
  // turning N renders/sec into <=60 renders/sec regardless of the line
  // rate. When the tab is backgrounded, rAF pauses, so we do no work
  // until the user returns. Cleanup on unmount cancels any pending frame.
  const linesBufferRef = useRef<JobLine[]>([]);
  const flushScheduledRef = useRef<number | null>(null);
  // Cumulative stderr counter for the active job. Persisted to the
  // JobHistoryEntry on terminal status so the Recent Jobs drawer can
  // surface "5 stderr" inline next to the exit code - useful for
  // spotting noisy jobs even when exit code is 0. Counted from the
  // raw stream rather than `lines` state because the visible buffer
  // ring-trims at MAX_OUTPUT_LINES; we want the TRUE total.
  const stderrCountRef = useRef(0);
  // Cumulative count of lines the CLIENT buffer dropped on rollover.
  // Mirrors the server's "[server dropped N earlier lines]" marker.
  // Without this, a user running a chatty eval (>1000 lines) sees the
  // tail and has no clue earlier output was hidden - they could miss
  // the stack-trace start. Reset on every fresh Run / Re-run.
  const clientDroppedRef = useRef(0);

  const flushLineBuffer = useCallback(() => {
    flushScheduledRef.current = null;
    const buf = linesBufferRef.current;
    if (buf.length === 0) return;
    linesBufferRef.current = [];
    setLines((prev) => {
      const result = mergeLinesWithOverflow(
        prev,
        buf,
        clientDroppedRef.current,
        MAX_OUTPUT_LINES,
      );
      clientDroppedRef.current = result.dropped;
      return result.lines;
    });
  }, []);

  const enqueueLine = useCallback(
    (line: JobLine) => {
      if (line.kind === 'stderr') {
        stderrCountRef.current += 1;
      }
      const buf = linesBufferRef.current;
      buf.push(line);
      // When the tab is backgrounded rAF pauses; the buffer would otherwise
      // grow unboundedly while lines keep streaming. Cap at MAX_OUTPUT_LINES
      // since anything beyond that is going to be sliced off on flush anyway.
      if (buf.length > MAX_OUTPUT_LINES) {
        buf.splice(0, buf.length - MAX_OUTPUT_LINES);
      }
      if (flushScheduledRef.current != null) return;
      if (typeof requestAnimationFrame === 'function') {
        flushScheduledRef.current = requestAnimationFrame(flushLineBuffer);
      } else {
        // jsdom / SSR fallback: fire on the next macrotask.
        flushScheduledRef.current = window.setTimeout(flushLineBuffer, 16);
      }
    },
    [flushLineBuffer],
  );

  // Output streaming auto-scroll via the shared chat-style hook. Sticks
  // to bottom on new lines unless the user has scrolled up to inspect
  // history (e.g. mid-stream stack trace), in which case we render a
  // "Jump to latest" pill so they can resume tailing in one click.
  const {
    scrollRef: outputRef,
    onScroll: onOutputScroll,
    scrolledUp: outputScrolledUp,
    jumpToBottom: jumpToOutputBottom,
  } = useStickToBottom(lines.length);

  // Resume mode: re-attach the WS to the existing job. We do a best-effort
  // GET /api/jobs/{id} first so we can dim the row + show an error if the
  // backend evicted the record (e.g. server restart killed the in-memory
  // store), instead of staring at an empty terminal forever.
  useEffect(() => {
    if (!resumeJobId) return;
    let cancelled = false;
    void (async () => {
      const snap = await fetchJobStatus(resumeJobId);
      if (cancelled) return;
      if (snap.kind === 'notFound') {
        patchJob(resumeJobId, { status: 'unknown' });
        setError('Backend no longer has this job in memory. Live tail unavailable.');
        setStatus('failed');
        return;
      }
      if (snap.kind === 'found') {
        patchJob(resumeJobId, {
          status: snap.status,
          exit_code: snap.exit_code ?? undefined,
          duration: snap.duration != null ? String(snap.duration) : undefined,
        });
        setStatus(snap.status);
        if (snap.exit_code !== undefined && snap.exit_code !== null) {
          setExitInfo({ code: snap.exit_code ?? undefined, duration: snap.duration ?? undefined });
        }
      }
      // Open the WS regardless (even for terminal jobs - the backend will
      // replay buffered lines from `?since=0`-style reconnect semantics).
      subRef.current?.close();
      subRef.current = subscribeJob(resumeJobId, {
        onLine: enqueueLine,
        onStatus: (s: JobStatus) => {
          setStatus(s.status);
          const isTerminal =
            s.status === 'complete' ||
            s.status === 'failed' ||
            s.status === 'cancelled';
          patchJob(resumeJobId, {
            status: s.status,
            exit_code: s.exit_code ?? undefined,
            duration: s.duration != null ? String(s.duration) : undefined,
            // Persist the cumulative stderr count once the job finishes
            // so the drawer row can show "N stderr" without rerunning.
            ...(isTerminal ? { stderr_count: stderrCountRef.current } : {}),
          });
          if (s.status === 'complete' || s.status === 'failed') {
            setExitInfo({ code: s.exit_code, duration: s.duration });
          }
        },
        onError: () => {
          // Don't clobber a more-informative `notFound` error that may have
          // already been set by the snapshot fetch above.
          setError((prev) => prev ?? 'Lost connection to job stream.');
        },
        onReconnecting: handleReconnecting,
        onReconnected: handleReconnected,
      });
    })();
    return () => {
      cancelled = true;
    };
    // Intentionally only on first mount; the parent re-keys this component
    // when a new resume target arrives, so the effect re-runs naturally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tear down the WS subscription on unmount or when starting a fresh job.
  // Also cancel any pending rAF flush + reconnect-arm timer so we don't
  // setState after unmount.
  useEffect(() => {
    return () => {
      subRef.current?.close();
      subRef.current = null;
      if (flushScheduledRef.current != null) {
        if (typeof cancelAnimationFrame === 'function') {
          cancelAnimationFrame(flushScheduledRef.current);
        } else {
          window.clearTimeout(flushScheduledRef.current);
        }
        flushScheduledRef.current = null;
      }
      if (reconnectArmTimerRef.current != null) {
        window.clearTimeout(reconnectArmTimerRef.current);
        reconnectArmTimerRef.current = null;
      }
      linesBufferRef.current = [];
    };
  }, []);

  const isWriteCommand = !READ_ONLY_ALLOWLIST.has(cli.id);
  const isRunning = status === 'running' || status === 'queued';
  const hasJob = jobId !== null;
  const missing = useMemo(() => missingRequired(cli.params, values), [
    cli.params,
    values,
  ]);
  const formValid = missing.length === 0;

  // "Has the user touched the form since opening?" - drives the Reset
  // button visibility. JSON-equality is fine for the small per-command
  // forms we have (~10 fields max); the alternative (deep-compare per
  // field) would be more code with no real win here. Reset only shows
  // when there's something to reset, so a fresh-opened pristine form
  // doesn't carry visual noise.
  const isDirty = useMemo(
    () => JSON.stringify(values) !== JSON.stringify(initialValues),
    [values, initialValues],
  );

  // Reset = "back to where I opened the form" (which includes the
  // schema default + any draft + any seed - the same priority chain
  // captured in initialValues). Also clears the persisted draft so
  // the next open is a clean state rather than an immediate restore
  // of the values the user just discarded.
  const onReset = useCallback(() => {
    setValues(initialValues);
    if (cli.id) clearDraft(cli.id);
  }, [initialValues, cli.id]);

  // Build the args dict actually sent to the backend (drop empties, coerce).
  const submitArgs = useMemo<Record<string, unknown>>(() => {
    const out: Record<string, unknown> = {};
    for (const p of cli.params) {
      const v = coerce(p.kind, values[p.name]);
      if (v === undefined) continue;
      if (Array.isArray(v) && v.length === 0) continue;
      // Send `false` for explicit booleans? Backend ignores false flags
      // anyway (see ``_argv_for_tool``). Keep `true` only.
      if (typeof v === 'boolean' && !v) continue;
      out[p.name] = v;
    }
    return out;
  }, [cli.params, values]);

  const preview = useMemo(
    () => previewCommand(cli.id, submitArgs),
    [cli.id, submitArgs],
  );

  const setField = useCallback((name: string, value: unknown) => {
    setValues((v) => ({ ...v, [name]: value }));
  }, []);

  const onRun = useCallback(async () => {
    if (submitting || !formValid) return;
    setSubmitting(true);
    setError(null);
    setLines([]);
    setExitInfo(null);
    setStatus('queued');
    try {
      const { job_id } = await startJob(cli.id, submitArgs);
      setJobId(job_id);
      setStatus('running');
      // Persist a `queued` entry to local history so the Recent Jobs drawer
      // can surface this run if the user navigates away. Status is patched
      // on every WS status event below.
      const startedAtIso = new Date().toISOString();
      setJobStartedAtIso(startedAtIso);
      const entry: JobHistoryEntry = {
        job_id,
        cli_id: cli.id,
        cli_args: submitArgs,
        started_at_iso: startedAtIso,
        status: 'queued',
      };
      upsertJob(entry);
      // Spawn succeeded: the user is done editing this command. Drop
      // the draft so the next open of run-eval (or whichever cli) is
      // a fresh form rather than restoring last-run's values into
      // what the user might think is a clean slate.
      clearDraft(cli.id);
      // Open the WS subscription. Lines flow through enqueueLine, which
      // batches by animation frame so chatty jobs (100s of lines/sec) cap
      // at one render per ~16ms instead of one per line.
      subRef.current?.close();
      subRef.current = subscribeJob(job_id, {
        onLine: enqueueLine,
        onReconnecting: handleReconnecting,
        onReconnected: handleReconnected,
        onStatus: (s: JobStatus) => {
          setStatus(s.status);
          const isTerminal =
            s.status === 'complete' ||
            s.status === 'failed' ||
            s.status === 'cancelled';
          patchJob(job_id, {
            status: s.status,
            exit_code: s.exit_code ?? undefined,
            duration: s.duration != null ? String(s.duration) : undefined,
            ...(isTerminal ? { stderr_count: stderrCountRef.current } : {}),
          });
          if (s.status === 'complete' || s.status === 'failed') {
            setExitInfo({ code: s.exit_code, duration: s.duration });
          }
          // Fire an OS notification when the user is on another tab
          // and the job has reached a terminal state. notifyJobTerminal
          // is internally a no-op without permission / when the tab is
          // foreground, so this is safe to always call.
          // Notify on complete/failed only. We deliberately skip
          // 'cancelled': the user just clicked Cancel, so a popup
          // telling them their job got cancelled is at best redundant
          // and at worst spammy. notifyJobTerminal still gates on
          // permission + hidden-tab, so this is a foreground-friendly
          // shape on top of those.
          if (s.status === 'complete' || s.status === 'failed') {
            notifyJobTerminal({
              jobId: job_id,
              cliId: cli.id,
              status: s.status,
              exitCode: s.exit_code,
              durationS: s.duration ?? null,
            });
          }
        },
        onError: () => {
          setError('Lost connection to job stream.');
        },
      });
    } catch (e) {
      // Capacity 503: don't render a raw "POST /api/cli/run 503: ..."
      // string. Show a clear message with cap state and reset to idle
      // so the user can hit Run again after a few seconds without
      // first having to clear an error state.
      if (e instanceof CapacityError) {
        setError(
          `Job queue full (${e.running} / ${e.maxConcurrent} running). Try again in a few seconds.`,
        );
        // Revert to the pre-click state so the Run button is reachable
        // again. We avoid 'failed' because nothing actually failed on
        // the server - the spawn was rejected before the subprocess
        // was created.
        setStatus('queued');
      } else {
        setError(errorMessage(e));
        setStatus('failed');
      }
    } finally {
      setSubmitting(false);
    }
  }, [submitting, formValid, cli.id, submitArgs]);

  const onCancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
      setStatus('cancelled');
      patchJob(jobId, { status: 'cancelled' });
    } catch (e) {
      setError(errorMessage(e));
    }
  }, [jobId]);

  const onRerun = useCallback(() => {
    // Tear down current subscription and reset job-side state. Keep form values
    // so the user can tweak + re-run quickly.
    subRef.current?.close();
    subRef.current = null;
    // Discard any lines still queued for the previous job; they are stale
    // now and would otherwise flush as the first lines of the new run.
    if (flushScheduledRef.current != null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(flushScheduledRef.current);
      } else {
        window.clearTimeout(flushScheduledRef.current);
      }
      flushScheduledRef.current = null;
    }
    linesBufferRef.current = [];
    stderrCountRef.current = 0;
    clientDroppedRef.current = 0;
    // Drop any lingering "Reconnecting" indicator from the previous run.
    if (reconnectArmTimerRef.current != null) {
      window.clearTimeout(reconnectArmTimerRef.current);
      reconnectArmTimerRef.current = null;
    }
    setReconnecting(false);
    setJobId(null);
    setJobStartedAtIso(null);
    setLines([]);
    setError(null);
    setExitInfo(null);
    setStatus('queued');
  }, []);

  const onCloseClick = useCallback(() => {
    subRef.current?.close();
    subRef.current = null;
    onClose();
  }, [onClose]);

  // Move focus into the runner on open and restore it on close.
  // aria-modal alone does not move focus - keyboard users were left
  // with focus on the trigger button behind the runner, so a Tab
  // press walked them out into the page underneath. The close
  // button is the safest landing target: it does not steal Enter
  // away from form fields the user might want to type into next,
  // but pressing Enter immediately bails out of the runner if
  // that's what the user actually wants.
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null;
    const t = window.setTimeout(() => closeBtnRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(t);
      if (
        prevFocus &&
        prevFocus !== document.body &&
        document.contains(prevFocus)
      ) {
        prevFocus.focus();
      }
    };
  }, []);

  // Clear visible scrollback (terminal Ctrl+L idiom). Does NOT cancel
  // the running job - only flushes the lines state and the rAF buffer
  // so the user can scroll to a fresh slate while the job keeps
  // streaming. The job's own `lines` state survives unaffected on the
  // server side; reattaching with ?since=0 would re-replay everything.
  const clearOutput = useCallback(() => {
    if (flushScheduledRef.current != null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(flushScheduledRef.current);
      } else {
        window.clearTimeout(flushScheduledRef.current);
      }
      flushScheduledRef.current = null;
    }
    linesBufferRef.current = [];
    setLines([]);
  }, []);

  // Close on Escape - matches palette behavior. Ctrl+L (or Cmd+L on
  // macOS) clears the visible scrollback, mirroring terminal idiom.
  // We intentionally only intercept Ctrl+L when the user is NOT
  // typing in an editable element (form mode) so the form retains
  // standard browser shortcut behavior.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onCloseClick();
        return;
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        !e.altKey &&
        !e.shiftKey &&
        e.key.toLowerCase() === 'l'
      ) {
        const target = e.target as HTMLElement | null;
        const tag = target?.tagName?.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || target?.isContentEditable) {
          return;
        }
        e.preventDefault();
        clearOutput();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCloseClick, clearOutput]);

  const pill = statusPill(status);

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(26,24,18,0.35)',
        zIndex: 900,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
      // Backdrop click is a no-op while a job is actively running -
      // accidental misclicks shouldn't drop the live output stream
      // mid-execution. Users can still close via the × button or Esc
      // (both intentional gestures); on close the job continues on
      // the backend and the user can re-attach from Recent Jobs.
      onClick={(ev) => {
        if (ev.target !== ev.currentTarget) return;
        if (status === 'running' || status === 'queued') return;
        onCloseClick();
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 560,
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
        {/* HEADER */}
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
            <Eyebrow>Run command</Eyebrow>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 14,
                color: E.text0,
                marginTop: 2,
                fontWeight: 500,
              }}
            >
              evalyn {cli.id}
            </div>
          </div>
          {hasJob && (
            <Pill mono color={pill.color} bg={pill.bg}>
              <StatusDot
                status={status === 'running' ? 'running' : status}
                animated={status === 'running'}
              />
              {pill.label}
            </Pill>
          )}
          <button
            ref={closeBtnRef}
            type="button"
            onClick={onCloseClick}
            aria-label="Close command runner"
            title={
              status === 'running' || status === 'queued'
                ? 'Close panel (job keeps running - re-attach from Recent Jobs)'
                : 'Close panel'
            }
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

        {/* BODY */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            padding: '14px 18px',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}
        >
          {commandSummary(cli) && !hasJob && (
            <div style={{ fontSize: 12.5, color: E.text2, lineHeight: 1.5 }}>
              {commandSummary(cli)}
            </div>
          )}

          {isWriteCommand && !hasJob && (
            <div
              style={{
                background: E.warnDim,
                border: `1px solid ${E.warn}33`,
                color: E.warn,
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 12,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span>⚠</span>
              This command may write to disk or make external calls. Review the
              preview below before running.
            </div>
          )}

          {error && (
            <div
              role="alert"
              style={{
                background: E.failDim,
                border: `1px solid ${E.fail}33`,
                color: E.fail,
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 12,
                fontFamily: E.fMono,
                whiteSpace: 'pre-wrap',
              }}
            >
              {error}
            </div>
          )}

          {/* Form OR output */}
          {!hasJob ? (
            <FormSection
              params={cli.params}
              values={values}
              onField={setField}
            />
          ) : (
            <OutputSection
              lines={lines}
              status={status}
              exitInfo={exitInfo}
              preview={preview}
              outputRef={outputRef}
              onOutputScroll={onOutputScroll}
              scrolledUp={outputScrolledUp}
              jumpToBottom={jumpToOutputBottom}
              reconnecting={reconnecting}
              startedAtIso={jobStartedAtIso}
              jobId={jobId}
            />
          )}
        </div>

        {/* FOOTER */}
        <div
          style={{
            padding: '12px 18px',
            borderTop: `1px solid ${E.hair}`,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            background: E.panel2,
            flexShrink: 0,
          }}
        >
          {!hasJob && (
            <PreviewLine
              preview={preview}
              open={previewOpen}
              onToggle={() => setPreviewOpen((v) => !v)}
            />
          )}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ flex: 1 }} />
            {!hasJob && (
              <>
                {isDirty && (
                  <Btn
                    kind="ghost"
                    size="md"
                    onClick={onReset}
                    title="Reset to the form's opening values"
                  >
                    Reset
                  </Btn>
                )}
                <Btn kind="ghost" size="md" onClick={onCloseClick}>
                  Cancel
                </Btn>
                <Btn
                  kind="primary"
                  size="md"
                  onClick={onRun}
                  disabled={submitting || !formValid}
                  aria-busy={submitting}
                  title={
                    !formValid
                      ? `Missing required: ${missing.join(', ')}`
                      : isWriteCommand
                        ? 'This command writes - review preview first'
                        : undefined
                  }
                >
                  {submitting ? 'Starting...' : 'Run'}
                </Btn>
              </>
            )}
            {hasJob && isRunning && (
              <Btn kind="danger" size="md" onClick={onCancel}>
                Cancel job
              </Btn>
            )}
            {hasJob && !isRunning && (
              <>
                <Btn kind="ghost" size="md" onClick={onCloseClick}>
                  Close
                </Btn>
                <Btn kind="primary" size="md" onClick={onRerun}>
                  Re-run
                </Btn>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface FormSectionProps {
  params: CliParam[];
  values: Record<string, unknown>;
  onField: (name: string, value: unknown) => void;
}

function FormSection({ params, values, onField }: FormSectionProps) {
  if (params.length === 0) {
    return (
      <div style={{ fontSize: 13, color: E.text3, fontStyle: 'italic' }}>
        This command takes no arguments.
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {params.map((p) => (
        <ParamField
          key={p.name}
          param={p}
          value={values[p.name]}
          onChange={(v) => onField(p.name, v)}
        />
      ))}
    </div>
  );
}

interface ParamFieldProps {
  param: CliParam;
  value: unknown;
  onChange: (v: unknown) => void;
}

function ParamField({ param, value, onChange }: ParamFieldProps) {
  // useId-prefixed so multiple ParamFields on the same page don't
  // collide on htmlFor/id even when two fields share a param.name.
  // The visible label text is unchanged; this is purely the
  // SR-side association so "Tab into this input" announces the
  // field name instead of bare "spin button" or "edit, blank".
  const fieldId = `${useId()}-${param.name}`;
  const labelEl = (
    <label
      htmlFor={fieldId}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        marginBottom: 4,
        cursor: 'default',
      }}
    >
      <span
        style={{
          fontFamily: E.fMono,
          fontSize: 12,
          color: E.text1,
          fontWeight: 500,
        }}
      >
        {param.name}
      </span>
      {param.required && (
        <span
          title="Required"
          aria-label="Required"
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: E.fail,
          }}
        />
      )}
      <span
        style={{
          fontFamily: E.fMono,
          fontSize: 10,
          color: E.text3,
          marginLeft: 'auto',
        }}
      >
        {param.kind}
      </span>
    </label>
  );

  const help = param.help ?? undefined;

  return (
    <div>
      {labelEl}
      {renderInput(param, value, onChange, fieldId)}
      {help && (
        <div style={{ fontSize: 11, color: E.text3, marginTop: 4, lineHeight: 1.4 }}>
          {help}
        </div>
      )}
    </div>
  );
}

const inputStyle: CSSProperties = {
  width: '100%',
  padding: '7px 10px',
  background: E.ink,
  border: `1px solid ${E.hair2}`,
  borderRadius: 6,
  fontSize: 13,
  fontFamily: E.fSans,
  color: E.text0,
  outline: 'none',
  boxSizing: 'border-box',
};

function renderInput(
  param: CliParam,
  value: unknown,
  onChange: (v: unknown) => void,
  /** Optional element id so the parent's `<label htmlFor={id}>`
   * binds to the actual interactive input. The bool branch wraps
   * its own label so it ignores this. */
  id?: string,
): ReactElement {
  const { kind, options } = param;
  if (kind === 'bool') {
    return (
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 12,
          color: E.text2,
          cursor: 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        Enable {param.name}
      </label>
    );
  }
  if (kind === 'select') {
    return (
      <select
        id={id}
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      >
        <option value="">— select —</option>
        {(options ?? []).map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  }
  if (kind === 'multiselect') {
    const arr = Array.isArray(value) ? (value as string[]) : [];
    const opts = options ?? [];
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {opts.length === 0 ? (
          <input
            id={id}
            type="text"
            value={arr.join(',')}
            placeholder="comma-separated values"
            onChange={(e) =>
              onChange(
                e.target.value
                  .split(',')
                  .map((s) => s.trim())
                  .filter(Boolean),
              )
            }
            style={inputStyle}
          />
        ) : (
          opts.map((opt) => {
            const on = arr.includes(opt);
            return (
              <button
                key={opt}
                type="button"
                onClick={() =>
                  onChange(on ? arr.filter((x) => x !== opt) : [...arr, opt])
                }
                style={{
                  padding: '4px 10px',
                  borderRadius: 999,
                  fontSize: 11,
                  fontFamily: E.fMono,
                  border: `1px solid ${on ? E.ember : E.hair2}`,
                  background: on ? E.emberDim : 'transparent',
                  color: on ? E.ember : E.text2,
                  cursor: 'pointer',
                }}
              >
                {opt}
              </button>
            );
          })
        )}
      </div>
    );
  }
  if (kind === 'number') {
    // Decimal (not numeric) - CLI param ranges may include floats
    // (e.g. temperature 0..1, step 0.05), and inputMode="decimal"
    // shows a numeric keypad with a decimal point on iOS / Android.
    // inputMode="numeric" would force-strip the decimal on those
    // keyboards, which is wrong for any non-integer step.
    return (
      <input
        id={id}
        type="number"
        inputMode="decimal"
        value={value === undefined || value === null ? '' : String(value)}
        onChange={(e) => onChange(e.target.value)}
        step={param.range?.step}
        min={param.range?.min}
        max={param.range?.max}
        style={inputStyle}
      />
    );
  }
  if (kind === 'long-text') {
    return (
      <textarea
        id={id}
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        style={{ ...inputStyle, resize: 'vertical', fontFamily: E.fMono }}
      />
    );
  }
  // string + path -> text input
  return (
    <input
      id={id}
      type="text"
      value={typeof value === 'string' ? value : ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder={kind === 'path' ? 'path/to/file' : ''}
      style={inputStyle}
    />
  );
}

interface PreviewLineProps {
  preview: string;
  open: boolean;
  onToggle: () => void;
}

function PreviewLine({ preview, open, onToggle }: PreviewLineProps) {
  // Local copy-state. The form-mode preview is rendered before any
  // job is spawned, so we can't lean on the OutputSection toolbar's
  // Copy button (it only mounts in the post-spawn view). This makes
  // the affordance reachable from the moment the form is filled.
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>(
    'idle',
  );
  const onCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!preview) return;
    try {
      await copyToClipboard(preview);
      setCopyState('copied');
      window.setTimeout(() => setCopyState('idle'), 2000);
    } catch {
      setCopyState('error');
      window.setTimeout(() => setCopyState('idle'), 3000);
    }
  };
  return (
    <div
      style={{
        background: E.ink,
        border: `1px solid ${E.hair2}`,
        borderRadius: 6,
        display: 'flex',
        alignItems: 'flex-start',
        gap: 0,
        width: '100%',
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-label={
          open ? 'Collapse the command preview' : 'Expand the command preview'
        }
        style={{
          flex: 1,
          padding: '6px 10px',
          textAlign: 'left',
          cursor: 'pointer',
          background: 'transparent',
          border: 'none',
          display: 'flex',
          alignItems: 'flex-start',
          gap: 8,
          minWidth: 0,
        }}
      >
        <span
          style={{
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.text3,
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          {open ? '▾' : '▸'}
        </span>
        <span
          style={{
            fontFamily: E.fMono,
            fontSize: 11,
            color: E.text1,
            flex: 1,
            minWidth: 0,
            ...(open
              ? { whiteSpace: 'pre-wrap', wordBreak: 'break-all' }
              : {
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }),
          }}
        >
          {preview}
        </span>
      </button>
      <button
        type="button"
        onClick={onCopy}
        aria-label="Copy the shell command to clipboard"
        title={
          copyState === 'copied'
            ? 'Copied!'
            : copyState === 'error'
              ? 'Browser blocked clipboard access'
              : 'Copy command'
        }
        style={{
          flexShrink: 0,
          padding: '0 10px',
          fontFamily: E.fMono,
          fontSize: 10.5,
          color:
            copyState === 'copied'
              ? E.pass
              : copyState === 'error'
                ? E.fail
                : E.text3,
          background: 'transparent',
          border: 'none',
          borderLeft: `1px solid ${E.hair2}`,
          cursor: 'pointer',
          alignSelf: 'stretch',
          whiteSpace: 'nowrap',
        }}
        onMouseEnter={(e) => {
          if (copyState === 'idle') {
            e.currentTarget.style.color = E.text1;
          }
        }}
        onMouseLeave={(e) => {
          if (copyState === 'idle') {
            e.currentTarget.style.color = E.text3;
          }
        }}
      >
        {copyState === 'copied' ? '✓' : copyState === 'error' ? '✗' : 'copy'}
      </button>
    </div>
  );
}

interface OutputSectionProps {
  lines: JobLine[];
  status: JobStatusKind;
  exitInfo: { code?: number; duration?: number } | null;
  preview: string;
  outputRef: React.RefObject<HTMLDivElement | null>;
  onOutputScroll: () => void;
  scrolledUp: boolean;
  jumpToBottom: () => void;
  /** True when the WS subscriber is reconnecting after an unexpected
   * close and the disconnect has lasted past the parent's debounce
   * threshold (1.5s). Renders a small pill in the preview row so the
   * user knows the gap in output is recoverable, not a hung eval. */
  reconnecting?: boolean;
  /** ISO timestamp of when the active job started. Drives the live
   * "running 12s" counter via useLiveDuration. Null when no job is
   * active (form mode) or after Re-run resets state. */
  startedAtIso?: string | null;
  /** Server-assigned job_id for the active subscription. Surfaces a
   * "Copy ID" button in the preview action row so a user filing a
   * support ticket or grep-ing logs can grab it in one click. Null
   * when no job has been spawned yet. */
  jobId?: string | null;
}

function OutputSection({
  lines,
  status,
  exitInfo,
  preview,
  outputRef,
  onOutputScroll,
  scrolledUp,
  jumpToBottom,
  reconnecting = false,
  startedAtIso = null,
  jobId = null,
}: OutputSectionProps) {
  // Inline clipboard state for the Copy button. Idle -> copied flips
  // the label briefly; idle -> error covers the rare browser-blocked
  // case (non-secure context with no clipboard API and execCommand
  // disabled).
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle');
  // Independent state for the "Copy ID" action so a successful output
  // copy doesn't tint the ID button (and vice versa).
  const [copyIdState, setCopyIdState] = useState<'idle' | 'copied' | 'error'>(
    'idle',
  );
  // Independent state for the "Copy command" action.
  const [copyCmdState, setCopyCmdState] = useState<
    'idle' | 'copied' | 'error'
  >('idle');

  async function handleCopyJobId() {
    if (!jobId) return;
    try {
      await copyToClipboard(jobId);
      setCopyIdState('copied');
      window.setTimeout(() => setCopyIdState('idle'), 2000);
    } catch {
      setCopyIdState('error');
      window.setTimeout(() => setCopyIdState('idle'), 3000);
    }
  }

  // Independent state for the "Copy log URL" action so it doesn't
  // tint the other copy buttons (and vice versa).
  const [copyUrlState, setCopyUrlState] = useState<
    'idle' | 'copied' | 'error'
  >('idle');

  async function handleCopyLogUrl() {
    if (!jobId || typeof window === 'undefined') return;
    // Absolute URL so a teammate clicking the pasted link from
    // chat/email lands on THIS dashboard rather than whatever origin
    // their browser currently has open.
    //
    // ?download=1 sets Content-Disposition so the link triggers a
    // save-as. ?include_meta=1 prepends a # header with job_id, cli,
    // started_at, status, exit_code so the recipient's downloaded
    // file is self-describing - they can tell what run produced it
    // without the chat context the link arrived with. The # prefix
    // means log tools (less, awk, grep) treat the header as comments.
    const url = `${window.location.origin}/api/jobs/${encodeURIComponent(jobId)}/output.txt?download=1&include_meta=1`;
    try {
      await copyToClipboard(url);
      setCopyUrlState('copied');
      window.setTimeout(() => setCopyUrlState('idle'), 2000);
    } catch {
      setCopyUrlState('error');
      window.setTimeout(() => setCopyUrlState('idle'), 3000);
    }
  }

  async function handleCopyCommand() {
    // Copy the preview line WITHOUT the leading "$ " sigil so it's
    // ready to paste into a real terminal. The sigil exists only to
    // signal "this is a shell command" visually; including it would
    // make the paste fail.
    if (!preview) return;
    try {
      await copyToClipboard(preview);
      setCopyCmdState('copied');
      window.setTimeout(() => setCopyCmdState('idle'), 2000);
    } catch {
      setCopyCmdState('error');
      window.setTimeout(() => setCopyCmdState('idle'), 3000);
    }
  }

  // Live "running 12s" counter for queued/running jobs. Hidden once a
  // terminal status arrives (the existing exitInfo footer takes over
  // duration display). Shares the useLiveDuration hook with the
  // RecentJobsDrawer rows for consistent formatting.
  const isLive = status === 'queued' || status === 'running';
  const liveDuration = useLiveDuration(startedAtIso ?? '', isLive && !!startedAtIso);

  // Errors-only filter. When on, the output panel renders only stderr
  // (and system) lines so the user can find a buried failure in a
  // chatty eval without manual scrolling. Toggled by clicking the
  // stderr-count chip in the header. Resets implicitly when this
  // component unmounts (e.g. switching to a different command tab).
  const [stderrFilter, setStderrFilter] = useState(false);
  // Free-text output filter. Customer pain: 1000+ streamed lines and
  // browser Ctrl+F doesn't survive scroll/streaming additions cleanly.
  // No sessionStorage persistence here - each run produces a fresh
  // output buffer so persisting a filter across runs would be wrong
  // (the matching lines from the previous run aren't there). The
  // hook's onKeyDown handles two-step Esc directly on the input;
  // "/" hotkey at the window level lives below since it must work
  // even when focus is outside the input. 150ms debounce (vs the
  // hook's 120ms default) keeps the existing UX - heavy-output
  // runs feel slightly less twitchy.
  const {
    input: outputFilterInput,
    setInput: setOutputFilterInput,
    query: outputFilterQuery,
    inputRef: outputFilterRef,
    onKeyDown: onOutputFilterKeyDown,
  } = useSearchFilter({ debounceMs: 150 });
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || target?.isContentEditable) {
        return;
      }
      e.preventDefault();
      outputFilterRef.current?.focus();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [outputFilterRef]);

  async function handleCopy() {
    if (lines.length === 0) return;
    // Concatenate output lines into a single string. Stderr lines stay
    // interleaved with stdout - same order they arrived in - because
    // splitting them would lose the chronology that's often what the
    // user actually cares about ("error happened RIGHT after this log").
    // The Copy button copies ALL lines regardless of the stderr filter:
    // the filter is for scanning, the clipboard is for sharing.
    const text = lines.map((l) => l.text).join('\n');
    try {
      await copyToClipboard(text);
      setCopyState('copied');
      window.setTimeout(() => setCopyState('idle'), 2000);
    } catch {
      setCopyState('error');
      window.setTimeout(() => setCopyState('idle'), 3000);
    }
  }

  function handleDownload() {
    if (lines.length === 0) return;
    // Save the same flat-text shape Copy uses, plus a trailing newline
    // so the file ends correctly. Tools (less, tail, grep) expect this.
    const text = lines.map((l) => l.text).join('\n') + '\n';
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    // Derive the cli-id from the preview command ("evalyn <cli-id> ...")
    // for a recognisable filename. ISO timestamp keeps multiple downloads
    // from clobbering each other in the user's Downloads folder.
    const cliId = preview.split(/\s+/)[1] ?? 'output';
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const fname = `evalyn-${cliId}-${ts}.log`;
    const a = document.createElement('a');
    a.href = url;
    a.download = fname;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Defer revoke to next tick so Safari has time to start the download.
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  // At-a-glance error count. We surface this even mid-stream so the user
  // knows the run is producing errors before it finishes - a 30s eval
  // failing on item 3 shouldn't have to wait until exit code to be
  // visible.
  const stderrCount = useMemo(
    () => lines.reduce((n, l) => (l.kind === 'stderr' ? n + 1 : n), 0),
    [lines],
  );

  // Apply the errors-only filter. System lines (e.g. truncation markers)
  // are kept visible since they describe diagnostic state, not stdout
  // content. Memoized so the filter pass does not run on every unrelated
  // re-render (lines updates ~60Hz during streaming due to rAF batching).
  const visibleLines = useMemo(() => {
    let out = lines;
    if (stderrFilter) {
      out = out.filter((l) => l.kind === 'stderr' || l.kind === 'system');
    }
    if (outputFilterQuery) {
      // Text filter applies to ALL lines (including system) so a
      // user grep-ing for "ERROR" doesn't accidentally see a
      // "[server dropped N lines]" marker counted as a match.
      out = out.filter((l) =>
        l.text.toLowerCase().includes(outputFilterQuery),
      );
    }
    return out;
  }, [lines, stderrFilter, outputFilterQuery]);

  // Auto-clear the filter when no stderr exists. Two scenarios:
  //   (a) terminal status reached with zero stderr - filter would be
  //       stuck "on" with no way to view stdout if the user navigates
  //       away and back.
  //   (b) all stderr lines got evicted from the ring buffer mid-run
  //       (chatty stdout pushed them past MAX_OUTPUT_LINES). The chip
  //       vanishes when stderrCount drops to 0, so the user has no way
  //       to toggle the filter back off.
  // The chip only renders when stderrCount > 0, so a clear of an
  // accidentally-active filter never disagrees with user intent: the
  // user could only have switched the filter ON when stderrCount was
  // already > 0.
  useEffect(() => {
    if (stderrFilter && stderrCount === 0) {
      setStderrFilter(false);
    }
  }, [stderrFilter, stderrCount]);

  // Index of the first stderr line in the currently-visible output. Used
  // by the "Jump to first error" button (rendered next to the stderr
  // chip) to scroll the user directly to the first failure point without
  // losing the surrounding stdout chronology - useful when an eval
  // produces hundreds of stdout lines with one error buried in the
  // middle. -1 when no stderr is present.
  const firstStderrIndex = useMemo(() => {
    for (let i = 0; i < visibleLines.length; i++) {
      if (visibleLines[i].kind === 'stderr') return i;
    }
    return -1;
  }, [visibleLines]);

  const jumpToFirstError = useCallback(() => {
    const container = outputRef.current;
    if (!container) return;
    const node = container.querySelector<HTMLElement>('[data-first-stderr]');
    if (!node) return;
    // scrollIntoView on the line element scrolls its NEAREST scrollable
    // ancestor; the output panel IS that ancestor (overflow: auto), so
    // the page itself does not jump. block: 'center' keeps a few lines
    // of stdout context above the error.
    node.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [outputRef]);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1, minHeight: 0 }}>
        <div
          style={{
            fontFamily: E.fMono,
            fontSize: 11,
            color: E.text3,
            background: E.ink,
            padding: '6px 10px',
            borderRadius: 6,
            border: `1px solid ${E.hair2}`,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            flexShrink: 0,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
          }}
        >
          <span style={{ flex: 1, minWidth: 0 }}>$ {preview}</span>
          {liveDuration && (
            <span
              role="status"
              aria-live="off"
              aria-label={`Running for ${liveDuration}`}
              title="Elapsed time since job started"
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.ember,
                background: 'rgba(217, 99, 49, 0.10)',
                border: `1px solid rgba(217, 99, 49, 0.3)`,
                borderRadius: 4,
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: 500,
              }}
            >
              running {liveDuration}
            </span>
          )}
          {reconnecting && (
            <span
              role="status"
              aria-live="polite"
              aria-label="Reconnecting to job stream"
              title="WebSocket lost - retrying with backoff"
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.warn,
                background: 'rgba(217, 161, 79, 0.15)',
                border: `1px solid rgba(217, 161, 79, 0.4)`,
                borderRadius: 4,
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: 500,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: E.warn,
                  display: 'inline-block',
                }}
              />
              Reconnecting…
            </span>
          )}
          {lines.length > 0 && (
            <input
              ref={outputFilterRef}
              type="search"
              value={outputFilterInput}
              onChange={(e) => setOutputFilterInput(e.target.value)}
              onKeyDown={onOutputFilterKeyDown}
              placeholder="Filter output... (/ to focus)"
              aria-label="Filter output by substring"
              style={{
                flexShrink: 0,
                width: 160,
                padding: '2px 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.text0,
                background: E.panel2,
                border: `1px solid ${E.hair2}`,
                borderRadius: 4,
                lineHeight: 1.6,
                outline: 'none',
              }}
            />
          )}
          {outputFilterQuery && (
            <span
              aria-live="polite"
              style={{
                flexShrink: 0,
                fontSize: 10,
                fontFamily: E.fMono,
                color: visibleLines.length === 0 ? E.fail : E.text3,
              }}
              title={`${visibleLines.length} of ${lines.length} lines match`}
            >
              {visibleLines.length}/{lines.length}
            </span>
          )}
          {stderrCount > 0 && (
            <button
              type="button"
              onClick={() => setStderrFilter((v) => !v)}
              aria-pressed={stderrFilter}
              aria-label={
                stderrFilter
                  ? `Showing only stderr (${stderrCount} line${stderrCount === 1 ? '' : 's'}); click to show all output`
                  : `${stderrCount} stderr line${stderrCount === 1 ? '' : 's'}; click to filter to stderr only`
              }
              title={
                stderrFilter
                  ? 'Showing stderr only - click to show all'
                  : `Filter to ${stderrCount} stderr line${stderrCount === 1 ? '' : 's'}`
              }
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: '#ff9580',
                background: stderrFilter
                  ? 'rgba(255, 149, 128, 0.28)'
                  : 'rgba(255, 149, 128, 0.12)',
                border: `1px solid rgba(255, 149, 128, ${stderrFilter ? 0.7 : 0.3})`,
                borderRadius: 4,
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: stderrFilter ? 600 : 500,
                cursor: 'pointer',
              }}
            >
              {stderrFilter ? '✓ ' : ''}
              {stderrCount} stderr
            </button>
          )}
          {stderrCount > 0 && !stderrFilter && firstStderrIndex >= 0 && (
            <button
              type="button"
              onClick={jumpToFirstError}
              aria-label="Scroll to first stderr line"
              title="Jump to first error"
              style={{
                flexShrink: 0,
                padding: '0 6px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: '#ff9580',
                background: 'transparent',
                border: `1px solid rgba(255, 149, 128, 0.3)`,
                borderRadius: 4,
                cursor: 'pointer',
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
                fontWeight: 500,
              }}
            >
              ↥ first
            </button>
          )}
          {preview && (
            <button
              type="button"
              onClick={() => void handleCopyCommand()}
              aria-label="Copy the shell command to clipboard"
              title={
                copyCmdState === 'copied'
                  ? `Copied: ${preview}`
                  : copyCmdState === 'error'
                    ? 'Browser blocked clipboard access'
                    : 'Copy the shell command (paste into your terminal)'
              }
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color:
                  copyCmdState === 'copied'
                    ? E.pass
                    : copyCmdState === 'error'
                      ? E.fail
                      : E.text2,
                background: 'transparent',
                border: `1px solid ${E.hair2}`,
                borderRadius: 4,
                cursor: 'pointer',
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
              }}
            >
              {copyCmdState === 'copied'
                ? '✓ Cmd copied'
                : copyCmdState === 'error'
                  ? '✗ Failed'
                  : 'Copy command'}
            </button>
          )}
          {jobId && (
            <button
              type="button"
              onClick={() => void handleCopyJobId()}
              aria-label={`Copy job ID ${jobId} to clipboard`}
              title={
                copyIdState === 'copied'
                  ? `Copied: ${jobId}`
                  : copyIdState === 'error'
                    ? 'Browser blocked clipboard access'
                    : `Copy job ID (${jobId.slice(0, 8)}...) to clipboard`
              }
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color:
                  copyIdState === 'copied'
                    ? E.pass
                    : copyIdState === 'error'
                      ? E.fail
                      : E.text2,
                background: 'transparent',
                border: `1px solid ${E.hair2}`,
                borderRadius: 4,
                cursor: 'pointer',
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
              }}
            >
              {copyIdState === 'copied'
                ? '✓ ID copied'
                : copyIdState === 'error'
                  ? '✗ Failed'
                  : 'Copy ID'}
            </button>
          )}
          {jobId && (
            <button
              type="button"
              onClick={() => void handleCopyLogUrl()}
              aria-label="Copy a shareable download URL for this job's log"
              title={
                copyUrlState === 'copied'
                  ? 'Log URL copied'
                  : copyUrlState === 'error'
                    ? 'Browser blocked clipboard access'
                    : 'Copy a shareable URL that downloads this job\'s log'
              }
              style={{
                flexShrink: 0,
                padding: '0 8px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color:
                  copyUrlState === 'copied'
                    ? E.pass
                    : copyUrlState === 'error'
                      ? E.fail
                      : E.text2,
                background: 'transparent',
                border: `1px solid ${E.hair2}`,
                borderRadius: 4,
                cursor: 'pointer',
                lineHeight: 1.6,
                whiteSpace: 'nowrap',
              }}
            >
              {copyUrlState === 'copied'
                ? '✓ URL copied'
                : copyUrlState === 'error'
                  ? '✗ Failed'
                  : 'Copy log URL'}
            </button>
          )}
          <button
            type="button"
            onClick={() => void handleCopy()}
            disabled={lines.length === 0}
            aria-label="Copy output to clipboard"
            title={
              lines.length === 0
                ? 'Nothing to copy yet'
                : copyState === 'copied'
                  ? 'Output copied'
                  : copyState === 'error'
                    ? 'Browser blocked clipboard access'
                    : `Copy ${lines.length} line${lines.length === 1 ? '' : 's'} of output to clipboard`
            }
            style={{
              flexShrink: 0,
              padding: '0 8px',
              fontFamily: E.fMono,
              fontSize: 10.5,
              color: copyState === 'copied' ? E.pass : copyState === 'error' ? E.fail : E.text2,
              background: 'transparent',
              border: `1px solid ${E.hair2}`,
              borderRadius: 4,
              cursor: lines.length === 0 ? 'not-allowed' : 'pointer',
              opacity: lines.length === 0 ? 0.5 : 1,
              lineHeight: 1.6,
              whiteSpace: 'nowrap',
            }}
          >
            {copyState === 'copied'
              ? '✓ Copied'
              : copyState === 'error'
                ? '✗ Failed'
                : 'Copy output'}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={lines.length === 0}
            aria-label="Download output as .log file"
            title={
              lines.length === 0
                ? 'Nothing to download yet'
                : `Save ${lines.length} line${lines.length === 1 ? '' : 's'} as .log file`
            }
            style={{
              flexShrink: 0,
              padding: '0 8px',
              fontFamily: E.fMono,
              fontSize: 10.5,
              color: E.text2,
              background: 'transparent',
              border: `1px solid ${E.hair2}`,
              borderRadius: 4,
              cursor: lines.length === 0 ? 'not-allowed' : 'pointer',
              opacity: lines.length === 0 ? 0.5 : 1,
              lineHeight: 1.6,
              whiteSpace: 'nowrap',
            }}
          >
            ↓ .log
          </button>
        </div>
        <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex' }}>
        <div
          ref={outputRef}
          onScroll={onOutputScroll}
          style={{
            flex: 1,
            minHeight: 220,
            background: '#15140f',
            color: '#e8e2d2',
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            padding: '10px 12px',
            fontFamily: E.fMono,
            fontSize: 11.5,
            lineHeight: 1.5,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {lines.length === 0 && status === 'running' && (
            <span style={{ color: '#7a7466', fontStyle: 'italic' }}>
              Waiting for output...
            </span>
          )}
          {lines.length === 0 && status !== 'running' && (
            <span style={{ color: '#7a7466', fontStyle: 'italic' }}>
              (no output)
            </span>
          )}
          {lines.length > 0 && visibleLines.length === 0 && stderrFilter && (
            <span style={{ color: '#7a7466', fontStyle: 'italic' }}>
              No stderr lines yet. Toggle the chip above to show all output.
            </span>
          )}
          {visibleLines.map((l, i) => (
            <div
              key={i}
              data-first-stderr={i === firstStderrIndex ? 'true' : undefined}
              style={{
                color:
                  l.kind === 'stderr'
                    ? '#ff9580'
                    : l.kind === 'system'
                      ? '#7a7466'
                      : '#e8e2d2',
              }}
            >
              {l.text}
            </div>
          ))}
        </div>
        {scrolledUp && lines.length > 0 && (
          <button
            type="button"
            onClick={jumpToBottom}
            aria-label="Jump to latest output"
            style={{
              position: 'absolute',
              bottom: 10,
              left: '50%',
              transform: 'translateX(-50%)',
              padding: '4px 10px',
              borderRadius: 999,
              background: E.ember,
              color: E.emberInk,
              border: 'none',
              cursor: 'pointer',
              fontSize: 10.5,
              fontFamily: E.fMono,
              fontWeight: 500,
              boxShadow: `0 6px 18px rgba(217,106,44,0.32)`,
              zIndex: 5,
              animation: 'eRouteFallbackFadeIn 200ms ease',
              whiteSpace: 'nowrap',
            }}
          >
            ↓ Jump to latest
          </button>
        )}
        </div>
        {exitInfo && (
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text3,
              flexShrink: 0,
            }}
          >
            exit {exitInfo.code ?? '?'}
            {exitInfo.duration !== undefined &&
              ` · ${exitInfo.duration.toFixed(2)}s`}
          </div>
        )}
      </div>
    );
}
