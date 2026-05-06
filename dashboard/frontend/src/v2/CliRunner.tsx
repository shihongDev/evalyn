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
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from 'react';
import { E } from './tokens';
import { Btn, Eyebrow, Pill, StatusDot } from './ui';
import { useStickToBottom } from './hooks/useStickToBottom';
import type { CliParam, CliParamKind, CliSchema } from './api/cli';
import { commandSummary } from './api/cli';
import {
  cancelJob,
  fetchJobStatus,
  startJob,
  subscribeJob,
  type JobLine,
  type JobStatus,
  type JobStatusKind,
} from './api/jobs';
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

/** Render an argv preview the user can sanity-check before clicking Run. */
function previewCommand(cliId: string, values: Record<string, unknown>): string {
  const parts: string[] = ['evalyn', cliId];
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined || value === null || value === '') continue;
    const flag = `--${name.replace(/_/g, '-')}`;
    if (typeof value === 'boolean') {
      if (value) parts.push(flag);
      continue;
    }
    if (Array.isArray(value)) {
      if (value.length === 0) continue;
      parts.push(flag);
      for (const v of value) parts.push(String(v));
      continue;
    }
    parts.push(flag, String(value));
  }
  return parts.join(' ');
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

/** True if every required param has a non-empty value. */
function isFormValid(
  params: CliParam[],
  values: Record<string, unknown>,
): boolean {
  for (const p of params) {
    if (!p.required) continue;
    const v = coerce(p.kind, values[p.name]);
    if (v === undefined) return false;
    if (Array.isArray(v) && v.length === 0) return false;
    if (typeof v === 'string' && v.trim() === '') return false;
  }
  return true;
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
  const initialValues = useMemo<Record<string, unknown>>(() => {
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
      const seeded = seed?.[p.name];
      if (seeded !== undefined) v = seeded;
      out[p.name] = v;
    }
    return out;
  }, [cli, seed]);
  const [values, setValues] = useState<Record<string, unknown>>(initialValues);
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

  const subRef = useRef<{ close: () => void } | null>(null);
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
        onLine: (line) => {
          setLines((prev) => {
            const next = [...prev, line];
            if (next.length > MAX_OUTPUT_LINES) {
              return next.slice(next.length - MAX_OUTPUT_LINES);
            }
            return next;
          });
        },
        onStatus: (s: JobStatus) => {
          setStatus(s.status);
          patchJob(resumeJobId, {
            status: s.status,
            exit_code: s.exit_code ?? undefined,
            duration: s.duration != null ? String(s.duration) : undefined,
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
  useEffect(() => {
    return () => {
      subRef.current?.close();
      subRef.current = null;
    };
  }, []);

  const isWriteCommand = !READ_ONLY_ALLOWLIST.has(cli.id);
  const isRunning = status === 'running' || status === 'queued';
  const hasJob = jobId !== null;
  const formValid = useMemo(() => isFormValid(cli.params, values), [
    cli.params,
    values,
  ]);

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
      const entry: JobHistoryEntry = {
        job_id,
        cli_id: cli.id,
        cli_args: submitArgs,
        started_at_iso: new Date().toISOString(),
        status: 'queued',
      };
      upsertJob(entry);
      // Open the WS subscription. Hand the lines straight to setState; the
      // ring-buffer trim happens in the updater so we never keep more than
      // MAX_OUTPUT_LINES in memory.
      subRef.current?.close();
      subRef.current = subscribeJob(job_id, {
        onLine: (line) => {
          setLines((prev) => {
            const next = [...prev, line];
            if (next.length > MAX_OUTPUT_LINES) {
              return next.slice(next.length - MAX_OUTPUT_LINES);
            }
            return next;
          });
        },
        onStatus: (s: JobStatus) => {
          setStatus(s.status);
          patchJob(job_id, {
            status: s.status,
            exit_code: s.exit_code ?? undefined,
            duration: s.duration != null ? String(s.duration) : undefined,
          });
          if (s.status === 'complete' || s.status === 'failed') {
            setExitInfo({ code: s.exit_code, duration: s.duration });
          }
        },
        onError: () => {
          setError('Lost connection to job stream.');
        },
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus('failed');
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
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [jobId]);

  const onRerun = useCallback(() => {
    // Tear down current subscription and reset job-side state. Keep form values
    // so the user can tweak + re-run quickly.
    subRef.current?.close();
    subRef.current = null;
    setJobId(null);
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

  // Close on Escape - matches palette behavior.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onCloseClick();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCloseClick]);

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
            type="button"
            onClick={onCloseClick}
            aria-label="Close"
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
            ×
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
                <Btn kind="ghost" size="md" onClick={onCloseClick}>
                  Cancel
                </Btn>
                <Btn
                  kind="primary"
                  size="md"
                  onClick={onRun}
                  disabled={submitting || !formValid}
                  title={
                    !formValid
                      ? 'Fill in all required fields'
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
  const labelEl = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        marginBottom: 4,
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
    </div>
  );

  const help = param.help ?? undefined;

  return (
    <div>
      {labelEl}
      {renderInput(param, value, onChange)}
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
    return (
      <input
        type="number"
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
  return (
    <button
      type="button"
      onClick={onToggle}
      style={{
        background: E.ink,
        border: `1px solid ${E.hair2}`,
        borderRadius: 6,
        padding: '6px 10px',
        textAlign: 'left',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 8,
        width: '100%',
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
}: OutputSectionProps) {
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
          }}
        >
          $ {preview}
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
          {lines.map((l, i) => (
            <div
              key={i}
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
