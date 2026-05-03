/**
 * CliForm - the three-mode CLI invocation form (legacy tab-based view).
 *
 * Used when a CLI is opened as a separate tab. The Workspace view in
 * `Workspace.tsx` uses `CliFormBody` directly via its active form panel.
 *
 * Submit -> `runCli` schedules the run and opens a `job:<id>` tab. The form
 * tab is NOT closed: instead the form-body region swaps to a "Run completed"
 * card once the job reaches a terminal state. The user can re-run with the
 * same args, switch to the job tab, or reset and start a new run.
 */

import { useEffect, useRef, useState } from 'react';
import type { CliSchema } from '../types/catalog';
import { useStore } from '../store';
import { buildCli, defaultValues, type CliFormValues } from './buildCli';
import CliFormBody from './CliFormBody';
import RecentRunsStrip from './RecentRunsStrip';
import SavedPresets from './SavedPresets';

export interface CliFormProps {
  cli: CliSchema;
}

interface PostRunSummary {
  jobId: string;
  exitCode?: number;
  duration?: string;
  startedAt: number;
  completedAt: number;
}

/** Pretty wall-clock duration when the job didn't supply one. */
const formatElapsed = (startedAt: number, completedAt: number): string => {
  const ms = Math.max(0, completedAt - startedAt);
  const sec = Math.round(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${String(s).padStart(2, '0')}s`;
};

/** Required-field validation, factored so submit and the inline link agree. */
const computeFieldErrors = (
  cli: CliSchema,
  values: CliFormValues,
): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const p of cli.params) {
    if (!p.required) continue;
    const v = values[p.name];
    if (v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0)) {
      out[p.name] = 'required';
    }
  }
  return out;
};

const CliForm = ({ cli }: CliFormProps) => {
  const mode = useStore((s) => s.tweaks.cliFormMode);
  const runCli = useStore((s) => s.runCli);
  const closeTab = useStore((s) => s.closeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const tabs = useStore((s) => s.tabs);
  const openJobTab = useStore((s) => s.openJobTab);
  const jobs = useStore((s) => s.jobs);

  const [values, setValues] = useState<CliFormValues>(() => defaultValues(cli));
  const [submitting, setSubmitting] = useState(false);
  // Server-side error (non-2xx from /api/cli/run). Field-scoped errors live
  // on individual ParamFields; this banner is reserved for transport-level
  // failures that don't map to a single field.
  const [serverError, setServerError] = useState<string | null>(null);
  const [postRunSummary, setPostRunSummary] = useState<PostRunSummary | null>(null);
  // Triggered on submit attempt — until then we don't paint field-scoped
  // errors so the user isn't yelled at on first paint.
  const [validationActive, setValidationActive] = useState(false);
  // Track the active submission so the post-run effect knows which job to
  // watch and can ignore late events from earlier runs.
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Remember when this submission started so we can compute elapsed if the
  // exit event omits duration.
  const startedAtRef = useRef<number>(0);

  const cmd = buildCli(cli, values);

  const fieldErrors = validationActive ? computeFieldErrors(cli, values) : {};
  const errorCount = Object.keys(fieldErrors).length;

  // Watch the active job for completion; on terminal state, swap to the
  // completion card. The job tab opened by `runCli` stays open.
  useEffect(() => {
    if (!activeJobId) return;
    const job = jobs.get(activeJobId);
    if (!job || job.status === 'pending' || job.status === 'running') return;
    setPostRunSummary({
      jobId: activeJobId,
      exitCode: job.exitCode,
      duration: job.duration,
      startedAt: startedAtRef.current,
      completedAt: Date.now(),
    });
    setActiveJobId(null);
  }, [activeJobId, jobs]);

  const submitRun = async () => {
    setValidationActive(true);
    const errs = computeFieldErrors(cli, values);
    if (Object.keys(errs).length > 0) {
      // Field-scoped errors render in the form; the "fields need attention"
      // anchor lets the user jump to the first.
      return;
    }
    setServerError(null);
    setSubmitting(true);
    setPostRunSummary(null);
    startedAtRef.current = Date.now();
    try {
      const jobId = await runCli(cli.id, values);
      setActiveJobId(jobId);
    } catch (err) {
      setServerError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const onRun = () => void submitRun();
  const onCancel = () => closeTab(`cli:${cli.id}`);
  const onCopy = () => {
    // Ignore failures in restricted contexts (no clipboard API, denied perms).
    if (typeof navigator === 'undefined') return;
    navigator.clipboard?.writeText?.(cmd).catch(() => {});
  };

  /** "N fields need attention" anchor — scrolls the first errored field into view. */
  const focusFirstError = () => {
    if (errorCount === 0 || typeof document === 'undefined') return;
    const first = cli.params.find((p) => fieldErrors[p.name]);
    if (!first) return;
    const el = document.querySelector<HTMLElement>(`[data-param-name="${first.name}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.querySelector<HTMLElement>('input, select, textarea, button')?.focus();
  };

  /** Re-run the same args without leaving the form. */
  const onRerun = onRun;

  /** Reset the form to defaults and swap back to the form-body view. */
  const onNewRun = () => {
    setValues(defaultValues(cli));
    setPostRunSummary(null);
    setActiveJobId(null);
    setServerError(null);
    setValidationActive(false);
  };

  /** Open the job tab (or activate it if already present). */
  const onOpenJob = (jobId: string) => {
    const tabId = `job:${jobId}`;
    if (tabs.some((t) => t.id === tabId)) setActiveTab(tabId);
    else openJobTab(jobId);
  };

  return (
    <div style={{ padding: 28, maxWidth: 1080, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 4 }}>
        <span
          className="mono text-3"
          style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em' }}
        >
          {cli.group}
        </span>
        <span className="text-3">/</span>
        <span className="mono accent" style={{ fontSize: 11 }}>
          $ evalyn
        </span>
        <h1 style={{ fontFamily: 'var(--mono)', fontSize: 22, fontWeight: 500, margin: 0 }}>
          {cli.name}
        </h1>
      </div>
      <p
        style={{
          color: 'var(--text-2)',
          maxWidth: 720,
          fontSize: 13,
          lineHeight: 1.55,
          margin: '8px 0 24px',
        }}
      >
        {cli.blurb}
      </p>

      {/* Saved presets strip lives ABOVE recent runs so starred (sticky)
          items are the first thing the eye lands on. */}
      {!postRunSummary && <SavedPresets cliId={cli.id} />}
      {/* Recent runs strip lives between the header and the form body so
          the user can seed args from a previous invocation in one click. */}
      {!postRunSummary && <RecentRunsStrip cliId={cli.id} />}

      <div
        style={{
          background: 'var(--bg-2)',
          border: '1px solid var(--line)',
          borderRadius: 8,
          padding: 22,
        }}
      >
        {postRunSummary ? (
          <PostRunCard
            summary={postRunSummary}
            onOpenJob={() => onOpenJob(postRunSummary.jobId)}
            onRerun={onRerun}
            onNewRun={onNewRun}
            rerunDisabled={submitting}
          />
        ) : (
          <CliFormBody
            cli={cli}
            values={values}
            setValues={setValues}
            mode={mode}
            fieldErrors={fieldErrors}
          />
        )}
      </div>

      {/* Field-scoped validation summary: a short anchor that jumps to the
          first errored field. Replaces the previous "missing required: ..."
          banner so the detail lives ON the field, not at the top. */}
      {!postRunSummary && errorCount > 0 && (
        <div
          role="status"
          style={{
            marginTop: 12,
            fontSize: 12,
            color: 'var(--fail)',
          }}
        >
          <button
            type="button"
            onClick={focusFirstError}
            className="mono"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--fail)',
              cursor: 'pointer',
              padding: 0,
              textDecoration: 'underline',
              fontSize: 12,
            }}
          >
            {errorCount} {errorCount === 1 ? 'field needs' : 'fields need'} attention →
          </button>
        </div>
      )}

      {/* Server-side errors (non-2xx response) keep the top-banner treatment
          because they aren't field-scoped. */}
      {serverError && !postRunSummary && (
        <div
          role="alert"
          style={{
            marginTop: 12,
            padding: '8px 12px',
            border: '1px solid var(--fail)',
            borderRadius: 6,
            color: 'var(--fail)',
            fontSize: 12,
          }}
        >
          {serverError}
        </div>
      )}

      {!postRunSummary && (
        <div style={{ marginTop: 18, display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            className="btn primary"
            onClick={onRun}
            disabled={submitting}
          >
            {submitting ? '… running' : '▶ Run'}
          </button>
          <button type="button" className="btn" onClick={onCancel} disabled={submitting}>
            Cancel
          </button>
          {/* "Get CLI" promotes the existing copy-command affordance to a
              first-class button — bridges dashboard users to terminal users
              by giving them the literal `evalyn ...` invocation to paste
              into a script or shell. */}
          <button
            type="button"
            className="btn"
            onClick={onCopy}
            title="Copy the literal evalyn command for terminal use"
            aria-label="Get CLI command"
          >
            Get CLI
          </button>
          <span className="grow" />
          <code
            className="mono"
            title={cmd}
            style={{
              fontSize: 11,
              color: 'var(--text-2)',
              maxWidth: 560,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {cmd}
          </code>
        </div>
      )}
    </div>
  );
};

/* ---------- Post-run summary ----------------------------------------- */

/**
 * "Run completed" card. Replaces the form body (not the form tab) so the
 * user can stay in the form context while still having one-click jumps to
 * the job tab and a "rerun" / "new run" affordance.
 *
 * Note: the job tab opening behaviour from `runCli` is preserved — this
 * card sits ALONGSIDE the existing job tab.
 */
const PostRunCard = ({
  summary,
  onOpenJob,
  onRerun,
  onNewRun,
  rerunDisabled,
}: {
  summary: PostRunSummary;
  onOpenJob: () => void;
  onRerun: () => void;
  onNewRun: () => void;
  rerunDisabled: boolean;
}) => {
  const ok = summary.exitCode === 0;
  const duration = summary.duration ?? formatElapsed(summary.startedAt, summary.completedAt);
  return (
    <div role="region" aria-label="run completed">
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 10,
          marginBottom: 8,
        }}
      >
        <span
          aria-hidden
          style={{
            color: ok ? 'var(--ok, #6fbf73)' : 'var(--fail)',
            fontSize: 16,
          }}
        >
          {ok ? '✓' : '✗'}
        </span>
        <span style={{ fontSize: 14, color: 'var(--text-0)', fontWeight: 500 }}>
          {ok ? 'Run completed' : 'Run failed'}
        </span>
        <span className="text-3" style={{ fontSize: 12 }}>
          · {duration}
          {summary.exitCode != null && summary.exitCode !== 0
            ? ` · exit ${summary.exitCode}`
            : ''}
        </span>
      </div>
      <div className="mono text-2" style={{ fontSize: 12, marginBottom: 16 }}>
        {summary.jobId}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        <button type="button" className="btn primary" onClick={onOpenJob}>
          Open in Run viewer →
        </button>
        <button
          type="button"
          className="btn"
          onClick={onRerun}
          disabled={rerunDisabled}
        >
          Re-run with same args
        </button>
        <button type="button" className="btn ghost" onClick={onNewRun}>
          New run
        </button>
      </div>
    </div>
  );
};

export default CliForm;
