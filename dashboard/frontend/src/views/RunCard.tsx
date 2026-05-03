/**
 * RunCard - one entry in the Workspace run history feed.
 *
 * Default state is collapsed: a one-line summary (cliId, status, duration,
 * key arg diff vs the previous run for the same cliId). Expanding the card
 * reveals the full Terminal output, the assembled command, and three
 * actions: Edit (re-load args into the active form), Pin / Unpin, Remove.
 *
 * Onboarding (P0):
 * When a run finishes (exit code resolved), the expanded view appends a
 * tinted "next step" card:
 *   - On success: a sage-tinted recommendation pointing at the natural
 *     follow-up CLI (analyze after run-eval, cluster-failures after
 *     analyze, etc.). Click "Open <cli>" to swap the active form.
 *   - On failure: a fail-tinted card whose summary is derived from
 *     pattern-matching the last 50 lines of stderr against ~10 common
 *     signatures (missing API keys, dataset not found, rate limits, ...).
 *     A context-appropriate action button is offered when we can guess
 *     the right recovery; otherwise the card just exposes "See full
 *     output" which scrolls/highlights the embedded Terminal.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore, type RunRecord } from '../store';
import { api, type RunDetail, type RunMetricResult } from '../api';
import { buildCli } from './buildCli';
import Terminal from '../components/Terminal';
import { diffArgs, fmtArgVal } from './diffArgs';
import type { Job, JobLine } from '../types/jobs';

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'var(--accent)',
  running: 'var(--accent)',
  complete: 'var(--pass)',
  failed: 'var(--fail)',
  cancelled: 'var(--fail)',
};

const formatTime = (epochMs: number): string => {
  const d = new Date(epochMs);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
};

/* ===================================================================
 * Next-step recommendations (success path)
 * =================================================================== */

interface NextStep {
  cliId: string;
  reason: string;
}

/** Map cliId → recommended follow-up CLI shown after a successful run. */
const NEXT_STEP: Record<string, NextStep> = {
  quickstart: { cliId: 'list-calls', reason: 'See the traces you just captured.' },
  'one-click': { cliId: 'analyze', reason: 'Analyze your eval results.' },
  'run-eval': { cliId: 'analyze', reason: 'Analyze your eval results.' },
  analyze: { cliId: 'cluster-failures', reason: 'Group failures by similarity.' },
  'build-dataset': {
    cliId: 'run-eval',
    reason: 'Run an evaluation against your new dataset.',
  },
  calibrate: {
    cliId: 'run-eval',
    reason: 'Re-run the eval with calibrated metrics.',
  },
};

/* ===================================================================
 * Failure-pattern signatures (failure path)
 * =================================================================== */

type FailureAction =
  | { kind: 'open-cli'; label: string; cliId: string }
  | { kind: 'open-settings'; label: string }
  | { kind: 'output' };

interface FailurePattern {
  /** Lowercased substrings; ANY match → this pattern wins. */
  needles: string[];
  summary: string;
  action: FailureAction;
}

/**
 * Ordered most-specific → least-specific. First match wins. The
 * unmatched fallback is handled in `matchFailure` so it can include
 * the real exit code in the summary.
 */
const FAILURE_PATTERNS: FailurePattern[] = [
  {
    needles: ['anthropic_api_key'],
    summary: 'ANTHROPIC_API_KEY not set',
    action: { kind: 'open-settings', label: 'Add Anthropic key' },
  },
  {
    needles: ['openai_api_key'],
    summary: 'OPENAI_API_KEY not set',
    action: { kind: 'open-settings', label: 'Add OpenAI key' },
  },
  {
    needles: ['dataset not found', 'no such file'],
    summary: 'Dataset not found at the given path',
    action: { kind: 'open-cli', label: 'Open build-dataset', cliId: 'build-dataset' },
  },
  {
    needles: ['no traces', 'no calls found'],
    summary: 'No traces found in the project',
    action: { kind: 'open-cli', label: 'Open quickstart', cliId: 'quickstart' },
  },
  {
    needles: ['429', 'rate limit', 'ratelimiterror'],
    summary: 'Rate-limited by the provider',
    action: { kind: 'output' },
  },
  {
    needles: ['connection refused', 'network'],
    summary: 'Network error reaching the provider',
    action: { kind: 'output' },
  },
  {
    needles: ['out of memory', 'memoryerror'],
    summary: 'Out of memory',
    action: { kind: 'output' },
  },
  {
    needles: ['permission denied'],
    summary: 'Permission denied (file system)',
    action: { kind: 'output' },
  },
];

interface FailureMatch {
  summary: string;
  action: FailureAction;
}

/** Run the last 50 lines of stderr through the pattern table. */
const matchFailure = (lines: JobLine[] | undefined, exitCode?: number): FailureMatch => {
  // Scope to stderr; some CLIs emit failure context to stdout too, so
  // include both info + stderr + fail/warn streams to be permissive.
  const recent = (lines ?? [])
    .filter((l) => l.kind === 'stderr' || l.kind === 'fail' || l.kind === 'warn')
    .slice(-50)
    .map((l) => l.text.toLowerCase())
    .join('\n');

  if (recent.length > 0) {
    for (const p of FAILURE_PATTERNS) {
      if (p.needles.some((n) => recent.includes(n))) {
        return { summary: p.summary, action: p.action };
      }
    }
  }

  return {
    summary: `Failed (exit code ${exitCode ?? '?'})`,
    action: { kind: 'output' },
  };
};

/* ===================================================================
 * Next-step / failure card
 * =================================================================== */

interface NextStepCardProps {
  step: NextStep;
  onOpen: () => void;
  onSkip: () => void;
}

const NextStepCard = ({ step, onOpen, onSkip }: NextStepCardProps) => (
  <div
    role="region"
    aria-label="Suggested next step"
    style={{
      margin: '0 14px 14px',
      padding: '12px 14px',
      borderRadius: 8,
      background: 'var(--pass-soft, rgba(80, 160, 100, 0.08))',
      border: '1px solid var(--pass, rgba(80, 160, 100, 0.4))',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      fontFamily: 'var(--mono)',
      fontSize: 12,
    }}
  >
    <div style={{ color: 'var(--pass)', fontWeight: 500 }}>
      <span aria-hidden>{'✓'}</span> Done. Suggested next step:
    </div>
    <div style={{ color: 'var(--text-1)', fontSize: 12 }}>{step.reason}</div>
    <div style={{ display: 'flex', gap: 8 }}>
      <button
        type="button"
        className="btn sm"
        onClick={onOpen}
      >
        Open {step.cliId}
      </button>
      <button type="button" className="btn ghost sm" onClick={onSkip}>
        Skip
      </button>
    </div>
  </div>
);

interface FailureCardProps {
  match: FailureMatch;
  onOpenCli: (cliId: string) => void;
  onOpenSettings: () => void;
  onSeeOutput: () => void;
}

const FailureCard = ({
  match,
  onOpenCli,
  onOpenSettings,
  onSeeOutput,
}: FailureCardProps) => {
  const handlePrimary = () => {
    switch (match.action.kind) {
      case 'open-cli':
        onOpenCli(match.action.cliId);
        break;
      case 'open-settings':
        onOpenSettings();
        break;
      case 'output':
        onSeeOutput();
        break;
    }
  };

  // 'output' renders only "See full output"; other kinds get their action
  // button followed by "See full output" as a secondary affordance.
  // Use the inline ternary on `match.action.kind` for primaryLabel so TS
  // narrows `match.action`; a stored boolean does not carry the narrowing.
  const primaryLabel =
    match.action.kind === 'output' ? 'See full output' : match.action.label;
  const isOutputOnly = match.action.kind === 'output';

  return (
    <div
      role="region"
      aria-label="Failure recovery"
      style={{
        margin: '0 14px 14px',
        padding: '12px 14px',
        borderRadius: 8,
        background: 'var(--fail-soft, rgba(220, 90, 90, 0.08))',
        border: '1px solid var(--fail, rgba(220, 90, 90, 0.4))',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        fontFamily: 'var(--mono)',
        fontSize: 12,
      }}
    >
      <div style={{ color: 'var(--fail)', fontWeight: 500 }}>
        <span aria-hidden>{'✗'}</span> Failed: {match.summary}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" className="btn sm" onClick={handlePrimary}>
          {primaryLabel}
        </button>
        {!isOutputOnly && (
          <button type="button" className="btn ghost sm" onClick={onSeeOutput}>
            See full output
          </button>
        )}
      </div>
    </div>
  );
};

export interface RunCardProps {
  run: RunRecord;
  prev: RunRecord | null;
  defaultExpanded?: boolean;
}

/* ===================================================================
 * Promote-rows-to-dataset (P2 Braintrust-inspired)
 *
 * Multi-select on the failed-rows table. Floating action bar appears
 * when ≥1 row is selected; clicking "Add to dataset" opens a modal
 * with a default dataset name. Submitting POSTs to
 * /api/promote/run-failures and surfaces success/error inline.
 * =================================================================== */

const defaultDatasetName = (): string => {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `-${pad(d.getHours())}${pad(d.getMinutes())}`;
  return `regressions-${stamp}`;
};

interface PromoteModalProps {
  count: number;
  onCancel: () => void;
  onSubmit: (datasetName: string) => Promise<void> | void;
  busy: boolean;
  error: string | null;
}

const PromoteModal = ({
  count,
  onCancel,
  onSubmit,
  busy,
  error,
}: PromoteModalProps) => {
  const [name, setName] = useState<string>(defaultDatasetName);
  const trimmed = name.trim();
  const canSubmit = trimmed.length > 0 && !busy;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    void onSubmit(trimmed);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Add rows to dataset"
      onKeyDown={(e) => {
        if (e.key === 'Escape' && !busy) onCancel();
      }}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.32)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => {
        // Click on the backdrop (not the panel) cancels.
        if (e.target === e.currentTarget && !busy) onCancel();
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          background: 'var(--bg-1)',
          border: '1px solid var(--line)',
          borderRadius: 10,
          padding: '18px 20px',
          width: 380,
          maxWidth: '90vw',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          fontFamily: 'var(--mono)',
          fontSize: 12,
        }}
      >
        <div
          style={{ color: 'var(--text-0)', fontSize: 13, fontWeight: 500 }}
        >
          Add {count} {count === 1 ? 'row' : 'rows'} to dataset
        </div>
        <label
          style={{ display: 'flex', flexDirection: 'column', gap: 6 }}
        >
          <span className="text-2" style={{ fontSize: 11 }}>
            Dataset name
          </span>
          <input
            type="text"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
            data-testid="promote-dataset-name"
            style={{
              padding: '8px 10px',
              border: '1px solid var(--line)',
              borderRadius: 6,
              background: 'var(--bg-0)',
              color: 'var(--text-0)',
              fontFamily: 'var(--mono)',
              fontSize: 12,
            }}
          />
        </label>
        {error && (
          <div
            role="alert"
            style={{
              color: 'var(--fail)',
              fontSize: 11,
              padding: '6px 8px',
              background: 'var(--fail-soft, rgba(220,90,90,0.08))',
              borderRadius: 6,
            }}
          >
            {error}
          </div>
        )}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            type="button"
            className="btn ghost sm"
            onClick={onCancel}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn sm"
            disabled={!canSubmit}
            data-testid="promote-submit"
          >
            {busy ? 'Adding...' : 'Add to dataset'}
          </button>
        </div>
      </form>
    </div>
  );
};

const RunCard = ({ run, prev, defaultExpanded = false }: RunCardProps) => {
  const [expanded, setExpanded] = useState<boolean>(defaultExpanded);
  const [nextDismissed, setNextDismissed] = useState<boolean>(false);
  // Brief visual highlight on the embedded Terminal when the user
  // clicks "See full output" on the failure card. Auto-clears.
  const [outputHighlight, setOutputHighlight] = useState<boolean>(false);

  // Promote-to-dataset state. Selection is held local to the card so
  // each card has its own set; modal open/busy/error are also local.
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [promoteModalOpen, setPromoteModalOpen] = useState<boolean>(false);
  const [promoteBusy, setPromoteBusy] = useState<boolean>(false);
  const [promoteError, setPromoteError] = useState<string | null>(null);
  const [promoteSuccess, setPromoteSuccess] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  // Cancellation hook for in-flight promote: keeps a generation counter
  // so a stale resolution can't write back to a remounted/cancelled card.
  const promoteGen = useRef<number>(0);

  const job = useStore((s) => s.jobs.get(run.jobId));
  const cli = useStore((s) => s.catalog.find((c) => c.id === run.cliId));
  const removeRun = useStore((s) => s.removeRun);
  const pinRun = useStore((s) => s.pinRun);
  const unpinRun = useStore((s) => s.unpinRun);
  const editRunArgs = useStore((s) => s.editRunArgs);
  const selectActiveCli = useStore((s) => s.selectActiveCli);
  const openSettings = useStore((s) => s.openSettings);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const attachToChatInput = useStore((s) => s.attachToChatInput);
  const promoteRowsToDataset = useStore((s) => s.promoteRowsToDataset);
  const loadFileTree = useStore((s) => s.loadFileTree);

  // Brief visual confirm on the attach button after the user clicks it.
  // Auto-clears via the timeout so successive clicks re-pulse.
  const [attachPulse, setAttachPulse] = useState<boolean>(false);
  useEffect(() => {
    if (!attachPulse) return;
    const id = setTimeout(() => setAttachPulse(false), 600);
    return () => clearTimeout(id);
  }, [attachPulse]);

  const handleAttach = (e: React.MouseEvent) => {
    e.stopPropagation();
    attachToChatInput({
      id: `run:${run.id}`,
      kind: 'run',
      label: run.id,
      ref: run.id,
    });
    setAttachPulse(true);
  };

  // Fetch the run's results.json the first time the card expands. We
  // only attempt this once per (run.id, expanded) edge; a 404 is
  // expected for in-memory CLI invocations that didn't produce a stored
  // eval-run, in which case the row-level table is silently omitted.
  // Note: we deliberately do NOT include `runDetail`/`detailLoading` in
  // the deps - mutating them inside the effect would re-trigger the
  // cleanup and cancel the in-flight fetch.
  // Track the run.id that was fetched AND succeeded. If a prior attempt
  // failed (network blip, transient backend error) the latch was set
  // but `runDetail` stayed null — without re-checking, a re-expand
  // would skip the fetch forever and the user would have no recovery.
  const detailFetchedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!expanded) return;
    // Skip only when we already SUCCESSFULLY loaded this run's detail.
    // null runDetail (initial state OR prior error) → re-attempt.
    if (detailFetchedFor.current === run.id && runDetail !== null) return;
    let cancelled = false;
    api
      .runDetail(run.id)
      .then((d) => {
        if (!cancelled) {
          setRunDetail(d);
          detailFetchedFor.current = run.id;
        }
      })
      .catch(() => {
        // Swallow - card simply omits the row table. Leave the latch
        // unset so a future re-expand (e.g. user hopes the network
        // recovered) can retry.
        if (!cancelled) setRunDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [expanded, run.id, runDetail]);

  // Build a per-item-id summary from the run's metric_results: each
  // item has an aggregated pass flag (passed iff every metric passed)
  // and the strongest failure reason (when any metric failed).
  const itemRows = useMemo(() => {
    const list = runDetail?.metric_results ?? [];
    const grouped = new Map<string, RunMetricResult[]>();
    for (const m of list) {
      if (!m || typeof m.item_id !== 'string') continue;
      const arr = grouped.get(m.item_id);
      if (arr) arr.push(m);
      else grouped.set(m.item_id, [m]);
    }
    return Array.from(grouped.entries()).map(([itemId, metrics]) => {
      const failedMetric = metrics.find((m) => m.passed === false);
      const passed = !failedMetric;
      const reason =
        (failedMetric?.details?.reason as string | undefined) ?? '';
      return { itemId, passed, reason, metricsCount: metrics.length };
    });
  }, [runDetail]);

  const failedRows = useMemo(
    () => itemRows.filter((r) => !r.passed),
    [itemRows],
  );

  const toggleRow = (itemId: string) => {
    setSelectedRows((prev) => {
      const next = new Set(prev);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const clearSelection = () => setSelectedRows(new Set());

  const handlePromoteSubmit = async (datasetName: string) => {
    const myGen = ++promoteGen.current;
    setPromoteBusy(true);
    setPromoteError(null);
    try {
      const result = await promoteRowsToDataset(
        run.id,
        Array.from(selectedRows),
        datasetName,
      );
      if (myGen !== promoteGen.current) return; // cancelled / superseded
      setPromoteSuccess(
        `Added ${result.item_count} ${result.item_count === 1 ? 'row' : 'rows'} to ${result.dataset_name}.`,
      );
      setPromoteModalOpen(false);
      clearSelection();
      // The new dataset directory exists on disk under
      // .evalyn/data/datasets/<name>/. Refresh the file tree so it
      // appears in the Files sidebar without requiring a page reload.
      // Failure here is non-fatal — the dataset is already written.
      loadFileTree().catch(() => {});
    } catch (err) {
      if (myGen !== promoteGen.current) return;
      const msg = err instanceof Error ? err.message : 'Promote failed';
      let display = msg;
      if (msg.includes('409')) {
        display = 'A dataset with that name already exists. Choose another.';
      } else if (msg.includes('404')) {
        display = 'Run not found on disk. Cannot promote rows.';
      }
      setPromoteError(display);
    } finally {
      if (myGen === promoteGen.current) setPromoteBusy(false);
    }
  };

  // Auto-clear the success toast after a short window so the card
  // doesn't permanently pin the message.
  useEffect(() => {
    if (!promoteSuccess) return;
    const id = setTimeout(() => setPromoteSuccess(null), 4000);
    return () => clearTimeout(id);
  }, [promoteSuccess]);

  const cmd = useMemo(
    () => (cli ? buildCli(cli, run.args) : `evalyn ${run.cliId}`),
    [cli, run.args, run.cliId],
  );
  const diff = useMemo(() => diffArgs(prev?.args ?? null, run.args), [prev, run.args]);

  const statusKey: Job['status'] = job?.status ?? 'pending';

  // Resolve next-step / failure card content. We only render once the
  // job has actually finished (exitCode populated) so the cards do not
  // flash while the run is still streaming.
  const exitCode = job?.exitCode;
  const finished = exitCode != null;
  const isSuccess = finished && exitCode === 0;
  const isFailure = finished && exitCode !== 0;

  const nextStep = isSuccess ? NEXT_STEP[run.cliId] ?? null : null;
  const failureMatch = useMemo(
    () => (isFailure ? matchFailure(job?.lines, exitCode) : null),
    [isFailure, job?.lines, exitCode],
  );

  const outputRef = useRef<HTMLDivElement | null>(null);

  // Auto-clear the highlight after a short pulse.
  useEffect(() => {
    if (!outputHighlight) return;
    const id = setTimeout(() => setOutputHighlight(false), 1500);
    return () => clearTimeout(id);
  }, [outputHighlight]);

  const handleOpenCli = (cliId: string) => {
    selectActiveCli(cliId);
    // Mirror the CliCatalog onPick behavior — switch off any job/file
    // tab so the user lands in the Workspace form for the next CLI.
    setActiveTab(null);
  };

  const handleSeeOutput = () => {
    // Make sure the card is expanded; the output region lives inside
    // the expanded body. Then scroll + flash-highlight it.
    setExpanded(true);
    setOutputHighlight(true);
    requestAnimationFrame(() => {
      outputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  };

  return (
    <div
      data-run-id={run.id}
      style={{
        background: 'var(--bg-1)',
        border: '1px solid var(--line)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 14px',
          fontFamily: 'var(--mono)',
          fontSize: 12,
        }}
      >
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          aria-label={expanded ? 'Collapse run' : 'Expand run'}
          style={{
            all: 'unset',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            flex: 1,
            minWidth: 0,
            cursor: 'pointer',
          }}
        >
          <span
            className="text-3"
            style={{ width: 12, display: 'inline-block', textAlign: 'center' }}
          >
            {expanded ? '▾' : '▸'}
          </span>
          <span className="accent">$</span>
          <span style={{ color: 'var(--text-0)', fontWeight: 500 }}>{run.cliId}</span>
          <span className="text-3">·</span>
          <span style={{ color: STATUS_COLOR[statusKey] }}>{statusKey}</span>
          {job?.exitCode != null && <span className="text-3">· exit {job.exitCode}</span>}
          {job?.duration && <span className="text-3">· {job.duration}</span>}
          {diff.length > 0 && (
            <span
              className="text-3"
              style={{
                marginLeft: 6,
                fontSize: 11,
                color: 'var(--text-2)',
                maxWidth: 360,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              ·{' '}
              {diff
                .slice(0, 1)
                .map((d) => `${d.name} ${fmtArgVal(d.before)} → ${fmtArgVal(d.after)}`)
                .join(', ')}
              {diff.length > 1 ? ` +${diff.length - 1}` : ''}
            </span>
          )}
        </button>
        <button
          type="button"
          className="btn ghost sm"
          data-testid={`attach-run-${run.id}`}
          aria-label="Attach run to chat"
          title="Attach this run to the chat composer"
          onClick={handleAttach}
          style={{
            fontSize: 11,
            padding: '0 6px',
            height: 18,
            lineHeight: '18px',
            color: attachPulse ? 'var(--accent)' : 'var(--text-2)',
            transition: 'color 200ms ease',
            flexShrink: 0,
          }}
        >
          {attachPulse ? '✓ chat' : '+ chat'}
        </button>
        {run.pinned && (
          <span
            className="mono"
            style={{
              fontSize: 10,
              padding: '2px 7px',
              borderRadius: 999,
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
            }}
          >
            pinned
          </span>
        )}
        <span className="text-3" style={{ fontSize: 10 }}>
          {formatTime(run.startedAt)}
        </span>
      </div>

      {expanded && (
        <div style={{ borderTop: '1px solid var(--line)', background: 'var(--bg-0)' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 14px',
              borderBottom: '1px solid var(--line)',
              background: 'var(--bg-1)',
            }}
          >
            <code
              className="mono"
              style={{
                flex: 1,
                fontSize: 11,
                color: 'var(--text-2)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {cmd}
            </code>
            <button
              type="button"
              className="btn ghost sm"
              onClick={(e) => {
                e.stopPropagation();
                editRunArgs(run.id);
              }}
              title="Load these args into the active form"
            >
              Edit
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={(e) => {
                e.stopPropagation();
                if (run.pinned) unpinRun(run.id);
                else pinRun(run.id);
              }}
            >
              {run.pinned ? 'Unpin' : 'Pin'}
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={(e) => {
                e.stopPropagation();
                removeRun(run.id);
              }}
            >
              Remove
            </button>
          </div>
          <div
            ref={outputRef}
            style={{
              padding: '12px 14px',
              maxHeight: 360,
              overflow: 'auto',
              transition: 'box-shadow 0.4s ease',
              boxShadow: outputHighlight
                ? 'inset 0 0 0 2px var(--accent, #4d9fff)'
                : 'none',
            }}
          >
            <Terminal jobId={run.jobId} />
          </div>
          {failedRows.length > 0 && (
            <FailedRowsTable
              rows={failedRows}
              selected={selectedRows}
              onToggle={toggleRow}
              onToggleAll={(check) =>
                setSelectedRows(
                  check ? new Set(failedRows.map((r) => r.itemId)) : new Set(),
                )
              }
            />
          )}
          {promoteSuccess && (
            <div
              role="status"
              data-testid="promote-success"
              style={{
                margin: '0 14px 14px',
                padding: '8px 12px',
                borderRadius: 6,
                background: 'var(--pass-soft, rgba(80,160,100,0.08))',
                color: 'var(--pass)',
                fontFamily: 'var(--mono)',
                fontSize: 11,
              }}
            >
              {promoteSuccess}
            </div>
          )}
          {isSuccess && nextStep && !nextDismissed && (
            <NextStepCard
              step={nextStep}
              onOpen={() => handleOpenCli(nextStep.cliId)}
              onSkip={() => setNextDismissed(true)}
            />
          )}
          {isFailure && failureMatch && (
            <FailureCard
              match={failureMatch}
              onOpenCli={handleOpenCli}
              onOpenSettings={openSettings}
              onSeeOutput={handleSeeOutput}
            />
          )}
        </div>
      )}
      {selectedRows.size > 0 && (
        <PromoteActionBar
          count={selectedRows.size}
          onAdd={() => {
            setPromoteError(null);
            setPromoteModalOpen(true);
          }}
          onCancel={clearSelection}
        />
      )}
      {promoteModalOpen && (
        <PromoteModal
          count={selectedRows.size}
          busy={promoteBusy}
          error={promoteError}
          onCancel={() => {
            // Cancel a possibly-in-flight promote: bumping the gen
            // disconnects the resolver from updating state.
            promoteGen.current += 1;
            setPromoteBusy(false);
            setPromoteError(null);
            setPromoteModalOpen(false);
          }}
          onSubmit={handlePromoteSubmit}
        />
      )}
    </div>
  );
};

/* ===================================================================
 * Failed rows table + floating action bar (P2 promote UI)
 * =================================================================== */

interface FailedRow {
  itemId: string;
  passed: boolean;
  reason: string;
  metricsCount: number;
}

interface FailedRowsTableProps {
  rows: FailedRow[];
  selected: Set<string>;
  onToggle: (itemId: string) => void;
  onToggleAll: (check: boolean) => void;
}

const FailedRowsTable = ({
  rows,
  selected,
  onToggle,
  onToggleAll,
}: FailedRowsTableProps) => {
  const allChecked = rows.length > 0 && rows.every((r) => selected.has(r.itemId));
  const someChecked = rows.some((r) => selected.has(r.itemId));
  return (
    <div
      data-testid="failed-rows-table"
      style={{
        margin: '0 14px 14px',
        border: '1px solid var(--line)',
        borderRadius: 8,
        overflow: 'hidden',
        fontFamily: 'var(--mono)',
        fontSize: 11,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '8px 10px',
          background: 'var(--bg-1)',
          borderBottom: '1px solid var(--line)',
        }}
      >
        <input
          type="checkbox"
          aria-label="Select all failed rows"
          checked={allChecked}
          ref={(el) => {
            if (el) el.indeterminate = !allChecked && someChecked;
          }}
          onChange={(e) => onToggleAll(e.target.checked)}
        />
        <span className="text-2">
          Failed rows · {rows.length}
        </span>
      </div>
      <div style={{ maxHeight: 220, overflow: 'auto' }}>
        {rows.map((r) => {
          const checked = selected.has(r.itemId);
          return (
            <label
              key={r.itemId}
              data-testid={`failed-row-${r.itemId}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '6px 10px',
                cursor: 'pointer',
                background: checked ? 'var(--accent-soft, transparent)' : 'transparent',
                borderBottom: '1px solid var(--line)',
              }}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(r.itemId)}
                aria-label={`Select row ${r.itemId}`}
              />
              <span style={{ color: 'var(--fail)' }}>✗</span>
              <span
                style={{
                  fontFamily: 'var(--mono)',
                  color: 'var(--text-1)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  flexShrink: 0,
                  maxWidth: 180,
                }}
              >
                {r.itemId}
              </span>
              <span
                className="text-3"
                style={{
                  flex: 1,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {r.reason || '(no reason)'}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
};

interface PromoteActionBarProps {
  count: number;
  onAdd: () => void;
  onCancel: () => void;
}

const PromoteActionBar = ({ count, onAdd, onCancel }: PromoteActionBarProps) => (
  <div
    role="region"
    aria-label="Promote selection"
    data-testid="promote-action-bar"
    style={{
      position: 'sticky',
      bottom: 8,
      margin: '0 14px 14px',
      padding: '8px 12px',
      borderRadius: 8,
      background: 'var(--bg-1)',
      border: '1px solid var(--line)',
      boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      fontFamily: 'var(--mono)',
      fontSize: 12,
      zIndex: 5,
    }}
  >
    <span style={{ color: 'var(--text-1)' }}>
      {count} {count === 1 ? 'row' : 'rows'} selected
    </span>
    <span style={{ flex: 1 }} />
    <button
      type="button"
      className="btn ghost sm"
      onClick={onCancel}
      data-testid="promote-cancel"
    >
      Cancel
    </button>
    <button
      type="button"
      className="btn sm"
      onClick={onAdd}
      data-testid="promote-add"
    >
      Add to dataset
    </button>
  </div>
);

export default RunCard;
