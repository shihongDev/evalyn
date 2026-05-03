/**
 * Workspace - "Action + Run History" view.
 *
 * Replaces the earlier Notebook prototype. Shape:
 *   - Active form panel at the top (one CLI at a time, switchable).
 *   - Pinned-runs row (only visible when at least one run is pinned).
 *   - Feed of past runs, newest first. Each card collapses to a one-line
 *     summary; expanding reveals the streamed Terminal output and the
 *     Edit / Pin / Remove actions.
 *
 * Form values are component-local state. They are only lifted into the
 * store as part of a `RunRecord` when Run is clicked.
 */

import { useEffect, useMemo, useReducer } from 'react';
import { useStore, mergeFormValues, type RunRecord } from '../store';
import CliFormBody from './CliFormBody';
import { buildCli } from './buildCli';
import RunCard from './RunCard';
import { diffArgs, fmtArgVal } from './diffArgs';
import type { CliSchema } from '../types/catalog';

interface ActiveFormProps {
  cli: CliSchema;
}

type FormStateAction =
  | { kind: 'set'; cliId: string; values: Record<string, unknown> }
  | { kind: 'seed'; cliId: string; values: Record<string, unknown> }
  | { kind: 'submitting'; value: boolean }
  | { kind: 'error'; value: string | null };

interface FormState {
  valuesByCli: Record<string, Record<string, unknown>>;
  submitting: boolean;
  error: string | null;
}

const formReducer = (s: FormState, a: FormStateAction): FormState => {
  switch (a.kind) {
    case 'set':
    case 'seed':
      return {
        ...s,
        valuesByCli: { ...s.valuesByCli, [a.cliId]: a.values },
      };
    case 'submitting':
      return { ...s, submitting: a.value };
    case 'error':
      return { ...s, error: a.value };
  }
};

const initialFormState: FormState = {
  valuesByCli: {},
  submitting: false,
  error: null,
};

const ActiveForm = ({ cli }: ActiveFormProps) => {
  const mode = useStore((s) => s.tweaks.cliFormMode);
  const runActive = useStore((s) => s.runActive);
  const runHistory = useStore((s) => s.runHistory);
  const editTarget = useStore((s) => s.activeFormSeed);
  const clearActiveFormSeed = useStore((s) => s.clearActiveFormSeed);

  // Per-CLI form values are component-local — the store does not re-render
  // when the user types. Lifted to the store only when Run is clicked.
  const [state, dispatch] = useReducer(formReducer, initialFormState);
  const { valuesByCli, submitting, error } = state;

  // editRunArgs handoff: the store sets `activeFormSeed` to mark the run
  // whose args should populate this form. We treat the seed as an
  // external-system signal and dispatch the merge + clear once, which is
  // the linter-approved shape for useEffect.
  useEffect(() => {
    if (!editTarget) return;
    if (editTarget.cliId !== cli.id) return;
    dispatch({
      kind: 'seed',
      cliId: cli.id,
      values: mergeFormValues(cli, editTarget.args),
    });
    clearActiveFormSeed();
  }, [editTarget, cli, clearActiveFormSeed]);

  const cached = valuesByCli[cli.id];
  // mergeFormValues handles defaults for new params + drops removed ones,
  // so the form survives a catalog refresh.
  const values = useMemo(() => mergeFormValues(cli, cached), [cli, cached]);
  const setValues = (v: Record<string, unknown>) =>
    dispatch({ kind: 'set', cliId: cli.id, values: v });

  const cmd = useMemo(() => buildCli(cli, values), [cli, values]);

  // Diff vs. the previous run for the same CLI (most recent first).
  const previousRun = useMemo(() => {
    for (let i = runHistory.length - 1; i >= 0; i--) {
      if (runHistory[i].cliId === cli.id) return runHistory[i];
    }
    return null;
  }, [runHistory, cli.id]);
  const paramDiff = useMemo(
    () => (previousRun ? diffArgs(previousRun.args, values) : []),
    [previousRun, values],
  );

  const onRun = async () => {
    dispatch({ kind: 'error', value: null });
    dispatch({ kind: 'submitting', value: true });
    try {
      await runActive(values);
    } catch (err) {
      dispatch({
        kind: 'error',
        value: err instanceof Error ? err.message : String(err),
      });
    } finally {
      dispatch({ kind: 'submitting', value: false });
    }
  };

  const onCopy = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText && cmd) {
      navigator.clipboard.writeText(cmd).catch(() => {
        /* ignore clipboard errors in restricted contexts */
      });
    }
  };

  return (
    <div
      style={{
        background: 'var(--bg-1)',
        border: '1px solid var(--line)',
        borderRadius: 10,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '12px 18px',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-2)',
        }}
      >
        <span
          className="text-3"
          style={{
            fontSize: 10,
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
          }}
        >
          Active command
        </span>
        <span className="mono accent" style={{ fontSize: 12 }}>
          $ evalyn
        </span>
        <span className="mono" style={{ fontSize: 14, color: 'var(--text-0)', fontWeight: 500 }}>
          {cli.name}
        </span>
        <span className="text-3 mono" style={{ fontSize: 10 }}>
          {cli.group}
        </span>
        {paramDiff.length > 0 && (
          <span
            className="mono"
            title="changed since last run"
            style={{
              marginLeft: 8,
              fontSize: 10,
              padding: '3px 8px',
              borderRadius: 999,
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
              border: '1px solid var(--accent)',
            }}
          >
            changed: {paramDiff
              .slice(0, 2)
              .map((d) => `${d.name} ${fmtArgVal(d.before)} → ${fmtArgVal(d.after)}`)
              .join(', ')}
            {paramDiff.length > 2 ? ` +${paramDiff.length - 2}` : ''}
          </span>
        )}
        <span className="grow" />
        <CliSwitcher activeCliId={cli.id} />
      </div>
      <div style={{ padding: 22 }}>
        <p
          style={{
            color: 'var(--text-2)',
            fontSize: 12,
            lineHeight: 1.55,
            margin: '0 0 16px',
          }}
        >
          {cli.blurb}
        </p>
        <div
          style={{
            background: 'var(--bg-2)',
            border: '1px solid var(--line)',
            borderRadius: 8,
            padding: 22,
          }}
        >
          <CliFormBody cli={cli} values={values} setValues={setValues} mode={mode} />
        </div>

        {error && (
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
            {error}
          </div>
        )}

        <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            className="btn primary"
            onClick={onRun}
            disabled={submitting}
          >
            {submitting ? 'Running...' : 'Run'}
          </button>
          <span className="grow" />
          <span className="text-3 mono" style={{ fontSize: 11 }}>
            preview:
          </span>
          <code
            className="mono"
            style={{
              fontSize: 11,
              color: 'var(--text-2)',
              maxWidth: 420,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {cmd}
          </code>
          <button type="button" className="btn ghost icon" title="Copy" onClick={onCopy}>
            Copy
          </button>
        </div>
      </div>
    </div>
  );
};

interface SwitcherProps {
  activeCliId: string;
}

const CliSwitcher = ({ activeCliId }: SwitcherProps) => {
  const catalog = useStore((s) => s.catalog);
  const selectActiveCli = useStore((s) => s.selectActiveCli);
  return (
    <select
      aria-label="Active CLI"
      className="mono"
      value={activeCliId}
      onChange={(e) => selectActiveCli(e.target.value)}
      style={{
        background: 'var(--bg-1)',
        color: 'var(--text-0)',
        border: '1px solid var(--line-2)',
        borderRadius: 6,
        padding: '4px 10px',
        fontSize: 12,
        fontFamily: 'var(--mono)',
      }}
    >
      {catalog.map((c) => (
        <option key={c.id} value={c.id}>
          {c.id}
        </option>
      ))}
    </select>
  );
};

const EmptyForm = () => {
  const catalog = useStore((s) => s.catalog);
  const selectActiveCli = useStore((s) => s.selectActiveCli);
  return (
    <div
      style={{
        background: 'var(--bg-1)',
        border: '1px dashed var(--line-2)',
        borderRadius: 10,
        padding: 28,
        textAlign: 'center',
        color: 'var(--text-2)',
      }}
    >
      <div
        className="text-3"
        style={{
          fontSize: 11,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          marginBottom: 10,
        }}
      >
        Active command
      </div>
      <p style={{ fontSize: 13, margin: '0 0 14px' }}>
        Choose a command from the left to begin.
      </p>
      {catalog.length > 0 && (
        <button
          type="button"
          className="btn"
          onClick={() => selectActiveCli(catalog[0].id)}
        >
          Use {catalog[0].id}
        </button>
      )}
    </div>
  );
};

const Workspace = () => {
  const activeCliId = useStore((s) => s.activeCliId);
  const catalog = useStore((s) => s.catalog);
  const runHistory = useStore((s) => s.runHistory);

  const cli = activeCliId ? catalog.find((c) => c.id === activeCliId) ?? null : null;

  // Newest first. Pinned runs render in their own row above the feed.
  const ordered = useMemo(
    () => [...runHistory].reverse(),
    [runHistory],
  );
  const pinned = ordered.filter((r) => r.pinned);
  const unpinned = ordered.filter((r) => !r.pinned);

  // Build a map from runId -> previous run (chronological) for the same
  // cliId. Used by the card's one-line "key arg diff vs previous" summary.
  const prevByRunId = useMemo(() => {
    const out = new Map<string, RunRecord | null>();
    for (let i = 0; i < runHistory.length; i++) {
      const r = runHistory[i];
      let prev: RunRecord | null = null;
      for (let j = i - 1; j >= 0; j--) {
        if (runHistory[j].cliId === r.cliId) {
          prev = runHistory[j];
          break;
        }
      }
      out.set(r.id, prev);
    }
    return out;
  }, [runHistory]);

  return (
    <div
      style={{
        padding: '24px 28px 60px',
        maxWidth: 1080,
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 18,
      }}
    >
      {cli ? <ActiveForm cli={cli} /> : <EmptyForm />}

      {pinned.length > 0 && (
        <div>
          <div
            className="text-3 mono"
            style={{
              fontSize: 10,
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              marginBottom: 8,
            }}
          >
            Pinned · {pinned.length}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {pinned.map((r) => (
              <RunCard key={r.id} run={r} prev={prevByRunId.get(r.id) ?? null} defaultExpanded />
            ))}
          </div>
        </div>
      )}

      <div>
        <div
          className="text-3 mono"
          style={{
            fontSize: 10,
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: 8,
          }}
        >
          Run history · {unpinned.length}
        </div>
        {unpinned.length === 0 ? (
          <div
            style={{
              padding: 28,
              border: '1px dashed var(--line-2)',
              borderRadius: 10,
              color: 'var(--text-2)',
              textAlign: 'center',
              fontSize: 12,
            }}
          >
            Your runs will appear here.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {unpinned.map((r) => (
              <RunCard key={r.id} run={r} prev={prevByRunId.get(r.id) ?? null} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Workspace;
