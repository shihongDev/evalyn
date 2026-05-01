/**
 * CliForm - the three-mode CLI invocation form.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-cli-forms.jsx (CliForm + CliFormBody).
 * Modes (from `tweaks.cliFormMode`, default `preview`):
 *   form     two-column field grid only
 *   preview  fields on the left, live command + cost/duration estimate on right
 *   raw      assembled command in a textarea
 *
 * Submit -> `store.runCli(cli.id, values)` -> POST `/api/cli/run`.
 * On success the action opens a `job:<id>` tab; the form tab closes.
 */

import { useMemo, useState } from 'react';
import type { CliSchema } from '../types/catalog';
import { useStore } from '../store';
import { buildCli, defaultValues, isFilled, type CliFormValues } from './buildCli';
import ParamField from './ParamField';

export interface CliFormProps {
  cli: CliSchema;
}

/* ---------- Highlighter for the assembled command ----------- */

const CliHighlighted = ({ cmd }: { cmd: string }) => {
  const tokens = cmd.split(/(\s+)/);
  return (
    <span>
      {tokens.map((t, i) => {
        if (/^\s+$/.test(t)) return <span key={i}>{t}</span>;
        if (i === 0) return <span key={i} className="text-2">{t}</span>;
        if (i === 2 && tokens[0] === 'evalyn')
          return <span key={i} className="accent">{t}</span>;
        if (t.startsWith('--'))
          return <span key={i} style={{ color: 'var(--info)' }}>{t}</span>;
        return <span key={i}>{t}</span>;
      })}
    </span>
  );
};

/* ---------- Form body (mode-aware) -------------------------- */

interface BodyProps {
  cli: CliSchema;
  values: CliFormValues;
  setValues: (v: CliFormValues) => void;
  mode: 'form' | 'preview' | 'raw';
}

const CliFormBody = ({ cli, values, setValues, mode }: BodyProps) => {
  const cmd = useMemo(() => buildCli(cli, values), [cli, values]);
  const setVal = (name: string, v: unknown) => setValues({ ...values, [name]: v });
  const basic = cli.params.filter((p) => !p.advanced);
  const advanced = cli.params.filter((p) => p.advanced);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (mode === 'raw') {
    return (
      <div>
        <div className="label" style={{ display: 'flex', alignItems: 'center' }}>
          <span>raw command</span>
          <span className="grow" />
          <span className="kbd">⌘↵ to run</span>
        </div>
        <textarea
          className="textarea"
          rows={3}
          style={{
            fontSize: 13,
            padding: '10px 12px',
            background: 'var(--bg-1)',
            borderColor: 'var(--line-2)',
          }}
          value={cmd}
          onChange={() => {
            /* read-only preview of the assembled command */
          }}
          aria-label="raw command"
        />
        <div className="hint">
          Edit this string directly. Toggle to <b>form</b> or <b>preview</b> mode in Tweaks.
        </div>
      </div>
    );
  }

  const filledCount = cli.params.filter((p) => isFilled(p, values[p.name])).length;

  const fields = (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
      {basic.map((p) => (
        <div
          key={p.name}
          style={{
            gridColumn: ['multiselect', 'long-text'].includes(p.kind)
              ? '1 / -1'
              : undefined,
          }}
        >
          <ParamField
            param={p}
            value={values[p.name]}
            onChange={(v) => setVal(p.name, v)}
          />
        </div>
      ))}
      {advanced.length > 0 && (
        <div style={{ gridColumn: '1 / -1' }}>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '▾' : '▸'} {advanced.length} advanced{' '}
            {advanced.length === 1 ? 'param' : 'params'}
          </button>
        </div>
      )}
      {showAdvanced &&
        advanced.map((p) => (
          <div
            key={p.name}
            style={{
              gridColumn: ['multiselect', 'long-text'].includes(p.kind)
                ? '1 / -1'
                : undefined,
            }}
          >
            <ParamField
              param={p}
              value={values[p.name]}
              onChange={(v) => setVal(p.name, v)}
            />
          </div>
        ))}
    </div>
  );

  if (mode === 'preview') {
    return (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0,1fr) 380px',
          gap: 18,
          alignItems: 'start',
        }}
      >
        <div>{fields}</div>
        <div style={{ position: 'sticky', top: 0 }}>
          <div className="label">live command</div>
          <div
            style={{
              background: 'var(--bg-0)',
              border: '1px solid var(--line-2)',
              borderRadius: 6,
              padding: 14,
              fontFamily: 'var(--mono)',
              fontSize: 12,
              lineHeight: 1.7,
              wordBreak: 'break-word',
            }}
          >
            <CliHighlighted cmd={cmd} />
          </div>
          <div
            style={{
              marginTop: 10,
              padding: '10px 12px',
              background: 'var(--bg-2)',
              borderRadius: 6,
              border: '1px solid var(--line)',
            }}
          >
            <div
              className="text-3 mono"
              style={{
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                marginBottom: 6,
              }}
            >
              predicted
            </div>
            <div className="row" style={{ fontSize: 11 }}>
              <span className="text-2">cost</span>
              <span className="grow" />
              <span className="mono">≈ ${(0.4 + filledCount * 0.18).toFixed(2)}</span>
            </div>
            <div className="row" style={{ fontSize: 11, marginTop: 4 }}>
              <span className="text-2">duration</span>
              <span className="grow" />
              <span className="mono">~ {3 + Math.round(filledCount * 1.4)}m</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // mode === 'form'
  return fields;
};

/* ---------- Top-level CliForm ------------------------------- */

const CliForm = ({ cli }: CliFormProps) => {
  const mode = useStore((s) => s.tweaks.cliFormMode);
  const runCli = useStore((s) => s.runCli);
  const closeTab = useStore((s) => s.closeTab);

  const [values, setValues] = useState<CliFormValues>(() => defaultValues(cli));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cmd = buildCli(cli, values);

  // Required-field client validation. Server re-validates per spec §10.
  const missing = cli.params
    .filter((p) => p.required)
    .filter((p) => {
      const v = values[p.name];
      return v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0);
    });

  const onRun = async () => {
    if (missing.length > 0) {
      setError(`missing required: ${missing.map((p) => p.name).join(', ')}`);
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      await runCli(cli.id, values);
      closeTab(`cli:${cli.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const onCancel = () => closeTab(`cli:${cli.id}`);
  const onCopy = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(cmd).catch(() => {
        /* ignore clipboard errors in restricted contexts */
      });
    }
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
        <span className="grow" />
        <span className="text-3 mono" style={{ fontSize: 11 }}>
          preview:
        </span>
        <code
          className="mono"
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
        <button type="button" className="btn ghost icon" title="Copy" onClick={onCopy}>
          ⎘
        </button>
      </div>
    </div>
  );
};

export default CliForm;
