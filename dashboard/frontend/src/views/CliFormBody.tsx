/**
 * CliFormBody - mode-aware body of a CLI form.
 *
 * Extracted from CliForm.tsx so the Workspace's active form (`Workspace.tsx`)
 * can reuse the same field rendering. Renders fields in `form` / `preview`
 * / `raw` mode based on the tweak.
 *
 * The component is purely controlled: parent owns `values` and provides the
 * `setValues` setter. No store coupling beyond `tweaks.cliFormMode` lookups
 * by callers.
 */

import { useMemo, useState } from 'react';
import type { CliSchema } from '../types/catalog';
import { buildCli, type CliFormValues } from './buildCli';
import ParamField from './ParamField';
import { reverseParseCli } from '../lib/reverseParseCli';

/* ---------- Paste-raw modal ----------------------------------------- */

// Errors from reverseParseCli that mean the input is structurally wrong
// (no command, wrong command, empty input). Anything else is a soft warning
// we surface but still allow the partial parse to apply.
const FATAL_PARSE_PREFIXES = [
  'expected first token',
  'command `',
  'missing CLI subcommand',
  'empty input',
];

interface PasteRawModalProps {
  cli: CliSchema;
  onApply: (values: Record<string, unknown>) => void;
  onClose: () => void;
}

/**
 * Modal that reverse-parses a literal `evalyn ...` command into form values.
 * On Apply, the merged values are handed back to the parent which calls
 * `setValues`. Soft errors (unknown flags, type coercion failures) are
 * shown inline so the user can still dismiss and apply a partial parse.
 */
const PasteRawModal = ({ cli, onApply, onClose }: PasteRawModalProps) => {
  const [text, setText] = useState('');
  const [errors, setErrors] = useState<string[]>([]);

  const onParse = () => {
    const result = reverseParseCli(text, cli);
    setErrors(result.errors);
    const fatal = result.errors.some((e) =>
      FATAL_PARSE_PREFIXES.some((prefix) => e.startsWith(prefix)),
    );
    if (fatal) return;
    onApply(result.values);
    onClose();
  };

  return (
    <div
      data-testid="paste-raw-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Paste a CLI"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-1)',
          border: '1px solid var(--line)',
          borderRadius: 8,
          width: 'min(640px, 90vw)',
          padding: 18,
          boxShadow: 'var(--shadow-soft)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
          <span style={{ fontFamily: 'var(--serif)', fontSize: 16 }}>Paste a CLI</span>
          <span className="grow" />
          <button
            type="button"
            className="btn ghost icon"
            aria-label="Close"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <textarea
          data-testid="paste-raw-textarea"
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={`evalyn ${cli.id} --dataset evals/spam.jsonl ...`}
          spellCheck={false}
          style={{
            width: '100%',
            boxSizing: 'border-box',
            fontFamily: 'var(--mono)',
            fontSize: 12,
            padding: '8px 10px',
            background: 'var(--bg-0)',
            border: '1px solid var(--line-2)',
            borderRadius: 5,
            color: 'var(--text-1)',
            resize: 'vertical',
          }}
        />
        {errors.length > 0 && (
          <ul
            data-testid="paste-raw-errors"
            style={{
              margin: '8px 0 0',
              paddingLeft: 18,
              fontSize: 11,
              color: 'var(--warn)',
              lineHeight: 1.55,
            }}
          >
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        )}
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button
            type="button"
            className="btn primary"
            data-testid="paste-raw-apply"
            onClick={onParse}
            disabled={!text.trim()}
          >
            Parse and load into form
          </button>
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

/* ---------- Highlighter for the assembled command ----------- */

export const CliHighlighted = ({ cmd }: { cmd: string }) => {
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

export interface CliFormBodyProps {
  cli: CliSchema;
  values: CliFormValues;
  setValues: (v: CliFormValues) => void;
  mode: 'form' | 'preview' | 'raw';
  /**
   * Field-scoped errors keyed by `param.name`. Forwarded to ParamField so
   * the row paints a red left-border + inline message. Pass `{}` (or omit)
   * when there is nothing to flag.
   */
  fieldErrors?: Record<string, string>;
}

/** Wide-row kinds need to span both grid columns to fit their contents. */
const WIDE_KINDS = new Set(['multiselect', 'long-text']);
const gridColumnFor = (kind: string): string | undefined =>
  WIDE_KINDS.has(kind) ? '1 / -1' : undefined;

const CliFormBody = ({ cli, values, setValues, mode, fieldErrors }: CliFormBodyProps) => {
  const errorFor = (name: string): string | undefined => fieldErrors?.[name];
  const cmd = useMemo(() => buildCli(cli, values), [cli, values]);
  const setVal = (name: string, v: unknown) => setValues({ ...values, [name]: v });
  const [pasteOpen, setPasteOpen] = useState(false);

  // Merge parsed values onto current values so the user keeps anything the
  // pasted command did not override. The parser only emits keys it
  // recognised, so this is a strict "shallow merge of recognized fields".
  const onPasteApply = (parsed: Record<string, unknown>) => {
    setValues({ ...values, ...parsed });
  };

  const pasteRawButton = (
    <button
      type="button"
      className="btn ghost sm"
      data-testid="paste-raw-open"
      onClick={() => setPasteOpen(true)}
      title="Paste a literal evalyn command and parse it into the form"
      aria-label="Paste a CLI"
      style={{ fontSize: 11 }}
    >
      ✎ paste raw
    </button>
  );

  const modal = pasteOpen ? (
    <PasteRawModal
      cli={cli}
      onApply={onPasteApply}
      onClose={() => setPasteOpen(false)}
    />
  ) : null;
  // Approach D: required + curated `essential` params render in the default
  // form; everything else lives behind a single "Show all options" disclosure.
  // The `advanced` flag is informational only (rendered as a chip below) — it
  // no longer drives partitioning.
  const defaultVisible = cli.params.filter((p) => p.required || p.essential);
  const extra = cli.params.filter((p) => !p.required && !p.essential);
  const [showExtra, setShowExtra] = useState(false);
  // Auto-expand the "Show all options" disclosure when any errored field
  // lives inside it — otherwise the inline error would be invisible.
  const hasHiddenError = extra.some((p) => errorFor(p.name));
  const expanded = showExtra || hasHiddenError;

  if (mode === 'raw') {
    const onCopy = () => {
      // best-effort copy; ignore failures (older browsers / insecure context)
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        void navigator.clipboard.writeText(cmd);
      }
    };
    return (
      <div>
        <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>raw command</span>
          <span className="grow" />
          {pasteRawButton}
          <button
            type="button"
            className="btn ghost sm"
            onClick={onCopy}
            aria-label="copy raw command"
          >
            Copy
          </button>
        </div>
        <pre
          aria-label="raw command"
          style={{
            margin: 0,
            fontSize: 13,
            padding: '10px 12px',
            background: 'var(--bg-1)',
            border: '1px solid var(--line-2)',
            borderRadius: 5,
            fontFamily: 'var(--mono)',
            color: 'var(--text-1)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          <CliHighlighted cmd={cmd} />
        </pre>
        <div className="hint">
          The command that will run, with values from the form above.
        </div>
        {modal}
      </div>
    );
  }

  const fieldsToolbar = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: 10,
      }}
    >
      <span className="grow" />
      {pasteRawButton}
    </div>
  );

  const fields = (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
      {defaultVisible.map((p) => (
        <div key={p.name} style={{ gridColumn: gridColumnFor(p.kind) }}>
          <ParamField
            param={p}
            value={values[p.name]}
            onChange={(v) => setVal(p.name, v)}
            error={errorFor(p.name)}
          />
        </div>
      ))}
      {extra.length > 0 && (
        <div style={{ gridColumn: '1 / -1' }}>
          <button
            type="button"
            className="btn ghost sm"
            aria-expanded={expanded}
            onClick={() => setShowExtra(!showExtra)}
          >
            {expanded ? '▾' : '▸'} Show all options ({extra.length})
          </button>
        </div>
      )}
      {expanded &&
        extra.map((p) => (
          <div key={p.name} style={{ gridColumn: gridColumnFor(p.kind) }}>
            <ParamField
              param={p}
              value={values[p.name]}
              onChange={(v) => setVal(p.name, v)}
              error={errorFor(p.name)}
            />
            {p.advanced && (
              <span
                className="mono text-3"
                style={{
                  fontSize: 9,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  marginTop: 2,
                  display: 'inline-block',
                }}
              >
                advanced
              </span>
            )}
          </div>
        ))}
    </div>
  );

  if (mode === 'preview') {
    return (
      <div>
        {fieldsToolbar}
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
          </div>
        </div>
        {modal}
      </div>
    );
  }

  // mode === 'form'
  return (
    <div>
      {fieldsToolbar}
      {fields}
      {modal}
    </div>
  );
};

export default CliFormBody;
