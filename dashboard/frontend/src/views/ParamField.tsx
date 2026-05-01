/**
 * ParamField - dispatches a control on `param.kind`.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-cli-forms.jsx (ParamField).
 * All seven kinds are handled:
 *   bool         two-button toggle (true / false)
 *   string       <input>
 *   number       <input type=number>
 *   select       native <select>
 *   multiselect  chip toggles
 *   path         <input> with `./...` placeholder
 *   long-text    <textarea>
 */

import type { ParamSchema } from '../types/catalog';

export interface ParamFieldProps {
  param: ParamSchema;
  value: unknown;
  onChange: (value: unknown) => void;
}

const Label = ({ param, htmlFor }: { param: ParamSchema; htmlFor?: string }) => (
  <label htmlFor={htmlFor} className="label">
    <span style={{ color: 'var(--text-1)' }}>{param.name.replace(/_/g, ' ')}</span>
    {param.required && (
      <span className="accent" style={{ marginLeft: 4 }}>
        *
      </span>
    )}
    {param.advanced && (
      <span
        className="text-3 mono"
        style={{
          marginLeft: 6,
          fontSize: 9,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}
      >
        advanced
      </span>
    )}
  </label>
);

const ParamField = ({ param, value, onChange }: ParamFieldProps) => {
  const id = `p-${param.name}`;
  const help = param.help ? <div className="hint">{param.help}</div> : null;

  if (param.kind === 'bool') {
    return (
      <div>
        <Label param={param} />
        <div style={{ display: 'flex', gap: 6 }}>
          {[true, false].map((b) => {
            const on = value === b;
            return (
              <button
                key={String(b)}
                type="button"
                onClick={() => onChange(b)}
                className="btn sm"
                style={{
                  background: on ? 'var(--accent-soft)' : 'var(--bg-2)',
                  color: on ? 'var(--accent)' : 'var(--text-2)',
                  borderColor: on ? 'rgba(255,122,61,0.4)' : 'var(--line)',
                }}
              >
                {b ? 'true' : 'false'}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }

  if (param.kind === 'select') {
    return (
      <div>
        <Label param={param} htmlFor={id} />
        <select
          id={id}
          className="select"
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
        >
          {(param.options ?? []).map((opt) => (
            <option key={opt} value={opt}>
              {opt || '(any)'}
            </option>
          ))}
        </select>
        {help}
      </div>
    );
  }

  if (param.kind === 'multiselect') {
    const arr = Array.isArray(value) ? (value as string[]) : [];
    return (
      <div>
        <Label param={param} />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {(param.options ?? []).map((opt) => {
            const on = arr.includes(opt);
            return (
              <button
                key={opt}
                type="button"
                className="btn sm"
                onClick={() => onChange(on ? arr.filter((x) => x !== opt) : [...arr, opt])}
                style={{
                  background: on ? 'var(--accent-soft)' : 'var(--bg-2)',
                  color: on ? 'var(--accent)' : 'var(--text-2)',
                  borderColor: on ? 'rgba(255,122,61,0.4)' : 'var(--line)',
                }}
              >
                {on ? 'on ' : 'off '}
                {opt}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }

  if (param.kind === 'number') {
    return (
      <div>
        <Label param={param} htmlFor={id} />
        <input
          id={id}
          type="number"
          className="input"
          value={value === '' || value === undefined || value === null ? '' : String(value)}
          onChange={(e) =>
            onChange(e.target.value === '' ? '' : Number(e.target.value))
          }
        />
        {help}
      </div>
    );
  }

  if (param.kind === 'long-text') {
    return (
      <div>
        <Label param={param} htmlFor={id} />
        <textarea
          id={id}
          className="textarea"
          rows={4}
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
        />
        {help}
      </div>
    );
  }

  // string / path
  const placeholder = param.kind === 'path' ? './...' : '';
  return (
    <div>
      <Label param={param} htmlFor={id} />
      <input
        id={id}
        className="input"
        value={(value as string) ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
      {help}
    </div>
  );
};

export default ParamField;
