/**
 * ParamField - dispatches a control on `param.kind`.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-cli-forms.jsx (ParamField).
 * All seven kinds are handled:
 *   bool         three-state segmented control (default / on / off)
 *   string       <input>
 *   number       <input type=number>
 *   select       native <select>
 *   multiselect  chip toggles
 *   path         combobox: typeable input + dropdown of fileTree paths
 *   long-text    <textarea>
 */

import { useMemo, useState } from 'react';
import type { ParamSchema } from '../types/catalog';
import type { FileNode } from '../types/jobs';
import { useStore } from '../store';

export interface ParamFieldProps {
  param: ParamSchema;
  value: unknown;
  onChange: (value: unknown) => void;
  /**
   * Field-scoped error message. When set, renders a red left-border on the
   * row + an inline message below the control. The CliForm wraps each field
   * with a `data-param-name` attribute so the "fields need attention" anchor
   * can scroll to the first errored field.
   */
  error?: string;
}

/** Flatten the workspace file tree to a list of slash-separated paths. */
const flattenTree = (nodes: FileNode[], prefix = ''): string[] => {
  const out: string[] = [];
  for (const n of nodes) {
    const here = prefix ? `${prefix}/${n.name}` : n.name;
    out.push(here);
    if (n.kind === 'folder' && n.children?.length) {
      out.push(...flattenTree(n.children, here));
    }
  }
  return out;
};

const MAX_DROPDOWN_OPTIONS = 12;

/**
 * Path combobox: a typeable text input with a focus-triggered dropdown of
 * matching paths from the workspace file tree (`store.fileTree`). Typing is
 * non-blocking — any string value is allowed; the dropdown is a hint only.
 */
const PathCombobox = ({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (v: string) => void;
}) => {
  const fileTree = useStore((s) => s.fileTree);
  const [focused, setFocused] = useState(false);

  const allPaths = useMemo(() => flattenTree(fileTree), [fileTree]);
  const query = value.toLowerCase();
  const matches = useMemo(
    () => (query ? allPaths.filter((p) => p.toLowerCase().includes(query)) : allPaths),
    [allPaths, query],
  );
  const visible = matches.slice(0, MAX_DROPDOWN_OPTIONS);
  const overflow = matches.length - visible.length;
  const showDropdown = focused && allPaths.length > 0;

  return (
    <div style={{ position: 'relative' }}>
      <input
        id={id}
        className="input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        // Delay blur so option mousedown can fire first.
        onBlur={() => setTimeout(() => setFocused(false), 120)}
        placeholder="path/to/file or directory"
        autoComplete="off"
        role="combobox"
        aria-expanded={showDropdown}
        aria-autocomplete="list"
      />
      {showDropdown && (
        <div
          role="listbox"
          style={{
            position: 'absolute',
            top: 'calc(100% + 2px)',
            left: 0,
            right: 0,
            zIndex: 20,
            background: 'var(--bg-2)',
            border: '1px solid var(--line)',
            borderRadius: 5,
            maxHeight: 240,
            overflowY: 'auto',
            boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
          }}
        >
          {visible.length === 0 ? (
            <div
              className="hint"
              style={{ padding: '6px 10px', fontFamily: 'var(--mono)' }}
            >
              no matches in .evalyn/
            </div>
          ) : (
            visible.map((p) => (
              <div
                key={p}
                role="option"
                // Use mousedown so the click registers before the input blurs.
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(p);
                  setFocused(false);
                }}
                style={{
                  padding: '5px 10px',
                  fontFamily: 'var(--mono)',
                  fontSize: 12,
                  color: 'var(--text-1)',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'var(--bg-3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                {p}
              </div>
            ))
          )}
          {overflow > 0 && (
            <div
              className="hint"
              style={{
                padding: '5px 10px',
                fontFamily: 'var(--mono)',
                borderTop: '1px solid var(--line)',
              }}
            >
              ...and {overflow} more
            </div>
          )}
        </div>
      )}
    </div>
  );
};

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

/**
 * Three-state bool segmented control: [ default ] [ on ] [ off ].
 *
 * State machine:
 *   - "default" -> stored value === param.default; buildCli skips the flag
 *     because (cur === def) in its bool path.
 *   - "on"      -> stored value === true.
 *   - "off"     -> stored value === false.
 *
 * The displayed selection is computed from the current value relative to the
 * argparse default. We never mutate `param.default` itself.
 */
type BoolMode = 'default' | 'on' | 'off';

const BoolToggle = ({
  param,
  value,
  onChange,
}: {
  param: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
}) => {
  const def = !!param.default;
  // Track whether the user has explicitly picked a non-default segment.
  // Without this, clicking "on" or "off" with a value that happens to equal
  // the argparse default would collapse the visual to "default" — the
  // command output is correct (buildCli omits the flag either way) but the
  // user gets no feedback that their click registered.
  const [touched, setTouched] = useState(false);

  // Mode resolution:
  //   - touched: render the literal value (true→on, false→off, else default)
  //   - untouched: collapse to "default" if value matches def or is unset
  let mode: BoolMode;
  if (value !== true && value !== false) mode = 'default';
  else if (!touched && value === def) mode = 'default';
  else if (value === true) mode = 'on';
  else mode = 'off';

  const segments: Array<{ key: BoolMode; label: string; value: unknown }> = [
    { key: 'default', label: 'default', value: def },
    { key: 'on', label: 'on', value: true },
    { key: 'off', label: 'off', value: false },
  ];

  const handleClick = (segKey: BoolMode, segValue: unknown) => {
    // Picking "default" returns to untouched state so the literal value
    // re-collapses; picking on/off marks touched so visual lights up.
    setTouched(segKey !== 'default');
    onChange(segValue);
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 0 }} role="group" aria-label={param.name}>
        {segments.map((seg, i) => {
          const on = mode === seg.key;
          return (
            <button
              key={seg.key}
              type="button"
              onClick={() => handleClick(seg.key, seg.value)}
              className="btn sm"
              aria-pressed={on}
              style={{
                background: on ? 'var(--accent-soft)' : 'var(--bg-2)',
                color: on ? 'var(--accent)' : 'var(--text-2)',
                borderColor: on ? 'rgba(255,122,61,0.4)' : 'var(--line)',
                borderTopRightRadius: i === segments.length - 1 ? undefined : 0,
                borderBottomRightRadius: i === segments.length - 1 ? undefined : 0,
                borderTopLeftRadius: i === 0 ? undefined : 0,
                borderBottomLeftRadius: i === 0 ? undefined : 0,
                marginLeft: i === 0 ? 0 : -1,
                position: 'relative',
                zIndex: on ? 1 : 0,
              }}
            >
              {seg.label}
            </button>
          );
        })}
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        (default = {String(def)})
      </div>
    </div>
  );
};

/**
 * Number control: bare numeric input when no range hint is present, or a
 * numeric input + range slider + unit label when `param.range` is set.
 *
 * Both controls share a single onChange callback. The slider's min/max/step
 * come from `param.range`; if no `step` is specified the control infers one
 * as `(max - min) / 100` (rounded to an integer when both bounds are whole
 * numbers, so int-typed args stay int-typed).
 */
const NumberControl = ({
  id,
  param,
  value,
  onChange,
}: {
  id: string;
  param: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
}) => {
  const range = param.range;
  const display =
    value === '' || value === undefined || value === null ? '' : String(value);

  // Bare input when no range hint - preserves prior behavior exactly.
  if (!range) {
    return (
      <input
        id={id}
        type="number"
        className="input"
        value={display}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
      />
    );
  }

  // Step inference: explicit > min/max-derived. Keep ints integer when the
  // bounds suggest the param is integer-typed.
  const isIntRange =
    Number.isInteger(range.min) && Number.isInteger(range.max) && range.step === undefined;
  const inferredStep = isIntRange
    ? Math.max(1, Math.round((range.max - range.min) / 100))
    : (range.max - range.min) / 100;
  const step = range.step ?? inferredStep;

  // Slider needs a numeric value; clamp display value into [min, max] for the
  // slider only - the numeric input still shows whatever the user typed.
  const numeric = typeof value === 'number' ? value : Number(value);
  const sliderValue = Number.isFinite(numeric)
    ? Math.min(range.max, Math.max(range.min, numeric))
    : range.min;

  const handleChange = (raw: string) => {
    if (raw === '') {
      onChange('');
      return;
    }
    const next = Number(raw);
    onChange(Number.isFinite(next) ? next : raw);
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <input
        id={id}
        type="number"
        className="input"
        min={range.min}
        max={range.max}
        step={step}
        value={display}
        onChange={(e) => handleChange(e.target.value)}
        style={{ width: 96, flex: '0 0 auto' }}
      />
      <input
        type="range"
        aria-label={`${param.name} slider`}
        min={range.min}
        max={range.max}
        step={step}
        value={sliderValue}
        onChange={(e) => handleChange(e.target.value)}
        style={{ flex: '1 1 auto', minWidth: 80 }}
      />
      {param.unit && (
        <span
          className="text-3 mono"
          style={{ fontSize: 12, flex: '0 0 auto', whiteSpace: 'nowrap' }}
        >
          {param.unit}
        </span>
      )}
    </div>
  );
};

const ParamField = ({ param, value, onChange, error }: ParamFieldProps) => {
  const id = `p-${param.name}`;
  const help = param.help ? <div className="hint">{param.help}</div> : null;
  const errorNode = error ? (
    <div
      role="alert"
      className="hint"
      style={{ marginTop: 4, color: 'var(--fail)' }}
      data-param-error={param.name}
    >
      {error}
    </div>
  ) : null;
  // Field-scoped error visual: red left border on the row container so the
  // field is scannable when validation fails.
  const containerStyle: React.CSSProperties | undefined = error
    ? {
        borderLeft: '2px solid var(--fail)',
        paddingLeft: 8,
        marginLeft: -10,
      }
    : undefined;

  const renderControl = (): React.ReactNode => {
    switch (param.kind) {
      case 'bool':
        return <BoolToggle param={param} value={value} onChange={onChange} />;
      case 'select':
        return (
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
        );
      case 'multiselect': {
        const arr = Array.isArray(value) ? (value as string[]) : [];
        return (
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
        );
      }
      case 'number':
        return <NumberControl id={id} param={param} value={value} onChange={onChange} />;
      case 'long-text':
        return (
          <textarea
            id={id}
            className="textarea"
            rows={4}
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value)}
          />
        );
      case 'path':
        return <PathCombobox id={id} value={(value as string) ?? ''} onChange={onChange} />;
      default:
        return (
          <input
            id={id}
            className="input"
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value)}
          />
        );
    }
  };

  // For bool / multiselect we don't tie the label htmlFor to a single control.
  const labelHtmlFor =
    param.kind === 'bool' || param.kind === 'multiselect' ? undefined : id;

  return (
    <div style={containerStyle} data-param-name={param.name}>
      <Label param={param} htmlFor={labelHtmlFor} />
      {renderControl()}
      {help}
      {errorNode}
    </div>
  );
};

export default ParamField;
