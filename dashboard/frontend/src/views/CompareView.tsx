/**
 * CompareView - Braintrust-style side-by-side run comparison.
 *
 * Renders the response from `GET /api/compare?run_a=&run_b=` as a header
 * (pass-rate / cost deltas) plus an aligned per-row table. Each row is
 * collapsible: click the row chrome to reveal the full input + both
 * outputs + per-metric deltas inline.
 *
 * Sort modes:
 *  - default: input-hash order returned by the backend
 *  - regression: most negative delta first (worst regressions surface)
 *  - improvement: most positive delta first
 *
 * NaN deltas (missing on one side) sort to the bottom regardless of mode
 * so the comparable rows always lead.
 *
 * Spec source: docs/superpowers/research/2026-05-01-dashboard-ux/00-synthesis.md §8.1
 *              docs/superpowers/research/2026-05-01-dashboard-ux/03-competitive-scan.md (Braintrust diff-mode)
 */

import { Fragment, useEffect, useMemo, useState } from 'react';
import { api, type CompareResponse, type CompareRow, type CompareRowSide } from '../api';

type SortMode = 'default' | 'regression' | 'improvement';

interface CompareViewProps {
  runA: string;
  runB: string;
}

const fmtPct = (v: number | null | undefined): string => {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
};

const fmtSignedPct = (delta: number): string => {
  if (!Number.isFinite(delta)) return '—';
  const pts = (delta * 100).toFixed(1);
  return delta > 0 ? `+${pts} pp` : delta < 0 ? `${pts} pp` : '0 pp';
};

const fmtDelta = (delta: number): string => {
  if (!Number.isFinite(delta)) return '—';
  const sign = delta > 0 ? '+' : '';
  return `${sign}${delta.toFixed(2)}`;
};

const fmtCost = (v: number | null | undefined): string => {
  if (v == null || Number.isNaN(v)) return '—';
  return `$${v.toFixed(4)}`;
};

const deltaColor = (delta: number): string => {
  if (!Number.isFinite(delta) || delta === 0) return 'var(--text-2)';
  return delta > 0 ? 'var(--pass)' : 'var(--fail)';
};

const PASS_GLYPH = { true: '✓', false: '✗', null: '—' } as const;
const PASS_COLOR = {
  true: 'var(--pass)',
  false: 'var(--fail)',
  null: 'var(--text-3)',
} as const;

const passKey = (side: CompareRowSide | null): 'true' | 'false' | 'null' => {
  if (!side || (side.pass !== true && side.pass !== false)) return 'null';
  return side.pass ? 'true' : 'false';
};

const passGlyph = (side: CompareRowSide | null): string => PASS_GLYPH[passKey(side)];
const passColor = (side: CompareRowSide | null): string => PASS_COLOR[passKey(side)];

/**
 * Sort the rows. NaN/undefined deltas (one-sided rows) always trail the
 * comparable rows so the table never opens with a column of dashes.
 */
const sortRows = (rows: CompareRow[], mode: SortMode): CompareRow[] => {
  if (mode === 'default') return rows;
  const finite: CompareRow[] = [];
  const nan: CompareRow[] = [];
  for (const r of rows) {
    if (Number.isFinite(r.delta)) finite.push(r);
    else nan.push(r);
  }
  finite.sort((x, y) =>
    mode === 'regression' ? x.delta - y.delta : y.delta - x.delta,
  );
  return [...finite, ...nan];
};

const SECTION_LABEL_STYLE: React.CSSProperties = {
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
  marginBottom: 4,
};

const PRE_BASE: React.CSSProperties = {
  margin: 0,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  color: 'var(--text-1)',
  fontSize: 12,
  lineHeight: 1.5,
  overflow: 'auto',
};

const PRE_OUTPUT: React.CSSProperties = {
  ...PRE_BASE,
  maxHeight: 240,
  background: 'var(--bg-2)',
  padding: 10,
  borderRadius: 6,
};

/**
 * Render one row's expanded panel: full input, both outputs, per-metric
 * deltas. Pure presentational; lifted out so the row component stays small.
 */
const ExpandedRow = ({ row }: { row: CompareRow }) => {
  const allMetrics = new Set<string>();
  if (row.a) for (const k of Object.keys(row.a.metrics)) allMetrics.add(k);
  if (row.b) for (const k of Object.keys(row.b.metrics)) allMetrics.add(k);
  const metrics = [...allMetrics].sort();

  return (
    <div
      style={{
        background: 'var(--bg-1)',
        padding: '14px 18px',
        borderTop: '1px solid var(--line)',
        fontSize: 12,
      }}
    >
      <div style={{ marginBottom: 12 }}>
        <div className="text-3" style={SECTION_LABEL_STYLE}>Input</div>
        <pre className="mono" style={{ ...PRE_BASE, maxHeight: 200 }}>
          {row.input_preview}
        </pre>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 12 }}>
        <div>
          <div className="text-3" style={SECTION_LABEL_STYLE}>A · output</div>
          <pre className="mono" style={PRE_OUTPUT}>
            {row.a?.output ?? '(no output)'}
          </pre>
        </div>
        <div>
          <div className="text-3" style={SECTION_LABEL_STYLE}>B · output</div>
          <pre className="mono" style={PRE_OUTPUT}>
            {row.b?.output ?? '(no output)'}
          </pre>
        </div>
      </div>

      {metrics.length > 0 && (
        <div>
          <div className="text-3" style={SECTION_LABEL_STYLE}>Per-metric deltas</div>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-2)' }}>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Metric</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>A</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>B</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Δ</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const av = row.a?.metrics[m];
                const bv = row.b?.metrics[m];
                const delta =
                  typeof av === 'number' && typeof bv === 'number' ? bv - av : NaN;
                return (
                  <tr key={m} style={{ borderTop: '1px solid var(--line)' }}>
                    <td className="mono" style={{ padding: '4px 8px', color: 'var(--text-1)' }}>{m}</td>
                    <td className="mono" style={{ padding: '4px 8px', textAlign: 'right' }}>
                      {typeof av === 'number' ? av.toFixed(2) : '—'}
                    </td>
                    <td className="mono" style={{ padding: '4px 8px', textAlign: 'right' }}>
                      {typeof bv === 'number' ? bv.toFixed(2) : '—'}
                    </td>
                    <td
                      className="mono"
                      style={{ padding: '4px 8px', textAlign: 'right', color: deltaColor(delta) }}
                    >
                      {fmtDelta(delta)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const CompareView = ({ runA, runB }: CompareViewProps) => {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortMode, setSortMode] = useState<SortMode>('default');
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  // Guard against the malformed-tab id case (e.g. `compare:abc:`). The store
  // checks at open time but a stale tab in localStorage / a deeplink could
  // still land us here with empty ids.
  const valid = Boolean(runA && runB);

  useEffect(() => {
    if (!valid) {
      setLoading(false);
      setError('invalid run ids');
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .compare(runA, runB)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runA, runB, valid]);

  const sortedRows = useMemo(() => {
    if (!data) return [];
    return sortRows(data.items, sortMode);
  }, [data, sortMode]);

  const safeDelta = (a: number | null | undefined, b: number | null | undefined): number =>
    a == null || b == null ? NaN : b - a;

  const passDelta = data ? safeDelta(data.run_a.pass, data.run_b.pass) : NaN;
  const costDelta = data ? safeDelta(data.run_a.cost, data.run_b.cost) : NaN;

  const toggleRow = (hash: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(hash)) next.delete(hash);
      else next.add(hash);
      return next;
    });
  };

  if (!valid) {
    return (
      <div style={{ padding: 28, color: 'var(--fail)', fontSize: 13 }}>
        Compare tab is missing a run id. Close this tab and re-open it from a runs row.
      </div>
    );
  }

  if (loading) {
    return (
      <div
        data-testid="compare-loading"
        style={{ padding: 28, color: 'var(--text-2)', fontSize: 13 }}
      >
        Loading comparison...
      </div>
    );
  }

  if (error) {
    return (
      <div
        data-testid="compare-error"
        style={{ padding: 28, color: 'var(--fail)', fontSize: 13 }}
      >
        Failed to load comparison: {error}
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const sameRun = data.run_a.id === data.run_b.id;

  return (
    <div data-testid="compare-view" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div
        style={{
          padding: '16px 22px',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-1)',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <div>
            <div className="text-3" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Compare runs
            </div>
            <div className="mono" style={{ fontSize: 14, color: 'var(--text-0)', marginTop: 2 }}>
              {data.run_a.id.slice(0, 8)} <span className="text-3">vs</span> {data.run_b.id.slice(0, 8)}
            </div>
          </div>
          <label style={{ fontSize: 12, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 6 }}>
            Sort by
            <select
              aria-label="Sort rows"
              value={sortMode}
              onChange={(e) => setSortMode(e.target.value as SortMode)}
              style={{
                background: 'var(--bg-2)',
                color: 'var(--text-0)',
                border: '1px solid var(--line)',
                borderRadius: 4,
                padding: '4px 8px',
                fontSize: 12,
              }}
            >
              <option value="default">default</option>
              <option value="regression">regression first</option>
              <option value="improvement">improvement first</option>
            </select>
          </label>
        </div>
        {sameRun && (
          <div className="text-3" style={{ fontSize: 11, marginTop: 6, color: 'var(--accent)' }}>
            Comparing a run against itself; every row should show a 0 delta.
          </div>
        )}
        <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-1)', display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          <span>
            <span className="text-3">Pass rate:</span>{' '}
            <span className="mono">{fmtPct(data.run_a.pass)}</span>{' '}
            <span className="text-3">→</span>{' '}
            <span className="mono">{fmtPct(data.run_b.pass)}</span>{' '}
            <span className="mono" style={{ color: deltaColor(passDelta) }}>
              ({fmtSignedPct(passDelta)})
            </span>
          </span>
          {(data.run_a.cost != null || data.run_b.cost != null) && (
            <span>
              <span className="text-3">Cost:</span>{' '}
              <span className="mono">{fmtCost(data.run_a.cost)}</span>{' '}
              <span className="text-3">→</span>{' '}
              <span className="mono">{fmtCost(data.run_b.cost)}</span>{' '}
              <span className="mono" style={{ color: deltaColor(-costDelta) }}>
                ({Number.isFinite(costDelta) ? (costDelta >= 0 ? '+' : '') + costDelta.toFixed(4) : '—'})
              </span>
            </span>
          )}
          <span className="text-3">{data.items.length} rows</span>
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {data.items.length === 0 ? (
          <div
            data-testid="compare-empty"
            style={{ padding: 28, color: 'var(--text-2)', fontSize: 13 }}
          >
            No comparable rows. Either run is empty.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr
                style={{
                  background: 'var(--bg-1)',
                  color: 'var(--text-2)',
                  position: 'sticky',
                  top: 0,
                  zIndex: 1,
                }}
              >
                <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 500, width: 50 }}>#</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', fontWeight: 500 }}>Input</th>
                <th style={{ textAlign: 'center', padding: '8px 12px', fontWeight: 500, width: 70 }}>A</th>
                <th style={{ textAlign: 'center', padding: '8px 12px', fontWeight: 500, width: 70 }}>B</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', fontWeight: 500, width: 80 }}>Δ</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row, idx) => {
                const isOpen = expanded.has(row.input_hash);
                return (
                  <Fragment key={row.input_hash}>
                    <tr
                      data-testid="compare-row"
                      onClick={() => toggleRow(row.input_hash)}
                      style={{
                        cursor: 'pointer',
                        borderTop: '1px solid var(--line)',
                        background: isOpen ? 'var(--bg-2)' : 'transparent',
                      }}
                    >
                      <td className="mono text-3" style={{ padding: '8px 12px' }}>#{idx + 1}</td>
                      <td
                        className="mono"
                        style={{
                          padding: '8px 12px',
                          color: 'var(--text-1)',
                          maxWidth: 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {row.input_preview}
                      </td>
                      <td
                        className="mono"
                        style={{ padding: '8px 12px', textAlign: 'center', color: passColor(row.a) }}
                      >
                        {passGlyph(row.a)}
                      </td>
                      <td
                        className="mono"
                        style={{ padding: '8px 12px', textAlign: 'center', color: passColor(row.b) }}
                      >
                        {passGlyph(row.b)}
                      </td>
                      <td
                        className="mono"
                        style={{ padding: '8px 12px', textAlign: 'right', color: deltaColor(row.delta) }}
                      >
                        {fmtDelta(row.delta)}
                      </td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={5} style={{ padding: 0 }}>
                          <ExpandedRow row={row} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default CompareView;
