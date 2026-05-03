/**
 * Datasets - list of evaluation input sets, one card per dataset.
 * Wires v2.datasets() into the design from screens-3.jsx.
 *
 * Filter bar (added 2026-05-02): search by name, sort, hide-empty toggle,
 * and tag filter. Pattern mirrors ExperimentsList.tsx.
 */

import { useMemo, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StackBar, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { loadDemo } from '../api/demo';
import { useV2Resource } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { E } from '../tokens';

const COVERAGE_COLORS = [E.ember, E.steel, '#a78bfa', E.warn, E.text3];
const NEW_DATASET_HINT = 'Use `evalyn build-dataset` from the CLI';
const IMPORT_CSV_HINT = 'No CSV importer yet - use `evalyn build-dataset` from the CLI';
const COMING_SOON = 'Coming soon';

type SortOrder = 'name' | 'n-desc' | 'recent';

const SORT_OPTIONS: { value: SortOrder; label: string }[] = [
  { value: 'recent', label: 'Sort: Recent' },
  { value: 'name', label: 'Sort: Name (A-Z)' },
  { value: 'n-desc', label: 'Sort: Items ↓' },
];

const SELECT_STYLE = {
  background: 'transparent',
  color: E.text2,
  border: `1px solid ${E.hair2}`,
  borderRadius: 6,
  padding: '4px 8px',
  fontSize: 11,
  fontFamily: E.fSans,
  cursor: 'pointer',
  outline: 'none',
} as const;

export default function Datasets() {
  const project = useProject();
  const { data, err, reloading, isInitialLoad } = useV2Resource(
    'datasets',
    v2.datasets,
  );
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoErr, setDemoErr] = useState<string | null>(null);

  // Filter state.
  const [query, setQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<SortOrder>('recent');
  const [hideEmpty, setHideEmpty] = useState(true);
  const [tagFilter, setTagFilter] = useState<string>('any');

  const handleLoadDemo = async () => {
    setDemoErr(null);
    setDemoLoading(true);
    try {
      await loadDemo();
      window.location.reload();
    } catch (e) {
      setDemoErr(e instanceof Error ? e.message : String(e));
      setDemoLoading(false);
    }
  };

  // Derive tag option list from the union of all dataset tags.
  const tagOptions = useMemo(() => {
    if (!data) return [];
    const seen = new Set<string>();
    for (const d of data) for (const t of d.tags) seen.add(t);
    return Array.from(seen).sort();
  }, [data]);

  const emptyCount = useMemo(() => {
    if (!data) return 0;
    return data.reduce((s, d) => s + (d.n === 0 ? 1 : 0), 0);
  }, [data]);

  // Filter + sort pipeline. Memoized so 446 cards only re-render when inputs change.
  const filtered = useMemo(() => {
    if (!data) return null;
    const q = query.trim().toLowerCase();
    const matched = data.filter((d) => {
      if (q && !d.name.toLowerCase().includes(q)) return false;
      if (hideEmpty && d.n === 0) return false;
      if (tagFilter !== 'any' && !d.tags.includes(tagFilter)) return false;
      return true;
    });
    const sorted = [...matched];
    if (sortOrder === 'name') {
      sorted.sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortOrder === 'n-desc') {
      sorted.sort((a, b) => b.n - a.n);
    } else {
      // recent: last_used_iso desc, nulls last
      sorted.sort((a, b) => {
        const at = a.last_used_iso ? Date.parse(a.last_used_iso) || 0 : -1;
        const bt = b.last_used_iso ? Date.parse(b.last_used_iso) || 0 : -1;
        if (at === -1 && bt === -1) return 0;
        if (at === -1) return 1;
        if (bt === -1) return -1;
        return bt - at;
      });
    }
    return sorted;
  }, [data, query, sortOrder, hideEmpty, tagFilter]);

  const totalItems = filtered ? filtered.reduce((s, d) => s + d.n, 0) : 0;
  const totalCount = data ? data.length : 0;
  const shownCount = filtered ? filtered.length : 0;

  const clearFilters = () => {
    setQuery('');
    setTagFilter('any');
    setHideEmpty(false);
  };

  return (
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eyebrow>Evaluation inputs</Eyebrow>
              <UpdatingChip visible={reloading && !isInitialLoad} />
            </div>
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 32,
                fontWeight: 400,
                margin: 0,
                color: E.text0,
                letterSpacing: '-0.015em',
              }}
            >
              Datasets
            </h1>
            <p style={{ fontSize: 13, color: E.text2, marginTop: 4 }}>
              The questions you grade your agent against
              {data
                ? ` - showing ${shownCount} of ${totalCount} datasets - ${totalItems} items`
                : ''}
            </p>
          </div>
          <span style={{ flex: 1 }} />
          <Btn kind="secondary" size="md" disabled title={IMPORT_CSV_HINT}>
            Import CSV
          </Btn>
          <Btn kind="primary" size="md" disabled title={NEW_DATASET_HINT}>
            + New dataset
          </Btn>
        </div>

        {err && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading datasets</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {/* FILTER BAR - only show once we have data with at least one card */}
        {data && data.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginTop: 16,
              padding: 8,
              background: E.panel,
              border: `1px solid ${E.hair}`,
              borderRadius: 10,
            }}
          >
            <input
              placeholder="Search datasets by name..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: E.text1,
                fontSize: 13,
                padding: '4px 10px',
              }}
            />
            <button
              type="button"
              onClick={() => setHideEmpty((v) => !v)}
              aria-pressed={hideEmpty}
              title={
                hideEmpty
                  ? `Showing only datasets with items (${emptyCount} hidden)`
                  : 'Showing all datasets including empty ones'
              }
              style={{
                ...SELECT_STYLE,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                color: hideEmpty ? E.ember : E.text2,
                borderColor: hideEmpty ? E.emberRim : E.hair2,
                background: hideEmpty ? E.emberDim : 'transparent',
              }}
            >
              <span
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  background: hideEmpty ? E.ember : E.text3,
                  display: 'inline-block',
                }}
              />
              {hideEmpty ? `Hide ${emptyCount} empty` : `Show empty (${emptyCount})`}
            </button>
            <select
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              style={SELECT_STYLE}
              disabled={tagOptions.length === 0}
              aria-label="Filter by tag"
              title={tagOptions.length === 0 ? 'No tags on any dataset yet' : 'Filter by tag'}
            >
              <option value="any" style={{ background: E.panel }}>
                Tag: Any
              </option>
              {tagOptions.map((t) => (
                <option key={t} value={t} style={{ background: E.panel }}>
                  Tag: {t}
                </option>
              ))}
            </select>
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as SortOrder)}
              style={SELECT_STYLE}
              aria-label="Sort order"
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value} style={{ background: E.panel }}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {!data && !err && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 14,
              marginTop: 22,
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <Card key={i} style={{ padding: 18 }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <Skeleton w={32} h={32} style={{ borderRadius: 7 }} />
                  <div style={{ flex: 1 }}>
                    <Skeleton w="60%" h={13} />
                    <div style={{ marginTop: 6 }}>
                      <Skeleton w="40%" h={10} />
                    </div>
                  </div>
                  <Skeleton w={36} h={22} />
                </div>
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={7} />
                </div>
              </Card>
            ))}
          </div>
        )}

        {data && data.length === 0 && (
          <Card style={{ padding: 32, marginTop: 22, textAlign: 'center' }}>
            <div style={{ fontSize: 14, color: E.text1, marginBottom: 14 }}>
              No datasets yet. Load the demo to explore, or build one from the CLI.
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <Btn kind="primary" size="md" onClick={handleLoadDemo} disabled={demoLoading}>
                {demoLoading ? 'Loading...' : 'Load demo'}
              </Btn>
              <Btn kind="secondary" size="md" disabled title={IMPORT_CSV_HINT}>
                Import CSV
              </Btn>
            </div>
            {demoErr && (
              <div
                style={{
                  marginTop: 14,
                  padding: '8px 12px',
                  background: E.failDim,
                  border: `1px solid ${E.fail}33`,
                  borderRadius: 6,
                  color: E.fail,
                  fontSize: 12,
                  fontFamily: E.fMono,
                  textAlign: 'left',
                  wordBreak: 'break-word',
                }}
              >
                {demoErr}
              </div>
            )}
          </Card>
        )}

        {/* Filters returned 0 hits (but raw data has rows) */}
        {data && data.length > 0 && filtered && filtered.length === 0 && (
          <Card style={{ padding: 28, marginTop: 14, textAlign: 'center' }}>
            <Eyebrow>No datasets match</Eyebrow>
            <div style={{ marginTop: 8, fontSize: 13, color: E.text2 }}>
              Try a different search term, tag, or clear your filters.
            </div>
            <div style={{ marginTop: 14 }}>
              <Btn kind="secondary" size="sm" onClick={clearFilters}>
                Clear filters
              </Btn>
            </div>
          </Card>
        )}

        {filtered && filtered.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 14,
              marginTop: 14,
            }}
          >
            {filtered.map((s) => {
              const segments = s.coverage.map((c, i) => ({
                value: c.value,
                color: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
                label: `${c.label}: ${c.value}`,
              }));
              return (
                <Card key={s.name} hover style={{ padding: 18 }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 7,
                        background: E.panel2,
                        border: `1px solid ${E.hair2}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: E.text2,
                        fontSize: 14,
                      }}
                    >
                      ◫
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontFamily: E.fMono,
                          fontSize: 13,
                          color: E.text0,
                          fontWeight: 500,
                        }}
                      >
                        {s.name}
                      </div>
                      <div style={{ fontSize: 11, color: E.text3, marginTop: 2 }}>
                        {s.source}
                        {s.last_used_iso ? ` - last used ${s.last_used_iso}` : ''}
                      </div>
                    </div>
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>{s.n}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 5, marginTop: 10 }}>
                    {s.tags.map((t) => (
                      <Pill
                        key={t}
                        mono
                        style={{ fontSize: 10, background: E.panel2, color: E.text2 }}
                      >
                        {t}
                      </Pill>
                    ))}
                  </div>
                  {s.coverage.length > 0 && (
                    <>
                      <Eyebrow style={{ marginTop: 14, marginBottom: 8 }}>Coverage</Eyebrow>
                      <StackBar segments={segments} w={'100%'} h={7} />
                      <div
                        style={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: 10,
                          marginTop: 8,
                          fontSize: 10.5,
                          color: E.text2,
                          fontFamily: E.fMono,
                        }}
                      >
                        {s.coverage.map((c, i) => (
                          <span
                            key={c.label}
                            style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                          >
                            <span
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: 1.5,
                                background: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
                              }}
                            />
                            {c.label} {c.value}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                  <div style={{ display: 'flex', gap: 6, marginTop: 14 }}>
                    <Btn kind="secondary" size="sm" disabled title={COMING_SOON}>
                      Open
                    </Btn>
                    <Btn kind="ghost" size="sm" disabled title={COMING_SOON}>
                      Use in eval -&gt;
                    </Btn>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
