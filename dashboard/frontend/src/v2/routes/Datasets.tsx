/**
 * Datasets - list of evaluation input sets, one card per dataset.
 * Wires v2.datasets() into the design from screens-3.jsx.
 *
 * Filter bar (added 2026-05-02): search by name, sort, hide-empty toggle,
 * and tag filter. Pattern mirrors ExperimentsList.tsx.
 */

import { useMemo, useState, type ChangeEvent, type MouseEvent } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StackBar, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { loadDemo } from '../api/demo';
import { runCli } from '../api/cli';
import { upsertJob, type JobHistoryEntry } from '../jobsHistory';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { E } from '../tokens';

const COVERAGE_COLORS = [E.ember, E.steel, '#a78bfa', E.warn, E.text3];
const NEW_DATASET_HINT = 'Use `evalyn build-dataset` from the CLI';
const IMPORT_CSV_HINT = 'No CSV importer yet - use `evalyn build-dataset` from the CLI';
/** Above this many selected datasets, ask the user to confirm before firing. */
const BULK_CONFIRM_THRESHOLD = 50;
/** CLI id for the run-eval command spawned by the bulk toolbar. */
const RUN_EVAL_CLI_ID = 'run-eval';

type SortOrder = 'name' | 'n-desc' | 'recent';
const SORT_VALUES: readonly SortOrder[] = ['name', 'n-desc', 'recent'];

function buildEvalLink(name: string): string {
  return `/commands?prefill=run-eval&dataset=${encodeURIComponent(name)}`;
}

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
  const navigate = useNavigate();
  const { data, err, reloading, isInitialLoad } = useV2Resource(
    'datasets',
    v2.datasets,
  );

  const openDataset = (name: string) => {
    navigate(`/datasets/${encodeURIComponent(name)}`);
  };
  const prefetchDetail = (name: string) => {
    prefetchV2(`dataset:${name}`, () => v2.dataset(name));
  };
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoErr, setDemoErr] = useState<string | null>(null);

  // Bulk-evaluate selection state. We track names (string ids) rather than
  // indices because filter+sort can re-order the visible list at any time.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  const [bulkErr, setBulkErr] = useState<string | null>(null);
  const [bulkInfo, setBulkInfo] = useState<string | null>(null);

  const toggleSelected = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
    // Any user-driven selection change invalidates a previous result chip.
    setBulkErr(null);
    setBulkInfo(null);
  };

  const clearSelection = () => {
    setSelected(new Set());
    setBulkErr(null);
    setBulkInfo(null);
  };

  // Filter state - persisted in the URL so refresh + back-button keep filters.
  // Defaults: query='', sort='recent', hideEmpty=true, tag='any'. Default
  // values are stripped from the URL to keep it clean.
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') ?? '';
  const rawSort = searchParams.get('sort');
  const sortOrder: SortOrder = SORT_VALUES.includes(rawSort as SortOrder)
    ? (rawSort as SortOrder)
    : 'recent';
  // hideEmpty defaults to true; URL stores '0' to mean "show empty".
  const hideEmpty = searchParams.get('hideEmpty') !== '0';
  const tagFilter = searchParams.get('tag') ?? 'any';

  const updateParam = (key: string, value: string, isDefault: boolean) => {
    const sp = new URLSearchParams(searchParams);
    if (isDefault) sp.delete(key);
    else sp.set(key, value);
    setSearchParams(sp, { replace: true });
  };

  const setQuery = (next: string) => updateParam('q', next, next === '');
  const setSortOrder = (next: SortOrder) =>
    updateParam('sort', next, next === 'recent');
  const setHideEmpty = (next: boolean) =>
    updateParam('hideEmpty', next ? '1' : '0', next === true);
  const setTagFilter = (next: string) =>
    updateParam('tag', next, next === 'any');

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
    const sp = new URLSearchParams(searchParams);
    sp.delete('q');
    sp.delete('tag');
    // Setting hideEmpty=false (show all) - that's a non-default state, persist.
    sp.set('hideEmpty', '0');
    setSearchParams(sp, { replace: true });
  };

  // Subset of `filtered` that is "never evaluated AND has items". This is
  // the canonical bulk-eval target set - the user's 417-card backlog.
  const visibleNeverEvaluated = useMemo(() => {
    if (!filtered) return [];
    return filtered.filter((d) => d.n > 0 && d.last_used_iso == null);
  }, [filtered]);

  const selectAllNeverEvaluatedVisible = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const d of visibleNeverEvaluated) next.add(d.name);
      return next;
    });
    setBulkErr(null);
    setBulkInfo(null);
  };

  // Fire one POST /api/cli/run per selected dataset (in parallel). On
  // partial failure we report which datasets failed but never roll back
  // already-spawned jobs - they're real backend work the user wanted.
  const handleBulkEvaluate = async () => {
    if (bulkBusy || selected.size === 0) return;
    const targets = Array.from(selected);
    if (
      targets.length >= BULK_CONFIRM_THRESHOLD &&
      typeof window !== 'undefined' &&
      !window.confirm(
        `This will start ${targets.length} evaluation jobs. LLM-judge runs may incur cost. Continue?`,
      )
    ) {
      return;
    }
    setBulkBusy(true);
    setBulkErr(null);
    setBulkInfo(null);
    const results = await Promise.allSettled(
      targets.map(async (name) => {
        const args: Record<string, unknown> = { dataset: name };
        const { job_id } = await runCli(RUN_EVAL_CLI_ID, args);
        const entry: JobHistoryEntry = {
          job_id,
          cli_id: RUN_EVAL_CLI_ID,
          cli_args: args,
          started_at_iso: new Date().toISOString(),
          status: 'queued',
        };
        upsertJob(entry);
        return name;
      }),
    );
    const failed: string[] = [];
    let started = 0;
    results.forEach((r, i) => {
      if (r.status === 'fulfilled') started += 1;
      else failed.push(targets[i]);
    });
    if (failed.length === 0) {
      setBulkInfo(`Started ${started} job${started === 1 ? '' : 's'}. See Recent Jobs.`);
      // Successful bulk fire clears the selection so the user can keep browsing.
      setSelected(new Set());
    } else {
      const preview = failed.slice(0, 3).join(', ');
      const more = failed.length > 3 ? ` +${failed.length - 3} more` : '';
      setBulkErr(
        `${started} of ${targets.length} spawned. Failed: ${preview}${more} - check Settings or Recent Jobs.`,
      );
      // Drop only the successfully-spawned ones; keep the failures so the
      // user can retry without re-checking each box.
      setSelected(new Set(failed));
    }
    setBulkBusy(false);
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
              onClick={() => setHideEmpty(!hideEmpty)}
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

        {/* SELECT-ALL-VISIBLE row: shows the bulk-eval entry point even when
            nothing is selected yet. Mirrors the filter bar's compact style. */}
        {data && data.length > 0 && visibleNeverEvaluated.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginTop: 8,
              fontSize: 11,
              color: E.text2,
              fontFamily: E.fMono,
            }}
          >
            <button
              type="button"
              onClick={selectAllNeverEvaluatedVisible}
              style={{
                ...SELECT_STYLE,
                color: E.ember,
                borderColor: E.emberRim,
                background: E.emberDim,
              }}
              title="Tick the checkbox on every visible card that has items but has never been evaluated."
            >
              Select all visible ({visibleNeverEvaluated.length} never-evaluated)
            </button>
            {selected.size > 0 && (
              <button type="button" onClick={clearSelection} style={SELECT_STYLE}>
                Clear
              </button>
            )}
          </div>
        )}

        {/* BULK TOOLBAR: appears (slides in) only when selection is non-empty. */}
        <div
          style={{
            overflow: 'hidden',
            maxHeight: selected.size > 0 ? 140 : 0,
            opacity: selected.size > 0 ? 1 : 0,
            transition:
              'max-height 180ms ease-out, opacity 140ms ease-out, margin-top 180ms ease-out',
            marginTop: selected.size > 0 ? 10 : 0,
          }}
          aria-hidden={selected.size === 0}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '10px 12px',
              background: E.emberDim,
              border: `1px solid ${E.emberRim}`,
              borderRadius: 10,
              flexWrap: 'wrap',
            }}
          >
            <span
              style={{
                fontFamily: E.fMono,
                fontSize: 11,
                color: E.ember,
                letterSpacing: '0.04em',
              }}
            >
              {selected.size} selected
            </span>
            <Btn kind="bare" size="sm" onClick={clearSelection} title="Clear selection">
              Clear
            </Btn>
            <span style={{ flex: 1 }} />
            {bulkErr && (
              <span
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: E.fail,
                  background: E.failDim,
                  border: `1px solid ${E.fail}33`,
                  padding: '4px 8px',
                  borderRadius: 6,
                  maxWidth: 420,
                }}
              >
                {bulkErr}
              </span>
            )}
            {bulkInfo && !bulkErr && (
              <span
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: E.pass,
                  background: E.passDim,
                  padding: '4px 8px',
                  borderRadius: 6,
                }}
              >
                {bulkInfo}
              </span>
            )}
            <Btn
              kind="primary"
              size="sm"
              onClick={handleBulkEvaluate}
              disabled={bulkBusy || selected.size === 0}
            >
              {bulkBusy
                ? `Spawning ${selected.size}...`
                : `Evaluate ${selected.size} selected ->`}
            </Btn>
          </div>
        </div>

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
              const hasItems = s.n > 0;
              const neverEvaluated = hasItems && s.last_used_iso == null;
              const isSelected = selected.has(s.name);
              const goEvaluate = (e: MouseEvent<HTMLButtonElement>) => {
                e.stopPropagation();
                navigate(buildEvalLink(s.name));
              };
              const onCheckboxClick = (e: MouseEvent<HTMLInputElement>) => {
                // Card has an onClick that navigates; the checkbox must NOT.
                e.stopPropagation();
              };
              const onCheckboxChange = (_e: ChangeEvent<HTMLInputElement>) => {
                toggleSelected(s.name);
              };
              return (
                <div
                  key={s.name}
                  onMouseEnter={() => prefetchDetail(s.name)}
                  onFocus={() => prefetchDetail(s.name)}
                  style={{ position: 'relative' }}
                >
                <Card
                  hover
                  style={{
                    padding: 18,
                    background: isSelected ? E.emberDim : E.panel,
                    borderColor: isSelected ? E.emberRim : E.hair,
                    transition: 'background 140ms, border-color 140ms',
                  }}
                  onClick={() => openDataset(s.name)}
                >
                  {/* Selection checkbox - top-right, always visible. Uses
                      `position: absolute` so it sits over the card without
                      shifting the existing flex layout below. */}
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={onCheckboxChange}
                    onClick={onCheckboxClick}
                    aria-label={`Select dataset ${s.name} for bulk evaluate`}
                    title={isSelected ? 'Unselect' : 'Select for bulk evaluate'}
                    style={{
                      position: 'absolute',
                      top: 10,
                      right: 10,
                      width: 16,
                      height: 16,
                      cursor: 'pointer',
                      accentColor: E.ember,
                      zIndex: 2,
                    }}
                  />
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
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          flexWrap: 'wrap',
                        }}
                      >
                        <span
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 13,
                            color: E.text0,
                            fontWeight: 500,
                          }}
                        >
                          {s.name}
                        </span>
                        {neverEvaluated && (
                          <Pill
                            mono
                            color={E.ember}
                            bg={E.emberDim}
                            style={{ fontSize: 9.5, padding: '1px 7px' }}
                          >
                            Never evaluated
                          </Pill>
                        )}
                      </div>
                      <div style={{ fontSize: 11, color: E.text3, marginTop: 2 }}>
                        {s.source}
                        {s.last_used_iso ? ` - last used ${s.last_used_iso}` : ''}
                      </div>
                    </div>
                    <div
                      style={{
                        fontFamily: E.fSerif,
                        fontSize: 22,
                        color: E.text0,
                        // Reserve space for the absolute-positioned checkbox so
                        // the count never overlaps it on narrow cards.
                        marginRight: 22,
                      }}
                    >
                      {s.n}
                    </div>
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
                    {neverEvaluated ? (
                      <Btn kind="primary" size="sm" onClick={goEvaluate}>
                        Evaluate this dataset -&gt;
                      </Btn>
                    ) : hasItems ? (
                      <>
                        <Btn
                          kind="secondary"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            openDataset(s.name);
                          }}
                        >
                          Open
                        </Btn>
                        <Btn kind="ghost" size="sm" onClick={goEvaluate}>
                          Evaluate again -&gt;
                        </Btn>
                      </>
                    ) : (
                      <Btn
                        kind="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          openDataset(s.name);
                        }}
                      >
                        Open
                      </Btn>
                    )}
                  </div>
                </Card>
                </div>
              );
            })}
          </div>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
