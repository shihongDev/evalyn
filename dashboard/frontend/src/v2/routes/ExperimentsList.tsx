/**
 * ExperimentsList - browseable table of all evaluation runs.
 * Selection state is client-side; "Compare 2 selected" routes to RunDetail
 * with ?compare=<otherId>.
 *
 * Two views: Flat (the original 7-column table) and Grouped (one
 * collapsible section per dataset). Grouped is the default once the
 * project has any meaningful repetition, since N runs on the same
 * agent otherwise read as a wall of identical labels.
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Card, Eyebrow, Pill, Btn, StatusDot, Spark, Skeleton, UpdatingChip } from '../ui';
// Deep-link target for "+ New evaluation". Lane I of iter 4 ships a CLI
// runner consuming ``?prefill=run-eval``; until then the user lands on
// the Commands page (graceful fallback over a dead-end "Wizard" card).
const NEW_EVAL_TARGET = '/commands?prefill=run-eval';
import { v2 } from '../api/client';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { useRouteTour } from '../tour/useRouteTour';
import { RUN_EVAL_TOUR_ID } from '../tour/scripts/runEval';
import { preloadRunDetail } from '../routePreloads';
import { E } from '../tokens';
import type { ExperimentRow } from '../api/types';

// Columns: select, name, pass, delta, items, cost, spark.
// Duration column dropped - prod runs don't carry duration_s, every cell
// rendered as "-" (UX walkthrough finding).
const COLS = '24px 2.4fr 90px 90px 80px 90px 110px';

/** Escape one cell for CSV. Wraps in quotes if it contains comma,
 * double-quote, newline, or carriage return; doubles internal quotes
 * per RFC 4180. */
function csvCell(v: string | number | null | undefined): string {
  if (v == null) return '';
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

/** Serialize the rows the user is currently looking at (post-filter,
 * post-sort) as CSV bytes. The columns match the on-screen table plus
 * a few useful extras (id, status, when). */
function rowsToCsv(rows: ExperimentRow[]): string {
  const header = ['id', 'name', 'author', 'status', 'when', 'pass_pct', 'delta', 'items', 'cost', 'tags'];
  const lines = [header.join(',')];
  for (const r of rows) {
    lines.push(
      [
        csvCell(r.id),
        csvCell(r.name),
        csvCell(r.author),
        csvCell(r.status),
        csvCell(r.when_iso),
        csvCell(r.pass),
        csvCell(r.delta),
        csvCell(r.items),
        csvCell(r.cost),
        csvCell(r.tags.join('; ')),
      ].join(','),
    );
  }
  // Trailing newline so the final row ends cleanly when opened in Excel
  // / Numbers / Sheets.
  return lines.join('\n') + '\n';
}

function downloadFile(name: string, mime: string, contents: string): void {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Defer revoke so Safari has time to start the download before the
  // blob URL is invalidated.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

type StatusFilter = 'any' | 'completed' | 'running' | 'failed' | 'warn';
type SortOrder = 'recent' | 'oldest' | 'pass-desc' | 'pass-asc';
type ViewMode = 'flat' | 'grouped';

const STATUS_VALUES: readonly StatusFilter[] = ['any', 'completed', 'running', 'failed', 'warn'];
const SORT_VALUES: readonly SortOrder[] = ['recent', 'oldest', 'pass-desc', 'pass-asc'];
const VIEW_VALUES: readonly ViewMode[] = ['flat', 'grouped'];

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: 'any', label: 'Status: Any' },
  { value: 'completed', label: 'Status: Completed' },
  { value: 'running', label: 'Status: Running' },
  { value: 'failed', label: 'Status: Failed' },
  { value: 'warn', label: 'Status: Warn' },
];

const SORT_OPTIONS: { value: SortOrder; label: string }[] = [
  { value: 'recent', label: 'Sort: Recent' },
  { value: 'oldest', label: 'Sort: Oldest' },
  { value: 'pass-desc', label: 'Sort: Pass ↓' },
  { value: 'pass-asc', label: 'Sort: Pass ↑' },
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

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function deltaColor(d: string): string {
  if (d === 'baseline') return E.steel;
  if (d === '-' || d === '—' || d === '') return E.text3;
  if (d.startsWith('-') || d.startsWith('−')) return E.fail;
  return E.pass;
}

function sparkColor(status: string): string {
  if (status === 'warn') return E.warn;
  if (status === 'running') return E.ember;
  if (status === 'failed') return E.fail;
  return E.pass;
}

/**
 * Run ids look like '20260330-012500_cd347c59'. Inside a dataset group
 * the dataset name is redundant, so each row gets the parsed timestamp
 * plus the short hash - "Mar 30, 01:25  cd347c59". Falls back to the
 * raw id if the prefix doesn't parse.
 */
function friendlyRunLabel(id: string): string {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:_(.+))?$/);
  if (!m) return id;
  const [, , mo, d, h, mi, , tail] = m;
  const month = MONTHS[parseInt(mo, 10) - 1] ?? mo;
  const day = parseInt(d, 10);
  const hh = h;
  const mm = mi;
  const hash = tail ? ` ${tail}` : '';
  return `${month} ${day}, ${hh}:${mm}${hash}`;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

interface RowProps {
  r: ExperimentRow;
  i: number;
  isSelected: boolean;
  onToggle: () => void;
  onNavigate: () => void;
  /** When true, hide the dataset name from the row label (it's in the group header). */
  hideName?: boolean;
}

function Row({ r, i, isSelected, onToggle, onNavigate, hideName }: RowProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: COLS,
        padding: '14px 18px',
        borderTop: i ? `1px solid ${E.hair}` : 'none',
        alignItems: 'center',
        gap: 8,
        background: isSelected ? E.emberDim : 'transparent',
        cursor: 'pointer',
        transition: 'background 140ms',
      }}
      onMouseEnter={(e) => {
        // Subtle hover bg to signal interactivity. Skip when selected
        // (it already has its own emberDim bg) to avoid a visual blip.
        if (!isSelected) {
          e.currentTarget.style.background = E.panel2;
        }
        // Warm both the RunDetail chunk and this run's payload so the
        // click->paint path is in the cache by the time the user
        // actually navigates. Browser dedupes concurrent imports/fetches.
        void preloadRunDetail();
        prefetchV2(`experiment:${r.id}`, () => v2.experiment(r.id));
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.background = 'transparent';
        }
      }}
      onClick={(ev) => {
        if ((ev.target as HTMLElement).tagName === 'INPUT') return;
        onNavigate();
      }}
    >
      <input
        type="checkbox"
        checked={isSelected}
        onChange={onToggle}
        onClick={(ev) => ev.stopPropagation()}
        style={{ accentColor: E.ember }}
      />
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <StatusDot status={r.status} animated={r.status === 'running'} />
          <span style={{ fontSize: 13.5, color: E.text0, fontWeight: 500 }}>
            {hideName ? friendlyRunLabel(r.id) : r.name}
          </span>
          {r.tags.map((t) => (
            <Pill
              key={t}
              mono
              style={{ fontSize: 9.5, padding: '1px 7px', background: E.panel3, color: E.text2 }}
            >
              {t}
            </Pill>
          ))}
          {r.err && (
            <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 9.5 }}>
              {r.err}
            </Pill>
          )}
        </div>
        <div style={{ fontSize: 11, color: E.text3, marginTop: 3, fontFamily: E.fMono }}>
          {hideName ? `${r.author} - ${r.when_iso}` : `${r.id} - ${r.author} - ${r.when_iso}`}
        </div>
      </div>
      <div style={{ fontFamily: E.fSerif, fontSize: 17, color: r.pass != null ? E.text0 : E.text3 }}>
        {r.pass != null ? `${r.pass}%` : '-'}
      </div>
      <div style={{ fontFamily: E.fMono, fontSize: 12, color: deltaColor(r.delta) }}>{r.delta}</div>
      <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>{r.items}</div>
      <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>{r.cost}</div>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        {r.spark && r.spark.length > 0 ? (
          <Spark data={r.spark} color={sparkColor(r.status)} dot w={90} h={24} />
        ) : (
          <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>n/a</span>
        )}
      </div>
    </div>
  );
}

function TableHeader() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: COLS,
        padding: '11px 18px',
        borderBottom: `1px solid ${E.hair}`,
        fontFamily: E.fMono,
        fontSize: 10,
        color: E.text3,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
      }}
    >
      <span></span>
      <span>Experiment</span>
      <span>Pass</span>
      <span>Δ</span>
      <span>Items</span>
      <span>Cost</span>
      <span style={{ textAlign: 'right' }}>Trend</span>
    </div>
  );
}

interface DatasetGroup {
  name: string;
  runs: ExperimentRow[]; // sorted newest-first (matches default sort)
  medianPass: number | null;
  spark: number[]; // pass rates oldest -> newest (for trend)
  latestStatus: string;
}

/**
 * Build per-dataset groups from already-filtered+sorted rows. The input
 * order is preserved within each group (so newest-first stays
 * newest-first), but for the spark we re-derive an oldest-first
 * sequence of pass rates so the trend reads left-to-right in time.
 */
function buildGroups(rows: ExperimentRow[]): DatasetGroup[] {
  const map = new Map<string, ExperimentRow[]>();
  for (const r of rows) {
    const arr = map.get(r.name);
    if (arr) arr.push(r);
    else map.set(r.name, [r]);
  }
  const groups: DatasetGroup[] = [];
  for (const [name, runs] of map) {
    const passes = runs.map((r) => r.pass).filter((p): p is number => p != null);
    const oldestFirst = [...runs].sort(
      (a, b) => (Date.parse(a.when_iso) || 0) - (Date.parse(b.when_iso) || 0),
    );
    const spark = oldestFirst
      .map((r) => r.pass)
      .filter((p): p is number => p != null);
    const latest = oldestFirst[oldestFirst.length - 1];
    groups.push({
      name,
      runs,
      medianPass: median(passes),
      spark,
      latestStatus: latest?.status ?? 'completed',
    });
  }
  // Sort groups by run count desc - the noisy ones go first.
  groups.sort((a, b) => b.runs.length - a.runs.length);
  return groups;
}

const COLLAPSE_THRESHOLD = 20;

export default function ExperimentsList() {
  const { data: rows, err, reloading, isInitialLoad } = useV2Resource(
    'experiments',
    v2.experiments,
  );
  useRouteTour(RUN_EVAL_TOUR_ID, !!(rows && !err));
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Filter / view state lives in the URL so refresh + back-button preserve it.
  // Default values are stripped from the URL to keep it tidy. Unknown values
  // fall back to defaults silently. View mode in URL = explicit override; if
  // absent we fall back to the data-driven `suggestedMode` below.
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') ?? '';
  const statusRaw = searchParams.get('status');
  const statusFilter: StatusFilter = STATUS_VALUES.includes(statusRaw as StatusFilter)
    ? (statusRaw as StatusFilter)
    : 'any';
  const tagFilter = searchParams.get('tag') ?? 'any';
  const authorFilter = searchParams.get('author') ?? 'any';
  const sortRaw = searchParams.get('sort');
  const sortOrder: SortOrder = SORT_VALUES.includes(sortRaw as SortOrder)
    ? (sortRaw as SortOrder)
    : 'recent';
  const viewRaw = searchParams.get('view');
  const viewOverride: ViewMode | null = VIEW_VALUES.includes(viewRaw as ViewMode)
    ? (viewRaw as ViewMode)
    : null;

  const updateParam = (key: string, value: string, isDefault: boolean) => {
    const sp = new URLSearchParams(searchParams);
    if (isDefault) sp.delete(key);
    else sp.set(key, value);
    setSearchParams(sp, { replace: true });
  };
  const setQuery = (next: string) => updateParam('q', next, next === '');
  const setStatusFilter = (next: StatusFilter) =>
    updateParam('status', next, next === 'any');
  const setTagFilter = (next: string) => updateParam('tag', next, next === 'any');
  const setAuthorFilter = (next: string) =>
    updateParam('author', next, next === 'any');
  const setSortOrder = (next: SortOrder) =>
    updateParam('sort', next, next === 'recent');
  const setViewMode = (next: ViewMode) => updateParam('view', next, false);

  // Per-group collapsed state. A group id present in the set is collapsed.
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  // Derive tag/author option lists from the row set.
  const tagOptions = useMemo(() => {
    if (!rows) return [];
    const seen = new Set<string>();
    for (const r of rows) for (const t of r.tags) seen.add(t);
    return Array.from(seen).sort();
  }, [rows]);

  const authorOptions = useMemo(() => {
    if (!rows) return [];
    return Array.from(new Set(rows.map((r) => r.author).filter(Boolean))).sort();
  }, [rows]);

  // Default view mode: grouped if there are >= 2 distinct datasets AND
  // at least one dataset has >= 3 runs. Otherwise flat.
  const suggestedMode: ViewMode = useMemo(() => {
    if (!rows || rows.length === 0) return 'flat';
    const counts = new Map<string, number>();
    for (const r of rows) counts.set(r.name, (counts.get(r.name) ?? 0) + 1);
    const distinct = counts.size;
    let maxCount = 0;
    for (const c of counts.values()) if (c > maxCount) maxCount = c;
    return distinct >= 2 && maxCount >= 3 ? 'grouped' : 'flat';
  }, [rows]);

  const view: ViewMode = viewOverride ?? suggestedMode;

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    setSelected(new Set());
  }

  function compareSelected() {
    if (selected.size !== 2) return;
    const [a, b] = Array.from(selected);
    navigate(`/experiments/${encodeURIComponent(a)}?compare=${encodeURIComponent(b)}`);
  }

  // State-dependent label for the Compare button. Telling the user
  // exactly how many more they need to pick (or which is the wrong count)
  // is more helpful than "Compare 0 selected" which reads as broken.
  const compareLabel: string =
    selected.size === 0
      ? 'Compare runs...'
      : selected.size === 1
        ? 'Pick 1 more to compare'
        : selected.size === 2
          ? 'Compare 2 →'
          : `Pick exactly 2 (${selected.size} selected)`;

  const filtered = useMemo(() => {
    if (!rows) return null;
    const q = query.trim().toLowerCase();
    const matched = rows.filter((r) => {
      if (q) {
        const hit =
          r.name.toLowerCase().includes(q) ||
          r.author.toLowerCase().includes(q) ||
          r.tags.some((t) => t.toLowerCase().includes(q));
        if (!hit) return false;
      }
      if (statusFilter !== 'any' && r.status !== statusFilter) return false;
      if (tagFilter !== 'any' && !r.tags.includes(tagFilter)) return false;
      if (authorFilter !== 'any' && r.author !== authorFilter) return false;
      return true;
    });
    // Sort in-place on a fresh copy.
    const sorted = [...matched];
    if (sortOrder === 'recent' || sortOrder === 'oldest') {
      sorted.sort((a, b) => {
        const at = Date.parse(a.when_iso) || 0;
        const bt = Date.parse(b.when_iso) || 0;
        return sortOrder === 'recent' ? bt - at : at - bt;
      });
    } else {
      sorted.sort((a, b) => {
        const ap = a.pass ?? -1;
        const bp = b.pass ?? -1;
        return sortOrder === 'pass-desc' ? bp - ap : ap - bp;
      });
    }
    return sorted;
  }, [rows, query, statusFilter, tagFilter, authorFilter, sortOrder]);

  // Compute grouped view from filtered rows. Each group preserves the
  // current sort order within it.
  const groups = useMemo(() => {
    if (!filtered) return null;
    return buildGroups(filtered);
  }, [filtered]);

  function isGroupCollapsed(g: DatasetGroup): boolean {
    // After seeding (see effect below), large groups are pre-added to
    // `collapsed`. From then on the set is the single source of truth.
    return collapsed.has(g.name);
  }

  function toggleGroup(name: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function expandAll() {
    setCollapsed(new Set());
  }

  function collapseAll() {
    if (!groups) return;
    setCollapsed(new Set(groups.map((g) => g.name)));
  }

  // Per-group default-collapsed state for big groups. Seed once on
  // first render that has groups, so the page doesn't render hundreds
  // of rows on initial load. After that the user owns the set.
  const [seededDefaults, setSeededDefaults] = useState(false);
  useEffect(() => {
    if (seededDefaults) return;
    if (!groups || groups.length === 0) return;
    const big = groups.filter((g) => g.runs.length > COLLAPSE_THRESHOLD).map((g) => g.name);
    if (big.length > 0) {
      setCollapsed((prev) => {
        const next = new Set(prev);
        for (const n of big) next.add(n);
        return next;
      });
    }
    setSeededDefaults(true);
  }, [groups, seededDefaults]);

  return (
    <AppShell contextChip={{ name: 'Experiments', version: '' }}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, marginBottom: 6 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eyebrow>All evaluations</Eyebrow>
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
                lineHeight: 1.1,
              }}
            >
              Experiments
            </h1>
          </div>
          <span style={{ flex: 1 }} />
          <Btn
            kind="secondary"
            size="md"
            onClick={() => {
              if (!filtered || filtered.length === 0) return;
              // Filename matches what the user is looking at: filtered count,
              // current sort, and a date stamp so repeated exports don't clobber.
              const stamp = new Date().toISOString().slice(0, 10);
              downloadFile(
                `evalyn-experiments-${stamp}.csv`,
                'text/csv;charset=utf-8',
                rowsToCsv(filtered),
              );
            }}
            disabled={!filtered || filtered.length === 0}
            title={
              !filtered || filtered.length === 0
                ? 'Nothing to export with the current filters'
                : `Download the ${filtered.length} currently visible row${filtered.length === 1 ? '' : 's'} as CSV (opens in Excel, Numbers, Sheets)`
            }
          >
            ↗ Export CSV
          </Btn>
          <Btn kind="primary" size="md" onClick={() => navigate(NEW_EVAL_TARGET)} data-coachmark="experiments-new-button">
            ＋ New evaluation
          </Btn>
        </div>

        {err && (
          <Card style={{ marginTop: 14, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading experiments</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
          </Card>
        )}

        {/* FILTER + COMPARE BAR */}
        <div
          data-coachmark="experiments-filters"
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
          {/* View toggle - segmented control */}
          <div
            style={{
              display: 'inline-flex',
              border: `1px solid ${E.hair2}`,
              borderRadius: 6,
              overflow: 'hidden',
            }}
            role="group"
            aria-label="View mode"
          >
            {(['flat', 'grouped'] as const).map((m) => {
              const active = view === m;
              return (
                <button
                  key={m}
                  onClick={() => setViewMode(m)}
                  style={{
                    background: active ? E.panel3 : 'transparent',
                    color: active ? E.text0 : E.text2,
                    border: 'none',
                    padding: '4px 10px',
                    fontSize: 11,
                    fontFamily: E.fSans,
                    cursor: 'pointer',
                    fontWeight: active ? 500 : 400,
                  }}
                  aria-pressed={active}
                >
                  {m === 'flat' ? 'Flat' : 'Grouped'}
                </button>
              );
            })}
          </div>
          {view === 'grouped' && groups && groups.length > 0 && (
            <>
              <Btn kind="secondary" size="sm" onClick={expandAll}>
                Expand all
              </Btn>
              <Btn kind="secondary" size="sm" onClick={collapseAll}>
                Collapse all
              </Btn>
            </>
          )}
          <input
            placeholder="Search by name, author, tag..."
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
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
            style={SELECT_STYLE}
            aria-label="Filter by status"
          >
            {STATUS_OPTIONS.map((o) => (
              <option key={o.value} value={o.value} style={{ background: E.panel }}>
                {o.label}
              </option>
            ))}
          </select>
          <select
            value={tagFilter}
            onChange={(e) => setTagFilter(e.target.value)}
            style={SELECT_STYLE}
            disabled={tagOptions.length === 0}
            aria-label="Filter by tag"
            title={tagOptions.length === 0 ? 'No tags on any run yet' : 'Filter by tag'}
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
            value={authorFilter}
            onChange={(e) => setAuthorFilter(e.target.value)}
            style={SELECT_STYLE}
            disabled={authorOptions.length === 0}
            aria-label="Filter by author"
            title={authorOptions.length === 0 ? 'No authors recorded yet' : 'Filter by author'}
          >
            <option value="any" style={{ background: E.panel }}>
              Author: Any
            </option>
            {authorOptions.map((a) => (
              <option key={a} value={a} style={{ background: E.panel }}>
                Author: {a}
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
          <span style={{ width: 1, height: 20, background: E.hair2 }} />
          {selected.size > 0 && (
            <Btn
              kind="ghost"
              size="sm"
              onClick={clearSelection}
              title="Clear all selected runs"
            >
              ✕ Clear ({selected.size})
            </Btn>
          )}
          <Btn
            kind="secondary"
            size="sm"
            disabled={selected.size !== 2}
            onClick={compareSelected}
            title={
              selected.size === 2
                ? 'Open both runs side-by-side'
                : selected.size === 0
                  ? 'Tick the checkbox on two runs to enable compare'
                  : selected.size === 1
                    ? 'Tick one more run to enable compare'
                    : `Compare needs exactly 2 runs (${selected.size} ticked)`
            }
          >
            {compareLabel}
          </Btn>
        </div>

        {/* LOADING SKELETON */}
        {!rows && !err && (
          <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }}>
            {[0, 1, 2, 3, 4].map((i) => (
              <div
                key={i}
                style={{
                  display: 'grid',
                  gridTemplateColumns: COLS,
                  padding: '14px 18px',
                  borderTop: i ? `1px solid ${E.hair}` : 'none',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Skeleton w={14} h={14} />
                <div>
                  <Skeleton w={`${60 + ((i * 7) % 30)}%`} h={13} />
                  <div style={{ marginTop: 6 }}>
                    <Skeleton w={180} h={10} />
                  </div>
                </div>
                <Skeleton w={40} h={14} />
                <Skeleton w={30} h={11} />
                <Skeleton w={36} h={11} />
                <Skeleton w={28} h={11} />
                <Skeleton w={40} h={11} />
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Skeleton w={70} h={20} />
                </div>
              </div>
            ))}
          </Card>
        )}

        {rows && rows.length === 0 && (
          <Card style={{ marginTop: 14, padding: 28, textAlign: 'center' }}>
            <Eyebrow>No experiments yet</Eyebrow>
            <div style={{ marginTop: 8, fontSize: 14, color: E.text2 }}>
              Run your first evaluation to see results here.
            </div>
            <div style={{ marginTop: 14 }}>
              <Btn kind="primary" size="md" onClick={() => navigate(NEW_EVAL_TARGET)}>
                Run your first eval
              </Btn>
            </div>
          </Card>
        )}

        {/* FLAT VIEW */}
        {view === 'flat' && filtered && filtered.length > 0 && (
          <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }} data-coachmark="experiments-list">
            <TableHeader />
            {filtered.map((r, i) => (
              <Row
                key={r.id}
                r={r}
                i={i}
                isSelected={selected.has(r.id)}
                onToggle={() => toggle(r.id)}
                onNavigate={() => navigate(`/experiments/${encodeURIComponent(r.id)}`)}
              />
            ))}
          </Card>
        )}

        {/* GROUPED VIEW */}
        {view === 'grouped' && groups && groups.length > 0 && (
          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {groups.map((g) => {
              const isCollapsed = isGroupCollapsed(g);
              const medianText =
                g.medianPass != null ? `median ${g.medianPass.toFixed(1)}%` : 'median -';
              return (
                <Card key={g.name} style={{ padding: 0, overflow: 'hidden' }}>
                  {/* Group header bar */}
                  <button
                    type="button"
                    onClick={() => toggleGroup(g.name)}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.background = E.panel2;
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                    }}
                    aria-expanded={!isCollapsed}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '14px 18px',
                      minHeight: 52,
                      background: 'transparent',
                      border: 'none',
                      borderBottom: !isCollapsed ? `1px solid ${E.hair}` : 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'background 120ms',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 12,
                        color: E.text3,
                        width: 14,
                        display: 'inline-block',
                      }}
                      aria-hidden
                    >
                      {isCollapsed ? '▸' : '▾'}
                    </span>
                    <span
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 14,
                        color: E.text0,
                        fontWeight: 500,
                      }}
                    >
                      {g.name}
                    </span>
                    <Pill
                      mono
                      style={{ fontSize: 10, padding: '1px 7px', background: E.panel3, color: E.text2 }}
                    >
                      {g.runs.length} {g.runs.length === 1 ? 'run' : 'runs'}
                    </Pill>
                    <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
                      {medianText}
                    </span>
                    <span style={{ flex: 1 }} />
                    {g.spark.length > 0 ? (
                      <Spark data={g.spark} color={sparkColor(g.latestStatus)} dot w={120} h={26} />
                    ) : (
                      <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>n/a</span>
                    )}
                  </button>

                  {/* Group body */}
                  {!isCollapsed && (
                    <div>
                      <TableHeader />
                      {g.runs.map((r, i) => (
                        <Row
                          key={r.id}
                          r={r}
                          i={i}
                          isSelected={selected.has(r.id)}
                          onToggle={() => toggle(r.id)}
                          onNavigate={() => navigate(`/experiments/${encodeURIComponent(r.id)}`)}
                          hideName
                        />
                      ))}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        )}

        {/* EMPTY FILTER RESULT */}
        {filtered && filtered.length === 0 && rows && rows.length > 0 && (
          <Card
            style={{
              marginTop: 14,
              padding: 18,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div style={{ flex: 1, fontSize: 13, color: E.text2 }}>
              No experiments match
              {query && (
                <>
                  {' '}<b style={{ color: E.ember }}>"{query}"</b>
                </>
              )}
              {' '}with the current filters.
              <div style={{ fontSize: 11, color: E.text3, marginTop: 2, fontFamily: E.fMono }}>
                {rows.length} experiment{rows.length === 1 ? '' : 's'} hidden
              </div>
            </div>
            {query && (
              <Btn kind="secondary" size="sm" onClick={() => setQuery('')}>
                Clear search
              </Btn>
            )}
            <Btn kind="ghost" size="sm" onClick={() => setStatusFilter('any')}>
              Reset filter
            </Btn>
          </Card>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
