/**
 * ExperimentsList - lineage view of evaluation runs.
 *
 * Replaces the old flat sortable table. The page is organised as a
 * vertical timeline ("lineage stream") with a sticky right rail that
 * holds a compare drawer + a project pulse card. Saved-view chips above
 * the stream let the user pivot between common slices ("All runs",
 * "My week", "Below ship gate", "Cost spike") without re-building
 * filters every visit; the active view persists in localStorage.
 *
 * Selection: clicking a node circle toggles the run as selected. The
 * compare drawer fills as soon as 2 runs are picked; "Open full diff"
 * navigates to RunDetail with `?compare=`.
 *
 * Data source: `v2.experiments()` (the existing list endpoint). The
 * proposal envisages a richer `/experiments/lineage` endpoint that
 * carries config_diff + project pulse_spark; until that lands the page
 * synthesises a minimal lineage from the list rows. config_diff stays
 * empty in this mode (the legacy payload doesn't capture it) and the
 * pulse spark is the recent pass-rate sequence.
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import {
  Card,
  Eyebrow,
  Pill,
  Btn,
  StatusDot,
  Spark,
  StackBar,
  Skeleton,
  UpdatingChip,
} from '../ui';
import { v2 } from '../api/client';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { listCli, type CliSchema } from '../api/cli';
import { openCliRunner } from '../cliRunnerBridge';
import { preloadRunDetail } from '../routePreloads';
import { useRouteTour } from '../tour/useRouteTour';
import { RUN_EVAL_TOUR_ID } from '../tour/scripts/runEval';
import { E } from '../tokens';
import type { ExperimentRow, RunStatus } from '../api/types';

// Deep-link target used as a graceful fallback if the CLI catalog hasn't
// loaded yet (matches the legacy implementation's behaviour).
const NEW_EVAL_TARGET = '/commands?prefill=run-eval';
const SAVED_VIEW_LS_KEY = 'evalyn:experiments:savedView';

type SavedView = 'all' | 'week' | 'gate' | 'cost';

const SAVED_VIEW_VALUES: readonly SavedView[] = ['all', 'week', 'gate', 'cost'];

/** Locally-defined lineage shapes. The richer `/experiments/lineage`
 * endpoint described in the design doc is not wired yet; we synthesise
 * this from `v2.experiments()` until it lands. */
interface FailureMix {
  pass: number;
  warn: number;
  fail: number;
}

interface LineageRun {
  id: string;
  name: string;
  author: string;
  /** Humanised "2h" / "3d" string. */
  when: string;
  status: RunStatus;
  pass: number | null;
  /** Count completed, or "128/400" while running. */
  items: number | string;
  failure_mix: FailureMix | null;
  /** What changed vs the previous run. Empty in fallback mode. */
  config_diff: string;
  tags: string[];
  pinned?: boolean;
  live?: boolean;
  /** ISO timestamp - used for client-side sorting / "My week". */
  when_iso: string;
}

interface Lineage {
  runs: LineageRun[];
  median_pass: number | null;
  pulse_spark: number[];
}

/**
 * Convert an ExperimentRow (legacy /experiments shape) to a LineageRun.
 * The legacy endpoint doesn't carry per-run failure mix or config diff,
 * so those stay null/empty and the UI falls back to a sparser node
 * card.
 */
function rowToLineage(r: ExperimentRow): LineageRun {
  let when = '';
  const t = Date.parse(r.when_iso);
  if (!Number.isNaN(t)) {
    const ms = Date.now() - t;
    const days = ms / 86_400_000;
    if (days < 1 / 24) when = `${Math.max(0, Math.round(ms / 60_000))}m`;
    else if (days < 2) when = `${Math.round(days * 24)}h`;
    else when = `${Math.round(days)}d`;
  }
  return {
    id: r.id,
    name: r.name,
    author: r.author,
    when,
    status: r.status,
    pass: r.pass,
    items: r.items,
    failure_mix: null,
    config_diff: '',
    tags: r.tags ?? [],
    pinned: false,
    live: r.status === 'running',
    when_iso: r.when_iso,
  };
}

/** Build a synthesised Lineage from the legacy /experiments list. Sorts
 * runs newest-first and computes a project median + pulse spark. */
function synthesizeLineage(rows: ExperimentRow[]): Lineage {
  const runs = rows.map(rowToLineage);
  // Sort newest-first; runs with unparseable timestamps drift to the
  // end so the freshest data stays at the top of the timeline.
  runs.sort((a, b) => {
    const at = Date.parse(a.when_iso) || -Infinity;
    const bt = Date.parse(b.when_iso) || -Infinity;
    return bt - at;
  });
  const passes = runs
    .map((r) => r.pass)
    .filter((p): p is number => p != null);
  const sortedP = [...passes].sort((a, b) => a - b);
  const median: number | null =
    sortedP.length === 0
      ? null
      : sortedP.length % 2
        ? sortedP[Math.floor(sortedP.length / 2)]
        : (sortedP[sortedP.length / 2 - 1] + sortedP[sortedP.length / 2]) / 2;
  // Pulse spark: pass rates oldest -> newest, up to 30 points.
  const oldestFirst = [...runs].reverse();
  const spark = oldestFirst
    .map((r) => r.pass)
    .filter((p): p is number => p != null)
    .slice(-30);
  return { runs, median_pass: median, pulse_spark: spark };
}

/** Days-since-now from an ISO timestamp. Infinity when unparseable so
 * "My week" predicates (<= 7) safely exclude unknown dates. */
function daysSince(iso: string): number {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return Infinity;
  return (Date.now() - t) / 86_400_000;
}

/** Compare drawer placeholder lines for the empty state. */
function CompareEmpty() {
  return (
    <div
      style={{
        marginTop: 10,
        padding: 10,
        background: E.panel2,
        borderRadius: 6,
        border: `1px dashed ${E.hair2}`,
        fontFamily: E.fMono,
        fontSize: 11,
        color: E.text3,
        lineHeight: 1.5,
      }}
    >
      <div>Click any 2 nodes to compare.</div>
      <div style={{ marginTop: 6 }}>
        Diff highlights what changed between runs (pass rate, tags, run id).
      </div>
    </div>
  );
}

interface NodeProps {
  run: LineageRun;
  selected: boolean;
  onToggleSelect: () => void;
  onOpen: () => void;
}

function LineageNode({ run, selected, onToggleSelect, onOpen }: NodeProps) {
  const isBaseline = run.tags?.includes('baseline');
  const isRunning = run.status === 'running';
  const isFailed = run.status === 'failed';
  const ringColor = isRunning
    ? E.ember
    : isFailed
      ? E.fail
      : isBaseline
        ? E.steel
        : E.text3;
  const fm = run.failure_mix;
  const segs = fm
    ? [
        { value: fm.pass, color: E.pass, label: `pass ${fm.pass}` },
        { value: fm.warn, color: E.warn, label: `warn ${fm.warn}` },
        { value: fm.fail, color: E.fail, label: `fail ${fm.fail}` },
      ]
    : [];
  const segTotal = segs.reduce((s, x) => s + x.value, 0);

  return (
    <div
      style={{ position: 'relative', marginBottom: 12 }}
      onMouseEnter={() => {
        // Warm both the RunDetail chunk and the per-run payload so click
        // -> paint stays under the perception threshold.
        void preloadRunDetail();
        prefetchV2(`experiment:${run.id}`, () => v2.experiment(run.id));
      }}
    >
      {/* spine node */}
      <button
        type="button"
        onClick={(ev) => {
          ev.stopPropagation();
          onToggleSelect();
        }}
        title={selected ? 'Click to deselect' : 'Click to select for compare'}
        aria-pressed={selected}
        style={{
          position: 'absolute',
          left: -22,
          top: 16,
          width: 16,
          height: 16,
          padding: 0,
          borderRadius: '50%',
          background: selected ? ringColor : E.panel,
          border: `2px solid ${ringColor}`,
          boxShadow: isRunning
            ? `0 0 0 4px ${E.emberDim}`
            : selected
              ? `0 0 0 3px ${E.emberRim}`
              : 'none',
          cursor: 'pointer',
          animation: isRunning ? 'eDotPulse 1.6s ease-in-out infinite' : 'none',
        }}
      />
      <Card
        accent={selected || run.pinned}
        style={{
          padding: '12px 14px',
          // Highlight the card subtly when selected so the spine ring +
          // card surface read as one unit.
          background: selected ? E.emberDim : E.panel,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 6,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
            {run.when ? `${run.when} ago` : '-'}
          </span>
          <span style={{ color: E.text4 }}>·</span>
          <button
            type="button"
            onClick={onOpen}
            style={{
              background: 'transparent',
              border: 'none',
              padding: 0,
              cursor: 'pointer',
              fontSize: 13,
              color: E.text0,
              fontWeight: 500,
              fontFamily: E.fSans,
              textAlign: 'left',
            }}
          >
            {run.name}
          </button>
          {isBaseline && (
            <Pill bg={E.steelDim} color={E.steel} mono>
              baseline
            </Pill>
          )}
          {run.live && (
            <Pill bg={E.emberDim} color={E.ember} mono>
              live
            </Pill>
          )}
          {run.tags
            ?.filter((t: string) => t !== 'baseline')
            .map((t: string) => (
              <Pill key={t} mono style={{ fontSize: 10 }}>
                {t}
              </Pill>
            ))}
          <span style={{ flex: 1 }} />
          <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
            {run.id} · {run.author}
          </span>
        </div>

        {run.config_diff && (
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 10.5,
              color: E.text2,
              padding: '4px 8px',
              background: E.panel2,
              borderRadius: 4,
              marginBottom: 8,
              borderLeft: `2px solid ${E.ember}`,
            }}
          >
            Δ {run.config_diff}
          </div>
        )}

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '90px 1fr 80px',
            alignItems: 'center',
            gap: 12,
          }}
        >
          <div>
            {run.pass != null ? (
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                <span
                  style={{
                    fontFamily: E.fSerif,
                    fontSize: 22,
                    color: E.text0,
                    lineHeight: 1,
                  }}
                >
                  {run.pass.toFixed(1)}
                </span>
                <span style={{ fontSize: 11, color: E.text3 }}>%</span>
              </div>
            ) : (
              <div style={{ fontSize: 11, color: E.ember, fontFamily: E.fMono }}>
                {String(run.items)}
              </div>
            )}
          </div>
          {segTotal > 0 ? (
            <div>
              <StackBar segments={segs} w="100%" h={6} />
              <div
                style={{
                  marginTop: 4,
                  fontFamily: E.fMono,
                  fontSize: 9.5,
                  color: E.text3,
                  display: 'flex',
                  gap: 10,
                  flexWrap: 'wrap',
                }}
              >
                <span style={{ color: E.pass }}>● pass {fm!.pass}</span>
                <span style={{ color: E.warn }}>● warn {fm!.warn}</span>
                <span style={{ color: E.fail }}>● fail {fm!.fail}</span>
              </div>
            </div>
          ) : (
            <div style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
              {isRunning ? 'collecting items...' : 'no item breakdown'}
            </div>
          )}
          <Btn
            kind="ghost"
            size="sm"
            onClick={onOpen}
            style={{ justifyContent: 'flex-end' }}
          >
            Open →
          </Btn>
        </div>
      </Card>
    </div>
  );
}

interface CompareCardProps {
  selected: LineageRun[];
  onClear: () => void;
  onOpenFullDiff: () => void;
}

function CompareCard({ selected, onClear, onOpenFullDiff }: CompareCardProps) {
  return (
    <Card style={{ padding: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Eyebrow>Compare drawer</Eyebrow>
        <span style={{ flex: 1 }} />
        {selected.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            title="Clear selection"
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              fontSize: 11,
              color: E.text3,
              fontFamily: E.fMono,
              padding: 0,
            }}
          >
            clear
          </button>
        )}
      </div>
      {selected.length === 0 && <CompareEmpty />}
      {selected.length === 1 && (
        <>
          <div
            style={{
              marginTop: 8,
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text2,
            }}
          >
            1 of 2 selected. Pick one more to compare.
          </div>
          <div
            style={{
              marginTop: 8,
              padding: 10,
              background: E.panel2,
              borderRadius: 6,
              border: `1px solid ${E.hair2}`,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 11.5,
              color: E.text1,
            }}
          >
            <StatusDot status={selected[0].status} label={`Status: ${selected[0].status}`} />
            <span style={{ fontFamily: E.fMono }}>{selected[0].id}</span>
            <span style={{ flex: 1 }} />
            <span
              style={{
                fontFamily: E.fMono,
                color: selected[0].pass != null ? E.text0 : E.text3,
              }}
            >
              {selected[0].pass != null ? selected[0].pass.toFixed(1) : '-'}
            </span>
          </div>
        </>
      )}
      {selected.length === 2 && (
        <>
          <div
            style={{
              marginTop: 10,
              padding: 10,
              background: E.panel2,
              borderRadius: 6,
              border: `1px dashed ${E.hair2}`,
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 11.5,
                color: E.text1,
                marginBottom: 4,
              }}
            >
              <StatusDot status={selected[0].status} label={`Status: ${selected[0].status}`} />
              <span style={{ fontFamily: E.fMono }}>{selected[0].id}</span>
              <span style={{ flex: 1 }} />
              <span
                style={{
                  fontFamily: E.fMono,
                  color: selected[0].pass != null ? E.pass : E.text3,
                }}
              >
                {selected[0].pass != null ? selected[0].pass.toFixed(1) : '-'}
              </span>
            </div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 11.5,
                color: E.text1,
                marginBottom: 8,
              }}
            >
              <StatusDot status={selected[1].status} label={`Status: ${selected[1].status}`} />
              <span style={{ fontFamily: E.fMono }}>{selected[1].id}</span>
              <span style={{ flex: 1 }} />
              <span
                style={{
                  fontFamily: E.fMono,
                  color: selected[1].pass != null ? E.pass : E.text3,
                }}
              >
                {selected[1].pass != null ? selected[1].pass.toFixed(1) : '-'}
              </span>
            </div>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 10,
                color: E.text2,
                lineHeight: 1.6,
                paddingTop: 8,
                borderTop: `1px solid ${E.hair2}`,
              }}
            >
              <div style={{ color: E.fail }}>- {selected[0].id}</div>
              <div style={{ color: E.pass }}>+ {selected[1].id}</div>
              {selected[0].pass != null && selected[1].pass != null && (
                <div style={{ color: E.text3, marginTop: 4 }}>
                  Δ pass {(selected[1].pass - selected[0].pass).toFixed(1)} pts
                </div>
              )}
            </div>
          </div>
          <Btn
            kind="primary"
            size="sm"
            onClick={onOpenFullDiff}
            style={{ marginTop: 10, width: '100%', justifyContent: 'center' }}
          >
            Open full diff →
          </Btn>
        </>
      )}
    </Card>
  );
}

interface ChipProps {
  label: string;
  count: number | null;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}

function ViewChip({ label, count, active, onClick, disabled, title }: ChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        padding: '5px 10px',
        borderRadius: 999,
        border: `1px solid ${active ? E.ember : E.hair2}`,
        background: active ? E.emberDim : E.panel,
        color: active ? E.ember : E.text2,
        fontSize: 11.5,
        fontWeight: active ? 500 : 400,
        fontFamily: E.fSans,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
      }}
      aria-pressed={active}
    >
      {label}
      {count != null && (
        <span
          style={{
            fontFamily: E.fMono,
            fontSize: 10,
            color: active ? E.ember : E.text3,
          }}
        >
          {count}
        </span>
      )}
    </button>
  );
}

export default function ExperimentsList() {
  const navigate = useNavigate();

  // Data: legacy /experiments list endpoint. Synthesise lineage shape
  // client-side until the richer /experiments/lineage endpoint is wired
  // up across the stack.
  const { data: rows, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'experiments',
    v2.experiments,
  );

  // Fire the route tour once we have any data.
  useRouteTour(RUN_EVAL_TOUR_ID, !!(rows && !err));

  const data: Lineage | null = useMemo(() => {
    if (!rows) return null;
    return synthesizeLineage(rows);
  }, [rows]);

  // Selection: at most 2 ids.
  const [selected, setSelected] = useState<string[]>([]);
  const [hint, setHint] = useState<string | null>(null);

  function toggleSelected(id: string) {
    setHint(null);
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  }

  function clearSelected() {
    setSelected([]);
    setHint(null);
  }

  // Saved view persisted to localStorage. Defensive read in case storage
  // is disabled (private browsing).
  const [savedView, setSavedView] = useState<SavedView>(() => {
    try {
      const raw = localStorage.getItem(SAVED_VIEW_LS_KEY);
      if (raw && SAVED_VIEW_VALUES.includes(raw as SavedView)) {
        return raw as SavedView;
      }
    } catch {
      // ignore
    }
    return 'all';
  });

  useEffect(() => {
    try {
      localStorage.setItem(SAVED_VIEW_LS_KEY, savedView);
    } catch {
      // ignore
    }
  }, [savedView]);

  // Compute counts per saved view from the unfiltered run set.
  const counts = useMemo(() => {
    const runs = data?.runs ?? [];
    const week = runs.filter((r) => daysSince(r.when_iso) <= 7).length;
    const gate = runs.filter((r) => r.pass != null && r.pass < 80).length;
    return { all: runs.length, week, gate };
  }, [data]);

  // Apply the active saved view to the lineage list.
  const visible = useMemo(() => {
    const runs = data?.runs ?? [];
    if (savedView === 'week')
      return runs.filter((r) => daysSince(r.when_iso) <= 7);
    if (savedView === 'gate')
      return runs.filter((r) => r.pass != null && r.pass < 80);
    if (savedView === 'cost') return runs; // legacy payload has no cost
    return runs;
  }, [data, savedView]);

  const selectedRuns: LineageRun[] = useMemo(() => {
    const runs = data?.runs ?? [];
    return selected
      .map((id) => runs.find((r) => r.id === id))
      .filter((r): r is LineageRun => !!r);
  }, [selected, data]);

  function openCompare() {
    if (selected.length !== 2) {
      setHint('Select 2 runs first.');
      return;
    }
    const [a, b] = selected;
    navigate(
      `/experiments/${encodeURIComponent(a)}?compare=${encodeURIComponent(b)}`,
    );
  }

  // "+ New evaluation" - load the CLI catalog lazily and open the runner.
  // Falls back to the /commands deep link if the catalog can't load (so
  // the button is never a dead-end).
  const [newEvalBusy, setNewEvalBusy] = useState(false);
  async function handleNewEvaluation() {
    if (newEvalBusy) return;
    setNewEvalBusy(true);
    try {
      const cmds: CliSchema[] = await listCli();
      const runEval = cmds.find((c) => c.id === 'run-eval');
      if (runEval) {
        openCliRunner(runEval);
      } else {
        navigate(NEW_EVAL_TARGET);
      }
    } catch {
      navigate(NEW_EVAL_TARGET);
    } finally {
      setNewEvalBusy(false);
    }
  }

  const noRuns = data && data.runs.length === 0;

  return (
    <AppShell contextChip={{ name: 'Experiments', version: '' }}>
      <div style={{ padding: 24 }}>
        {/* Header row */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            marginBottom: 4,
          }}
        >
          <Eyebrow>Experiments - lineage view</Eyebrow>
          <UpdatingChip
            visible={reloading && !isInitialLoad}
            error={data ? err : null}
            onRetry={refetch}
          />
          <span style={{ flex: 1 }} />
          <Btn
            kind="ghost"
            size="sm"
            onClick={openCompare}
            disabled={selected.length === 0}
            title={
              selected.length === 2
                ? 'Open both runs side-by-side'
                : 'Click any 2 nodes to enable compare'
            }
          >
            ⇄ Compare 2
          </Btn>
          <Btn
            kind="primary"
            size="sm"
            onClick={handleNewEvaluation}
            disabled={newEvalBusy}
            data-coachmark="experiments-new-button"
          >
            + New evaluation
          </Btn>
        </div>

        {/* Page title */}
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 28,
            fontWeight: 400,
            margin: '4px 0 14px',
            color: E.text0,
            letterSpacing: '-0.015em',
            lineHeight: 1.1,
          }}
        >
          How quality has moved
        </h1>

        {hint && (
          <div
            style={{
              marginBottom: 10,
              padding: '6px 10px',
              background: E.emberDim,
              border: `1px solid ${E.emberRim}`,
              borderRadius: 6,
              fontSize: 11.5,
              color: E.ember,
              fontFamily: E.fMono,
              display: 'inline-block',
            }}
          >
            {hint}
          </div>
        )}

        {/* Saved views chip row */}
        <div
          style={{
            display: 'flex',
            gap: 6,
            marginBottom: 14,
            flexWrap: 'wrap',
          }}
          data-coachmark="experiments-filters"
        >
          <ViewChip
            label="All runs"
            count={counts.all}
            active={savedView === 'all'}
            onClick={() => setSavedView('all')}
          />
          <ViewChip
            label="↑ My week"
            count={counts.week}
            active={savedView === 'week'}
            onClick={() => setSavedView('week')}
          />
          <ViewChip
            label="Below ship gate"
            count={counts.gate}
            active={savedView === 'gate'}
            onClick={() => setSavedView('gate')}
          />
          {/* Cost spike: the legacy payload exposes cost only as a
              formatted string ("$2.41"), which can't be reliably
              compared to a median. Render the chip disabled so the
              slot stays in the layout but the user understands the
              filter is intentionally inactive until the lineage
              endpoint ships numeric cost. */}
          <ViewChip
            label="Cost spike"
            count={null}
            active={false}
            onClick={() => {
              /* no-op */
            }}
            disabled
            title="Per-run numeric cost is not available on this endpoint yet"
          />
          <ViewChip
            label="+ Save current view"
            count={null}
            active={false}
            onClick={() => {
              /* placeholder for the persistent custom views feature */
            }}
            disabled
            title="Custom views are coming soon"
          />
        </div>

        {/* Inline error */}
        {err && (
          <Card
            style={{
              marginBottom: 14,
              padding: 14,
              borderColor: E.fail,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div style={{ flex: 1 }}>
              <Eyebrow style={{ color: E.fail }}>
                Error loading experiments
              </Eyebrow>
              <div
                style={{
                  fontFamily: E.fMono,
                  fontSize: 12,
                  color: E.text2,
                  marginTop: 4,
                }}
              >
                {err}
              </div>
            </div>
            <Btn kind="secondary" size="sm" onClick={refetch}>
              Retry
            </Btn>
          </Card>
        )}

        {/* Loading skeleton */}
        {!data && !err && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 280px',
              gap: 14,
            }}
          >
            <div style={{ paddingLeft: 22 }}>
              {[0, 1, 2].map((i) => (
                <Card key={i} style={{ padding: 14, marginBottom: 10 }}>
                  <Skeleton w="40%" h={12} />
                  <div style={{ marginTop: 8 }}>
                    <Skeleton w="60%" h={10} />
                  </div>
                  <div style={{ marginTop: 14 }}>
                    <Skeleton w="100%" h={6} />
                  </div>
                </Card>
              ))}
            </div>
            <Card style={{ padding: 14 }}>
              <Skeleton w="50%" h={10} />
              <div style={{ marginTop: 10 }}>
                <Skeleton w="100%" h={60} />
              </div>
            </Card>
          </div>
        )}

        {/* Empty state */}
        {noRuns && (
          <Card style={{ padding: 28, textAlign: 'center' }}>
            <Eyebrow>No experiments yet</Eyebrow>
            <div style={{ marginTop: 8, fontSize: 14, color: E.text2 }}>
              Run your first evaluation to see the lineage view.
            </div>
            <div style={{ marginTop: 14 }}>
              <Btn kind="primary" size="md" onClick={handleNewEvaluation}>
                Run your first eval
              </Btn>
            </div>
          </Card>
        )}

        {/* Main grid */}
        {data && data.runs.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 280px',
              gap: 14,
            }}
            data-coachmark="experiments-list"
          >
            {/* Lineage stream */}
            <div style={{ position: 'relative', paddingLeft: 22 }}>
              {/* Spine: dashed via CSS gradient. */}
              <div
                style={{
                  position: 'absolute',
                  left: 7,
                  top: 8,
                  bottom: 0,
                  width: 2,
                  background: `linear-gradient(${E.hair2} 50%, transparent 0) 0/2px 6px`,
                }}
              />
              {visible.length === 0 ? (
                <Card
                  style={{
                    padding: 18,
                    fontSize: 12.5,
                    color: E.text2,
                    fontFamily: E.fSans,
                  }}
                >
                  No runs match the current view.
                  <Btn
                    kind="ghost"
                    size="sm"
                    onClick={() => setSavedView('all')}
                    style={{ marginLeft: 10 }}
                  >
                    Reset to All runs
                  </Btn>
                </Card>
              ) : (
                visible.map((run) => (
                  <LineageNode
                    key={run.id}
                    run={run}
                    selected={selected.includes(run.id)}
                    onToggleSelect={() => toggleSelected(run.id)}
                    onOpen={() =>
                      navigate(`/experiments/${encodeURIComponent(run.id)}`)
                    }
                  />
                ))
              )}
            </div>

            {/* Right rail */}
            <div
              style={{
                position: 'sticky',
                top: 0,
                alignSelf: 'flex-start',
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}
            >
              <CompareCard
                selected={selectedRuns}
                onClear={clearSelected}
                onOpenFullDiff={openCompare}
              />
              <Card style={{ padding: 14 }}>
                <Eyebrow>Pulse</Eyebrow>
                <div
                  style={{
                    fontFamily: E.fSerif,
                    fontSize: 24,
                    color: E.text0,
                    marginTop: 4,
                  }}
                >
                  {data.median_pass != null
                    ? data.median_pass.toFixed(1)
                    : '-'}
                  <span style={{ fontSize: 12, color: E.text3 }}>%</span>
                </div>
                <div style={{ fontSize: 10.5, color: E.text3, marginBottom: 8 }}>
                  median across {counts.all} run{counts.all === 1 ? '' : 's'}
                </div>
                {data.pulse_spark.length > 0 ? (
                  <Spark
                    data={data.pulse_spark}
                    w={250}
                    h={40}
                    color={E.steel}
                  />
                ) : (
                  <div
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 10.5,
                      color: E.text3,
                    }}
                  >
                    not enough runs for a trend yet
                  </div>
                )}
              </Card>
            </div>
          </div>
        )}

        {/* "Why this view" footer - explains the lineage redesign in
            human terms; mirrors the proposal canvas. */}
        {data && data.runs.length > 0 && (
          <div
            style={{
              marginTop: 14,
              padding: '12px 16px',
              background: E.panel,
              border: `1px dashed ${E.hair2}`,
              borderRadius: 8,
              fontSize: 11.5,
              color: E.text2,
              lineHeight: 1.5,
            }}
          >
            <span
              style={{
                color: E.text3,
                fontFamily: E.fMono,
                fontSize: 10.5,
              }}
            >
              why this view →
            </span>{' '}
            Tables sort runs; lineage shows what changed between them. Each
            node carries its run id, pass rate and tags so you can read the
            cause of any pass-rate move at a glance. Pick two nodes to
            compare without losing your place.
          </div>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
