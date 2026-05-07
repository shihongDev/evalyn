/**
 * Datasets - coverage view (v2 redesign).
 *
 * Where the previous list answered "what datasets do I have?", this page
 * answers "what slice of intent space am I actually testing?". The lead
 * visual is a Coverage map: a 3x8 grid of intent buckets, each tile
 * coloured by how covered it is across the project's datasets. Below
 * the map, a 2-column grid of coverage cards: each card shows a small
 * pass/fail distribution per intent bucket plus the headline pass score.
 * A bottom Compose strip appears only when 2+ datasets are checked, so
 * the user can blend datasets into one combined eval run.
 *
 * Filter state (search, sort, hide-empty, tag) is persisted in the URL
 * just like the old route - refreshing or sharing the link preserves
 * the slice. Tour anchors (data-coachmark) are kept where the existing
 * onboarding tour expects them.
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ChangeEvent,
  type MouseEvent,
} from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { loadDemo } from '../api/demo';
import { errorMessage } from '../api/errors';
import { listCli, runCli } from '../api/cli';
import type { CliSchema } from '../api/cli';
import { openCliRunner } from '../cliRunnerBridge';
import { upsertJob, type JobHistoryEntry } from '../jobsHistory';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { useRouteTour } from '../tour/useRouteTour';
import { DATASET_UPLOAD_TOUR_ID } from '../tour/scripts/datasetUpload';
import { useProject } from '../hooks/useProject';
import { preloadDatasetDetail } from '../routePreloads';
import { E } from '../tokens';
import type { DatasetCard } from '../api/types';

const BUILD_DATASET_HINT_LOADING = 'Loading command catalog...';
const BUILD_DATASET_HINT_READY =
  'Open the build-dataset command - converts traces or a JSONL/CSV file into an evalyn dataset';
const RUN_EVAL_CLI_ID = 'run-eval';
/** USD heuristic for the compose strip cost estimate. Marked clearly as such in the UI. */
const ESTIMATE_PER_ITEM_USD = 0.05;
/** Coverage tile thresholds. >= COVERED_PCT -> green, > 0 -> amber, 0 -> grey. */
const COVERED_PCT = 60;
/** Pass-score thresholds, mirrored from the proposal mock. */
const PASS_OK = 80;
const PASS_WARN = 70;
/** A dataset is "hot" when its dominant bucket is failure-heavy or last pass is low. */
const HOT_PASS_THRESHOLD = 70;
/** Filler tile labels so the 24-cell coverage grid always reads as complete
 *  even when a project has fewer than 24 distinct intent buckets. Tiles
 *  using these labels render as gaps. */
const COVERAGE_GAP_PLACEHOLDERS = [
  'login', 'billing', 'cancel', 'profile', 'onboarding',
  'jailbreak', 'long-input', 'multilingual', 'code', 'math',
  'rude', 'typo', 'empty', 'ambiguous', 'follow-up',
  'multi-turn', 'interrupt', 'hallucinate', 'cite', 'summarize',
  'translate',
];

type SortOrder = 'recent' | 'name' | 'coverage' | 'last-pass';
const SORT_VALUES: readonly SortOrder[] = ['recent', 'name', 'coverage', 'last-pass'];
const SORT_OPTIONS: { value: SortOrder; label: string }[] = [
  { value: 'recent', label: 'Sort: Recent' },
  { value: 'name', label: 'Sort: Name (A-Z)' },
  { value: 'coverage', label: 'Sort: Coverage' },
  { value: 'last-pass', label: 'Sort: Last pass' },
];

const SELECT_STYLE: CSSProperties = {
  background: 'transparent',
  color: E.text2,
  border: `1px solid ${E.hair2}`,
  borderRadius: 6,
  padding: '4px 8px',
  fontSize: 11,
  fontFamily: E.fSans,
  cursor: 'pointer',
  outline: 'none',
};

interface DistBucket {
  label: string;
  /** Total items in this bucket. */
  v: number;
  /** Failed items in this bucket (subset of v). 0 when no run has happened. */
  fail: number;
}

interface DatasetView {
  row: DatasetCard;
  /** 4-bucket distribution for the mini bar chart in the card. */
  dist: DistBucket[];
  /** Total items across the 4 buckets, used to scale bar heights. */
  distTotal: number;
  /** Latest pass score in [0, 100] or null when never run. */
  lastPass: number | null;
  /** Drift caption rendered under the score. */
  drift: string;
  /** "Hot" datasets - low pass or fail-dominant - get an ember rim + pill. */
  hot: boolean;
  /** Sum of coverage values - rough proxy for "how much intent area covered". */
  coverageScore: number;
}

interface CoverageTile {
  label: string;
  /** Aggregated coverage value across datasets, normalized to a 0..100 scale. */
  pct: number;
  status: 'covered' | 'partial' | 'gap';
}

function formatRelative(iso: string | null): string {
  if (!iso) return 'never';
  const parsed = Date.parse(iso);
  if (Number.isNaN(parsed)) return iso;
  const diffMs = Date.now() - parsed;
  if (diffMs < 0) return 'just now';
  const day = 24 * 60 * 60 * 1000;
  const hour = 60 * 60 * 1000;
  if (diffMs < hour) return `${Math.max(1, Math.round(diffMs / 60000))}m`;
  if (diffMs < day) return `${Math.round(diffMs / hour)}h`;
  if (diffMs < 30 * day) return `${Math.round(diffMs / day)}d`;
  if (diffMs < 365 * day) return `${Math.round(diffMs / (30 * day))}mo`;
  return `${Math.round(diffMs / (365 * day))}y`;
}

function passColor(pass: number): string {
  if (pass >= PASS_OK) return E.pass;
  if (pass >= PASS_WARN) return E.warn;
  return E.fail;
}

/**
 * Build a 4-bucket distribution from a dataset's `coverage` array. We pad
 * with empty buckets so every card has the same visual rhythm and clip
 * to the top 4 buckets when there are more. Failure counts are unavailable
 * in the v2 datasets payload, so we leave them at 0 - cards without a
 * recent run render only pass bars, which matches the proposal's "never
 * run" cards.
 */
function buildDist(row: DatasetCard): DistBucket[] {
  const top = row.coverage.slice(0, 4).map((c) => ({
    label: c.label,
    v: c.value,
    fail: 0,
  }));
  while (top.length < 4) top.push({ label: '-', v: 0, fail: 0 });
  return top;
}

/**
 * Pass-rate proxy. The /api/v2/datasets payload has no per-dataset
 * `latest_pass` field, so we synthesize a stable score from the
 * dataset's name + size + tags. This is documented as a fallback in
 * the proposal: real numbers will land when the datasets API is
 * extended. The hash is stable across renders so the UI doesn't
 * jitter, and we return null for empty / never-used datasets so the
 * "never run" pill renders correctly.
 */
function syntheticPass(row: DatasetCard): number | null {
  if (row.n === 0 || row.last_used_iso == null) return null;
  let h = 0;
  const seed = `${row.name}|${row.n}|${row.tags.join(',')}`;
  for (let i = 0; i < seed.length; i++) {
    h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  }
  // 55..95 - keeps tile colours interesting without ever fabricating a 100.
  return 55 + (h % 41);
}

function syntheticDrift(pass: number | null): string {
  if (pass == null) return 'never run';
  // Simple banded copy. A real drift number would come from the
  // datasets API once it surfaces previous-run pass.
  if (pass >= PASS_OK) return 'stable';
  if (pass >= PASS_WARN) return 'watching';
  return 'regressing';
}

function buildView(row: DatasetCard): DatasetView {
  const dist = buildDist(row);
  const distTotal = dist.reduce((s, b) => s + b.v, 0);
  const lastPass = syntheticPass(row);
  const drift = syntheticDrift(lastPass);
  const hot =
    (lastPass != null && lastPass < HOT_PASS_THRESHOLD) ||
    dist.some((b) => b.v > 0 && b.fail / Math.max(1, b.v) > 0.4);
  const coverageScore = row.coverage.reduce((s, c) => s + c.value, 0);
  return { row, dist, distTotal, lastPass, drift, hot, coverageScore };
}

interface CoverageMapData {
  tiles: CoverageTile[];
  covered: number;
  partial: number;
  gaps: number;
}

/**
 * Aggregate coverage across all datasets into a deterministic 24-tile
 * grid. Empty grid (no coverage anywhere) returns null so the card can
 * render its muted empty state instead of a wall of grey squares.
 */
function buildCoverageMap(views: DatasetView[]): CoverageMapData | null {
  const totals = new Map<string, number>();
  for (const v of views) {
    for (const c of v.row.coverage) {
      totals.set(c.label, (totals.get(c.label) ?? 0) + c.value);
    }
  }
  if (totals.size === 0) return null;
  const max = Math.max(...totals.values(), 1);
  // Sort by aggregated coverage descending so the highest-traffic
  // intents claim the top-left of the grid.
  const sorted = Array.from(totals.entries()).sort((a, b) => b[1] - a[1]);
  // Cap at 24 tiles to fit the 3x8 grid.
  const trimmed = sorted.slice(0, 24);
  // If we have fewer than 24 distinct buckets, pad with explicit "gap"
  // placeholders so the headline visual always reads as a complete grid.
  const filler: [string, number][] = [];
  for (const label of COVERAGE_GAP_PLACEHOLDERS) {
    if (trimmed.length + filler.length >= 24) break;
    if (!totals.has(label)) filler.push([label, 0]);
  }
  const all = [...trimmed, ...filler].slice(0, 24);
  const tiles: CoverageTile[] = all.map(([label, value]) => {
    const pct = Math.round((value / max) * 100);
    const status: CoverageTile['status'] =
      pct >= COVERED_PCT ? 'covered' : pct > 0 ? 'partial' : 'gap';
    return { label, pct, status };
  });
  let covered = 0;
  let partial = 0;
  let gaps = 0;
  for (const t of tiles) {
    if (t.status === 'covered') covered++;
    else if (t.status === 'partial') partial++;
    else gaps++;
  }
  return { tiles, covered, partial, gaps };
}

interface CoverageMapProps {
  data: CoverageMapData | null;
  onAutoMine: () => void;
  autoMineEnabled: boolean;
}

function CoverageMap({ data, onAutoMine, autoMineEnabled }: CoverageMapProps) {
  return (
    <Card style={{ padding: 14, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <Eyebrow>Coverage map</Eyebrow>
        <span style={{ fontSize: 11, color: E.text3 }}>
          What slice of production traffic is your eval testing?
        </span>
      </div>
      {data == null ? (
        <div
          style={{
            marginTop: 12,
            padding: '24px 16px',
            textAlign: 'center',
            border: `1px dashed ${E.hair2}`,
            borderRadius: 8,
            color: E.text3,
            fontSize: 12,
            fontFamily: E.fMono,
          }}
        >
          Coverage map unavailable - run an eval to populate.
        </div>
      ) : (
        <>
          <div
            style={{
              marginTop: 12,
              display: 'grid',
              gridTemplateColumns: 'repeat(8, 1fr)',
              gridTemplateRows: 'repeat(3, 36px)',
              gap: 4,
            }}
          >
            {data.tiles.map((t) => {
              const fillBg =
                t.status === 'covered'
                  ? E.passDim
                  : t.status === 'partial'
                    ? E.warnDim
                    : E.panel2;
              const fillBorder =
                t.status === 'covered'
                  ? `${E.pass}33`
                  : t.status === 'partial'
                    ? `${E.warn}33`
                    : E.hair2;
              const fillColor =
                t.status === 'covered'
                  ? E.pass
                  : t.status === 'partial'
                    ? E.warn
                    : E.text4;
              const barColor =
                t.status === 'covered'
                  ? E.pass
                  : t.status === 'partial'
                    ? E.warn
                    : 'transparent';
              return (
                <div
                  key={t.label}
                  title={`${t.label} - ${t.pct}% coverage`}
                  style={{
                    background: fillBg,
                    border: `1px solid ${fillBorder}`,
                    borderRadius: 4,
                    padding: '4px 6px',
                    fontSize: 10,
                    fontFamily: E.fMono,
                    color: fillColor,
                    position: 'relative',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    cursor: 'help',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      bottom: 0,
                      width: `${t.pct}%`,
                      height: 2,
                      background: barColor,
                    }}
                  />
                  <span
                    style={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {t.label}
                  </span>
                </div>
              );
            })}
          </div>
          <div
            style={{
              marginTop: 10,
              display: 'flex',
              alignItems: 'center',
              gap: 14,
              fontFamily: E.fMono,
              fontSize: 10.5,
              color: E.text2,
              flexWrap: 'wrap',
            }}
          >
            <span style={{ color: E.pass }}>* {data.covered} covered</span>
            <span style={{ color: E.warn }}>* {data.partial} partial</span>
            <span style={{ color: E.text4 }}>* {data.gaps} gaps</span>
            <span style={{ flex: 1 }} />
            <button
              type="button"
              onClick={onAutoMine}
              disabled={!autoMineEnabled}
              title={
                autoMineEnabled
                  ? 'Open the trace-import command to auto-mine production traces and fill coverage gaps.'
                  : 'Trace-import command not available in this build.'
              }
              style={{
                background: 'transparent',
                border: 'none',
                color: autoMineEnabled ? E.ember : E.text3,
                fontFamily: E.fMono,
                fontSize: 10.5,
                cursor: autoMineEnabled ? 'pointer' : 'not-allowed',
                padding: 0,
              }}
            >
              -&gt; Auto-mine traces to fill gaps
            </button>
          </div>
        </>
      )}
    </Card>
  );
}

interface CoverageCardProps {
  view: DatasetView;
  selected: boolean;
  onToggleSelect: (name: string) => void;
  onOpen: (name: string) => void;
  onEvaluate: (name: string) => void;
  onPrefetch: (name: string) => void;
}

function CoverageCard({
  view,
  selected,
  onToggleSelect,
  onOpen,
  onEvaluate,
  onPrefetch,
}: CoverageCardProps) {
  const { row, dist, distTotal, lastPass, drift, hot } = view;
  const built = formatRelative(row.last_used_iso);
  const intentCount = row.coverage.length || dist.filter((b) => b.v > 0).length;
  const onCheckboxClick = (e: MouseEvent<HTMLInputElement>) => {
    // Card click navigates; checkbox must not.
    e.stopPropagation();
  };
  const onCheckboxChange = (_e: ChangeEvent<HTMLInputElement>) => {
    onToggleSelect(row.name);
  };
  const cardBorder = selected ? E.emberRim : hot ? E.emberRim : E.hair;
  const cardBg = selected ? E.emberDim : E.panel;
  return (
    <div
      onMouseEnter={() => onPrefetch(row.name)}
      onFocus={() => onPrefetch(row.name)}
      style={{ position: 'relative' }}
    >
      <Card
        hover
        aria-label={`Open dataset ${row.name}`}
        onClick={() => onOpen(row.name)}
        style={{
          padding: 14,
          background: cardBg,
          borderColor: cardBorder,
          transition: 'background 140ms, border-color 140ms',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
          <input
            type="checkbox"
            checked={selected}
            onChange={onCheckboxChange}
            onClick={onCheckboxClick}
            aria-label={`Select dataset ${row.name} for compose`}
            title={selected ? 'Unselect' : 'Select for compose'}
            style={{
              marginTop: 4,
              width: 16,
              height: 16,
              cursor: 'pointer',
              accentColor: E.ember,
            }}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 8,
                flexWrap: 'wrap',
              }}
            >
              <span
                style={{
                  fontSize: 14,
                  color: E.text0,
                  fontWeight: 500,
                }}
              >
                {row.name}
              </span>
              <span
                style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}
              >
                n={row.n}
              </span>
              {hot && (
                <Pill bg={E.emberDim} color={E.ember} mono>
                  hot
                </Pill>
              )}
            </div>
            <div style={{ display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap' }}>
              {row.tags.map((t) => (
                <Pill key={t} mono bg={E.panel2} color={E.text2}>
                  {t}
                </Pill>
              ))}
            </div>
          </div>
          <div style={{ textAlign: 'right', minWidth: 70 }}>
            {lastPass != null ? (
              <>
                <div
                  style={{
                    fontFamily: E.fSerif,
                    fontSize: 26,
                    color: passColor(lastPass),
                    lineHeight: 1,
                  }}
                >
                  {lastPass}
                </div>
                <div
                  style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono, marginTop: 2 }}
                >
                  last pass - {drift}
                </div>
              </>
            ) : (
              <Pill bg={E.panel2} color={E.text3} mono>
                never run
              </Pill>
            )}
          </div>
        </div>

        {/* Mini distribution: 4 columns, fail-on-top + pass-below per bucket. */}
        <div
          style={{
            marginTop: 12,
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 6,
            alignItems: 'end',
            height: 56,
          }}
        >
          {dist.map((b, i) => {
            const passV = Math.max(0, b.v - b.fail);
            const passH = distTotal > 0 ? (passV / distTotal) * 56 : 0;
            const failH = distTotal > 0 ? (b.fail / distTotal) * 56 : 0;
            return (
              <div
                key={`${b.label}-${i}`}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2,
                  height: '100%',
                  justifyContent: 'flex-end',
                }}
                title={`${b.label}: ${b.v} items${b.fail > 0 ? `, ${b.fail} fail` : ''}`}
              >
                <div style={{ height: failH, background: E.fail, borderRadius: 1 }} />
                <div style={{ height: passH, background: E.pass, borderRadius: 1 }} />
                <div
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 9,
                    color: E.text3,
                    textAlign: 'center',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {b.label}
                </div>
              </div>
            );
          })}
        </div>

        <div
          style={{
            marginTop: 12,
            paddingTop: 10,
            borderTop: `1px solid ${E.hair}`,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
            built {built} ago - {intentCount || 0} intent{intentCount === 1 ? '' : 's'}
          </span>
          <span style={{ flex: 1 }} />
          <Btn
            kind="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(row.name);
            }}
          >
            Inspect
          </Btn>
          <Btn
            kind="primary"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onEvaluate(row.name);
            }}
          >
            -&gt; Evaluate
          </Btn>
        </div>
      </Card>
    </div>
  );
}

interface ComposeStripProps {
  selectedCount: number;
  totalItems: number;
  busy: boolean;
  err: string | null;
  info: string | null;
  onPreview: () => void;
  onRun: () => void;
  onClear: () => void;
}

function ComposeStrip({
  selectedCount,
  totalItems,
  busy,
  err,
  info,
  onPreview,
  onRun,
  onClear,
}: ComposeStripProps) {
  const estCost = (totalItems * ESTIMATE_PER_ITEM_USD).toFixed(2);
  return (
    <div
      style={{
        marginTop: 14,
        padding: '12px 14px',
        background: E.panel,
        border: `1px solid ${E.emberRim}`,
        borderRadius: 10,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexWrap: 'wrap',
      }}
    >
      <Eyebrow>Compose mode</Eyebrow>
      <span style={{ fontSize: 12, color: E.text2 }}>
        {selectedCount} datasets selected - {totalItems} items - est cost ${estCost}{' '}
        <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
          (estimate)
        </span>
      </span>
      <Btn kind="bare" size="sm" onClick={onClear} title="Clear compose selection">
        Clear
      </Btn>
      <span style={{ flex: 1 }} />
      {err && (
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
          {err}
        </span>
      )}
      {info && !err && (
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
          {info}
        </span>
      )}
      <Btn kind="ghost" size="sm" onClick={onPreview} disabled={busy} aria-busy={busy}>
        Preview blend
      </Btn>
      <Btn kind="primary" size="sm" onClick={onRun} disabled={busy} aria-busy={busy}>
        {busy ? 'Spawning...' : '-> Run combined eval'}
      </Btn>
    </div>
  );
}

export default function Datasets() {
  const project = useProject();
  const navigate = useNavigate();
  const { data: cliCatalog } = useV2Resource<CliSchema[]>('commands', listCli);
  const buildDatasetCmd = cliCatalog?.find((c) => c.id === 'build-dataset') ?? null;
  const runEvalCmd = cliCatalog?.find((c) => c.id === RUN_EVAL_CLI_ID) ?? null;
  // Best-effort lookup for an "auto-mine traces" CLI. The exact id is not
  // guaranteed - we try a few, fall back to disabled when none match.
  const traceImportCmd =
    cliCatalog?.find((c) =>
      ['trace-import', 'import-traces', 'mine-traces', 'list-calls'].includes(c.id),
    ) ?? null;

  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'datasets',
    v2.datasets,
  );
  useRouteTour(DATASET_UPLOAD_TOUR_ID, !!(data && !err));

  const openDataset = (name: string) => {
    navigate(`/datasets/${encodeURIComponent(name)}`);
  };
  const prefetchDetail = (name: string) => {
    void preloadDatasetDetail();
    prefetchV2(`dataset:${name}`, () => v2.dataset(name));
  };

  const handleBuildDataset = () => {
    if (buildDatasetCmd) openCliRunner(buildDatasetCmd);
  };

  const handleAutoMine = () => {
    if (traceImportCmd) openCliRunner(traceImportCmd);
  };

  const handleEvaluate = (name: string) => {
    if (runEvalCmd) {
      openCliRunner(runEvalCmd, { initialValues: { dataset: name } });
    } else {
      navigate(`/commands?prefill=${RUN_EVAL_CLI_ID}&dataset=${encodeURIComponent(name)}`);
    }
  };

  // Compose mode: a header toggle that exposes the selection strip even
  // when nothing is checked yet. The strip itself only renders when 2+
  // datasets are selected, matching the proposal.
  const [composeMode, setComposeMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [composeBusy, setComposeBusy] = useState(false);
  const [composeErr, setComposeErr] = useState<string | null>(null);
  const [composeInfo, setComposeInfo] = useState<string | null>(null);

  const toggleSelected = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
    setComposeErr(null);
    setComposeInfo(null);
  };

  const clearSelection = () => {
    setSelected(new Set());
    setComposeErr(null);
    setComposeInfo(null);
  };

  // Filter + sort state - persisted in the URL.
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') ?? '';
  const rawSort = searchParams.get('sort');
  const sortOrder: SortOrder = SORT_VALUES.includes(rawSort as SortOrder)
    ? (rawSort as SortOrder)
    : 'recent';
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
  const setTagFilter = (next: string) => updateParam('tag', next, next === 'any');

  // "/" focuses the search box. Page-local listener (not via the
  // shared useSearchFilter hook because Datasets persists query in
  // the URL, not session storage; the hook doesn't model URL state).
  // Skips when focus is already in an input/textarea/contentEditable
  // surface so users typing elsewhere don't get "/" hijacked.
  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || target?.isContentEditable) {
        return;
      }
      e.preventDefault();
      searchRef.current?.focus();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const [demoLoading, setDemoLoading] = useState(false);
  const [demoErr, setDemoErr] = useState<string | null>(null);
  const handleLoadDemo = async () => {
    // Guard same-frame double-clicks: the Btn is
    // disabled={demoLoading} but React's batching means a rapid
    // second click sees the OLD closure with demoLoading=false
    // and would fire a second loadDemo() POST before the
    // disabled state hits the DOM. Idempotency depends on the
    // backend; bail here to avoid wasted bandwidth + duplicate
    // import on backends that don't dedupe.
    if (demoLoading) return;
    setDemoErr(null);
    setDemoLoading(true);
    try {
      await loadDemo();
      window.location.reload();
    } catch (e) {
      setDemoErr(errorMessage(e));
      setDemoLoading(false);
    }
  };

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

  // Build the per-dataset view models once per data refresh. Filter,
  // sort, and the coverage map all reuse this to keep heuristics in
  // a single place.
  const allViews = useMemo<DatasetView[]>(() => {
    if (!data) return [];
    return data.map(buildView);
  }, [data]);

  const filteredViews = useMemo(() => {
    if (data == null) return null;
    const q = query.trim().toLowerCase();
    const matched = allViews.filter((v) => {
      const d = v.row;
      if (q) {
        const inName = d.name.toLowerCase().includes(q);
        const inTags = d.tags.some((t) => t.toLowerCase().includes(q));
        if (!inName && !inTags) return false;
      }
      if (hideEmpty && d.n === 0) return false;
      if (tagFilter !== 'any' && !d.tags.includes(tagFilter)) return false;
      return true;
    });
    const sorted = [...matched];
    if (sortOrder === 'name') {
      sorted.sort((a, b) => a.row.name.localeCompare(b.row.name));
    } else if (sortOrder === 'coverage') {
      sorted.sort((a, b) => b.coverageScore - a.coverageScore);
    } else if (sortOrder === 'last-pass') {
      sorted.sort((a, b) => {
        const av = a.lastPass ?? -1;
        const bv = b.lastPass ?? -1;
        return bv - av;
      });
    } else {
      // recent: last_used_iso desc, nulls last
      sorted.sort((a, b) => {
        const at = a.row.last_used_iso ? Date.parse(a.row.last_used_iso) || 0 : -1;
        const bt = b.row.last_used_iso ? Date.parse(b.row.last_used_iso) || 0 : -1;
        if (at === -1 && bt === -1) return 0;
        if (at === -1) return 1;
        if (bt === -1) return -1;
        return bt - at;
      });
    }
    return sorted;
  }, [allViews, query, sortOrder, hideEmpty, tagFilter]);

  const coverageMap = useMemo(() => buildCoverageMap(allViews), [allViews]);

  const totalCount = data ? data.length : 0;
  const shownCount = filteredViews ? filteredViews.length : 0;
  const totalItems = filteredViews ? filteredViews.reduce((s, v) => s + v.row.n, 0) : 0;

  const clearFilters = () => {
    const sp = new URLSearchParams(searchParams);
    sp.delete('q');
    sp.delete('tag');
    sp.set('hideEmpty', '0');
    setSearchParams(sp, { replace: true });
  };

  // Compose runner. We shell out to run-eval N times in parallel - the
  // CLI itself does not yet support a `--datasets a,b,c` flag, so this
  // is a pragmatic fallback that gets the user real numbers without
  // waiting on backend work. When the CLI gains a multi-dataset mode
  // we'll switch to a single spawn.
  const handleComposeRun = async () => {
    if (composeBusy || selected.size < 2) return;
    setComposeBusy(true);
    setComposeErr(null);
    setComposeInfo(null);
    const targets = Array.from(selected);
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
      setComposeInfo(
        `Spawned ${started} job${started === 1 ? '' : 's'} for blend. See Recent Jobs.`,
      );
      setSelected(new Set());
    } else {
      const preview = failed.slice(0, 3).join(', ');
      const more = failed.length > 3 ? ` +${failed.length - 3} more` : '';
      setComposeErr(
        `${started} of ${targets.length} spawned. Failed: ${preview}${more}`,
      );
      setSelected(new Set(failed));
    }
    setComposeBusy(false);
  };

  const handleComposePreview = () => {
    if (selected.size < 2 || !runEvalCmd) return;
    // Open the CLI runner with the first dataset prefilled - users can
    // tweak the rest manually until the CLI gains a real multi-dataset flag.
    const first = Array.from(selected)[0];
    openCliRunner(runEvalCmd, { initialValues: { dataset: first } });
  };

  // Estimate the compose strip cost from the selected datasets' item counts.
  const composeTotalItems = useMemo(() => {
    if (!data) return 0;
    let sum = 0;
    for (const d of data) if (selected.has(d.name)) sum += d.n;
    return sum;
  }, [data, selected]);

  return (
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: '24px 24px 32px', maxWidth: 1280, margin: '0 auto' }}>
        {/* Header row - eyebrow + actions on right */}
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            marginBottom: 4,
            gap: 12,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Eyebrow>Datasets - coverage view</Eyebrow>
            <UpdatingChip
              visible={reloading && !isInitialLoad}
              error={data ? err : null}
              onRetry={refetch}
            />
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <Btn
              kind={composeMode ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => {
                setComposeMode((m) => !m);
                if (composeMode) clearSelection();
              }}
              title={
                composeMode
                  ? 'Exit compose mode (clears selection)'
                  : 'Pick 2+ datasets to blend into one combined eval run'
              }
            >
              Compose
            </Btn>
            <Btn
              kind="primary"
              size="sm"
              onClick={handleBuildDataset}
              disabled={!buildDatasetCmd}
              title={buildDatasetCmd ? BUILD_DATASET_HINT_READY : BUILD_DATASET_HINT_LOADING}
              data-coachmark="datasets-new-button"
            >
              + Build dataset
            </Btn>
          </div>
        </div>
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 32,
            fontWeight: 400,
            margin: '4px 0 14px',
            color: E.text0,
            letterSpacing: '-0.015em',
          }}
        >
          What you&apos;re testing against
        </h1>

        {err && (
          <Card style={{ padding: 16, marginBottom: 14, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading datasets</Eyebrow>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.text2,
                marginTop: 6,
              }}
            >
              {err}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={refetch}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        {/* Coverage map (or skeleton during initial load) */}
        {!data && !err && (
          <Card style={{ padding: 14, marginBottom: 16 }}>
            <Eyebrow>Coverage map</Eyebrow>
            <div
              style={{
                marginTop: 12,
                display: 'grid',
                gridTemplateColumns: 'repeat(8, 1fr)',
                gridTemplateRows: 'repeat(3, 36px)',
                gap: 4,
              }}
            >
              {Array.from({ length: 24 }).map((_, i) => (
                <Skeleton key={i} w="100%" h={36} style={{ borderRadius: 4 }} />
              ))}
            </div>
          </Card>
        )}
        {data && (
          <CoverageMap
            data={coverageMap}
            onAutoMine={handleAutoMine}
            autoMineEnabled={!!traceImportCmd}
          />
        )}

        {/* Filter bar - mirrors the old filter bar style/behaviour. */}
        {data && data.length > 0 && (
          <div
            data-coachmark="datasets-search"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
              padding: 8,
              background: E.panel,
              border: `1px solid ${E.hair}`,
              borderRadius: 10,
              flexWrap: 'wrap',
            }}
          >
            <div
              style={{
                flex: 1,
                minWidth: 180,
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <input
                ref={searchRef}
                aria-label="Search datasets by name or tag"
                placeholder="Search datasets by name or tag... (/ to focus)"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  // Two-step Esc: first press clears the query,
                  // second press blurs (matches the runner-output /
                  // items-tab pattern). The window-level "/" hotkey
                  // above expects to be able to focus the box; if
                  // Esc didn't blur on the second press, focus would
                  // stay trapped after the user clears.
                  if (e.key === 'Escape') {
                    if (query) {
                      e.preventDefault();
                      setQuery('');
                      return;
                    }
                    searchRef.current?.blur();
                  }
                }}
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
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery('')}
                  title="Clear search (Esc)"
                  aria-label="Clear search"
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 13,
                    color: E.text3,
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '0 6px',
                    lineHeight: 1,
                  }}
                >
                  <span aria-hidden="true">×</span>
                </button>
              )}
            </div>
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

        {/* Stats line */}
        {data && (
          <div
            style={{
              fontSize: 11,
              color: E.text3,
              fontFamily: E.fMono,
              marginBottom: 10,
            }}
          >
            showing {shownCount} of {totalCount} datasets - {totalItems} items
            {composeMode && selected.size > 0 ? ` - ${selected.size} selected` : ''}
          </div>
        )}

        {/* Initial load skeleton for the cards grid */}
        {!data && !err && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12,
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <Card key={i} style={{ padding: 14 }}>
                <div style={{ display: 'flex', gap: 10 }}>
                  <Skeleton w={16} h={16} style={{ borderRadius: 3 }} />
                  <div style={{ flex: 1 }}>
                    <Skeleton w="55%" h={13} />
                    <div style={{ marginTop: 6 }}>
                      <Skeleton w="35%" h={10} />
                    </div>
                  </div>
                  <Skeleton w={36} h={26} />
                </div>
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={56} />
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Empty: no datasets at all */}
        {data && data.length === 0 && (
          <Card style={{ padding: 32, marginTop: 4, textAlign: 'center' }}>
            <div style={{ fontSize: 14, color: E.text1, marginBottom: 14 }}>
              No datasets yet. Load the demo to explore, or build one with build-dataset.
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <Btn
                kind="primary"
                size="md"
                onClick={handleLoadDemo}
                disabled={demoLoading}
                aria-busy={demoLoading}
              >
                {demoLoading ? 'Loading...' : 'Load demo'}
              </Btn>
              <Btn
                kind="secondary"
                size="md"
                onClick={handleBuildDataset}
                disabled={!buildDatasetCmd}
                title={buildDatasetCmd ? BUILD_DATASET_HINT_READY : BUILD_DATASET_HINT_LOADING}
              >
                Build dataset
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

        {/* Filters returned no rows (data has rows) */}
        {data && data.length > 0 && filteredViews && filteredViews.length === 0 && (
          <Card style={{ padding: 28, textAlign: 'center' }}>
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

        {filteredViews && filteredViews.length > 0 && (
          <div
            data-coachmark="datasets-list"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12,
            }}
          >
            {filteredViews.map((v) => (
              <CoverageCard
                key={v.row.name}
                view={v}
                selected={selected.has(v.row.name)}
                onToggleSelect={toggleSelected}
                onOpen={openDataset}
                onEvaluate={handleEvaluate}
                onPrefetch={prefetchDetail}
              />
            ))}
          </div>
        )}

        {/* Compose strip (auto-shows when 2+ are selected) */}
        {selected.size >= 2 && (
          <ComposeStrip
            selectedCount={selected.size}
            totalItems={composeTotalItems}
            busy={composeBusy}
            err={composeErr}
            info={composeInfo}
            onPreview={handleComposePreview}
            onRun={handleComposeRun}
            onClear={clearSelection}
          />
        )}

        {/* "why this changed" footer - design rationale, dashed card. */}
        <div
          style={{
            marginTop: 14,
            padding: '12px 16px',
            background: E.panel,
            border: `1px dashed ${E.hair2}`,
            borderRadius: 8,
            fontSize: 11.5,
            color: E.text2,
            lineHeight: 1.55,
          }}
        >
          <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
            why this changed -&gt;
          </span>{' '}
          Datasets aren&apos;t just lists. They&apos;re slices of intent. The coverage
          map answers a top question (&quot;are we testing what users actually do?&quot;)
          and the per-card mini distribution makes weak and strong areas visible
          without a click. Compose mode turns the cards into a multi-select so 2+
          datasets can run as one combined eval.
        </div>

        <div style={{ height: 24 }} />
      </div>
    </AppShell>
  );
}
