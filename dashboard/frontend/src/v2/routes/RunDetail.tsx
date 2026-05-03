/**
 * RunDetail - the deepest read view: headline stats, pass-rate vs baseline,
 * failure clusters, sub-metric breakdown, confusion matrix, failed item preview.
 *
 * Compare overlay: when ?compare=<otherId> is in the URL, a second
 * ExperimentDetail is fetched in parallel and several sections render
 * side-by-side (headline, pass timeline legend, sub-metrics, failures donut).
 */

import { useCallback, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import {
  Card,
  Eyebrow,
  Glossary,
  Pill,
  Btn,
  Spinner,
  StatusDot,
  Donut,
  LineChart,
  Skeleton,
  UpdatingChip,
} from '../ui';
import type { LineSeries } from '../ui';
import { v2 } from '../api/client';
import { listCli } from '../api/cli';
import type { CliSchema } from '../api/cli';
import type {
  ExperimentDetail,
  ExperimentItemsFilter,
  ExperimentItemsResponse,
  ExperimentItemsSort,
} from '../api/types';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { openCliRunner } from '../cliRunnerBridge';
import { E } from '../tokens';

const CLUSTER_COLOR: Record<string, string> = {
  fail: E.fail,
  warn: E.warn,
  steel: E.steel,
  violet: '#a78bfa',
  text3: E.text3,
};

const SERIES_COLOR: Record<string, string> = {
  ember: E.ember,
  steel: E.steel,
  fail: E.fail,
};

/** Numeric movement smaller than this is rendered as neutral (no win/loss). */
const NEUTRAL_DELTA_EPS = 0.05;

function deltaColor(kind: 'pass' | 'fail' | 'warn' | 'info'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  if (kind === 'warn') return E.warn;
  return E.steel;
}

/** Format a numeric delta as "+1.2" / "-3.4" with neutral-zero smoothing. */
function formatDelta(diff: number, suffix: string = ''): string {
  if (Math.abs(diff) < NEUTRAL_DELTA_EPS) return `0${suffix}`;
  const sign = diff > 0 ? '+' : '';
  return `${sign}${diff.toFixed(1)}${suffix}`;
}

/**
 * Color a numeric delta. inverse=true means lower-is-better
 * (hallucination-style metrics).
 */
function numericDeltaColor(diff: number, inverse: boolean = false): string {
  if (Math.abs(diff) < NEUTRAL_DELTA_EPS) return E.text3;
  const better = inverse ? diff < 0 : diff > 0;
  return better ? E.pass : E.fail;
}

/**
 * Strip a trailing % off the headline value strings the backend returns
 * ("89.3%") so we can compare numerically. Returns null if not a percent.
 */
function parseHeadlineNumber(s: string): number | null {
  const m = s.match(/^(-?\d+(?:\.\d+)?)/);
  return m ? Number(m[1]) : null;
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [searchParams] = useSearchParams();
  const compareParam = searchParams.get('compare');
  // Sanity: ignore compare param when it's the same id as the primary run.
  const compareWith =
    compareParam && compareParam !== runId ? compareParam : null;
  const navigate = useNavigate();
  const project = useProject();
  const fetcher = useCallback(() => v2.experiment(runId ?? ''), [runId]);
  const {
    data: detail,
    err,
    reloading,
    isInitialLoad,
  } = useV2Resource<ExperimentDetail>(`experiment:${runId ?? ''}`, fetcher);

  // Compare-run fetch (parallel) - only enabled when compareWith is set.
  const compareFetcher = useCallback(
    () => v2.experiment(compareWith ?? ''),
    [compareWith],
  );
  const {
    data: compareDetail,
    err: compareErr,
  } = useV2Resource<ExperimentDetail>(
    `experiment:${compareWith ?? ''}`,
    compareFetcher,
    { enabled: Boolean(compareWith) },
  );
  const compareActive = Boolean(compareWith && compareDetail && !compareErr);
  const [activeTab, setActiveTab] = useState(0);
  const [rerunBusy, setRerunBusy] = useState(false);

  function clearCompare(): void {
    navigate(`/experiments/${encodeURIComponent(runId ?? '')}`);
  }
  function swapCompare(): void {
    if (!compareWith || !runId) return;
    navigate(
      `/experiments/${encodeURIComponent(compareWith)}?compare=${encodeURIComponent(runId)}`,
    );
  }

  if (err && !detail) {
    return (
      <AppShell breadcrumb={['Experiments', runId ?? '']}>
        <div style={{ padding: '32px 36px' }}>
          <Card style={{ padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Run not found</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
            <div style={{ marginTop: 12 }}>
              <Btn kind="secondary" size="sm" onClick={() => navigate('/experiments')}>
                ← Back to experiments
              </Btn>
            </div>
          </Card>
        </div>
      </AppShell>
    );
  }

  if (!detail) {
    return (
      <AppShell breadcrumb={['Experiments', runId ?? '']}>
        <div style={{ padding: '28px 36px' }}>
          <Skeleton w={220} h={11} />
          <div style={{ marginTop: 8 }}>
            <Skeleton w={420} h={32} />
          </div>
          <div style={{ marginTop: 14, display: 'flex', gap: 8 }}>
            <Skeleton w={120} h={20} style={{ borderRadius: 999 }} />
            <Skeleton w={140} h={20} style={{ borderRadius: 999 }} />
            <Skeleton w={100} h={20} style={{ borderRadius: 999 }} />
          </div>
          <div
            style={{
              marginTop: 24,
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 14,
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <Card key={i} style={{ padding: 16 }}>
                <Skeleton w={100} h={11} />
                <div style={{ marginTop: 10 }}>
                  <Skeleton w={80} h={28} />
                </div>
                <div style={{ marginTop: 6 }}>
                  <Skeleton w={120} h={11} />
                </div>
              </Card>
            ))}
          </div>
          <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 14 }}>
            <Card style={{ padding: 18 }}>
              <Skeleton w={220} h={11} />
              <div style={{ marginTop: 14 }}>
                <Skeleton w="100%" h={200} style={{ borderRadius: 6 }} />
              </div>
            </Card>
            <Card style={{ padding: 18 }}>
              <Skeleton w={180} h={11} />
              <div style={{ marginTop: 14 }}>
                <Skeleton w={120} h={120} style={{ borderRadius: '50%' }} />
              </div>
            </Card>
          </div>
        </div>
      </AppShell>
    );
  }

  const tabs = [
    'Summary',
    `Items - ${detail.dataset.n}`,
    `Failures - ${detail.failure_clusters.total_failures}`,
    'Compare',
    'Trace',
  ];

  async function handleRerun() {
    if (!detail || rerunBusy) return;
    setRerunBusy(true);
    try {
      // Pull the catalog (cached after first hit) and look up `run-eval`.
      // We can't reconstruct the original argv from disk - results.json
      // doesn't carry the launching command - so we seed dataset + any
      // model/rubric metadata we happen to know. The user edits in the
      // form before clicking Run.
      const cmds: CliSchema[] = await listCli();
      const runEval = cmds.find((c) => c.id === 'run-eval');
      if (!runEval) {
        window.alert(
          'Cannot re-run: the `run-eval` command is not in this build of the CLI catalog.',
        );
        return;
      }
      const initialValues: Record<string, unknown> = {
        dataset: detail.dataset.name,
      };
      // Best-effort extras: only set fields that exist on the schema AND
      // have a non-null value on the run record. Avoids polluting the
      // form with empty strings or fields the schema doesn't declare.
      const paramNames = new Set(runEval.params.map((p) => p.name));
      if (detail.model?.id && paramNames.has('model')) {
        initialValues.model = detail.model.id;
      }
      if (detail.rubric && paramNames.has('rubric')) {
        initialValues.rubric = detail.rubric;
      }
      openCliRunner(runEval, { initialValues });
    } catch (e: unknown) {
      window.alert(`Failed to open re-run form:\n${String(e)}`);
    } finally {
      setRerunBusy(false);
    }
  }

  const headerExtra = (
    <div style={{ display: 'flex', gap: 6 }}>
      <Btn kind="ghost" size="sm" disabled title="Coming soon - export run as a shareable link">
        ↗ Share
      </Btn>
      <Btn kind="ghost" size="sm" disabled title="Coming soon - clone this run's config as a new evaluation">
        ⎘ Duplicate
      </Btn>
      <Btn
        kind="primary"
        size="sm"
        onClick={handleRerun}
        onMouseEnter={() => prefetchV2('commands', listCli)}
        disabled={rerunBusy}
        title={
          rerunBusy
            ? 'Loading...'
            : `Open the run-eval form pre-filled for ${detail.dataset.name}`
        }
      >
        {rerunBusy ? (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <Spinner size={10} /> Loading
          </span>
        ) : (
          '↻ Re-run'
        )}
      </Btn>
    </div>
  );

  const passSeries: LineSeries[] = detail.pass_timeline.series.map((s) => ({
    color: SERIES_COLOR[s.color_kind] ?? E.text2,
    width: s.color_kind === 'ember' ? 2 : 1.5,
    fill: s.color_kind === 'ember',
    data: s.data,
  }));

  // When comparing, append the OTHER run's primary (ember-kind) series
  // recolored steel so the timeline shows both lines on the same axes.
  if (compareActive && compareDetail) {
    const otherPrimary =
      compareDetail.pass_timeline.series.find((s) => s.color_kind === 'ember') ??
      compareDetail.pass_timeline.series[0];
    if (otherPrimary && otherPrimary.data.length > 0) {
      passSeries.push({
        color: E.steel,
        width: 1.75,
        dashed: true,
        data: otherPrimary.data,
      });
    }
  }

  const donutSegments = detail.failure_clusters.clusters.map((c) => ({
    value: c.count,
    color: CLUSTER_COLOR[c.color_kind] ?? E.text3,
    label: c.label,
  }));

  const compareDonutSegments = compareDetail
    ? compareDetail.failure_clusters.clusters.map((c) => ({
        value: c.count,
        color: CLUSTER_COLOR[c.color_kind] ?? E.text3,
        label: c.label,
      }))
    : [];

  // Build a colour-coded view of clusters across both runs.
  // ember = appears in this run only (introduced/persisting here)
  // pass  = appears in other run only (we fixed it)
  // steel = appears in both runs (shared / persistent)
  type CompareCluster = {
    label: string;
    thisCount: number;
    otherCount: number;
    color: string;
  };
  const compareClusters: CompareCluster[] = (() => {
    if (!compareDetail) return [];
    const thisMap = new Map<string, number>();
    const otherMap = new Map<string, number>();
    for (const c of detail.failure_clusters.clusters) thisMap.set(c.label, c.count);
    for (const c of compareDetail.failure_clusters.clusters) otherMap.set(c.label, c.count);
    const keys = new Set<string>([...thisMap.keys(), ...otherMap.keys()]);
    return Array.from(keys).map((label) => {
      const thisCount = thisMap.get(label) ?? 0;
      const otherCount = otherMap.get(label) ?? 0;
      let color: string;
      if (thisCount > 0 && otherCount === 0) color = E.ember;
      else if (otherCount > 0 && thisCount === 0) color = E.pass;
      else color = E.text3;
      return { label, thisCount, otherCount, color };
    }).sort((a, b) => b.thisCount + b.otherCount - (a.thisCount + a.otherCount));
  })();

  return (
    <AppShell
      contextChip={project ?? undefined}
      breadcrumb={['Experiments', detail.name]}
      headerExtra={headerExtra}
    >
      <div style={{ padding: '28px 36px 0' }}>
        {/* HEADER */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <StatusDot status={detail.status} animated={detail.status === 'running'} />
              <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}>
                {detail.id} - {detail.status} {detail.finished_at_iso} - {detail.duration} - {detail.cost}
              </span>
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
              {detail.name}
            </h1>
            <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap', alignItems: 'center' }}>
              <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                {detail.dataset.name} - {detail.dataset.n}
              </Pill>
              {detail.model && (
                <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                  {detail.model.id} - temp {detail.model.temp}
                </Pill>
              )}
              {detail.rubric && (
                <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                  {detail.rubric}
                </Pill>
              )}
              {detail.baseline_id && (
                <>
                  <span style={{ color: E.text4, fontSize: 11 }}>-</span>
                  <span style={{ fontSize: 11, color: E.text3 }}>vs.</span>
                  <Pill mono style={{ background: E.panel2, color: E.steel }}>
                    {detail.baseline_id}
                  </Pill>
                </>
              )}
              {compareWith && (
                <>
                  <span style={{ color: E.text4, fontSize: 11 }}>-</span>
                  <span style={{ fontSize: 11, color: E.text3 }}>compare overlay:</span>
                  <Pill mono style={{ background: E.panel2, color: E.ember }}>
                    {compareWith}
                  </Pill>
                  <Btn kind="bare" size="sm" onClick={clearCompare}>
                    Clear
                  </Btn>
                </>
              )}
            </div>
          </div>
        </div>

        {/* TABS */}
        <div style={{ display: 'flex', gap: 2, marginTop: 22, borderBottom: `1px solid ${E.hair}` }}>
          {tabs.map((t, i) => {
            const isActive = i === activeTab;
            // Summary (0) and Items (1) render in-place. Failures (2)
            // deep-links into the first cluster if one exists; the rest honest-stub.
            const firstCluster = detail.failure_clusters.clusters[0];
            const isSummary = i === 0;
            const isItems = i === 1;
            const isFailures = i === 2;
            const isComingSoon = !isSummary && !isItems && !isFailures;
            const failuresDisabled = isFailures && !firstCluster;
            const disabled = isComingSoon || failuresDisabled;
            const title = isComingSoon
              ? 'Coming soon - the Summary tab covers this surface today'
              : failuresDisabled
                ? 'No failures in this run'
                : isFailures
                  ? `Open the first failure cluster (${firstCluster?.label ?? ''})`
                  : undefined;
            return (
              <button
                key={t}
                type="button"
                disabled={disabled}
                title={title}
                onClick={() => {
                  if (disabled) return;
                  if (isFailures && firstCluster) {
                    navigate(
                      `/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(firstCluster.id)}`,
                    );
                    return;
                  }
                  setActiveTab(i);
                }}
                style={{
                  padding: '9px 14px',
                  fontSize: 12.5,
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  background: 'transparent',
                  border: 'none',
                  color: disabled ? E.text3 : isActive ? E.text0 : E.text2,
                  fontWeight: isActive ? 500 : 400,
                  borderBottom: `2px solid ${isActive ? E.ember : 'transparent'}`,
                  marginBottom: -1,
                  opacity: disabled ? 0.55 : 1,
                }}
              >
                {t}
              </button>
            );
          })}
        </div>
      </div>

      {activeTab === 1 && (
        <ItemsTab runId={detail.id} compareActive={compareActive} onClearCompare={clearCompare} />
      )}

      {activeTab === 0 && <div style={{ padding: '20px 36px' }}>
        {/* COMPARE BADGE BAR */}
        {compareWith && compareErr && !compareDetail && (
          <Card style={{ padding: 12, marginBottom: 14, borderColor: E.fail }}>
            <div style={{ fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
              Could not load compare run {compareWith}: not found
            </div>
            <div style={{ marginTop: 8 }}>
              <Btn kind="ghost" size="sm" onClick={clearCompare}>
                Clear compare
              </Btn>
            </div>
          </Card>
        )}
        {compareActive && compareDetail && (
          <Card
            style={{
              padding: 12,
              marginBottom: 14,
              background: E.steelDim,
              borderColor: E.steel,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <Eyebrow style={{ color: E.steel }}>Compare with</Eyebrow>
              <StatusDot status={compareDetail.status} />
              <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>
                {compareDetail.id}
              </span>
              <span style={{ fontSize: 12.5, color: E.text1, fontWeight: 500 }}>
                {compareDetail.name}
              </span>
              <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                {compareDetail.dataset.name} - {compareDetail.dataset.n}
              </Pill>
              <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
                {compareDetail.finished_at_iso}
              </span>
              <span style={{ flex: 1 }} />
              <Btn kind="ghost" size="sm" onClick={swapCompare} title="Swap A and B">
                ↔ Swap A/B
              </Btn>
              <Btn kind="bare" size="sm" onClick={clearCompare}>
                Clear compare
              </Btn>
            </div>
          </Card>
        )}

        {/* HEADLINE STAT ROW */}
        {detail.headline.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${detail.headline.length}, 1fr)`,
              gap: 14,
            }}
          >
            {detail.headline.map((s, i) => {
              const other =
                compareActive && compareDetail
                  ? compareDetail.headline[i] ?? null
                  : null;
              if (!other) {
                return (
                  <Card key={s.label} style={{ padding: 16 }}>
                    <Eyebrow>{s.label}</Eyebrow>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 6 }}>
                      <span style={{ fontFamily: E.fSerif, fontSize: 28, color: E.text0 }}>{s.value}</span>
                      <span style={{ fontFamily: E.fMono, fontSize: 11, color: deltaColor(s.delta_kind) }}>{s.delta}</span>
                    </div>
                    <div style={{ fontSize: 11, color: E.text3, marginTop: 4 }}>{s.sub}</div>
                  </Card>
                );
              }
              const a = parseHeadlineNumber(s.value);
              const b = parseHeadlineNumber(other.value);
              const diff = a != null && b != null ? a - b : null;
              const arrow =
                diff == null
                  ? '·'
                  : Math.abs(diff) < NEUTRAL_DELTA_EPS
                    ? '='
                    : diff > 0
                      ? '▲'
                      : '▼';
              const arrowColor =
                diff == null ? E.text3 : numericDeltaColor(diff);
              return (
                <Card key={s.label} style={{ padding: 14 }}>
                  <Eyebrow>{s.label}</Eyebrow>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'baseline',
                      gap: 6,
                      marginTop: 6,
                      fontFamily: E.fSerif,
                    }}
                  >
                    <span style={{ fontSize: 22, color: E.ember }}>{s.value}</span>
                    <span
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 12,
                        color: arrowColor,
                      }}
                      title={
                        diff != null ? `Δ ${formatDelta(diff)}` : undefined
                      }
                    >
                      {arrow}
                    </span>
                    <span style={{ fontSize: 22, color: E.steel }}>{other.value}</span>
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color: E.text3,
                      marginTop: 4,
                      fontFamily: E.fMono,
                      letterSpacing: '0.04em',
                    }}
                  >
                    this | other{diff != null ? ` - ${formatDelta(diff)}` : ''}
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* MAIN GRID: timeline + clusters */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 14, marginTop: 14 }}>
          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <Eyebrow>
                Pass rate - {compareActive ? 'this run vs. compare' : 'this run vs. baseline'}
              </Eyebrow>
              <span style={{ flex: 1 }} />
              <div style={{ display: 'flex', gap: 12, fontSize: 10, fontFamily: E.fMono, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                {detail.pass_timeline.series.map((s) => {
                  const c = SERIES_COLOR[s.color_kind] ?? E.text2;
                  const label =
                    compareActive && s.color_kind === 'ember' ? `${s.label} (${detail.id})` : s.label;
                  return (
                    <span key={s.label} style={{ color: c, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                      <span style={{ width: 8, height: 2, background: c }} />
                      {label}
                    </span>
                  );
                })}
                {compareActive && compareDetail && (
                  <span style={{ color: E.steel, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 8, height: 0, borderTop: `1px dashed ${E.steel}` }} />
                    {compareDetail.id}
                  </span>
                )}
                <span style={{ color: E.text4, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 8, height: 0, borderTop: `1px dashed ${E.text4}` }} />
                  <Glossary term="Ship gate is the minimum quality threshold for shipping. Runs above it are deemed releasable.">
                    ship gate
                  </Glossary>
                </span>
              </div>
            </div>
            {passSeries.length > 0 && (
              <LineChart
                w={620}
                h={200}
                yMin={detail.pass_timeline.y_min}
                yMax={detail.pass_timeline.y_max}
                baseline={detail.pass_timeline.ship_gate}
                xLabels={detail.pass_timeline.x_labels}
                series={passSeries}
              />
            )}
            <div style={{ fontSize: 11, color: E.text3, marginTop: 8 }}>
              Cumulative pass rate as items were graded.
            </div>
          </Card>

          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <Eyebrow>
                Failure{' '}
                <Glossary term="A failure cluster groups items that failed for the same reason. v2 buckets by metric; LLM clustering coming.">
                  clusters
                </Glossary>{' '}
                by metric - {detail.failure_clusters.total_failures} of {detail.failure_clusters.total_items}
              </Eyebrow>
              <span style={{ flex: 1 }} />
              <Btn
                kind="bare"
                size="sm"
                onClick={() => {
                  const first = detail.failure_clusters.clusters[0];
                  if (first) navigate(`/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(first.id)}`);
                }}
                disabled={detail.failure_clusters.clusters.length === 0}
              >
                Open all →
              </Btn>
            </div>
            {detail.failure_clusters.clusters.length === 0 && !compareActive ? (
              <div style={{ fontSize: 12.5, color: E.text3, padding: '12px 0' }}>No failure clusters.</div>
            ) : compareActive && compareDetail ? (
              <div>
                {/* Side-by-side donuts */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 10, color: E.ember, fontFamily: E.fMono, letterSpacing: '0.08em', marginBottom: 6 }}>
                      THIS - {detail.id}
                    </div>
                    <Donut
                      size={100}
                      thick={14}
                      segments={donutSegments}
                      center={
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontFamily: E.fSerif, fontSize: 18, color: E.text0 }}>
                            {detail.failure_clusters.total_failures}
                          </div>
                          <div style={{ fontSize: 8, color: E.text3, fontFamily: E.fMono, letterSpacing: '0.08em' }}>
                            FAILS
                          </div>
                        </div>
                      }
                    />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 10, color: E.steel, fontFamily: E.fMono, letterSpacing: '0.08em', marginBottom: 6 }}>
                      OTHER - {compareDetail.id}
                    </div>
                    <Donut
                      size={100}
                      thick={14}
                      segments={compareDonutSegments}
                      center={
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontFamily: E.fSerif, fontSize: 18, color: E.text0 }}>
                            {compareDetail.failure_clusters.total_failures}
                          </div>
                          <div style={{ fontSize: 8, color: E.text3, fontFamily: E.fMono, letterSpacing: '0.08em' }}>
                            FAILS
                          </div>
                        </div>
                      }
                    />
                  </div>
                </div>
                {/* Cluster diff list */}
                <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {compareClusters.length === 0 ? (
                    <div style={{ fontSize: 12.5, color: E.text3 }}>No failure clusters in either run.</div>
                  ) : (
                    compareClusters.slice(0, 6).map((c) => (
                      <div
                        key={c.label}
                        style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}
                        title={
                          c.color === E.ember
                            ? 'Only in this run'
                            : c.color === E.pass
                              ? 'Only in other (we fixed it)'
                              : 'In both runs'
                        }
                      >
                        <span
                          style={{
                            width: 7,
                            height: 7,
                            borderRadius: 2,
                            background: c.color,
                            flexShrink: 0,
                          }}
                        />
                        <span
                          style={{
                            flex: 1,
                            color: E.text1,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {c.label}
                        </span>
                        <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.ember, width: 26, textAlign: 'right' }}>
                          {c.thisCount}
                        </span>
                        <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text4 }}>|</span>
                        <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.steel, width: 26, textAlign: 'right' }}>
                          {c.otherCount}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
                <Donut
                  size={120}
                  thick={18}
                  segments={donutSegments}
                  center={
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>
                        {detail.failure_clusters.total_failures}
                      </div>
                      <div style={{ fontSize: 9, color: E.text3, fontFamily: E.fMono, letterSpacing: '0.08em' }}>
                        FAILS
                      </div>
                    </div>
                  }
                />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {detail.failure_clusters.clusters.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() =>
                        navigate(`/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(c.id)}`)
                      }
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        fontSize: 12,
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        textAlign: 'left',
                      }}
                    >
                      <span
                        style={{
                          width: 7,
                          height: 7,
                          borderRadius: 2,
                          background: CLUSTER_COLOR[c.color_kind] ?? E.text3,
                          flexShrink: 0,
                        }}
                      />
                      <span
                        style={{
                          flex: 1,
                          color: E.text1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {c.label}
                      </span>
                      {c.regression && (
                        <Pill
                          mono
                          color={E.fail}
                          bg={E.failDim}
                          style={{ fontSize: 9 }}
                          title="A regression is a metric that got WORSE compared to the baseline run."
                        >
                          regress
                        </Pill>
                      )}
                      <span
                        style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2, width: 26, textAlign: 'right' }}
                      >
                        {c.count}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>

        {/* SUB-METRICS + CONFUSION */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 14 }}>
          <Card style={{ padding: 18 }}>
            <Eyebrow>
              Sub-metric breakdown{compareActive ? ' - this vs. other' : ''}
            </Eyebrow>
            {detail.sub_metrics.length === 0 ? (
              <div style={{ marginTop: 14, fontSize: 12.5, color: E.text3 }}>No sub-metrics.</div>
            ) : (
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
                {detail.sub_metrics.map((m) => {
                  const otherMetric =
                    compareActive && compareDetail
                      ? compareDetail.sub_metrics.find((om) => om.label === m.label) ?? null
                      : null;
                  const baseline = m.baseline ?? 0;
                  const baselineRef = otherMetric ? otherMetric.value : m.baseline;
                  const better = baselineRef == null
                    ? E.steel
                    : m.inverse
                      ? m.value < baselineRef ? E.pass : E.fail
                      : m.value > baselineRef ? E.pass : E.steel;
                  return (
                    <div key={m.label}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'baseline',
                          marginBottom: 5,
                          fontSize: 12,
                        }}
                      >
                        <span style={{ color: E.text1 }}>
                          {m.label}
                          {m.inverse ? ' (lower=better)' : ''}
                        </span>
                        <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', fontFamily: E.fMono }}>
                          {otherMetric ? (
                            <>
                              <span style={{ color: E.ember, fontSize: 12 }}>{m.value}%</span>
                              <span style={{ color: E.text4, fontSize: 11 }}>|</span>
                              <span style={{ color: E.steel, fontSize: 12 }}>{otherMetric.value}%</span>
                            </>
                          ) : (
                            <>
                              {m.baseline != null && (
                                <span style={{ color: E.steel, fontSize: 11 }}>base {m.baseline}%</span>
                              )}
                              <span style={{ color: E.text0, fontSize: 13 }}>{m.value}%</span>
                            </>
                          )}
                        </div>
                      </div>
                      {otherMetric ? (
                        // Compare mode: two stacked bars (this ember, other steel).
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                          <div style={{ position: 'relative', height: 6, background: E.panel2, borderRadius: 3 }}>
                            <div
                              style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                bottom: 0,
                                width: `${m.value}%`,
                                background: better,
                                borderRadius: 3,
                              }}
                            />
                          </div>
                          <div style={{ position: 'relative', height: 6, background: E.panel2, borderRadius: 3 }}>
                            <div
                              style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                bottom: 0,
                                width: `${otherMetric.value}%`,
                                background: E.steel,
                                opacity: 0.7,
                                borderRadius: 3,
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <div style={{ position: 'relative', height: 6, background: E.panel2, borderRadius: 3 }}>
                          {m.baseline != null && (
                            <div
                              style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                bottom: 0,
                                width: `${baseline}%`,
                                background: E.steel,
                                opacity: 0.4,
                                borderRadius: 3,
                              }}
                            />
                          )}
                          <div
                            style={{
                              position: 'absolute',
                              left: 0,
                              top: 0,
                              bottom: 0,
                              width: `${m.value}%`,
                              background: better,
                              borderRadius: 3,
                            }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <Eyebrow>Item-level comparison vs. baseline</Eyebrow>
              <span style={{ flex: 1 }} />
              <Pill mono style={{ background: E.panel2, color: E.text2, fontSize: 10 }}>
                {detail.dataset.n} items
              </Pill>
            </div>
            {detail.confusion ? (
              <>
                <div
                  style={{
                    marginTop: 14,
                    display: 'grid',
                    gridTemplateColumns: '70px 1fr 1fr',
                    gridTemplateRows: '24px 1fr 1fr',
                    gap: 4,
                    fontSize: 11,
                    fontFamily: E.fMono,
                  }}
                >
                  <div></div>
                  <div
                    style={{
                      color: E.text3,
                      textAlign: 'center',
                      display: 'flex',
                      alignItems: 'flex-end',
                      justifyContent: 'center',
                    }}
                  >
                    new PASS
                  </div>
                  <div
                    style={{
                      color: E.text3,
                      textAlign: 'center',
                      display: 'flex',
                      alignItems: 'flex-end',
                      justifyContent: 'center',
                    }}
                  >
                    new FAIL
                  </div>

                  <div
                    style={{
                      color: E.text3,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      paddingRight: 8,
                    }}
                  >
                    base PASS
                  </div>
                  <div
                    style={{
                      background: E.passDim,
                      border: `1px solid ${E.pass}33`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>
                      {detail.confusion.base_pass_v_pass}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>kept passing</div>
                  </div>
                  <div
                    style={{
                      background: E.failDim,
                      border: `1px solid ${E.fail}33`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.fail }}>
                      {detail.confusion.base_pass_v_fail}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>regressed</div>
                  </div>

                  <div
                    style={{
                      color: E.text3,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      paddingRight: 8,
                    }}
                  >
                    base FAIL
                  </div>
                  <div
                    style={{
                      background: E.emberDim,
                      border: `1px solid ${E.emberRim}`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.ember }}>
                      {detail.confusion.base_fail_v_pass}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>fixed</div>
                  </div>
                  <div
                    style={{
                      background: E.panel2,
                      border: `1px solid ${E.hair}`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text2 }}>
                      {detail.confusion.base_fail_v_fail}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>still failing</div>
                  </div>
                </div>
                <div
                  style={{
                    marginTop: 14,
                    padding: 10,
                    background: E.panel2,
                    borderRadius: 6,
                    fontSize: 11.5,
                    color: E.text2,
                    lineHeight: 1.5,
                  }}
                >
                  {/* net_delta is signed; show explicit + prefix when positive */}
                  <b style={{ color: E.text0 }}>
                    Net {detail.confusion.net_delta >= 0 ? '+' : ''}
                    {detail.confusion.net_delta} items
                  </b>{' '}
                  moved to passing.
                  {detail.confusion.base_pass_v_fail > 0
                    ? ` The ${detail.confusion.base_pass_v_fail} regressions are the ones to investigate before shipping.`
                    : ''}
                </div>
              </>
            ) : (
              <div style={{ marginTop: 14, fontSize: 12.5, color: E.text3 }}>
                No baseline to compare against.
              </div>
            )}
          </Card>
        </div>

        {/* FAILED ITEMS PREVIEW */}
        {detail.failed_items_preview.length > 0 && (
          <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '12px 18px',
                borderBottom: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Failed items</span>
              <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 10 }}>
                {detail.failure_clusters.total_failures}
              </Pill>
              <span style={{ flex: 1 }} />
              <Btn kind="ghost" size="sm" disabled title="Coming soon - filter failed items by cluster, severity, or rubric">
                Filter ▾
              </Btn>
              <Btn kind="ghost" size="sm" disabled title="Coming soon - regroup the failed-items list by cluster, rubric, or input pattern">
                Group: cluster ▾
              </Btn>
            </div>
            {detail.failed_items_preview.map((s, i) => (
              <div
                key={s.id}
                style={{
                  padding: '14px 18px',
                  borderTop: i ? `1px solid ${E.hair}` : 'none',
                  display: 'grid',
                  gridTemplateColumns: '70px 1fr 90px 60px',
                  gap: 14,
                  alignItems: 'flex-start',
                }}
              >
                <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3, paddingTop: 2 }}>{s.id}</div>
                <div style={{ fontSize: 12.5, lineHeight: 1.55 }}>
                  <div style={{ color: E.text1, marginBottom: 4 }}>
                    <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>USER</span>
                    {s.user}
                  </div>
                  <div style={{ color: E.text3, fontSize: 11.5, marginBottom: 4 }}>
                    <span style={{ fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>EXPECT</span>
                    {s.expected}
                  </div>
                  <div style={{ color: E.text1, fontSize: 12 }}>
                    <span style={{ color: E.fail, fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>GOT</span>
                    <span style={{ background: E.failDim, padding: '1px 5px', borderRadius: 3 }}>{s.got}</span>
                  </div>
                </div>
                <div>
                  <Pill mono color={E.fail} bg={E.failDim}>
                    {s.cluster}
                  </Pill>
                </div>
                <div style={{ textAlign: 'right', fontFamily: E.fSerif, fontSize: 16, color: E.fail }}>{s.score}</div>
              </div>
            ))}
            <div
              style={{
                padding: '11px 18px',
                borderTop: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Btn
                kind="bare"
                size="sm"
                disabled
                title="Coming soon - a flat all-failures view; for now drill into a specific cluster from the donut above"
              >
                View all {detail.failure_clusters.total_failures} failures →
              </Btn>
            </div>
          </Card>
        )}

        <div style={{ height: 30 }} />
      </div>}
    </AppShell>
  );
}

// ---------- Items tab ----------

const PAGE_SIZE = 50;
const FILTERS: { key: ExperimentItemsFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'passed', label: 'Passed' },
  { key: 'failed', label: 'Failed' },
];

interface ItemsTabProps {
  runId: string;
  compareActive?: boolean;
  onClearCompare?: () => void;
}

function ItemsTab({ runId, compareActive, onClearCompare }: ItemsTabProps) {
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<ExperimentItemsFilter>('all');
  const [sort, setSort] = useState<ExperimentItemsSort>('item_id');
  const [search, setSearch] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetcher = useCallback(
    () => v2.experimentItems(runId, { offset, limit: PAGE_SIZE, filter, sort }),
    [runId, offset, filter, sort],
  );
  const { data, err, reloading, isInitialLoad } = useV2Resource<ExperimentItemsResponse>(
    `experimentItems:${runId}:${offset}:${filter}:${sort}`,
    fetcher,
  );

  if (err && !data) {
    return (
      <div style={{ padding: '20px 36px' }}>
        <Card style={{ padding: 16, borderColor: E.fail }}>
          <Eyebrow style={{ color: E.fail }}>Failed to load items</Eyebrow>
          <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
        </Card>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: '20px 36px' }}>
        <Skeleton w={320} h={28} />
        <div style={{ marginTop: 14 }}>
          <Skeleton w="100%" h={300} style={{ borderRadius: 6 }} />
        </div>
      </div>
    );
  }

  const visibleItems = search.trim()
    ? data.items.filter((it) => it.id.toLowerCase().includes(search.trim().toLowerCase()))
    : data.items;

  const showingFrom = data.total === 0 ? 0 : offset + 1;
  const showingTo = Math.min(offset + data.items.length, data.total);
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < data.total;

  function changeFilter(next: ExperimentItemsFilter) {
    setFilter(next);
    setOffset(0);
    setExpandedId(null);
  }

  function changeSort(next: ExperimentItemsSort) {
    setSort(next);
    setOffset(0);
    setExpandedId(null);
  }

  const gridCols = `120px 1fr ${data.metric_ids.map(() => '64px').join(' ')} 64px`;

  return (
    <div style={{ padding: '20px 36px' }}>
      {compareActive && (
        <Card
          style={{
            padding: 10,
            marginBottom: 14,
            background: E.steelDim,
            borderColor: E.steel,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
            <span style={{ color: E.steel, fontFamily: E.fMono, fontSize: 10, letterSpacing: '0.08em' }}>
              COMPARE MODE ACTIVE
            </span>
            <span style={{ color: E.text2 }}>
              Per-item compare grid is not yet wired here - showing this run's items only.
            </span>
            <span style={{ flex: 1 }} />
            {onClearCompare && (
              <Btn kind="bare" size="sm" onClick={onClearCompare}>
                Clear compare
              </Btn>
            )}
          </div>
        </Card>
      )}
      {/* TOOLBAR */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {FILTERS.map((f) => {
            const active = filter === f.key;
            return (
              <button
                key={f.key}
                type="button"
                onClick={() => changeFilter(f.key)}
                style={{
                  padding: '5px 12px',
                  fontSize: 12,
                  borderRadius: 999,
                  border: `1px solid ${active ? E.ember : E.hair}`,
                  background: active ? E.emberDim : E.panel,
                  color: active ? E.ember : E.text2,
                  cursor: 'pointer',
                  fontFamily: E.fSans,
                }}
              >
                {f.label}
              </button>
            );
          })}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              fontSize: 11,
              color: E.text3,
              fontFamily: E.fMono,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            Sort
          </span>
          <select
            value={sort}
            onChange={(e) => changeSort(e.target.value as ExperimentItemsSort)}
            style={{
              fontSize: 12,
              padding: '4px 8px',
              borderRadius: 6,
              border: `1px solid ${E.hair}`,
              background: E.panel,
              color: E.text1,
              fontFamily: E.fSans,
            }}
          >
            <option value="item_id">Item ID</option>
            <option value="score">Score (worst first)</option>
          </select>
        </div>

        <input
          type="text"
          placeholder="Search item id..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            flex: 1,
            minWidth: 180,
            maxWidth: 320,
            padding: '6px 10px',
            fontSize: 12,
            fontFamily: E.fMono,
            borderRadius: 6,
            border: `1px solid ${E.hair}`,
            background: E.panel,
            color: E.text1,
          }}
        />

        <span style={{ flex: 1 }} />

        <span style={{ fontSize: 11.5, color: E.text2, fontFamily: E.fMono }}>
          Showing {showingFrom}-{showingTo} of {data.total}
        </span>
        <UpdatingChip visible={reloading && !isInitialLoad} />
      </div>

      {data.total === 0 ? (
        <Card style={{ padding: 24 }}>
          <div style={{ fontSize: 13, color: E.text2 }}>No items match this filter.</div>
        </Card>
      ) : (
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          {/* HEADER ROW */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: gridCols,
              gap: 10,
              padding: '10px 16px',
              borderBottom: `1px solid ${E.hair}`,
              fontSize: 10,
              fontFamily: E.fMono,
              color: E.text3,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              alignItems: 'center',
            }}
          >
            <div>ID</div>
            <div>Input</div>
            {data.metric_ids.map((mid) => (
              <div
                key={mid}
                title={mid}
                style={{
                  textAlign: 'center',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {mid.length > 6 ? `${mid.slice(0, 6)}...` : mid}
              </div>
            ))}
            <div style={{ textAlign: 'right' }}>Score</div>
          </div>

          {visibleItems.length === 0 ? (
            <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>
              No items match the search.
            </div>
          ) : (
            visibleItems.map((it) => {
              const expanded = expandedId === it.id;
              const totalScore = it.per_metric.reduce(
                (sum, pm) => sum + (pm.score ?? 0),
                0,
              );
              const denom = it.per_metric.length || 1;
              const fillPct = Math.max(0, Math.min(1, totalScore / denom));
              return (
                <div key={it.id}>
                  <button
                    type="button"
                    onClick={() => setExpandedId(expanded ? null : it.id)}
                    style={{
                      display: 'grid',
                      width: '100%',
                      gridTemplateColumns: gridCols,
                      gap: 10,
                      padding: '10px 16px',
                      borderTop: `1px solid ${E.hair}`,
                      alignItems: 'center',
                      background: expanded ? E.panel2 : 'transparent',
                      cursor: 'pointer',
                      textAlign: 'left',
                      fontFamily: E.fSans,
                    }}
                  >
                    <div
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 11,
                        color: E.text2,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={it.id}
                    >
                      {it.id.slice(0, 12)}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        color: E.text1,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={it.input_preview}
                    >
                      {it.input_preview || <span style={{ color: E.text4 }}>-</span>}
                    </div>
                    {it.per_metric.map((pm) => {
                      let dotColor: string;
                      if (pm.passed === true) dotColor = E.pass;
                      else if (pm.passed === false) dotColor = E.fail;
                      else dotColor = E.text4;
                      const titleText =
                        pm.score != null
                          ? `${pm.metric_id}: ${pm.score.toFixed(2)}${pm.passed === false ? ' (failed)' : pm.passed === true ? ' (passed)' : ''}`
                          : `${pm.metric_id}: no result`;
                      return (
                        <div
                          key={pm.metric_id}
                          style={{ display: 'flex', justifyContent: 'center' }}
                          title={titleText}
                        >
                          <span
                            style={{
                              width: 10,
                              height: 10,
                              borderRadius: '50%',
                              background: dotColor,
                              display: 'inline-block',
                            }}
                          />
                        </div>
                      );
                    })}
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'flex-end',
                        alignItems: 'center',
                      }}
                    >
                      <span
                        style={{
                          width: 36,
                          height: 4,
                          borderRadius: 2,
                          background: E.panel3,
                          position: 'relative',
                          overflow: 'hidden',
                        }}
                      >
                        <span
                          style={{
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: `${fillPct * 100}%`,
                            background: it.any_failed ? E.fail : E.pass,
                          }}
                        />
                      </span>
                    </div>
                  </button>

                  {expanded && (
                    <div
                      style={{
                        padding: '14px 18px 18px',
                        borderTop: `1px solid ${E.hair}`,
                        background: E.panel2,
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr 1fr',
                        gap: 14,
                      }}
                    >
                      <ExpandPanel label="Input" body={it.input_preview} accent={E.text2} />
                      <ExpandPanel
                        label="Expected"
                        body={it.expected_preview}
                        accent={E.steel}
                      />
                      <ExpandPanel
                        label="Output"
                        body={it.output_preview}
                        accent={it.any_failed ? E.fail : E.pass}
                      />
                    </div>
                  )}
                </div>
              );
            })
          )}
        </Card>
      )}

      {/* PAGINATION */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 14 }}>
        <Btn
          kind="secondary"
          size="sm"
          onClick={() => {
            setOffset(Math.max(0, offset - PAGE_SIZE));
            setExpandedId(null);
          }}
          disabled={!canPrev}
        >
          ← Prev
        </Btn>
        <Btn
          kind="secondary"
          size="sm"
          onClick={() => {
            setOffset(offset + PAGE_SIZE);
            setExpandedId(null);
          }}
          disabled={!canNext}
        >
          Next →
        </Btn>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
          page {Math.floor(offset / PAGE_SIZE) + 1} of{' '}
          {Math.max(1, Math.ceil(data.total / PAGE_SIZE))}
        </span>
      </div>

      <div style={{ height: 30 }} />
    </div>
  );
}

interface ExpandPanelProps {
  label: string;
  body: string | null;
  accent: string;
}

function ExpandPanel({ label, body, accent }: ExpandPanelProps) {
  return (
    <div>
      <Eyebrow style={{ color: accent }}>{label}</Eyebrow>
      <div
        style={{
          marginTop: 6,
          padding: 10,
          background: E.panel,
          border: `1px solid ${E.hair}`,
          borderRadius: 6,
          fontFamily: E.fMono,
          fontSize: 11.5,
          color: E.text1,
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          minHeight: 40,
        }}
      >
        {body && body.length > 0 ? body : <span style={{ color: E.text4 }}>-</span>}
      </div>
    </div>
  );
}
