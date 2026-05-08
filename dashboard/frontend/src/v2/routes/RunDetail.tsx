/**
 * RunDetail - the deepest read view: headline stats, pass-rate vs baseline,
 * failure clusters, sub-metric breakdown, confusion matrix, failed item preview.
 *
 * Compare overlay: when ?compare=<otherId> is in the URL, a second
 * ExperimentDetail is fetched in parallel and several sections render
 * side-by-side (headline, pass timeline legend, sub-metrics, failures donut).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
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
import { errorMessage } from '../api/errors';
import { listCli } from '../api/cli';
import type { CliSchema } from '../api/cli';
import type {
  ExperimentDetail,
  ExperimentItemRow,
  ExperimentItemsFilter,
  ExperimentItemsResponse,
  ExperimentItemsSort,
} from '../api/types';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { useFlashState } from '../hooks/useFlashState';
import { useProject } from '../hooks/useProject';
import { useSearchFilter } from '../hooks/useSearchFilter';
import { preloadFailureCluster } from '../routePreloads';
import { openCliRunner } from '../cliRunnerBridge';
import { copyToClipboard } from '../clipboard';
import { E } from '../tokens';
import { linkifyText, makeUrlCounter } from '../textRender';

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
  const [searchParams, setSearchParams] = useSearchParams();
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
    refetch,
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

  // Prefetch the items list once detail lands so a click on the
  // Items tab hits a warm cache. The Items tab does its own fetch on
  // mount; without this, the user sees a loading spinner on the FIRST
  // tab click. Established pattern from Datasets/ExperimentsList/
  // Home/AppShell.
  //
  // Cache key + fetcher must match what the tab itself will use. The
  // limits differ between the regular ItemsTab (PAGE_SIZE=50 per page)
  // and ItemsCompareTab (ITEMS_COMPARE_LIMIT=200 per side); we
  // dispatch on compareActive so the prefetch matches the user's path.
  // Both tabs use the same cache key shape ``:0:all:item_id`` for the
  // first page so this is a single warmup either way.
  useEffect(() => {
    if (!detail || !runId) return;
    if (compareActive && compareWith) {
      prefetchV2(`experimentItems:${runId}:0:all:item_id`, () =>
        v2.experimentItems(runId, {
          offset: 0,
          limit: ITEMS_COMPARE_LIMIT,
          filter: 'all',
          sort: 'item_id',
        }),
      );
      prefetchV2(`experimentItems:${compareWith}:0:all:item_id`, () =>
        v2.experimentItems(compareWith, {
          offset: 0,
          limit: ITEMS_COMPARE_LIMIT,
          filter: 'all',
          sort: 'item_id',
        }),
      );
    } else {
      prefetchV2(`experimentItems:${runId}:0:all:item_id`, () =>
        v2.experimentItems(runId, {
          offset: 0,
          limit: PAGE_SIZE,
          filter: 'all',
          sort: 'item_id',
        }),
      );
    }
  }, [detail, compareActive, compareWith, runId]);

  // activeTab lives in the URL (?tab=items) so a shared run link
  // lands the recipient on the same tab the sender was looking at,
  // and reload / browser-back keep their place. Summary (0) is the
  // default and stays unparameterised so /experiments/<id> remains
  // a clean default URL. Failures (2) deep-links into a cluster on
  // click, so it never becomes the activeTab; only Summary <-> Items
  // is a real URL-persisted toggle.
  const tabParam = searchParams.get('tab');
  const activeTab = tabParam === 'items' ? 1 : 0;
  const setActiveTab = useCallback(
    (idx: number) => {
      setSearchParams(
        (prev) => {
          const u = new URLSearchParams(prev);
          if (idx === 1) u.set('tab', 'items');
          else u.delete('tab');
          return u;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
  // When the user clicks "View all N failures" from the Summary tab we
  // both switch to the Items tab and seed its filter to "failed". The
  // Items tab manages its own filter state internally so we re-mount it
  // (via a per-jump nonce) to apply the seed each time the user clicks
  // through. Without the nonce, a user who toggled to "passed" inside
  // Items and then re-clicked "View failures" would stay on "passed".
  const [itemsSeed, setItemsSeed] = useState<{
    filter: ExperimentItemsFilter;
    nonce: number;
  } | null>(null);
  const [rerunBusy, setRerunBusy] = useState(false);

  function jumpToFailedItems() {
    setItemsSeed({ filter: 'failed', nonce: Date.now() });
    setActiveTab(1);
  }
  // Inline status for header actions - replaces window.alert dialogs that
  // jarred against the v2 design. shareState toggles the "Share" button
  // label, rerunErr surfaces failures from the run-eval form-open path.
  const [shareState, flashShareState] = useFlashState<'idle' | 'copied' | 'error'>('idle');
  // Independent state for the inline-id copy affordance below the
  // status-dot row. Tracking separately from `shareState` so the
  // Share button (URL) and the id click (id only) report their own
  // success/failure without one button's click affecting the other's
  // visual feedback.
  const [idCopyState, flashIdCopyState] = useFlashState<'idle' | 'copied' | 'error'>('idle');
  const [rerunErr, setRerunErr] = useState<string | null>(null);

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

  // Compare lives as a flag on the Items tab (header opens compare via
  // ?compare=...), not as a standalone tab. Trace has no implementation.
  // Keeping disabled stubs in the tab strip just confused users into
  // hovering them looking for a hidden surface, so they are dropped.
  const tabs = [
    'Summary',
    `Items - ${detail.dataset.n}`,
    `Failures - ${detail.failure_clusters.total_failures}`,
  ];

  async function handleRerun() {
    if (!detail || rerunBusy) return;
    setRerunBusy(true);
    setRerunErr(null);
    try {
      // Pull the catalog (cached after first hit) and look up `run-eval`.
      // We can't reconstruct the original argv from disk - results.json
      // doesn't carry the launching command - so we seed dataset + any
      // model/rubric metadata we happen to know. The user edits in the
      // form before clicking Run.
      const cmds: CliSchema[] = await listCli();
      const runEval = cmds.find((c) => c.id === 'run-eval');
      if (!runEval) {
        setRerunErr(
          'The `run-eval` command is not in this build of the CLI catalog.',
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
      setRerunErr(`Failed to open re-run form: ${errorMessage(e)}`);
    } finally {
      setRerunBusy(false);
    }
  }

  async function handleShare() {
    // The deep-link URL for a run is its current pathname - already shareable
    // since the dashboard routes are stable. Copy to clipboard so the user
    // can paste straight into Slack/email/etc.
    try {
      await copyToClipboard(window.location.href);
      flashShareState('copied', 2000);
    } catch {
      flashShareState('error', 3000);
    }
  }

  async function handleCopyId() {
    // Copy just the run id - useful when pasting into a CLI like
    // `evalyn run-detail <id>` or referencing in chat. The Share
    // button copies the full URL; this is for the id-only case
    // operators routinely need.
    if (!detail) return;
    try {
      await copyToClipboard(detail.id);
      flashIdCopyState('copied', 2000);
    } catch {
      flashIdCopyState('error', 3000);
    }
  }

  const headerExtra = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      {rerunErr && (
        <span
          role="alert"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 11,
            color: E.fail,
            fontFamily: E.fMono,
            maxWidth: 320,
          }}
          title={rerunErr}
        >
          <span
            style={{
              maxWidth: 280,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {rerunErr}
          </span>
          <button
            type="button"
            onClick={() => setRerunErr(null)}
            aria-label="Dismiss error"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'currentColor',
              cursor: 'pointer',
              fontSize: 13,
              lineHeight: 1,
              padding: '0 2px',
              opacity: 0.7,
            }}
          >
            ×
          </button>
        </span>
      )}
      <Btn
        kind="ghost"
        size="sm"
        onClick={() => void handleShare()}
        title={
          shareState === 'copied'
            ? 'URL copied to clipboard'
            : shareState === 'error'
              ? 'Browser blocked clipboard access'
              : 'Copy this run\'s URL to your clipboard'
        }
        aria-label={
          shareState === 'copied'
            ? 'Run URL copied to clipboard'
            : shareState === 'error'
              ? 'Failed to copy run URL'
              : "Copy this run's URL to clipboard"
        }
      >
        {shareState === 'copied' ? '✓ Copied' : shareState === 'error' ? '✗ Failed' : '↗ Share'}
      </Btn>
      <Btn
        kind="primary"
        size="sm"
        onClick={handleRerun}
        onMouseEnter={() => prefetchV2('commands', listCli)}
        onFocus={() => prefetchV2('commands', listCli)}
        disabled={rerunBusy}
        title={
          rerunBusy
            ? 'Loading...'
            : `Open the run-eval form pre-filled for ${detail.dataset.name}`
        }
        aria-busy={rerunBusy}
        aria-label={
          rerunBusy
            ? 'Loading re-run form'
            : `Re-run with the same dataset (${detail.dataset.name})`
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

  // Memoize derived series/segments/clusters so unrelated parent state
  // changes (tab clicks, Share button label, rerun status) don't
  // recompute work that only depends on the loaded run details.
  const passSeries = useMemo<LineSeries[]>(() => {
    const series: LineSeries[] = detail.pass_timeline.series.map((s) => ({
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
        series.push({
          color: E.steel,
          width: 1.75,
          dashed: true,
          data: otherPrimary.data,
        });
      }
    }
    return series;
  }, [detail, compareActive, compareDetail]);

  const donutSegments = useMemo(
    () =>
      detail.failure_clusters.clusters.map((c) => ({
        value: c.count,
        color: CLUSTER_COLOR[c.color_kind] ?? E.text3,
        label: c.label,
      })),
    [detail],
  );

  const compareDonutSegments = useMemo(
    () =>
      compareDetail
        ? compareDetail.failure_clusters.clusters.map((c) => ({
            value: c.count,
            color: CLUSTER_COLOR[c.color_kind] ?? E.text3,
            label: c.label,
          }))
        : [],
    [compareDetail],
  );

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
  const compareClusters = useMemo<CompareCluster[]>(() => {
    if (!compareDetail) return [];
    const thisMap = new Map<string, number>();
    const otherMap = new Map<string, number>();
    for (const c of detail.failure_clusters.clusters) thisMap.set(c.label, c.count);
    for (const c of compareDetail.failure_clusters.clusters) otherMap.set(c.label, c.count);
    const keys = new Set<string>([...thisMap.keys(), ...otherMap.keys()]);
    return Array.from(keys)
      .map((label) => {
        const thisCount = thisMap.get(label) ?? 0;
        const otherCount = otherMap.get(label) ?? 0;
        let color: string;
        if (thisCount > 0 && otherCount === 0) color = E.ember;
        else if (otherCount > 0 && thisCount === 0) color = E.pass;
        else color = E.text3;
        return { label, thisCount, otherCount, color };
      })
      .sort((a, b) => b.thisCount + b.otherCount - (a.thisCount + a.otherCount));
  }, [detail, compareDetail]);

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
                <button
                  type="button"
                  onClick={() => void handleCopyId()}
                  // Stable aria-label - the previous version flipped
                  // to "Run id copied to clipboard" on success, which
                  // caused "name changed mid-focus" announcements on
                  // some SR engines. Now the success cue lives in a
                  // sibling .eSr live region (below) and the button
                  // name stays consistent across press cycles.
                  aria-label={`Copy run id ${detail.id} to clipboard`}
                  title={
                    idCopyState === 'copied'
                      ? 'Copied to clipboard'
                      : idCopyState === 'error'
                        ? 'Browser blocked clipboard access'
                        : 'Click to copy just the run id (without the URL)'
                  }
                  style={{
                    fontFamily: 'inherit',
                    fontSize: 'inherit',
                    color:
                      idCopyState === 'copied'
                        ? E.pass
                        : idCopyState === 'error'
                          ? E.fail
                          : 'inherit',
                    background: 'transparent',
                    border: 'none',
                    padding: 0,
                    cursor: 'pointer',
                    // Subtle dotted underline so it reads as
                    // interactive without shouting. Mirrors the
                    // pattern on the SystemStatusCard "Failures
                    // (24h)" click-through row.
                    textDecoration: 'underline',
                    textDecorationStyle: 'dotted',
                    textUnderlineOffset: 2,
                  }}
                >
                  {idCopyState === 'copied' ? (
                    <>
                      <span aria-hidden="true">✓ </span>{detail.id}
                    </>
                  ) : (
                    detail.id
                  )}
                </button>
                {idCopyState === 'copied' && (
                  <span role="status" aria-live="polite" className="eSr">
                    Run id copied to clipboard
                  </span>
                )}
                {idCopyState === 'error' && (
                  <span role="alert" className="eSr">
                    Copy failed
                  </span>
                )}
                {' - '}
                {detail.status} {detail.finished_at_iso} - {detail.duration} - {detail.cost}
              </span>
              <UpdatingChip
                visible={reloading && !isInitialLoad}
                error={detail ? err : null}
                onRetry={refetch}
              />
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

        {/* TABS - proper ARIA tablist with arrow-key navigation, mirroring
            the Reports AudienceTabs pattern. Failures is borderline since
            it can deep-link to a cluster, but visually it lives in the
            same strip; the click handler (and Enter activation) keeps the
            existing nav behaviour intact. */}
        <div
          role="tablist"
          aria-label="Run sections"
          onKeyDown={(e) => {
            // Find next non-disabled tab in the requested direction.
            const moveBy = (delta: 1 | -1): number | null => {
              const n = tabs.length;
              for (let step = 1; step <= n; step += 1) {
                const i = (activeTab + delta * step + n) % n;
                const isFailuresI = i === 2;
                const disabledI = isFailuresI && !detail.failure_clusters.clusters[0];
                if (!disabledI) return i;
              }
              return null;
            };
            let nextIdx: number | null = null;
            if (e.key === 'ArrowRight') nextIdx = moveBy(1);
            else if (e.key === 'ArrowLeft') nextIdx = moveBy(-1);
            else if (e.key === 'Home') nextIdx = 0;
            else if (e.key === 'End') nextIdx = tabs.length - 1;
            if (nextIdx === null) return;
            e.preventDefault();
            setActiveTab(nextIdx);
            // Defer focus so React commits the new selected state first.
            window.setTimeout(() => {
              const next = document.querySelector<HTMLButtonElement>(
                `[data-rundetail-tab="${nextIdx}"]`,
              );
              next?.focus();
            }, 0);
          }}
          style={{ display: 'flex', gap: 2, marginTop: 22, borderBottom: `1px solid ${E.hair}` }}
        >
          {tabs.map((t, i) => {
            const isActive = i === activeTab;
            // Summary (0) and Items (1) render in-place. Failures (2)
            // deep-links into the first cluster if one exists; if no
            // failures, the tab is honestly disabled.
            const firstCluster = detail.failure_clusters.clusters[0];
            const isFailures = i === 2;
            const disabled = isFailures && !firstCluster;
            const title = disabled
              ? 'No failures in this run'
              : isFailures
                ? `Open the first failure cluster (${firstCluster?.label ?? ''})`
                : undefined;
            // Hover/focus warmup for the Failures tab: prefetches
            // the FailureCluster chunk + the first cluster's
            // detail JSON so the cross-route navigation feels
            // instant. No-op when disabled (no cluster) or when
            // i !== 2 (Summary/Items render in-place; their data
            // is already prefetched by the page-level effect).
            const warmFailuresTab = isFailures && firstCluster
              ? () => {
                  void preloadFailureCluster();
                  prefetchV2(
                    `cluster:${detail.id}:${firstCluster.id}`,
                    () => v2.cluster(detail.id, firstCluster.id),
                  );
                }
              : undefined;
            return (
              <button
                key={t}
                id={`rundetail-tab-${i}`}
                type="button"
                role="tab"
                aria-selected={isActive}
                aria-controls={i < 2 ? `rundetail-panel-${i}` : undefined}
                tabIndex={isActive ? 0 : -1}
                data-rundetail-tab={i}
                disabled={disabled}
                title={title}
                onMouseEnter={warmFailuresTab}
                onFocus={warmFailuresTab}
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
        <div id="rundetail-panel-1" role="tabpanel" aria-labelledby="rundetail-tab-1">
          {compareWith ? (
            <ItemsCompareTab
              runId={detail.id}
              otherId={compareWith}
              onClearCompare={clearCompare}
              onSwapCompare={swapCompare}
            />
          ) : (
            <ItemsTab
              // Remount on each cross-tab "View failures" click so the
              // seeded filter applies even if the user already switched
              // it inside ItemsTab on a prior visit. The nonce is what
              // makes successive jumps work; otherwise React would keep
              // the same instance and ignore the new initialFilter.
              key={`items-${itemsSeed?.nonce ?? 0}`}
              runId={detail.id}
              initialFilter={itemsSeed?.filter}
            />
          )}
        </div>
      )}

      {activeTab === 0 && <div id="rundetail-panel-0" role="tabpanel" aria-labelledby="rundetail-tab-0" style={{ padding: '20px 36px' }}>
        {/* COMPARE BADGE BAR */}
        {compareWith && compareErr && !compareDetail && (
          <Card style={{ padding: 12, marginBottom: 14, borderColor: E.fail }}>
            <div role="alert" style={{ fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
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
              <StatusDot
                status={compareDetail.status}
                label={`Compare run status: ${compareDetail.status}`}
              />
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
                title={`Cumulative pass rate as items were graded (ship gate ${detail.pass_timeline.ship_gate}%, ending at ${passSeries[0].data.at(-1)?.toFixed(1) ?? '?'}%).`}
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
                          title={c.label}
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
                  {detail.failure_clusters.clusters.map((c) => {
                    // Warm the FailureCluster chunk + this cluster's
                    // payload on hover/focus. Same pattern as the
                    // tab-strip's warmFailuresTab at line 743 - the
                    // tab warms the FIRST cluster, this list warms
                    // each one as the user hovers over it. prefetchV2
                    // dedupes by cache key so re-warming a cluster is
                    // free, and clicking a pre-warmed row paints the
                    // cluster page below the perception threshold.
                    const warm = () => {
                      void preloadFailureCluster();
                      prefetchV2(
                        `cluster:${detail.id}:${c.id}`,
                        () => v2.cluster(detail.id, c.id),
                      );
                    };
                    return (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() =>
                        navigate(`/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(c.id)}`)
                      }
                      onMouseEnter={warm}
                      onFocus={warm}
                      aria-label={`Open cluster ${c.label}: ${c.count} failure${c.count === 1 ? '' : 's'}${c.regression ? ', regressed from baseline' : ''}`}
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
                        title={c.label}
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
                    );
                  })}
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
              <Btn
                kind="ghost"
                size="sm"
                onClick={jumpToFailedItems}
                title="Open the Items tab pre-filtered to failed items - filter and sort options live there"
              >
                Filter &amp; sort <span aria-hidden="true">→</span>
              </Btn>
            </div>
            <div role="list" aria-label={`Top ${detail.failed_items_preview.length} failed items in this run`}>
            {detail.failed_items_preview.map((s, i) => (
              <div
                key={s.id}
                role="listitem"
                aria-label={`Item ${s.id}, cluster ${s.cluster}, score ${s.score}: user said "${s.user}", expected "${s.expected}", got "${s.got}"`}
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
            </div>
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
                onClick={jumpToFailedItems}
                title="Open the Items tab filtered to failed items"
              >
                View all {detail.failure_clusters.total_failures} failures <span aria-hidden="true">→</span>
              </Btn>
            </div>
          </Card>
        )}

        <div style={{ height: 30 }} />
      </div>}
    </AppShell>
  );
}

// ---------- Items compare tab ----------

/**
 * Per-item side-by-side grid for compare mode. Aligns rows by item_id
 * (same dataset → same item_ids), then for each metric computes a
 * "fixed" / "regressed" / "unchanged" verdict against the other run.
 *
 * Fetches up to ITEMS_COMPARE_LIMIT items per side. If either side has
 * more, a footer notes the cap; users can drill via filter pills.
 */

const ITEMS_COMPARE_LIMIT = 200;

type CompareFilter = 'all' | 'regressed' | 'fixed' | 'unchanged';

const COMPARE_FILTERS: { key: CompareFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'regressed', label: 'Regressed' },
  { key: 'fixed', label: 'Fixed' },
  { key: 'unchanged', label: 'Unchanged' },
];

interface CompareDiff {
  fixed: number;
  regressed: number;
  unchanged: number;
}

interface CompareRow {
  id: string;
  thisRow: ExperimentItemRow;
  otherRow: ExperimentItemRow | null;
  diff: CompareDiff;
}

function computeDiff(
  thisRow: ExperimentItemRow,
  otherRow: ExperimentItemRow | null,
): CompareDiff {
  if (!otherRow) return { fixed: 0, regressed: 0, unchanged: 0 };
  const otherByMetric = new Map(
    otherRow.per_metric.map((pm) => [pm.metric_id, pm]),
  );
  let fixed = 0;
  let regressed = 0;
  let unchanged = 0;
  for (const tpm of thisRow.per_metric) {
    const opm = otherByMetric.get(tpm.metric_id);
    if (!opm) {
      unchanged += 1;
      continue;
    }
    if (opm.passed === false && tpm.passed === true) fixed += 1;
    else if (opm.passed === true && tpm.passed === false) regressed += 1;
    else unchanged += 1;
  }
  return { fixed, regressed, unchanged };
}

interface ItemsCompareTabProps {
  runId: string;
  otherId: string;
  onClearCompare: () => void;
  onSwapCompare: () => void;
}

function ItemsCompareTab({
  runId,
  otherId,
  onClearCompare,
  onSwapCompare,
}: ItemsCompareTabProps) {
  const [filter, setFilter] = useState<CompareFilter>('all');

  const thisFetcher = useCallback(
    () =>
      v2.experimentItems(runId, {
        offset: 0,
        limit: ITEMS_COMPARE_LIMIT,
        filter: 'all',
        sort: 'item_id',
      }),
    [runId],
  );
  const otherFetcher = useCallback(
    () =>
      v2.experimentItems(otherId, {
        offset: 0,
        limit: ITEMS_COMPARE_LIMIT,
        filter: 'all',
        sort: 'item_id',
      }),
    [otherId],
  );
  const {
    data: thisItems,
    err: thisErr,
    refetch: thisRefetch,
    reloading: thisReloading,
    isInitialLoad: thisInitial,
  } = useV2Resource<ExperimentItemsResponse>(
    `experimentItems:${runId}:0:all:item_id`,
    thisFetcher,
  );
  const {
    data: otherItems,
    err: otherErr,
    refetch: otherRefetch,
    reloading: otherReloading,
    isInitialLoad: otherInitial,
  } = useV2Resource<ExperimentItemsResponse>(
    `experimentItems:${otherId}:0:all:item_id`,
    otherFetcher,
  );

  // Aligned rows + summary counts. One pass computes the rows, sorts
  // them regressed-first, and tallies the toolbar/footer chip counts in
  // the same sweep so unrelated parent state (filter pill, search input)
  // doesn't trigger three extra ``allRows.filter(...)`` walks per render.
  const { rows: allRows, totals } = useMemo<{
    rows: CompareRow[];
    totals: { regressed: number; fixed: number; missing: number };
  }>(() => {
    if (!thisItems || !otherItems) {
      return { rows: [], totals: { regressed: 0, fixed: 0, missing: 0 } };
    }
    const otherById = new Map(otherItems.items.map((i) => [i.id, i]));
    const rows: CompareRow[] = thisItems.items.map((thisRow) => {
      const otherRow = otherById.get(thisRow.id) ?? null;
      return { id: thisRow.id, thisRow, otherRow, diff: computeDiff(thisRow, otherRow) };
    });
    rows.sort((a, b) => {
      // regressed first (DESC), then fixed (DESC), then id
      if (b.diff.regressed !== a.diff.regressed)
        return b.diff.regressed - a.diff.regressed;
      if (b.diff.fixed !== a.diff.fixed) return b.diff.fixed - a.diff.fixed;
      return a.id.localeCompare(b.id);
    });
    let regressed = 0;
    let fixed = 0;
    let missing = 0;
    for (const r of rows) {
      if (r.diff.regressed > 0) regressed++;
      if (r.diff.fixed > 0) fixed++;
      if (r.otherRow === null) missing++;
    }
    return { rows, totals: { regressed, fixed, missing } };
  }, [thisItems, otherItems]);

  // Apply filter pill on top of the aligned rows.
  const visibleRows = useMemo<CompareRow[]>(() => {
    if (filter === 'all') return allRows;
    if (filter === 'regressed') return allRows.filter((r) => r.diff.regressed > 0);
    if (filter === 'fixed') return allRows.filter((r) => r.diff.fixed > 0);
    return allRows.filter((r) => r.diff.regressed === 0 && r.diff.fixed === 0);
  }, [allRows, filter]);

  // Loading / error gates -------------------------------------------------
  if (thisErr && otherErr) {
    return (
      <div style={{ padding: '20px 36px' }}>
        <Card style={{ padding: 16, borderColor: E.fail }}>
          <Eyebrow style={{ color: E.fail }}>
            Could not load items for compare
          </Eyebrow>
          <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2, marginTop: 6 }}>
            {runId}: {thisErr}
          </div>
          <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2, marginTop: 4 }}>
            {otherId}: {otherErr}
          </div>
          <div style={{ marginTop: 12 }}>
            <Btn kind="secondary" size="sm" onClick={onClearCompare}>
              Clear compare
            </Btn>
          </div>
        </Card>
      </div>
    );
  }

  if (!thisItems || !otherItems) {
    return (
      <div style={{ padding: '20px 36px' }}>
        <Skeleton w={320} h={28} />
        <div style={{ marginTop: 14 }}>
          <Skeleton w="100%" h={300} style={{ borderRadius: 6 }} />
        </div>
      </div>
    );
  }

  // Empty-side guard: if either run has zero items we can't align.
  if (thisItems.total === 0 || otherItems.total === 0) {
    const emptyId = thisItems.total === 0 ? runId : otherId;
    return (
      <div style={{ padding: '20px 36px' }}>
        <Card style={{ padding: 16, borderColor: E.warn }}>
          <Eyebrow style={{ color: E.warn }}>Cannot compare</Eyebrow>
          <div style={{ fontSize: 12.5, color: E.text2, marginTop: 6 }}>
            Run <span style={{ fontFamily: E.fMono }}>{emptyId}</span> has no
            items - cannot align side-by-side.
          </div>
          <div style={{ marginTop: 12 }}>
            <Btn kind="secondary" size="sm" onClick={onClearCompare}>
              Clear compare
            </Btn>
          </div>
        </Card>
      </div>
    );
  }

  // Aggregate counts for the summary chip in the toolbar (computed in
  // the allRows useMemo above to avoid extra walks per render).
  const totalsRegressed = totals.regressed;
  const totalsFixed = totals.fixed;
  const totalsUnchanged = allRows.length - totalsRegressed - totalsFixed;
  const reloading = thisReloading || otherReloading;
  const isInitialLoad = thisInitial || otherInitial;
  // Surface either side's background-refresh error. The retry kicks
  // both refetches since the user doesn't think of them as separate.
  const combinedErr = thisErr ?? otherErr ?? null;
  const refetchBoth = useCallback(() => {
    void thisRefetch();
    void otherRefetch();
  }, [thisRefetch, otherRefetch]);

  // Soft "missing on the other side" notice: rows where the other
  // run doesn't carry this item_id at all (e.g. dataset extended).
  const missingOnOther = totals.missing;

  return (
    <div style={{ padding: '20px 36px' }}>
      {/* COMPARE BAR */}
      <Card
        style={{
          padding: 10,
          marginBottom: 14,
          background: E.steelDim,
          borderColor: E.steel,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, flexWrap: 'wrap' }}>
          <span style={{ color: E.steel, fontFamily: E.fMono, fontSize: 10, letterSpacing: '0.08em' }}>
            COMPARING
          </span>
          <span style={{ fontFamily: E.fMono, color: E.ember }}>{runId}</span>
          <span style={{ color: E.text4 }}>vs</span>
          <span style={{ fontFamily: E.fMono, color: E.steel }}>{otherId}</span>
          <span style={{ flex: 1 }} />
          <Btn kind="ghost" size="sm" onClick={onSwapCompare} title="Swap A and B">
            ↔ Swap A/B
          </Btn>
          <Btn kind="bare" size="sm" onClick={onClearCompare}>
            Clear compare
          </Btn>
        </div>
      </Card>

      {/* TOOLBAR: filter pills + summary */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {COMPARE_FILTERS.map((f) => {
            const active = filter === f.key;
            const count =
              f.key === 'all'
                ? allRows.length
                : f.key === 'regressed'
                  ? totalsRegressed
                  : f.key === 'fixed'
                    ? totalsFixed
                    : totalsUnchanged;
            return (
              <button
                key={f.key}
                type="button"
                onClick={() => setFilter(f.key)}
                // Filter-pill toggle - sighted users see ember bg
                // when active, SR users get aria-pressed + a label
                // that includes the count (visible "Regressed 12"
                // would otherwise read with no toggle context).
                aria-pressed={active}
                aria-label={`${f.label}, ${count} item${count === 1 ? '' : 's'}`}
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
                {f.label}{' '}
                <span style={{ fontFamily: E.fMono, color: active ? E.ember : E.text3, fontSize: 11 }}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        <span style={{ flex: 1 }} />

        {missingOnOther > 0 && (
          <span
            style={{
              fontSize: 11,
              color: E.warn,
              fontFamily: E.fMono,
            }}
            title={`${missingOnOther} item(s) in this run are absent from ${otherId} - rendered with "missing" indicator on the other side`}
          >
            {missingOnOther} missing on other
          </span>
        )}
        <span style={{ fontSize: 11.5, color: E.text2, fontFamily: E.fMono }}>
          {visibleRows.length} of {allRows.length} rows
        </span>
        <UpdatingChip
          visible={reloading && !isInitialLoad}
          error={thisItems && otherItems ? combinedErr : null}
          onRetry={refetchBoth}
        />
      </div>

      {visibleRows.length === 0 ? (
        <Card
          style={{
            padding: 18,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}
        >
          <div style={{ flex: 1, fontSize: 13, color: E.text2 }}>
            No items in the <b style={{ color: E.ember }}>{filter}</b> bucket.
            <div style={{ fontSize: 11, color: E.text3, marginTop: 2, fontFamily: E.fMono }}>
              {allRows.length} row{allRows.length === 1 ? '' : 's'} total - try a different filter.
            </div>
          </div>
          {filter !== 'all' && (
            <Btn kind="secondary" size="sm" onClick={() => setFilter('all')}>
              Show all
            </Btn>
          )}
        </Card>
      ) : (
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <CompareTable
            rows={visibleRows}
            thisLabel={runId}
            otherLabel={otherId}
          />
        </Card>
      )}

      {/* PAGINATION FOOTER (cap notice) */}
      {(thisItems.total > ITEMS_COMPARE_LIMIT ||
        otherItems.total > ITEMS_COMPARE_LIMIT) && (
        <div
          style={{
            marginTop: 12,
            padding: 10,
            background: E.panel2,
            borderRadius: 6,
            fontSize: 11.5,
            color: E.text2,
            fontFamily: E.fMono,
            textAlign: 'center',
          }}
        >
          Showing {ITEMS_COMPARE_LIMIT} of {Math.max(thisItems.total, otherItems.total)} items - the
          remainder is not aligned in this view. Filter the source runs to focus on a subset.
        </div>
      )}

      <div style={{ height: 30 }} />
    </div>
  );
}

interface CompareTableProps {
  rows: CompareRow[];
  thisLabel: string;
  otherLabel: string;
}

/**
 * Three-column compare table: this | other | diff.
 * Each side renders a row of small per-metric dots using the union
 * of metric_ids across both rows so the dot positions line up.
 */
function CompareTable({ rows, thisLabel, otherLabel }: CompareTableProps) {
  const gridCols = '110px 1fr 1px 1fr 90px';
  return (
    <div>
      {/* HEADER */}
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
        <div>Item</div>
        <div style={{ color: E.ember }} title={thisLabel}>
          This - {thisLabel}
        </div>
        <div style={{ background: E.hair2, height: 14 }} />
        <div style={{ color: E.steel }} title={otherLabel}>
          Other - {otherLabel}
        </div>
        <div style={{ textAlign: 'right' }}>Diff</div>
      </div>

      {rows.map((r) => (
        <CompareRowView key={r.id} row={r} gridCols={gridCols} />
      ))}
    </div>
  );
}

interface CompareRowViewProps {
  row: CompareRow;
  gridCols: string;
}

function CompareRowView({ row, gridCols }: CompareRowViewProps) {
  // Union of metric_ids across both sides, preserving this-side order
  // first so the dominant run's metric layout determines the column order.
  const metricIds = useMemo<string[]>(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const pm of row.thisRow.per_metric) {
      if (!seen.has(pm.metric_id)) {
        seen.add(pm.metric_id);
        ordered.push(pm.metric_id);
      }
    }
    if (row.otherRow) {
      for (const pm of row.otherRow.per_metric) {
        if (!seen.has(pm.metric_id)) {
          seen.add(pm.metric_id);
          ordered.push(pm.metric_id);
        }
      }
    }
    return ordered;
  }, [row]);

  const otherByMetric = useMemo(() => {
    if (!row.otherRow) return new Map<string, ExperimentItemRow['per_metric'][number]>();
    return new Map(row.otherRow.per_metric.map((pm) => [pm.metric_id, pm]));
  }, [row]);
  const thisByMetric = useMemo(
    () => new Map(row.thisRow.per_metric.map((pm) => [pm.metric_id, pm])),
    [row],
  );

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: gridCols,
        gap: 10,
        padding: '8px 16px',
        borderTop: `1px solid ${E.hair}`,
        alignItems: 'center',
        minHeight: 36,
      }}
    >
      {/* Item id (mono, narrow, sticky-feel via color) */}
      <div
        style={{
          fontFamily: E.fMono,
          fontSize: 11,
          color: E.text2,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={row.id}
      >
        {row.id.slice(0, 12)}
      </div>

      {/* This run dots */}
      <DotRow
        metricIds={metricIds}
        byMetric={thisByMetric}
        absentLabel="not graded in this run"
      />

      {/* Divider */}
      <div style={{ background: E.hair2, alignSelf: 'stretch', width: 1 }} />

      {/* Other run dots (or "missing" if no row at all) */}
      {row.otherRow ? (
        <DotRow
          metricIds={metricIds}
          byMetric={otherByMetric}
          absentLabel="not graded in other run"
        />
      ) : (
        <div
          style={{
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.warn,
            letterSpacing: '0.06em',
          }}
          title="This item id is not present in the other run"
        >
          missing
        </div>
      )}

      {/* Diff cell */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <DiffChip diff={row.diff} hasOther={row.otherRow !== null} />
      </div>
    </div>
  );
}

interface DotRowProps {
  metricIds: string[];
  byMetric: Map<string, ExperimentItemRow['per_metric'][number]>;
  absentLabel: string;
}

/** A row of small per-metric status dots. Hover via title= for tooltip. */
function DotRow({ metricIds, byMetric, absentLabel }: DotRowProps) {
  return (
    <div style={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
      {metricIds.map((mid) => {
        const pm = byMetric.get(mid);
        let dotColor: string;
        let titleText: string;
        if (!pm) {
          dotColor = E.text4;
          titleText = `${mid}: ${absentLabel}`;
        } else if (pm.passed === true) {
          dotColor = E.pass;
          titleText = `${mid}: passed${pm.score != null ? ` (${pm.score.toFixed(2)})` : ''}`;
        } else if (pm.passed === false) {
          dotColor = E.fail;
          titleText = `${mid}: failed${pm.score != null ? ` (${pm.score.toFixed(2)})` : ''}`;
        } else {
          dotColor = E.text4;
          titleText = `${mid}: no result`;
        }
        return (
          <span
            key={mid}
            title={titleText}
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: dotColor,
              display: 'inline-block',
              flexShrink: 0,
            }}
          />
        );
      })}
    </div>
  );
}

interface DiffChipProps {
  diff: CompareDiff;
  hasOther: boolean;
}

/**
 * Diff summary chip:
 *   "+N" green when only fixed
 *   "-N" red  when only regressed
 *   "+N -M" mixed  when both
 *   "·"  muted  when no change (or other side missing)
 */
function DiffChip({ diff, hasOther }: DiffChipProps) {
  if (!hasOther) {
    return (
      <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text4 }}>-</span>
    );
  }
  const { fixed, regressed } = diff;
  if (fixed === 0 && regressed === 0) {
    return (
      <span
        style={{ fontFamily: E.fMono, fontSize: 13, color: E.text4 }}
        title="No metrics changed verdict"
      >
        ·
      </span>
    );
  }
  return (
    <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
      {fixed > 0 && (
        <span
          title={`${fixed} metric(s) fixed (failed in other, passed here)`}
          style={{
            fontFamily: E.fMono,
            fontSize: 11,
            color: E.pass,
            background: E.passDim,
            padding: '1px 6px',
            borderRadius: 3,
          }}
        >
          +{fixed}
        </span>
      )}
      {regressed > 0 && (
        <span
          title={`${regressed} metric(s) regressed (passed in other, failed here)`}
          style={{
            fontFamily: E.fMono,
            fontSize: 11,
            color: E.fail,
            background: E.failDim,
            padding: '1px 6px',
            borderRadius: 3,
          }}
        >
          -{regressed}
        </span>
      )}
    </span>
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
  /** Initial filter to apply on first mount. Used when the user
   * arrives via "View all N failures" from the Summary tab so they
   * land on the failed-only view without an extra click. */
  initialFilter?: ExperimentItemsFilter;
}

function ItemsTab({ runId, initialFilter }: ItemsTabProps) {
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<ExperimentItemsFilter>(initialFilter ?? 'all');
  const [sort, setSort] = useState<ExperimentItemsSort>('item_id');
  // Page-local item-id search via the shared useSearchFilter hook.
  // No sessionStorage here - the search is run-specific and a stale
  // query from a different run/tab would confuse. Hook provides the
  // debounce + Esc handling + focus ref; the "/" hotkey below
  // mirrors the drawer / experiments / runner-output pattern.
  const {
    input: search,
    setInput: setSearch,
    query: searchQuery,
    inputRef: searchRef,
    onKeyDown: onSearchKeyDown,
  } = useSearchFilter({});
  // Window-level "/" focuses the search box. Skipped when the user
  // is already typing in another input/textarea/contentEditable
  // surface (would otherwise eat their actual character).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || target?.isContentEditable) return;
      e.preventDefault();
      searchRef.current?.focus();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [searchRef]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetcher = useCallback(
    () => v2.experimentItems(runId, { offset, limit: PAGE_SIZE, filter, sort }),
    [runId, offset, filter, sort],
  );
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource<ExperimentItemsResponse>(
    `experimentItems:${runId}:${offset}:${filter}:${sort}`,
    fetcher,
  );

  // Prefetch the NEXT and PREVIOUS pages once the current one
  // resolves so paging in either direction is instant. Without
  // these, a "did I miss something" back-and-forth scroll pays a
  // round-trip every click. Lazy (after data lands so we know
  // `total`) and bounds-checked on both ends. useV2Resource dedupes
  // by cache key so a back-and-forth oscillation is fetch-free
  // after the first traversal.
  useEffect(() => {
    if (!data) return;
    const nextOffset = offset + PAGE_SIZE;
    if (nextOffset < data.total) {
      prefetchV2(
        `experimentItems:${runId}:${nextOffset}:${filter}:${sort}`,
        () => v2.experimentItems(runId, {
          offset: nextOffset,
          limit: PAGE_SIZE,
          filter,
          sort,
        }),
      );
    }
    const prevOffset = offset - PAGE_SIZE;
    if (prevOffset >= 0) {
      prefetchV2(
        `experimentItems:${runId}:${prevOffset}:${filter}:${sort}`,
        () => v2.experimentItems(runId, {
          offset: prevOffset,
          limit: PAGE_SIZE,
          filter,
          sort,
        }),
      );
    }
  }, [data, offset, filter, sort, runId]);

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

  const visibleItems = searchQuery
    ? data.items.filter((it) => it.id.toLowerCase().includes(searchQuery))
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
                // Same aria-pressed pattern as the compare filters
                // above - SR users hear the toggle state instead of
                // inferring from the bg-color flip alone.
                aria-pressed={active}
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
            aria-label="Sort items"
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

        <div
          style={{
            flex: 1,
            minWidth: 180,
            maxWidth: 320,
            display: 'flex',
            alignItems: 'center',
            border: `1px solid ${E.hair}`,
            borderRadius: 6,
            background: E.panel,
          }}
        >
          <input
            ref={searchRef}
            type="text"
            aria-label="Filter items on this page by id"
            placeholder="Filter this page by id... (/ to focus)"
            title={
              searchQuery
                ? `Filtering ${visibleItems.length} of ${data.items.length} items on this page (page-local; items on other pages aren't searched)`
                : 'Filter visible items on this page by id (substring match). Items on other pages aren\'t searched - use Prev/Next to scan more.'
            }
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={onSearchKeyDown}
            style={{
              flex: 1,
              padding: '6px 10px',
              fontSize: 12,
              fontFamily: E.fMono,
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: E.text1,
            }}
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              title="Clear search (Esc)"
              aria-label="Clear search"
              style={{
                fontFamily: E.fMono,
                fontSize: 13,
                color: E.text3,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: '0 8px',
                lineHeight: 1,
              }}
            >
              <span aria-hidden="true">×</span>
            </button>
          )}
        </div>

        <span style={{ flex: 1 }} />

        <span style={{ fontSize: 11.5, color: E.text2, fontFamily: E.fMono }}>
          Showing {showingFrom}-{showingTo} of {data.total}
        </span>
        <UpdatingChip
          visible={reloading && !isInitialLoad}
          error={data ? err : null}
          onRetry={refetch}
        />
      </div>

      {data.total === 0 ? (
        <Card
          style={{
            padding: 18,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}
        >
          <div style={{ flex: 1, fontSize: 13, color: E.text2 }}>
            No items match the <b style={{ color: E.ember }}>{filter}</b> filter.
            <div style={{ fontSize: 11, color: E.text3, marginTop: 2, fontFamily: E.fMono }}>
              Try switching to "All" to see every item.
            </div>
          </div>
          {filter !== 'all' && (
            <Btn kind="secondary" size="sm" onClick={() => changeFilter('all')}>
              Show all
            </Btn>
          )}
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
            <div
              style={{
                padding: 18,
                display: 'flex',
                alignItems: 'center',
                gap: 12,
              }}
            >
              <div style={{ flex: 1, fontSize: 12.5, color: E.text3 }}>
                {searchQuery ? (
                  // Search active + no matches: clarify scope
                  // (page-local) AND offer the page-paging escape.
                  // Branch on `searchQuery` (debounced) not `search`
                  // (raw) so the message matches what the filter is
                  // actually doing - otherwise clearing search via
                  // Esc flashes the wrong empty-state for ~120ms
                  // while the debounced query catches up.
                  <>
                    No items match{' '}
                    <b style={{ color: E.ember, fontFamily: E.fMono }}>"{search}"</b>
                    {' '}on this page.
                    {canNext || canPrev ? (
                      <span style={{ color: E.text4 }}>
                        {' '}Items on other pages aren&apos;t searched -
                        use Prev/Next to scan more.
                      </span>
                    ) : null}
                  </>
                ) : data.total === 0 ? (
                  // Truly empty result set (e.g. filter=failed but
                  // no failures across the whole run).
                  <>
                    No items match the current filter
                    {filter !== 'all' && (
                      <>
                        {' ('}
                        <b style={{ color: E.ember, fontFamily: E.fMono }}>
                          {filter}
                        </b>
                        {')'}
                      </>
                    )}
                    .
                  </>
                ) : (
                  // Empty page within a non-empty result set
                  // (e.g. filter changed but offset wasn't reset
                  // - shouldn't normally happen since changeFilter
                  // resets offset, but defensive against future
                  // refactors).
                  <>This page is empty.</>
                )}
              </div>
              {searchQuery && (
                <Btn kind="secondary" size="sm" onClick={() => setSearch('')}>
                  Clear search
                </Btn>
              )}
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
        {body && body.length > 0
          ? linkifyText(body, makeUrlCounter())
          : <span style={{ color: E.text4 }}>-</span>}
      </div>
    </div>
  );
}
