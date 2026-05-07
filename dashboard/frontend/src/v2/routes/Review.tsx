/**
 * Review - cluster-led human review.
 *
 * Top-level layout (top -> bottom):
 *  1. Header eyebrow + keyboard hint.
 *  2. Page H1 "<N> failures - <M> patterns".
 *  3. Cluster strip (4-up grid, color-coded by kind, ember outline on active).
 *  4. Two-column body:
 *       Left: cluster item list with select-all + per-item checkboxes.
 *       Right: trace pane (side-by-side user/assistant), pattern banner,
 *              batch verdict bar, single-item verdict row, note field,
 *              and a 2-up "calibration boost / session" footer.
 *  5. "why this changed" dashed footer.
 *
 * Keyboard:
 *  - P/F/S       - pass/fail/skip the active item.
 *  - Shift+P/F   - batch verdict on every selected item in the active cluster.
 *  - N or Enter  - advance to next item.
 *  - Esc         - leave to /.
 *  - Backspace   - dismiss the feedback chip (best-effort undo).
 *
 * Data:
 *  - useV2Resource('reviewQueue', v2.reviewQueue) gives the flat ReviewItem
 *    list (still the source of truth for trace + judge data).
 *  - v2.experimentClusters(runId) gives semantic clusters for the most
 *    common source_run_id across the queue. Cluster items carry only
 *    {id, text}; we map their `id` -> full ReviewItem by item_id.
 *  - When clusters() returns nothing (small/no failure run, or the
 *    backend hasn't shipped the route yet), we fall back to a single
 *    synthetic "All failures" cluster so the page still works.
 *
 * Preserved tour anchors: review-queue / review-item / review-verdict-buttons.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, UpdatingChip } from '../ui';
import { Bar, Donut } from '../ui/charts';
import { v2 } from '../api/client';
import type {
  CalibrationSuggestion,
  ClustersResponse,
  ReviewItem,
  ReviewQueue,
  SemanticCluster,
} from '../api/types';
import { listCli, type CliSchema } from '../api/cli';
import { useV2Resource } from '../hooks/useV2Resource';
import { useRouteTour } from '../tour/useRouteTour';
import { REVIEW_FAILURES_TOUR_ID } from '../tour/scripts/reviewFailures';
import { useProject } from '../hooks/useProject';
import { openCliRunner } from '../cliRunnerBridge';
import { linkifyText, makeUrlCounter } from '../textRender';
import { E } from '../tokens';

type Verdict = 'pass' | 'fail' | 'skip';

interface FeedbackChipState {
  kind: 'progress' | 'ready';
  metric: string;
  count: number;
  threshold: number;
  suggestion: CalibrationSuggestion | null;
}

interface BatchToastState {
  verdict: Verdict;
  total: number;
  failed: number;
}

const FEEDBACK_DISMISS_MS = 4000;
const BATCH_DISMISS_MS = 4000;
const SYNTHETIC_OTHER_ID = '__other__';
const SYNTHETIC_ALL_ID = '__all__';
/** Cluster strip slot count - includes the trailing "Other" bucket. */
const MAX_VISIBLE_CLUSTERS = 4;

/* ====================================================================
 * Cluster construction
 * ==================================================================*/

interface UiCluster {
  id: string;
  label: string;
  kind: 'fail' | 'warn' | 'info';
  /** 0..1 confidence; null means "don't render conf line" (Other / synthetic). */
  confidence: number | null;
  pattern_hint: string | null;
  /** ReviewItems mapped from cluster member ids; ordered by queue order. */
  items: ReviewItem[];
}

/**
 * Build the list of clusters used by the strip + item list. Pads with a
 * synthetic "Other" bucket holding any queue items that didn't land in a
 * server-provided cluster (or all queue items, if clusters() was empty).
 */
function buildUiClusters(
  queueItems: ReviewItem[],
  clustersData: ClustersResponse | null,
): UiCluster[] {
  // Index queue items by id once - the map is used to (a) resolve cluster
  // membership and (b) compute the "Other" bucket of orphans.
  const byId = new Map<string, ReviewItem>();
  queueItems.forEach((it) => byId.set(it.item_id, it));

  // Empty server response -> collapse everything into one synthetic
  // cluster. Keeps the UI shape stable when the run is too small to
  // cluster, or when the /clusters endpoint hasn't shipped yet.
  const serverClusters: SemanticCluster[] = clustersData?.clusters ?? [];
  if (serverClusters.length === 0) {
    if (queueItems.length === 0) return [];
    return [
      {
        id: SYNTHETIC_ALL_ID,
        label: 'All failures',
        kind: 'fail',
        confidence: null,
        pattern_hint: null,
        items: queueItems.slice(),
      },
    ];
  }

  // Map server clusters -> UiCluster, dropping members the queue doesn't
  // contain (the queue is a sliding window; clusters may reference items
  // outside it).
  const used = new Set<string>();
  const mapped: UiCluster[] = serverClusters.map((c) => {
    const items: ReviewItem[] = [];
    for (const m of c.items) {
      const it = byId.get(m.id);
      if (it && !used.has(it.item_id)) {
        items.push(it);
        used.add(it.item_id);
      }
    }
    return {
      id: c.id,
      label: c.label,
      kind: c.kind,
      confidence: c.confidence,
      pattern_hint: c.pattern_hint,
      items,
    };
  });

  // Drop empty mapped clusters (every member was outside the queue window).
  const nonEmpty = mapped.filter((c) => c.items.length > 0);

  // Limit to MAX_VISIBLE_CLUSTERS - 1 so the trailing "Other" bucket
  // has a slot. Anything past the cap rolls into Other along with
  // queue items that no server cluster claimed.
  const visible = nonEmpty.slice(0, MAX_VISIBLE_CLUSTERS - 1);
  const overflow = nonEmpty.slice(MAX_VISIBLE_CLUSTERS - 1);

  const otherItems: ReviewItem[] = [];
  for (const c of overflow) otherItems.push(...c.items);
  for (const it of queueItems) {
    if (!used.has(it.item_id)) otherItems.push(it);
  }

  if (otherItems.length > 0) {
    visible.push({
      id: SYNTHETIC_OTHER_ID,
      label: 'Other',
      kind: 'info',
      confidence: null,
      pattern_hint: null,
      items: otherItems,
    });
  }

  return visible;
}

/** Tally source_run_id across queue items, return the most common. */
function pickRunId(items: ReviewItem[]): string | null {
  if (items.length === 0) return null;
  const counts = new Map<string, number>();
  for (const it of items) {
    counts.set(it.source_run_id, (counts.get(it.source_run_id) ?? 0) + 1);
  }
  let best: string | null = null;
  let bestN = 0;
  for (const [k, v] of counts) {
    if (v > bestN) {
      best = k;
      bestN = v;
    }
  }
  return best;
}

function clusterAccent(kind: UiCluster['kind']): string {
  if (kind === 'fail') return E.fail;
  if (kind === 'warn') return E.warn;
  return E.steel;
}

function pillColor(kind: 'pass' | 'fail' | 'warn'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  return E.warn;
}

/**
 * Highlight `highlights` substrings inside `text`. Wraps each match in
 * a coloured span. Inline rather than imported because the trace pane
 * needs colour control (warn-tinted vs fail-tinted) per call.
 */
function renderHighlighted(
  text: string,
  highlights: string[],
  bg: string,
  fg: string,
) {
  const urlCounter = makeUrlCounter();
  if (!highlights || highlights.length === 0) return linkifyText(text, urlCounter);
  const parts: { text: string; mark: boolean }[] = [{ text, mark: false }];
  for (const h of highlights) {
    if (!h) continue;
    const next: { text: string; mark: boolean }[] = [];
    for (const p of parts) {
      if (p.mark) {
        next.push(p);
        continue;
      }
      const segs = p.text.split(h);
      for (let i = 0; i < segs.length; i++) {
        if (segs[i]) next.push({ text: segs[i], mark: false });
        if (i < segs.length - 1) next.push({ text: h, mark: true });
      }
    }
    parts.length = 0;
    parts.push(...next);
  }
  return parts.map((p, i) =>
    p.mark ? (
      <span
        key={i}
        style={{ background: bg, color: fg, padding: '1px 4px', borderRadius: 3 }}
      >
        {p.text}
      </span>
    ) : (
      <span key={i}>{linkifyText(p.text, urlCounter)}</span>
    ),
  );
}

/* ====================================================================
 * Top-level component
 * ==================================================================*/

export default function Review() {
  const project = useProject();
  const {
    data: queue,
    err: queueErr,
    reloading,
    isInitialLoad,
    refetch: refetchQueue,
  } = useV2Resource<ReviewQueue>('reviewQueue', v2.reviewQueue);
  useRouteTour(REVIEW_FAILURES_TOUR_ID, !!(queue && !queueErr));
  const { data: cmds } = useV2Resource<CliSchema[]>('commands', listCli);
  const navigate = useNavigate();

  const items = queue?.items ?? [];
  const runId = useMemo(() => pickRunId(items), [items]);

  // Fetch clusters for the dominant run id. Memoize the fetcher so the
  // resource hook's effect identity is stable per runId. The hook is
  // gated behind `enabled: !!runId` to avoid firing when the queue is
  // empty.
  const clustersFetcher = useCallback(() => {
    if (!runId) return Promise.resolve<ClustersResponse>({ clusters: [] });
    // The /clusters route may not be live on every backend - swallow
    // failures so the page falls back to the synthetic "All failures"
    // cluster instead of error-stating the whole pane.
    return v2.experimentClusters(runId).catch(() => ({ clusters: [] }));
  }, [runId]);
  const {
    data: clustersData,
    err: clustersErr,
    isInitialLoad: clustersInitial,
  } = useV2Resource<ClustersResponse>(
    runId ? `experimentClusters:${runId}` : 'experimentClusters:none',
    clustersFetcher,
    { enabled: !!runId },
  );

  const uiClusters = useMemo(
    () => buildUiClusters(items, clustersData),
    [items, clustersData],
  );

  // ===== Local UI state ============================================
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [calibrateErr, setCalibrateErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastChip, setLastChip] = useState<FeedbackChipState | null>(null);
  const [batchToast, setBatchToast] = useState<BatchToastState | null>(null);
  const dismissTimerRef = useRef<number | null>(null);
  const batchTimerRef = useRef<number | null>(null);

  /** Active cluster id; defaults to the first non-empty cluster. */
  const [activeClusterId, setActiveClusterId] = useState<string | null>(null);
  /** Active item id within the active cluster. */
  const [activeItemId, setActiveItemId] = useState<string | null>(null);
  /** Item ids the user has selected for batch verdict (per-cluster). */
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [note, setNote] = useState('');

  // Default active cluster: first one with items. Only set when unset
  // or when the previously-active cluster vanished.
  useEffect(() => {
    if (uiClusters.length === 0) {
      if (activeClusterId !== null) setActiveClusterId(null);
      return;
    }
    if (
      activeClusterId === null ||
      !uiClusters.find((c) => c.id === activeClusterId)
    ) {
      setActiveClusterId(uiClusters[0].id);
    }
  }, [uiClusters, activeClusterId]);

  const activeCluster = useMemo(
    () => uiClusters.find((c) => c.id === activeClusterId) ?? null,
    [uiClusters, activeClusterId],
  );

  // When the active cluster changes, default the selection to all items
  // in the cluster + first item as active. Matches the proposal's
  // "select-all on entry" UX.
  useEffect(() => {
    if (!activeCluster) {
      setSelectedIds(new Set());
      setActiveItemId(null);
      return;
    }
    setSelectedIds(new Set(activeCluster.items.map((it) => it.item_id)));
    setActiveItemId(activeCluster.items[0]?.item_id ?? null);
    setNote('');
    // Intentionally only react to activeClusterId. activeCluster identity
    // changes whenever queue refetches; that would clobber the user's
    // selection mid-review.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeClusterId]);

  const activeItem: ReviewItem | null = useMemo(() => {
    if (!activeCluster) return null;
    return (
      activeCluster.items.find((it) => it.item_id === activeItemId) ??
      activeCluster.items[0] ??
      null
    );
  }, [activeCluster, activeItemId]);

  const totalFailures = items.length;
  const totalPatterns = uiClusters.length;

  const err = submitErr ?? queueErr;

  // ===== Calibration banner helpers =================================
  const openCalibrate = useCallback(
    (s: CalibrationSuggestion) => {
      const cmd = cmds?.find((c) => c.id === 'calibrate');
      if (!cmd) {
        setCalibrateErr(
          'The calibrate command is not in this build of the CLI catalog.',
        );
        return;
      }
      const paramNames = new Set(cmd.params.map((p) => p.name));
      const initialValues: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(s.cli_args)) {
        if (paramNames.has(key)) initialValues[key] = value;
      }
      setCalibrateErr(null);
      openCliRunner(cmd, { initialValues });
    },
    [cmds],
  );

  const scheduleChipDismiss = useCallback(() => {
    if (dismissTimerRef.current != null) {
      window.clearTimeout(dismissTimerRef.current);
    }
    dismissTimerRef.current = window.setTimeout(() => {
      setLastChip(null);
      dismissTimerRef.current = null;
    }, FEEDBACK_DISMISS_MS);
  }, []);

  const scheduleBatchDismiss = useCallback(() => {
    if (batchTimerRef.current != null) {
      window.clearTimeout(batchTimerRef.current);
    }
    batchTimerRef.current = window.setTimeout(() => {
      setBatchToast(null);
      batchTimerRef.current = null;
    }, BATCH_DISMISS_MS);
  }, []);

  // ===== Verdict submit (single + batch) ============================
  /**
   * Submit a verdict for a specific item. Inner helper used by both
   * single-item P/F/S and the Shift+P/F batch path.
   *
   * Returns the post-state metric chip (or null if no metric snapshot
   * was available). The caller decides whether to render the chip -
   * batch path renders a single batch toast instead.
   */
  const submitOne = useCallback(
    async (
      item: ReviewItem,
      verdict: Verdict,
      noteForItem: string | null,
    ): Promise<FeedbackChipState | null> => {
      const itemMetric = item.judge_breakdown[0]?.label ?? null;
      const beforeMatch =
        queue?.calibration_suggestions.find((s) => s.metric_id === itemMetric) ?? null;

      await v2.submitVerdict({
        item_id: item.item_id,
        source_run_id: item.source_run_id,
        verdict,
        note: noteForItem,
      });

      if (!itemMetric) return null;
      const fresh = await v2.reviewQueue().catch(() => null);
      if (!fresh) return null;
      const afterMatch =
        fresh.calibration_suggestions.find((s) => s.metric_id === itemMetric) ?? null;
      if (!afterMatch) return null;
      const beforeCount = beforeMatch?.verdict_count ?? 0;
      if (afterMatch.verdict_count <= beforeCount) return null;
      const ready = afterMatch.verdict_count >= afterMatch.threshold;
      return {
        kind: ready ? 'ready' : 'progress',
        metric: itemMetric,
        count: afterMatch.verdict_count,
        threshold: afterMatch.threshold,
        suggestion: afterMatch,
      };
    },
    [queue],
  );

  /** Advance the active item pointer to the next item in the cluster. */
  const advanceActive = useCallback(() => {
    if (!activeCluster) return;
    const idx = activeCluster.items.findIndex((it) => it.item_id === activeItemId);
    const next = activeCluster.items[idx + 1];
    if (next) setActiveItemId(next.item_id);
  }, [activeCluster, activeItemId]);

  /** P/F/S - single-item verdict on the active item. */
  const submitActive = useCallback(
    async (verdict: Verdict) => {
      if (!activeItem || submitting) return;
      setSubmitting(true);
      try {
        const chip = await submitOne(activeItem, verdict, note.trim() || null);
        setNote('');
        setSubmitErr(null);
        // Drop the just-submitted item from the selection so the count
        // shown on the batch bar matches "still to do".
        setSelectedIds((prev) => {
          if (!prev.has(activeItem.item_id)) return prev;
          const next = new Set(prev);
          next.delete(activeItem.item_id);
          return next;
        });
        advanceActive();
        await refetchQueue();
        if (chip) {
          setLastChip(chip);
          scheduleChipDismiss();
        }
      } catch (e) {
        setSubmitErr(String(e));
      } finally {
        setSubmitting(false);
      }
    },
    [
      activeItem,
      submitting,
      submitOne,
      note,
      advanceActive,
      refetchQueue,
      scheduleChipDismiss,
    ],
  );

  /** Shift+P / Shift+F - apply the same verdict to every selected item. */
  const submitBatch = useCallback(
    async (verdict: Verdict) => {
      if (!activeCluster || submitting) return;
      const targets = activeCluster.items.filter((it) => selectedIds.has(it.item_id));
      if (targets.length === 0) return;
      setSubmitting(true);
      let failed = 0;
      let lastChipResult: FeedbackChipState | null = null;
      for (const it of targets) {
        try {
          const chip = await submitOne(it, verdict, null);
          if (chip) lastChipResult = chip;
        } catch (e) {
          failed += 1;
          setSubmitErr(String(e));
        }
      }
      // Clear submitted ids out of the selection.
      setSelectedIds((prev) => {
        const next = new Set(prev);
        for (const it of targets) next.delete(it.item_id);
        return next;
      });
      // Move the active pointer past every submitted id.
      const submittedSet = new Set(targets.map((t) => t.item_id));
      const remaining = activeCluster.items.find((it) => !submittedSet.has(it.item_id));
      if (remaining) setActiveItemId(remaining.item_id);
      setBatchToast({ verdict, total: targets.length, failed });
      scheduleBatchDismiss();
      if (lastChipResult) {
        setLastChip(lastChipResult);
        scheduleChipDismiss();
      }
      await refetchQueue();
      setSubmitting(false);
    },
    [
      activeCluster,
      submitting,
      selectedIds,
      submitOne,
      refetchQueue,
      scheduleBatchDismiss,
      scheduleChipDismiss,
    ],
  );

  // Clear pending dismiss timers on unmount.
  useEffect(() => {
    return () => {
      if (dismissTimerRef.current != null) window.clearTimeout(dismissTimerRef.current);
      if (batchTimerRef.current != null) window.clearTimeout(batchTimerRef.current);
    };
  }, []);

  // Keyboard shortcuts.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')) return;
      const k = e.key.toLowerCase();
      if (e.shiftKey && (k === 'p' || k === 'f')) {
        e.preventDefault();
        void submitBatch(k === 'p' ? 'pass' : 'fail');
        return;
      }
      if (k === 'p') void submitActive('pass');
      else if (k === 'f') void submitActive('fail');
      else if (k === 's') void submitActive('skip');
      else if (k === 'n' || k === 'enter') {
        e.preventDefault();
        advanceActive();
      } else if (k === 'escape') {
        navigate('/');
      } else if (k === 'backspace') {
        setLastChip(null);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [submitActive, submitBatch, advanceActive, navigate]);

  // ===== Derived display data ======================================
  // Calibration boost donut: pick the first metric on the active item;
  // its suggestion (if any) drives the donut/segment counts.
  const boostInfo = useMemo(() => {
    if (!activeItem || !queue) return null;
    const metricLabels = activeItem.judge_breakdown.map((b) => b.label);
    if (metricLabels.length === 0) return null;
    const sug = queue.calibration_suggestions.find((s) =>
      metricLabels.includes(s.metric_id),
    );
    if (!sug) {
      return {
        metrics: metricLabels.slice(0, 2),
        verdictCount: 0,
        threshold: 10,
        toGo: 10,
      };
    }
    return {
      metrics: metricLabels.slice(0, 2),
      verdictCount: sug.verdict_count,
      threshold: sug.threshold,
      toGo: Math.max(0, sug.threshold - sug.verdict_count),
    };
  }, [activeItem, queue]);

  // Session counts: how far through the queue the reviewer is. The
  // queue itself is a sliding window so we can't show "100% done" -
  // we approximate progress as "(idx of active item in flat queue)".
  const sessionInfo = useMemo(() => {
    if (!activeCluster) return null;
    const total = items.length;
    const idx = activeCluster.items.findIndex((it) => it.item_id === activeItemId);
    const flatIdx = items.findIndex((it) => it.item_id === activeItemId);
    return {
      doneInCluster: idx >= 0 ? idx : 0,
      totalInCluster: activeCluster.items.length,
      flatIdx: flatIdx >= 0 ? flatIdx : 0,
      total,
    };
  }, [activeCluster, activeItemId, items]);

  return (
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: '24px' }}>
        {/* Header eyebrow + keyboard hint =========================== */}
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            marginBottom: 4,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Eyebrow>Review - cluster mode</Eyebrow>
            <UpdatingChip
              visible={reloading && !isInitialLoad}
              error={queue ? queueErr : null}
              onRetry={refetchQueue}
            />
          </div>
          <span
            style={{
              fontFamily: E.fMono,
              fontSize: 10.5,
              color: E.text3,
              letterSpacing: '0.02em',
            }}
          >
            P pass - F fail - S skip - shift+P/F batch
          </span>
        </div>

        {/* H1 ======================================================= */}
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 30,
            fontWeight: 400,
            margin: '4px 0 14px',
            color: E.text0,
            letterSpacing: '-0.015em',
          }}
        >
          {totalFailures} failures - {totalPatterns} patterns
        </h1>

        {/* Feedback chips ============================================ */}
        {lastChip && (
          <div style={{ marginBottom: 12, animation: 'eFadeIn 180ms ease' }}>
            {lastChip.kind === 'progress' ? (
              <Pill
                mono
                color={E.text1}
                bg={E.passDim}
                style={{
                  fontSize: 11.5,
                  padding: '6px 12px',
                  border: `1px solid ${E.pass}33`,
                }}
              >
                + 1 verdict on{' '}
                <span style={{ color: E.text0, marginLeft: 4, marginRight: 4 }}>
                  {lastChip.metric}
                </span>
                <span style={{ color: E.text3, marginLeft: 4 }}>
                  ({lastChip.count} of {lastChip.threshold} - calibration unlocks at{' '}
                  {lastChip.threshold})
                </span>
              </Pill>
            ) : (
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 14px',
                  borderRadius: 999,
                  background: E.emberDim,
                  border: `1px solid ${E.ember}55`,
                  color: E.text0,
                  fontSize: 12,
                }}
              >
                <span style={{ fontFamily: E.fMono, color: E.ember }}>
                  Calibration ready for{' '}
                  <span style={{ color: E.text0 }}>{lastChip.metric}</span>
                </span>
                <Btn
                  kind="primary"
                  size="sm"
                  onClick={() => {
                    if (lastChip.suggestion) openCalibrate(lastChip.suggestion);
                  }}
                  disabled={!cmds || !lastChip.suggestion}
                  title={
                    cmds
                      ? `Open the calibrate form pre-filled for ${lastChip.metric}`
                      : 'Loading CLI catalog...'
                  }
                >
                  Run it now -&gt;
                </Btn>
              </div>
            )}
          </div>
        )}

        {batchToast && batchToast.total > 0 && (
          <div
            style={{
              marginBottom: 12,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 12px',
              borderRadius: 999,
              background: batchToast.failed > 0 ? E.failDim : E.passDim,
              border: `1px solid ${
                batchToast.failed > 0 ? E.fail + '33' : E.pass + '33'
              }`,
              color: batchToast.failed > 0 ? E.fail : E.pass,
              fontSize: 11.5,
              fontFamily: E.fMono,
            }}
          >
            Submitted {batchToast.total - batchToast.failed} / {batchToast.total} as{' '}
            {batchToast.verdict}
            {batchToast.failed > 0 && ` (${batchToast.failed} failed)`}
          </div>
        )}

        {calibrateErr && (
          <div
            role="alert"
            style={{
              marginBottom: 12,
              padding: '10px 14px',
              borderRadius: 8,
              background: E.failDim,
              border: `1px solid ${E.fail}55`,
              color: E.fail,
              fontFamily: E.fMono,
              fontSize: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <span style={{ flex: 1, lineHeight: 1.5 }}>{calibrateErr}</span>
            <button
              type="button"
              onClick={() => setCalibrateErr(null)}
              aria-label="Dismiss error"
              style={{
                background: 'transparent',
                border: 'none',
                color: 'currentColor',
                cursor: 'pointer',
                fontSize: 14,
                lineHeight: 1,
                padding: '0 2px',
                opacity: 0.7,
              }}
            >
              x
            </button>
          </div>
        )}

        {/* Calibration suggestion banner (preserved from v1) ======== */}
        {queue?.calibration_suggestions && queue.calibration_suggestions.length > 0 && (
          <div
            style={{
              marginBottom: 14,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            {queue.calibration_suggestions.map((s) => (
              <Card key={`${s.dataset}-${s.metric_id}`} accent style={{ padding: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  <div style={{ flex: 1 }}>
                    <Eyebrow style={{ color: E.ember }}>Calibration ready</Eyebrow>
                    <div
                      style={{
                        marginTop: 4,
                        fontSize: 13,
                        color: E.text1,
                        lineHeight: 1.5,
                      }}
                    >
                      {s.verdict_count} verdicts collected on{' '}
                      <span style={{ fontFamily: E.fMono, color: E.text0 }}>
                        {s.metric_id}
                      </span>{' '}
                      <span style={{ color: E.text3 }}>
                        (threshold: {s.threshold}, dataset: {s.dataset})
                      </span>
                      .
                    </div>
                  </div>
                  <Btn
                    kind="primary"
                    size="md"
                    onClick={() => openCalibrate(s)}
                    disabled={!cmds}
                    title={
                      cmds
                        ? `Open the calibrate form pre-filled for ${s.metric_id}`
                        : 'Loading CLI catalog...'
                    }
                  >
                    Run calibrate -&gt;
                  </Btn>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Error fallback ============================================ */}
        {err && !queue && (
          <Card style={{ padding: 16, marginBottom: 18, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error</Eyebrow>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.text2,
                marginTop: 6,
                marginBottom: 10,
              }}
            >
              {err}
            </div>
            <Btn kind="ghost" size="sm" onClick={() => refetchQueue()}>
              Retry
            </Btn>
          </Card>
        )}
        {submitErr && queue && (
          <Card style={{ padding: 12, marginBottom: 14, borderColor: E.fail }}>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.fail,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span style={{ flex: 1 }}>{submitErr}</span>
              <Btn kind="ghost" size="sm" onClick={() => setSubmitErr(null)}>
                Dismiss
              </Btn>
            </div>
          </Card>
        )}

        {/* Loading skeleton ========================================== */}
        {(!queue || (clustersInitial && !!runId && !clustersErr)) && !err && (
          <>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 8,
                marginBottom: 14,
              }}
            >
              {[0, 1, 2, 3].map((i) => (
                <Card key={i} style={{ padding: 12 }}>
                  <Skeleton w={120} h={11} />
                  <div style={{ marginTop: 10 }}>
                    <Skeleton w={50} h={22} />
                  </div>
                </Card>
              ))}
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '320px 1fr',
                gap: 14,
              }}
            >
              <Card style={{ padding: 14 }}>
                <Skeleton w={140} h={11} />
                <div style={{ marginTop: 10 }}>
                  <Skeleton w="100%" h={40} />
                </div>
                <div style={{ marginTop: 10 }}>
                  <Skeleton w="100%" h={40} />
                </div>
              </Card>
              <Card style={{ padding: 14 }}>
                <Skeleton w={200} h={14} />
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={120} style={{ borderRadius: 8 }} />
                </div>
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={40} />
                </div>
              </Card>
            </div>
          </>
        )}

        {/* Empty / done ============================================== */}
        {queue && uiClusters.length === 0 && (
          <Card style={{ padding: 32, textAlign: 'center' }}>
            <div
              style={{
                fontFamily: E.fSerif,
                fontSize: 22,
                color: E.text0,
                marginBottom: 8,
              }}
            >
              Inbox zero
            </div>
            <div style={{ fontSize: 13, color: E.text2 }}>
              The judge is confident on every recent item.
            </div>
          </Card>
        )}

        {queue && uiClusters.length > 0 && (
          <>
            {/* ===== Cluster strip =================================== */}
            <ClusterStrip
              clusters={uiClusters}
              activeId={activeClusterId}
              onSelect={(id) => setActiveClusterId(id)}
            />

            {clustersErr && (
              <div
                style={{
                  marginBottom: 12,
                  fontSize: 11.5,
                  color: E.warn,
                  fontFamily: E.fMono,
                }}
              >
                Could not load semantic clusters - showing all failures in one bucket.
              </div>
            )}

            {/* ===== Body grid ====================================== */}
            <div
              data-coachmark="review-queue"
              style={{
                display: 'grid',
                gridTemplateColumns: '320px 1fr',
                gap: 14,
              }}
            >
              <ClusterItemList
                cluster={activeCluster}
                activeItemId={activeItemId}
                selectedIds={selectedIds}
                onSelectItem={(id) => setActiveItemId(id)}
                onToggleSelected={(id) => {
                  setSelectedIds((prev) => {
                    const next = new Set(prev);
                    if (next.has(id)) next.delete(id);
                    else next.add(id);
                    return next;
                  });
                }}
                onToggleAll={() => {
                  if (!activeCluster) return;
                  setSelectedIds((prev) => {
                    const all = activeCluster.items.map((it) => it.item_id);
                    const allOn = all.every((id) => prev.has(id));
                    if (allOn) return new Set();
                    return new Set(all);
                  });
                }}
              />

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <Card style={{ padding: 14 }} data-coachmark="review-item">
                  {activeItem ? (
                    <TracePane
                      item={activeItem}
                      cluster={activeCluster}
                      selectedCount={selectedIds.size}
                      onConfirmPattern={() => {
                        // No backend route yet for "confirm pattern"; show a
                        // UI-only ack toast so the click feels responsive.
                        setBatchToast({ verdict: 'pass', total: 0, failed: 0 });
                        scheduleBatchDismiss();
                      }}
                      onBatch={(verdict) => void submitBatch(verdict)}
                      onReviewIndividually={() => {
                        // Drop the bulk selection so each P/F lands on the
                        // single active item only.
                        setSelectedIds(new Set());
                      }}
                      onSingle={(verdict) => void submitActive(verdict)}
                      submitting={submitting}
                      note={note}
                      onNote={setNote}
                    />
                  ) : (
                    <div style={{ padding: 24, color: E.text3, fontSize: 12 }}>
                      No item selected.
                    </div>
                  )}
                </Card>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 12,
                  }}
                >
                  <CalibrationBoostCard info={boostInfo} />
                  <SessionCard info={sessionInfo} />
                </div>
              </div>
            </div>

            {/* ===== Why this changed footer ======================== */}
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
                style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}
              >
                why this changed -&gt;
              </span>{' '}
              Reviewing items 1-by-1 when many share the same bug is a waste.
              Cluster-led mode lets you confirm the pattern, batch the verdict,
              and only pull individual items into focus when they're ambiguous.
              Side-by-side trace (input + tools + reply + judge) makes the
              decision faster and more correct.
            </div>
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}

/* ====================================================================
 * Subcomponents
 * ==================================================================*/

interface ClusterStripProps {
  clusters: UiCluster[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

function ClusterStrip({ clusters, activeId, onSelect }: ClusterStripProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 8,
        marginBottom: 14,
      }}
    >
      {clusters.map((c) => {
        const active = c.id === activeId;
        return (
          <button
            type="button"
            key={c.id}
            onClick={() => onSelect(c.id)}
            style={{
              textAlign: 'left',
              padding: '10px 12px',
              background: active ? E.panel : E.panel2,
              border: `1px solid ${active ? E.ember : E.hair2}`,
              borderLeft: `3px solid ${clusterAccent(c.kind)}`,
              borderRadius: 6,
              cursor: 'pointer',
              position: 'relative',
              fontFamily: E.fSans,
            }}
          >
            <div
              style={{
                fontSize: 12,
                color: active ? E.text0 : E.text1,
                fontWeight: 500,
                marginBottom: 4,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              title={c.label}
            >
              {c.label}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
              <span style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>
                {c.items.length}
              </span>
              {c.confidence != null && (
                <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
                  conf {c.confidence.toFixed(2)}
                </span>
              )}
            </div>
            {active && (
              <div
                style={{
                  position: 'absolute',
                  bottom: -1,
                  left: 0,
                  right: 0,
                  height: 2,
                  background: E.ember,
                }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

interface ClusterItemListProps {
  cluster: UiCluster | null;
  activeItemId: string | null;
  selectedIds: Set<string>;
  onSelectItem: (id: string) => void;
  onToggleSelected: (id: string) => void;
  onToggleAll: () => void;
}

const VISIBLE_ITEM_LIMIT = 8;

function ClusterItemList({
  cluster,
  activeItemId,
  selectedIds,
  onSelectItem,
  onToggleSelected,
  onToggleAll,
}: ClusterItemListProps) {
  if (!cluster) {
    return (
      <Card style={{ padding: 16 }}>
        <Eyebrow>No cluster selected</Eyebrow>
      </Card>
    );
  }
  const visible = cluster.items.slice(0, VISIBLE_ITEM_LIMIT);
  const remaining = Math.max(0, cluster.items.length - visible.length);
  const allSelected =
    cluster.items.length > 0 &&
    cluster.items.every((it) => selectedIds.has(it.item_id));
  return (
    <Card style={{ padding: 0, overflow: 'hidden', alignSelf: 'flex-start' }}>
      <div
        style={{
          padding: '10px 12px',
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <input
          type="checkbox"
          checked={allSelected}
          onChange={onToggleAll}
          aria-label="Select all in cluster"
          style={{ accentColor: E.ember, cursor: 'pointer' }}
        />
        <Eyebrow>
          {cluster.label} ({cluster.items.length})
        </Eyebrow>
      </div>
      {visible.map((it) => {
        const sel = selectedIds.has(it.item_id);
        const isActive = it.item_id === activeItemId;
        const preview = it.user_text.replace(/\s+/g, ' ').trim();
        return (
          <div
            key={it.item_id}
            onClick={() => onSelectItem(it.item_id)}
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 8,
              padding: '8px 12px',
              borderBottom: `1px solid ${E.hair}`,
              background: isActive ? E.emberDim : 'transparent',
              borderLeft: isActive ? `2px solid ${E.ember}` : '2px solid transparent',
              cursor: 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={sel}
              onChange={(e) => {
                e.stopPropagation();
                onToggleSelected(it.item_id);
              }}
              onClick={(e) => e.stopPropagation()}
              style={{ marginTop: 3, accentColor: E.ember, cursor: 'pointer' }}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontFamily: E.fMono,
                  fontSize: 10,
                  color: E.text3,
                }}
              >
                {it.item_id}
              </div>
              <div
                style={{
                  fontSize: 11.5,
                  color: isActive ? E.text0 : E.text2,
                  lineHeight: 1.45,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {preview}
              </div>
            </div>
          </div>
        );
      })}
      {remaining > 0 && (
        <div
          style={{
            padding: '8px 12px',
            fontFamily: E.fMono,
            fontSize: 10.5,
            color: E.text3,
            textAlign: 'center',
          }}
        >
          ... {remaining} more
        </div>
      )}
    </Card>
  );
}

interface TracePaneProps {
  item: ReviewItem;
  cluster: UiCluster | null;
  selectedCount: number;
  onConfirmPattern: () => void;
  onBatch: (verdict: Verdict) => void;
  onReviewIndividually: () => void;
  onSingle: (verdict: Verdict) => void;
  submitting: boolean;
  note: string;
  onNote: (s: string) => void;
}

function TracePane({
  item,
  cluster,
  selectedCount,
  onConfirmPattern,
  onBatch,
  onReviewIndividually,
  onSingle,
  submitting,
  note,
  onNote,
}: TracePaneProps) {
  const failPills = item.judge_breakdown.filter((b) => b.kind === 'fail');
  // Heuristic tool-call summary: scan judge_reasoning for "tool(name)"
  // patterns. The current ReviewItem shape doesn't carry the raw trace,
  // so this is best-effort and silently skipped if empty.
  const toolCallSummary = useMemo(() => {
    const text = item.judge_reasoning ?? '';
    const matches = text.match(/[a-z_]+\([^)]*\)/g);
    return matches ? matches.slice(0, 2).join(' ') : null;
  }, [item.judge_reasoning]);

  const patternHint = cluster?.pattern_hint;
  const patternCount = cluster?.items.length ?? 0;

  return (
    <>
      {/* Top row: pills + item id ================================ */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 10,
          flexWrap: 'wrap',
        }}
      >
        {failPills.length === 0 ? (
          <Pill mono bg={E.warnDim} color={E.warn}>
            judge uncertain - {item.judge_confidence.toFixed(2)}
          </Pill>
        ) : (
          failPills.slice(0, 3).map((b) => (
            <Pill key={b.label} mono bg={E.failDim} color={pillColor(b.kind)}>
              fail - {b.label} {b.score.toFixed(2)}
            </Pill>
          ))
        )}
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
          {item.item_id} - {item.source_run_label}
        </span>
      </div>

      {/* Side-by-side trace ===================================== */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 10,
          marginBottom: 10,
        }}
      >
        <div>
          <Eyebrow>User message</Eyebrow>
          <div
            style={{
              marginTop: 4,
              padding: 10,
              background: E.panel2,
              borderRadius: 6,
              fontSize: 12,
              color: E.text1,
              lineHeight: 1.55,
            }}
          >
            {renderHighlighted(item.user_text, item.highlights, E.warnDim, E.warn)}
          </div>
          <div style={{ marginTop: 8 }}>
            <Eyebrow>Tool calls{toolCallSummary ? ' (1)' : ' (0)'}</Eyebrow>
            <div
              style={{
                marginTop: 4,
                padding: 8,
                background: E.panel2,
                borderRadius: 6,
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: toolCallSummary ? E.text2 : E.text3,
                lineHeight: 1.6,
                minHeight: 22,
              }}
            >
              {toolCallSummary ?? '(no tool calls captured)'}
            </div>
          </div>
        </div>
        <div>
          <Eyebrow>Assistant reply</Eyebrow>
          <div
            style={{
              marginTop: 4,
              padding: 10,
              background: E.failDim,
              border: `1px solid ${E.fail}33`,
              borderRadius: 6,
              fontSize: 12,
              color: E.text1,
              lineHeight: 1.55,
            }}
          >
            {renderHighlighted(item.agent_response, item.highlights, E.failDim, E.fail)}
          </div>
          <div style={{ marginTop: 8 }}>
            <Eyebrow>Judge says</Eyebrow>
            <div
              style={{
                marginTop: 4,
                padding: 10,
                border: `1px solid ${E.hair2}`,
                borderRadius: 6,
                fontSize: 11,
                color: E.text2,
                lineHeight: 1.55,
                fontFamily: E.fMono,
              }}
            >
              {item.judge_reasoning}
            </div>
          </div>
        </div>
      </div>

      {/* Auto-detected pattern banner =========================== */}
      {patternHint && (
        <div
          style={{
            padding: '8px 12px',
            background: E.warnDim,
            border: `1px solid ${E.warn}33`,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 12,
          }}
        >
          <span style={{ fontSize: 12, color: E.text1, flex: 1 }}>
            <strong>Auto-detected pattern:</strong> {patternHint}. Common across{' '}
            {patternCount} items in this cluster.
          </span>
          <Btn kind="ghost" size="sm" onClick={onConfirmPattern}>
            Confirm pattern
          </Btn>
        </div>
      )}

      {/* Batch verdict bar ====================================== */}
      <div
        data-coachmark="review-verdict-buttons"
        style={{
          padding: 12,
          background: E.panel2,
          borderRadius: 6,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          border: `1px solid ${E.hair2}`,
          flexWrap: 'wrap',
        }}
      >
        <span style={{ fontSize: 12.5, color: E.text1 }}>
          <strong>{selectedCount} selected</strong> - apply same verdict
        </span>
        <span style={{ flex: 1 }} />
        <Btn
          kind="danger"
          size="sm"
          onClick={() => onBatch('fail')}
          disabled={submitting || selectedCount === 0}
        >
          Shift+F Fail all {selectedCount}
        </Btn>
        <Btn
          kind="primary"
          size="sm"
          onClick={() => onBatch('pass')}
          disabled={submitting || selectedCount === 0}
        >
          Shift+P Pass all {selectedCount}
        </Btn>
        <Btn kind="ghost" size="sm" onClick={onReviewIndividually}>
          Review individually
        </Btn>
      </div>

      {/* Single-item verdict row + note ========================= */}
      <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <button
          type="button"
          disabled={submitting}
          onClick={() => onSingle('pass')}
          title="Pass this item (P)"
          style={singleBtnStyle('pass', submitting)}
        >
          <span>Pass</span>
          <span aria-hidden="true" style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>P</span>
        </button>
        <button
          type="button"
          disabled={submitting}
          onClick={() => onSingle('fail')}
          title="Fail this item (F)"
          style={singleBtnStyle('fail', submitting)}
        >
          <span>Fail</span>
          <span aria-hidden="true" style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>F</span>
        </button>
        <button
          type="button"
          disabled={submitting}
          onClick={() => onSingle('skip')}
          title="Skip this item (S)"
          style={singleBtnStyle('skip', submitting)}
        >
          <span>Skip</span>
          <span aria-hidden="true" style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>S</span>
        </button>
      </div>

      {/* Workflow hotkeys that don't fit on a button label. P/F/S
          are visible inside the verdict buttons themselves; this
          line surfaces the navigation/undo keys that previously
          had no on-screen documentation. */}
      <div
        style={{
          marginTop: 8,
          fontSize: 10.5,
          fontFamily: E.fMono,
          color: E.text3,
          display: 'flex',
          gap: 12,
          flexWrap: 'wrap',
        }}
      >
        <span>↵ / N next item</span>
        <span>⌫ undo last verdict</span>
        <span>esc back to home</span>
      </div>

      <div style={{ marginTop: 10 }}>
        <Eyebrow>Note for the team (optional)</Eyebrow>
        <textarea
          value={note}
          onChange={(e) => onNote(e.target.value)}
          aria-label="Note for the team (optional)"
          placeholder="Optional - leave a note about this verdict"
          style={{
            width: '100%',
            marginTop: 6,
            padding: 10,
            background: E.panel2,
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            color: E.text1,
            fontSize: 12.5,
            fontFamily: E.fSans,
            resize: 'vertical',
            minHeight: 50,
            outline: 'none',
          }}
        />
      </div>
    </>
  );
}

function singleBtnStyle(verdict: Verdict, submitting: boolean) {
  const base = {
    flex: verdict === 'fail' ? 2 : 1,
    padding: '12px',
    borderRadius: 6,
    fontSize: 12.5,
    fontWeight: 500 as const,
    cursor: submitting ? 'not-allowed' : 'pointer',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 2,
    opacity: submitting ? 0.6 : 1,
  };
  if (verdict === 'pass') {
    return {
      ...base,
      background: E.passDim,
      border: `1px solid ${E.pass}33`,
      color: E.pass,
    };
  }
  if (verdict === 'fail') {
    return {
      ...base,
      background: E.failDim,
      border: `1px solid ${E.fail}55`,
      color: E.fail,
    };
  }
  return {
    ...base,
    background: E.panel2,
    border: `1px solid ${E.hair2}`,
    color: E.text2,
  };
}

interface CalibrationBoostInfo {
  metrics: string[];
  verdictCount: number;
  threshold: number;
  toGo: number;
}

function CalibrationBoostCard({ info }: { info: CalibrationBoostInfo | null }) {
  if (!info) {
    return (
      <Card style={{ padding: 12 }}>
        <Eyebrow>Calibration boost</Eyebrow>
        <div style={{ marginTop: 8, fontSize: 11.5, color: E.text3 }}>
          No metric breakdown for this item.
        </div>
      </Card>
    );
  }
  return (
    <Card style={{ padding: 12 }}>
      <Eyebrow>Calibration boost</Eyebrow>
      <div
        style={{
          marginTop: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <Donut
          segments={[
            { value: info.verdictCount, color: E.ember },
            { value: Math.max(0, info.threshold - info.verdictCount), color: E.panel3 },
          ]}
          size={42}
          thickness={6}
        />
        <div style={{ fontSize: 11.5, color: E.text2, lineHeight: 1.45 }}>
          Each verdict counts toward
          <br />
          {info.metrics.map((m, i) => (
            <span key={m}>
              <strong style={{ color: E.text0 }}>{m}</strong>
              {i < info.metrics.length - 1 ? ' and ' : ''}
            </span>
          ))}
          .
          <br />
          <span style={{ color: E.ember }}>{info.toGo} to threshold</span>
        </div>
      </div>
    </Card>
  );
}

interface SessionInfo {
  doneInCluster: number;
  totalInCluster: number;
  flatIdx: number;
  total: number;
}

function SessionCard({ info }: { info: SessionInfo | null }) {
  if (!info) {
    return (
      <Card style={{ padding: 12 }}>
        <Eyebrow>Session</Eyebrow>
        <div style={{ marginTop: 8, fontSize: 11.5, color: E.text3 }}>
          No session data.
        </div>
      </Card>
    );
  }
  const remaining = Math.max(0, info.total - info.flatIdx);
  const pct = info.total === 0 ? 0 : (info.flatIdx / Math.max(1, info.total)) * 100;
  return (
    <Card style={{ padding: 12 }}>
      <Eyebrow>Session</Eyebrow>
      <div
        style={{
          marginTop: 8,
          display: 'flex',
          alignItems: 'baseline',
          gap: 8,
        }}
      >
        <span style={{ fontFamily: E.fSerif, fontSize: 24, color: E.text0 }}>
          {info.flatIdx}
        </span>
        <span style={{ fontSize: 11, color: E.text3 }}>of {info.total}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>
          ~{remaining} left
        </span>
      </div>
      <div style={{ marginTop: 6 }}>
        <Bar value={pct} max={100} w="100%" h={4} color={E.ember} />
      </div>
    </Card>
  );
}
