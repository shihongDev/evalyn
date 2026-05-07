/**
 * Annotate landing - calibration cockpit.
 *
 * Reframes Annotate from "pick a source" to "make your judges
 * trustworthy". Per-judge progress rings up top - pick the metric
 * you want to improve. Smart queue ranks items by information gain
 * so each verdict moves the calibration needle as much as possible.
 *
 * Three layers:
 *   1. Per-judge rings (one per metric, kappa vs target).
 *   2. Smart queue (information-gain ranked items for the active
 *      metric) + right rail with in-progress sessions and the
 *      ranking weights that produced this batch.
 *   3. "Why this changed" footer note.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import {
  Bar,
  Btn,
  Card,
  Donut,
  Eyebrow,
  Pill,
  Skeleton,
  StatusDot,
  UpdatingChip,
} from '../ui';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { v2 } from '../api/client';
import { annotationApi } from '../api/annotation';
import { errorMessage } from '../api/errors';
import type {
  AnnotationSessionList,
  AnnotationSessionMeta,
  RubricList,
  RubricRow,
  SmartQueue,
} from '../api/types';
import { preloadAnnotateSession } from '../routePreloads';
import { E } from '../tokens';

// ---------- helpers ------------------------------------------------------

const KAPPA_TARGET = 0.8;
const KAPPA_LOW = 0.65;
// Heuristic: how many additional verdicts we expect each one of these
// items to translate into kappa points. Used purely to give the user a
// rough "~N more to threshold" hint - the real estimator lives backend-
// side once calibrate runs land enough data to fit a curve.
const VERDICTS_PER_KAPPA_POINT = 25;
const SAMPLE_TARGET = 30;

interface JudgeRing {
  /** Backing rubric row - kept for navigation hand-offs. */
  row: RubricRow;
  metricId: string;
  name: string;
  /** Effective kappa (0 if null so the ring renders empty). */
  kappa: number;
  /** True kappa availability - some rubrics have no calibration yet. */
  hasKappa: boolean;
  target: number;
  drift: number;
  sampleSize: number;
  /** Estimated annotations needed to clear the trust threshold. */
  need: number;
  done: boolean;
  low: boolean;
}

function buildJudgeRings(rows: RubricRow[]): JudgeRing[] {
  return rows.map((r) => {
    const kappa = r.kappa ?? 0;
    const hasKappa = r.kappa != null;
    const target = KAPPA_TARGET;
    const done = hasKappa && kappa >= target;
    const low = hasKappa && kappa < KAPPA_LOW;
    let need = 0;
    if (!done) {
      if (!hasKappa || r.sample_size < SAMPLE_TARGET) {
        // Not enough verdicts yet to even fit a kappa - show the
        // sample-size gap so the user knows roughly how many they need.
        need = Math.max(SAMPLE_TARGET - r.sample_size, 5);
      } else {
        const gap = Math.max(0, target - kappa);
        need = Math.max(1, Math.ceil(gap * VERDICTS_PER_KAPPA_POINT));
      }
    }
    return {
      row: r,
      metricId: r.id,
      name: r.name,
      kappa,
      hasKappa,
      target,
      drift: r.drift_per_week,
      sampleSize: r.sample_size,
      need,
      done,
      low,
    };
  });
}

function pickDefaultMetric(rings: JudgeRing[]): string | null {
  if (rings.length === 0) return null;
  // Lowest-kappa metric most needs work, so default the queue to it.
  // Treat "missing kappa" as 0 so brand-new metrics float to the top.
  const sorted = [...rings].sort((a, b) => a.kappa - b.kappa);
  return sorted[0].metricId;
}

// ---------- per-judge ring -----------------------------------------------

function JudgeProgressRing({
  ring,
  active,
  onSelect,
}: {
  ring: JudgeRing;
  active: boolean;
  onSelect: () => void;
}) {
  const ringColor = ring.done ? E.pass : ring.low ? E.fail : E.ember;
  const borderColor = active
    ? E.emberRim
    : ring.done
      ? E.pass + '33'
      : ring.low
        ? E.fail + '33'
        : E.hair;
  return (
    <Card
      hover
      aria-label={`Judge ${ring.name}: ${
        ring.done
          ? 'trusted'
          : ring.low
            ? 'needs work'
            : `${ring.need} more annotations to threshold`
      }. Click to select.`}
      aria-pressed={active}
      style={{
        padding: 14,
        borderColor,
        cursor: 'pointer',
        outline: active ? `1px solid ${E.emberRim}` : 'none',
      }}
      onClick={onSelect}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
        <Donut
          segments={[
            { value: ring.kappa, color: ringColor },
            { value: Math.max(0, 1 - ring.kappa), color: E.panel3 },
          ]}
          size={56}
          thickness={7}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            title={ring.name}
            style={{
              fontSize: 12.5,
              color: E.text0,
              fontWeight: 500,
              marginBottom: 2,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {ring.name}
          </div>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: ring.done ? E.pass : E.text2,
              marginBottom: 6,
            }}
          >
            {ring.hasKappa
              ? `k ${ring.kappa.toFixed(2)} -> ${ring.target.toFixed(2)}`
              : `k -.-- -> ${ring.target.toFixed(2)}`}
          </div>
          {ring.done ? (
            <Pill bg={E.passDim} color={E.pass} mono>
              trusted
            </Pill>
          ) : ring.low ? (
            <Pill bg={E.failDim} color={E.fail} mono>
              needs work
            </Pill>
          ) : (
            <span
              style={{
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.ember,
              }}
            >
              ~{ring.need} more to threshold
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}

// ---------- smart queue card ---------------------------------------------

function SmartQueueRow({
  rank,
  score,
  judgeScore,
  text,
  why,
  onOpen,
  divider,
}: {
  rank: number;
  score: number;
  judgeScore: number;
  text: string;
  why: string;
  onOpen: () => void;
  divider: boolean;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '40px 1fr 88px',
        alignItems: 'center',
        gap: 12,
        padding: '10px 16px',
        borderBottom: divider ? `1px solid ${E.hair}` : 'none',
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: E.fSerif, fontSize: 18, color: E.text0, lineHeight: 1 }}>
          {rank}
        </div>
        <div style={{ fontFamily: E.fMono, fontSize: 9, color: E.text3, marginTop: 2 }}>
          * {score.toFixed(2)}
        </div>
      </div>
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontSize: 12.5,
            color: E.text1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={text}
        >
          "{text}"
        </div>
        <div
          title={`judge says ${judgeScore.toFixed(2)} - ${why}`}
          style={{
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.text3,
            marginTop: 3,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          judge says {judgeScore.toFixed(2)} - {why}
        </div>
      </div>
      <Btn kind="secondary" size="sm" onClick={onOpen} style={{ justifyContent: 'center' }}>
        Open
      </Btn>
    </div>
  );
}

function SmartQueueCard({
  metricId,
  metricLabel,
  queue,
  loading,
  err,
  onRetry,
  onStartSession,
  onOpenItem,
  onCustomizeRanking,
  showRankingHint,
  starting,
  rings,
  onPickAnother,
}: {
  metricId: string | null;
  metricLabel: string;
  queue: SmartQueue | null;
  loading: boolean;
  err: string | null;
  onRetry: () => void;
  onStartSession: () => void;
  onOpenItem: (itemId: string) => void;
  onCustomizeRanking: () => void;
  showRankingHint: boolean;
  starting: boolean;
  rings: JudgeRing[];
  onPickAnother: (metricId: string) => void;
}) {
  const items = queue?.items ?? [];
  const itemCount = items.length;
  const estToThreshold = queue?.est_to_threshold ?? null;
  // ~30 seconds per item is the design's anchor for "how long is this
  // session". Adjust as we collect telemetry.
  const estMinutes = estToThreshold != null ? Math.max(1, Math.round(estToThreshold * 0.5)) : null;

  return (
    <Card style={{ padding: 0, overflow: 'hidden' }}>
      <div
        style={{
          padding: '12px 16px',
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexWrap: 'wrap',
        }}
      >
        <Eyebrow>Smart queue</Eyebrow>
        {metricId && (
          <Pill bg={E.emberDim} color={E.ember} mono>
            {metricLabel} - {itemCount} items
          </Pill>
        )}
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
          ranked by information gain
        </span>
      </div>

      {loading && (
        <div style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Skeleton w={28} h={28} />
              <div style={{ flex: 1 }}>
                <Skeleton w="80%" h={11} />
                <div style={{ marginTop: 6 }}>
                  <Skeleton w="40%" h={10} />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && err && (
        <div style={{ padding: 18 }}>
          <div
            style={{
              padding: 12,
              border: `1px solid ${E.fail}33`,
              background: E.failDim,
              borderRadius: 8,
              fontSize: 12,
              color: E.fail,
              fontFamily: E.fMono,
              marginBottom: 10,
            }}
          >
            {err}
          </div>
          <Btn kind="secondary" size="sm" onClick={onRetry}>
            Retry
          </Btn>
        </div>
      )}

      {!loading && !err && metricId == null && (
        <div style={{ padding: 22, fontSize: 12.5, color: E.text3 }}>
          Pick a judge above to see ranked items.
        </div>
      )}

      {!loading && !err && metricId != null && itemCount === 0 && (
        <div style={{ padding: 22 }}>
          <div style={{ fontSize: 13, color: E.text1, fontWeight: 500, marginBottom: 6 }}>
            All caught up
          </div>
          <div style={{ fontSize: 12, color: E.text3, lineHeight: 1.55, marginBottom: 12 }}>
            No items need annotation right now for {metricLabel}. Pick another judge to keep
            calibrating.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {rings
              .filter((r) => r.metricId !== metricId && !r.done)
              .slice(0, 4)
              .map((r) => (
                <Btn
                  key={r.metricId}
                  kind="secondary"
                  size="sm"
                  onClick={() => onPickAnother(r.metricId)}
                >
                  {r.name}
                </Btn>
              ))}
          </div>
        </div>
      )}

      {!loading && !err && itemCount > 0 &&
        items.map((it, i) => (
          <SmartQueueRow
            key={`${it.run_id}:${it.item_id}:${i}`}
            rank={it.rank}
            score={it.score}
            judgeScore={it.judge_score}
            text={it.text}
            why={it.why}
            divider={i < items.length - 1}
            onOpen={() => onOpenItem(it.item_id)}
          />
        ))}

      <div
        style={{
          padding: '12px 16px',
          background: E.panel2,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          borderTop: `1px solid ${E.hair}`,
          flexWrap: 'wrap',
        }}
      >
        <Btn
          kind="primary"
          size="sm"
          onClick={onStartSession}
          disabled={starting || metricId == null || itemCount === 0}
          onMouseEnter={() => void preloadAnnotateSession()}
          onFocus={() => void preloadAnnotateSession()}
        >
          {starting ? 'Starting...' : 'Start session'}
        </Btn>
        <div style={{ position: 'relative' }}>
          <Btn kind="ghost" size="sm" onClick={onCustomizeRanking}>
            Customize ranking
          </Btn>
          {showRankingHint && (
            <div
              role="tooltip"
              style={{
                position: 'absolute',
                top: 'calc(100% + 6px)',
                left: 0,
                background: E.panel,
                border: `1px solid ${E.hair2}`,
                borderRadius: 6,
                padding: '6px 10px',
                fontSize: 11,
                color: E.text2,
                fontFamily: E.fMono,
                whiteSpace: 'nowrap',
                boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                zIndex: 2,
              }}
            >
              Coming soon
            </div>
          )}
        </div>
        <span style={{ flex: 1 }} />
        {estToThreshold != null && (
          <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
            est ~{estMinutes} min - ~{estToThreshold} items to k {'>='} 0.8
          </span>
        )}
      </div>
    </Card>
  );
}

// ---------- in-progress sessions sidebar ---------------------------------

function InProgressSessions({
  sessions,
  loading,
  onResume,
}: {
  sessions: AnnotationSessionMeta[];
  loading: boolean;
  onResume: (id: string) => void;
}) {
  return (
    <Card style={{ padding: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Eyebrow>In progress</Eyebrow>
        <Pill mono>{sessions.length}</Pill>
      </div>
      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {loading && sessions.length === 0 && (
          <>
            <Skeleton w="100%" h={36} />
            <Skeleton w="100%" h={36} />
          </>
        )}
        {!loading && sessions.length === 0 && (
          <div style={{ fontSize: 11.5, color: E.text3, lineHeight: 1.55 }}>
            No sessions in progress. Start one from the smart queue.
          </div>
        )}
        {sessions.map((s) => {
          const pct =
            s.items_total > 0
              ? Math.min(100, Math.round((s.items_done / s.items_total) * 100))
              : 0;
          const label = s.metric_ids[0] ?? s.source_id;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => onResume(s.id)}
              onMouseEnter={() => {
                void preloadAnnotateSession();
                // Warm the session-detail JSON + items list (200
                // items per page, the heavy fetch on the route)
                // so resume click->paint is instant. Cache keys
                // mirror what AnnotateSession's useV2Resource
                // calls use, so prefetchV2 dedupes correctly.
                prefetchV2(`annotation/session:${s.id}`, () =>
                  annotationApi.getSession(s.id),
                );
                prefetchV2(`annotation/items:${s.id}`, () =>
                  annotationApi.getItems(s.id, { limit: 200 }),
                );
              }}
              onFocus={() => {
                void preloadAnnotateSession();
                prefetchV2(`annotation/session:${s.id}`, () =>
                  annotationApi.getSession(s.id),
                );
                prefetchV2(`annotation/items:${s.id}`, () =>
                  annotationApi.getItems(s.id, { limit: 200 }),
                );
              }}
              title={`Resume - ${pct}% complete`}
              aria-label={`Resume ${label}: ${s.items_done} of ${s.items_total} items annotated, ${pct}% complete`}
              style={{
                padding: 10,
                background: E.panel2,
                borderRadius: 6,
                border: `1px solid ${E.hair2}`,
                textAlign: 'left',
                cursor: 'pointer',
                font: 'inherit',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginBottom: 6,
                }}
              >
                <StatusDot status="running" size={6} />
                <span
                  style={{
                    fontSize: 12,
                    color: E.text0,
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    minWidth: 0,
                    flex: 1,
                  }}
                  title={label}
                >
                  {label}
                </span>
                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 10,
                    color: E.text3,
                  }}
                >
                  {s.items_done}/{s.items_total}
                </span>
              </div>
              <Bar value={pct} w="100%" h={3} color={E.ember} />
            </button>
          );
        })}
      </div>
    </Card>
  );
}

// ---------- ranking weights card -----------------------------------------

function RankingWeights({ queue }: { queue: SmartQueue | null }) {
  // Fallback weights mirror the SmartQueue contract so the card still
  // renders before the queue lands or when picking a brand-new metric.
  const fallback = { uncertainty: 0.5, disagreement: 0.3, coverage: 0.15, random: 0.05 };
  const weights = queue?.weights ?? fallback;
  const rows: { label: string; value: number; muted?: boolean }[] = [
    { label: 'uncertainty', value: weights.uncertainty },
    { label: 'judge <-> judge', value: weights.disagreement },
    { label: 'coverage gaps', value: weights.coverage },
    { label: 'random fill', value: weights.random, muted: true },
  ];
  return (
    <Card style={{ padding: 14 }}>
      <Eyebrow>Why these items?</Eyebrow>
      <div
        style={{
          marginTop: 8,
          fontFamily: E.fMono,
          fontSize: 10.5,
          color: E.text2,
          lineHeight: 1.7,
        }}
      >
        {rows.map((r) => (
          <div
            key={r.label}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          >
            <span>{r.label}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Bar value={r.value * 100} max={100} w={48} h={3} color={r.muted ? E.text3 : E.ember} />
              <span style={{ color: r.muted ? E.text3 : E.ember }}>
                {Math.round(r.value * 100)}%
              </span>
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ---------- empty state --------------------------------------------------

function NoRubricsEmpty({ onCreate }: { onCreate: () => void }) {
  return (
    <Card style={{ padding: 28, marginTop: 24 }}>
      <Eyebrow>No judges yet</Eyebrow>
      <div
        style={{
          fontFamily: E.fSerif,
          fontSize: 22,
          color: E.text0,
          marginTop: 6,
          marginBottom: 10,
        }}
      >
        You need a rubric before you can calibrate.
      </div>
      <div style={{ fontSize: 13, color: E.text2, lineHeight: 1.55, marginBottom: 14, maxWidth: 540 }}>
        A rubric defines what "good" looks like for one judge. Once you have one and a
        handful of annotations, this page surfaces the items that move the calibration
        needle most.
      </div>
      <Btn kind="primary" size="md" onClick={onCreate}>
        Create your first rubric
      </Btn>
    </Card>
  );
}

// ---------- main page ----------------------------------------------------

export default function Annotate() {
  const navigate = useNavigate();

  const {
    data: sessionList,
    err: sessionErr,
    refetch: refetchSessions,
    reloading: sessionReloading,
    isInitialLoad: sessionInitial,
  } = useV2Resource<AnnotationSessionList>('annotation/sessions', () =>
    annotationApi.listSessions(),
  );

  const {
    data: rubrics,
    err: rubricsErr,
    refetch: refetchRubrics,
    reloading: rubricsReloading,
    isInitialLoad: rubricsInitial,
  } = useV2Resource<RubricList>('rubrics', v2.rubrics);

  const rings = useMemo(() => buildJudgeRings(rubrics ?? []), [rubrics]);

  // Selection lives in URL (?metric=quality) for shareable links.
  // Pattern matches Metrics' ?metric=, Reports ?audience=, RunDetail
  // ?tab=, Commands ?cmd=. The default-pick effect below writes
  // ?metric=<lowest-kappa> on mount when no URL param is set, so
  // /annotate is always shareable from arrival forward.
  const [searchParams, setSearchParams] = useSearchParams();
  const activeMetricId = searchParams.get('metric');
  const setActiveMetricId = useCallback(
    (id: string | null) => {
      setSearchParams(
        (prev) => {
          const u = new URLSearchParams(prev);
          if (id) u.set('metric', id);
          else u.delete('metric');
          return u;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  // Default the active metric to the lowest-kappa ring once rubrics
  // land. Keep an explicit user pick across rubric refetches by only
  // setting when the current pick is missing/invalid.
  useEffect(() => {
    if (!rings.length) {
      if (activeMetricId !== null) setActiveMetricId(null);
      return;
    }
    const stillExists = activeMetricId && rings.some((r) => r.metricId === activeMetricId);
    if (!stillExists) {
      setActiveMetricId(pickDefaultMetric(rings));
    }
  }, [rings, activeMetricId, setActiveMetricId]);

  // Smart queue, parameterized by the active metric. Cache key includes
  // the metric so switching metrics doesn't flash old data.
  const queueKey = activeMetricId
    ? `annotation/smart-queue:${activeMetricId}`
    : 'annotation/smart-queue:none';
  const {
    data: queue,
    err: queueErr,
    refetch: refetchQueue,
    isInitialLoad: queueInitial,
  } = useV2Resource<SmartQueue>(
    queueKey,
    () => v2.annotationSmartQueue(activeMetricId ?? undefined, 5),
    { enabled: !!activeMetricId },
  );

  const sessions = sessionList?.sessions ?? [];
  const inProgress = sessions.filter((s) => s.status === 'in_progress');

  // Active metric label (for pill + estimate copy). Falls back to the
  // raw id when the rubric list is still loading.
  const activeRing = rings.find((r) => r.metricId === activeMetricId) ?? null;
  const activeLabel = activeRing?.name ?? activeMetricId ?? '';

  const [starting, setStarting] = useState(false);
  const [startErr, setStartErr] = useState<string | null>(null);
  const [showRankingHint, setShowRankingHint] = useState(false);

  // Find or create a session for the active metric; return its id.
  // Reusing an in-progress session keeps the user on a single thread
  // for that metric instead of fragmenting work across many.
  async function ensureSession(): Promise<string | null> {
    if (!activeMetricId) return null;
    const existing = inProgress.find(
      (s) => s.metric_ids.includes(activeMetricId) && s.source_kind === 'custom',
    );
    if (existing) return existing.id;
    try {
      const session = await annotationApi.createSession({
        source_kind: 'custom',
        source_id: activeMetricId,
        metric_ids: [activeMetricId],
      });
      void refetchSessions();
      return session.id;
    } catch (e) {
      setStartErr(errorMessage(e));
      return null;
    }
  }

  async function handleStartSession() {
    if (!activeMetricId || starting) return;
    setStarting(true);
    setStartErr(null);
    try {
      const id = await ensureSession();
      if (id) navigate(`/annotate/${encodeURIComponent(id)}`);
    } finally {
      setStarting(false);
    }
  }

  async function handleOpenItem(itemId: string) {
    if (!activeMetricId) return;
    setStartErr(null);
    const id = await ensureSession();
    if (id) {
      navigate(
        `/annotate/${encodeURIComponent(id)}?item=${encodeURIComponent(itemId)}`,
      );
    }
  }

  function handleCustomizeRanking() {
    setShowRankingHint(true);
    window.setTimeout(() => setShowRankingHint(false), 1800);
  }

  const showInitialSkeleton = rubricsInitial && !rubrics;
  const showRubricsError = !!rubricsErr && !rubrics;

  return (
    <AppShell>
      <div style={{ padding: '24px 24px 40px', maxWidth: 1200 }}>
        {/* Header: eyebrow + updating chip */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <Eyebrow>Annotate - calibration cockpit</Eyebrow>
          <UpdatingChip
            visible={
              (sessionReloading && !sessionInitial) || (rubricsReloading && !rubricsInitial)
            }
            error={(sessionList ? sessionErr : null) || (rubrics ? rubricsErr : null)}
            onRetry={() => {
              void refetchSessions();
              void refetchRubrics();
            }}
          />
        </div>

        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 38,
            fontWeight: 400,
            margin: 0,
            color: E.text0,
            letterSpacing: '-0.015em',
            lineHeight: 1.05,
          }}
        >
          Make your judges trustworthy
        </h1>
        <p
          style={{
            fontSize: 13.5,
            color: E.text2,
            marginTop: 8,
            marginBottom: 0,
            lineHeight: 1.55,
            maxWidth: 640,
          }}
        >
          Each verdict you give the smart queue moves a judge closer to its kappa target.
          Pick the judge that needs the most help, then run the ranked items below.
        </p>

        {/* Hard rubric load error - show retry but no ring grid. */}
        {showRubricsError && (
          <Card style={{ marginTop: 22, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Could not load judges</Eyebrow>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.text2,
                marginTop: 6,
              }}
            >
              {rubricsErr}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={refetchRubrics}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        {/* Per-judge progress rings */}
        {!showRubricsError && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12,
              marginTop: 18,
              marginBottom: 14,
            }}
          >
            {showInitialSkeleton &&
              Array.from({ length: 4 }).map((_, i) => (
                <Card key={i} style={{ padding: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                    <Skeleton w={56} h={56} style={{ borderRadius: '50%' }} />
                    <div style={{ flex: 1 }}>
                      <Skeleton w="60%" h={12} />
                      <div style={{ marginTop: 6 }}>
                        <Skeleton w="80%" h={10} />
                      </div>
                      <div style={{ marginTop: 10 }}>
                        <Skeleton w="50%" h={14} />
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            {!showInitialSkeleton &&
              rings.map((r) => (
                <JudgeProgressRing
                  key={r.metricId}
                  ring={r}
                  active={r.metricId === activeMetricId}
                  onSelect={() => setActiveMetricId(r.metricId)}
                />
              ))}
          </div>
        )}

        {/* Empty state - no rubrics at all */}
        {!showRubricsError && !showInitialSkeleton && rings.length === 0 && (
          <NoRubricsEmpty onCreate={() => navigate('/metrics')} />
        )}

        {/* Smart queue + sidebar - only when at least one judge exists. */}
        {!showRubricsError && rings.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 280px',
              gap: 14,
            }}
          >
            <SmartQueueCard
              metricId={activeMetricId}
              metricLabel={activeLabel}
              queue={queue}
              loading={queueInitial && !queue && !!activeMetricId}
              err={queue ? null : queueErr}
              onRetry={refetchQueue}
              onStartSession={handleStartSession}
              onOpenItem={handleOpenItem}
              onCustomizeRanking={handleCustomizeRanking}
              showRankingHint={showRankingHint}
              starting={starting}
              rings={rings}
              onPickAnother={(id) => setActiveMetricId(id)}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <InProgressSessions
                sessions={inProgress}
                loading={sessionInitial && !sessionList}
                onResume={(id) => navigate(`/annotate/${encodeURIComponent(id)}`)}
              />
              <RankingWeights queue={queue} />
            </div>
          </div>
        )}

        {/* Error toasts for ad-hoc actions (createSession failures). */}
        {startErr && (
          <Card style={{ marginTop: 14, padding: 12, borderColor: E.fail }}>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.fail,
              }}
            >
              Could not start session: {startErr}
            </div>
          </Card>
        )}

        {/* "why this changed" footer */}
        {!showRubricsError && rings.length > 0 && (
          <div
            style={{
              marginTop: 16,
              padding: '12px 16px',
              background: E.panel,
              border: `1px dashed ${E.hair2}`,
              borderRadius: 8,
              fontSize: 11.5,
              color: E.text2,
              lineHeight: 1.6,
              maxWidth: 900,
            }}
          >
            <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
              why this changed -&gt;
            </span>{' '}
            Annotation is most expensive when items are picked at random. Active learning
            surfaces the items where one human judgment moves the needle most. The cockpit
            reframes the activity from "session 088" to "make {activeLabel || 'this judge'}{' '}
            trustworthy".
          </div>
        )}
      </div>
    </AppShell>
  );
}
