/**
 * Home - triage cockpit. Conversational brief + pulse strip + today's
 * queue + live streaming + pinned dimensions. Sources from v2.home().
 *
 * The proposal (HomeAfter in /tmp/evalyn-proposal/tab-home.jsx) reverses
 * the prior "passive snapshot" layout: the action queue leads, the brief
 * is conversational, and the pulse strip is glanceable. Anything that
 * needs a number-plus-context gets a sparkline inline.
 */

import { useMemo } from 'react';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { useRouteTour } from '../tour/useRouteTour';
import { FIRST_RUN_TOUR_ID } from '../tour/scripts/firstRun';
import {
  Card,
  Eyebrow,
  Pill,
  Btn,
  Spark,
  Bar,
  Skeleton,
  Spinner,
  UpdatingChip,
} from '../ui';
import { Welcome } from '../ui/Welcome';
import { v2 } from '../api/client';
import { useV2Resource, prefetchV2 } from '../hooks/useV2Resource';
import { preloadRunDetail } from '../routePreloads';
import { E } from '../tokens';
import type { HomeSnapshot } from '../api/types';

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

type SeverityKind = 'fail' | 'warn' | 'info';
type PulseKind = 'pass' | 'fail' | 'warn' | 'info';

function shortTime(iso: string): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const diffMs = Date.now() - then;
  const m = Math.floor(diffMs / 60_000);
  if (m < 1) return 'now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function priorityForSeverity(sev: SeverityKind): 'P0' | 'P1' | 'P2' {
  if (sev === 'fail') return 'P0';
  if (sev === 'warn') return 'P1';
  return 'P2';
}

function pulseColor(kind: PulseKind): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  if (kind === 'warn') return E.warn;
  return E.steel;
}

function fmtUsd(n: number): string {
  if (n >= 100) return `$${Math.round(n)}`;
  return `$${n.toFixed(2)}`;
}

/* ------------------------------------------------------------------ */
/* derived snapshot summaries (defensive accessors)                   */
/* ------------------------------------------------------------------ */

interface PulseCellData {
  label: string;
  value: string;
  delta: string;
  kind: PulseKind;
  spark: number[];
  onClick?: () => void;
}

interface BriefHighlights {
  topAttention: HomeSnapshot['attention'][number] | null;
  topClusterAttention: HomeSnapshot['attention'][number] | null;
  topClusterRunId: string | null;
  topClusterId: string | null;
  /** A reasonable "compare A <-> B" pair (latest two completed runs). */
  compareA: string | null;
  compareB: string | null;
}

/** Parse `/experiments/<id>/cluster/<cid>` if present. */
function parseClusterTarget(target: string): { runId: string; clusterId: string } | null {
  const m = /^\/experiments\/([^/]+)\/cluster\/([^/?#]+)/.exec(target);
  if (!m) return null;
  return { runId: decodeURIComponent(m[1]), clusterId: decodeURIComponent(m[2]) };
}

function deriveBriefHighlights(snap: HomeSnapshot): BriefHighlights {
  const failAttention = snap.attention.find((a) => a.severity === 'fail') ?? null;
  const topAttention = failAttention ?? snap.attention[0] ?? null;

  let topClusterAttention: BriefHighlights['topClusterAttention'] = null;
  let topClusterRunId: string | null = null;
  let topClusterId: string | null = null;
  for (const a of snap.attention) {
    const parsed = parseClusterTarget(a.cta_target);
    if (parsed) {
      topClusterAttention = a;
      topClusterRunId = parsed.runId;
      topClusterId = parsed.clusterId;
      break;
    }
  }

  // Latest two completed runs (active_experiments is "top 3 most recent");
  // pick the two most recent ids that are not the running one.
  const completed = snap.active_experiments.filter((e) => e.status !== 'running');
  const compareA = completed[0]?.id ?? null;
  const compareB = completed[1]?.id ?? null;

  return {
    topAttention,
    topClusterAttention,
    topClusterRunId,
    topClusterId,
    compareA,
    compareB,
  };
}

/* ------------------------------------------------------------------ */
/* Brief header                                                       */
/* ------------------------------------------------------------------ */

interface BriefHeaderProps {
  snap: HomeSnapshot;
  reloading: boolean;
  onRegenerate: () => void;
  onNavigate: (path: string) => void;
}

function BriefHeader({ snap, reloading, onRegenerate, onNavigate }: BriefHeaderProps) {
  const briefTime = useMemo(() => {
    if (!snap.brief) return '';
    return shortTime(snap.brief.generated_at_iso);
  }, [snap.brief]);
  const highlights = useMemo(() => deriveBriefHighlights(snap), [snap]);

  // Pick canonical CTAs. Backend supplies brief.actions but we also build
  // typed fallbacks so the cluster + compare buttons always render when the
  // underlying data is present.
  const clusterCta = highlights.topClusterAttention;
  const clusterRunId = highlights.topClusterRunId;
  const clusterId = highlights.topClusterId;
  const compareA = highlights.compareA;
  const compareB = highlights.compareB;

  // Backend-supplied "Open run" action (primary) is also a useful target
  // for a single-button case where there is no cluster.
  const fallbackPrimary = snap.brief?.actions.find((a) => a.kind === 'primary') ?? null;

  const clusterLabel = clusterCta
    ? clusterCta.title.length > 38
      ? clusterCta.title.slice(0, 36) + '...'
      : clusterCta.title
    : null;

  return (
    <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
      <div
        style={{
          width: 38,
          height: 38,
          borderRadius: 8,
          background: E.emberDim,
          border: `1px solid ${E.emberRim}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: E.ember,
          fontFamily: E.fSerif,
          fontSize: 18,
          flexShrink: 0,
          position: 'relative',
        }}
        aria-hidden
      >
        e
        <span
          style={{
            position: 'absolute',
            right: -2,
            bottom: -2,
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: E.ember,
            border: `2px solid ${E.ink}`,
            animation: 'eDotPulse 1.6s ease-in-out infinite',
          }}
        />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <Eyebrow>{snap.brief ? `Brief - ${briefTime || 'just now'}` : 'Brief'}</Eyebrow>
          {snap.brief && (
            <Pill bg={E.passDim} color={E.pass} mono>
              fresh
            </Pill>
          )}
        </div>
        {snap.brief ? (
          <BriefProse snap={snap} highlights={highlights} />
        ) : (
          <p
            style={{
              fontFamily: E.fSerif,
              fontSize: 22,
              color: E.text2,
              lineHeight: 1.35,
              margin: 0,
              maxWidth: 820,
              letterSpacing: '-0.005em',
            }}
          >
            No brief yet. Run an evaluation or load the demo to generate the morning brief.
          </p>
        )}
        <div
          style={{ display: 'flex', gap: 6, marginTop: 12, flexWrap: 'wrap' }}
          data-coachmark="home-copilot-brief"
        >
          {clusterCta && clusterRunId && clusterId ? (
            <Btn
              kind="primary"
              size="sm"
              onClick={() =>
                onNavigate(
                  `/experiments/${encodeURIComponent(clusterRunId)}/cluster/${encodeURIComponent(clusterId)}`,
                )
              }
            >
              {`Open ${clusterLabel ?? 'top failure cluster'} ->`}
            </Btn>
          ) : fallbackPrimary ? (
            <Btn kind="primary" size="sm" onClick={() => onNavigate(fallbackPrimary.intent)}>
              {`${fallbackPrimary.label} ->`}
            </Btn>
          ) : null}
          {compareA && compareB && (
            <Btn
              kind="secondary"
              size="sm"
              onClick={() =>
                onNavigate(
                  `/experiments/${encodeURIComponent(compareA)}?compare=${encodeURIComponent(compareB)}`,
                )
              }
              title={`Compare ${compareA} vs ${compareB}`}
            >
              {`Compare ${compareA} <-> ${compareB}`}
            </Btn>
          )}
          <Btn
            kind="ghost"
            size="sm"
            onClick={onRegenerate}
            disabled={reloading}
            title={reloading ? 'Refreshing...' : 'Regenerate the morning brief'}
          >
            {reloading ? <Spinner size={11} /> : '↻'} Regenerate brief
          </Btn>
        </div>
      </div>
    </div>
  );
}

/**
 * BriefProse - render the brief as a serif paragraph composed from the
 * snapshot's structured fields, with inline emphasis on the quality figure
 * and the largest negative sub-metric. Backend body_md trails as a
 * supporting paragraph in case it carries extra context.
 */
function BriefProse({
  snap,
  highlights,
}: {
  snap: HomeSnapshot;
  highlights: BriefHighlights;
}) {
  const body = snap.brief?.body_md ?? '';
  const q = snap.quality;
  const qStr = q.current != null ? q.current.toFixed(1) : null;
  const qDelta = q.delta_30d;

  // Find the first negative sub-metric for the fail-tinted highlight
  const negSub = snap.sub_metrics.find((m) => {
    const goodDirection = m.inverse ? m.delta < 0 : m.delta > 0;
    return !goodDirection && m.delta !== 0;
  });

  const subline = highlights.topClusterAttention
    ? highlights.topClusterAttention.title
    : highlights.topAttention?.title ?? null;

  if (qStr) {
    return (
      <p
        style={{
          fontFamily: E.fSerif,
          fontSize: 22,
          color: E.text0,
          lineHeight: 1.35,
          margin: 0,
          maxWidth: 820,
          letterSpacing: '-0.005em',
        }}
      >
        <span style={{ color: E.text2 }}>Good morning.</span>{' '}
        Quality is{' '}
        <span
          style={{
            borderBottom: `2px solid ${qDelta != null && qDelta < 0 ? E.fail : E.pass}`,
          }}
        >
          {qDelta != null && qDelta >= 0 ? 'holding at' : 'down to'} {qStr}
          {qDelta != null && (
            <>
              {' '}
              <span style={{ color: E.text3, fontFamily: E.fSans, fontSize: 14 }}>
                ({qDelta >= 0 ? '+' : ''}
                {qDelta.toFixed(1)} pts vs. 30d)
              </span>
            </>
          )}
        </span>
        {negSub && (
          <>
            {' '}
            but{' '}
            <span
              style={{
                background: E.failDim,
                color: E.fail,
                padding: '0 4px',
                borderRadius: 3,
              }}
            >
              {negSub.label.toLowerCase()} dropped {Math.abs(negSub.delta).toFixed(1)} pts
            </span>
          </>
        )}
        .
        {subline && (
          <>
            {' '}
            <span style={{ color: E.text2 }}>{subline}</span>
            {highlights.topClusterAttention && (
              <span style={{ color: E.text3 }}> - worth a look first.</span>
            )}
          </>
        )}
        {body && (
          <span
            style={{
              display: 'block',
              fontFamily: E.fSans,
              fontSize: 13,
              color: E.text2,
              lineHeight: 1.55,
              marginTop: 10,
              maxWidth: 760,
            }}
          >
            {body}
          </span>
        )}
      </p>
    );
  }

  // Fallback: no quality number, just print the brief body.
  return (
    <p
      style={{
        fontFamily: E.fSerif,
        fontSize: 22,
        color: E.text0,
        lineHeight: 1.35,
        margin: 0,
        maxWidth: 820,
        letterSpacing: '-0.005em',
      }}
    >
      {body || 'No quality data yet.'}
    </p>
  );
}

/* ------------------------------------------------------------------ */
/* Pulse strip                                                        */
/* ------------------------------------------------------------------ */

interface PulseStripProps {
  cells: PulseCellData[];
}

function PulseStrip({ cells }: PulseStripProps) {
  return (
    <div
      style={{
        marginTop: 22,
        background: E.panel,
        border: `1px solid ${E.hair}`,
        borderRadius: 10,
        padding: '14px 4px',
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
      }}
      data-coachmark="home-quality"
    >
      {cells.map((c, i) => (
        <PulseCell key={c.label} cell={c} hasDivider={i < cells.length - 1} />
      ))}
    </div>
  );
}

function PulseCell({ cell, hasDivider }: { cell: PulseCellData; hasDivider: boolean }) {
  const color = pulseColor(cell.kind);
  const interactive = cell.onClick != null;
  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!interactive) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      cell.onClick?.();
    }
  };
  return (
    <div
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={cell.onClick}
      onKeyDown={onKey}
      style={{
        padding: '0 18px',
        borderRight: hasDivider ? `1px solid ${E.hair}` : 'none',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        cursor: interactive ? 'pointer' : 'default',
      }}
    >
      <Eyebrow>{cell.label}</Eyebrow>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span
          style={{
            fontFamily: E.fSerif,
            fontSize: 28,
            color: E.text0,
            lineHeight: 1,
            fontWeight: 400,
          }}
        >
          {cell.value}
        </span>
        <span style={{ fontFamily: E.fMono, fontSize: 11, color }}>{cell.delta}</span>
      </div>
      {cell.spark.length > 1 ? (
        <Spark data={cell.spark} w={170} h={26} color={color} />
      ) : (
        <div style={{ height: 26 }} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Today's queue                                                      */
/* ------------------------------------------------------------------ */

interface QueueRowProps {
  item: HomeSnapshot['attention'][number];
  isLast: boolean;
  onNavigate: (path: string) => void;
}

function QueueRow({ item, isLast, onNavigate }: QueueRowProps) {
  const prio = priorityForSeverity(item.severity);
  const bg = item.severity === 'fail' ? E.failDim : item.severity === 'warn' ? E.warnDim : E.panel2;
  const color = item.severity === 'fail' ? E.fail : item.severity === 'warn' ? E.warn : E.text2;
  const actionLabel = item.cta || 'Open';

  return (
    <div
      role="button"
      tabIndex={0}
      // Compose a single SR-friendly summary so the row reads as one
      // action ("Priority P0: <title>, <subtitle>, Open") rather than
      // a sequence of orphan strings followed by a duplicate button.
      aria-label={`Priority ${prio}: ${item.title}${item.subtitle ? `, ${item.subtitle}` : ''}, ${actionLabel}`}
      // title= mirrors the visible truncated text so sighted users
      // can read a long title or subtitle that's been ellipsized in
      // the row's narrow middle column.
      title={`${item.title}${item.subtitle ? `\n${item.subtitle}` : ''}`}
      onClick={() => onNavigate(item.cta_target)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onNavigate(item.cta_target);
        }
      }}
      onMouseEnter={(ev) => {
        ev.currentTarget.style.background = E.panel2;
      }}
      onMouseLeave={(ev) => {
        ev.currentTarget.style.background = 'transparent';
      }}
      style={{
        display: 'grid',
        gridTemplateColumns: '40px 1fr 100px',
        alignItems: 'center',
        gap: 12,
        padding: '11px 16px',
        borderBottom: isLast ? 'none' : `1px solid ${E.hair}`,
        cursor: 'pointer',
        background: 'transparent',
        transition: 'background 140ms',
      }}
    >
      <Pill mono bg={bg} color={color}>
        {prio}
      </Pill>
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontSize: 13,
            color: E.text0,
            fontWeight: 500,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {item.title}
        </div>
        <div
          style={{
            fontFamily: E.fMono,
            fontSize: 10.5,
            color: E.text3,
            marginTop: 2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {item.subtitle}
        </div>
      </div>
      {/* The Btn is a visual affordance - the row is the actual click
          target and is announced by AT via the aria-label above.
          Mark the inner Btn tabIndex=-1 + aria-hidden so keyboard
          users get one stop per row and SR users hear one summary
          instead of a duplicate "Open button" right after. */}
      <Btn
        kind="secondary"
        size="sm"
        tabIndex={-1}
        aria-hidden
        onClick={(ev) => {
          ev.stopPropagation();
          onNavigate(item.cta_target);
        }}
        style={{ justifyContent: 'center', width: '100%' }}
      >
        {actionLabel} -&gt;
      </Btn>
    </div>
  );
}

interface TodaysQueueProps {
  items: HomeSnapshot['attention'];
  onNavigate: (path: string) => void;
}

function TodaysQueue({ items, onNavigate }: TodaysQueueProps) {
  return (
    <Card style={{ padding: 0, overflow: 'hidden' }} data-coachmark="home-experiments">
      <div
        style={{
          padding: '12px 16px',
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <Eyebrow>Today's queue</Eyebrow>
        <Pill bg={E.emberDim} color={E.ember} mono>
          {items.length} {items.length === 1 ? 'item' : 'items'}
        </Pill>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
          ranked by severity
        </span>
      </div>
      {items.length === 0 ? (
        <div style={{ padding: 22, fontSize: 12.5, color: E.text3 }}>
          Nothing waiting. Your inbox is clean.
        </div>
      ) : (
        items.slice(0, 5).map((item, i, arr) => (
          <QueueRow
            key={`${item.title}-${i}`}
            item={item}
            isLast={i === arr.length - 1}
            onNavigate={onNavigate}
          />
        ))
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Live streaming card                                                */
/* ------------------------------------------------------------------ */

interface LiveStreamingCardProps {
  experiments: HomeSnapshot['active_experiments'];
  onNavigate: (path: string) => void;
}

function LiveStreamingCard({ experiments, onNavigate }: LiveStreamingCardProps) {
  const live = experiments.find((e) => e.status === 'running') ?? null;
  return (
    <Card style={{ padding: 14 }} data-coachmark="home-activity">
      <Eyebrow>Live - streaming now</Eyebrow>
      {live ? (
        <button
          type="button"
          onClick={() => onNavigate(`/experiments/${encodeURIComponent(live.id)}`)}
          onMouseEnter={() => {
            void preloadRunDetail();
            prefetchV2(`experiment:${live.id}`, () => v2.experiment(live.id));
          }}
          title={`Open ${live.name}`}
          aria-label={`Open live run ${live.name}`}
          style={{
            display: 'block',
            textAlign: 'left',
            width: '100%',
            marginTop: 10,
            background: E.panel2,
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            padding: '10px 12px',
            cursor: 'pointer',
            font: 'inherit',
            color: 'inherit',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: E.ember,
                animation: 'eDotPulse 1.4s ease-in-out infinite',
                flexShrink: 0,
              }}
            />
            <span
              style={{
                fontSize: 12.5,
                color: E.text0,
                fontWeight: 500,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                minWidth: 0,
                flex: 1,
              }}
            >
              {live.name}
            </span>
            {live.progress && (
              <span
                style={{
                  fontFamily: E.fMono,
                  fontSize: 10.5,
                  color: E.text3,
                  flexShrink: 0,
                }}
              >
                {live.progress.done}/{live.progress.total}
              </span>
            )}
          </div>
          <div style={{ marginTop: 8 }}>
            <Bar
              value={
                live.progress && live.progress.total > 0
                  ? Math.round((live.progress.done / live.progress.total) * 100)
                  : 0
              }
              max={100}
              w="100%"
              h={4}
              color={E.ember}
            />
          </div>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 10,
              color: E.text3,
              marginTop: 6,
              lineHeight: 1.6,
            }}
          >
            {live.pass != null
              ? `pass ${live.pass}%${live.delta_pts != null ? ` - ${live.delta_pts >= 0 ? '+' : ''}${live.delta_pts} pts` : ''}`
              : 'streaming...'}
          </div>
        </button>
      ) : (
        <div style={{ marginTop: 10, fontSize: 12, color: E.text3, fontStyle: 'italic' }}>
          No live runs.
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Pinned dimensions                                                  */
/* ------------------------------------------------------------------ */

function PinnedDimensions({ subMetrics }: { subMetrics: HomeSnapshot['sub_metrics'] }) {
  const top = subMetrics.slice(0, 4);
  return (
    <Card style={{ padding: 14 }} data-coachmark="home-submetrics">
      <Eyebrow>Pinned dimensions</Eyebrow>
      {top.length === 0 ? (
        <div style={{ marginTop: 10, fontSize: 12, color: E.text3, fontStyle: 'italic' }}>
          No sub-metrics yet.
        </div>
      ) : (
        <div
          style={{
            marginTop: 10,
            display: 'flex',
            flexDirection: 'column',
            gap: 9,
          }}
        >
          {top.map((m) => {
            const color = m.value >= 80 ? E.pass : m.value >= 70 ? E.warn : E.fail;
            // Synthesize a small spark from the value+delta. Real timeline
            // data isn't part of HomeSnapshot.sub_metrics; we render a
            // 6-pt approximation: from (value - delta) to value.
            const start = Math.max(0, Math.min(100, m.value - m.delta));
            const end = Math.max(0, Math.min(100, m.value));
            const spark = [start, start, (start + end) / 2, end, end, end];
            return (
              <div
                key={m.label}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 50px 50px',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span
                  style={{
                    fontSize: 11.5,
                    color: E.text1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={m.label}
                >
                  {m.label}
                </span>
                <Spark data={spark} w={50} h={14} color={color} />
                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 11,
                    color,
                    textAlign: 'right',
                  }}
                >
                  {Math.round(m.value)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Skeleton + empty + error                                           */
/* ------------------------------------------------------------------ */

function HomeSkeleton() {
  const cellStyle: CSSProperties = {
    padding: '0 18px',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  };
  return (
    <div style={{ padding: '24px 28px', maxWidth: 1180 }}>
      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        <Skeleton w={38} h={38} style={{ borderRadius: 8 }} />
        <div style={{ flex: 1 }}>
          <Skeleton w={180} h={11} />
          <div style={{ marginTop: 10 }}>
            <Skeleton w="80%" h={26} />
          </div>
          <div style={{ marginTop: 10 }}>
            <Skeleton w="60%" h={26} />
          </div>
        </div>
      </div>
      <div
        style={{
          marginTop: 22,
          background: E.panel,
          border: `1px solid ${E.hair}`,
          borderRadius: 10,
          padding: '14px 4px',
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
        }}
      >
        {[0, 1, 2, 3].map((i) => (
          <div key={i} style={{ ...cellStyle, borderRight: i < 3 ? `1px solid ${E.hair}` : 'none' }}>
            <Skeleton w={80} h={11} />
            <Skeleton w={120} h={28} />
            <Skeleton w={170} h={26} />
          </div>
        ))}
      </div>
      <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 14 }}>
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '12px 16px', borderBottom: `1px solid ${E.hair}` }}>
            <Skeleton w={140} h={11} />
          </div>
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              style={{
                padding: '11px 16px',
                borderBottom: i < 4 ? `1px solid ${E.hair}` : 'none',
                display: 'flex',
                gap: 12,
                alignItems: 'center',
              }}
            >
              <Skeleton w={28} h={16} />
              <div style={{ flex: 1 }}>
                <Skeleton w="70%" h={13} />
                <div style={{ marginTop: 4 }}>
                  <Skeleton w="40%" h={10} />
                </div>
              </div>
              <Skeleton w={70} h={22} />
            </div>
          ))}
        </Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {[0, 1].map((i) => (
            <Card key={i} style={{ padding: 14 }}>
              <Skeleton w={120} h={11} />
              <div style={{ marginTop: 10 }}>
                <Skeleton w="100%" h={48} style={{ borderRadius: 6 }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function ErrorView({ err, onRetry }: { err: string; onRetry: () => void }) {
  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
      <Card style={{ padding: 16, borderColor: E.fail }}>
        <Eyebrow style={{ color: E.fail }}>Error loading home</Eyebrow>
        <div
          style={{
            fontFamily: E.fMono,
            fontSize: 12,
            color: E.text2,
            marginTop: 6,
            lineHeight: 1.55,
          }}
        >
          {err}
        </div>
        <div style={{ marginTop: 12 }}>
          <Btn kind="secondary" size="sm" onClick={onRetry}>
            ↻ Retry
          </Btn>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Empty state (no quality data yet)                                  */
/* ------------------------------------------------------------------ */

function EmptyHome({ snap, navigate }: { snap: HomeSnapshot; navigate: (p: string) => void }) {
  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
      <Eyebrow style={{ marginBottom: 8 }}>Project home</Eyebrow>
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
        {snap.project.name}
      </h1>
      <p style={{ fontSize: 14, color: E.text2, marginTop: 8, lineHeight: 1.55, maxWidth: 640 }}>
        No evaluation runs yet. Load the demo to explore a populated workspace, or run your first
        eval from the CLI.
      </p>
      <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 18 }}>
        <Welcome />
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '14px 18px', borderBottom: `1px solid ${E.hair}` }}>
            <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Recent activity</span>
          </div>
          <div
            style={{
              padding: 22,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 10,
            }}
          >
            <span style={{ fontSize: 22, color: E.text4, fontFamily: E.fSerif, lineHeight: 1 }} aria-hidden>
              ◆
            </span>
            <span style={{ fontSize: 13, color: E.text2, textAlign: 'center' }}>No runs yet.</span>
            <span
              style={{
                fontSize: 11,
                color: E.text3,
                textAlign: 'center',
                fontFamily: E.fMono,
                maxWidth: 280,
              }}
            >
              Run an evaluation from the CLI or load the demo to populate this feed.
            </span>
            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
              <Btn kind="secondary" size="sm" onClick={() => navigate('/commands')}>
                Browse commands
              </Btn>
              <Btn kind="ghost" size="sm" onClick={() => navigate('/datasets')}>
                Datasets -&gt;
              </Btn>
            </div>
          </div>
        </Card>
      </div>
      <div style={{ height: 30 }} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Footer note                                                        */
/* ------------------------------------------------------------------ */

function WhyChangedNote({ children }: { children: ReactNode }) {
  return (
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
        maxWidth: 1180,
      }}
    >
      <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
        why this changed -&gt;
      </span>{' '}
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Top-level Home component                                           */
/* ------------------------------------------------------------------ */

function buildPulseCells(snap: HomeSnapshot, navigate: (p: string) => void): PulseCellData[] {
  const q = snap.quality;

  // Quality
  const qSpark = q.timeline.slice(-12).map((p) => p.y);
  const qDelta = q.delta_30d;
  const qKind: PulseKind =
    qDelta == null ? 'info' : qDelta >= 0 ? 'pass' : qDelta < -2 ? 'fail' : 'warn';
  const qualityCell: PulseCellData = {
    label: 'Quality',
    value: q.current != null ? `${q.current.toFixed(1)}%` : '--',
    delta:
      qDelta == null
        ? '--'
        : qDelta === 0
          ? 'flat'
          : `${qDelta >= 0 ? '+' : ''}${qDelta.toFixed(1)}`,
    kind: qKind,
    spark: qSpark,
    onClick: () => navigate('/experiments'),
  };

  // Calibration: count drift entries from sub_metrics. Heuristic - "drift"
  // is any sub-metric whose absolute delta exceeds 2 pts in the wrong
  // direction. Total = total sub_metrics.
  const total = snap.sub_metrics.length;
  const drift = snap.sub_metrics.filter((m) => {
    const goodDirection = m.inverse ? m.delta < 0 : m.delta > 0;
    return !goodDirection && Math.abs(m.delta) > 2;
  }).length;
  const calibratedCount = Math.max(0, total - drift);
  const calibrationKind: PulseKind = drift === 0 ? 'pass' : drift >= 2 ? 'fail' : 'warn';
  const calibrationCell: PulseCellData = {
    label: 'Calibration',
    value: total > 0 ? `${calibratedCount} / ${total}` : '--',
    delta: drift > 0 ? `${drift} drift` : 'stable',
    kind: calibrationKind,
    // Synthesize a flat-ish trend; real per-day calibration history isn't
    // in HomeSnapshot. Stub a 6-pt declining trend if drift > 0.
    spark:
      total > 0
        ? Array.from({ length: 6 }, (_, i) =>
            Math.max(0, total - Math.max(0, drift - (5 - i)) * 0.5),
          )
        : [],
    onClick: () => navigate('/metrics'),
  };

  // Review backlog: count attention items with cta_target starting /review.
  const reviewItems = snap.attention.filter(
    (a) => a.cta_target.startsWith('/review') || /review/i.test(a.cta),
  );
  const backlog = reviewItems.length;
  // Try to find a "+N today" hint in subtitle text; otherwise omit it.
  const todayMatch = reviewItems
    .map((a) => /\+(\d+)\s*today/i.exec(a.subtitle ?? ''))
    .find((m) => m);
  const todayN = todayMatch ? parseInt(todayMatch[1], 10) : null;
  const backlogKind: PulseKind = backlog === 0 ? 'pass' : backlog >= 5 ? 'fail' : 'warn';
  const backlogCell: PulseCellData = {
    label: 'Review backlog',
    value: `${backlog}`,
    delta: todayN != null ? `+${todayN} today` : backlog > 0 ? 'pending' : 'clear',
    kind: backlogKind,
    spark: backlog > 0 ? [0, 1, 1, 2, 3, Math.min(backlog, 5)] : [],
    onClick: () => navigate('/review'),
  };

  // Spend MTD
  let spendCell: PulseCellData;
  if (snap.cost) {
    const c = snap.cost;
    const value = fmtUsd(c.total_30d);
    let delta: string;
    if (c.projected_monthly != null) {
      delta = `proj ${fmtUsd(c.projected_monthly)}`;
    } else if (c.runs_total > 0) {
      delta = `${c.runs_with_cost}/${c.runs_total} runs`;
    } else {
      delta = '--';
    }
    spendCell = {
      label: 'Spend MTD',
      value,
      delta,
      kind: 'info',
      spark: c.daily_30d.slice(-12),
    };
  } else {
    spendCell = {
      label: 'Spend MTD',
      value: '--',
      delta: 'no runs',
      kind: 'info',
      spark: [],
    };
  }

  return [qualityCell, calibrationCell, backlogCell, spendCell];
}

export default function Home() {
  const { data: snap, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'home',
    v2.home,
  );
  const navigate = useNavigate();
  // useTour() is mounted globally in AppShell. Each tabbed route declares
  // which tour should fire on first visit via useRouteTour.
  useRouteTour(FIRST_RUN_TOUR_ID, !!(snap && !err));

  const pulseCells = useMemo(
    () => (snap ? buildPulseCells(snap, navigate) : []),
    [snap, navigate],
  );

  // Hard error (no cached payload) - render error card with retry
  if (err && !snap) {
    return (
      <AppShell>
        <ErrorView err={err} onRetry={() => void refetch()} />
      </AppShell>
    );
  }

  // First load - render the skeleton until data lands
  if (!snap) {
    return (
      <AppShell>
        <HomeSkeleton />
      </AppShell>
    );
  }

  // No quality data yet - keep the friendly empty state from the prior
  // implementation. The proposal's triage UI requires real signal.
  if (snap.quality.current == null) {
    return (
      <AppShell contextChip={snap.project}>
        <EmptyHome snap={snap} navigate={navigate} />
      </AppShell>
    );
  }

  return (
    <AppShell contextChip={snap.project}>
      <div style={{ padding: '24px 28px', maxWidth: 1180 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <Eyebrow>Project home - {snap.project.name}</Eyebrow>
          <UpdatingChip
            visible={reloading && !isInitialLoad}
            error={snap ? err : null}
            onRetry={refetch}
          />
        </div>

        {/* Conversational brief header */}
        <BriefHeader
          snap={snap}
          reloading={reloading}
          onRegenerate={() => void refetch()}
          onNavigate={(p) => navigate(p)}
        />

        {/* Pulse strip */}
        <PulseStrip cells={pulseCells} />

        {/* Today's queue + live streaming + pinned dimensions */}
        <div
          style={{
            marginTop: 18,
            display: 'grid',
            gridTemplateColumns: '1.4fr 1fr',
            gap: 14,
          }}
        >
          <TodaysQueue items={snap.attention} onNavigate={(p) => navigate(p)} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <LiveStreamingCard
              experiments={snap.active_experiments}
              onNavigate={(p) => navigate(p)}
            />
            <PinnedDimensions subMetrics={snap.sub_metrics} />
          </div>
        </div>

        {/* "Why this changed" footer note */}
        <WhyChangedNote>
          The triage cockpit puts the action queue first. Brief and pulse strip frame the day; the
          queue, live runs, and pinned dimensions surface what to do next without making you read a
          chart.
        </WhyChangedNote>

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
