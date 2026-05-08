/**
 * Metrics - Trust Report.
 *
 * A redesigned /metrics route. Top-down: an inline trust scoreboard
 * answering "which judges should you trust today?" then a visual rubric
 * editor with a calibration progress donut and confusion matrix on the
 * right rail. Selection state lives in this component so clicking a row
 * in the scoreboard switches the editor without a route change.
 *
 * The new /api/v2/rubrics/trust + saveRubric endpoints aren't on the
 * client yet so the scoreboard is computed from /rubrics + per-rubric
 * /calibration loads, the confusion matrix is derived from FP/FN%, and
 * "Save rubric" is honestly disabled with a "(soon)" suffix. We swap
 * to the dedicated endpoints once the persistence + trust hooks land.
 * The editor still tracks dirty/discard locally so users can explore
 * weight changes; the beforeunload warning + "unsaved" pill make it
 * clear edits do not survive navigation.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import type { CSSProperties } from 'react';
import { AppShell } from '../AppShell';
import {
  Btn,
  Card,
  Donut,
  Eyebrow,
  Pill,
  Skeleton,
  UpdatingChip,
} from '../ui';
import { v2 } from '../api/client';
import type { RubricDetail, RubricRow } from '../api/types';
import { useV2Resource } from '../hooks/useV2Resource';
import { useFlashState } from '../hooks/useFlashState';
import { useRouteTour } from '../tour/useRouteTour';
import { READ_METRICS_TOUR_ID } from '../tour/scripts/readMetrics';
import { useProject } from '../hooks/useProject';
import { copyToClipboard } from '../clipboard';
import { E } from '../tokens';

/* ------------------------------ shared types --------------------------- */

interface TrustRow {
  id: string;
  name: string;
  metric_type: 'llm' | 'programmatic';
  score: number;
  kappa: number | null;
  drift_per_week: number;
  sample_size: number | string;
  needs_work: boolean;
  has_kappa: boolean;
}

interface DimensionDraft {
  label: string;
  weight: number;
  fp: number;
  fn: number;
}

/* ------------------------------ utilities ------------------------------ */

function parseLabelKappa(label: string): number | null {
  const m = label.match(/k\s*=\s*([0-9.]+)/i);
  if (!m) return null;
  const v = Number.parseFloat(m[1]);
  return Number.isFinite(v) ? v : null;
}

function deriveTrustScore(kappa: number | null, isProgrammatic: boolean): number {
  if (isProgrammatic) return 1.0;
  if (kappa == null) return 0;
  return Math.max(0, Math.min(1, kappa));
}

function trustColor(score: number): string {
  if (score >= 0.8) return E.pass;
  if (score >= 0.7) return E.warn;
  return E.fail;
}

function fmtDrift(d: number): string {
  if (!d || Math.abs(d) < 0.005) return '-';
  const sign = d > 0 ? '+' : '';
  return `${sign}${d.toFixed(2)} / wk`;
}

function fmtKappa(k: number | null, isProg: boolean): string {
  if (isProg) return 'prog';
  if (k == null) return '-';
  return k.toFixed(2);
}

function fmtSample(n: number | string, isProg: boolean): string {
  if (isProg) return '-';
  return `n=${n}`;
}

/* ----------------------- TrustScoreboard sub-component ----------------- */

interface TrustScoreboardCardProps {
  rows: TrustRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Called when the user clicks "Run an evaluation" in the empty
   * state. The card itself doesn't know the route - the parent owns
   * navigation - so this is a pure callback. */
  onRunEval: () => void;
  loading: boolean;
}

function TrustScoreboardCard({
  rows,
  selectedId,
  onSelect,
  onRunEval,
  loading,
}: TrustScoreboardCardProps) {
  return (
    <Card
      style={{ padding: 14, marginBottom: 14 }}
      data-coachmark="metrics-list"
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          marginBottom: 12,
        }}
      >
        <Eyebrow>Calibration health</Eyebrow>
        <span style={{ fontFamily: E.fMono, fontSize: 10.5, color: E.text3 }}>
          sorted by trust gap - re-evaluated nightly
        </span>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              style={{
                display: 'grid',
                gridTemplateColumns: '160px 1fr 80px 80px 100px',
                alignItems: 'center',
                gap: 14,
              }}
            >
              <Skeleton w="60%" h={12} />
              <Skeleton w="100%" h={12} />
              <Skeleton w="60%" h={11} />
              <Skeleton w="60%" h={11} />
              <Skeleton w="60%" h={11} />
            </div>
          ))}
        </div>
      )}

      {!loading && rows.length === 0 && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 12,
            padding: '24px 0 8px',
            textAlign: 'center',
          }}
        >
          <div style={{ color: E.text2, fontSize: 13, maxWidth: 360 }}>
            No metrics yet. Run your first evaluation - the trust report
            fills in once judges have something to score.
          </div>
          <Btn kind="primary" size="sm" onClick={onRunEval}>
            Run an evaluation →
          </Btn>
        </div>
      )}

      {!loading && rows.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {rows.map((t) => {
            const isProg = t.metric_type === 'programmatic';
            const isActive = t.id === selectedId;
            const tColor = trustColor(t.score);
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => onSelect(t.id)}
                aria-pressed={isActive}
                aria-label={`${t.name} - trust score ${Math.round(t.score * 100)} of 100${t.needs_work ? ', needs work' : ''}`}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = E.panel2;
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'transparent';
                }}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '160px 1fr 80px 80px 100px',
                  alignItems: 'center',
                  gap: 14,
                  padding: '6px 8px',
                  border: 'none',
                  borderRadius: 6,
                  background: isActive ? E.emberDim : 'transparent',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'background 140ms',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span
                    style={{
                      fontSize: 13,
                      color: E.text0,
                      fontWeight: 500,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {t.name}
                  </span>
                  {t.needs_work && (
                    <Pill bg={E.warnDim} color={E.warn} mono>
                      needs work
                    </Pill>
                  )}
                </div>

                <div style={{ position: 'relative', height: 12 }}>
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: E.panel3,
                      borderRadius: 3,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${Math.max(0, Math.min(1, t.score)) * 100}%`,
                        background: tColor,
                        transition: 'width 200ms ease-out',
                      }}
                    />
                  </div>
                  <div
                    style={{
                      position: 'absolute',
                      left: '70%',
                      top: -2,
                      bottom: -2,
                      width: 1,
                      background: E.text4,
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      left: '80%',
                      top: -2,
                      bottom: -2,
                      width: 1,
                      background: E.text3,
                    }}
                  />
                </div>

                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 11,
                    color: isProg ? E.text3 : tColor,
                    textAlign: 'right',
                  }}
                >
                  k {fmtKappa(t.kappa, isProg)}
                </span>
                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 11,
                    color:
                      t.drift_per_week > 0.005
                        ? E.pass
                        : t.drift_per_week < -0.005
                          ? E.fail
                          : E.text3,
                    textAlign: 'right',
                  }}
                >
                  {fmtDrift(t.drift_per_week)}
                </span>
                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 11,
                    color: E.text3,
                    textAlign: 'right',
                  }}
                >
                  {fmtSample(t.sample_size, isProg)}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div
        style={{
          marginTop: 10,
          paddingTop: 10,
          borderTop: `1px solid ${E.hair}`,
          display: 'flex',
          gap: 14,
          fontFamily: E.fMono,
          fontSize: 10,
          color: E.text3,
        }}
      >
        <span>0.7 = annotation threshold</span>
        <span>0.8 = ship-ready</span>
        <span style={{ flex: 1 }} />
        {/* Placeholder for the future "re-run kappa & drift across all
            metrics" action - not wired yet. Render as plain muted
            text rather than a fake button so AT does not announce a
            non-functional control and sighted users do not expect
            anything to happen on click. */}
        <span
          aria-disabled="true"
          title="Trust audit endpoint is not wired yet"
          style={{ color: E.text3, fontStyle: 'italic', cursor: 'default' }}
        >
          Trust audit (soon)
        </span>
      </div>
    </Card>
  );
}

/* ------------------------ DimensionRow sub-component ------------------- */

interface DimensionRowProps {
  dim: DimensionDraft;
  index: number;
  onWeightChange: (next: number) => void;
  onRemove: () => void;
  onDragStart: () => void;
  onDragOver: () => void;
  onDragEnd: () => void;
  isDragging: boolean;
  isDropTarget: boolean;
}

function DimensionRow({
  dim,
  index,
  onWeightChange,
  onRemove,
  onDragStart,
  onDragOver,
  onDragEnd,
  isDragging,
  isDropTarget,
}: DimensionRowProps) {
  const fp = Math.max(0, Math.min(100, Math.round(dim.fp)));
  const fn = Math.max(0, Math.min(100, Math.round(dim.fn)));
  const totalBad = Math.min(16, Math.round(((fp + fn) * 16) / 100));
  const fpCells = Math.min(totalBad, Math.round((fp * 16) / 100));
  const fnCells = Math.max(0, totalBad - fpCells);
  const hot = fp >= 10 || fn >= 10;
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(index));
        onDragStart();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        onDragOver();
      }}
      onDragEnd={onDragEnd}
      onDrop={(e) => {
        e.preventDefault();
        onDragEnd();
      }}
      style={{
        display: 'grid',
        gridTemplateColumns: '24px 130px 1fr 120px 110px 28px',
        gap: 10,
        alignItems: 'center',
        padding: '10px 12px',
        background: isDropTarget ? E.emberDim : E.panel2,
        border: `1px solid ${hot ? E.failDim : E.hair2}`,
        borderRadius: 6,
        opacity: isDragging ? 0.4 : 1,
        transition: 'opacity 120ms, background 140ms',
      }}
    >
      <span
        title="Drag to reorder"
        style={{
          color: E.text4,
          cursor: 'grab',
          fontFamily: E.fMono,
          userSelect: 'none',
          textAlign: 'center',
        }}
      >
        ::
      </span>
      <span
        style={{
          fontSize: 13,
          color: E.text0,
          fontWeight: 500,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {dim.label || '(unnamed)'}
      </span>

      <div
        style={{
          height: 6,
          background: E.panel3,
          borderRadius: 999,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${Math.max(0, Math.min(100, dim.weight))}%`,
            height: '100%',
            background: E.ember,
            transition: 'width 160ms ease-out',
          }}
        />
      </div>

      <div
        style={{ display: 'flex', alignItems: 'center', gap: 4 }}
        title="Adjust weight (5% steps)"
      >
        <button
          type="button"
          onClick={() => onWeightChange(Math.max(0, dim.weight - 5))}
          aria-label="Decrease weight"
          style={{
            background: E.panel3,
            border: `1px solid ${E.hair2}`,
            color: E.text2,
            borderRadius: 4,
            width: 18,
            height: 18,
            fontSize: 11,
            cursor: 'pointer',
            lineHeight: 1,
            padding: 0,
          }}
        >
          -
        </button>
        <input
          type="number"
          inputMode="numeric"
          // The visible label (dim.label) is two columns away with a
          // progress bar between, so SR users tabbing here would
          // otherwise hear "44, spin button" with no context. Anchor
          // it to the dimension by name + the unit (% weight).
          aria-label={`Weight % for ${dim.label || 'dimension'}`}
          value={Math.round(dim.weight)}
          min={0}
          max={100}
          step={1}
          onChange={(e) => {
            // An empty input would coerce to 0 via Number(""), which
            // means the field snaps to 0 the moment the user clears
            // it to retype - they can't pause mid-edit. Treat empty
            // as "no change" so the user can clear-and-retype freely;
            // the onBlur path (or a step click) will commit.
            if (e.target.value === '') return;
            const v = Number(e.target.value);
            if (!Number.isFinite(v)) return;
            onWeightChange(Math.max(0, Math.min(100, v)));
          }}
          style={{
            fontFamily: E.fMono,
            fontSize: 12,
            color: E.text1,
            width: 44,
            background: 'transparent',
            border: `1px solid ${E.hair}`,
            borderRadius: 4,
            padding: '2px 4px',
            textAlign: 'center',
          }}
        />
        <button
          type="button"
          onClick={() => onWeightChange(Math.min(100, dim.weight + 5))}
          aria-label="Increase weight"
          style={{
            background: E.panel3,
            border: `1px solid ${E.hair2}`,
            color: E.text2,
            borderRadius: 4,
            width: 18,
            height: 18,
            fontSize: 11,
            cursor: 'pointer',
            lineHeight: 1,
            padding: 0,
          }}
        >
          +
        </button>
        <span
          style={{
            fontFamily: E.fMono,
            fontSize: 11,
            color: E.text3,
            marginLeft: 2,
          }}
        >
          %
        </span>
      </div>

      <div>
        <div
          style={{
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.text3,
            display: 'flex',
            gap: 6,
          }}
        >
          <span style={{ color: E.fail }}>FP {fp}%</span>
          <span style={{ color: E.warn }}>FN {fn}%</span>
        </div>
        <div style={{ display: 'flex', gap: 2, marginTop: 3 }}>
          {Array.from({ length: 16 }).map((_, i) => (
            <span
              key={i}
              style={{
                width: 4,
                height: 6,
                background:
                  i < fpCells ? E.fail : i < fpCells + fnCells ? E.warn : E.pass,
                borderRadius: 1,
              }}
            />
          ))}
        </div>
      </div>

      <button
        type="button"
        onClick={onRemove}
        title="Remove dimension"
        aria-label="Remove dimension"
        style={{
          background: 'transparent',
          border: 'none',
          color: E.text3,
          cursor: 'pointer',
          fontSize: 14,
          padding: 0,
        }}
      >
        ...
      </button>
    </div>
  );
}

/* --------------------- ConfusionMatrix sub-component ------------------- */

interface ConfusionMatrixProps {
  cm: { tp: number; tn: number; fp: number; fn: number } | null;
  sampleSize: number;
}

function ConfusionMatrix({ cm, sampleSize }: ConfusionMatrixProps) {
  const cellBase: CSSProperties = {
    padding: 8,
    textAlign: 'center',
    border: `1px solid ${E.hair2}`,
    fontFamily: E.fMono,
    fontSize: 12,
  };
  const tp = cm?.tp ?? 0;
  const tn = cm?.tn ?? 0;
  const fp = cm?.fp ?? 0;
  const fn = cm?.fn ?? 0;

  let worstNote: string | null = null;
  if (cm && (fp > 0 || fn > 0)) {
    if (fp >= fn) {
      worstNote = 'Worst: false positives - judge marks failing items as passing.';
    } else {
      worstNote = 'Worst: false negatives - judge marks passing items as failing.';
    }
  }

  return (
    <Card style={{ padding: 14 }} data-coachmark="metrics-chart">
      <Eyebrow>
        Confusion matrix - {sampleSize} sample{sampleSize === 1 ? '' : 's'}
      </Eyebrow>
      <div
        style={{
          marginTop: 10,
          display: 'grid',
          gridTemplateColumns: 'auto 1fr 1fr',
          fontFamily: E.fMono,
          fontSize: 11,
          color: E.text2,
          gap: 0,
        }}
      >
        <div />
        <div style={{ textAlign: 'center', color: E.text3, padding: 4 }}>
          pred ok
        </div>
        <div style={{ textAlign: 'center', color: E.text3, padding: 4 }}>
          pred no
        </div>
        <div style={{ color: E.text3, padding: 4 }}>true ok</div>
        <div style={{ ...cellBase, background: E.passDim, color: E.pass }}>
          {tp}
        </div>
        <div style={{ ...cellBase, background: E.warnDim, color: E.warn }}>
          {fn}
        </div>
        <div style={{ color: E.text3, padding: 4 }}>true no</div>
        <div style={{ ...cellBase, background: E.failDim, color: E.fail }}>
          {fp}
        </div>
        <div style={{ ...cellBase, background: E.passDim, color: E.pass }}>
          {tn}
        </div>
      </div>
      {!cm && (
        <div
          style={{
            marginTop: 8,
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.text3,
            lineHeight: 1.6,
          }}
        >
          No annotations yet - run evalyn annotate to populate the matrix.
        </div>
      )}
      {worstNote && (
        <div
          style={{
            marginTop: 8,
            fontFamily: E.fMono,
            fontSize: 10,
            color: E.text3,
            lineHeight: 1.6,
          }}
        >
          {worstNote}
        </div>
      )}
    </Card>
  );
}

/* -------------------- CalibrationProgress sub-component ---------------- */

interface CalibrationProgressProps {
  metricId: string;
  sampleSize: number;
  threshold?: number;
}

function CalibrationProgress({
  metricId,
  sampleSize,
  threshold = 10,
}: CalibrationProgressProps) {
  const navigate = useNavigate();
  const done = Math.min(threshold, Math.max(0, sampleSize));
  const remaining = Math.max(0, threshold - done);
  return (
    <Card style={{ padding: 14 }}>
      <Eyebrow>Calibration progress</Eyebrow>
      <div
        style={{
          marginTop: 10,
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        <Donut
          segments={[
            { value: done, color: E.ember },
            { value: remaining, color: E.panel3 },
          ]}
          size={92}
          thickness={11}
        />
      </div>
      <div
        style={{
          textAlign: 'center',
          fontFamily: E.fMono,
          fontSize: 12,
          color: E.text1,
          marginTop: 6,
        }}
      >
        <strong>{done}</strong> of {threshold} to threshold
      </div>
      <Btn
        kind="primary"
        size="sm"
        style={{ marginTop: 10, width: '100%', justifyContent: 'center' }}
        onClick={() =>
          navigate(`/annotate?metric=${encodeURIComponent(metricId)}`)
        }
        disabled={remaining === 0}
        title={
          remaining === 0
            ? 'Already at the calibration threshold'
            : `Open the annotation queue scoped to ${metricId}`
        }
      >
        {remaining === 0 ? 'At threshold' : `Annotate ${remaining} more ->`}
      </Btn>
    </Card>
  );
}

/* -------------------------- RubricEditor card ------------------------- */

interface RubricEditorProps {
  detail: RubricDetail | null;
  loading: boolean;
  err: string | null;
  /** Re-fetch the rubric detail after a load failure. */
  onRetry: () => void;
  draft: DimensionDraft[];
  setDraft: (next: DimensionDraft[]) => void;
  dirty: boolean;
  onDiscard: () => void;
  drift: number;
}

function RubricEditor({
  detail,
  loading,
  err,
  onRetry,
  draft,
  setDraft,
  dirty,
  onDiscard,
  drift,
}: RubricEditorProps) {
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [dropIdx, setDropIdx] = useState<number | null>(null);
  // Preview Copy state. Stable aria-label + sibling .eSr live region
  // matches the canonical pattern used by CliRunner / RunDetail /
  // FailureCluster - avoids the "name changed mid-press" announcement
  // some SR engines emit when a button's accessible name flips.
  const [copyState, flashCopyState] = useFlashState<'idle' | 'copied' | 'error'>('idle');

  const totalWeight = useMemo(
    () =>
      draft.reduce((s, d) => s + (Number.isFinite(d.weight) ? d.weight : 0), 0),
    [draft],
  );
  const sumOk = Math.abs(totalWeight - 100) < 0.5;

  const previewCmd = useMemo(() => {
    if (!detail) return '';
    const pairs = draft
      .filter((d) => d.label.trim().length > 0)
      .map(
        (d) =>
          `${d.label.trim().toLowerCase().replace(/\s+/g, '_')}=${Math.round(d.weight)}`,
      )
      .join(',');
    return [
      `evalyn select-metrics ${detail.id} \\`,
      `  --weights ${pairs || '<none>'} \\`,
      `  --judge gpt-4o-mini --threshold 0.7`,
    ].join('\n');
  }, [detail, draft]);

  function moveDimension(from: number, to: number) {
    if (from === to || from < 0 || to < 0) return;
    const next = draft.slice();
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    setDraft(next);
  }

  function updateWeight(idx: number, weight: number) {
    const next = draft.slice();
    next[idx] = { ...next[idx], weight };
    setDraft(next);
  }

  function removeDimension(idx: number) {
    const next = draft.slice();
    next.splice(idx, 1);
    setDraft(next);
  }

  function addDimension() {
    const remaining = Math.max(0, Math.round(100 - totalWeight));
    setDraft([
      ...draft,
      {
        label: `dimension_${draft.length + 1}`,
        weight: remaining > 0 ? remaining : 10,
        fp: 0,
        fn: 0,
      },
    ]);
  }

  const lastEvalLabel =
    detail && detail.calibration.sample_size > 0
      ? 'recently'
      : 'never (no annotations)';
  const isProgrammatic =
    !!detail && detail.dimensions.length > 0 && detail.dimensions[0].kind === 'prog';

  return (
    <Card style={{ padding: 18 }} data-coachmark="metrics-rubric">
      {loading && (
        <>
          <Skeleton w="40%" h={14} />
          <div
            style={{
              marginTop: 14,
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
            }}
          >
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  padding: 12,
                  background: E.panel2,
                  borderRadius: 8,
                  border: `1px solid ${E.hair}`,
                }}
              >
                <Skeleton w="50%" h={13} />
                <div style={{ marginTop: 6 }}>
                  <Skeleton w="80%" h={11} />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      {!loading && err && (
        <div role="alert">
          <Eyebrow style={{ color: E.fail }}>Error loading rubric</Eyebrow>
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
            <Btn kind="secondary" size="sm" onClick={onRetry}>
              Retry
            </Btn>
          </div>
        </div>
      )}
      {!loading && !err && detail && (
        <>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 4,
            }}
          >
            <Eyebrow>Editing</Eyebrow>
            {Math.abs(drift) >= 0.005 && (
              <Pill
                bg={drift < 0 ? E.warnDim : E.passDim}
                color={drift < 0 ? E.warn : E.pass}
                mono
              >
                drift {fmtDrift(drift)}
              </Pill>
            )}
            <span style={{ flex: 1 }} />
            {dirty && (
              <Pill
                bg={E.warnDim}
                color={E.warn}
                mono
                title="Local edits only - rubric persistence is not wired up yet"
              >
                unsaved (local)
              </Pill>
            )}
          </div>
          <h2
            style={{
              fontFamily: E.fSerif,
              fontSize: 26,
              margin: '4px 0 4px',
              color: E.text0,
              fontWeight: 400,
            }}
          >
            {detail.name}
          </h2>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text3,
              marginBottom: 14,
            }}
          >
            {detail.id} - {isProgrammatic ? 'programmatic' : 'LLM judge'} -
            last evaluated {lastEvalLabel}
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 8,
            }}
          >
            <Eyebrow>Dimensions - weights sum to 100%</Eyebrow>
            <span
              style={{
                fontFamily: E.fMono,
                fontSize: 11,
                color: sumOk ? E.text3 : E.warn,
              }}
              title={
                sumOk ? 'Total weight' : 'Total must equal 100% before saving'
              }
            >
              total {Math.round(totalWeight)}%
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {draft.length === 0 && (
              <div
                style={{
                  padding: 14,
                  border: `1px dashed ${E.hair2}`,
                  borderRadius: 6,
                  fontSize: 12,
                  color: E.text3,
                  textAlign: 'center',
                }}
              >
                No dimensions yet. Click below to add one.
              </div>
            )}
            {draft.map((d, idx) => (
              <DimensionRow
                key={`${d.label}-${idx}`}
                dim={d}
                index={idx}
                isDragging={dragIdx === idx}
                isDropTarget={dropIdx === idx && dragIdx !== idx}
                onWeightChange={(w) => updateWeight(idx, w)}
                onRemove={() => removeDimension(idx)}
                onDragStart={() => setDragIdx(idx)}
                onDragOver={() => setDropIdx(idx)}
                onDragEnd={() => {
                  if (dragIdx != null && dropIdx != null && dragIdx !== dropIdx) {
                    moveDimension(dragIdx, dropIdx);
                  }
                  setDragIdx(null);
                  setDropIdx(null);
                }}
              />
            ))}
            <button
              type="button"
              onClick={addDimension}
              style={{
                padding: '8px 12px',
                border: `1px dashed ${E.hair2}`,
                borderRadius: 6,
                fontSize: 12,
                color: E.text3,
                textAlign: 'center',
                cursor: 'pointer',
                background: 'transparent',
                fontFamily: E.fSans,
              }}
            >
              + Add dimension
            </button>
          </div>

          <div
            style={{
              marginTop: 14,
              borderRadius: 6,
              background: E.text0,
              border: `1px solid ${E.text0}`,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 0,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                flex: 1,
                padding: 12,
                fontFamily: E.fMono,
                fontSize: 11,
                color: E.ink,
                lineHeight: 1.7,
                whiteSpace: 'pre',
                overflowX: 'auto',
                minWidth: 0,
              }}
            >
              <span style={{ color: E.text4 }}>
                # preview - evalyn select-metrics
              </span>
              {'\n'}
              {previewCmd}
            </div>
            <button
              type="button"
              onClick={async () => {
                try {
                  await copyToClipboard(previewCmd);
                  flashCopyState('copied', 2000);
                } catch {
                  flashCopyState('error', 3000);
                }
              }}
              aria-label="Copy the shell command to clipboard"
              title={
                copyState === 'copied'
                  ? 'Copied!'
                  : copyState === 'error'
                    ? 'Browser blocked clipboard access'
                    : 'Copy command'
              }
              style={{
                flexShrink: 0,
                padding: '0 12px',
                fontFamily: E.fMono,
                fontSize: 10.5,
                color:
                  copyState === 'copied'
                    ? E.pass
                    : copyState === 'error'
                      ? E.fail
                      : E.text4,
                background: 'transparent',
                border: 'none',
                borderLeft: `1px solid ${E.panel4}`,
                cursor: 'pointer',
                alignSelf: 'stretch',
                whiteSpace: 'nowrap',
              }}
              onMouseEnter={(e) => {
                if (copyState === 'idle') {
                  e.currentTarget.style.color = E.panel;
                }
              }}
              onMouseLeave={(e) => {
                if (copyState === 'idle') {
                  e.currentTarget.style.color = E.text4;
                }
              }}
              onFocus={(e) => {
                if (copyState === 'idle') {
                  e.currentTarget.style.color = E.panel;
                }
              }}
              onBlur={(e) => {
                if (copyState === 'idle') {
                  e.currentTarget.style.color = E.text4;
                }
              }}
            >
              {copyState === 'copied'
                ? '✓'
                : copyState === 'error'
                  ? '✗'
                  : 'copy'}
            </button>
            {/* SR live region - announces success/error transitions
                without changing the button's accessible name. Polite
                so it doesn't interrupt other speech; atomic so the
                full message reads at once. */}
            <span
              role="status"
              aria-live="polite"
              aria-atomic="true"
              className="eSr"
            >
              {copyState === 'copied'
                ? 'Command copied to clipboard.'
                : copyState === 'error'
                  ? 'Could not copy command. Browser blocked clipboard access.'
                  : ''}
            </span>
          </div>

          <div
            style={{
              marginTop: 12,
              display: 'flex',
              gap: 6,
              alignItems: 'center',
            }}
          >
            <Btn
              kind="primary"
              size="sm"
              disabled
              title="Rubric persistence is not wired up yet - your edits live in this tab only and will be lost on navigation"
            >
              Save rubric (soon)
            </Btn>
            {/* Placeholder for the future "replay this rubric on the
                most recent run" action. Honestly disabled until the
                replay endpoint exists - clicking did nothing before
                the fix and the user got no feedback. */}
            <Btn
              kind="secondary"
              size="sm"
              disabled
              title="Replay on recent run is not wired up yet"
            >
              Test on recent run (soon)
            </Btn>
            <Btn
              kind="ghost"
              size="sm"
              onClick={onDiscard}
              disabled={!dirty}
              title="Revert to the last loaded rubric"
            >
              Discard
            </Btn>
          </div>
        </>
      )}
    </Card>
  );
}

/* ============================== ROOT ROUTE =========================== */

export default function Metrics() {
  const project = useProject();
  const navigate = useNavigate();

  const {
    data: list,
    err: listErr,
    refetch: listRefetch,
    reloading: listReloading,
    isInitialLoad: listInitial,
  } = useV2Resource<RubricRow[]>('rubrics', v2.rubrics);

  // Selection lives in the URL (?metric=quality) so links are shareable
  // and a reload / browser-back keeps the user on the row they were
  // looking at. The fallback init effect below picks the first trust
  // row when no URL param is set, matching the prior default-to-top
  // behaviour. setSearchParams uses replace so clicking through the
  // scoreboard doesn't pollute browser history with one entry per row.
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedId = searchParams.get('metric');
  const setSelectedId = useCallback(
    (id: string | null) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (id) next.set('metric', id);
          else next.delete('metric');
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const detailFetcher = useCallback(
    () => v2.rubric(selectedId ?? ''),
    [selectedId],
  );
  const {
    data: detail,
    err: detailErr,
    refetch: detailRefetch,
  } = useV2Resource<RubricDetail>(
    `rubric:${selectedId ?? ''}`,
    detailFetcher,
    { enabled: !!selectedId },
  );

  const trustRows: TrustRow[] = useMemo(() => {
    if (!list) return [];
    return list
      .map((r) => {
        const isProg = r.kind === 'Programmatic';
        const labelKappa = parseLabelKappa(r.calibration_label);
        const detailKappa =
          detail && detail.id === r.id ? detail.calibration.kappa : null;
        const kappa = detailKappa ?? labelKappa;
        const score = deriveTrustScore(kappa, isProg);
        const sample =
          detail && detail.id === r.id ? detail.calibration.sample_size : '-';
        return {
          id: r.id,
          name: r.name,
          metric_type: isProg ? 'programmatic' : 'llm',
          score,
          kappa,
          drift_per_week: 0,
          sample_size: sample,
          needs_work: !isProg && (kappa == null || kappa < 0.7),
          has_kappa: kappa != null,
        } as TrustRow;
      })
      .sort((a, b) => a.score - b.score);
  }, [list, detail]);

  useEffect(() => {
    if (selectedId !== null) return;
    if (trustRows.length === 0) return;
    setSelectedId(trustRows[0].id);
  }, [trustRows, selectedId]);

  useRouteTour(READ_METRICS_TOUR_ID, !!(list && !listErr));

  const [draft, setDraft] = useState<DimensionDraft[]>([]);
  const [draftLoadedFor, setDraftLoadedFor] = useState<string | null>(null);

  const buildInitialDraft = useCallback(
    (d: RubricDetail): DimensionDraft[] => {
      const fp = d.calibration.false_positives_pct ?? 0;
      const fn = d.calibration.false_negatives_pct ?? 0;
      if (d.dimensions.length === 0) {
        return [{ label: d.name, weight: 100, fp, fn }];
      }
      return d.dimensions.map((dim) => ({
        label: dim.label,
        weight: dim.weight_pct,
        fp,
        fn,
      }));
    },
    [],
  );

  useEffect(() => {
    if (!detail) return;
    if (draftLoadedFor === detail.id) return;
    setDraft(buildInitialDraft(detail));
    setDraftLoadedFor(detail.id);
  }, [detail, draftLoadedFor, buildInitialDraft]);

  useEffect(() => {
    setDraftLoadedFor(null);
  }, [selectedId]);

  const initialDraftSerialized = useMemo(() => {
    if (!detail) return '';
    return JSON.stringify(buildInitialDraft(detail));
  }, [detail, buildInitialDraft]);
  const draftSerialized = useMemo(() => JSON.stringify(draft), [draft]);
  const dirty = !!detail && draftSerialized !== initialDraftSerialized;

  function handleDiscard(): void {
    if (!detail) return;
    setDraft(buildInitialDraft(detail));
  }

  // Warn the user before they leave the page (tab close, refresh,
  // hard nav) if the rubric draft has unsaved edits. Modern browsers
  // ignore the custom message text and show their own generic
  // "leave site?" dialog, but the `returnValue` assignment is what
  // triggers the prompt at all. SPA route changes within the
  // dashboard are not guarded here - those are intentional navigations
  // and the dirty pill in the editor card is enough warning.
  useEffect(() => {
    if (!dirty) return;
    function onBeforeUnload(e: BeforeUnloadEvent) {
      e.preventDefault();
      e.returnValue = 'You have unsaved rubric edits.';
    }
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [dirty]);

  const cm: { tp: number; tn: number; fp: number; fn: number } | null =
    useMemo(() => {
      if (!detail) return null;
      const n = detail.calibration.sample_size;
      if (!n || n <= 0) return null;
      const fpPct = detail.calibration.false_positives_pct ?? 0;
      const fnPct = detail.calibration.false_negatives_pct ?? 0;
      const fp = Math.round((fpPct / 100) * n);
      const fn = Math.round((fnPct / 100) * n);
      const correct = Math.max(0, n - fp - fn);
      const tp = Math.ceil(correct / 2);
      const tn = correct - tp;
      return { tp, tn, fp, fn };
    }, [detail]);

  const trustLoading = !list && !listErr;

  return (
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Eyebrow>Metrics - trust report</Eyebrow>
          <UpdatingChip
            visible={listReloading && !listInitial}
            error={list ? listErr : null}
            onRetry={listRefetch}
          />
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
          Which judges should you trust today?
        </h1>

        {listErr && !list && (
          <Card style={{ padding: 16, marginBottom: 14, borderColor: E.fail }}>
            <div role="alert">
              <Eyebrow style={{ color: E.fail }}>Error loading metrics</Eyebrow>
              <div
                style={{
                  fontFamily: E.fMono,
                  fontSize: 12,
                  color: E.text2,
                  marginTop: 6,
                }}
              >
                {listErr}
              </div>
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={() => void listRefetch()}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        <TrustScoreboardCard
          rows={trustRows}
          selectedId={selectedId}
          onSelect={(id) => setSelectedId(id)}
          onRunEval={() => navigate('/commands?prefill=run-eval')}
          loading={trustLoading}
        />

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 320px',
            gap: 14,
          }}
        >
          <RubricEditor
            detail={detail ?? null}
            loading={!!selectedId && !detail && !detailErr}
            err={detailErr}
            onRetry={() => void detailRefetch()}
            draft={draft}
            setDraft={setDraft}
            dirty={dirty}
            onDiscard={handleDiscard}
            drift={
              trustRows.find((r) => r.id === selectedId)?.drift_per_week ?? 0
            }
          />

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {detail ? (
              <>
                <CalibrationProgress
                  metricId={detail.id}
                  sampleSize={detail.calibration.sample_size}
                />
                <ConfusionMatrix
                  cm={cm}
                  sampleSize={detail.calibration.sample_size}
                />
              </>
            ) : (
              <>
                <Card style={{ padding: 14 }}>
                  <Skeleton w="50%" h={11} />
                  <div
                    style={{
                      marginTop: 14,
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                  >
                    <Skeleton w={92} h={92} style={{ borderRadius: '50%' }} />
                  </div>
                </Card>
                <Card style={{ padding: 14 }}>
                  <Skeleton w="60%" h={11} />
                  <div style={{ marginTop: 12 }}>
                    <Skeleton w="100%" h={64} />
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>

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
            why this changed -&gt;
          </span>{' '}
          Calibration is the trust contract for every number on every other
          page. Surfacing it as a scoreboard up front, with a visible
          confusion matrix and per-dimension FP/FN, turns kappa from a number
          into a story. The CLI command preview teaches by example.
        </div>

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
