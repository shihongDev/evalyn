/**
 * AnnotateSession - the active per-item annotation surface.
 *
 * Layout: input/output preview on top, one row per metric below with the
 * AI's pre-label and the user's verdict toggles. Keyboard-first:
 *   - 1, 2, 3, ... cycle that row's verdict (pass -> fail -> skip -> pass)
 *   - A   accept all AI verdicts (pre-fill + ready to submit)
 *   - N or Enter   save the current verdict and advance
 *   - Backspace / U   undo (revert to AI pre-label)
 *   - <- / -> nav between items without saving
 *   - Esc   leave the session (back to /annotate)
 *
 * Resume: visiting /annotate/:id picks up wherever progress left off.
 * The first un-annotated item becomes the cursor.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StatusDot } from '../ui';
import { useV2Resource } from '../hooks/useV2Resource';
import { annotationApi } from '../api/annotation';
import type {
  AnnotationItemRow,
  AnnotationItemsResponse,
  AnnotationLabel,
  AnnotationLabelEntry,
  AnnotationSessionMeta,
} from '../api/types';
import { E } from '../tokens';

const LABEL_CYCLE: Record<AnnotationLabel, AnnotationLabel> = {
  pass: 'fail',
  fail: 'skip',
  skip: 'pass',
};

const LABEL_BG: Record<AnnotationLabel, string> = {
  pass: E.passDim,
  fail: E.failDim,
  skip: E.panel3,
};

const LABEL_FG: Record<AnnotationLabel, string> = {
  pass: E.pass,
  fail: E.fail,
  skip: E.text3,
};

const LABEL_GLYPH: Record<AnnotationLabel, string> = {
  pass: '✓',
  fail: '✗',
  skip: '·',
};

function deriveDefaults(item: AnnotationItemRow, metricIds: string[]): Record<string, AnnotationLabel> {
  const out: Record<string, AnnotationLabel> = {};
  // Already-saved verdicts win.
  for (const lab of item.user_labels) {
    if (lab.metric_id && lab.label) out[lab.metric_id] = lab.label;
  }
  // Otherwise, pre-fill from AI pre-label when present.
  for (const ai of item.ai_labels) {
    if (out[ai.metric_id] !== undefined) continue;
    if (ai.label === 'pass' || ai.label === 'fail') out[ai.metric_id] = ai.label;
  }
  // Ensure every metric has SOME state - default to skip when no signal.
  for (const mid of metricIds) {
    if (out[mid] === undefined) out[mid] = 'skip';
  }
  return out;
}

function aiVerdictMap(item: AnnotationItemRow): Map<string, AnnotationLabel | null> {
  const m = new Map<string, AnnotationLabel | null>();
  for (const ai of item.ai_labels) m.set(ai.metric_id, ai.label);
  return m;
}

export default function AnnotateSession() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const sessionKey = `annotation/session:${sessionId ?? ''}`;
  const { data: session, err: sessionErr, refetch: refetchSession } = useV2Resource<AnnotationSessionMeta>(
    sessionKey,
    () => annotationApi.getSession(sessionId!),
    { enabled: Boolean(sessionId) },
  );

  const itemsKey = `annotation/items:${sessionId ?? ''}`;
  const { data: itemsResp, err: itemsErr, refetch: refetchItems } =
    useV2Resource<AnnotationItemsResponse>(
      itemsKey,
      () => annotationApi.getItems(sessionId!, { limit: 200 }),
      { enabled: Boolean(sessionId) },
    );

  const items = itemsResp?.items ?? [];
  const metricIds = itemsResp?.metric_ids ?? session?.metric_ids ?? [];

  // Cursor: start at the first un-annotated item (resume), else 0.
  const [cursor, setCursor] = useState(0);
  const initialCursorRef = useRef(false);
  useEffect(() => {
    if (initialCursorRef.current || items.length === 0) return;
    const idx = items.findIndex((i) => !i.annotated);
    setCursor(idx >= 0 ? idx : 0);
    initialCursorRef.current = true;
  }, [items]);

  // Per-item local verdict state (keyed by item_id).
  const [verdicts, setVerdicts] = useState<Record<string, Record<string, AnnotationLabel>>>({});
  // Initialize a new item's verdicts the first time we land on it.
  const ensureItemDefaults = useCallback(
    (item: AnnotationItemRow): Record<string, AnnotationLabel> => {
      if (verdicts[item.item_id]) return verdicts[item.item_id];
      const initial = deriveDefaults(item, metricIds);
      setVerdicts((prev) => ({ ...prev, [item.item_id]: initial }));
      return initial;
    },
    [verdicts, metricIds],
  );

  const currentItem = items[cursor];
  const currentVerdict = currentItem ? ensureItemDefaults(currentItem) : {};

  const [submitting, setSubmitting] = useState(false);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [finalizeErr, setFinalizeErr] = useState<string | null>(null);

  const cycleMetric = useCallback(
    (metricId: string) => {
      if (!currentItem) return;
      setVerdicts((prev) => {
        const item = prev[currentItem.item_id] ?? deriveDefaults(currentItem, metricIds);
        const cur = item[metricId] ?? 'skip';
        return {
          ...prev,
          [currentItem.item_id]: { ...item, [metricId]: LABEL_CYCLE[cur] },
        };
      });
    },
    [currentItem, metricIds],
  );

  const acceptAllAi = useCallback(() => {
    if (!currentItem) return;
    setVerdicts((prev) => {
      const next: Record<string, AnnotationLabel> = {};
      const ai = aiVerdictMap(currentItem);
      for (const mid of metricIds) {
        const aiLabel = ai.get(mid);
        next[mid] = aiLabel === 'pass' || aiLabel === 'fail' ? aiLabel : 'skip';
      }
      return { ...prev, [currentItem.item_id]: next };
    });
  }, [currentItem, metricIds]);

  const undoToAi = useCallback(() => {
    if (!currentItem) return;
    setVerdicts((prev) => ({
      ...prev,
      [currentItem.item_id]: deriveDefaults(currentItem, metricIds),
    }));
  }, [currentItem, metricIds]);

  const submitVerdict = useCallback(async (): Promise<boolean> => {
    if (!currentItem || !sessionId) return false;
    const labels: AnnotationLabelEntry[] = [];
    const skipped: string[] = [];
    const ai = aiVerdictMap(currentItem);
    const item = verdicts[currentItem.item_id] ?? deriveDefaults(currentItem, metricIds);
    for (const mid of metricIds) {
      const label = item[mid] ?? 'skip';
      if (label === 'skip') {
        skipped.push(mid);
        continue;
      }
      const aiLabel = ai.get(mid);
      labels.push({
        metric_id: mid,
        label,
        used_ai_verdict: aiLabel === label,
      });
    }
    setSubmitting(true);
    setSubmitErr(null);
    try {
      await annotationApi.postVerdict(sessionId, {
        item_id: currentItem.item_id,
        labels,
        skipped_metrics: skipped,
      });
      // Refresh both session (progress counter) and items (annotated flag)
      // in the background. We don't await refetch since the verdict just
      // posted - UI advances immediately.
      void refetchSession();
      void refetchItems();
      return true;
    } catch (e) {
      setSubmitErr(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setSubmitting(false);
    }
  }, [currentItem, sessionId, verdicts, metricIds, refetchItems, refetchSession]);

  const goNext = useCallback(async () => {
    const ok = await submitVerdict();
    if (ok && cursor < items.length - 1) {
      setCursor((c) => Math.min(items.length - 1, c + 1));
    }
  }, [submitVerdict, cursor, items.length]);

  const goPrev = useCallback(() => {
    setCursor((c) => Math.max(0, c - 1));
  }, []);

  const goNextNoSave = useCallback(() => {
    setCursor((c) => Math.min(items.length - 1, c + 1));
  }, [items.length]);

  // Keyboard handler. Bypassed when focus is in a textarea / input.
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')) return;
      if (!currentItem) return;

      // Number keys cycle the corresponding metric (1 = first metric, etc).
      if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (idx < metricIds.length) {
          e.preventDefault();
          cycleMetric(metricIds[idx]);
        }
        return;
      }
      const k = e.key.toLowerCase();
      if (k === 'a') {
        e.preventDefault();
        acceptAllAi();
      } else if (k === 'n' || e.key === 'Enter') {
        e.preventDefault();
        void goNext();
      } else if (k === 'u' || e.key === 'Backspace') {
        e.preventDefault();
        undoToAi();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goPrev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        goNextNoSave();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        navigate('/annotate');
      }
    }
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentItem, metricIds, cycleMetric, acceptAllAi, goNext, undoToAi, goPrev, goNextNoSave, navigate]);

  // beforeunload: warn if there are unsaved verdicts (cursor points to an
  // item whose UI state hasn't been submitted yet).
  useEffect(() => {
    function handler(e: BeforeUnloadEvent) {
      if (!currentItem || currentItem.annotated) return;
      e.preventDefault();
      e.returnValue = 'You have unsaved verdicts.';
    }
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [currentItem]);

  const progress = useMemo(() => {
    if (!session) return null;
    const pct = session.items_total > 0 ? (session.items_done / session.items_total) * 100 : 0;
    return { done: session.items_done, total: session.items_total, pct };
  }, [session]);

  async function finalizeSession() {
    if (!sessionId) return;
    setFinalizing(true);
    setFinalizeErr(null);
    try {
      await annotationApi.finalize(sessionId);
      navigate('/annotate');
    } catch (e) {
      setFinalizeErr(e instanceof Error ? e.message : String(e));
    } finally {
      setFinalizing(false);
    }
  }

  if (sessionErr || itemsErr) {
    return (
      <AppShell breadcrumb={['Annotate', sessionId ?? '']}>
        <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
          <Card style={{ padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading session</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {sessionErr || itemsErr}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={() => navigate('/annotate')}>
                ← Back to annotate
              </Btn>
            </div>
          </Card>
        </div>
      </AppShell>
    );
  }

  if (!session || !itemsResp) {
    return (
      <AppShell breadcrumb={['Annotate', sessionId ?? '']}>
        <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
          <Skeleton w={300} h={28} />
          <div style={{ marginTop: 16 }}>
            <Skeleton w="100%" h={120} style={{ borderRadius: 8 }} />
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell breadcrumb={['Annotate', `${session.source_kind}:${session.source_id}`]}>
      <div style={{ padding: '24px 36px 30px', maxWidth: 1100 }}>
        {/* HEADER */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <Eyebrow>Annotation session</Eyebrow>
          <Pill mono color={E.text3} bg={E.panel3}>
            {session.id}
          </Pill>
          <span style={{ flex: 1 }} />
          {progress && (
            <span style={{ fontSize: 12, color: E.text2, fontFamily: E.fMono }}>
              {progress.done}/{progress.total} ({progress.pct.toFixed(0)}%)
            </span>
          )}
          <Btn
            kind="primary"
            size="sm"
            onClick={finalizeSession}
            disabled={finalizing || progress?.done === 0}
          >
            {finalizing ? 'Finalizing...' : 'Finish & save'}
          </Btn>
          <Btn kind="ghost" size="sm" onClick={() => navigate('/annotate')}>
            Exit
          </Btn>
        </div>

        {/* PROGRESS BAR */}
        <div
          style={{
            height: 4,
            background: E.panel2,
            borderRadius: 2,
            overflow: 'hidden',
            marginBottom: 18,
          }}
        >
          <div
            style={{
              width: `${progress?.pct ?? 0}%`,
              height: '100%',
              background: E.ember,
              transition: 'width 200ms',
            }}
          />
        </div>

        {finalizeErr && (
          <Card style={{ padding: 12, marginBottom: 14, borderColor: E.fail }}>
            <div style={{ fontSize: 12, color: E.fail, fontFamily: E.fMono }}>{finalizeErr}</div>
          </Card>
        )}

        {/* ITEM CARD */}
        {currentItem ? (
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '14px 18px',
                borderBottom: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <StatusDot status={currentItem.annotated ? 'pass' : 'idle'} size={6} />
              <span style={{ fontSize: 12, color: E.text2, fontFamily: E.fMono }}>
                Item {cursor + 1} of {items.length}
              </span>
              <Pill mono color={E.text3} bg={E.panel3} style={{ fontSize: 10 }}>
                {currentItem.item_id.slice(0, 12)}
              </Pill>
              <span style={{ flex: 1 }} />
              <Btn kind="ghost" size="sm" onClick={goPrev} disabled={cursor === 0}>
                ←
              </Btn>
              <Btn kind="ghost" size="sm" onClick={goNextNoSave} disabled={cursor === items.length - 1}>
                →
              </Btn>
            </div>

            <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <Eyebrow>Input</Eyebrow>
                <div
                  style={{
                    marginTop: 4,
                    padding: 10,
                    background: E.panel2,
                    borderRadius: 6,
                    fontSize: 13,
                    color: E.text1,
                    lineHeight: 1.5,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    maxHeight: 160,
                    overflow: 'auto',
                  }}
                >
                  {currentItem.input_preview || '(empty)'}
                </div>
              </div>
              {currentItem.expected_preview && (
                <div>
                  <Eyebrow>Expected</Eyebrow>
                  <div
                    style={{
                      marginTop: 4,
                      padding: 10,
                      background: E.panel2,
                      borderRadius: 6,
                      fontSize: 13,
                      color: E.text2,
                      lineHeight: 1.5,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      maxHeight: 120,
                      overflow: 'auto',
                    }}
                  >
                    {currentItem.expected_preview}
                  </div>
                </div>
              )}
              {currentItem.output_preview && (
                <div>
                  <Eyebrow>Output</Eyebrow>
                  <div
                    style={{
                      marginTop: 4,
                      padding: 10,
                      background: E.panel,
                      border: `1px solid ${E.hair}`,
                      borderRadius: 6,
                      fontSize: 13,
                      color: E.text1,
                      lineHeight: 1.5,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      maxHeight: 220,
                      overflow: 'auto',
                    }}
                  >
                    {currentItem.output_preview}
                  </div>
                </div>
              )}
            </div>

            {/* METRIC ROWS */}
            <div style={{ borderTop: `1px solid ${E.hair}` }}>
              {metricIds.map((mid, idx) => {
                const userLabel: AnnotationLabel = currentVerdict[mid] ?? 'skip';
                const aiEntry = currentItem.ai_labels.find((a) => a.metric_id === mid);
                const aiLabel = aiEntry?.label ?? null;
                const matchesAi = aiLabel === userLabel && (aiLabel === 'pass' || aiLabel === 'fail');
                return (
                  <div
                    key={mid}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '34px 1fr 110px 130px',
                      alignItems: 'center',
                      gap: 12,
                      padding: '11px 18px',
                      borderTop: idx ? `1px solid ${E.hair}` : 'none',
                    }}
                  >
                    <kbd
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 11,
                        color: E.text2,
                        background: E.panel2,
                        border: `1px solid ${E.hair2}`,
                        borderRadius: 4,
                        padding: '2px 6px',
                        textAlign: 'center',
                      }}
                    >
                      {idx + 1}
                    </kbd>
                    <span style={{ fontFamily: E.fMono, fontSize: 13, color: E.text1 }}>{mid}</span>
                    <Pill
                      mono
                      color={aiLabel ? LABEL_FG[aiLabel] : E.text3}
                      bg={aiLabel ? LABEL_BG[aiLabel] : E.panel3}
                      style={{ fontSize: 10 }}
                    >
                      AI: {aiLabel ?? 'n/a'}
                    </Pill>
                    <button
                      type="button"
                      onClick={() => cycleMetric(mid)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 6,
                        padding: '6px 10px',
                        borderRadius: 6,
                        background: LABEL_BG[userLabel],
                        color: LABEL_FG[userLabel],
                        border: `1px solid ${LABEL_FG[userLabel]}33`,
                        fontFamily: E.fMono,
                        fontSize: 12,
                        fontWeight: 500,
                        cursor: 'pointer',
                      }}
                    >
                      {LABEL_GLYPH[userLabel]} {userLabel}
                      {matchesAi && (
                        <span style={{ fontSize: 9, opacity: 0.6 }}>(AI)</span>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>

            {/* FOOTER */}
            <div
              style={{
                padding: '14px 18px',
                borderTop: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <Btn kind="secondary" size="sm" onClick={acceptAllAi}>
                A · Accept all AI verdicts
              </Btn>
              <Btn kind="secondary" size="sm" onClick={undoToAi}>
                U · Undo
              </Btn>
              <span style={{ flex: 1 }} />
              {submitErr && (
                <span style={{ fontSize: 11, color: E.fail, fontFamily: E.fMono }}>{submitErr}</span>
              )}
              <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
                ←/→ nav · 1-9 cycle metric · Esc exit
              </span>
              <Btn kind="primary" size="sm" onClick={goNext} disabled={submitting}>
                {submitting ? 'Saving...' : 'N · Save & next →'}
              </Btn>
            </div>
          </Card>
        ) : (
          <Card style={{ padding: 18 }}>
            <div style={{ fontSize: 13, color: E.text3 }}>
              No items in this session. Try starting a new session.
            </div>
          </Card>
        )}
      </div>
    </AppShell>
  );
}
