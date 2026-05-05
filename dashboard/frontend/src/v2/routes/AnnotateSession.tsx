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

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

/**
 * Compact, scannable keyboard cheat chip rendered in the footer.
 *
 * Single hover-revealed tooltip with the full hotkey table. Default
 * surface stays a single line so the footer doesn't overwhelm.
 */
function KeyHints() {
  const [open, setOpen] = useState(false);
  const KEYS: Array<[string, string]> = [
    ['1-9', 'cycle metric'],
    ['A', 'accept all AI'],
    ['N / ⏎', 'save + next'],
    ['U / ⌫', 'undo'],
    ['S', 'skip all + next'],
    ['/', 'focus note'],
    ['← / →', 'navigate'],
    ['Esc', 'exit'],
  ];
  return (
    <div
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        style={{
          fontFamily: E.fMono,
          fontSize: 11,
          color: E.text3,
          padding: '4px 8px',
          borderRadius: 4,
          border: `1px solid ${E.hair}`,
          background: E.panel2,
          cursor: 'help',
          userSelect: 'none',
        }}
      >
        ⌨ keys
      </span>
      {open && (
        <div
          role="tooltip"
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 6px)',
            right: 0,
            background: E.panel,
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            padding: '8px 10px',
            boxShadow: '0 6px 18px rgba(20,18,14,0.10)',
            display: 'grid',
            gridTemplateColumns: 'auto auto',
            columnGap: 14,
            rowGap: 4,
            zIndex: 50,
            minWidth: 200,
          }}
        >
          {KEYS.map(([k, d]) => (
            <Fragment key={k}>
              <kbd
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: E.text1,
                  background: E.panel2,
                  border: `1px solid ${E.hair2}`,
                  borderRadius: 3,
                  padding: '1px 6px',
                  textAlign: 'center',
                }}
              >
                {k}
              </kbd>
              <span style={{ fontSize: 11, color: E.text2 }}>{d}</span>
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

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

  // Seed local notes from any persisted note on each item the first time
  // we see it. User edits override; we only re-seed for unseen item ids.
  useEffect(() => {
    if (items.length === 0) return;
    setNotes((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const it of items) {
        if (next[it.item_id] === undefined && it.note) {
          next[it.item_id] = it.note;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
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

  // Per-item free-text notes. Reset only when the user clears the field.
  const [notes, setNotes] = useState<Record<string, string>>({});
  // Per-item flip animation key: bump when a metric cycles so the pill
  // gets a fresh CSS animation. Stored as a counter rather than a flag
  // so consecutive cycles each trigger.
  const [flipKey, setFlipKey] = useState<Record<string, number>>({});
  // "Saved" pulse counter: bumps on successful verdict POST. The current
  // value is rendered as a key on the badge so each save retriggers
  // the keyframe animation. `savedFlash` is the rendering toggle - it
  // unmounts the badge after 1.4s so the flash auto-clears even when
  // prefers-reduced-motion suppresses the fade-out keyframe.
  const [savedTick, setSavedTick] = useState(0);
  const [savedFlash, setSavedFlash] = useState(false);
  useEffect(() => {
    if (savedTick === 0) return;
    setSavedFlash(true);
    const t = setTimeout(() => setSavedFlash(false), 1400);
    return () => clearTimeout(t);
  }, [savedTick]);

  const [submitting, setSubmitting] = useState(false);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [finalizeErr, setFinalizeErr] = useState<string | null>(null);

  // Refs for the optional note textarea so "/" can focus it without
  // tripping the global keyboard handler.
  const noteRef = useRef<HTMLTextAreaElement | null>(null);

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
      // Bump flip key to retrigger the pill flip animation.
      setFlipKey((prev) => ({
        ...prev,
        [`${currentItem.item_id}:${metricId}`]:
          (prev[`${currentItem.item_id}:${metricId}`] ?? 0) + 1,
      }));
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

  // Submit the current item's verdicts. Optionally accepts an explicit
  // verdict map (override) so callers like the "S" hotkey can submit
  // synthetic state without waiting for React to commit setVerdicts.
  const submitVerdict = useCallback(
    async (override?: Record<string, AnnotationLabel>): Promise<boolean> => {
      if (!currentItem || !sessionId) return false;
      const labels: AnnotationLabelEntry[] = [];
      const skipped: string[] = [];
      const ai = aiVerdictMap(currentItem);
      const item = override ?? verdicts[currentItem.item_id] ?? deriveDefaults(currentItem, metricIds);
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
      const noteForItem = (notes[currentItem.item_id] ?? '').trim();
      try {
        await annotationApi.postVerdict(sessionId, {
          item_id: currentItem.item_id,
          labels,
          skipped_metrics: skipped,
          note: noteForItem || null,
        });
        setSavedTick((t) => t + 1);
        void refetchSession();
        void refetchItems();
        return true;
      } catch (e) {
        setSubmitErr(e instanceof Error ? e.message : String(e));
        return false;
      } finally {
        setSubmitting(false);
      }
    },
    [currentItem, sessionId, verdicts, metricIds, notes, refetchItems, refetchSession],
  );

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
      } else if (k === '/') {
        // "/" focuses the per-item note textarea so the user can type
        // a quick rationale without reaching for the mouse.
        e.preventDefault();
        noteRef.current?.focus();
      } else if (k === 's') {
        // "s" marks every metric on this item as skip and advances.
        // Useful for items the annotator wants to defer / can't judge.
        // We build the skip-all map locally and pass it as an override
        // so submitVerdict doesn't race React's commit cycle.
        e.preventDefault();
        const skipAll: Record<string, AnnotationLabel> = {};
        for (const mid of metricIds) skipAll[mid] = 'skip';
        setVerdicts((prev) => ({ ...prev, [currentItem.item_id]: skipAll }));
        void (async () => {
          const ok = await submitVerdict(skipAll);
          if (ok && cursor < items.length - 1) {
            setCursor((c) => Math.min(items.length - 1, c + 1));
          }
        })();
      }
    }
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [
    currentItem,
    metricIds,
    cycleMetric,
    acceptAllAi,
    goNext,
    undoToAi,
    goPrev,
    goNextNoSave,
    navigate,
    submitVerdict,
    cursor,
    items.length,
  ]);

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
          {/* "Saved" pulse: rendered for 1.4s after each successful POST.
              `savedFlash` toggles mount/unmount; `savedTick` keys the node
              so consecutive saves retrigger the fade-in keyframe. */}
          {savedFlash && (
            <span
              key={savedTick}
              style={{
                fontFamily: E.fMono,
                fontSize: 11,
                color: E.pass,
                background: E.passDim,
                border: `1px solid ${E.pass}33`,
                borderRadius: 4,
                padding: '2px 7px',
                animation: 'eSavedPop 1.4s ease-out forwards',
                pointerEvents: 'none',
              }}
            >
              ✓ Saved
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

        {/* SCRUBBER STRIP - one dot per item, click to jump.
            Color encodes state: ember = current, pass green = annotated,
            hairline = todo. Caps at 60 dots; if more, we render a sliding
            window around the cursor so the strip stays scannable. */}
        {items.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              marginBottom: 6,
              flexWrap: 'wrap',
            }}
            aria-label="Item progress strip"
          >
            {(() => {
              const MAX_DOTS = 60;
              let start = 0;
              let end = items.length;
              if (items.length > MAX_DOTS) {
                start = Math.max(0, cursor - Math.floor(MAX_DOTS / 2));
                end = Math.min(items.length, start + MAX_DOTS);
                start = Math.max(0, end - MAX_DOTS);
              }
              return items.slice(start, end).map((it, j) => {
                const i = start + j;
                const isCurrent = i === cursor;
                const bg = isCurrent
                  ? E.ember
                  : it.annotated
                    ? E.pass
                    : E.hair2;
                const border = isCurrent ? `2px solid ${E.ember}` : 'none';
                return (
                  <button
                    key={it.item_id}
                    type="button"
                    onClick={() => setCursor(i)}
                    title={`Item ${i + 1}${it.annotated ? ' · annotated' : ''}`}
                    style={{
                      width: isCurrent ? 12 : 8,
                      height: isCurrent ? 12 : 8,
                      borderRadius: 50,
                      background: bg,
                      border,
                      padding: 0,
                      cursor: 'pointer',
                      transition: 'all 160ms',
                      outline: 'none',
                    }}
                  />
                );
              });
            })()}
            {items.length > 60 && (
              <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3, marginLeft: 6 }}>
                window {Math.max(1, cursor - 29)}-{Math.min(items.length, cursor + 30)} of {items.length}
              </span>
            )}
          </div>
        )}

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

        {/* ITEM CARD - keyed on cursor so each navigation triggers
            the slide-in animation. Cheap retrigger via key swap. */}
        {currentItem ? (
          <Card
            key={currentItem.item_id}
            style={{
              padding: 0,
              overflow: 'hidden',
              animation: 'eItemSlideIn 240ms ease-out',
            }}
          >
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
                const fk = flipKey[`${currentItem.item_id}:${mid}`] ?? 0;
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
                      key={`${currentItem.item_id}:${mid}:${fk}`}
                      type="button"
                      onClick={() => cycleMetric(mid)}
                      title={`Click or press ${idx + 1} to cycle (pass → fail → skip)`}
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
                        // Animation only after the user has cycled at least
                        // once - prevents a noisy flip cascade on item arrival.
                        animation: fk > 0 ? 'eVerdictFlip 220ms ease-out' : undefined,
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

            {/* NOTE - free-text rationale, optional. Persisted with the
                next verdict POST. "/" focuses without breaking the
                global hotkeys. */}
            <div style={{ borderTop: `1px solid ${E.hair}`, padding: '12px 18px' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  marginBottom: 6,
                }}
              >
                <Eyebrow>Note (optional)</Eyebrow>
                <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>
                  press / to focus
                </span>
                <span style={{ flex: 1 }} />
                {(notes[currentItem.item_id] ?? '').length > 0 && (
                  <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>
                    {(notes[currentItem.item_id] ?? '').length} chars
                  </span>
                )}
              </div>
              <textarea
                ref={noteRef}
                value={notes[currentItem.item_id] ?? ''}
                onChange={(e) =>
                  setNotes((prev) => ({ ...prev, [currentItem.item_id]: e.target.value }))
                }
                placeholder="Why this verdict? (saved with N)"
                rows={2}
                style={{
                  width: '100%',
                  resize: 'vertical',
                  padding: '8px 10px',
                  background: E.panel2,
                  border: `1px solid ${E.hair}`,
                  borderRadius: 6,
                  fontFamily: E.fMono,
                  fontSize: 12,
                  color: E.text1,
                  lineHeight: 1.5,
                  outline: 'none',
                  minHeight: 36,
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = E.ember;
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = E.hair;
                }}
              />
            </div>

            {/* FOOTER */}
            <div
              style={{
                padding: '14px 18px',
                borderTop: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 8,
              }}
            >
              <Btn kind="secondary" size="sm" onClick={acceptAllAi}>
                A · Accept all AI
              </Btn>
              <Btn kind="secondary" size="sm" onClick={undoToAi}>
                U · Undo
              </Btn>
              <span style={{ flex: 1, minWidth: 8 }} />
              {submitErr && (
                <span style={{ fontSize: 11, color: E.fail, fontFamily: E.fMono }}>{submitErr}</span>
              )}
              <KeyHints />
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
