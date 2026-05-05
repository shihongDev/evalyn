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

import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StatusDot } from '../ui';
import { useV2Resource } from '../hooks/useV2Resource';
import { annotationApi } from '../api/annotation';
import type {
  AnnotationEvidence,
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

/** Backend hard-caps preview content at this many chars. Anything at
 * exactly this length is almost certainly truncated, so the Pane shows
 * a "(truncated)" hint to set expectations. Keep in lockstep with
 * _PREVIEW_CHAR_CAP in dashboard/evalyn_dashboard/api/v2/annotation.py. */
const PREVIEW_CHAR_CAP = 8000;

/**
 * Item content pane: Eyebrow label, char count, expand/collapse toggle,
 * scrollable text body. Default collapsed at the supplied maxHeight so
 * the layout stays compact; expanded the cap is removed and the pane
 * grows to fit. State resets on item nav (parent Card keys on item_id).
 */
const Pane = function Pane({
  label,
  text,
  collapsedMaxH,
  emphasized,
  muted,
  bodyRef,
}: {
  label: string;
  text: string;
  collapsedMaxH: number;
  emphasized?: boolean; // output pane gets the contrast bg
  muted?: boolean; // expected pane uses softer text color
  /** Optional ref forwarded to the content div - lets the parent
   * attach selection listeners without duplicating Pane's layout. */
  bodyRef?: (el: HTMLDivElement | null) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  // True when the rendered content actually overflows the collapsed
  // maxHeight, measured from the DOM. Without this the Expand button
  // would appear even on short content (clicks would do nothing visible).
  const [overflowing, setOverflowing] = useState(false);
  const innerRef = useRef<HTMLDivElement | null>(null);

  const length = text.length;
  // Content at exactly the backend cap is almost certainly truncated.
  const probablyTruncated = length >= PREVIEW_CHAR_CAP;
  const naturalCap = expanded ? undefined : collapsedMaxH;

  // Measure overflow each time the text changes. ResizeObserver also
  // catches the case where layout reflows after fonts load.
  useLayoutEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    const measure = () => {
      // scrollHeight excludes maxHeight clipping; clientHeight reflects
      // the visible box. > means there's content the user can't see.
      setOverflowing(el.scrollHeight > el.clientHeight + 1);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [text, collapsedMaxH]);

  // The Expand toggle is only meaningful when (a) we're collapsed and
  // there's hidden content, OR (b) we're already expanded (so the user
  // can re-collapse). Otherwise we hide it entirely.
  const showToggle = overflowing || expanded;

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 4,
        }}
      >
        <Eyebrow>{label}</Eyebrow>
        <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>
          {length.toLocaleString()} chars
          {probablyTruncated && (
            <span style={{ color: E.ember, marginLeft: 4 }}>(truncated by server)</span>
          )}
        </span>
        <span style={{ flex: 1 }} />
        {showToggle && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            style={{
              fontFamily: E.fMono,
              fontSize: 10,
              color: E.text2,
              background: 'transparent',
              border: `1px solid ${E.hair}`,
              borderRadius: 4,
              padding: '2px 8px',
              cursor: 'pointer',
            }}
          >
            {expanded ? 'Collapse' : 'Show all'}
          </button>
        )}
      </div>
      <div
        ref={(el) => {
          innerRef.current = el;
          if (bodyRef) bodyRef(el);
        }}
        style={{
          padding: 10,
          background: emphasized ? E.panel : E.panel2,
          border: emphasized ? `1px solid ${E.hair}` : 'none',
          borderRadius: 6,
          fontSize: 13,
          color: muted ? E.text2 : E.text1,
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          maxHeight: naturalCap,
          overflow: expanded ? 'visible' : 'auto',
        }}
      >
        {text || '(empty)'}
      </div>
    </div>
  );
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

/** localStorage key for in-progress draft state. Per-session so multiple
 * tabs don't clobber each other. Stored as JSON: {verdicts, notes, savedAt}. */
function draftKey(sessionId: string): string {
  return `evalyn:annotation:draft:${sessionId}`;
}

interface DraftBlob {
  verdicts: Record<string, Record<string, AnnotationLabel>>;
  notes: Record<string, string>;
  evidence: Record<string, AnnotationEvidence[]>;
  savedAt: number;
}

function readDraft(sessionId: string): DraftBlob | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(draftKey(sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as DraftBlob;
    if (!parsed || typeof parsed !== 'object') return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeDraft(sessionId: string, blob: Omit<DraftBlob, 'savedAt'>): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      draftKey(sessionId),
      JSON.stringify({ ...blob, savedAt: Date.now() }),
    );
  } catch {
    // Quota / private mode - drafts are best-effort.
  }
}

/** Format a duration in seconds as a compact human string. */
function formatDuration(sec: number): string {
  if (sec < 60) return `${Math.max(1, Math.round(sec))}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function clearDraft(sessionId: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(draftKey(sessionId));
  } catch {
    // ignore
  }
}

/** True when the user picked the opposite of an AI pass/fail verdict.
 * AI = null/skip and user = anything is NOT a disagreement (no signal
 * to disagree with). User = skip with AI = pass/fail is also not a
 * disagreement (the user opted out, didn't override). */
function isDisagreement(userLabel: AnnotationLabel, aiLabel: AnnotationLabel | null): boolean {
  if (aiLabel !== 'pass' && aiLabel !== 'fail') return false;
  if (userLabel !== 'pass' && userLabel !== 'fail') return false;
  return userLabel !== aiLabel;
}

/** Count disagreements across all annotated items. Reads the persisted
 * user_labels from the server (not local UI state) so the badge reflects
 * what's saved, not what's currently being typed. */
function countOverrides(items: AnnotationItemRow[]): number {
  let n = 0;
  for (const it of items) {
    if (!it.annotated) continue;
    const ai = aiVerdictMap(it);
    for (const ul of it.user_labels) {
      if (isDisagreement(ul.label, ai.get(ul.metric_id) ?? null)) n += 1;
    }
  }
  return n;
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
    // Same pattern for evidence: only seed unseen item ids so local
    // edits in this tab take precedence over the server snapshot.
    setEvidence((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const it of items) {
        if (next[it.item_id] === undefined && it.evidence && it.evidence.length > 0) {
          next[it.item_id] = it.evidence;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [items]);

  // Per-item local verdict state (keyed by item_id). Hydrated lazily from
  // the localStorage draft so an accidental tab close doesn't lose
  // in-progress UI state.
  const [verdicts, setVerdicts] = useState<Record<string, Record<string, AnnotationLabel>>>(() => {
    if (!sessionId) return {};
    return readDraft(sessionId)?.verdicts ?? {};
  });
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
  // Hydrated from localStorage draft alongside verdicts.
  const [notes, setNotes] = useState<Record<string, string>>(() => {
    if (!sessionId) return {};
    return readDraft(sessionId)?.notes ?? {};
  });
  // Per-item evidence snippets - text the user highlighted in the
  // output pane, optionally tagged with a metric. Hydrated from draft
  // and seeded from server records when present (similar to notes).
  const [evidence, setEvidence] = useState<Record<string, AnnotationEvidence[]>>(() => {
    if (!sessionId) return {};
    return readDraft(sessionId)?.evidence ?? {};
  });
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

  // Persist verdicts + notes + evidence to localStorage on every change.
  // We keep it simple - every change writes synchronously. For typical
  // session sizes (<300 items) the JSON cost is negligible.
  useEffect(() => {
    if (!sessionId) return;
    writeDraft(sessionId, { verdicts, notes, evidence });
  }, [sessionId, verdicts, notes, evidence]);

  const [submitting, setSubmitting] = useState(false);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [finalizeErr, setFinalizeErr] = useState<string | null>(null);

  // Refs for the optional note textarea so "/" can focus it without
  // tripping the global keyboard handler.
  const noteRef = useRef<HTMLTextAreaElement | null>(null);

  // Output pane element ref - so the selection listener can decide
  // whether the user's selection is inside the output (eligible for
  // "mark as evidence") vs in some other text on the page.
  const outputBodyRef = useRef<HTMLDivElement | null>(null);
  // Floating popover state when the user has selected text in the
  // output pane. null = no popover. The snippet is captured at
  // selection time so it survives the user clicking the popover
  // (which would normally collapse the selection).
  const [evidencePopover, setEvidencePopover] = useState<{
    snippet: string;
    x: number;
    y: number;
  } | null>(null);

  // Listen for mouseup anywhere; if a non-empty selection lives inside
  // the output pane, show the popover near the selection's bottom edge.
  // Listener is per-component-mount so it's torn down on unmount.
  useEffect(() => {
    function onMouseUp() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) {
        setEvidencePopover(null);
        return;
      }
      const text = sel.toString();
      if (!text || text.trim().length === 0) {
        setEvidencePopover(null);
        return;
      }
      // Selection must be inside the output pane to count as evidence.
      const anchor = sel.anchorNode;
      const focus = sel.focusNode;
      const out = outputBodyRef.current;
      if (!out || !anchor || !focus) return;
      if (!out.contains(anchor) || !out.contains(focus)) {
        setEvidencePopover(null);
        return;
      }
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      // Position the popover horizontally centered under the selection,
      // capped within the viewport so it doesn't clip off-screen.
      const x = Math.max(160, Math.min(window.innerWidth - 160, rect.left + rect.width / 2));
      const y = rect.bottom + 8;
      // Cap snippet length client-side so we don't ship megabytes.
      const snippet = text.length > 2000 ? text.slice(0, 2000) : text;
      setEvidencePopover({ snippet, x, y });
    }
    document.addEventListener('mouseup', onMouseUp);
    return () => document.removeEventListener('mouseup', onMouseUp);
  }, []);

  // Reset the popover when navigating items - selection is per-item.
  useEffect(() => {
    setEvidencePopover(null);
  }, [cursor]);

  // Add a piece of evidence to the current item. metricId === null
  // means "item-level evidence not tied to any specific metric".
  const addEvidence = useCallback(
    (snippet: string, metricId: string | null, note: string | null) => {
      if (!currentItem) return;
      setEvidence((prev) => {
        const cur = prev[currentItem.item_id] ?? [];
        const next: AnnotationEvidence = { snippet, metric_id: metricId, note };
        return { ...prev, [currentItem.item_id]: [...cur, next] };
      });
      setEvidencePopover(null);
      window.getSelection()?.removeAllRanges();
    },
    [currentItem],
  );

  // Remove an evidence entry by index for the current item.
  const removeEvidence = useCallback(
    (idx: number) => {
      if (!currentItem) return;
      setEvidence((prev) => {
        const cur = prev[currentItem.item_id] ?? [];
        const next = cur.slice(0, idx).concat(cur.slice(idx + 1));
        return { ...prev, [currentItem.item_id]: next };
      });
    },
    [currentItem],
  );

  // Per-item timing for the "Xs/item · ~Ymin left" header chip.
  // We capture wall-clock between successful saves into a 10-deep ring,
  // capping each delta at 5 minutes so an idle break doesn't poison
  // the average. timingTick exists only to force a re-render when the
  // ring updates - the actual data lives in refs to avoid render churn.
  const lastSaveAtRef = useRef<number>(Date.now());
  const itemDurationsRef = useRef<number[]>([]);
  const [timingTick, setTimingTick] = useState(0);

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
      const evidenceForItem = evidence[currentItem.item_id] ?? [];
      try {
        await annotationApi.postVerdict(sessionId, {
          item_id: currentItem.item_id,
          labels,
          skipped_metrics: skipped,
          note: noteForItem || null,
          evidence: evidenceForItem,
        });
        setSavedTick((t) => t + 1);
        // Capture per-item wall-clock for the avg/ETA header chip.
        const now = Date.now();
        const deltaSec = (now - lastSaveAtRef.current) / 1000;
        if (deltaSec > 0 && deltaSec < 300) {
          // Cap at 5 min to filter out idle gaps that would skew the avg.
          itemDurationsRef.current.push(deltaSec);
          if (itemDurationsRef.current.length > 10) itemDurationsRef.current.shift();
          setTimingTick((t) => t + 1);
        }
        lastSaveAtRef.current = now;
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
    [currentItem, sessionId, verdicts, metricIds, notes, evidence, refetchItems, refetchSession],
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

  // Disagreement count across all annotated items - shows the user how
  // often they're overriding AI pre-labels. Helps catch calibration drift.
  const overridesCount = useMemo(() => countOverrides(items), [items]);

  // True when every loaded item has been annotated. Drives the
  // celebratory all-done state below.
  const allDone = items.length > 0 && items.every((it) => it.annotated);

  // Rolling average per-item duration (seconds) and ETA. Recomputed
  // on every successful save (timingTick is the trigger). Null until
  // the first sample lands so we don't render misleading numbers.
  const timing = useMemo(() => {
    const arr = itemDurationsRef.current;
    if (arr.length === 0) return null;
    const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
    const remaining = progress ? Math.max(0, progress.total - progress.done) * avg : 0;
    return { avgSec: avg, remainingSec: remaining };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timingTick, progress?.total, progress?.done]);

  async function finalizeSession() {
    if (!sessionId) return;
    setFinalizing(true);
    setFinalizeErr(null);
    try {
      await annotationApi.finalize(sessionId);
      // Draft is no longer useful once verdicts are merged on disk.
      clearDraft(sessionId);
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
          {timing && progress && progress.done < progress.total && (
            <span
              style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}
              title={`Rolling average across your last ${itemDurationsRef.current.length} item${itemDurationsRef.current.length === 1 ? '' : 's'}.`}
            >
              ~{formatDuration(timing.avgSec)}/item · ~{formatDuration(timing.remainingSec)} left
            </span>
          )}
          {overridesCount > 0 && (
            <Pill
              mono
              color={E.ember}
              bg="#fcefe2"
              style={{ fontSize: 10 }}
              title={`You overrode the AI's pass/fail on ${overridesCount} verdict${overridesCount === 1 ? '' : 's'}.`}
            >
              {overridesCount} override{overridesCount === 1 ? '' : 's'}
            </Pill>
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

        {/* All-done celebratory banner. Stays above the item card so the
            user can still ←/→ through items to revise before finalizing. */}
        {allDone && (
          <Card
            style={{
              padding: 16,
              marginBottom: 14,
              borderColor: E.ember,
              background: '#fcefe2',
              display: 'flex',
              alignItems: 'center',
              gap: 14,
            }}
          >
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 50,
                background: E.ember,
                color: '#fff8f1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 18,
                flexShrink: 0,
              }}
            >
              ✓
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: E.fSerif, fontSize: 16, color: E.text1 }}>
                All {items.length} item{items.length === 1 ? '' : 's'} annotated.
              </div>
              <div
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: E.text2,
                  marginTop: 2,
                }}
              >
                {overridesCount > 0
                  ? `You overrode AI on ${overridesCount} verdict${overridesCount === 1 ? '' : 's'}. `
                  : ''}
                Finish & save to merge into the dataset annotations.
              </div>
            </div>
            <Btn
              kind="primary"
              size="md"
              onClick={finalizeSession}
              disabled={finalizing}
            >
              {finalizing ? 'Finalizing...' : 'Finish & save →'}
            </Btn>
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

            <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <Pane label="Input" text={currentItem.input_preview ?? ''} collapsedMaxH={260} />
              {currentItem.expected_preview && (
                <Pane
                  label="Expected"
                  text={currentItem.expected_preview}
                  collapsedMaxH={220}
                  muted
                />
              )}
              {currentItem.output_preview && (
                <Pane
                  label="Output"
                  text={currentItem.output_preview}
                  collapsedMaxH={420}
                  emphasized
                  bodyRef={(el) => {
                    outputBodyRef.current = el;
                  }}
                />
              )}
            </div>

            {/* METRIC ROWS */}
            <div style={{ borderTop: `1px solid ${E.hair}` }}>
              {metricIds.map((mid, idx) => {
                const userLabel: AnnotationLabel = currentVerdict[mid] ?? 'skip';
                const aiEntry = currentItem.ai_labels.find((a) => a.metric_id === mid);
                const aiLabel = aiEntry?.label ?? null;
                const aiScore = aiEntry?.score ?? null;
                const matchesAi = aiLabel === userLabel && (aiLabel === 'pass' || aiLabel === 'fail');
                const overridingAi = isDisagreement(userLabel, aiLabel);
                const fk = flipKey[`${currentItem.item_id}:${mid}`] ?? 0;
                return (
                  <div
                    key={mid}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '34px 1fr 110px 130px',
                      alignItems: 'center',
                      gap: 12,
                      padding: '11px 14px 11px 14px',
                      borderTop: idx ? `1px solid ${E.hair}` : 'none',
                      // 4px ember stripe on the left when this row's
                      // verdict overrides the AI - draws the eye to
                      // disagreement without changing layout.
                      borderLeft: overridingAi ? `4px solid ${E.ember}` : '4px solid transparent',
                      background: overridingAi ? '#fcefe2' : undefined,
                      transition: 'background 160ms, border-left-color 160ms',
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
                      title={
                        aiScore != null
                          ? `Model confidence: ${aiScore.toFixed(2)}`
                          : 'No AI verdict for this metric.'
                      }
                    >
                      AI: {aiLabel ?? 'n/a'}
                      {aiScore != null && (
                        <span style={{ marginLeft: 4, opacity: 0.65 }}>
                          {aiScore.toFixed(2)}
                        </span>
                      )}
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

            {/* EVIDENCE - text the user highlighted from the output as
                justification. Each entry can be metric-tagged (or item-
                level when metric_id is null). Click X to remove.
                Hidden entirely when there's no evidence yet to keep the
                surface uncluttered for users who don't use this feature. */}
            {(() => {
              const list = evidence[currentItem.item_id] ?? [];
              if (list.length === 0) return null;
              return (
                <div
                  style={{
                    borderTop: `1px solid ${E.hair}`,
                    padding: '12px 18px',
                    background: '#faf6ec',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      marginBottom: 8,
                    }}
                  >
                    <Eyebrow>Evidence</Eyebrow>
                    <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>
                      {list.length} snippet{list.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {list.map((ev, i) => (
                      <div
                        key={`${i}-${ev.snippet.slice(0, 12)}`}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: 8,
                          padding: '6px 8px',
                          background: '#fff8eb',
                          border: `1px solid ${E.hair}`,
                          borderRadius: 6,
                        }}
                      >
                        <Pill
                          mono
                          color={ev.metric_id ? E.ember : E.text3}
                          bg={ev.metric_id ? '#fcefe2' : E.panel3}
                          style={{ fontSize: 10, flexShrink: 0 }}
                        >
                          {ev.metric_id ?? 'item'}
                        </Pill>
                        <div
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 12,
                            color: E.text1,
                            flex: 1,
                            wordBreak: 'break-word',
                            whiteSpace: 'pre-wrap',
                            lineHeight: 1.4,
                          }}
                        >
                          "{ev.snippet}"
                          {ev.note && (
                            <div
                              style={{
                                fontSize: 11,
                                color: E.text2,
                                marginTop: 2,
                                fontStyle: 'italic',
                              }}
                            >
                              — {ev.note}
                            </div>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={() => removeEvidence(i)}
                          title="Remove this evidence"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: E.text3,
                            cursor: 'pointer',
                            fontFamily: E.fMono,
                            fontSize: 14,
                            padding: '0 4px',
                            lineHeight: 1,
                          }}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

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

      {/* FLOATING EVIDENCE POPOVER - appears anchored to a text selection
          inside the Output pane. Lets the user attach the snippet to a
          specific metric (or item-level) before saving. Position is
          fixed to viewport so it survives page scroll between selection
          and click. */}
      {evidencePopover && currentItem && (
        <EvidencePopover
          snippet={evidencePopover.snippet}
          x={evidencePopover.x}
          y={evidencePopover.y}
          metricIds={metricIds}
          onSave={(metricId, note) =>
            addEvidence(evidencePopover.snippet, metricId, note)
          }
          onCancel={() => setEvidencePopover(null)}
        />
      )}
    </AppShell>
  );
}

/**
 * Floating popover for "mark this selection as evidence". Renders fixed
 * to viewport coordinates so it tracks the selection without needing
 * to know about page scroll. Closed by Save, Cancel, or Esc.
 */
function EvidencePopover({
  snippet,
  x,
  y,
  metricIds,
  onSave,
  onCancel,
}: {
  snippet: string;
  x: number;
  y: number;
  metricIds: string[];
  onSave: (metricId: string | null, note: string | null) => void;
  onCancel: () => void;
}) {
  // Default to "item-level" (no metric) - prompts the user to pick.
  const [metricId, setMetricId] = useState<string | null>(null);
  const [note, setNote] = useState('');

  // Close on Esc. Capture-phase so it beats the global keyboard handler.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        onCancel();
      }
    }
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [onCancel]);

  return (
    <div
      role="dialog"
      aria-label="Mark as evidence"
      // Stop propagation so the global mouseup-selection listener
      // doesn't react to clicks inside the popover. Without these,
      // clicking the metric dropdown or note input would shift the
      // browser selection and immediately close the popover.
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'fixed',
        left: x,
        top: y,
        transform: 'translateX(-50%)',
        zIndex: 100,
        background: '#fbf7ee',
        border: `1px solid ${E.hair2}`,
        borderRadius: 8,
        boxShadow: '0 8px 24px rgba(20,18,14,0.16)',
        padding: 12,
        minWidth: 280,
        maxWidth: 360,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      <div
        style={{
          fontFamily: E.fMono,
          fontSize: 11,
          color: E.text2,
          background: E.panel2,
          border: `1px solid ${E.hair}`,
          borderRadius: 4,
          padding: '6px 8px',
          maxHeight: 80,
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        "{snippet.length > 240 ? `${snippet.slice(0, 240)}…` : snippet}"
      </div>
      <label
        style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: E.text2 }}
      >
        <span style={{ fontFamily: E.fMono }}>For metric:</span>
        <select
          value={metricId ?? ''}
          onChange={(e) => setMetricId(e.target.value || null)}
          style={{
            flex: 1,
            fontFamily: E.fMono,
            fontSize: 12,
            padding: '4px 6px',
            background: E.panel,
            border: `1px solid ${E.hair}`,
            borderRadius: 4,
            color: E.text1,
          }}
        >
          <option value="">(item-level)</option>
          {metricIds.map((mid) => (
            <option key={mid} value={mid}>
              {mid}
            </option>
          ))}
        </select>
      </label>
      <input
        type="text"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Optional note (why?)"
        style={{
          fontFamily: E.fMono,
          fontSize: 12,
          padding: '6px 8px',
          background: E.panel,
          border: `1px solid ${E.hair}`,
          borderRadius: 4,
          color: E.text1,
          outline: 'none',
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = E.ember;
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = E.hair;
        }}
        // Submit on Enter; Esc handled at document level.
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            onSave(metricId, note.trim() || null);
          }
        }}
      />
      <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
        <Btn kind="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Btn>
        <Btn
          kind="primary"
          size="sm"
          onClick={() => onSave(metricId, note.trim() || null)}
        >
          Add evidence
        </Btn>
      </div>
    </div>
  );
}
