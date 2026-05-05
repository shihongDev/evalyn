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
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
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

/** Render text with each evidence snippet wrapped in a highlight span.
 *
 * Builds a span tree by:
 *   1. Finding every (start,end) range where any snippet matches in text.
 *   2. Merging overlapping ranges so we never nest <mark>s.
 *   3. Walking the merged ranges to emit alternating plain/marked nodes.
 *
 * Each highlight gets the snippet index as a data-attr so click handlers
 * (or future hover effects) can flash the matching evidence row. Empty or
 * not-found snippets are silently skipped - we never crash on bad input.
 */
function renderWithHighlights(
  text: string,
  snippets: string[],
  onClick: (snippetIdx: number) => void,
): React.ReactNode {
  if (!text || snippets.length === 0) return text;
  type Range = { start: number; end: number; snippetIdx: number };
  const ranges: Range[] = [];
  snippets.forEach((s, i) => {
    if (!s || s.trim().length === 0) return;
    let cursor = 0;
    while (cursor < text.length) {
      const idx = text.indexOf(s, cursor);
      if (idx === -1) break;
      ranges.push({ start: idx, end: idx + s.length, snippetIdx: i });
      cursor = idx + s.length;
    }
  });
  if (ranges.length === 0) return text;
  ranges.sort((a, b) => a.start - b.start || a.end - b.end);
  // Merge overlaps; keep the first snippet idx that opened the range
  // so the click handler points at SOMETHING, even if multiple snippets
  // happen to overlap on the same text.
  const merged: Range[] = [];
  for (const r of ranges) {
    const top = merged[merged.length - 1];
    if (!top || top.end < r.start) {
      merged.push({ ...r });
    } else if (r.end > top.end) {
      top.end = r.end;
    }
  }
  const nodes: React.ReactNode[] = [];
  let cur = 0;
  merged.forEach((r, i) => {
    if (cur < r.start) nodes.push(text.slice(cur, r.start));
    nodes.push(
      <mark
        key={`m${i}`}
        data-evidence-idx={r.snippetIdx}
        onClick={() => onClick(r.snippetIdx)}
        title="Saved evidence — click to locate in the list below"
        style={{
          background: '#fff8c8',
          color: 'inherit',
          borderBottom: '2px solid #d96a2c',
          padding: '0 1px',
          borderRadius: 2,
          cursor: 'pointer',
        }}
      >
        {text.slice(r.start, r.end)}
      </mark>,
    );
    cur = r.end;
  });
  if (cur < text.length) nodes.push(text.slice(cur));
  return nodes;
}

/** Compact word-level diff between two strings using LCS DP.
 *
 * Returns a flat list of {type, text} ops:
 *   - 'eq'  : in both strings (rendered normally)
 *   - 'del' : in `a` (expected), missing from `b` (output)
 *   - 'ins' : in `b` (output), not in `a` (expected)
 *
 * Caps each side at `maxWords`. Returns null when either side exceeds
 * the cap so the UI can render a "diff disabled - too large" notice
 * instead of locking the page on a giant DP table.
 *
 * Memory: Uint16Array of (m+1)*(n+1). At maxWords=1500 this is ~4.5 MB,
 * well within main-thread budgets for an interactive toggle.
 */
type DiffOp = { type: 'eq' | 'del' | 'ins'; text: string };

function wordDiff(a: string, b: string, maxWords = 1500): DiffOp[] | null {
  // Split keeping the whitespace runs so we reconstruct spacing on render.
  const aw = a.split(/(\s+)/).filter((s) => s.length > 0);
  const bw = b.split(/(\s+)/).filter((s) => s.length > 0);
  if (aw.length > maxWords || bw.length > maxWords) return null;
  const m = aw.length;
  const n = bw.length;
  if (m === 0 && n === 0) return [];
  // dp[i*(n+1)+j] = LCS length from aw[i:] vs bw[j:]
  const dp = new Uint16Array((m + 1) * (n + 1));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      const cell = i * (n + 1) + j;
      if (aw[i] === bw[j]) {
        dp[cell] = dp[(i + 1) * (n + 1) + (j + 1)] + 1;
      } else {
        const down = dp[(i + 1) * (n + 1) + j];
        const right = dp[i * (n + 1) + (j + 1)];
        dp[cell] = down > right ? down : right;
      }
    }
  }
  const out: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (aw[i] === bw[j]) {
      out.push({ type: 'eq', text: aw[i] });
      i++;
      j++;
    } else if (dp[(i + 1) * (n + 1) + j] >= dp[i * (n + 1) + (j + 1)]) {
      out.push({ type: 'del', text: aw[i] });
      i++;
    } else {
      out.push({ type: 'ins', text: bw[j] });
      j++;
    }
  }
  while (i < m) out.push({ type: 'del', text: aw[i++] });
  while (j < n) out.push({ type: 'ins', text: bw[j++] });
  return out;
}

/** Render a word-level diff of (expected, output) as inline spans.
 *
 * Color semantics (chosen to match the cream palette + pass/fail colors):
 *   - 'eq'  : neutral text, no background.
 *   - 'ins' : in output, missing from expected → leaf-green tint.
 *   - 'del' : in expected, missing from output → cinnabar tint, struck.
 *
 * On too-large content, returns a notice instead of a giant DP table.
 */
function renderDiffBody(expected: string, output: string): React.ReactNode {
  const ops = wordDiff(expected, output);
  if (ops === null) {
    return (
      <em style={{ fontSize: 12, color: '#94907f' }}>
        Diff disabled — content too long. Toggle off to see raw output.
      </em>
    );
  }
  if (ops.length === 0) {
    return (
      <em style={{ fontSize: 12, color: '#94907f' }}>
        Both expected and output are empty.
      </em>
    );
  }
  return ops.map((op, i) => {
    if (op.type === 'eq') return <span key={i}>{op.text}</span>;
    if (op.type === 'ins') {
      return (
        <span
          key={i}
          title="In output, not in expected"
          style={{
            background: '#e8f5e9',
            color: '#2e7d32',
            borderBottom: '1.5px solid #2e7d32',
            padding: '0 1px',
            borderRadius: 2,
          }}
        >
          {op.text}
        </span>
      );
    }
    return (
      <span
        key={i}
        title="In expected, missing from output"
        style={{
          background: '#fde2e2',
          color: '#a83232',
          textDecoration: 'line-through',
          padding: '0 1px',
          borderRadius: 2,
        }}
      >
        {op.text}
      </span>
    );
  });
}

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
  renderBody,
  headerExtras,
}: {
  label: string;
  text: string;
  collapsedMaxH: number;
  emphasized?: boolean; // output pane gets the contrast bg
  muted?: boolean; // expected pane uses softer text color
  /** Optional ref forwarded to the content div - lets the parent
   * attach selection listeners without duplicating Pane's layout. */
  bodyRef?: (el: HTMLDivElement | null) => void;
  /** Optional override for how the text body is rendered. When provided
   * we render this instead of the raw text - lets the parent decorate
   * the text (e.g., inline evidence highlights). The char count and
   * truncation hint are still computed from the underlying `text`. */
  renderBody?: (text: string) => React.ReactNode;
  /** Extra controls rendered in the pane header, before the Expand
   * toggle. Use for pane-specific actions (e.g., a "Diff vs expected"
   * toggle on the Output pane) that share the visual slot with Expand. */
  headerExtras?: React.ReactNode;
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
        {headerExtras}
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
        {text ? (renderBody ? renderBody(text) : text) : '(empty)'}
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
    ['⌘/Ctrl S', 'save in place'],
    ['U / ⌫', 'undo'],
    ['S', 'skip all + next'],
    ['B', 'bookmark item'],
    ['D', 'next override'],
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

/** Bookmarks live in their own localStorage entry per session. We keep
 * them separate from the draft so finalize-clearing-the-draft doesn't
 * also wipe the user's "items I want to revisit" set, and so future
 * cross-session bookmark views can read these directly. */
function bookmarksKey(sessionId: string): string {
  return `evalyn:annotation:bookmarks:${sessionId}`;
}

function readBookmarks(sessionId: string): Record<string, true> {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(bookmarksKey(sessionId));
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return {};
    // Only keep entries whose value is literally true to defend against
    // a malformed payload (e.g. someone hand-edited localStorage).
    const out: Record<string, true> = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (v === true && typeof k === 'string') out[k] = true;
    }
    return out;
  } catch {
    return {};
  }
}

function writeBookmarks(sessionId: string, marks: Record<string, true>): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(bookmarksKey(sessionId), JSON.stringify(marks));
  } catch {
    // best-effort
  }
}

/** Cursor resume: store the last viewed item_id per session so revisits
 * land where the user left off, not back at "first un-annotated". We
 * persist the item_id (not the index) so reordering of items doesn't
 * mis-aim the cursor on the next load. */
function cursorKey(sessionId: string): string {
  return `evalyn:annotation:cursor:${sessionId}`;
}

function readCursorItemId(sessionId: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(cursorKey(sessionId));
  } catch {
    return null;
  }
}

function writeCursorItemId(sessionId: string, itemId: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(cursorKey(sessionId), itemId);
  } catch {
    // best-effort
  }
}

/** Filter persistence so a refresh keeps the user's chosen view. */
function filterStorageKey(sessionId: string): string {
  return `evalyn:annotation:filter:${sessionId}`;
}

const VALID_FILTERS = new Set(['all', 'todo', 'done', 'bookmarked', 'overrides']);

function readFilter(sessionId: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    const v = window.localStorage.getItem(filterStorageKey(sessionId));
    return v && VALID_FILTERS.has(v) ? v : null;
  } catch {
    return null;
  }
}

function writeFilter(sessionId: string, value: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(filterStorageKey(sessionId), value);
  } catch {
    // best-effort
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
  // Deep-link param: /annotate/:sessionId?item=:id. Mount precedence
  // is URL > localStorage cursor > first un-annotated > 0. We use
  // replace (not push) when syncing the URL on cursor change so the
  // back button isn't polluted with an entry per item visited.
  const [searchParams, setSearchParams] = useSearchParams();
  const urlItemId = searchParams.get('item');

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

  // Cursor: precedence on initial mount is URL ?item= > localStorage
  // cursor > first un-annotated > 0. Persisting by item_id (not index)
  // means reordered/re-paginated items don't mis-aim on the next load.
  const [cursor, setCursor] = useState(0);
  const initialCursorRef = useRef(false);
  useEffect(() => {
    if (initialCursorRef.current || items.length === 0) return;
    // 1. URL param wins - it's the most explicit user signal (someone
    //    shared a deep link or the user bookmarked a specific item).
    const fromUrl = urlItemId ? items.findIndex((i) => i.item_id === urlItemId) : -1;
    if (fromUrl >= 0) {
      setCursor(fromUrl);
      initialCursorRef.current = true;
      return;
    }
    // 2. Saved cursor from a prior visit.
    const savedId = sessionId ? readCursorItemId(sessionId) : null;
    const savedIdx = savedId ? items.findIndex((i) => i.item_id === savedId) : -1;
    if (savedIdx >= 0) {
      setCursor(savedIdx);
      initialCursorRef.current = true;
      return;
    }
    // 3. First un-annotated, then 0.
    const idx = items.findIndex((i) => !i.annotated);
    setCursor(idx >= 0 ? idx : 0);
    initialCursorRef.current = true;
  }, [items, sessionId, urlItemId]);

  // Persist the current item_id on every cursor change: localStorage
  // (across reloads) + URL ?item= (shareable). Use replace() on the
  // URL to avoid a history entry per item navigated.
  useEffect(() => {
    if (!sessionId || items.length === 0) return;
    const item = items[cursor];
    if (!item) return;
    writeCursorItemId(sessionId, item.item_id);
    if (searchParams.get('item') !== item.item_id) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('item', item.item_id);
          return next;
        },
        { replace: true },
      );
    }
  }, [sessionId, cursor, items, searchParams, setSearchParams]);

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
  // Per-item bookmarks: items the user has flagged for revisit. Lives
  // in its own localStorage entry (see bookmarksKey) so finalize doesn't
  // clear it. Map shape lets us check membership in O(1).
  const [bookmarks, setBookmarks] = useState<Record<string, true>>(() => {
    if (!sessionId) return {};
    return readBookmarks(sessionId);
  });
  useEffect(() => {
    if (!sessionId) return;
    writeBookmarks(sessionId, bookmarks);
  }, [sessionId, bookmarks]);
  const toggleBookmark = useCallback(
    (itemId: string) => {
      setBookmarks((prev) => {
        const next = { ...prev };
        if (next[itemId]) {
          delete next[itemId];
        } else {
          next[itemId] = true;
        }
        return next;
      });
    },
    [],
  );
  const bookmarkCount = useMemo(() => Object.keys(bookmarks).length, [bookmarks]);

  // Filter mode for navigation + scrubber. Lets the annotator focus
  // on subsets without losing the rest. Default 'all' keeps the
  // existing behavior so adding this is non-disruptive. Persisted
  // per session so a refresh keeps the user's chosen view.
  type FilterKind = 'all' | 'todo' | 'done' | 'bookmarked' | 'overrides';
  const [filter, setFilter] = useState<FilterKind>(() => {
    if (!sessionId) return 'all';
    return (readFilter(sessionId) as FilterKind | null) ?? 'all';
  });
  useEffect(() => {
    if (!sessionId) return;
    writeFilter(sessionId, filter);
  }, [sessionId, filter]);

  const matchesFilter = useCallback(
    (item: AnnotationItemRow): boolean => {
      switch (filter) {
        case 'all':
          return true;
        case 'todo':
          return !item.annotated;
        case 'done':
          return item.annotated;
        case 'bookmarked':
          return bookmarks[item.item_id] === true;
        case 'overrides': {
          if (!item.annotated) return false;
          const ai = aiVerdictMap(item);
          return item.user_labels.some((ul) =>
            isDisagreement(ul.label, ai.get(ul.metric_id) ?? null),
          );
        }
      }
    },
    [filter, bookmarks],
  );

  // Counts per filter for the chip badges. Recomputed when items or
  // bookmarks change. Cheap (single pass).
  const filterCounts = useMemo(() => {
    let todo = 0;
    let done = 0;
    let bookmarked = 0;
    let overrides = 0;
    for (const it of items) {
      if (!it.annotated) todo += 1;
      else done += 1;
      if (bookmarks[it.item_id]) bookmarked += 1;
      if (it.annotated) {
        const ai = aiVerdictMap(it);
        if (
          it.user_labels.some((ul) =>
            isDisagreement(ul.label, ai.get(ul.metric_id) ?? null),
          )
        ) {
          overrides += 1;
        }
      }
    }
    return { all: items.length, todo, done, bookmarked, overrides };
  }, [items, bookmarks]);

  // When the filter changes and the cursor is on a non-matching item,
  // jump to the first matching item from current cursor (search forward
  // first, then wrap to start). If nothing matches we leave cursor in
  // place and the no-match notice in the render handles the rest.
  useEffect(() => {
    if (items.length === 0) return;
    if (matchesFilter(items[cursor])) return;
    let idx = -1;
    for (let i = cursor + 1; i < items.length; i++) {
      if (matchesFilter(items[i])) {
        idx = i;
        break;
      }
    }
    if (idx === -1) {
      for (let i = 0; i < items.length; i++) {
        if (matchesFilter(items[i])) {
          idx = i;
          break;
        }
      }
    }
    if (idx >= 0) setCursor(idx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);
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

  // Pane element refs - the selection listener accepts a selection
  // inside any of these panes (Input, Expected, Output) as evidence.
  // Originally only Output was eligible, but the user's reasoning
  // for marking fail can come from any pane (e.g., ambiguous input).
  const outputBodyRef = useRef<HTMLDivElement | null>(null);
  const inputBodyRef = useRef<HTMLDivElement | null>(null);
  const expectedBodyRef = useRef<HTMLDivElement | null>(null);
  // Toggle between raw output and a word-level diff against expected.
  // Per-item: the toggle remounts the Pane (key includes diff state)
  // so the user gets a clean entrance. Reset when item changes via
  // the cursor effect below.
  const [showDiff, setShowDiff] = useState(false);
  useEffect(() => {
    setShowDiff(false);
  }, [cursor]);

  // Index of the evidence row to briefly flash. Set when the user
  // clicks an inline highlight in the output pane; auto-clears after
  // the eItemSlideIn-style attention pulse runs.
  const [flashEvidenceIdx, setFlashEvidenceIdx] = useState<number | null>(null);
  // Ref array for evidence row elements so we can scrollIntoView
  // when an inline highlight is clicked.
  const evidenceRowRefs = useRef<Array<HTMLDivElement | null>>([]);
  const flashEvidenceRow = useCallback((idx: number) => {
    setFlashEvidenceIdx(idx);
    const el = evidenceRowRefs.current[idx];
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Auto-clear so the same row can flash again on a second click.
    window.setTimeout(() => {
      setFlashEvidenceIdx((cur) => (cur === idx ? null : cur));
    }, 1200);
  }, []);
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
  // any of the eligible panes (Input, Expected, Output), show the
  // popover near the selection's bottom edge. Both endpoints of the
  // selection must be inside the same pane - cross-pane selections
  // are ambiguous and we'd rather decline than guess.
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
      const anchor = sel.anchorNode;
      const focus = sel.focusNode;
      if (!anchor || !focus) return;
      const eligibles: Array<HTMLDivElement | null> = [
        outputBodyRef.current,
        inputBodyRef.current,
        expectedBodyRef.current,
      ];
      const inSamePane = eligibles.some(
        (pane) => pane && pane.contains(anchor) && pane.contains(focus),
      );
      if (!inSamePane) {
        setEvidencePopover(null);
        return;
      }
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const x = Math.max(160, Math.min(window.innerWidth - 160, rect.left + rect.width / 2));
      const y = rect.bottom + 8;
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

  // Find the next index from `start` (exclusive) that matches the
  // current filter. Returns -1 if none. dir = +1 forward, -1 back.
  const findInFilter = useCallback(
    (start: number, dir: 1 | -1): number => {
      for (let i = start + dir; i >= 0 && i < items.length; i += dir) {
        if (matchesFilter(items[i])) return i;
      }
      return -1;
    },
    [items, matchesFilter],
  );

  const goNext = useCallback(async () => {
    const ok = await submitVerdict();
    if (!ok) return;
    const next = findInFilter(cursor, 1);
    if (next >= 0) setCursor(next);
  }, [submitVerdict, cursor, findInFilter]);

  const goPrev = useCallback(() => {
    const prev = findInFilter(cursor, -1);
    if (prev >= 0) setCursor(prev);
  }, [cursor, findInFilter]);

  const goNextNoSave = useCallback(() => {
    const next = findInFilter(cursor, 1);
    if (next >= 0) setCursor(next);
  }, [cursor, findInFilter]);

  // Keyboard handler. Bypassed when focus is in a textarea / input.
  // Also bypassed when any modifier key is held - those combinations
  // belong to the browser (Cmd+B, Ctrl+A, etc.) and we shouldn't steal
  // them. ArrowLeft/Right + Esc are still handled because they have no
  // standard modifier semantics worth preserving on this surface.
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      // Cmd/Ctrl+S = save without advancing. Caught BEFORE every
      // other guard so it works while the user is typing in the
      // note textarea (the most common time they'd want it) and so
      // we can preventDefault the browser's "save page" dialog.
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        if (!currentItem) return;
        e.preventDefault();
        void submitVerdict();
        return;
      }
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')) return;
      if (!currentItem) return;
      // Skip when any other modifier combo is held - leaves browser
      // shortcuts (Cmd+B for bookmarks bar, Ctrl+R for reload, etc.)
      // intact.
      if (e.metaKey || e.ctrlKey || e.altKey) return;

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
      } else if (k === 'b') {
        // "b" toggles a bookmark on the current item. Bookmarks are a
        // separate signal from verdicts - "I want to revisit this", not
        // "this is my decision". Visible in the scrubber + header count.
        e.preventDefault();
        toggleBookmark(currentItem.item_id);
      } else if (k === 'd') {
        // "d" jumps to the next item where the user disagreed with the
        // AI's pass/fail. Wraps around from start if no match forward.
        // No-op when there are zero disagreements anywhere - the
        // header pill already tells the user that count is 0.
        e.preventDefault();
        const findOverride = (start: number, dir: 1 | -1): number => {
          for (let i = start + dir; i >= 0 && i < items.length; i += dir) {
            const it = items[i];
            if (!it.annotated) continue;
            const ai = aiVerdictMap(it);
            if (
              it.user_labels.some((ul) =>
                isDisagreement(ul.label, ai.get(ul.metric_id) ?? null),
              )
            ) {
              return i;
            }
          }
          return -1;
        };
        let next = findOverride(cursor, 1);
        if (next === -1) next = findOverride(-1, 1);
        if (next >= 0 && next !== cursor) setCursor(next);
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
    items,
    toggleBookmark,
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

  // Per-metric verdict distribution across annotated items. Lets the
  // user notice their own pattern (e.g., "I'm marking 90% pass on
  // factuality, am I being too lenient?"). Computed from server state.
  const distribution = useMemo(() => {
    const out: Record<string, { pass: number; fail: number; skip: number }> = {};
    for (const mid of metricIds) out[mid] = { pass: 0, fail: 0, skip: 0 };
    for (const it of items) {
      if (!it.annotated) continue;
      const skippedSet = new Set(it.skipped_metrics);
      const labelMap = new Map<string, AnnotationLabel>();
      for (const ul of it.user_labels) {
        if (ul.metric_id) labelMap.set(ul.metric_id, ul.label);
      }
      for (const mid of metricIds) {
        const bucket = out[mid];
        if (!bucket) continue;
        if (skippedSet.has(mid)) {
          bucket.skip += 1;
        } else if (labelMap.has(mid)) {
          const lab = labelMap.get(mid)!;
          if (lab === 'pass') bucket.pass += 1;
          else if (lab === 'fail') bucket.fail += 1;
          else bucket.skip += 1;
        }
      }
    }
    return out;
  }, [items, metricIds]);

  // Stats panel toggle. Default hidden so the surface stays calm; users
  // open it on demand when they want to inspect their own pattern.
  const [showStats, setShowStats] = useState(false);

  // Scrubber hover preview - shows a small floating tooltip with the
  // item's input preview after a 400ms hover. Helps users navigate
  // long sessions without clicking through every dot. Touch devices
  // skip the preview because they have no hover state.
  const [scrubberHover, setScrubberHover] = useState<{
    idx: number;
    x: number;
    y: number;
  } | null>(null);
  const scrubberHoverTimerRef = useRef<number | null>(null);
  const onScrubberDotEnter = useCallback((idx: number, target: HTMLElement) => {
    if (scrubberHoverTimerRef.current) {
      window.clearTimeout(scrubberHoverTimerRef.current);
    }
    const rect = target.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top - 8;
    scrubberHoverTimerRef.current = window.setTimeout(() => {
      setScrubberHover({ idx, x, y });
    }, 400);
  }, []);
  const onScrubberDotLeave = useCallback(() => {
    if (scrubberHoverTimerRef.current) {
      window.clearTimeout(scrubberHoverTimerRef.current);
      scrubberHoverTimerRef.current = null;
    }
    setScrubberHover(null);
  }, []);
  // Cleanup the timer on unmount so a pending tooltip doesn't fire
  // after the component is gone.
  useEffect(() => {
    return () => {
      if (scrubberHoverTimerRef.current) {
        window.clearTimeout(scrubberHoverTimerRef.current);
      }
    };
  }, []);

  // True when every loaded item has been annotated. Drives the
  // celebratory all-done state below.
  const allDone = items.length > 0 && items.every((it) => it.annotated);

  // Copy-link feedback: when the user copies a deep-link to clipboard,
  // we replace the button glyph with "✓ Copied" briefly so the action
  // is acknowledged without a separate toast layer.
  const [copiedTick, setCopiedTick] = useState(0);
  const copyItemLink = useCallback(() => {
    if (typeof window === 'undefined' || !currentItem) return;
    if (!navigator.clipboard?.writeText) return;
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('item', currentItem.item_id);
      // Only flash "Copied" if the write actually succeeded, so an
      // insecure context (no clipboard permission) doesn't lie to
      // the user. Promise rejection is silent - the user just won't
      // see feedback and can try again or copy the URL bar manually.
      navigator.clipboard.writeText(url.toString()).then(
        () => setCopiedTick((t) => t + 1),
        () => {
          // ignore - browser denied
        },
      );
    } catch {
      // ignore - URL ctor or property access failed
    }
  }, [currentItem]);
  // Auto-clear the copied feedback after 1.5s. Counter-keyed so
  // consecutive copies retrigger cleanly.
  useEffect(() => {
    if (copiedTick === 0) return;
    const t = window.setTimeout(() => setCopiedTick(0), 1500);
    return () => window.clearTimeout(t);
  }, [copiedTick]);

  // Finalize confirmation gate. We only require the user to confirm
  // when there are still un-annotated items - clicking Finish then is
  // an irreversible "I'm done with what I have" action and we don't
  // want it to fire on a stray click. allDone path stays one-click.
  const [confirmFinalize, setConfirmFinalize] = useState(false);
  // When items refetch and the session becomes all-done, drop any
  // pending confirmation - the gate is no longer needed.
  useEffect(() => {
    if (allDone) setConfirmFinalize(false);
  }, [allDone]);

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
          {bookmarkCount > 0 && (
            <Pill
              mono
              color={E.ember}
              bg="#fcefe2"
              style={{ fontSize: 10 }}
              title={`${bookmarkCount} item${bookmarkCount === 1 ? '' : 's'} bookmarked for revisit.`}
            >
              ★ {bookmarkCount}
            </Pill>
          )}
          <Btn
            kind="ghost"
            size="sm"
            onClick={() => setShowStats((v) => !v)}
            title="Toggle the per-metric verdict distribution panel"
          >
            Stats {showStats ? '▴' : '▾'}
          </Btn>
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
            onClick={() => {
              if (allDone) {
                void finalizeSession();
              } else {
                setConfirmFinalize(true);
              }
            }}
            disabled={finalizing || progress?.done === 0}
          >
            {finalizing ? 'Finalizing...' : 'Finish & save'}
          </Btn>
          <Btn kind="ghost" size="sm" onClick={() => navigate('/annotate')}>
            Exit
          </Btn>
        </div>

        {/* FILTER CHIPS - subset the visible items so annotators can
            focus. Selecting a filter dims out non-matching scrubber
            dots and makes ←/→/N skip them. Counts are live. Disabled
            when their match count is 0 (and we're not currently on it). */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            marginBottom: 8,
            flexWrap: 'wrap',
          }}
        >
          {(
            [
              ['all', 'All', filterCounts.all],
              ['todo', 'Todo', filterCounts.todo],
              ['done', 'Done', filterCounts.done],
              ['bookmarked', '★ Bookmarks', filterCounts.bookmarked],
              ['overrides', 'Overrides', filterCounts.overrides],
            ] as Array<[FilterKind, string, number]>
          ).map(([k, label, count]) => {
            const active = filter === k;
            const disabled = count === 0 && !active && k !== 'all';
            return (
              <button
                key={k}
                type="button"
                disabled={disabled}
                onClick={() => setFilter(k)}
                title={disabled ? 'No items in this filter' : `Show ${label.toLowerCase()}`}
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  padding: '4px 10px',
                  borderRadius: 14,
                  border: `1px solid ${active ? E.ember : E.hair2}`,
                  background: active ? '#fcefe2' : E.panel,
                  color: active ? E.ember : disabled ? E.text3 : E.text2,
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  opacity: disabled ? 0.5 : 1,
                  transition: 'all 160ms',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                {label}
                <span
                  style={{
                    fontSize: 10,
                    color: active ? E.ember : E.text3,
                    background: active ? '#fff8f1' : E.panel2,
                    border: `1px solid ${active ? E.ember + '33' : E.hair}`,
                    borderRadius: 8,
                    padding: '0 5px',
                    minWidth: 14,
                    textAlign: 'center',
                  }}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* STATS PANEL - toggleable per-metric verdict distribution.
            Hidden by default to keep the surface calm; toggled via the
            Stats button in the header. Bars are stacked horizontal:
            green = pass, red = fail, neutral = skip. Numbers on the
            right show the raw counts. */}
        {showStats && (
          <Card
            style={{
              padding: 12,
              marginBottom: 10,
              animation: 'eItemSlideIn 200ms ease-out',
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
              <Eyebrow>Verdict distribution</Eyebrow>
              <span
                style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}
              >
                across your saved verdicts so far
              </span>
            </div>
            {(() => {
              const anyData = metricIds.some((mid) => {
                const d = distribution[mid];
                return d && d.pass + d.fail + d.skip > 0;
              });
              if (!anyData) {
                return (
                  <div style={{ fontSize: 12, color: E.text3 }}>
                    No verdicts saved yet — start annotating to see distribution.
                  </div>
                );
              }
              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {metricIds.map((mid) => {
                    const d = distribution[mid] ?? { pass: 0, fail: 0, skip: 0 };
                    const total = d.pass + d.fail + d.skip;
                    if (total === 0) return null;
                    const pct = (n: number) => `${Math.round((n / total) * 100)}%`;
                    return (
                      <div
                        key={mid}
                        style={{
                          display: 'grid',
                          gridTemplateColumns: '140px 1fr 130px',
                          alignItems: 'center',
                          gap: 10,
                        }}
                      >
                        <span
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 12,
                            color: E.text2,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={mid}
                        >
                          {mid}
                        </span>
                        <div
                          style={{
                            display: 'flex',
                            height: 12,
                            borderRadius: 6,
                            overflow: 'hidden',
                            background: E.panel2,
                          }}
                          title={`${d.pass} pass · ${d.fail} fail · ${d.skip} skip`}
                        >
                          {d.pass > 0 && (
                            <div
                              style={{
                                width: pct(d.pass),
                                background: E.pass,
                                transition: 'width 200ms',
                              }}
                            />
                          )}
                          {d.fail > 0 && (
                            <div
                              style={{
                                width: pct(d.fail),
                                background: E.fail,
                                transition: 'width 200ms',
                              }}
                            />
                          )}
                          {d.skip > 0 && (
                            <div
                              style={{
                                width: pct(d.skip),
                                background: E.text3,
                                transition: 'width 200ms',
                              }}
                            />
                          )}
                        </div>
                        <span
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 11,
                            color: E.text2,
                            textAlign: 'right',
                          }}
                        >
                          <span style={{ color: E.pass }}>{d.pass}✓</span>{' '}
                          <span style={{ color: E.fail }}>{d.fail}✗</span>{' '}
                          <span style={{ color: E.text3 }}>{d.skip}·</span>
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </Card>
        )}

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
                const isBookmarked = bookmarks[it.item_id] === true;
                const inFilter = matchesFilter(it);
                const bg = isCurrent
                  ? E.ember
                  : it.annotated
                    ? E.pass
                    : E.hair2;
                const border = isCurrent ? `2px solid ${E.ember}` : 'none';
                // Wrap each dot in a column so we can stack a small
                // bookmark glyph above bookmarked items without affecting
                // the dot's own size/alignment.
                return (
                  <div
                    key={it.item_id}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 1,
                      // Dim out items outside the active filter so the
                      // annotator can scan the in-filter set quickly.
                      // Current item stays full opacity even if filtered.
                      opacity: inFilter || isCurrent ? 1 : 0.25,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 9,
                        lineHeight: 1,
                        color: isBookmarked ? E.ember : 'transparent',
                        height: 10,
                        userSelect: 'none',
                      }}
                      aria-hidden
                    >
                      ★
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        // Clearing the hover preview avoids it lingering
                        // on the just-selected item if the mouse lifts
                        // before mouseleave fires.
                        onScrubberDotLeave();
                        setCursor(i);
                      }}
                      onMouseEnter={(e) => onScrubberDotEnter(i, e.currentTarget)}
                      onMouseLeave={onScrubberDotLeave}
                      title={`Item ${i + 1}${it.annotated ? ' · annotated' : ''}${isBookmarked ? ' · bookmarked' : ''}`}
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
                  </div>
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

        {/* Finalize confirmation panel - only shown when the user
            clicked Finish & save while items remain todo. Spelling
            out the breakdown makes the action's blast radius obvious
            before they commit. */}
        {confirmFinalize && progress && (
          <Card
            style={{
              padding: 14,
              marginBottom: 14,
              borderColor: E.ember,
              background: '#fcefe2',
              animation: 'eItemSlideIn 200ms ease-out',
            }}
          >
            <div style={{ fontFamily: E.fSerif, fontSize: 15, color: E.text1, marginBottom: 6 }}>
              Finalize this session?
            </div>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.text2,
                marginBottom: 10,
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: E.pass }}>{progress.done} done</span>
              {' · '}
              <span style={{ color: E.text3 }}>
                {Math.max(0, progress.total - progress.done)} still todo
              </span>
              {bookmarkCount > 0 && (
                <>
                  {' · '}
                  <span style={{ color: E.ember }}>★ {bookmarkCount} bookmarked</span>
                </>
              )}
              <div style={{ marginTop: 4, color: E.text3, fontSize: 11 }}>
                Finalizing merges your saved verdicts into the dataset
                annotations file. Items not yet annotated will not be
                included.
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <Btn
                kind="primary"
                size="sm"
                onClick={() => {
                  void finalizeSession();
                  setConfirmFinalize(false);
                }}
                disabled={finalizing}
              >
                {finalizing ? 'Finalizing...' : 'Yes, finalize now'}
              </Btn>
              <Btn
                kind="secondary"
                size="sm"
                onClick={() => setConfirmFinalize(false)}
                disabled={finalizing}
              >
                Cancel
              </Btn>
            </div>
          </Card>
        )}

        {/* No-match notice for the active filter. Shown when the filter
            isn't 'all' and zero items match. The item card below still
            renders so the user has context, but a one-click escape
            hatch back to 'all' is right here. */}
        {filter !== 'all' && filterCounts[filter] === 0 && items.length > 0 && (
          <Card
            style={{
              padding: 12,
              marginBottom: 14,
              borderColor: E.hair2,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div style={{ fontSize: 12, color: E.text2, flex: 1 }}>
              No items match the <b style={{ color: E.ember }}>{filter}</b> filter.
            </div>
            <Btn kind="secondary" size="sm" onClick={() => setFilter('all')}>
              Switch to All
            </Btn>
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
              {/* Bookmark toggle - filled star when set, hollow otherwise.
                  Pressed B is the keyboard shortcut. */}
              <button
                type="button"
                onClick={() => toggleBookmark(currentItem.item_id)}
                title={
                  bookmarks[currentItem.item_id]
                    ? 'Remove bookmark (B)'
                    : 'Bookmark this item for revisit (B)'
                }
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: 16,
                  color: bookmarks[currentItem.item_id] ? E.ember : E.text3,
                  padding: '0 4px',
                  lineHeight: 1,
                  transition: 'color 160ms, transform 160ms',
                  transform: bookmarks[currentItem.item_id] ? 'scale(1.06)' : 'none',
                }}
              >
                {bookmarks[currentItem.item_id] ? '★' : '☆'}
              </button>
              {/* Copy-link button. Builds a URL with ?item= set to the
                  current item id and writes it to the clipboard.
                  Shows a brief "Copied" pill so the user knows it worked
                  without a separate toast layer. */}
              <button
                type="button"
                onClick={copyItemLink}
                title="Copy a sharable link to this item"
                style={{
                  background: copiedTick > 0 ? '#fcefe2' : 'transparent',
                  border: `1px solid ${copiedTick > 0 ? E.ember : 'transparent'}`,
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: copiedTick > 0 ? E.ember : E.text3,
                  padding: '2px 6px',
                  lineHeight: 1,
                  transition: 'all 160ms',
                }}
              >
                {copiedTick > 0 ? '✓ Copied' : '🔗 Copy link'}
              </button>
              <span style={{ flex: 1 }} />
              <Btn kind="ghost" size="sm" onClick={goPrev} disabled={cursor === 0}>
                ←
              </Btn>
              <Btn kind="ghost" size="sm" onClick={goNextNoSave} disabled={cursor === items.length - 1}>
                →
              </Btn>
            </div>

            <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <Pane
                label="Input"
                text={currentItem.input_preview ?? ''}
                collapsedMaxH={260}
                bodyRef={(el) => {
                  inputBodyRef.current = el;
                }}
                renderBody={(t) =>
                  renderWithHighlights(
                    t,
                    (evidence[currentItem.item_id] ?? []).map((e) => e.snippet),
                    flashEvidenceRow,
                  )
                }
              />
              {currentItem.expected_preview && (
                <Pane
                  label="Expected"
                  text={currentItem.expected_preview}
                  collapsedMaxH={220}
                  muted
                  bodyRef={(el) => {
                    expectedBodyRef.current = el;
                  }}
                  renderBody={(t) =>
                    renderWithHighlights(
                      t,
                      (evidence[currentItem.item_id] ?? []).map((e) => e.snippet),
                      flashEvidenceRow,
                    )
                  }
                />
              )}
              {currentItem.output_preview && (
                <div
                  key={showDiff ? 'diff' : 'raw'}
                  style={{ animation: 'eItemSlideIn 200ms ease-out' }}
                >
                  <Pane
                    label="Output"
                    text={currentItem.output_preview}
                    collapsedMaxH={420}
                    emphasized
                    bodyRef={(el) => {
                      outputBodyRef.current = el;
                    }}
                    headerExtras={
                      currentItem.expected_preview ? (
                        <button
                          type="button"
                          onClick={() => setShowDiff((v) => !v)}
                          title={
                            showDiff
                              ? 'Show the raw output text'
                              : 'Show a word-level diff against the expected output'
                          }
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 10,
                            color: showDiff ? E.ember : E.text2,
                            background: showDiff ? '#fcefe2' : 'transparent',
                            border: `1px solid ${showDiff ? E.ember : E.hair}`,
                            borderRadius: 4,
                            padding: '2px 8px',
                            cursor: 'pointer',
                          }}
                        >
                          {showDiff ? '✓ Diff vs expected' : 'Diff vs expected'}
                        </button>
                      ) : null
                    }
                    renderBody={(t) =>
                      showDiff && currentItem.expected_preview
                        ? renderDiffBody(currentItem.expected_preview, t)
                        : renderWithHighlights(
                            t,
                            (evidence[currentItem.item_id] ?? []).map((e) => e.snippet),
                            flashEvidenceRow,
                          )
                    }
                  />
                </div>
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
                // Count evidence snippets attached to THIS specific metric.
                // Item-level evidence (metric_id == null) doesn't count
                // toward any single row's badge.
                const evCount = (evidence[currentItem.item_id] ?? []).filter(
                  (e) => e.metric_id === mid,
                ).length;
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
                    <span
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 13,
                        color: E.text1,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      {mid}
                      {evCount > 0 && (
                        <span
                          title={`${evCount} evidence snippet${evCount === 1 ? '' : 's'} attached`}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 2,
                            fontFamily: E.fMono,
                            fontSize: 10,
                            color: E.ember,
                            background: '#fcefe2',
                            border: `1px solid ${E.ember}33`,
                            borderRadius: 4,
                            padding: '1px 5px',
                          }}
                        >
                          📎 {evCount}
                        </span>
                      )}
                    </span>
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
                        ref={(el) => {
                          evidenceRowRefs.current[i] = el;
                        }}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: 8,
                          padding: '6px 8px',
                          background: flashEvidenceIdx === i ? '#fff0d5' : '#fff8eb',
                          border: `1px solid ${flashEvidenceIdx === i ? E.ember : E.hair}`,
                          borderRadius: 6,
                          transition: 'background 240ms, border-color 240ms',
                          animation: flashEvidenceIdx === i ? 'eEvidenceFlash 1.2s ease-out' : undefined,
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

      {/* SCRUBBER HOVER PREVIEW - lightweight tooltip with the item's
          input preview + status. Only shown after a 400ms hover dwell
          so a quick pass over the strip doesn't flash a tooltip per
          dot. Position is fixed to viewport (above the dot). */}
      {scrubberHover && items[scrubberHover.idx] && (
        <div
          role="tooltip"
          style={{
            position: 'fixed',
            left: scrubberHover.x,
            top: scrubberHover.y,
            transform: 'translateX(-50%) translateY(-100%)',
            zIndex: 60,
            background: '#fbf7ee',
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            boxShadow: '0 6px 18px rgba(20,18,14,0.12)',
            padding: '8px 10px',
            maxWidth: 320,
            pointerEvents: 'none',
            animation: 'eItemSlideIn 140ms ease-out',
          }}
        >
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text2,
              marginBottom: 4,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span>Item {scrubberHover.idx + 1}</span>
            <span style={{ color: E.text3 }}>·</span>
            <span
              style={{
                color: items[scrubberHover.idx].annotated ? E.pass : E.text3,
              }}
            >
              {items[scrubberHover.idx].annotated ? 'annotated' : 'todo'}
            </span>
            {bookmarks[items[scrubberHover.idx].item_id] && (
              <>
                <span style={{ color: E.text3 }}>·</span>
                <span style={{ color: E.ember }}>★ bookmarked</span>
              </>
            )}
          </div>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text1,
              lineHeight: 1.4,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              maxHeight: 80,
              overflow: 'hidden',
            }}
          >
            {(items[scrubberHover.idx].input_preview ?? '').slice(0, 180) || '(empty)'}
            {(items[scrubberHover.idx].input_preview ?? '').length > 180 && '…'}
          </div>
        </div>
      )}

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
