/**
 * CoPilotThread - full-screen co-pilot conversation route.
 *
 * Two-column layout: left thread list (260px), right conversation pane (flex 1).
 * Reuses `useCoPilotThread` for live messages and `Bubble`/`ToolBlock`/`PlanCard`
 * for rendering. Asks AppShell to hide the right dock since this route owns the
 * full surface.
 *
 * Thread history is client-side only (see threadHistory.ts) - the backend has
 * no list endpoint, so we maintain a small directory in localStorage that
 * upserts after each turn so it survives reloads.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { E } from '../tokens';
import { Btn, Eyebrow } from '../ui';
import { Bubble, PlanCard, ThinkingBubble, ToolBlock } from '../copilot/atoms';
import { linkifyText, makeUrlCounter } from '../textRender';
import { MOD_KEY } from '../platform';
import { useCoPilotThread } from '../copilot/useCoPilotThread';
import { useStickToBottom } from '../hooks/useStickToBottom';
import { useSearchFilter } from '../hooks/useSearchFilter';
import {
  loadThreadIndex,
  upsertThread,
  type ThreadIndexEntry,
} from '../copilot/threadHistory';
import {
  deriveContextChips,
  prependContextTag,
  type ContextKind,
} from '../copilot/contextChips';
import {
  clearCoPilotDraft,
  loadCoPilotDraft,
  saveCoPilotDraft,
} from '../copilot/copilotDrafts';

const DISABLED_CHIP_TITLE =
  'Open this on a run/cluster/dataset page to attach as context';

interface ContextChipButtonProps {
  label: string;
  enabled: boolean;
  title?: string;
  onClick: () => void;
}

function ContextChipButton({ label, enabled, title, onClick }: ContextChipButtonProps) {
  return (
    <button
      type="button"
      onClick={enabled ? onClick : undefined}
      disabled={!enabled}
      title={enabled ? title : DISABLED_CHIP_TITLE}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '2px 8px',
        borderRadius: 999,
        fontSize: 10,
        fontFamily: E.fMono,
        color: enabled ? E.text2 : E.text3,
        background: E.panel2,
        border: 'none',
        cursor: enabled ? 'pointer' : 'not-allowed',
        opacity: enabled ? 1 : 0.55,
        transition: 'color 120ms, background 120ms',
      }}
      onMouseEnter={(e) => {
        if (!enabled) return;
        e.currentTarget.style.color = E.text1;
      }}
      onMouseLeave={(e) => {
        if (!enabled) return;
        e.currentTarget.style.color = E.text2;
      }}
    >
      {label}
    </button>
  );
}

const DAY_MS = 24 * 60 * 60 * 1000;

const EXAMPLE_PROMPTS = [
  "What's my pass rate trend?",
  'Show me the worst failure cluster',
  'Explain my rubric',
];

function inferTitle(firstUserText: string | null): string {
  if (!firstUserText) return 'New thread';
  const trimmed = firstUserText.trim().replace(/\s+/g, ' ');
  if (trimmed.length === 0) return 'New thread';
  return trimmed.length <= 50 ? trimmed : `${trimmed.slice(0, 50)}...`;
}

function groupByRecency(entries: ThreadIndexEntry[]): {
  today: ThreadIndexEntry[];
  earlier: ThreadIndexEntry[];
} {
  const now = Date.now();
  const today: ThreadIndexEntry[] = [];
  const earlier: ThreadIndexEntry[] = [];
  for (const e of entries) {
    const t = Date.parse(e.created_at_iso);
    if (Number.isNaN(t)) {
      earlier.push(e);
      continue;
    }
    if (now - t < DAY_MS) today.push(e);
    else earlier.push(e);
  }
  const sortDesc = (a: ThreadIndexEntry, b: ThreadIndexEntry) =>
    Date.parse(b.created_at_iso) - Date.parse(a.created_at_iso);
  today.sort(sortDesc);
  earlier.sort(sortDesc);
  return { today, earlier };
}

function formatRelative(iso: string, turnCount: number): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return `${turnCount} turns`;
  const delta = Date.now() - t;
  const mins = Math.floor(delta / 60000);
  if (mins < 1) return `${turnCount} turns - just now`;
  if (mins < 60) return `${turnCount} turns - ${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${turnCount} turns - ${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${turnCount} turns - ${days}d ago`;
}

export default function CoPilotThread() {
  const params = useParams<{ threadId?: string }>();
  const routeThreadId = params.threadId ?? null;
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();

  const {
    threadId,
    messages,
    pending,
    status,
    error,
    send,
    confirm,
    resetTo,
    clearError,
    reconnecting,
  } = useCoPilotThread({ initialThreadId: routeThreadId });

  const [draft, setDraft] = useState(() => loadCoPilotDraft(threadId));
  const [historyTick, setHistoryTick] = useState(0);
  const [index, setIndex] = useState<ThreadIndexEntry[]>(() => loadThreadIndex());

  // Sparse SR live announcements for status transitions. The
  // conversation log itself is aria-live="off" (per-chunk streaming
  // would spam) so without this, blind users would have no idea
  // when a response started or finished. Two events:
  //   idle -> streaming: ThinkingBubble's own role="status" label
  //                      already covers this, so we skip here.
  //   streaming -> idle: agent finished. Announce so users know
  //                      to start reading the message.
  // The error path is covered by the role="alert" block below.
  const [ariaStatus, setAriaStatus] = useState('');
  const prevStatusRef = useRef(status);
  useEffect(() => {
    if (prevStatusRef.current === 'streaming' && status === 'idle') {
      setAriaStatus('Co-pilot finished responding');
      const t = window.setTimeout(() => setAriaStatus(''), 1200);
      prevStatusRef.current = status;
      return () => window.clearTimeout(t);
    }
    prevStatusRef.current = status;
  }, [status]);

  // When the active thread changes (sidebar click, URL nav), load
  // that thread's draft so switching threads mid-compose doesn't
  // erase the in-progress text on either side. Mirrors the dock.
  useEffect(() => {
    setDraft(loadCoPilotDraft(threadId));
  }, [threadId]);

  // Auto-save the draft on every change (debounced 250ms). Empty
  // drafts are removed from storage rather than persisted as empty
  // strings (the helper handles that).
  useEffect(() => {
    const handle = window.setTimeout(() => {
      saveCoPilotDraft(threadId, draft);
    }, 250);
    return () => window.clearTimeout(handle);
  }, [threadId, draft]);
  const createdAtRef = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const useSuggestion = (prompt: string) => {
    setDraft(prompt);
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const contextChips = deriveContextChips(location.pathname);

  const attachContext = (kind: ContextKind, id: string) => {
    setDraft((d) => prependContextTag(d, kind, id));
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      const len = el.value.length;
      el.setSelectionRange(len, len);
    });
  };

  // ?prefill=... seeds the composer (e.g. from /commands "Ask co-pilot"
  // links). Only fires when the thread is empty so we don't overwrite an
  // in-flight typed message. Strips the param from the URL after consuming
  // it so a refresh doesn't re-prefill.
  useEffect(() => {
    const prefill = searchParams.get('prefill');
    if (!prefill || messages.length > 0) return;
    setDraft(prefill);
    requestAnimationFrame(() => textareaRef.current?.focus());
    const next = new URLSearchParams(searchParams);
    next.delete('prefill');
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams, messages.length]);

  // Auto-focus the composer on first mount so the user can start typing
  // immediately. Skipped if a prefill is in the URL (the prefill effect
  // above handles focus) or if the user is mid-thread (focusing then
  // would steal focus from any read action they were taking). Empty
  // deps - we run once per route mount, NOT on every thread switch
  // via the sidebar (which keeps the route mounted; param-only nav).
  useEffect(() => {
    if (searchParams.get('prefill')) return;
    requestAnimationFrame(() => textareaRef.current?.focus());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reattach the hook when the URL thread id changes (sidebar click).
  useEffect(() => {
    if (routeThreadId !== threadId) {
      resetTo(routeThreadId);
      createdAtRef.current = null;
    }
  }, [routeThreadId, threadId, resetTo]);

  // Reload history when we just touched it (avoids stale list after upsert).
  useEffect(() => {
    setIndex(loadThreadIndex());
  }, [historyTick]);

  // Persist thread metadata after each idle settle (final lands or status returns to idle).
  useEffect(() => {
    if (!threadId) return;
    if (status === 'streaming') return;
    if (messages.length === 0) return;
    const firstUser = messages.find((m) => m.role === 'you');
    const title = inferTitle(firstUser ? firstUser.text : null);
    if (createdAtRef.current == null) {
      const existing = loadThreadIndex().find((e) => e.id === threadId);
      createdAtRef.current = existing?.created_at_iso ?? new Date().toISOString();
    }
    upsertThread({
      id: threadId,
      title,
      turn_count: messages.length,
      created_at_iso: createdAtRef.current,
    });
    setHistoryTick((n) => n + 1);
  }, [threadId, status, messages]);

  // Thread title filter via the shared useSearchFilter hook. No
  // sessionStorage key (filter is per-visit; persisting across
  // visits would surprise users coming back to a previously-narrow
  // sidebar). No "/" hotkey at the window level - the runner-output
  // and items-tab already claim "/" when their surfaces are active,
  // and the thread sidebar is always visible alongside; adding
  // another "/" handler would race them. Hook still provides
  // debounced query (120ms) + two-step Esc + focus ref.
  const {
    input: threadFilter,
    setInput: setThreadFilter,
    query: threadFilterQuery,
    inputRef: threadFilterRef,
    onKeyDown: onThreadFilterKeyDown,
  } = useSearchFilter({});

  const filteredIndex = useMemo(() => {
    if (!threadFilterQuery) return index;
    return index.filter((e) => e.title.toLowerCase().includes(threadFilterQuery));
  }, [index, threadFilterQuery]);

  const grouped = useMemo(() => groupByRecency(filteredIndex), [filteredIndex]);

  const activeTitle = useMemo(() => {
    if (threadId) {
      const hit = index.find((e) => e.id === threadId);
      if (hit) return hit.title;
    }
    const firstUser = messages.find((m) => m.role === 'you');
    return inferTitle(firstUser ? firstUser.text : null);
  }, [threadId, messages, index]);

  const submit = () => {
    // Defensive guard: the submit button is `disabled` when
    // streaming or a pending confirm is in flight, but React's
    // batched state updates leave a same-frame double-click able
    // to slip through with the stale closure. Match the dock's
    // version of this guard so a rapid second click is a no-op.
    if (status === 'streaming' || pending != null) return;
    const t = draft.trim();
    if (!t) return;
    setDraft('');
    // Synchronous clear avoids a race between the debounced save
    // above and the navigate-after-send that may flip threadId.
    clearCoPilotDraft(threadId);
    void send(t);
  };

  const handleSelectThread = (id: string) => {
    if (id === threadId) return;
    createdAtRef.current = null;
    resetTo(id);
    navigate(`/copilot/${id}`);
  };

  const handleNewThread = () => {
    createdAtRef.current = null;
    resetTo(null);
    navigate('/copilot');
  };

  const sendDisabled = !draft.trim() || status === 'streaming' || pending != null;

  // Auto-scroll to the bottom on new messages / streaming updates,
  // unless the user has scrolled up to read history.
  const {
    scrollRef: convRef,
    onScroll: onConvScroll,
    scrolledUp,
    jumpToBottom,
  } = useStickToBottom(messages);

  return (
    <AppShell hideCoPilot breadcrumb={['Co-pilot', activeTitle]}>
      {/* Sparse status live region for streaming transitions. The
          conversation log is aria-live="off" so this carries the
          "started / finished" beats without per-chunk spam. */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="eSr"
      >
        {ariaStatus}
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '260px 1fr',
          height: '100%',
          minHeight: 0,
        }}
      >
        {/* Thread list */}
        <div
          style={{
            borderRight: `1px solid ${E.hair}`,
            background: E.panel,
            padding: '20px 12px',
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Btn
            kind="secondary"
            size="sm"
            onClick={handleNewThread}
            style={{ width: '100%', justifyContent: 'center', marginBottom: 8 }}
          >
            ＋ New thread
          </Btn>

          {/* Filter only renders once there are at least a handful of
              threads - below that, scrolling is just as fast and the
              extra input adds visual weight for no real benefit. */}
          {index.length >= 5 && (
            <input
              ref={threadFilterRef}
              type="text"
              value={threadFilter}
              onChange={(e) => setThreadFilter(e.target.value)}
              onKeyDown={onThreadFilterKeyDown}
              placeholder="Filter threads..."
              aria-label="Filter threads"
              style={{
                width: '100%',
                padding: '6px 10px',
                marginBottom: 10,
                fontSize: 12,
                fontFamily: E.fSans,
                background: E.panel2,
                color: E.text1,
                border: `1px solid ${E.hair2}`,
                borderRadius: 6,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          )}

          {/* Empty list AND no filter -> first-time copy. Filter active
              and zero hits -> a different "no matches" message that
              surfaces a Clear button. Splitting these two states keeps
              the empty case from sounding broken when the user has
              threads but the filter just doesn't match. */}
          {grouped.today.length === 0 && grouped.earlier.length === 0 && (
            // Branch on the debounced query (matches what
            // filteredIndex / grouped is actually using); using the
            // raw `threadFilter` would flash "No conversations yet"
            // for ~120ms after clearing via Esc. Same anti-pattern
            // ItemsTab had before its useSearchFilter retrofit.
            threadFilterQuery ? (
              <div
                style={{
                  padding: '12px 10px',
                  fontSize: 12,
                  color: E.text3,
                  lineHeight: 1.55,
                }}
              >
                No threads match "{threadFilter}".
                <button
                  type="button"
                  onClick={() => setThreadFilter('')}
                  style={{
                    display: 'block',
                    marginTop: 6,
                    background: 'transparent',
                    border: 'none',
                    color: E.ember,
                    cursor: 'pointer',
                    padding: 0,
                    fontSize: 11.5,
                    fontFamily: E.fMono,
                  }}
                >
                  Clear filter
                </button>
              </div>
            ) : (
              <div
                style={{
                  padding: '12px 10px',
                  fontSize: 12,
                  color: E.text3,
                  lineHeight: 1.55,
                }}
              >
                No conversations yet. Start one on the right.
              </div>
            )
          )}

          {grouped.today.length > 0 && (
            <>
              <Eyebrow style={{ padding: '0 8px', marginBottom: 6 }}>Today</Eyebrow>
              {grouped.today.map((t) => {
                const isActive = t.id === threadId;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => handleSelectThread(t.id)}
                    aria-current={isActive ? 'page' : undefined}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = E.panel2;
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                    // Focus parity for keyboard users tabbing the
                    // thread sidebar. Active-item guard mirrors the
                    // hover handler - the active thread already has
                    // its own panel3 bg.
                    onFocus={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = E.panel2;
                      }
                    }}
                    onBlur={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                    title={t.title}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '10px 10px',
                      borderRadius: 7,
                      border: 'none',
                      cursor: 'pointer',
                      background: isActive ? E.panel3 : 'transparent',
                      marginBottom: 2,
                      transition: 'background 140ms',
                    }}
                  >
                    <div
                      style={{
                        fontSize: 12.5,
                        color: isActive ? E.text0 : E.text1,
                        fontWeight: isActive ? 500 : 400,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {t.title}
                    </div>
                    <div
                      style={{
                        fontSize: 10.5,
                        color: E.text3,
                        marginTop: 2,
                        fontFamily: E.fMono,
                      }}
                    >
                      {formatRelative(t.created_at_iso, t.turn_count)}
                    </div>
                  </button>
                );
              })}
            </>
          )}

          {grouped.earlier.length > 0 && (
            <>
              <Eyebrow style={{ padding: '14px 8px 4px' }}>Earlier this week</Eyebrow>
              {grouped.earlier.map((t) => {
                const isActive = t.id === threadId;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => handleSelectThread(t.id)}
                    aria-current={isActive ? 'page' : undefined}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = E.panel2;
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                    onFocus={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = E.panel2;
                      }
                    }}
                    onBlur={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                    title={t.title}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 6,
                      border: 'none',
                      cursor: 'pointer',
                      background: isActive ? E.panel3 : 'transparent',
                      fontSize: 12,
                      color: isActive ? E.text0 : E.text2,
                      fontWeight: isActive ? 500 : 400,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      transition: 'background 140ms',
                    }}
                  >
                    {t.title}
                  </button>
                );
              })}
            </>
          )}
        </div>

        {/* Conversation pane */}
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div
            style={{
              padding: '20px 36px',
              borderBottom: `1px solid ${E.hair}`,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 12,
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <h1
                style={{
                  fontFamily: E.fSerif,
                  fontSize: 24,
                  fontWeight: 400,
                  margin: 0,
                  color: E.text0,
                  letterSpacing: '-0.01em',
                }}
              >
                {activeTitle}
              </h1>
              <div
                style={{
                  fontSize: 11.5,
                  color: E.text3,
                  fontFamily: E.fMono,
                  marginTop: 4,
                }}
              >
                {threadId ? `thread: ${threadId}` : 'new thread - send a message to start'}
              </div>
            </div>
            {reconnecting && (
              <span
                role="status"
                aria-label="Agent socket reconnecting"
                title="The agent WebSocket dropped. The dashboard is reconnecting; in-flight responses may be delayed."
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '3px 8px',
                  borderRadius: 999,
                  background: 'rgba(229, 161, 79, 0.12)',
                  border: `1px solid rgba(229, 161, 79, 0.3)`,
                  color: E.warn,
                  fontFamily: E.fMono,
                  fontSize: 10,
                  letterSpacing: '0.04em',
                  flexShrink: 0,
                  marginTop: 6,
                }}
              >
                <span aria-hidden style={{ lineHeight: 1, fontSize: 11 }}>◌</span>
                Reconnecting
              </span>
            )}
          </div>

          <div
            style={{
              position: 'relative',
              flex: 1,
              minHeight: 0,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
          <div
            ref={convRef}
            onScroll={onConvScroll}
            // role="log" gives SR users a landmark to jump to with
            // their screen-reader cursor. We override aria-live to
            // "off" because role="log" implies polite live region
            // by default - and the agent streams text-deltas
            // chunk-by-chunk, which would announce every keystroke
            // worth of incoming text. Users can cursor-walk the
            // completed conversation; the explicit ariaStatus live
            // region in this component handles state-change beats.
            role="log"
            aria-label="Conversation"
            aria-live="off"
            style={{ flex: 1, overflowY: 'auto', padding: '24px 36px' }}
          >
            <div style={{ maxWidth: 720 }}>
              {messages.length === 0 && (
                <div style={{ padding: '12px 4px' }}>
                  <div
                    style={{
                      color: E.text3,
                      fontSize: 13,
                      lineHeight: 1.6,
                    }}
                  >
                    Ask anything about your evals. The co-pilot can read runs, datasets,
                    and rubrics on its own. Anything that writes will pause for your
                    confirmation first.
                  </div>
                  <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {EXAMPLE_PROMPTS.map((p) => (
                      <button
                        key={p}
                        type="button"
                        onClick={() => useSuggestion(p)}
                        title="Use this prompt"
                        aria-label={`Use prompt: ${p}`}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          padding: '5px 12px',
                          borderRadius: 999,
                          fontSize: 12,
                          fontFamily: E.fMono,
                          color: E.text2,
                          background: E.panel2,
                          border: `1px solid ${E.hair2}`,
                          cursor: 'pointer',
                          transition: 'border-color 120ms, color 120ms',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = E.ember;
                          e.currentTarget.style.color = E.text1;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = E.hair2;
                          e.currentTarget.style.color = E.text2;
                        }}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((m) => (
                <Bubble key={m.id} who={m.role}>
                  {linkifyText(m.text, makeUrlCounter())}
                  {m.tools.length > 0 && <ToolBlock entries={m.tools} />}
                  {m.pending_confirm && (
                    <PlanCard
                      pending={m.pending_confirm}
                      onApprove={() => void confirm(true)}
                      onReject={() => void confirm(false)}
                    />
                  )}
                </Bubble>
              ))}
              {/* The empty beat between user-message-sent and the
                  agent's first chunk used to look like nothing was
                  happening. Show pulsing dots in an agent bubble so
                  the user knows the request is in flight. As soon
                  as text_delta arrives, useCoPilotThread creates the
                  real agent bubble and last-message becomes 'agent'
                  - the dots vanish naturally. */}
              {status === 'streaming' &&
                messages.length > 0 &&
                messages[messages.length - 1].role === 'you' && <ThinkingBubble />}
              {error && (
                <div
                  role="alert"
                  style={{
                    marginTop: 12,
                    padding: '10px 14px',
                    background: E.failDim,
                    border: `1px solid ${E.fail}33`,
                    borderRadius: 8,
                    color: E.fail,
                    fontSize: 12.5,
                    fontFamily: E.fMono,
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 8,
                  }}
                >
                  <span style={{ flex: 1, lineHeight: 1.5 }}>{error}</span>
                  <button
                    type="button"
                    onClick={clearError}
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
                      flexShrink: 0,
                    }}
                  >
                    ×
                  </button>
                </div>
              )}
            </div>
          </div>
          {scrolledUp && (
            <button
              type="button"
              onClick={jumpToBottom}
              aria-label="Jump to latest message"
              style={{
                position: 'absolute',
                bottom: 16,
                left: '50%',
                transform: 'translateX(-50%)',
                padding: '6px 14px',
                borderRadius: 999,
                background: E.ember,
                color: E.emberInk,
                border: 'none',
                cursor: 'pointer',
                fontSize: 11.5,
                fontFamily: E.fMono,
                fontWeight: 500,
                boxShadow: `0 6px 18px rgba(217,106,44,0.32)`,
                zIndex: 5,
                animation: 'eRouteFallbackFadeIn 200ms ease',
              }}
            >
              ↓ Jump to latest
            </button>
          )}
          </div>

          {/* Composer */}
          <div style={{ padding: '14px 36px 22px', borderTop: `1px solid ${E.hair}` }}>
            <div
              style={{
                background: E.panel,
                border: `1px solid ${E.hair2}`,
                borderRadius: 12,
                padding: 14,
                maxWidth: 760,
              }}
            >
              {/* Attach-chips prepend a `[context: ...]` tag to the draft based on
                  the current route. See ../copilot/contextChips.ts. */}
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {contextChips.map((c) => (
                  <ContextChipButton
                    key={c.kind}
                    label={c.label}
                    enabled={c.id != null}
                    title={c.enabledTitle}
                    onClick={() => {
                      if (c.id) attachContext(c.kind, c.id);
                    }}
                  />
                ))}
              </div>
              <textarea
                ref={textareaRef}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    submit();
                  }
                }}
                aria-label="Co-pilot message composer"
                placeholder="Continue the conversation, or paste a new question..."
                disabled={status === 'streaming' || pending != null}
                rows={3}
                style={{
                  width: '100%',
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: E.text1,
                  fontSize: 13,
                  fontFamily: E.fSans,
                  resize: 'none',
                  lineHeight: 1.55,
                  // Auto-grow with content. The full thread route has
                  // more vertical room than the dock so the cap is
                  // higher (320 vs 200).
                  fieldSizing: 'content',
                  minHeight: 60,
                  maxHeight: 320,
                }}
              />
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginTop: 8,
                }}
              >
                <span
                  style={{
                    fontSize: 10.5,
                    color: E.text3,
                    fontFamily: E.fMono,
                  }}
                >
                  <span aria-hidden="true">{MOD_KEY}+Enter</span>
                  <span className="eSr">{MOD_KEY === '⌘' ? 'Cmd Enter' : 'Ctrl Enter'}</span>
                  {' to send'}
                </span>
                <span style={{ flex: 1 }} />
                <Btn
                  kind="primary"
                  size="sm"
                  onClick={submit}
                  disabled={sendDisabled}
                  // Mirror the CoPilotDock Send-button pattern at
                  // line 581-588 - the dock got proper a11y polish
                  // (aria-busy / spelled-out keys / aria-hidden
                  // glyph) and the full thread route was missed.
                  aria-busy={status === 'streaming' || pending != null}
                  aria-label={`Send message (${MOD_KEY === '⌘' ? 'Cmd Enter' : 'Ctrl Enter'})`}
                  title={
                    status === 'streaming'
                      ? 'Co-pilot is responding - wait for it to finish'
                      : pending != null
                        ? 'A tool action is pending - resolve it above'
                        : !draft.trim()
                          ? 'Type a message to enable Send'
                          : `Send (${MOD_KEY} ⏎)`
                  }
                  style={{ width: 30, height: 30, padding: 0, justifyContent: 'center' }}
                >
                  <span aria-hidden="true">↑</span>
                </Btn>
              </div>
            </div>
            <div
              style={{
                marginTop: 6,
                fontSize: 10,
                color: E.text3,
                fontFamily: E.fMono,
                textAlign: 'center',
                maxWidth: 760,
              }}
            >
              Read-only commands run automatically: writes need your confirmation
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
