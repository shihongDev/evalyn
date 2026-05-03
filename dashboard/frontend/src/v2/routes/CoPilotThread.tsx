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
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { E } from '../tokens';
import { Btn, Eyebrow, Pill } from '../ui';
import { Bubble, PlanCard, ToolBlock } from '../copilot/atoms';
import { useCoPilotThread } from '../copilot/useCoPilotThread';
import {
  loadThreadIndex,
  upsertThread,
  type ThreadIndexEntry,
} from '../copilot/threadHistory';

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

  const { threadId, messages, pending, status, error, send, confirm, resetTo } =
    useCoPilotThread({ initialThreadId: routeThreadId });

  const [draft, setDraft] = useState('');
  const [historyTick, setHistoryTick] = useState(0);
  const [index, setIndex] = useState<ThreadIndexEntry[]>(() => loadThreadIndex());
  const createdAtRef = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const useSuggestion = (prompt: string) => {
    setDraft(prompt);
    requestAnimationFrame(() => textareaRef.current?.focus());
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

  const grouped = useMemo(() => groupByRecency(index), [index]);

  const activeTitle = useMemo(() => {
    if (threadId) {
      const hit = index.find((e) => e.id === threadId);
      if (hit) return hit.title;
    }
    const firstUser = messages.find((m) => m.role === 'you');
    return inferTitle(firstUser ? firstUser.text : null);
  }, [threadId, messages, index]);

  const submit = () => {
    const t = draft.trim();
    if (!t) return;
    setDraft('');
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

  return (
    <AppShell hideCoPilot breadcrumb={['Co-pilot', activeTitle]}>
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
            style={{ width: '100%', justifyContent: 'center', marginBottom: 14 }}
          >
            ＋ New thread
          </Btn>

          {grouped.today.length === 0 && grouped.earlier.length === 0 && (
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
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '10px 10px',
                      borderRadius: 7,
                      border: 'none',
                      cursor: 'pointer',
                      background: isActive ? E.panel3 : 'transparent',
                      marginBottom: 2,
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
          <div style={{ padding: '20px 36px', borderBottom: `1px solid ${E.hair}` }}>
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

          <div style={{ flex: 1, overflowY: 'auto', padding: '24px 36px' }}>
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
                  {m.text}
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
              {error && (
                <div
                  style={{
                    marginTop: 12,
                    padding: '10px 14px',
                    background: E.failDim,
                    border: `1px solid ${E.fail}33`,
                    borderRadius: 8,
                    color: E.fail,
                    fontSize: 12.5,
                    fontFamily: E.fMono,
                  }}
                >
                  {error}
                </div>
              )}
            </div>
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
              {/* Attach-chips are visual placeholders; first-class context attachment is post-v2. */}
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                <Pill
                  mono
                  style={{ fontSize: 10, padding: '2px 8px', background: E.panel2, color: E.text2 }}
                >
                  + this run
                </Pill>
                <Pill
                  mono
                  style={{ fontSize: 10, padding: '2px 8px', background: E.panel2, color: E.text2 }}
                >
                  ＠ dataset
                </Pill>
                <Pill
                  mono
                  style={{ fontSize: 10, padding: '2px 8px', background: E.panel2, color: E.text2 }}
                >
                  + rubric
                </Pill>
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
                  ⌘+Enter to send
                </span>
                <span style={{ flex: 1 }} />
                <Btn
                  kind="primary"
                  size="sm"
                  onClick={submit}
                  disabled={sendDisabled}
                  style={{ width: 30, height: 30, padding: 0, justifyContent: 'center' }}
                >
                  ↑
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
