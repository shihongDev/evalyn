/**
 * ChatPanel — agent chat dock-right.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-chat.jsx. Visual chrome stays
 * close to the mock; data wiring is replaced with the Zustand store's
 * agent slice (Lane C2).
 *
 * Renders:
 *   - Header: "Ask agent" title + settings gear + close button.
 *   - Provider error banner (when `agent.error` is set) with link to
 *     SettingsModal.
 *   - Scrollable history of ChatMessage entries:
 *       * text bubbles (user / assistant)
 *       * inline tool-call cards with status pill
 *       * confirmation cards (approve / reject buttons)
 *       * final-suggestion cards (clickable -> open CliForm tab)
 *   - Composer at the bottom (textarea + send button).
 *
 * Width 420px, full-height, border-left. The dock-right placement is the
 * only one shipped (per spec §8: "dock-right only").
 */

import { useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import type {
  ChatMessage,
  FinalSuggestion,
  ToolCall,
  ToolCallStatus,
} from '../types/agent';

const STATUS_LABEL: Record<ToolCallStatus, string> = {
  proposed: 'proposed',
  awaiting_confirmation: 'awaiting confirmation',
  running: 'running…',
  complete: 'complete',
  rejected: 'rejected',
  error: 'error',
};

const STATUS_TONE: Record<ToolCallStatus, string> = {
  proposed: 'text-2',
  awaiting_confirmation: 'warn',
  running: 'accent',
  complete: 'pass',
  rejected: 'text-3',
  error: 'fail',
};

/* --------------------------- subcomponents -------------------------- */

const Avatar = ({ role }: { role: ChatMessage['role'] }) => {
  if (role === 'user') {
    return (
      <div
        aria-hidden="true"
        style={{
          width: 26,
          height: 26,
          borderRadius: 4,
          background: 'var(--bg-3)',
          border: '1px solid var(--line)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 10,
          color: 'var(--text-2)',
          flexShrink: 0,
        }}
      >
        YOU
      </div>
    );
  }
  return (
    <div
      aria-hidden="true"
      style={{
        width: 26,
        height: 26,
        borderRadius: 4,
        background: 'var(--accent)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 13,
        color: '#1a1305',
        fontWeight: 600,
        flexShrink: 0,
      }}
    >
      e
    </div>
  );
};

/** Tiny markdown renderer: supports `**bold**` and `` `code` ``. */
const Markdownish = ({ text }: { text: string }) => {
  const parts: { t: string; m: 'b' | 'c' | null }[] = [];
  let buf = '';
  let mode: 'b' | 'c' | null = null;
  let i = 0;
  while (i < text.length) {
    if (text.startsWith('**', i)) {
      if (buf) parts.push({ t: buf, m: mode });
      buf = '';
      mode = mode === 'b' ? null : 'b';
      i += 2;
      continue;
    }
    if (text[i] === '`') {
      if (buf) parts.push({ t: buf, m: mode });
      buf = '';
      mode = mode === 'c' ? null : 'c';
      i++;
      continue;
    }
    buf += text[i++];
  }
  if (buf) parts.push({ t: buf, m: mode });
  return (
    <div style={{ fontSize: 13.5, lineHeight: 1.6, color: 'var(--text-1)', whiteSpace: 'pre-wrap' }}>
      {parts.map((p, j) => {
        if (p.m === 'b') return <b key={j}>{p.t}</b>;
        if (p.m === 'c')
          return (
            <code
              key={j}
              className="mono"
              style={{
                background: 'var(--bg-3)',
                padding: '1px 5px',
                borderRadius: 3,
                fontSize: 12,
              }}
            >
              {p.t}
            </code>
          );
        return <span key={j}>{p.t}</span>;
      })}
    </div>
  );
};

const ToolCallCard = ({ call }: { call: ToolCall }) => {
  const confirmAgent = useStore((s) => s.confirmAgent);
  const isAwaiting = call.status === 'awaiting_confirmation';
  const tone = STATUS_TONE[call.status];

  return (
    <div
      data-testid={`tool-call-${call.id}`}
      style={{
        background: 'var(--bg-0)',
        border: '1px solid var(--line)',
        borderRadius: 6,
        marginTop: 10,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-2)',
        }}
      >
        <span className={`mono ${tone}`} style={{ fontSize: 11 }}>
          {call.status === 'complete'
            ? '✓'
            : call.status === 'error' || call.status === 'rejected'
              ? '✗'
              : call.status === 'running'
                ? '▸'
                : '·'}
        </span>
        <code className="mono" style={{ fontSize: 11, color: 'var(--text-1)' }}>
          $ <span className="accent">{call.tool}</span>{' '}
          <span className="text-2">{call.previewCmd.replace(/^evalyn\s+\S+\s*/, '')}</span>
        </code>
        <span className="grow" />
        <span className={`chip mono ${tone}`} style={{ fontSize: 10 }}>
          {STATUS_LABEL[call.status]}
        </span>
      </div>

      {isAwaiting && (
        <div
          style={{
            display: 'flex',
            gap: 8,
            padding: '10px 12px',
            borderBottom: '1px solid var(--line)',
            background: 'var(--warn-soft)',
          }}
        >
          <span className="text-1" style={{ fontSize: 12, flex: 1 }}>
            This command writes to disk. Approve to run?
          </span>
          <button
            type="button"
            className="btn primary sm"
            data-testid={`approve-${call.id}`}
            onClick={() => void confirmAgent(true)}
          >
            Approve
          </button>
          <button
            type="button"
            className="btn ghost sm"
            data-testid={`reject-${call.id}`}
            onClick={() => void confirmAgent(false)}
          >
            Reject
          </button>
        </div>
      )}

      {call.output && (
        <pre
          className="mono"
          style={{
            margin: 0,
            padding: 12,
            fontSize: 11,
            lineHeight: 1.7,
            color: 'var(--text-1)',
            whiteSpace: 'pre-wrap',
            overflow: 'auto',
            maxHeight: 320,
          }}
        >
          {call.output}
        </pre>
      )}

      {call.error && (
        <div
          style={{
            padding: '8px 12px',
            color: 'var(--fail)',
            background: 'var(--fail-soft)',
            fontSize: 12,
            fontFamily: 'var(--mono)',
          }}
        >
          {call.error}
        </div>
      )}
    </div>
  );
};

const SuggestionCard = ({ suggestion }: { suggestion: FinalSuggestion }) => {
  const openCli = useStore((s) => s.openCli);

  const handleClick = () => {
    // openCli adds the tab. Pre-fill of args is the responsibility of
    // CliForm; for now we stash the args on a session-scoped key so the
    // form view can pick them up. Phase 4 may wire this through the store.
    try {
      const key = `cli:prefill:${suggestion.cliId}`;
      sessionStorage.setItem(key, JSON.stringify(suggestion.args));
    } catch {
      // ignore — sessionStorage may be unavailable
    }
    openCli(suggestion.cliId);
  };

  return (
    <button
      type="button"
      data-testid={`suggestion-${suggestion.cliId}`}
      onClick={handleClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 12px',
        background: 'var(--accent-soft)',
        border: '1px solid rgba(255,122,61,0.25)',
        borderRadius: 6,
        textAlign: 'left',
        width: '100%',
        cursor: 'pointer',
        color: 'inherit',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-1)', marginBottom: 3 }}>
          {suggestion.label}
        </div>
        <code
          className="mono text-3 truncate"
          style={{ fontSize: 10.5, display: 'block' }}
        >
          {`open form: ${suggestion.cliId}`}
        </code>
      </div>
      <span className="btn primary sm" aria-hidden="true">
        Open →
      </span>
    </button>
  );
};

const ChatTurn = ({ message }: { message: ChatMessage }) => (
  <div style={{ display: 'flex', gap: 10, marginBottom: 18 }}>
    <Avatar role={message.role} />
    <div style={{ flex: 1, paddingTop: 3, minWidth: 0 }}>
      {message.text && <Markdownish text={message.text} />}
      {message.toolCall && <ToolCallCard call={message.toolCall} />}
      {message.finalSuggestions && message.finalSuggestions.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
          {message.finalSuggestions.map((s, i) => (
            <SuggestionCard key={`${s.cliId}-${i}`} suggestion={s} />
          ))}
        </div>
      )}
    </div>
  </div>
);

const ChatHeader = () => {
  const setChatVisible = useStore((s) => s.setChatVisible);
  const openSettings = useStore((s) => s.openSettings);
  const resetChat = useStore((s) => s.resetChat);
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 14px',
        borderBottom: '1px solid var(--line)',
        background: 'var(--bg-2)',
      }}
    >
      <div style={{ width: 22, height: 22, position: 'relative' }} aria-hidden="true">
        <div
          style={{
            position: 'absolute',
            inset: 0,
            border: '1.5px solid var(--accent)',
            borderRadius: '50%',
          }}
        />
        <div
          style={{
            position: 'absolute',
            inset: 4,
            border: '1px dashed var(--text-2)',
            borderRadius: '50%',
          }}
        />
      </div>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 17, color: 'var(--text-0)' }}>
        Ask <em className="accent">agent</em>
      </div>
      <span className="grow" />
      <button
        type="button"
        className="btn ghost icon"
        title="New conversation"
        aria-label="New conversation"
        onClick={resetChat}
      >
        ＋
      </button>
      <button
        type="button"
        className="btn ghost icon"
        title="Agent settings"
        aria-label="Agent settings"
        onClick={openSettings}
      >
        ⚙
      </button>
      <button
        type="button"
        className="btn ghost icon"
        title="Hide chat"
        aria-label="Hide chat"
        onClick={() => setChatVisible(false)}
      >
        ×
      </button>
    </div>
  );
};

const ErrorBanner = () => {
  const error = useStore((s) => s.agent.error);
  const openSettings = useStore((s) => s.openSettings);
  if (!error) return null;
  // Provider auth / rate limit / budget — link to SettingsModal.
  const isProviderError = error.kind === 'auth' || error.kind === 'rate_limit';
  return (
    <div
      role="alert"
      data-testid="agent-error-banner"
      style={{
        background: 'var(--fail-soft)',
        borderBottom: '1px solid var(--fail)',
        color: 'var(--fail)',
        padding: '8px 14px',
        fontSize: 12,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <span className="mono" style={{ fontSize: 11 }}>
        ✗
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        {error.provider ? `${error.provider}: ` : ''}
        {error.message}
      </span>
      {isProviderError && (
        <button
          type="button"
          className="btn ghost sm"
          onClick={openSettings}
          style={{ color: 'var(--fail)' }}
        >
          Open settings
        </button>
      )}
    </div>
  );
};

const ChatComposer = () => {
  const sendChatMessage = useStore((s) => s.sendChatMessage);
  const status = useStore((s) => s.agent.status);
  const [value, setValue] = useState('');
  const disabled = status === 'streaming' || status === 'awaiting_confirmation';

  const onSend = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setValue('');
    void sendChatMessage(trimmed);
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div
      style={{
        borderTop: '1px solid var(--line)',
        padding: '12px 14px 14px',
        background: 'var(--bg-2)',
      }}
    >
      <div
        style={{
          background: 'var(--bg-1)',
          border: '1px solid var(--line-2)',
          borderRadius: 8,
          padding: '8px 10px',
        }}
      >
        <textarea
          rows={2}
          aria-label="Ask the agent"
          className="textarea"
          style={{
            background: 'transparent',
            border: 0,
            padding: 0,
            resize: 'none',
            width: '100%',
            fontFamily: 'var(--sans)',
            fontSize: 13.5,
            color: 'var(--text-1)',
            minHeight: 36,
            outline: 'none',
          }}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask anything · paste a CLI · @-mention a run"
          disabled={disabled}
        />
        <div style={{ display: 'flex', alignItems: 'center', marginTop: 6, gap: 6 }}>
          <span className="grow" />
          <span className="text-3 mono" style={{ fontSize: 10 }}>
            ↵ send · ⌘K palette
          </span>
          <button
            type="button"
            className="btn primary icon"
            aria-label="Send message"
            style={{ width: 30, height: 30, fontSize: 14 }}
            onClick={onSend}
            disabled={disabled || !value.trim()}
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
};

/* ------------------------------ panel ------------------------------- */

const ChatPanel = () => {
  const messages = useStore((s) => s.agent.messages);
  const status = useStore((s) => s.agent.status);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom when messages change.
  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, status]);

  return (
    <aside
      data-testid="chat-panel"
      style={{
        width: 420,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-1)',
        borderLeft: '1px solid var(--line)',
        minHeight: 0,
      }}
    >
      <ChatHeader />
      <ErrorBanner />
      <div
        ref={scrollRef}
        style={{ flex: 1, overflowY: 'auto', padding: '16px 18px' }}
      >
        {messages.length === 0 && (
          <div
            className="mono text-3"
            style={{ fontSize: 11, lineHeight: 1.7, padding: '24px 0' }}
          >
            {'// Ask anything about your evals.'}
            <div style={{ marginTop: 6 }}>
              {'// Tools call CLIs; writes require your approval.'}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <ChatTurn key={m.id} message={m} />
        ))}
        {status === 'streaming' && (
          <div
            data-testid="agent-thinking"
            style={{
              display: 'flex',
              gap: 10,
              marginTop: 8,
              color: 'var(--text-3)',
              fontSize: 12,
            }}
          >
            <Avatar role="assistant" />
            <span>thinking</span>
            <span>…</span>
          </div>
        )}
      </div>
      <ChatComposer />
    </aside>
  );
};

export default ChatPanel;
