/**
 * CoPilotDock - the right-side dock variant of the co-pilot.
 *
 * Layout adapts via the `mode` prop:
 *   - 'docked'  : in-grid 420px column on desktop (default).
 *   - 'overlay' : fixed 420px panel slid in from the right (tablet).
 *   - 'sheet'   : fixed bottom sheet, ~75vh tall (mobile).
 *
 * Wired to /api/agent/chat + /ws/agent/{tid} via useCoPilotThread.
 */

import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { E } from '../tokens';
import { Btn, Pill } from '../ui';
import { Bubble, PlanCard, ThinkingBubble, ToolBlock } from './atoms';
import { linkifyText, makeUrlCounter } from '../textRender';
import { MOD_KEY } from '../platform';
import { useCoPilotThread } from './useCoPilotThread';
import { useStickToBottom } from '../hooks/useStickToBottom';
import { deriveContextChips, prependContextTag, type ContextKind } from './contextChips';
import {
  clearCoPilotDraft,
  loadCoPilotDraft,
  saveCoPilotDraft,
} from './copilotDrafts';

export type CoPilotDockMode = 'docked' | 'overlay' | 'sheet';

const EXAMPLE_PROMPTS = [
  "What's my pass rate trend?",
  'Show me the worst failure cluster',
  'Explain my rubric',
];

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
        background: E.panel3,
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

interface CoPilotDockProps {
  onClose: () => void;
  /** Layout mode. Defaults to 'docked' for backwards compatibility. */
  mode?: CoPilotDockMode;
}

export function CoPilotDock({ onClose, mode = 'docked' }: CoPilotDockProps) {
  const {
    messages,
    pending,
    send,
    confirm,
    status,
    threadId,
    error,
    clearError,
    reconnecting,
  } = useCoPilotThread();
  // Initialize draft from localStorage so a refresh / dock-reopen
  // doesn't lose half-typed work. Per-thread keying so a draft on
  // thread A doesn't leak into thread B if the user switches.
  const [draft, setDraft] = useState(() => loadCoPilotDraft(threadId));
  const navigate = useNavigate();
  const location = useLocation();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-focus the composer when the dock first opens. The dock is
  // conditionally mounted (AppShell renders it only when dockOpen is
  // true) so first-mount = first-open, which is exactly when the user
  // wants the textarea ready for typing.
  useEffect(() => {
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  // When the active thread changes (navigation, new thread creation),
  // load that thread's draft. Lets users switch between threads
  // mid-compose without losing either draft.
  useEffect(() => {
    setDraft(loadCoPilotDraft(threadId));
  }, [threadId]);

  // Auto-save the draft on every change, debounced 250ms so each
  // keystroke isn't a JSON.stringify + setItem pair. Empty drafts
  // are removed from storage rather than persisted as empty
  // (cleaner localStorage for users who type and then delete).
  useEffect(() => {
    const handle = window.setTimeout(() => {
      saveCoPilotDraft(threadId, draft);
    }, 250);
    return () => window.clearTimeout(handle);
  }, [threadId, draft]);

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
      // Place cursor at the end so the user can keep typing immediately.
      const len = el.value.length;
      el.setSelectionRange(len, len);
    });
  };

  const handleExpand = () => {
    navigate(threadId ? `/copilot/${threadId}` : '/copilot');
  };

  const submit = () => {
    // Defensive guard: the submit Btn and the textarea are both
    // `disabled` while status === 'streaming' or pending != null,
    // which prevents user-driven double-fires under normal latency.
    // But React's batched state updates mean a rapid double-click
    // within the same paint frame sees the OLD closure with
    // status='idle', and both invocations would pass through.
    // Bail explicitly so the second click is a no-op even pre-render.
    if (status === 'streaming' || pending != null) return;
    const t = draft.trim();
    if (!t) return;
    setDraft('');
    // Clear synchronously: the debounced save effect would otherwise
    // race with the navigate-after-send and persist a stale draft
    // back into storage if the threadId changes mid-flight.
    clearCoPilotDraft(threadId);
    void send(t);
  };

  // Auto-scroll the dock conversation pane on new messages / streaming
  // updates. Same hook as the full /copilot route, smaller container.
  const {
    scrollRef: convRef,
    onScroll: onConvScroll,
    scrolledUp,
    jumpToBottom,
  } = useStickToBottom(messages);

  // Per-mode wrapper styles. The dock body markup itself is identical across
  // modes; only positioning + animation change here.
  const wrapperStyle: CSSProperties =
    mode === 'overlay'
      ? {
          position: 'fixed',
          top: 56,
          right: 0,
          bottom: 0,
          width: 420,
          maxWidth: '92vw',
          background: E.panel,
          borderLeft: `1px solid ${E.hair}`,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          zIndex: 800,
          boxShadow: '-8px 0 24px rgba(0,0,0,0.16)',
          animation: 'eSlideInRight 220ms ease',
        }
      : mode === 'sheet'
        ? {
            position: 'fixed',
            left: 0,
            right: 0,
            bottom: 0,
            height: '75vh',
            background: E.panel,
            borderTop: `1px solid ${E.hair}`,
            borderTopLeftRadius: 14,
            borderTopRightRadius: 14,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            zIndex: 800,
            boxShadow: '0 -8px 24px rgba(0,0,0,0.18)',
            animation: 'eSlideUp 220ms ease',
          }
        : {
            background: E.panel,
            borderLeft: `1px solid ${E.hair}`,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          };

  // Backdrop click closes the floating modes. Docked mode renders no backdrop.
  const backdrop =
    mode === 'docked' ? null : (
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          top: mode === 'overlay' ? 56 : 0,
          background: 'rgba(20, 18, 14, 0.45)',
          zIndex: 790,
          animation: 'eFadeIn 140ms ease',
        }}
      />
    );

  return (
    <>
      {backdrop}
      <div style={wrapperStyle}>
      <div
        style={{
          padding: '14px 18px',
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <div
          style={{
            width: 24,
            height: 24,
            borderRadius: 6,
            background: `linear-gradient(135deg, ${E.ember}, #b8501f)`,
            color: E.emberInk,
            fontWeight: 700,
            fontFamily: E.fSerif,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
          }}
        >
          e
        </div>
        <span style={{ fontFamily: E.fSerif, fontSize: 17, color: E.text0 }}>Co-pilot</span>
        <Pill mono color={E.text3} bg={E.panel2} style={{ fontSize: 10 }}>
          {messages.length === 0 ? 'new' : `${messages.length} turns`}
        </Pill>
        {reconnecting && (
          <span
            role="status"
            aria-live="polite"
            aria-label="Reconnecting to agent stream"
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
              whiteSpace: 'nowrap',
            }}
          >
            <span aria-hidden style={{ lineHeight: 1, fontSize: 11 }}>◌</span>
            Reconnecting
          </span>
        )}
        <span style={{ flex: 1 }} />
        <button
          type="button"
          onClick={handleExpand}
          title="Open full thread"
          aria-label="Open full thread"
          style={{
            color: E.text3,
            fontSize: 13,
            padding: 4,
            cursor: 'pointer',
            background: 'transparent',
            border: 'none',
            lineHeight: 1,
          }}
        >
          <span aria-hidden="true">⛶</span>
        </button>
        <button
          type="button"
          onClick={onClose}
          title="Close co-pilot"
          aria-label="Close co-pilot"
          style={{
            color: E.text3,
            fontSize: 16,
            padding: 4,
            cursor: 'pointer',
            background: 'transparent',
            border: 'none',
          }}
        >
          <span aria-hidden="true">×</span>
        </button>
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
        style={{ flex: 1, overflowY: 'auto', padding: '18px 18px 8px' }}
      >
        {messages.length === 0 && (
          <div style={{ padding: '20px 4px' }}>
            <div style={{ color: E.text3, fontSize: 12.5, lineHeight: 1.55 }}>
              Ask anything about your evals. The co-pilot can read runs, datasets, and rubrics on
              its own. Anything that writes will pause for your confirmation first.
            </div>
            <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
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
                    alignSelf: 'flex-start',
                    padding: '4px 10px',
                    borderRadius: 999,
                    fontSize: 11,
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
        {/* See CoPilotThread for context on the empty-beat indicator. */}
        {status === 'streaming' &&
          messages.length > 0 &&
          messages[messages.length - 1].role === 'you' && <ThinkingBubble />}
        {error && (
          <div
            role="alert"
            style={{
              marginTop: 12,
              padding: '8px 12px',
              background: E.failDim,
              border: `1px solid ${E.fail}33`,
              borderRadius: 6,
              color: E.fail,
              fontSize: 11.5,
              fontFamily: E.fMono,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 6,
              lineHeight: 1.4,
            }}
          >
            <span style={{ flex: 1 }}>{error}</span>
            <button
              type="button"
              onClick={clearError}
              aria-label="Dismiss error"
              style={{
                background: 'transparent',
                border: 'none',
                color: 'currentColor',
                cursor: 'pointer',
                fontSize: 13,
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
      {scrolledUp && (
        <button
          type="button"
          onClick={jumpToBottom}
          aria-label="Jump to latest message"
          style={{
            position: 'absolute',
            bottom: 12,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '5px 12px',
            borderRadius: 999,
            background: E.ember,
            color: E.emberInk,
            border: 'none',
            cursor: 'pointer',
            fontSize: 11,
            fontFamily: E.fMono,
            fontWeight: 500,
            boxShadow: `0 6px 18px rgba(217,106,44,0.32)`,
            zIndex: 5,
            animation: 'eRouteFallbackFadeIn 200ms ease',
            whiteSpace: 'nowrap',
          }}
        >
          ↓ Jump to latest
        </button>
      )}
      </div>

      <div style={{ padding: 14, borderTop: `1px solid ${E.hair}` }}>
        <div
          style={{
            background: E.panel2,
            border: `1px solid ${E.hair2}`,
            borderRadius: 10,
            padding: '10px 12px',
          }}
        >
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
            aria-label="Co-pilot prompt"
            placeholder="Ask anything about your evals…"
            disabled={status === 'streaming' || pending != null}
            rows={2}
            style={{
              width: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: E.text1,
              fontSize: 13,
              fontFamily: E.fSans,
              resize: 'none',
              // Auto-grow with content. Bounded so a long question
              // doesn't push the input out of the dock - older
              // browsers fall back to rows=2.
              fieldSizing: 'content',
              minHeight: 38,
              maxHeight: 200,
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
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
            <span style={{ flex: 1 }} />
            <Btn
              kind="primary"
              size="sm"
              onClick={submit}
              disabled={!draft.trim() || status === 'streaming' || pending != null}
              aria-busy={status === 'streaming' || pending != null}
              // Spell out the keys for SR (Cmd Enter / Ctrl Enter) -
              // raw ⌘ / ⏎ glyphs are inconsistent across engines.
              aria-label={`Send message (${MOD_KEY === '⌘' ? 'Cmd Enter' : 'Ctrl Enter'})`}
              title={`Send (${MOD_KEY} ⏎)`}
              style={{ width: 26, height: 26, padding: 0, justifyContent: 'center' }}
            >
              <span aria-hidden="true">↑</span>
            </Btn>
          </div>
        </div>
        <div
          style={{
            marginTop: 8,
            fontSize: 10,
            color: E.text3,
            fontFamily: E.fMono,
            textAlign: 'center',
          }}
        >
          {MOD_KEY} ⏎ to send · read-only commands run automatically · writes ask first
        </div>
      </div>
      </div>
    </>
  );
}
