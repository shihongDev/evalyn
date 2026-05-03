/**
 * InlineChat (Cmd+I).
 *
 * Ephemeral chat popover anchored near the focused/clicked element. Inspired
 * by Cursor 3 / Copilot Chat's Cmd+I — complements the dock-right ChatPanel
 * for "explain this row" / "rerun with X tweaked" style asks where the user
 * doesn't want to context-switch into a long-form chat thread.
 *
 * Behaviour:
 *   - Mounted only when `inlineChatOpen=true` (controlled by store).
 *   - Textarea autofocuses on open. Esc closes without sending. Click outside
 *     closes too. Send pipes the text into `sendChatMessage`, opens the
 *     dock-right chat panel so the user sees the response stream, and closes
 *     the popover.
 *   - Position prefers below the anchor; flips above when there isn't enough
 *     room. With a null anchor the popover centers in the viewport.
 *   - The popover keeps focus inside itself (textarea) and restores focus to
 *     the previously-focused element on close, mirroring CommandPalette.
 *
 * NOTE: the Cmd+I global shortcut is registered in App.tsx (parallel to the
 * Cmd+K shortcut) and dispatches into the store actions rather than being
 * handled here, so the keybind works even when the popover is closed.
 */

import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { useStore } from '../store';

/** Popover dimensions (kept in sync with the rendered <div> below). */
const POPOVER_WIDTH = 480;
const POPOVER_MAX_HEIGHT = 360;
/** Vertical gap between the anchor and the flipped popover edge. */
const FLIP_GAP = 8;

/** Compute fixed-position offsets for the popover given an anchor point.
 *
 *  Returns CSS top/left in viewport coordinates. When the anchor is null
 *  we center horizontally and place the popover at ~30% viewport height
 *  (modal-like) so it isn't visually buried at the very top of the screen.
 *
 *  The function is exported only for tests; production code reads it via
 *  the component itself. */
const computePosition = (
  anchor: { x: number; y: number } | null,
  viewport: { width: number; height: number },
): { top: number; left: number } => {
  if (!anchor) {
    return {
      top: Math.max(16, viewport.height * 0.3),
      left: Math.max(16, (viewport.width - POPOVER_WIDTH) / 2),
    };
  }
  // Prefer below the anchor; flip above when below would overflow.
  const fitsBelow = anchor.y + POPOVER_MAX_HEIGHT <= viewport.height;
  const top = fitsBelow
    ? anchor.y
    : Math.max(16, anchor.y - POPOVER_MAX_HEIGHT - FLIP_GAP);
  // Center horizontally on the anchor, but clamp inside the viewport.
  let left = anchor.x - POPOVER_WIDTH / 2;
  if (left < 16) left = 16;
  if (left + POPOVER_WIDTH > viewport.width - 16) {
    left = Math.max(16, viewport.width - POPOVER_WIDTH - 16);
  }
  return { top, left };
};

const InlineChat = () => {
  const open = useStore((s) => s.inlineChatOpen);
  const anchor = useStore((s) => s.inlineChatAnchor);
  const closeInlineChat = useStore((s) => s.closeInlineChat);
  const sendChatMessage = useStore((s) => s.sendChatMessage);
  const setChatVisible = useStore((s) => s.setChatVisible);

  const [value, setValue] = useState('');
  const [sending, setSending] = useState(false);
  // Recomputed on open and on window resize so a stretched/shrunk window
  // doesn't leave the popover off-screen.
  const [viewport, setViewport] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
  }));

  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const titleId = useId();

  // On open: capture previously focused element, reset value, focus textarea.
  // On close: restore focus to that element.
  useEffect(() => {
    if (open) {
      previousFocusRef.current = (document.activeElement as HTMLElement) ?? null;
      setValue('');
      setSending(false);
      const id = setTimeout(() => textareaRef.current?.focus(), 0);
      return () => clearTimeout(id);
    }
    // Guard: focus restoration is best-effort. The element may have been
    // removed (e.g. if the user navigated). Optional chaining handles both
    // missing ref and missing focus().
    previousFocusRef.current?.focus?.();
    return undefined;
  }, [open]);

  // Track viewport size while open so the position stays valid across
  // window resizes. We listen lazily (only when mounted) to avoid leaking a
  // listener for the lifetime of the app.
  useLayoutEffect(() => {
    if (!open) return undefined;
    const onResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    };
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [open]);

  // Click-outside: close when a pointerdown lands outside the popover.
  // We use pointerdown (not click) so the outside element doesn't also
  // receive the click — matches the behaviour of most popover libraries.
  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (popoverRef.current && popoverRef.current.contains(target)) return;
      closeInlineChat();
    };
    // Defer registration by a microtask: the same event that opened the
    // popover (e.g. a Cmd+I keydown landing on a button via Enter) won't
    // also close it.
    const id = setTimeout(() => {
      window.addEventListener('pointerdown', onPointerDown);
    }, 0);
    return () => {
      clearTimeout(id);
      window.removeEventListener('pointerdown', onPointerDown);
    };
  }, [open, closeInlineChat]);

  if (!open) return null;

  const { top, left } = computePosition(anchor, viewport);

  const submit = () => {
    if (sending) return;
    const text = value.trim();
    if (!text) {
      // Empty submit is a no-op but we keep the popover open so the user
      // can either type something or press Esc.
      return;
    }
    setSending(true);
    void sendChatMessage(text);
    setChatVisible(true);
    closeInlineChat();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      closeInlineChat();
      return;
    }
    // Cmd+I inside the popover: the global App.tsx handler is attached at
    // window-level and would otherwise see the same keypress and toggle the
    // popover back open. We stop the native event so the window listener
    // never fires for this keystroke. (React's synthetic stopPropagation
    // alone doesn't stop the underlying DOM event from reaching window
    // listeners attached outside React's delegation tree.)
    if ((e.metaKey || e.ctrlKey) && (e.key === 'i' || e.key === 'I')) {
      e.preventDefault();
      e.nativeEvent.stopImmediatePropagation();
      closeInlineChat();
      return;
    }
    // Enter (without shift) submits; Shift+Enter inserts a newline.
    if (e.key === 'Enter' && !e.shiftKey && e.target === textareaRef.current) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div
      ref={popoverRef}
      role="dialog"
      aria-modal="false"
      aria-labelledby={titleId}
      data-testid="inline-chat"
      onKeyDown={onKeyDown}
      style={{
        position: 'fixed',
        top,
        left,
        width: POPOVER_WIDTH,
        maxWidth: 'calc(100vw - 32px)',
        maxHeight: POPOVER_MAX_HEIGHT,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-2)',
        border: '1px solid var(--line)',
        borderRadius: 12,
        boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
        overflow: 'hidden',
        zIndex: 60,
      }}
    >
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--line)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span
          id={titleId}
          className="text-3 mono"
          style={{
            fontSize: 10,
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
          }}
        >
          Inline chat
        </span>
        <span className="text-3" style={{ fontSize: 11 }}>
          <kbd className="mono">esc</kbd> close
        </span>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: 12 }}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask the agent…"
          aria-label="Inline chat message"
          data-testid="inline-chat-input"
          rows={3}
          style={{
            width: '100%',
            border: '1px solid var(--line)',
            borderRadius: 8,
            background: 'var(--bg-1)',
            color: 'var(--text-0)',
            fontFamily: 'var(--mono)',
            fontSize: 13,
            padding: '8px 10px',
            resize: 'vertical',
            minHeight: 60,
            maxHeight: 220,
            outline: 'none',
          }}
        />
      </div>

      <div
        style={{
          padding: '8px 12px',
          borderTop: '1px solid var(--line)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          background: 'var(--bg-1)',
        }}
      >
        <span className="text-3" style={{ fontSize: 11 }}>
          <kbd className="mono">↵</kbd> send · <kbd className="mono">shift+↵</kbd> newline
        </span>
        <button
          type="button"
          className="btn primary"
          data-testid="inline-chat-send"
          onClick={submit}
          disabled={sending || value.trim().length === 0}
          style={{ height: 28, padding: '0 14px', fontSize: 12 }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default InlineChat;
