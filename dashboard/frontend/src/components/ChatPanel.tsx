/**
 * ChatPanel — agent chat dock-right.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-chat.jsx and progressively
 * upgraded toward Claude.ai-parity (P1 chat surface upgrade, see
 * docs/superpowers/research/2026-05-01-dashboard-ux/). Visual chrome stays
 * close to the mock; data wiring is the Zustand store's agent slice.
 *
 * Renders:
 *   - Header: "Ask Evalyn" title + new-conversation / settings / close.
 *   - Optional non-blocking API-key banner (no active provider).
 *   - Provider error banner (when `agent.error` is set).
 *   - Scrollable history of ChatMessage entries:
 *       * text bubbles with hover-revealed actions (copy / retry / time)
 *       * collapsible tool-call cards with one-line summary + peek
 *       * confirmation cards (approve / reject)
 *       * final-suggestion cards (open CliForm tab)
 *   - Smart auto-scroll: pinned to bottom when within 80px; otherwise a
 *     "↓ N new" pill appears bottom-right.
 *   - Streaming caret (▌) at the end of in-progress assistant text.
 *   - Empty-state suggestion chips (cold/warm-start variants).
 *   - Composer at the bottom (textarea + send button).
 *
 * Width 420px, full-height, border-left. Dock-right placement only.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  chatEmptyStateSuggestions,
  serializeChatAttachments,
  useStore,
} from '../store';
import type { ChatThread } from '../store';
import type {
  ChatMessage,
  FinalSuggestion,
  ToolCall,
  ToolCallStatus,
} from '../types/agent';
import AttachChip from './AttachChip';
import { parseSlashCommand, SLASH_HELP_TEXT } from '../lib/slashCommands';
import { api } from '../api';

const STATUS_LABEL: Record<ToolCallStatus, string> = {
  proposed: 'proposed',
  awaiting_confirmation: 'awaiting confirmation',
  running: 'running...',
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

const STATUS_GLYPH: Record<ToolCallStatus, string> = {
  proposed: '·',
  awaiting_confirmation: '·',
  running: '▸',
  complete: '✓',
  rejected: '✗',
  error: '✗',
};

/** Format an ISO timestamp as HH:MM. Falls back to '' on invalid input. */
const formatTime = (iso?: string): string => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
};

/** "5m ago", "3h ago", "2d ago", or HH:MM if same day. */
const formatModifiedAgo = (epochMs: number): string => {
  const now = Date.now();
  const diffSec = Math.max(0, Math.floor((now - epochMs) / 1000));
  if (diffSec < 60) return 'just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  const d = new Date(epochMs);
  return `${d.getMonth() + 1}/${d.getDate()}`;
};

/** Truncate at `n` chars with ellipsis. */
const truncate = (s: string, n: number): string =>
  s.length <= n ? s : `${s.slice(0, n - 1)}...`;

/** Render an argv value for the read-only confirmation card display. */
const formatArgValue = (v: unknown): string => {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean' || typeof v === 'number') return String(v);
  return JSON.stringify(v);
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

/**
 * Tiny markdown renderer: supports `**bold**` and `` `code` ``. When
 * `streaming` is true a blinking caret is appended after the content so the
 * user sees the assistant is still typing without a layout reflow.
 */
const Markdownish = ({ text, streaming }: { text: string; streaming?: boolean }) => {
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
      {streaming && (
        <span aria-hidden="true" data-testid="streaming-caret" className="caret-streaming">
          ▌
        </span>
      )}
    </div>
  );
};

/**
 * Inline args editor used inside the confirmation card. Renders one row per
 * arg with a plain text input. v1 keeps it stringly-typed; per-kind controls
 * (sliders, file pickers, etc.) are deferred until the form-side P1 lands.
 *
 * The editor owns its draft state until the user clicks Save; on Save we
 * call `onSave(values)` with a fully-coerced map (booleans stay booleans,
 * arrays stay arrays — only string-typed values were rendered as inputs).
 */
const ArgsEditor = ({
  args,
  callId,
  onSave,
  onCancel,
}: {
  args: Record<string, unknown>;
  callId: string;
  onSave: (next: Record<string, unknown>) => void;
  onCancel: () => void;
}) => {
  // Hold each value as a string in local state so the inputs stay controlled
  // even when the underlying type was non-string. Arrays/objects round-trip
  // through JSON; on Save the original type is restored from `args[k]`.
  const initial = useMemo<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(args)) out[k] = formatArgValue(v);
    return out;
  }, [args]);

  const [draft, setDraft] = useState<Record<string, string>>(initial);

  const handleSave = () => {
    const next: Record<string, unknown> = {};
    for (const [k, raw] of Object.entries(draft)) {
      const original = args[k];
      if (typeof original === 'boolean') {
        next[k] = raw === 'true';
      } else if (typeof original === 'number') {
        // An empty/whitespace input is the user's way of saying "drop the
        // value." `Number('')` would silently coerce to 0 and the agent
        // would still pass `--flag 0` — usually NOT what the user meant.
        // Keep the raw empty string so buildCli's "skip empty" path fires.
        if (raw.trim() === '') {
          next[k] = '';
          continue;
        }
        const n = Number(raw);
        next[k] = Number.isFinite(n) ? n : raw;
      } else if (Array.isArray(original) || (original !== null && typeof original === 'object')) {
        // Best-effort JSON parse; fall back to the original value if the
        // user typed garbage so we don't drop the field silently. Also
        // require the parsed shape to match the original's container kind
        // (array vs object) — otherwise typing "5" into an array field
        // would silently swap the type and the agent would mis-handle it.
        try {
          const parsed = JSON.parse(raw);
          const originalIsArray = Array.isArray(original);
          if (originalIsArray !== Array.isArray(parsed)) {
            next[k] = original;
          } else {
            next[k] = parsed;
          }
        } catch {
          next[k] = original;
        }
      } else {
        next[k] = raw;
      }
    }
    onSave(next);
  };

  const keys = Object.keys(args);
  return (
    <div
      data-testid={`args-editor-${callId}`}
      style={{ display: 'flex', flexDirection: 'column', gap: 6 }}
    >
      {keys.length === 0 && (
        <div className="text-3" style={{ fontSize: 11 }}>
          (no arguments)
        </div>
      )}
      {keys.map((k) => (
        <div
          key={k}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <code
            className="mono text-2"
            style={{ fontSize: 11, minWidth: 92, flexShrink: 0 }}
          >
            --{k.replace(/_/g, '-')}
          </code>
          <input
            type="text"
            data-testid={`args-editor-input-${callId}-${k}`}
            value={draft[k] ?? ''}
            onChange={(e) => setDraft((prev) => ({ ...prev, [k]: e.target.value }))}
            style={{
              flex: 1,
              minWidth: 0,
              fontFamily: 'var(--mono)',
              fontSize: 11,
              padding: '3px 6px',
              background: 'var(--bg-1)',
              border: '1px solid var(--line)',
              borderRadius: 3,
              color: 'var(--text-1)',
            }}
          />
        </div>
      ))}
      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        <button
          type="button"
          className="btn primary sm"
          data-testid={`args-editor-save-${callId}`}
          onClick={(e) => {
            e.stopPropagation();
            handleSave();
          }}
        >
          Save
        </button>
        <button
          type="button"
          className="btn ghost sm"
          data-testid={`args-editor-cancel-${callId}`}
          onClick={(e) => {
            e.stopPropagation();
            onCancel();
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

/**
 * Collapsible tool-call card.
 *
 * Default state:
 *   - `complete` → collapsed (one-line summary + 8-line peek)
 *   - `error`    → expanded and locked open (errored output is critical)
 *   - everything else → expanded (running / awaiting confirmation must be visible)
 *
 * Click anywhere on the header row toggles collapse, except on the
 * approve/reject buttons inside the awaiting-confirmation strip.
 */
const ToolCallCard = ({ call }: { call: ToolCall }) => {
  const confirmAgent = useStore((s) => s.confirmAgent);
  const pendingConfirmation = useStore((s) => s.agent.pendingConfirmation);
  const isAwaiting = call.status === 'awaiting_confirmation';
  const isErrored = call.status === 'error';
  const tone = STATUS_TONE[call.status];

  // Inline args edit state. `editedArgs` is null until the user has clicked
  // Save in the editor; the Approve buttons read from editedArgs ?? call.args
  // so an unsaved edit never leaks into the confirm payload.
  const [editing, setEditing] = useState<boolean>(false);
  const [editedArgs, setEditedArgs] = useState<Record<string, unknown> | null>(null);

  // The runtime emits the curated bullets on confirmation_required; surface
  // them through the store. Fall back to a generic line if absent (keeps the
  // card useful when an older runtime version omits the field).
  const sideEffects =
    pendingConfirmation && pendingConfirmation.toolCallId === call.id
      ? pendingConfirmation.sideEffects
      : undefined;
  const bullets =
    sideEffects && sideEffects.length > 0
      ? sideEffects
      : ['This is a write command and may modify your project on disk.'];

  const liveArgs = editedArgs ?? call.args;

  // Recomputed each render so a status transition from `running` -> `complete`
  // triggers auto-collapse.
  const defaultExpanded = call.status !== 'complete';
  const [expanded, setExpanded] = useState<boolean>(defaultExpanded);
  const [touched, setTouched] = useState<boolean>(false);

  // Errors lock the card open so the failure context stays visible.
  const effectiveExpanded = isErrored || (touched ? expanded : defaultExpanded);

  const lines = call.output ? call.output.split('\n') : [];
  const lineCount = lines.length;
  const peek = lines.slice(0, 8).join('\n');
  const hasMoreThanPeek = lineCount > 8;

  const summary = call.previewCmd ?? `evalyn ${call.tool}`;

  let headerGlyph: string;
  if (isErrored) headerGlyph = STATUS_GLYPH[call.status];
  else headerGlyph = effectiveExpanded ? '▾' : '▸';

  const onHeaderToggle = () => {
    if (isErrored) return;
    setTouched(true);
    setExpanded((v) => (touched ? !v : !defaultExpanded));
  };

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
      <button
        type="button"
        data-testid={`tool-call-header-${call.id}`}
        onClick={onHeaderToggle}
        aria-expanded={effectiveExpanded}
        aria-disabled={isErrored ? 'true' : undefined}
        style={{
          all: 'unset',
          cursor: isErrored ? 'default' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          borderBottom: effectiveExpanded || call.output ? '1px solid var(--line)' : 'none',
          background: 'var(--bg-2)',
          width: '100%',
          boxSizing: 'border-box',
        }}
      >
        <span className={`mono ${tone}`} style={{ fontSize: 11 }} aria-hidden="true">
          {headerGlyph}
        </span>
        <code
          className="mono truncate"
          style={{ fontSize: 11, color: 'var(--text-1)', flex: 1, minWidth: 0 }}
        >
          $ {summary}
        </code>
        <span className={`chip mono ${tone}`} style={{ fontSize: 10 }}>
          {STATUS_LABEL[call.status]}
          {call.output ? ` · ${lineCount} line${lineCount === 1 ? '' : 's'}` : ''}
        </span>
      </button>

      {isAwaiting && (
        <div
          data-testid={`confirm-card-${call.id}`}
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            padding: '12px 14px',
            borderBottom: '1px solid var(--line)',
            background: 'var(--warn-soft)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* COMMAND block */}
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 6,
              }}
            >
              <span
                className="mono text-3"
                style={{ fontSize: 10, fontWeight: 600, letterSpacing: 0.5 }}
              >
                COMMAND
              </span>
              {editedArgs && !editing && (
                <span
                  className="chip mono accent"
                  data-testid={`args-edited-chip-${call.id}`}
                  style={{ fontSize: 10 }}
                >
                  args edited
                </span>
              )}
              <span className="grow" />
              {!editing && (
                <button
                  type="button"
                  className="btn ghost sm"
                  data-testid={`edit-args-${call.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditing(true);
                  }}
                  style={{ fontSize: 11 }}
                >
                  edit args
                </button>
              )}
            </div>
            {editing ? (
              <ArgsEditor
                args={liveArgs}
                callId={call.id}
                onSave={(next) => {
                  setEditedArgs(next);
                  setEditing(false);
                }}
                onCancel={() => setEditing(false)}
              />
            ) : (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2,
                }}
              >
                <code
                  className="mono"
                  style={{ fontSize: 11.5, color: 'var(--text-1)' }}
                >
                  evalyn {call.tool}
                </code>
                {Object.entries(liveArgs).map(([k, v]) => (
                  <code
                    key={k}
                    className="mono text-2"
                    style={{
                      fontSize: 11,
                      paddingLeft: 16,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all',
                    }}
                  >
                    --{k.replace(/_/g, '-')}{'  '}
                    {formatArgValue(v)}
                  </code>
                ))}
              </div>
            )}
          </div>

          {/* THIS WILL block */}
          {!editing && (
            <div>
              <div
                className="mono text-3"
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: 0.5,
                  marginBottom: 4,
                }}
              >
                THIS WILL
              </div>
              <ul
                data-testid={`side-effects-${call.id}`}
                style={{
                  margin: 0,
                  paddingLeft: 18,
                  fontSize: 12,
                  color: 'var(--text-1)',
                  lineHeight: 1.55,
                }}
              >
                {bullets.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Action buttons. Hidden while the inline editor is open (the
              editor's Save returns to read-only state before the user can
              confirm). */}
          {!editing && (
            <div
              style={{
                display: 'flex',
                gap: 6,
                flexWrap: 'wrap',
                alignItems: 'center',
              }}
            >
              <button
                type="button"
                className="btn ghost sm"
                data-testid={`reject-${call.id}`}
                onClick={(e) => {
                  e.stopPropagation();
                  void confirmAgent(false, call.id);
                }}
              >
                Reject
              </button>
              <button
                type="button"
                className="btn primary sm"
                data-testid={`approve-${call.id}`}
                onClick={(e) => {
                  e.stopPropagation();
                  void confirmAgent(
                    true,
                    call.id,
                    editedArgs ? { argsOverride: editedArgs } : undefined,
                  );
                }}
              >
                Approve once
              </button>
              <button
                type="button"
                className="btn ghost sm"
                data-testid={`approve-session-${call.id}`}
                title={`Approve this and don't ask again this session for ${call.tool}`}
                onClick={(e) => {
                  e.stopPropagation();
                  void confirmAgent(true, call.id, {
                    autoApproveSession: true,
                    ...(editedArgs ? { argsOverride: editedArgs } : {}),
                  });
                }}
              >
                Approve · session
              </button>
            </div>
          )}
        </div>
      )}

      {/* Peek: shown when collapsed and there's output. Muted color so the
          user can decide whether to expand. */}
      {!effectiveExpanded && call.output && (
        <pre
          data-testid={`tool-call-peek-${call.id}`}
          className="mono text-3"
          style={{
            margin: 0,
            padding: '8px 12px',
            fontSize: 10.5,
            lineHeight: 1.65,
            whiteSpace: 'pre-wrap',
            overflow: 'hidden',
            maxHeight: 8 * 1.65 * 10.5 + 16,
            background: 'var(--bg-1)',
          }}
        >
          {peek}
          {hasMoreThanPeek ? '\n...' : ''}
        </pre>
      )}

      {effectiveExpanded && call.output && (
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

/**
 * Per-message hover-revealed action row. Uses opacity rather than display
 * so the card height stays constant whether or not the user is hovering.
 */
const MessageActions = ({
  message,
  onCopy,
  onRetry,
  onBranch,
  hovered,
}: {
  message: ChatMessage;
  onCopy: () => void;
  onRetry?: () => void;
  onBranch?: () => void;
  hovered: boolean;
}) => {
  const ts = formatTime(message.ts);
  return (
    <div
      data-testid={`message-actions-${message.id}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        marginTop: 6,
        height: 18,
        opacity: hovered ? 1 : 0,
        visibility: hovered ? 'visible' : 'hidden',
        transition: 'opacity 120ms ease',
      }}
    >
      {ts && (
        <span className="text-3 mono" style={{ fontSize: 10 }} aria-label="Sent at">
          {ts}
        </span>
      )}
      <button
        type="button"
        className="btn ghost sm"
        title="Copy message"
        aria-label="Copy message"
        data-testid={`copy-${message.id}`}
        onClick={onCopy}
        style={{ fontSize: 11, padding: '0 6px', height: 18, lineHeight: '18px' }}
      >
        ⎘
      </button>
      {onRetry && (
        <button
          type="button"
          className="btn ghost sm"
          title="Retry from previous user message"
          aria-label="Retry"
          data-testid={`retry-${message.id}`}
          onClick={onRetry}
          style={{ fontSize: 11, padding: '0 6px', height: 18, lineHeight: '18px' }}
        >
          ↻
        </button>
      )}
      {onBranch && (
        <button
          type="button"
          className="btn ghost sm"
          title="Branch a new thread from this message"
          aria-label="Branch from here"
          data-testid={`branch-${message.id}`}
          onClick={onBranch}
          style={{ fontSize: 11, padding: '0 6px', height: 18, lineHeight: '18px' }}
        >
          branch
        </button>
      )}
    </div>
  );
};

/**
 * One message row. Tracks its own hover state so the actions row reveals
 * without re-rendering the whole list.
 */
const ChatTurn = ({
  message,
  onRetry,
  onBranch,
}: {
  message: ChatMessage;
  onRetry?: () => void;
  onBranch?: () => void;
}) => {
  const [hovered, setHovered] = useState(false);

  // Build a copy-able representation: text content plus any tool output.
  const copyText = (): void => {
    const parts: string[] = [];
    if (message.text) parts.push(message.text);
    if (message.toolCall?.output) parts.push(message.toolCall.output);
    const txt = parts.join('\n\n');
    if (!txt) return;
    try {
      void navigator.clipboard?.writeText(txt);
    } catch {
      // navigator.clipboard may be missing in jsdom; silent fail keeps the
      // happy-path test simple and avoids polluting the panel.
    }
  };

  return (
    <div
      data-testid={`chat-turn-${message.id}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setHovered(true)}
      onBlur={() => setHovered(false)}
      style={{ display: 'flex', gap: 10, marginBottom: 18 }}
    >
      <Avatar role={message.role} />
      <div style={{ flex: 1, paddingTop: 3, minWidth: 0 }}>
        {message.text && <Markdownish text={message.text} streaming={message.streaming} />}
        {message.toolCall && <ToolCallCard call={message.toolCall} />}
        {message.finalSuggestions && message.finalSuggestions.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
            {message.finalSuggestions.map((s, i) => (
              <SuggestionCard key={`${s.cliId}-${i}`} suggestion={s} />
            ))}
          </div>
        )}
        <MessageActions
          message={message}
          hovered={hovered}
          onCopy={copyText}
          onRetry={message.role === 'assistant' ? onRetry : undefined}
          onBranch={message.role === 'assistant' ? onBranch : undefined}
        />
      </div>
    </div>
  );
};

/**
 * Read a thread's persisted messages out of localStorage. Mirrors the
 * storage layout in store.ts (`evalyn:threads:msgs:${id}`). Returns [] when
 * storage is unavailable or the entry is missing/corrupt.
 */
const _readPersistedMessages = (id: string): ChatMessage[] => {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(`evalyn:threads:msgs:${id}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMessage[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

type SaveStatus = 'idle' | 'saving' | 'ok' | 'error';

const SAVE_LABEL: Record<SaveStatus, string> = {
  idle: '↓ disk',
  saving: '…',
  ok: '✓ saved',
  error: '✗ failed',
};

const SAVE_TITLE: Record<SaveStatus, string> = {
  idle: 'Save thread to disk as markdown',
  saving: 'Save thread to disk as markdown',
  ok: 'Saved to disk',
  error: 'Save failed',
};

const SAVE_CLASS: Record<SaveStatus, string> = {
  idle: '',
  saving: '',
  ok: ' pass',
  error: ' fail',
};

/**
 * Dropdown listing persisted threads; opens below the title. Active row is
 * marked with a filled bullet. Click outside closes; Esc closes.
 */
const ThreadSwitcherDropdown = ({
  threads,
  activeThreadId,
  liveMessages,
  onSelect,
  onNew,
  onClose,
}: {
  threads: ChatThread[];
  activeThreadId: string | null;
  /** In-memory messages for the active thread (not yet flushed to storage). */
  liveMessages: ChatMessage[];
  onSelect: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      const el = containerRef.current;
      if (el && !el.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  const sorted = useMemo(
    () => [...threads].sort((a, b) => b.modifiedAt - a.modifiedAt),
    [threads],
  );

  const [saveStatus, setSaveStatus] = useState<Record<string, SaveStatus>>({});

  const onSaveDisk = async (thread: ChatThread, e: React.MouseEvent) => {
    e.stopPropagation();
    setSaveStatus((m) => ({ ...m, [thread.id]: 'saving' }));
    // For the active thread, the in-memory log is the freshest source. For
    // background threads, read from the persisted snapshot.
    const messages =
      thread.id === activeThreadId && liveMessages.length > 0
        ? liveMessages
        : _readPersistedMessages(thread.id);
    try {
      await api.saveThreadToDisk({
        id: thread.id,
        title: thread.title,
        messages,
      });
      setSaveStatus((m) => ({ ...m, [thread.id]: 'ok' }));
      setTimeout(
        () => setSaveStatus((m) => ({ ...m, [thread.id]: 'idle' })),
        2000,
      );
    } catch (err) {
      console.error('save thread failed', err);
      setSaveStatus((m) => ({ ...m, [thread.id]: 'error' }));
      setTimeout(
        () => setSaveStatus((m) => ({ ...m, [thread.id]: 'idle' })),
        3000,
      );
    }
  };

  return (
    <div
      ref={containerRef}
      data-testid="thread-switcher-dropdown"
      role="menu"
      style={{
        position: 'absolute',
        top: '100%',
        left: 0,
        marginTop: 4,
        width: 320,
        background: 'var(--bg-1)',
        border: '1px solid var(--line)',
        borderRadius: 8,
        boxShadow: 'var(--shadow-soft)',
        zIndex: 50,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          maxHeight: 280,
          overflowY: 'auto',
        }}
      >
        {sorted.length === 0 && (
          <div
            className="text-3"
            style={{ padding: '12px 14px', fontSize: 12 }}
          >
            No saved threads yet.
          </div>
        )}
        {sorted.map((t) => {
          const active = t.id === activeThreadId;
          const status = saveStatus[t.id] ?? 'idle';
          // Wrapping the row in a div-not-button avoids the nested-button
          // a11y violation when we add an inner Save button. The outer div
          // gets a click handler that calls onSelect, while the inner Save
          // button stops propagation.
          return (
            <div
              key={t.id}
              data-testid={`thread-row-${t.id}`}
              role="menuitem"
              tabIndex={0}
              onClick={() => onSelect(t.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(t.id);
                }
              }}
              style={{
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                width: '100%',
                boxSizing: 'border-box',
                padding: '8px 12px',
                background: active ? 'var(--bg-2)' : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (!active) (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-2)';
              }}
              onMouseLeave={(e) => {
                if (!active) (e.currentTarget as HTMLDivElement).style.background = 'transparent';
              }}
            >
              <span
                aria-hidden="true"
                className="mono"
                style={{
                  fontSize: 11,
                  color: active ? 'var(--accent)' : 'var(--text-3)',
                  width: 12,
                  textAlign: 'center',
                }}
              >
                {active ? '●' : '○'}
              </span>
              <span
                className="truncate"
                style={{
                  flex: 1,
                  minWidth: 0,
                  fontSize: 13,
                  color: 'var(--text-1)',
                }}
              >
                {truncate(t.title, 50)}
              </span>
              <button
                type="button"
                data-testid={`thread-save-disk-${t.id}`}
                aria-label={`Save thread ${t.title} to disk`}
                title={SAVE_TITLE[status]}
                onClick={(e) => void onSaveDisk(t, e)}
                disabled={status === 'saving'}
                className={`btn ghost sm${SAVE_CLASS[status]}`}
                style={{ fontSize: 10, padding: '0 6px' }}
              >
                {SAVE_LABEL[status]}
              </button>
              <span
                className="text-3 mono"
                style={{ fontSize: 10, flexShrink: 0 }}
              >
                {formatModifiedAgo(t.modifiedAt)}
              </span>
            </div>
          );
        })}
      </div>
      <div
        style={{
          display: 'flex',
          gap: 6,
          padding: '8px 10px',
          borderTop: '1px solid var(--line)',
          background: 'var(--bg-2)',
        }}
      >
        <button
          type="button"
          className="btn ghost sm"
          data-testid="thread-switcher-new"
          onClick={onNew}
          style={{ fontSize: 12 }}
        >
          + New thread
        </button>
      </div>
    </div>
  );
};

const ChatHeader = () => {
  const setChatVisible = useStore((s) => s.setChatVisible);
  const openSettings = useStore((s) => s.openSettings);
  const newThread = useStore((s) => s.newThread);
  const loadThread = useStore((s) => s.loadThread);
  const threads = useStore((s) => s.threads);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const liveMessages = useStore((s) => s.agent.messages);
  const [open, setOpen] = useState(false);

  const handleSelect = (id: string) => {
    loadThread(id);
    setOpen(false);
  };
  const handleNew = () => {
    newThread();
    setOpen(false);
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 14px',
        borderBottom: '1px solid var(--line)',
        background: 'var(--bg-2)',
        position: 'relative',
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
      <div style={{ position: 'relative' }}>
        <button
          type="button"
          data-testid="thread-switcher-toggle"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="Switch thread"
          onClick={() => setOpen((v) => !v)}
          style={{
            all: 'unset',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontFamily: 'var(--serif)',
            fontSize: 17,
            color: 'var(--text-0)',
            padding: '2px 4px',
            borderRadius: 4,
          }}
        >
          <span>Ask Evalyn</span>
          <span
            aria-hidden="true"
            className="mono text-3"
            style={{ fontSize: 11 }}
          >
            {open ? '▴' : '▾'}
          </span>
        </button>
        {open && (
          <ThreadSwitcherDropdown
            threads={threads}
            activeThreadId={activeThreadId}
            liveMessages={liveMessages}
            onSelect={handleSelect}
            onNew={handleNew}
            onClose={() => setOpen(false)}
          />
        )}
      </div>
      <span className="grow" />
      <button
        type="button"
        className="btn ghost icon"
        title="New conversation"
        aria-label="New conversation"
        onClick={() => newThread()}
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

/**
 * Non-blocking API-key banner. Visible only when no provider is active and
 * the user has not dismissed it for this session. Sits between the header
 * and the message list. Uses warn (amber) tones to signal "soft gate".
 */
const ApiKeyBanner = () => {
  const active = useStore((s) => s.settings.active);
  const dismissed = useStore((s) => s.chatBannerDismissed);
  const openSettings = useStore((s) => s.openSettings);
  const dismissChatBanner = useStore((s) => s.dismissChatBanner);
  if (active || dismissed) return null;
  return (
    <div
      role="status"
      data-testid="api-key-banner"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 14px',
        background: 'var(--warn-soft)',
        borderBottom: '1px solid var(--warn)',
        color: 'var(--warn)',
        fontSize: 12,
      }}
    >
      <span aria-hidden="true" className="mono" style={{ fontSize: 11 }}>
        ◐
      </span>
      <span style={{ flex: 1, minWidth: 0, color: 'var(--text-1)' }}>
        Add an LLM key to use the chat agent.
      </span>
      <button
        type="button"
        className="btn ghost sm"
        data-testid="api-key-banner-add"
        onClick={openSettings}
        style={{ color: 'var(--warn)' }}
      >
        Add key
      </button>
      <button
        type="button"
        className="btn ghost icon"
        aria-label="Dismiss"
        title="Dismiss for this session"
        data-testid="api-key-banner-dismiss"
        onClick={dismissChatBanner}
        style={{ width: 22, height: 22, fontSize: 12, color: 'var(--text-2)' }}
      >
        ✕
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
  const chatAttachments = useStore((s) => s.chatAttachments);
  const removeChatAttachment = useStore((s) => s.removeChatAttachment);
  const clearChatAttachments = useStore((s) => s.clearChatAttachments);
  const resetChat = useStore((s) => s.resetChat);
  const openSettings = useStore((s) => s.openSettings);
  const [value, setValue] = useState('');
  const disabled = status === 'streaming' || status === 'awaiting_confirmation';

  const onSend = () => {
    const trimmed = value.trim();
    // Allow send when at least one attachment is pending - empty body is
    // still a valid "ask about these refs" turn.
    if (!trimmed && chatAttachments.length === 0) return;

    // Slash-command interception: if the input starts with `/`, parse it
    // client-side. `kind: 'send'` rewrites the text before submission;
    // `kind: 'action'` triggers a side effect and skips the network turn.
    // Unrecognised slash inputs (parser returned null) fall through to the
    // normal send path so users can still ask `/why does X happen?` etc.
    if (trimmed.startsWith('/')) {
      const parsed = parseSlashCommand(trimmed);
      if (parsed?.kind === 'action') {
        switch (parsed.action) {
          case 'clear':
            resetChat();
            break;
          case 'settings':
            openSettings();
            break;
          case 'help':
            // Synthetic assistant turn - never sent over WS. Patch directly
            // into agent.messages so the existing scroll-to-bottom logic
            // picks it up. The id includes random bits to keep keys unique
            // when /help is used multiple times in a row.
            useStore.setState((s) => ({
              agent: {
                ...s.agent,
                messages: [
                  ...s.agent.messages,
                  {
                    id: `slash-help-${Date.now().toString(36)}-${Math.random()
                      .toString(36)
                      .slice(2, 6)}`,
                    role: 'assistant',
                    text: SLASH_HELP_TEXT,
                    ts: new Date().toISOString(),
                  },
                ],
              },
            }));
            break;
        }
        setValue('');
        // Do NOT clear chatAttachments here - the user's pinned context
        // chips persist across action commands.
        return;
      }
      if (parsed?.kind === 'send') {
        const prefix = serializeChatAttachments(chatAttachments, parsed.text);
        const payload = `${prefix}${parsed.text}`;
        setValue('');
        clearChatAttachments();
        void sendChatMessage(payload);
        return;
      }
      // null -> fall through to normal send below.
    }

    const prefix = serializeChatAttachments(chatAttachments, trimmed);
    const payload = `${prefix}${trimmed}`;
    setValue('');
    clearChatAttachments();
    void sendChatMessage(payload);
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
      {chatAttachments.length > 0 && (
        <div
          data-testid="chat-attachments-row"
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            marginBottom: 8,
          }}
        >
          {chatAttachments.map((a) => (
            <AttachChip
              key={a.id}
              attachment={a}
              onRemove={() => removeChatAttachment(a.id)}
            />
          ))}
        </div>
      )}
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
          placeholder="Message Evalyn..."
          disabled={disabled}
        />
        <div style={{ display: 'flex', alignItems: 'center', marginTop: 6, gap: 6 }}>
          {/* Rough token estimate: ~4 chars/token. `Array.from` counts unicode
              codepoints so emoji/CJK contribute one char each rather than the
              2-4 surrogate halves `string.length` would yield. */}
          {(() => {
            const tokens = Math.ceil(Array.from(value).length / 4);
            const overBudget = tokens > 100_000;
            return (
              <span
                className={`mono ${overBudget ? 'fail' : 'text-3'}`}
                data-testid="composer-token-count"
                style={{ fontSize: 10 }}
                title="Rough estimate: ~4 chars per token"
              >
                ~{tokens.toLocaleString()} tokens
              </span>
            );
          })()}
          <span className="grow" />
          <span className="text-3 mono" style={{ fontSize: 10 }}>
            ↵ send
          </span>
          <button
            type="button"
            className="btn primary icon"
            aria-label="Send message"
            style={{ width: 30, height: 30, fontSize: 14 }}
            onClick={onSend}
            disabled={
              disabled || (!value.trim() && chatAttachments.length === 0)
            }
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
};

/* ---------------------- empty-state suggestion chips ---------------------- */

const EmptyStateSuggestions = () => {
  const runs = useStore((s) => s.runs);
  const sendChatMessage = useStore((s) => s.sendChatMessage);
  const chips = useMemo(() => chatEmptyStateSuggestions(runs), [runs]);

  return (
    <div data-testid="chat-empty-state" style={{ padding: '24px 0' }}>
      <div
        style={{
          fontFamily: 'var(--serif)',
          fontSize: 18,
          lineHeight: 1.45,
          color: 'var(--text-0)',
        }}
      >
        Hi. I can read your traces, run evals, and explain results.
      </div>

      {chips.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            marginTop: 16,
          }}
        >
          {chips.map((chip) => (
            <button
              key={chip}
              type="button"
              data-testid={`chip-${chip}`}
              onClick={() => void sendChatMessage(chip)}
              style={{
                all: 'unset',
                cursor: 'pointer',
                display: 'block',
                padding: '8px 12px',
                background: 'var(--bg-2)',
                border: '1px solid var(--line)',
                borderRadius: 6,
                color: 'var(--text-1)',
                fontSize: 13,
                lineHeight: 1.4,
                transition: 'background var(--dur-base) var(--ease-soft)',
              }}
            >
              {chip}
            </button>
          ))}
        </div>
      )}

      <div
        className="text-3"
        style={{ fontSize: 12, lineHeight: 1.6, marginTop: 14 }}
      >
        I will always ask before running anything that writes data.
      </div>
    </div>
  );
};

/* ------------------------------ panel ------------------------------- */

const ChatPanel = () => {
  const messages = useStore((s) => s.agent.messages);
  const status = useStore((s) => s.agent.status);
  const sendChatMessage = useStore((s) => s.sendChatMessage);
  const branchFromMessage = useStore((s) => s.branchFromMessage);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  // "Pinned" means the user is reading at/near the bottom; we should keep
  // forcing scrollTop to the latest content. When scrolled up, we stop
  // jerking the view and instead show a "↓ N new" pill.
  const [pinned, setPinned] = useState(true);
  const [unread, setUnread] = useState(0);

  // Track whether the user is pinned to the bottom. Threshold from spec: 80px.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - (el.scrollTop + el.clientHeight);
    const isPinned = distance < 80;
    setPinned(isPinned);
    if (isPinned) setUnread(0);
  };

  // Auto-scroll only when pinned. When unpinned, count "new" arrivals so we
  // can render the pill. Using messages.length as a proxy is good enough:
  // streaming text deltas mutate an existing message (length unchanged) so
  // they don't bump the counter mid-stream.
  const lastMessageCount = useRef(messages.length);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (pinned) {
      el.scrollTop = el.scrollHeight;
      setUnread(0);
    } else if (messages.length > lastMessageCount.current) {
      setUnread((n) => n + (messages.length - lastMessageCount.current));
    }
    lastMessageCount.current = messages.length;
  }, [messages, status, pinned]);

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setPinned(true);
    setUnread(0);
  };

  // Retry-from-here: find the user message immediately preceding `idx` and
  // re-send its text. Skip if no such message exists.
  const retryFrom = (assistantIdx: number) => {
    for (let i = assistantIdx - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === 'user' && m.text) {
        void sendChatMessage(m.text);
        return;
      }
    }
  };

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
      <ApiKeyBanner />
      <ErrorBanner />
      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <div
          ref={scrollRef}
          onScroll={onScroll}
          style={{ height: '100%', overflowY: 'auto', padding: '16px 18px' }}
        >
          {messages.length === 0 && <EmptyStateSuggestions />}
          {messages.map((m, idx) => (
            <ChatTurn
              key={m.id}
              message={m}
              onRetry={m.role === 'assistant' ? () => retryFrom(idx) : undefined}
              onBranch={
                m.role === 'assistant' ? () => branchFromMessage(m.id) : undefined
              }
            />
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
              <span>Thinking...</span>
            </div>
          )}
        </div>

        {!pinned && unread > 0 && (
          <button
            type="button"
            data-testid="scroll-to-bottom"
            onClick={scrollToBottom}
            style={{
              position: 'absolute',
              right: 16,
              bottom: 12,
              padding: '4px 10px',
              borderRadius: 999,
              background: 'var(--accent)',
              color: '#1a1305',
              border: 'none',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: 'var(--shadow-soft)',
            }}
          >
            ↓ {unread} new
          </button>
        )}
      </div>
      <ChatComposer />
    </aside>
  );
};

export default ChatPanel;
