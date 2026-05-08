/**
 * Co-pilot rendering atoms - Bubble, ToolBlock, PlanCard, EvidenceTable.
 * Ported from /tmp/evalyn-v2/screens-5.jsx.
 *
 * Pure render-only - all state lives in useCoPilotThread.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { E } from '../tokens';
import { Btn, Card, Eyebrow, Pill, Spinner } from '../ui';
import type { PendingConfirmation, ToolBlockEntry } from './types';

export function Bubble({ who, children }: { who: 'you' | 'agent'; children: ReactNode }) {
  const isYou = who === 'you';
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 22 }}>
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: 7,
          flexShrink: 0,
          background: isYou ? E.panel3 : `linear-gradient(135deg, ${E.ember}, #b8501f)`,
          border: isYou ? `1px solid ${E.hair2}` : 'none',
          color: isYou ? E.text2 : E.emberInk,
          fontSize: isYou ? 10 : 14,
          fontFamily: isYou ? E.fMono : E.fSerif,
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {isYou ? 'YOU' : 'e'}
      </div>
      <div
        style={{
          flex: 1,
          minWidth: 0,
          fontSize: 13.5,
          color: E.text1,
          lineHeight: 1.6,
          paddingTop: 4,
          whiteSpace: 'pre-wrap',
        }}
      >
        {children}
      </div>
    </div>
  );
}

/**
 * Inline "thinking" indicator for the conversation pane. Shown when the
 * agent has received a turn but hasn't streamed any text yet - that
 * empty beat between send and first chunk used to make the user wonder
 * whether the message went through. Three dots pulse via the existing
 * eDotPulse keyframe, staggered with `animation-delay` to produce the
 * typewriter "..." effect. Reduced-motion users get static dots.
 */
export function ThinkingBubble() {
  return (
    <Bubble who="agent">
      <span
        role="status"
        aria-label="Co-pilot is thinking"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          fontSize: 18,
          color: E.text3,
          lineHeight: 1,
        }}
      >
        <span aria-hidden="true" style={{ animation: 'eDotPulse 1.2s ease-in-out infinite', animationDelay: '0s' }}>·</span>
        <span aria-hidden="true" style={{ animation: 'eDotPulse 1.2s ease-in-out infinite', animationDelay: '0.2s' }}>·</span>
        <span aria-hidden="true" style={{ animation: 'eDotPulse 1.2s ease-in-out infinite', animationDelay: '0.4s' }}>·</span>
      </span>
    </Bubble>
  );
}

function statusGlyph(status: ToolBlockEntry['status']): { symbol: string; color: string } {
  if (status === 'error') return { symbol: '✗', color: E.fail };
  if (status === 'complete') return { symbol: '✓', color: E.pass };
  if (status === 'running') return { symbol: '◐', color: E.ember };
  return { symbol: '·', color: E.ember };
}

function ToolRow({
  entry,
  expanded,
  onToggle,
  isLast,
}: {
  entry: ToolBlockEntry;
  expanded: boolean;
  onToggle: () => void;
  isLast: boolean;
}) {
  const [showFull, setShowFull] = useState(false);
  const { symbol, color } = statusGlyph(entry.status);
  const isRunning = entry.status === 'running';
  const isAwaiting = entry.status === 'awaiting_confirmation';
  const isProposed = entry.status === 'proposed';
  const isError = entry.status === 'error';
  const argEntries = Object.entries(entry.args ?? {});
  const hasArgs = argEntries.length > 0;
  const hasOutput = entry.output_full.length > 0 || entry.output_preview.length > 0;
  const previewIsTruncated = entry.output_full.length > entry.output_preview.length;
  const outputToShow = showFull ? entry.output_full : entry.output_preview;
  return (
    <div style={{ borderBottom: isLast ? 'none' : `1px solid ${E.hair}`, padding: '4px 0' }}>
      <div
        onClick={onToggle}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onToggle();
          }
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          minHeight: 22,
          padding: '3px 0',
          color: E.text2,
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <span
          style={{
            color,
            width: 12,
            display: 'inline-flex',
            justifyContent: 'center',
            animation: isRunning ? 'eDotPulse 1.6s ease-in-out infinite' : 'none',
          }}
        >
          {symbol}
        </span>
        <span
          style={{
            color: E.text3,
            fontSize: 9,
            width: 10,
            display: 'inline-flex',
            justifyContent: 'center',
          }}
        >
          {expanded ? '▾' : '▸'}
        </span>
        <span
          style={{
            flex: 1,
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: E.text1,
          }}
        >
          {entry.tool || entry.cmd}
        </span>
        {isAwaiting && (
          <Pill mono color={E.warn} bg={E.warnDim} style={{ fontSize: 9.5 }}>
            confirm
          </Pill>
        )}
        {isRunning && (
          <Pill mono color={E.ember} bg={E.emberDim} style={{ fontSize: 9.5 }}>
            running
          </Pill>
        )}
        {isProposed && (
          <Pill mono color={E.text3} bg={E.panel3} style={{ fontSize: 9.5 }}>
            proposed
          </Pill>
        )}
        <span style={{ color: E.text3, fontSize: 10, minWidth: 32, textAlign: 'right' }}>
          {entry.duration_s != null ? `${entry.duration_s.toFixed(1)}s` : ''}
        </span>
      </div>
      {expanded && (
        <div style={{ paddingLeft: 20, paddingTop: 6, paddingBottom: 6 }}>
          <div
            style={{
              fontSize: 11,
              color: E.text2,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              marginBottom: hasArgs || hasOutput || isError ? 8 : 0,
            }}
          >
            {entry.cmd || entry.tool}
          </div>
          {hasArgs && (
            <div style={{ marginBottom: hasOutput || isError ? 8 : 0 }}>
              <Eyebrow style={{ marginBottom: 4 }}>Args</Eyebrow>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {argEntries.map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      display: 'flex',
                      gap: 8,
                      fontSize: 11,
                      color: E.text2,
                      lineHeight: 1.45,
                    }}
                  >
                    <span style={{ color: E.text3, minWidth: 70, flexShrink: 0 }}>{k}</span>
                    <span
                      style={{
                        flex: 1,
                        minWidth: 0,
                        wordBreak: 'break-word',
                        whiteSpace: 'pre-wrap',
                        color: E.text1,
                      }}
                    >
                      {typeof v === 'string' ? v : JSON.stringify(v)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {hasOutput && (
            <div style={{ marginBottom: isError ? 8 : 0 }}>
              <Eyebrow style={{ marginBottom: 4 }}>Output</Eyebrow>
              <pre
                style={{
                  margin: 0,
                  padding: '6px 8px',
                  background: E.panel,
                  border: `1px solid ${E.hair}`,
                  borderRadius: 6,
                  fontFamily: E.fMono,
                  fontSize: 10.5,
                  color: E.text2,
                  maxHeight: showFull ? 240 : 120,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  lineHeight: 1.45,
                }}
              >
                {outputToShow}
              </pre>
              {previewIsTruncated && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowFull((v) => !v);
                  }}
                  style={{
                    marginTop: 4,
                    padding: 0,
                    background: 'transparent',
                    border: 'none',
                    color: E.ember,
                    fontSize: 10.5,
                    fontFamily: E.fMono,
                    cursor: 'pointer',
                  }}
                >
                  {showFull ? 'Show less' : 'Show full'}
                </button>
              )}
            </div>
          )}
          {isError && (
            <div
              style={{
                fontSize: 11,
                color: E.fail,
                fontFamily: E.fMono,
              }}
            >
              Failed{entry.exit_code != null ? ` (exit code ${entry.exit_code})` : ''}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ToolBlock({ entries }: { entries: ToolBlockEntry[] }) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  // Drop expanded ids whose entries have disappeared (e.g. thread reset).
  useEffect(() => {
    setExpanded((prev) => {
      const ids = new Set(entries.map((e) => e.tool_call_id));
      let changed = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (ids.has(id)) next.add(id);
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [entries]);

  if (entries.length === 0) return null;
  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  return (
    <div
      style={{
        marginTop: 10,
        background: E.panel2,
        border: `1px solid ${E.hair}`,
        borderRadius: 8,
        padding: '4px 10px',
        fontFamily: E.fMono,
        fontSize: 11.5,
      }}
    >
      {entries.map((e, i) => (
        <ToolRow
          key={e.tool_call_id}
          entry={e}
          expanded={expanded.has(e.tool_call_id)}
          onToggle={() => toggle(e.tool_call_id)}
          isLast={i === entries.length - 1}
        />
      ))}
    </div>
  );
}

export function PlanCard({
  pending,
  onApprove,
  onReject,
}: {
  pending: PendingConfirmation;
  onApprove: () => void;
  onReject: () => void;
}) {
  // Internal busy state. Once the user clicks Approve or Reject we
  // disable both buttons until the parent flips `pending` away (on
  // resolve OR error), at which point this whole card unmounts so
  // the local state goes with it. Without this, a slow network +
  // double-click would fire confirmAgentTool twice; the second
  // call 4xxs because the tool_call_id is already consumed but
  // wastes a round trip and surfaces a confusing error.
  //
  // The parent's `confirm()` is fire-and-forgotten from the click
  // handler so we can't await it here - that's why we track the
  // click locally rather than reading an `inFlight` prop.
  const [clicked, setClicked] = useState<'approve' | 'reject' | null>(null);
  const busy = clicked !== null;
  return (
    <Card style={{ marginTop: 10, padding: 0, overflow: 'hidden', background: E.panel2 }}>
      <div
        style={{
          padding: '10px 14px',
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <Eyebrow>Confirm before run · {pending.tool}</Eyebrow>
        <span style={{ flex: 1 }} />
        <Pill mono color={E.warn} bg={E.warnDim} style={{ fontSize: 9.5 }}>
          write
        </Pill>
      </div>
      <div
        style={{
          padding: '10px 14px',
          fontFamily: E.fMono,
          fontSize: 11.5,
          color: E.text2,
          background: E.panel,
          borderBottom: `1px solid ${E.hair}`,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {pending.preview_cmd}
      </div>
      {pending.side_effects && pending.side_effects.length > 0 && (
        <div style={{ padding: '10px 14px', fontSize: 12, color: E.text2, lineHeight: 1.55 }}>
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 10,
              color: E.text3,
              letterSpacing: '0.06em',
              marginBottom: 6,
            }}
          >
            THIS WILL
          </div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {pending.side_effects.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      <div
        style={{
          padding: '10px 14px',
          borderTop: `1px solid ${E.hair}`,
          display: 'flex',
          gap: 6,
        }}
      >
        <Btn
          kind="primary"
          size="sm"
          disabled={busy}
          aria-busy={clicked === 'approve'}
          onClick={() => {
            if (busy) return;
            setClicked('approve');
            onApprove();
          }}
        >
          {clicked === 'approve' ? (
            <>
              <Spinner size={11} /> Approving
            </>
          ) : (
            'Approve & run →'
          )}
        </Btn>
        <Btn
          kind="secondary"
          size="sm"
          disabled={busy}
          aria-busy={clicked === 'reject'}
          onClick={() => {
            if (busy) return;
            setClicked('reject');
            onReject();
          }}
        >
          {clicked === 'reject' ? (
            <>
              <Spinner size={11} /> Rejecting
            </>
          ) : (
            'Reject'
          )}
        </Btn>
      </div>
    </Card>
  );
}
