/**
 * CoPilotDock - the right-side 420px dock variant of the co-pilot.
 * Wired to /api/agent/chat + /ws/agent/{tid} via useCoPilotThread.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { E } from '../tokens';
import { Btn, Pill } from '../ui';
import { Bubble, PlanCard, ToolBlock } from './atoms';
import { useCoPilotThread } from './useCoPilotThread';

export function CoPilotDock({ onClose }: { onClose: () => void }) {
  const { messages, pending, send, confirm, status, threadId } = useCoPilotThread();
  const [draft, setDraft] = useState('');
  const navigate = useNavigate();

  const handleExpand = () => {
    navigate(threadId ? `/copilot/${threadId}` : '/copilot');
  };

  const submit = () => {
    const t = draft.trim();
    if (!t) return;
    setDraft('');
    void send(t);
  };

  return (
    <div
      style={{
        background: E.panel,
        borderLeft: `1px solid ${E.hair}`,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
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
        <span style={{ flex: 1 }} />
        <button
          type="button"
          onClick={handleExpand}
          title="Open full thread"
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
          ⛶
        </button>
        <button
          type="button"
          onClick={onClose}
          style={{
            color: E.text3,
            fontSize: 16,
            padding: 4,
            cursor: 'pointer',
            background: 'transparent',
            border: 'none',
          }}
        >
          ×
        </button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '18px 18px 8px' }}>
        {messages.length === 0 && (
          <div style={{ color: E.text3, fontSize: 12.5, padding: '20px 4px', lineHeight: 1.55 }}>
            Ask anything about your evals. The co-pilot can read runs, datasets, and rubrics on
            its own. Anything that writes will pause for your confirmation first.
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
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                submit();
              }
            }}
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
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
            <Pill mono style={{ fontSize: 10, padding: '2px 8px' }}>
              + this run
            </Pill>
            <Pill mono style={{ fontSize: 10, padding: '2px 8px' }}>
              ＠ dataset
            </Pill>
            <span style={{ flex: 1 }} />
            <Btn
              kind="primary"
              size="sm"
              onClick={submit}
              disabled={!draft.trim() || status === 'streaming' || pending != null}
              style={{ width: 26, height: 26, padding: 0, justifyContent: 'center' }}
            >
              ↑
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
          Read-only commands run automatically · writes ask first
        </div>
      </div>
    </div>
  );
}
