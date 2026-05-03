/**
 * Co-pilot rendering atoms - Bubble, ToolBlock, PlanCard, EvidenceTable.
 * Ported from /tmp/evalyn-v2/screens-5.jsx.
 *
 * Pure render-only - all state lives in useCoPilotThread.
 */

import type { ReactNode } from 'react';
import { E } from '../tokens';
import { Btn, Card, Eyebrow, Pill } from '../ui';
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

export function ToolBlock({ entries }: { entries: ToolBlockEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div
      style={{
        marginTop: 10,
        background: E.panel2,
        border: `1px solid ${E.hair}`,
        borderRadius: 8,
        padding: 10,
        fontFamily: E.fMono,
        fontSize: 11.5,
      }}
    >
      {entries.map((e) => {
        const ok = e.status === 'complete';
        const err = e.status === 'error';
        const running = e.status === 'running' || e.status === 'proposed';
        const symbol = err ? '✗' : ok ? '✓' : '·';
        const symbolColor = err ? E.fail : ok ? E.pass : E.ember;
        return (
          <div
            key={e.tool_call_id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '3px 0',
              color: E.text2,
            }}
          >
            <span style={{ color: symbolColor }}>{symbol}</span>
            <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {e.cmd}
            </span>
            <span style={{ color: E.text3, fontSize: 10 }}>
              {running ? 'running' : e.duration_s != null ? `${e.duration_s.toFixed(1)}s` : ''}
            </span>
          </div>
        );
      })}
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
        <Btn kind="primary" size="sm" onClick={onApprove}>
          Approve & run →
        </Btn>
        <Btn kind="secondary" size="sm" onClick={onReject}>
          Reject
        </Btn>
      </div>
    </Card>
  );
}
