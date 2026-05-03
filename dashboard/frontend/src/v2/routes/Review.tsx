/**
 * Review - human review queue. P/F/S keyboard shortcuts advance through the queue.
 * Wires v2.reviewQueue() and v2.submitVerdict() into the design from screens-3.jsx.
 */

import { useCallback, useEffect, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill } from '../ui';
import { v2 } from '../api/client';
import type { ReviewItem, ReviewQueue } from '../api/types';
import { E } from '../tokens';

type Verdict = 'pass' | 'fail' | 'skip';

function pillColor(kind: 'pass' | 'fail' | 'warn'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  return E.warn;
}

function highlightedText(text: string, highlights: string[]) {
  if (highlights.length === 0) return text;
  const parts: { text: string; mark: boolean }[] = [{ text, mark: false }];
  for (const h of highlights) {
    if (!h) continue;
    const next: { text: string; mark: boolean }[] = [];
    for (const p of parts) {
      if (p.mark) {
        next.push(p);
        continue;
      }
      const segs = p.text.split(h);
      for (let i = 0; i < segs.length; i++) {
        if (segs[i]) next.push({ text: segs[i], mark: false });
        if (i < segs.length - 1) next.push({ text: h, mark: true });
      }
    }
    parts.length = 0;
    parts.push(...next);
  }
  return parts.map((p, i) =>
    p.mark ? (
      <span
        key={i}
        style={{
          background: E.failDim,
          color: E.fail,
          padding: '1px 4px',
          borderRadius: 3,
        }}
      >
        {p.text}
      </span>
    ) : (
      <span key={i}>{p.text}</span>
    ),
  );
}

export default function Review() {
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    v2.reviewQueue()
      .then(setQueue)
      .catch((e: unknown) => setErr(String(e)));
  }, []);

  const items = queue?.items ?? [];
  const current: ReviewItem | null = items[idx] ?? null;

  const submit = useCallback(
    (verdict: Verdict) => {
      if (!current || submitting) return;
      setSubmitting(true);
      v2.submitVerdict({
        item_id: current.item_id,
        source_run_id: current.source_run_id,
        verdict,
        note: note.trim() ? note.trim() : null,
      })
        .then(() => {
          setNote('');
          setIdx((i) => i + 1);
        })
        .catch((e: unknown) => setErr(String(e)))
        .finally(() => setSubmitting(false));
    },
    [current, note, submitting],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')) return;
      const k = e.key.toLowerCase();
      if (k === 'p') submit('pass');
      else if (k === 'f') submit('fail');
      else if (k === 's') submit('skip');
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [submit]);

  return (
    <AppShell contextChip={{ name: 'Customer Support Agent', version: 'v0.3' }}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <div>
            <Eyebrow>Where the judge wants a second opinion</Eyebrow>
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 32,
                fontWeight: 400,
                margin: 0,
                color: E.text0,
                letterSpacing: '-0.015em',
              }}
            >
              Human review
            </h1>
            <p style={{ fontSize: 13, color: E.text2, marginTop: 4 }}>
              {items.length} items pending - sorted by judge uncertainty - your reviews calibrate the judge
            </p>
          </div>
          <span style={{ flex: 1 }} />
          {items.length > 0 && (
            <div
              style={{
                display: 'flex',
                gap: 4,
                alignItems: 'center',
                fontSize: 11,
                color: E.text3,
                fontFamily: E.fMono,
              }}
            >
              <span>
                {Math.min(idx + 1, items.length)} / {items.length}
              </span>
              <Btn
                kind="ghost"
                size="sm"
                onClick={() => setIdx((i) => Math.max(0, i - 1))}
                disabled={idx === 0}
              >
                &lt;-
              </Btn>
              <Btn
                kind="ghost"
                size="sm"
                onClick={() => setIdx((i) => Math.min(items.length, i + 1))}
                disabled={idx >= items.length}
              >
                -&gt;
              </Btn>
            </div>
          )}
        </div>

        {err && (
          <Card style={{ padding: 16, marginTop: 18, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {!queue && !err && (
          <div style={{ marginTop: 20, color: E.text3, fontSize: 13 }}>Loading...</div>
        )}

        {queue && (items.length === 0 || idx >= items.length) && (
          <Card style={{ padding: 32, marginTop: 22, textAlign: 'center' }}>
            <div
              style={{
                fontFamily: E.fSerif,
                fontSize: 22,
                color: E.text0,
                marginBottom: 8,
              }}
            >
              Inbox zero
            </div>
            <div style={{ fontSize: 13, color: E.text2 }}>
              The judge is confident on every recent item.
            </div>
          </Card>
        )}

        {queue && current && (
          <>
            <div
              style={{
                marginTop: 18,
                height: 4,
                background: E.panel2,
                borderRadius: 2,
                overflow: 'hidden',
                display: 'flex',
              }}
            >
              <div
                style={{
                  width: `${(idx / Math.max(1, items.length)) * 100}%`,
                  background: E.ember,
                }}
              />
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 320px',
                gap: 14,
                marginTop: 18,
              }}
            >
              <Card style={{ padding: 24 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    marginBottom: 18,
                  }}
                >
                  <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                    {current.item_id} - {current.category}
                  </Pill>
                  <Pill mono color={E.warn} bg={E.warnDim}>
                    judge uncertain - {current.judge_confidence.toFixed(2)}
                  </Pill>
                  <span style={{ flex: 1 }} />
                  <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
                    from {current.source_run_label}
                  </span>
                </div>

                <Eyebrow>User</Eyebrow>
                <div
                  style={{
                    marginTop: 6,
                    padding: 14,
                    background: E.panel2,
                    borderRadius: 8,
                    fontSize: 14,
                    color: E.text1,
                    lineHeight: 1.5,
                  }}
                >
                  {current.user_text}
                </div>

                <Eyebrow style={{ marginTop: 18 }}>Agent response</Eyebrow>
                <div
                  style={{
                    marginTop: 6,
                    padding: 14,
                    background: E.panel2,
                    borderRadius: 8,
                    fontSize: 14,
                    color: E.text1,
                    lineHeight: 1.55,
                    border: `1px solid ${E.hair}`,
                  }}
                >
                  {highlightedText(current.agent_response, current.highlights)}
                </div>

                <Eyebrow style={{ marginTop: 18 }}>Expected</Eyebrow>
                <div
                  style={{
                    marginTop: 6,
                    padding: 14,
                    background: E.passDim,
                    border: `1px solid ${E.pass}33`,
                    borderRadius: 8,
                    fontSize: 13,
                    color: E.text1,
                    lineHeight: 1.55,
                  }}
                >
                  {current.expected}
                </div>

                <div
                  style={{
                    marginTop: 22,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                  }}
                >
                  <Eyebrow>Your verdict</Eyebrow>
                  <span style={{ flex: 1 }} />
                  <span style={{ fontSize: 11, color: E.text3 }}>shortcut keys</span>
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={() => submit('pass')}
                    style={{
                      flex: 1,
                      padding: '14px',
                      background: E.passDim,
                      border: `1px solid ${E.pass}33`,
                      color: E.pass,
                      borderRadius: 8,
                      fontSize: 13,
                      fontWeight: 500,
                      cursor: submitting ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2,
                      opacity: submitting ? 0.6 : 1,
                    }}
                  >
                    <span>Pass</span>
                    <span style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>P</span>
                  </button>
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={() => submit('fail')}
                    style={{
                      flex: 2,
                      padding: '14px',
                      background: E.failDim,
                      border: `1px solid ${E.fail}55`,
                      color: E.fail,
                      borderRadius: 8,
                      fontSize: 13,
                      fontWeight: 500,
                      cursor: submitting ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2,
                      opacity: submitting ? 0.6 : 1,
                    }}
                  >
                    <span>Fail</span>
                    <span style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>F</span>
                  </button>
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={() => submit('skip')}
                    style={{
                      flex: 1,
                      padding: '14px',
                      background: E.panel2,
                      border: `1px solid ${E.hair2}`,
                      color: E.text2,
                      borderRadius: 8,
                      fontSize: 13,
                      fontWeight: 500,
                      cursor: submitting ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2,
                      opacity: submitting ? 0.6 : 1,
                    }}
                  >
                    <span>Skip</span>
                    <span style={{ fontSize: 10, fontFamily: E.fMono, opacity: 0.6 }}>S</span>
                  </button>
                </div>

                <div style={{ marginTop: 14 }}>
                  <Eyebrow>Note for the team (optional)</Eyebrow>
                  <textarea
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="e.g. add policy.md grounding for tier-related questions..."
                    style={{
                      width: '100%',
                      marginTop: 6,
                      padding: 10,
                      background: E.panel2,
                      border: `1px solid ${E.hair2}`,
                      borderRadius: 6,
                      color: E.text1,
                      fontSize: 12.5,
                      fontFamily: E.fSans,
                      resize: 'vertical',
                      minHeight: 60,
                      outline: 'none',
                    }}
                  />
                </div>
              </Card>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <Card style={{ padding: 16 }}>
                  <Eyebrow>Judge's reasoning</Eyebrow>
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: E.text2,
                      lineHeight: 1.55,
                    }}
                  >
                    {current.judge_reasoning}
                  </div>
                  <div
                    style={{
                      marginTop: 12,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 5,
                      fontSize: 11,
                      fontFamily: E.fMono,
                    }}
                  >
                    {current.judge_breakdown.map((b) => (
                      <div key={b.label} style={{ display: 'flex' }}>
                        <span style={{ flex: 1, color: E.text3 }}>{b.label}</span>
                        <span style={{ color: pillColor(b.kind) }}>{b.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card style={{ padding: 16 }}>
                  <Eyebrow>Reviewer panel</Eyebrow>
                  <div
                    style={{
                      marginTop: 10,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                    }}
                  >
                    {queue.reviewers.map((r) => (
                      <div
                        key={r.name}
                        style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                      >
                        <div
                          style={{
                            width: 22,
                            height: 22,
                            borderRadius: '50%',
                            background: r.you ? E.ember : E.panel3,
                            color: r.you ? E.emberInk : E.text1,
                            fontSize: 10,
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                          }}
                        >
                          {r.name[0]}
                        </div>
                        <span style={{ fontSize: 12, color: E.text1, flex: 1 }}>{r.name}</span>
                        <span
                          style={{ fontSize: 10, fontFamily: E.fMono, color: E.text3 }}
                        >
                          {r.done}/{r.total}
                        </span>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card style={{ padding: 16 }}>
                  <Eyebrow>Why this batch?</Eyebrow>
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: E.text2,
                      lineHeight: 1.55,
                    }}
                  >
                    {queue.rationale}
                  </div>
                  <Btn kind="bare" size="sm" style={{ marginTop: 8 }}>
                    How sampling works -&gt;
                  </Btn>
                </Card>
              </div>
            </div>
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
