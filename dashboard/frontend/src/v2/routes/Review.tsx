/**
 * Review - human review queue. P/F/S keyboard shortcuts advance through the queue.
 * Wires v2.reviewQueue() and v2.submitVerdict() into the design from screens-3.jsx.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Glossary, Pill, Skeleton, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import type { CalibrationSuggestion, ReviewItem, ReviewQueue } from '../api/types';
import { listCli, type CliSchema } from '../api/cli';
import { useV2Resource } from '../hooks/useV2Resource';
import { useRouteTour } from '../tour/useRouteTour';
import { REVIEW_FAILURES_TOUR_ID } from '../tour/scripts/reviewFailures';
import { useProject } from '../hooks/useProject';
import { openCliRunner } from '../cliRunnerBridge';
import { E } from '../tokens';

type Verdict = 'pass' | 'fail' | 'skip';

/**
 * Transient feedback shown after a successful verdict submit.
 *
 * - `progress`: a verdict was added on a metric that is still below
 *   threshold. Shows "+1 verdict on <metric> (X of N)".
 * - `ready`: the verdict pushed the metric to/over the calibration
 *   threshold. Shows a primary CTA that opens CliRunner pre-filled.
 *
 * Lives at the top of the page and auto-dismisses after 4s.
 */
interface FeedbackChipState {
  kind: 'progress' | 'ready';
  metric: string;
  count: number;
  threshold: number;
  /** The matching suggestion - kept around so the "ready" CTA can deep-link. */
  suggestion: CalibrationSuggestion | null;
}

const FEEDBACK_DISMISS_MS = 4000;

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
  const project = useProject();
  const {
    data: queue,
    err: queueErr,
    reloading,
    isInitialLoad,
    refetch: refetchQueue,
  } = useV2Resource<ReviewQueue>('reviewQueue', v2.reviewQueue);
  useRouteTour(REVIEW_FAILURES_TOUR_ID, !!(queue && !queueErr));
  // Pull the CLI catalog so the calibration suggestion banner can open
  // the runner pre-filled. Cached at the module level by useV2Resource;
  // every other consumer that hits the catalog reuses this data.
  const { data: cmds } = useV2Resource<CliSchema[]>('commands', listCli);
  const [submitErr, setSubmitErr] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [lastChip, setLastChip] = useState<FeedbackChipState | null>(null);
  const dismissTimerRef = useRef<number | null>(null);

  const openCalibrate = useCallback(
    (s: CalibrationSuggestion) => {
      const cmd = cmds?.find((c) => c.id === 'calibrate');
      if (!cmd) {
        window.alert(
          'Cannot open calibrate: the command is not in this build of the CLI catalog.',
        );
        return;
      }
      // Drop any keys the schema doesn't declare so the form doesn't
      // surface ghost fields. The introspector normalises CLI flags to
      // snake_case, matching the backend's `cli_args` keys directly.
      const paramNames = new Set(cmd.params.map((p) => p.name));
      const initialValues: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(s.cli_args)) {
        if (paramNames.has(key)) initialValues[key] = value;
      }
      openCliRunner(cmd, { initialValues });
    },
    [cmds],
  );

  // Submit errors take priority since they're the most recent user action.
  const err = submitErr ?? queueErr;

  const items = queue?.items ?? [];
  const current: ReviewItem | null = items[idx] ?? null;

  const scheduleChipDismiss = useCallback(() => {
    if (dismissTimerRef.current != null) {
      window.clearTimeout(dismissTimerRef.current);
    }
    dismissTimerRef.current = window.setTimeout(() => {
      setLastChip(null);
      dismissTimerRef.current = null;
    }, FEEDBACK_DISMISS_MS);
  }, []);

  const submit = useCallback(
    (verdict: Verdict) => {
      if (!current || submitting) return;
      // The metric_id isn't on ReviewItem directly - it's the label of the
      // first judge_breakdown entry (see backend _row_for_metric).
      const itemMetric = current.judge_breakdown[0]?.label ?? null;
      // Snapshot the suggestion for this metric BEFORE submit so we can
      // tell whether verdict_count actually moved.
      const beforeMatch =
        queue?.calibration_suggestions.find((s) => s.metric_id === itemMetric) ?? null;

      setSubmitting(true);
      v2.submitVerdict({
        item_id: current.item_id,
        source_run_id: current.source_run_id,
        verdict,
        note: note.trim() ? note.trim() : null,
      })
        .then(async () => {
          setNote('');
          setSubmitErr(null);
          setIdx((i) => i + 1);

          if (!itemMetric) return;

          // Refetch the queue so the suggestions banner + counts update,
          // then directly fetch a fresh snapshot to evaluate the post-state
          // synchronously (the hook's `data` only updates on the next
          // render). The `refetch` call de-dupes via the inflight map.
          const [, fresh] = await Promise.all([
            refetchQueue(),
            v2.reviewQueue().catch(() => null),
          ]);
          if (!fresh) return;

          const afterMatch =
            fresh.calibration_suggestions.find((s) => s.metric_id === itemMetric) ?? null;
          if (!afterMatch) return;

          const beforeCount = beforeMatch?.verdict_count ?? 0;
          if (afterMatch.verdict_count <= beforeCount) return;

          const ready = afterMatch.verdict_count >= afterMatch.threshold;
          setLastChip({
            kind: ready ? 'ready' : 'progress',
            metric: itemMetric,
            count: afterMatch.verdict_count,
            threshold: afterMatch.threshold,
            suggestion: afterMatch,
          });
          scheduleChipDismiss();
        })
        .catch((e: unknown) => setSubmitErr(String(e)))
        .finally(() => setSubmitting(false));
    },
    [current, note, submitting, queue, refetchQueue, scheduleChipDismiss],
  );

  // Clear any pending dismiss timer on unmount.
  useEffect(() => {
    return () => {
      if (dismissTimerRef.current != null) {
        window.clearTimeout(dismissTimerRef.current);
      }
    };
  }, []);

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
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eyebrow>Where the judge wants a second opinion</Eyebrow>
              <UpdatingChip visible={reloading && !isInitialLoad} />
            </div>
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
              {items.length} items pending - sorted by{' '}
              <Glossary term="A judge is the rubric implementation that grades each item - usually an LLM, sometimes programmatic.">
                judge
              </Glossary>{' '}
              uncertainty - your reviews{' '}
              <Glossary term="Calibration compares judge verdicts to human verdicts to validate the rubric.">
                calibrate
              </Glossary>{' '}
              the judge
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

        {lastChip && (
          <div
            style={{
              marginTop: 14,
              animation: 'eFadeIn 180ms ease',
            }}
          >
            {lastChip.kind === 'progress' ? (
              <Pill
                mono
                color={E.text1}
                bg={E.passDim}
                style={{
                  fontSize: 11.5,
                  padding: '6px 12px',
                  border: `1px solid ${E.pass}33`,
                }}
              >
                + 1 verdict on{' '}
                <span style={{ color: E.text0, marginLeft: 4, marginRight: 4 }}>
                  {lastChip.metric}
                </span>
                <span style={{ color: E.text3, marginLeft: 4 }}>
                  ({lastChip.count} of {lastChip.threshold} - calibration unlocks at{' '}
                  {lastChip.threshold})
                </span>
              </Pill>
            ) : (
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 14px',
                  borderRadius: 999,
                  background: E.emberDim,
                  border: `1px solid ${E.ember}55`,
                  color: E.text0,
                  fontSize: 12,
                }}
              >
                <span style={{ fontFamily: E.fMono, color: E.ember }}>
                  Calibration ready for{' '}
                  <span style={{ color: E.text0 }}>{lastChip.metric}</span>
                </span>
                <Btn
                  kind="primary"
                  size="sm"
                  onClick={() => {
                    if (lastChip.suggestion) openCalibrate(lastChip.suggestion);
                  }}
                  disabled={!cmds || !lastChip.suggestion}
                  title={
                    cmds
                      ? `Open the calibrate form pre-filled for ${lastChip.metric}`
                      : 'Loading CLI catalog...'
                  }
                >
                  Run it now -&gt;
                </Btn>
              </div>
            )}
          </div>
        )}

        {queue?.calibration_suggestions && queue.calibration_suggestions.length > 0 && (
          <div
            style={{
              marginTop: 18,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            {queue.calibration_suggestions.map((s) => (
              <Card
                key={`${s.dataset}-${s.metric_id}`}
                accent
                style={{ padding: 16 }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 14,
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <Eyebrow style={{ color: E.ember }}>Calibration ready</Eyebrow>
                    <div
                      style={{
                        marginTop: 4,
                        fontSize: 13,
                        color: E.text1,
                        lineHeight: 1.5,
                      }}
                    >
                      {s.verdict_count} verdicts collected on{' '}
                      <span style={{ fontFamily: E.fMono, color: E.text0 }}>
                        {s.metric_id}
                      </span>{' '}
                      <span style={{ color: E.text3 }}>
                        (threshold: {s.threshold}, dataset: {s.dataset})
                      </span>
                      .
                    </div>
                  </div>
                  <Btn
                    kind="primary"
                    size="md"
                    onClick={() => openCalibrate(s)}
                    disabled={!cmds}
                    title={
                      cmds
                        ? `Open the calibrate form pre-filled for ${s.metric_id}`
                        : 'Loading CLI catalog...'
                    }
                  >
                    Run calibrate -&gt;
                  </Btn>
                </div>
              </Card>
            ))}
          </div>
        )}

        {err && (
          <Card style={{ padding: 16, marginTop: 18, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {!queue && !err && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 320px',
              gap: 14,
              marginTop: 18,
            }}
          >
            <Card style={{ padding: 24 }}>
              <Skeleton w={180} h={14} />
              <div style={{ marginTop: 18 }}>
                <Skeleton w={80} h={11} />
                <div style={{ marginTop: 6 }}>
                  <Skeleton w="100%" h={60} style={{ borderRadius: 8 }} />
                </div>
              </div>
              <div style={{ marginTop: 18 }}>
                <Skeleton w={120} h={11} />
                <div style={{ marginTop: 6 }}>
                  <Skeleton w="100%" h={80} style={{ borderRadius: 8 }} />
                </div>
              </div>
              <div style={{ marginTop: 18 }}>
                <Skeleton w={80} h={11} />
                <div style={{ marginTop: 6 }}>
                  <Skeleton w="100%" h={50} style={{ borderRadius: 8 }} />
                </div>
              </div>
            </Card>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[0, 1, 2].map((i) => (
                <Card key={i} style={{ padding: 16 }}>
                  <Skeleton w={120} h={11} />
                  <div style={{ marginTop: 10 }}>
                    <Skeleton w="100%" h={11} />
                  </div>
                  <div style={{ marginTop: 6 }}>
                    <Skeleton w="80%" h={11} />
                  </div>
                </Card>
              ))}
            </div>
          </div>
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
              data-coachmark="review-queue"
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 320px',
                gap: 14,
                marginTop: 18,
              }}
            >
              <Card style={{ padding: 24 }} data-coachmark="review-item">
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
                <div data-coachmark="review-verdict-buttons" style={{ marginTop: 8, display: 'flex', gap: 8 }}>
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
                    placeholder="Optional - leave a note for the team about this verdict"
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
                  <Eyebrow>
                    <Glossary term="A judge is the rubric implementation that grades each item - usually an LLM, sometimes programmatic.">
                      Judge's
                    </Glossary>{' '}
                    reasoning
                  </Eyebrow>
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
                  <Btn
                    kind="bare"
                    size="sm"
                    style={{ marginTop: 8 }}
                    disabled
                    title="Coming soon - a glossary entry on uncertainty sampling and reviewer rotation"
                  >
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
