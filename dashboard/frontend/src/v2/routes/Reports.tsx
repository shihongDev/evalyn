/**
 * Reports - weekly auto-generated summary.
 * Wires v2.weeklyReport() into the design from screens-4.jsx.
 */

import { useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, Spinner, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { useV2Resource } from '../hooks/useV2Resource';
import { E } from '../tokens';
import { linkifyText, makeUrlCounter } from '../textRender';
import { copyToClipboard } from '../clipboard';
import type { WeeklyReport } from '../api/types';

function deltaPillColor(kind: 'pass' | 'fail' | 'warn' | 'info'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  if (kind === 'warn') return E.warn;
  return E.info;
}

function deltaPillBg(kind: 'pass' | 'fail' | 'warn' | 'info'): string {
  if (kind === 'pass') return E.passDim;
  if (kind === 'fail') return E.failDim;
  if (kind === 'warn') return E.warnDim;
  return E.infoDim;
}

/** Render the report as markdown that pastes cleanly into Slack, email,
 * Notion, or any other tool. Format chosen to look reasonable in both
 * raw-text and Markdown-rendered surfaces. */
function reportToMarkdown(r: WeeklyReport): string {
  const lines: string[] = [];
  lines.push(`# ${r.project_name} - week of ${r.week_label}`);
  lines.push('');
  lines.push(`_Drafted by co-pilot - ${r.generated_at_iso}_`);
  lines.push('');
  lines.push('## TL;DR');
  lines.push(r.tldr_md.trim());
  lines.push('');
  if (r.big_numbers.length > 0) {
    lines.push('## Highlights');
    for (const n of r.big_numbers) {
      lines.push(`- **${n.label}:** ${n.value} (${n.delta}) - ${n.sub}`);
    }
    lines.push('');
  }
  if (r.shipped.length > 0) {
    lines.push('## What we shipped');
    for (const s of r.shipped) lines.push(`- ${s.text}`);
    lines.push('');
  }
  if (r.blocking) {
    lines.push('## What\'s blocking');
    lines.push(`**${r.blocking.title}**`);
    lines.push('');
    lines.push(r.blocking.body_md.trim());
    lines.push('');
    lines.push(`Owner: ${r.blocking.owner} - ETA: ${r.blocking.eta}`);
    lines.push('');
  }
  if (r.up_next.length > 0) {
    lines.push('## Up next');
    for (const s of r.up_next) lines.push(`- ${s.text}`);
    lines.push('');
  }
  return lines.join('\n');
}

export default function Reports() {
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'weeklyReport',
    v2.weeklyReport,
  );
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle');

  async function handleCopy() {
    if (!data) return;
    try {
      await copyToClipboard(reportToMarkdown(data));
      setCopyState('copied');
      window.setTimeout(() => setCopyState('idle'), 2000);
    } catch {
      setCopyState('error');
      window.setTimeout(() => setCopyState('idle'), 3000);
    }
  }

  return (
    <AppShell contextChip={{ name: data?.project_name ?? 'Loading', version: '' }}>
      <div style={{ padding: '32px 36px', maxWidth: 1080 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Eyebrow>Weekly summary - auto-generated</Eyebrow>
          <UpdatingChip
            visible={reloading && !isInitialLoad}
            error={data ? err : null}
            onRetry={refetch}
          />
        </div>
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 34,
            fontWeight: 400,
            margin: 0,
            color: E.text0,
            letterSpacing: '-0.015em',
          }}
        >
          {data ? `Week of ${data.week_label} - ${data.project_name}` : <Skeleton w={420} h={34} />}
        </h1>
        {data && (
          <p style={{ fontSize: 13, color: E.text2, marginTop: 6 }}>
            Drafted by co-pilot - {data.generated_at_iso}
          </p>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 16, flexWrap: 'wrap' }}>
          <Btn
            kind="primary"
            size="md"
            onClick={() => void handleCopy()}
            disabled={!data}
            title={
              !data
                ? 'Wait for the report to load'
                : 'Copy as markdown - paste into Slack, email, Notion, or anywhere else'
            }
          >
            {copyState === 'copied' ? 'Copied' : 'Copy report'}
          </Btn>
          <Btn
            kind="ghost"
            size="md"
            onClick={() => void refetch()}
            disabled={reloading}
            title={reloading ? 'Regenerating...' : "Re-fetch the weekly report - it's recomputed on each load"}
          >
            {reloading ? (
              <>
                <Spinner size={11} /> Regenerating
              </>
            ) : (
              'Regenerate'
            )}
          </Btn>
          {copyState === 'copied' && (
            <Pill mono color={E.pass} bg={E.passDim}>
              Markdown on clipboard
            </Pill>
          )}
          {copyState === 'error' && (
            <Pill mono color={E.fail} bg={E.failDim}>
              Copy failed - browser blocked clipboard
            </Pill>
          )}
        </div>

        {err && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading report</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {!data && !err && (
          <>
            <Card style={{ padding: 24, marginTop: 22 }}>
              <Eyebrow>TL;DR</Eyebrow>
              <div style={{ marginTop: 10 }}>
                <Skeleton w="100%" h={22} />
                <div style={{ marginTop: 8 }}>
                  <Skeleton w="80%" h={22} />
                </div>
              </div>
            </Card>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 14,
                marginTop: 14,
              }}
            >
              {[0, 1, 2].map((i) => (
                <Card key={i} style={{ padding: 18 }}>
                  <Skeleton w={120} h={11} />
                  <div style={{ marginTop: 10 }}>
                    <Skeleton w={100} h={34} />
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <Skeleton w={140} h={11} />
                  </div>
                </Card>
              ))}
            </div>
            <Card style={{ padding: 24, marginTop: 14 }}>
              <Skeleton w={200} h={22} />
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} w={`${65 + ((i * 11) % 25)}%`} h={13} />
                ))}
              </div>
            </Card>
          </>
        )}

        {data && (
          <>
            <Card style={{ padding: 24, marginTop: 22 }}>
              <Eyebrow>TL;DR</Eyebrow>
              <div
                style={{
                  marginTop: 8,
                  fontFamily: E.fSerif,
                  fontSize: 22,
                  color: E.text0,
                  lineHeight: 1.4,
                  fontWeight: 400,
                  letterSpacing: '-0.01em',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {linkifyText(data.tldr_md, makeUrlCounter())}
              </div>
            </Card>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 14,
                marginTop: 14,
              }}
            >
              {data.big_numbers.map((s) => (
                <Card key={s.label} style={{ padding: 18 }}>
                  <Eyebrow>{s.label}</Eyebrow>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'baseline',
                      gap: 10,
                      marginTop: 6,
                    }}
                  >
                    <span style={{ fontFamily: E.fSerif, fontSize: 34, color: E.text0 }}>
                      {s.value}
                    </span>
                    <Pill mono color={deltaPillColor(s.delta_kind)} bg={deltaPillBg(s.delta_kind)}>
                      {s.delta}
                    </Pill>
                  </div>
                  <div style={{ fontSize: 11.5, color: E.text3, marginTop: 4 }}>{s.sub}</div>
                </Card>
              ))}
            </div>

            <Card style={{ padding: 24, marginTop: 14 }}>
              <h2
                style={{
                  fontFamily: E.fSerif,
                  fontSize: 22,
                  fontWeight: 400,
                  color: E.text0,
                  margin: 0,
                }}
              >
                What we shipped
              </h2>
              <ul
                style={{
                  marginTop: 10,
                  paddingLeft: 18,
                  fontSize: 13.5,
                  color: E.text1,
                  lineHeight: 1.8,
                }}
              >
                {data.shipped.map((s, i) => (
                  <li key={i}>{s.text}</li>
                ))}
              </ul>

              {data.blocking && (
                <>
                  <h2
                    style={{
                      fontFamily: E.fSerif,
                      fontSize: 22,
                      fontWeight: 400,
                      color: E.text0,
                      margin: '22px 0 0',
                    }}
                  >
                    What's blocking
                  </h2>
                  <div
                    style={{
                      marginTop: 10,
                      padding: 14,
                      background: E.failDim,
                      border: `1px solid ${E.fail}33`,
                      borderRadius: 8,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        marginBottom: 6,
                      }}
                    >
                      <span
                        style={{
                          width: 4,
                          height: 14,
                          borderRadius: 2,
                          background: E.fail,
                        }}
                      />
                      <span style={{ fontSize: 13.5, color: E.text0, fontWeight: 500 }}>
                        {data.blocking.title}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: 12.5,
                        color: E.text2,
                        marginLeft: 12,
                        lineHeight: 1.55,
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      {linkifyText(data.blocking.body_md, makeUrlCounter())}{' '}
                      <b style={{ color: E.text0 }}>Owner:</b> {data.blocking.owner}.{' '}
                      <b style={{ color: E.text0 }}>ETA:</b> {data.blocking.eta}.
                    </div>
                  </div>
                </>
              )}

              <h2
                style={{
                  fontFamily: E.fSerif,
                  fontSize: 22,
                  fontWeight: 400,
                  color: E.text0,
                  margin: '22px 0 0',
                }}
              >
                Up next
              </h2>
              <ul
                style={{
                  marginTop: 10,
                  paddingLeft: 18,
                  fontSize: 13.5,
                  color: E.text1,
                  lineHeight: 1.8,
                }}
              >
                {data.up_next.map((s, i) => (
                  <li key={i}>{s.text}</li>
                ))}
              </ul>
            </Card>
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
