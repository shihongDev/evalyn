/**
 * Reports - weekly auto-generated summary.
 * Wires v2.weeklyReport() into the design from screens-4.jsx.
 */

import { useEffect, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill } from '../ui';
import { v2 } from '../api/client';
import type { WeeklyReport } from '../api/types';
import { E } from '../tokens';

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

export default function Reports() {
  const [data, setData] = useState<WeeklyReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    v2.weeklyReport()
      .then(setData)
      .catch((e: unknown) => setErr(String(e)));
  }, []);

  return (
    <AppShell contextChip={{ name: data?.project_name ?? 'Loading', version: '' }}>
      <div style={{ padding: '32px 36px', maxWidth: 1080 }}>
        <Eyebrow>Weekly summary - auto-generated</Eyebrow>
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
          {data ? `Week of ${data.week_label} - ${data.project_name}` : 'Loading...'}
        </h1>
        {data && (
          <p style={{ fontSize: 13, color: E.text2, marginTop: 6 }}>
            Drafted by co-pilot - {data.generated_at_iso} - review and send to{' '}
            <b>#agent-quality</b>
          </p>
        )}

        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          <Btn kind="primary" size="md" disabled title="Coming soon">
            Send to Slack
          </Btn>
          <Btn kind="secondary" size="md" disabled title="Coming soon">
            Export PDF
          </Btn>
          <Btn kind="ghost" size="md">Regenerate</Btn>
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
          <div style={{ marginTop: 22, color: E.text3, fontSize: 13 }}>Loading...</div>
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
                {data.tldr_md}
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
                      {data.blocking.body_md}{' '}
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
