/**
 * Welcome - the empty-state hero card shown when the workspace has no runs yet.
 *
 * Renders a single full-width card with headline, subtitle, three numbered
 * guidance items, and two primary actions (load demo, open co-pilot). The
 * "Load demo" button calls POST /api/demo/load and reloads on success so the
 * populated UI renders from the freshly seeded data.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Btn, Card, Eyebrow } from './index';
import { loadDemo } from '../api/demo';
import { E } from '../tokens';

export function Welcome() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onLoadDemo = async () => {
    setErr(null);
    setLoading(true);
    try {
      await loadDemo();
      window.location.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  };

  return (
    <Card style={{ padding: 28 }}>
      <Eyebrow>Get started</Eyebrow>
      <h2
        style={{
          fontFamily: E.fSerif,
          fontSize: 28,
          fontWeight: 400,
          margin: '6px 0 0',
          color: E.text0,
          letterSpacing: '-0.012em',
          lineHeight: 1.1,
        }}
      >
        Welcome to evalyn
      </h2>
      <p
        style={{
          fontSize: 14,
          color: E.text2,
          marginTop: 10,
          lineHeight: 1.6,
          maxWidth: 640,
        }}
      >
        A quality dashboard for your AI agent. Track pass rates, debug regressions,
        calibrate judges - all from your real evaluation runs.
      </p>

      <ol
        style={{
          margin: '18px 0 0',
          padding: 0,
          listStyle: 'none',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          maxWidth: 640,
        }}
      >
        {[
          'Load demo data to explore.',
          <>
            Or run{' '}
            <code style={{ fontFamily: E.fMono, fontSize: 12.5, color: E.text1 }}>
              evalyn run-eval
            </code>{' '}
            from your terminal.
          </>,
          'The right-side co-pilot can answer questions about your runs.',
        ].map((item, i) => (
          <li
            key={i}
            style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: 10,
              fontSize: 13,
              color: E.text1,
              lineHeight: 1.55,
            }}
          >
            <span
              style={{
                fontFamily: E.fMono,
                fontSize: 11,
                color: E.text3,
                minWidth: 16,
              }}
            >
              {i + 1}.
            </span>
            <span>{item}</span>
          </li>
        ))}
      </ol>

      <div style={{ display: 'flex', gap: 8, marginTop: 22, flexWrap: 'wrap' }}>
        <Btn kind="primary" size="md" onClick={onLoadDemo} disabled={loading}>
          {loading ? 'Loading...' : 'Load demo data'}
        </Btn>
        <Btn kind="secondary" size="md" onClick={() => navigate('/copilot')}>
          Open co-pilot
        </Btn>
      </div>

      {err && (
        <div
          style={{
            marginTop: 14,
            padding: '8px 12px',
            background: E.failDim,
            border: `1px solid ${E.fail}33`,
            borderRadius: 6,
            color: E.fail,
            fontSize: 12,
            fontFamily: E.fMono,
            wordBreak: 'break-word',
          }}
        >
          {err}
        </div>
      )}
    </Card>
  );
}
