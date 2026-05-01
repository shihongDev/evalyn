/**
 * Welcome view — IDE-for-evaluations hero.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (lines 386-423).
 * Quick-action cards open CLI form tabs / run tabs via store actions; B2
 * later replaces the console.log stubs with real wiring.
 */

import { useStore } from '../store';

interface QuickAction {
  /** Eyebrow label. */
  t: string;
  /** Body line (e.g. "evalyn run-eval"). */
  b: string;
  /** Action label. */
  c: string;
  /** Click handler. */
  a: () => void;
}

const Welcome = () => {
  const openCli = useStore((s) => s.openCli);
  const openFile = useStore((s) => s.openFile);

  const actions: QuickAction[] = [
    {
      t: 'Latest run',
      b: '82dddcc3 · −4 pts',
      c: 'Open run',
      a: () => {
        console.log('welcome: open latest run (B2 stub)');
        openFile('82dddcc3.run');
      },
    },
    {
      t: 'Quick eval',
      b: 'evalyn run-eval',
      c: 'Open form',
      a: () => {
        console.log('welcome: open run-eval (B2 stub)');
        openCli('run-eval');
      },
    },
    {
      t: 'Calibrate judge',
      b: 'evalyn calibrate',
      c: 'Open form',
      a: () => {
        console.log('welcome: open calibrate (B2 stub)');
        openCli('calibrate');
      },
    },
    {
      t: 'Annotate',
      b: 'evalyn annotate',
      c: 'Open form',
      a: () => {
        console.log('welcome: open annotate (B2 stub)');
        openCli('annotate');
      },
    },
    {
      t: 'Build dataset',
      b: 'evalyn build-dataset',
      c: 'Open form',
      a: () => {
        console.log('welcome: open build-dataset (B2 stub)');
        openCli('build-dataset');
      },
    },
    {
      t: 'One-click',
      b: 'evalyn oneclick',
      c: 'Open form',
      a: () => {
        console.log('welcome: open oneclick (B2 stub)');
        openCli('oneclick');
      },
    },
  ];

  return (
    <div style={{ padding: '60px 80px', maxWidth: 920, margin: '0 auto' }}>
      <div
        className="text-3 mono"
        style={{
          fontSize: 11,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          marginBottom: 12,
        }}
      >
        localhost · workbench
      </div>
      <h1
        style={{
          fontFamily: 'var(--serif)',
          fontSize: 56,
          fontWeight: 400,
          lineHeight: 1.05,
          letterSpacing: '-0.02em',
          margin: 0,
          color: 'var(--text-0)',
        }}
      >
        An IDE for evaluations.
      </h1>
      <p
        style={{
          fontSize: 16,
          color: 'var(--text-2)',
          maxWidth: 620,
          lineHeight: 1.55,
          marginTop: 18,
        }}
      >
        Open a file or run a CLI. The agent on the right can translate questions into commands and
        chain them for you.
      </p>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: 12,
          marginTop: 32,
        }}
      >
        {actions.map((card) => (
          <div
            key={card.t}
            onClick={card.a}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') card.a();
            }}
            style={{
              padding: 18,
              background: 'var(--bg-2)',
              border: '1px solid var(--line)',
              borderRadius: 8,
              cursor: 'pointer',
            }}
          >
            <div
              className="text-3 mono"
              style={{
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}
            >
              {card.t}
            </div>
            <div className="mono" style={{ fontSize: 14, marginTop: 6, color: 'var(--text-0)' }}>
              {card.b}
            </div>
            <div className="accent mono" style={{ fontSize: 11, marginTop: 14 }}>
              {card.c} →
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          marginTop: 36,
          padding: 18,
          background: 'var(--bg-2)',
          border: '1px dashed var(--line-2)',
          borderRadius: 8,
        }}
      >
        <div className="row">
          <span className="text-2" style={{ fontSize: 12 }}>
            Try the agent: <i>"Why is gemini regressing?"</i>
          </span>
          <span className="grow" />
          <span className="kbd">⌘K</span>
          <span className="text-3" style={{ fontSize: 11, marginLeft: 4 }}>
            palette
          </span>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
