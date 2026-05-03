/**
 * Welcome view - landing page when no tab is open.
 *
 * Branches on workspace state: with no runs we offer onboarding paths;
 * with runs we surface the most recent one. No fake data is rendered.
 */

import { useState } from 'react';
import { api } from '../api';
import { useStore } from '../store';
import type { RunMeta } from '../types/jobs';

interface QuickAction {
  /** Eyebrow label. */
  title: string;
  /** Body line. */
  body: string;
  /** Action label. */
  cta: string;
  /** Click handler. */
  onAction: () => void;
}

const fmtPct = (v: number): string => `${(v * 100).toFixed(1)}%`;

const fmtDelta = (delta: number): string => {
  const pts = (delta * 100).toFixed(1);
  if (delta > 0) return `+${pts} pts`;
  if (delta < 0) return `${pts} pts`;
  return 'no change';
};

const Welcome = () => {
  const selectActiveCli = useStore((s) => s.selectActiveCli);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const runs = useStore((s) => s.runs);
  const loadRuns = useStore((s) => s.loadRuns);
  const loadFileTree = useStore((s) => s.loadFileTree);
  const sendChatMessage = useStore((s) => s.sendChatMessage);

  // Demo-load state. We keep these local since they only matter while
  // the user is on the Welcome view; once the demo is loaded the
  // workspace branches to a different render path.
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);

  const pick = (cliId: string) => {
    selectActiveCli(cliId);
    setActiveTab(null);
  };

  // Open a CLI form pre-seeded with args. Used by the latest-run card to
  // route the user into `analyze --run <id>` instead of the placeholder
  // run viewer (which is not yet implemented).
  const pickWithArgs = (cliId: string, args: Record<string, unknown>) => {
    useStore.setState({
      activeCliId: cliId,
      activeFormSeed: { cliId, args },
    });
    setActiveTab(null);
  };

  const loadDemo = async () => {
    if (demoLoading) return;
    setDemoLoading(true);
    setDemoError(null);
    try {
      await api.demoLoad();
      // Refresh both the runs list and the file tree so the rest of
      // the dashboard reflects the demo content immediately.
      await Promise.all([loadRuns(), loadFileTree()]);
      // Land on `analyze --dataset <demo-dataset-path>` so the user can
      // hit Run and see the loop close in one click. We seed the dataset
      // path (a real file on disk in the fixture) rather than --run
      // because the analyze CLI looks runs up via the SQLite traces
      // store, which the demo fixture does not seed. The dataset path
      // works against the demo's results.json on disk.
      pickWithArgs('analyze', { dataset: 'data/datasets/research-v1' });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // 409 means non-demo content already exists; surface friendly
      // copy instead of the raw HTTP body.
      setDemoError(
        msg.includes('409') ? 'Demo data already loaded.' : `Could not load demo: ${msg}`,
      );
    } finally {
      setDemoLoading(false);
    }
  };

  const hasRuns = runs.length > 0;
  const latest: RunMeta | null = hasRuns ? runs[0] : null;

  const latestCard: QuickAction = latest
    ? {
        title: 'Latest run',
        body: latest.delta != null
          ? `${latest.id} · ${fmtDelta(latest.delta)}`
          : `${latest.id} · pass ${fmtPct(latest.pass)}`,
        cta: 'Analyze',
        onAction: () => pickWithArgs('analyze', { run: latest.id }),
      }
    : {
        title: 'No runs yet',
        body: 'Run quickstart to capture your first traces.',
        cta: 'Open quickstart',
        onAction: () => pick('quickstart'),
      };

  const baseActions: QuickAction[] = [
    latestCard,
    { title: 'Quick eval', body: 'evalyn run-eval', cta: 'Open', onAction: () => pick('run-eval') },
    { title: 'Calibrate judge', body: 'evalyn calibrate', cta: 'Open', onAction: () => pick('calibrate') },
    { title: 'Annotate', body: 'evalyn annotate', cta: 'Open', onAction: () => pick('annotate') },
    { title: 'Build dataset', body: 'evalyn build-dataset', cta: 'Open', onAction: () => pick('build-dataset') },
    { title: 'One-click', body: 'evalyn one-click', cta: 'Open', onAction: () => pick('one-click') },
  ];

  // The "Try demo" card only shows when the workspace is empty so a
  // first-time user has a one-click path to a populated dashboard. Once
  // they have any runs, the card disappears (the demo is a starting
  // point, not a permanent fixture).
  const showDemoCard = !hasRuns;

  // Workspace-aware suggestion. Empty workspace gets a getting-started prompt;
  // populated workspace gets a "look at my latest run" prompt.
  const agentPrompt = hasRuns
    ? 'What does my latest run look like?'
    : 'How do I get started with evalyn?';

  return (
    <div style={{ padding: '60px 80px', maxWidth: 920, margin: '0 auto' }}>
      <div
        className="text-3"
        style={{
          fontSize: 11,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          marginBottom: 12,
        }}
      >
        Welcome to evalyn
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
        Run and analyze evaluations.
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
        Pick a command from the sidebar, or describe what you want and let the agent pick.
      </p>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: 12,
          marginTop: 32,
        }}
      >
        {showDemoCard && (
          <div
            data-testid="welcome-demo-card"
            onClick={() => void loadDemo()}
            role="button"
            aria-busy={demoLoading}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') void loadDemo();
            }}
            style={{
              padding: 18,
              background: 'var(--bg-2)',
              border: '1px solid var(--accent, var(--line))',
              borderRadius: 8,
              cursor: demoLoading ? 'progress' : 'pointer',
              opacity: demoLoading ? 0.7 : 1,
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
              Try demo
            </div>
            <div className="mono" style={{ fontSize: 14, marginTop: 6, color: 'var(--text-0)' }}>
              See a real eval in 30 seconds. No setup.
            </div>
            <div className="accent mono" style={{ fontSize: 11, marginTop: 14 }}>
              {demoLoading ? 'Loading demo...' : 'Load demo →'}
            </div>
            {demoError && (
              <div
                role="alert"
                className="mono"
                style={{ fontSize: 11, marginTop: 8, color: 'var(--danger, #c44918)' }}
              >
                {demoError}
              </div>
            )}
          </div>
        )}
        {baseActions.map((card) => (
          <div
            key={card.title}
            onClick={card.onAction}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') card.onAction();
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
              {card.title}
            </div>
            <div className="mono" style={{ fontSize: 14, marginTop: 6, color: 'var(--text-0)' }}>
              {card.body}
            </div>
            <div className="accent mono" style={{ fontSize: 11, marginTop: 14 }}>
              {card.cta} →
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
            Try the agent: <i>"{agentPrompt}"</i>
          </span>
          <span className="grow" />
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => void sendChatMessage(agentPrompt)}
            title="Send this prompt to the agent"
          >
            Ask the agent
          </button>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
