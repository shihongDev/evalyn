/**
 * TitleBar — top chrome.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (lines 28-55).
 * Replaces the mock's prop-based handlers with Zustand selectors.
 */

import { useStore } from '../store';

const ACTIVE_RUN_DATASET = 'datasets/customer-support.jsonl';

const TitleBar = () => {
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);
  const setTweaksOpen = useStore((s) => s.setTweaksOpen);
  const tabs = useStore((s) => s.tabs);
  const activeTabId = useStore((s) => s.activeTabId);
  const activeTab = tabs.find((t) => t.id === activeTabId) ?? null;

  const breadcrumbs = `${ACTIVE_RUN_DATASET} / runs / ${activeTab?.title ?? '—'}`;

  return (
    <div
      style={{
        height: 38,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '0 12px',
        background: 'var(--bg-2)',
        borderBottom: '1px solid var(--line)',
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', gap: 6 }}>
        <span style={{ width: 12, height: 12, background: '#ff5f56', borderRadius: '50%' }} />
        <span style={{ width: 12, height: 12, background: '#ffbd2e', borderRadius: '50%' }} />
        <span style={{ width: 12, height: 12, background: '#27c93f', borderRadius: '50%' }} />
      </div>
      <div style={{ width: 1, height: 18, background: 'var(--line)', margin: '0 6px' }} />
      <div style={{ width: 22, height: 22, position: 'relative' }}>
        <div
          style={{
            position: 'absolute',
            inset: 0,
            border: '1.5px solid var(--accent)',
            borderRadius: '50%',
          }}
        />
        <div
          style={{
            position: 'absolute',
            inset: 4,
            border: '1px dashed var(--text-2)',
            borderRadius: '50%',
          }}
        />
      </div>
      <span style={{ fontFamily: 'var(--serif)', fontSize: 16, color: 'var(--text-0)' }}>
        evalyn{' '}
        <em className="text-2" style={{ fontSize: 13 }}>
          workbench
        </em>
      </span>
      <span className="text-3" style={{ marginLeft: 14, fontSize: 11 }}>
        {'›'}
      </span>
      <span className="mono text-2 truncate" style={{ fontSize: 11, maxWidth: 480 }}>
        {breadcrumbs}
      </span>
      <span className="grow" />
      <button className="btn ghost sm" onClick={() => setPaletteOpen(true)}>
        <span style={{ fontSize: 11 }}>Search files, runs, CLIs</span>
        <span className="kbd" style={{ marginLeft: 8 }}>
          {'⌘K'}
        </span>
      </button>
      <button className="btn ghost icon" title="Workspace settings" type="button">
        {'⚙'}
      </button>
      <span className="chip pass dot" style={{ marginLeft: 6 }}>
        localhost:7401
      </span>
      <button
        className="btn ghost icon"
        title="Tweaks"
        type="button"
        onClick={() => setTweaksOpen(true)}
      >
        {'⌐'}
      </button>
    </div>
  );
};

export default TitleBar;
