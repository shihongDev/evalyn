/**
 * App shell — IDE layout.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (App, lines 573+).
 * Three-column grid: Sidebar | (EditorTabs / content / BottomPanel) | ChatPanel.
 * ChatPanel is a placeholder until Lane C2 ships in Phase 3.
 */

import { useEffect } from 'react';
import './styles/index.css';
import TitleBar from './components/TitleBar';
import EditorTabs from './components/EditorTabs';
import Sidebar from './components/Sidebar';
import BottomPanel from './components/BottomPanel';
import Welcome from './views/Welcome';
import { useStore } from './store';

const ChatPanelPlaceholder = () => (
  <aside
    style={{
      width: 420,
      background: 'var(--bg-1)',
      borderLeft: '1px solid var(--line)',
      display: 'flex',
      flexDirection: 'column',
      minHeight: 0,
    }}
  >
    <div
      style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--line)',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <span className="accent mono" style={{ fontSize: 11 }}>
        ✦ agent
      </span>
      <span className="grow" />
      <span className="text-3 mono" style={{ fontSize: 10 }}>
        Phase 3
      </span>
    </div>
    <div
      className="mono text-3"
      style={{ padding: '24px 18px', fontSize: 11, lineHeight: 1.7 }}
    >
      {'// ChatPanel ships in Phase 3 (Lane C2)'}
    </div>
  </aside>
);

const TabContent = () => {
  const tabs = useStore((s) => s.tabs);
  const activeTabId = useStore((s) => s.activeTabId);
  const activeTab = tabs.find((t) => t.id === activeTabId) ?? null;

  if (!activeTab) {
    return <Welcome />;
  }
  return (
    <div className="mono text-3" style={{ padding: 28, fontSize: 12, lineHeight: 1.7 }}>
      <div className="text-2">{activeTab.title}</div>
      <div style={{ marginTop: 12 }}>
        {`// ${activeTab.kind} content ships in Phase 2 (Lane B2)`}
      </div>
    </div>
  );
};

const App = () => {
  const theme = useStore((s) => s.tweaks.theme);
  const monoOnly = useStore((s) => s.tweaks.monoOnly);
  const showJobsPanel = useStore((s) => s.tweaks.showJobsPanel);
  const chatPlacement = useStore((s) => s.tweaks.chatPlacement);
  const chatVisible = useStore((s) => s.chatVisible);
  const setChatVisible = useStore((s) => s.setChatVisible);
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);

  // Apply theme + mono toggle to <body>, mirroring the mock's effect.
  useEffect(() => {
    document.body.dataset.theme = theme;
    document.body.dataset.mono = monoOnly ? 'true' : 'false';
  }, [theme, monoOnly]);

  // Cmd/Ctrl-K opens the palette; Escape closes it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setPaletteOpen(true);
      }
      if (e.key === 'Escape') {
        setPaletteOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setPaletteOpen]);

  const chatDocked = chatPlacement === 'dock' && chatVisible;

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-0)',
      }}
    >
      <TitleBar />

      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: `auto 1fr ${chatDocked ? '420px' : ''}`,
          minHeight: 0,
        }}
      >
        <Sidebar />

        <div
          style={{
            display: 'grid',
            gridTemplateRows: showJobsPanel ? 'auto 1fr 220px' : 'auto 1fr',
            minWidth: 0,
            minHeight: 0,
          }}
        >
          <EditorTabs />
          <div style={{ overflow: 'auto', background: 'var(--bg-0)' }}>
            <TabContent />
          </div>
          {showJobsPanel && <BottomPanel />}
        </div>

        {chatDocked && <ChatPanelPlaceholder />}
      </div>

      {!chatVisible && (
        <button
          type="button"
          onClick={() => setChatVisible(true)}
          className="btn primary"
          style={{
            position: 'fixed',
            right: 24,
            bottom: 24,
            height: 44,
            padding: '0 16px',
            borderRadius: 22,
            boxShadow: 'var(--shadow)',
            zIndex: 20,
          }}
        >
          ✦ Ask agent
        </button>
      )}
    </div>
  );
};

export default App;
