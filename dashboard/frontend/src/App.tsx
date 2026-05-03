/**
 * App shell.
 *
 * Two-column grid: Sidebar | (EditorTabs / TabContent) | ChatPanel.
 * Job output renders inline inside its job tab (no bottom panel split).
 */

import { useEffect } from 'react';
import './styles/index.css';
import TitleBar from './components/TitleBar';
import EditorTabs from './components/EditorTabs';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import CommandPalette from './components/CommandPalette';
import InlineChat from './components/InlineChat';
import SettingsModal from './components/SettingsModal';
import Terminal from './components/Terminal';
import Workspace from './views/Workspace';
import CompareView from './views/CompareView';
import { useStore } from './store';

const JobTabView = ({ jobId }: { jobId: string }) => {
  const job = useStore((s) => s.jobs.get(jobId));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div
        style={{
          padding: '14px 22px',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-1)',
          flexShrink: 0,
        }}
      >
        <div className="mono text-3" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          Output
        </div>
        <div className="mono" style={{ fontSize: 13, color: 'var(--text-0)', marginTop: 4 }}>
          {job?.cmd ?? `evalyn (job ${jobId.slice(0, 8)})`}
        </div>
        <div className="text-3" style={{ fontSize: 11, marginTop: 4 }}>
          status: <span className={
            job?.status === 'complete' ? 'pass' :
            job?.status === 'failed' || job?.status === 'cancelled' ? 'fail' :
            'accent'
          }>{job?.status ?? 'running'}</span>
          {job?.exitCode != null ? <span className="text-3"> · exit {job.exitCode}</span> : null}
          {job?.duration ? <span className="text-3"> · {job.duration}</span> : null}
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        <Terminal jobId={jobId} />
      </div>
    </div>
  );
};

const TabContent = () => {
  const tabs = useStore((s) => s.tabs);
  const activeTabId = useStore((s) => s.activeTabId);
  const activeTab = tabs.find((t) => t.id === activeTabId) ?? null;

  // Workspace is the default view. Sidebar CLI clicks set the active CLI
  // via `selectActiveCli` rather than opening a tab. Chat-suggestion clicks
  // open a `cli:` tab AND set the active CLI (see `openCli` in store.ts),
  // so the Workspace renders the form for the opened tab in either case.
  if (!activeTab || activeTab.kind === 'cli') {
    return <Workspace />;
  }

  if (activeTab.kind === 'job') {
    const jobId = activeTab.id.replace(/^job:/, '');
    return <JobTabView jobId={jobId} />;
  }

  if (activeTab.kind === 'compare') {
    // Tab id format: `compare:<runA>:<runB>`. Run ids may legitimately
    // contain underscores or dashes but never colons, so a 3-part split is
    // sufficient. A malformed id (missing B) renders an inline error.
    const parts = activeTab.id.split(':');
    const runA = parts[1] ?? '';
    const runB = parts[2] ?? '';
    return <CompareView runA={runA} runB={runB} />;
  }

  return (
    <div style={{ padding: 28, color: 'var(--text-2)', fontSize: 13 }}>
      <div style={{ fontSize: 16, color: 'var(--text-0)', marginBottom: 8 }}>
        {activeTab.title}
      </div>
      <div>This view is coming soon.</div>
    </div>
  );
};

const App = () => {
  const theme = useStore((s) => s.tweaks.theme);
  const monoOnly = useStore((s) => s.tweaks.monoOnly);
  const chatVisible = useStore((s) => s.chatVisible);
  const setChatVisible = useStore((s) => s.setChatVisible);
  const loadCatalog = useStore((s) => s.loadCatalog);
  const loadFileTree = useStore((s) => s.loadFileTree);
  const loadRuns = useStore((s) => s.loadRuns);
  const loadSettings = useStore((s) => s.loadSettings);
  const loadThreads = useStore((s) => s.loadThreads);
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);

  // Boot fetches: catalog / file tree / runs. Failures are non-fatal in Phase 2
  // because the corresponding endpoints land in Lane B1; the UI degrades to
  // empty placeholders. loadThreads reads from localStorage only — no network.
  useEffect(() => {
    loadCatalog().catch((err) => {
      console.warn('loadCatalog failed', err);
    });
    loadFileTree().catch((err) => {
      console.warn('loadFileTree failed', err);
    });
    loadRuns().catch((err) => {
      console.warn('loadRuns failed', err);
    });
    loadSettings().catch((err) => {
      console.warn('loadSettings failed', err);
    });
    loadThreads();
  }, [loadCatalog, loadFileTree, loadRuns, loadSettings, loadThreads]);

  // Apply theme + mono toggle to <body>, mirroring the mock's effect.
  useEffect(() => {
    document.body.dataset.theme = theme;
    document.body.dataset.mono = monoOnly ? 'true' : 'false';
  }, [theme, monoOnly]);

  // Cmd/Ctrl-K toggles the command palette; Cmd/Ctrl-I toggles the inline
  // chat popover anchored to the focused element. Escape closes the palette.
  // We intentionally toggle (not just open) so a second keypress dismisses
  // the surface — matches Linear / Vercel / Raycast / Cursor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        const { paletteOpen } = useStore.getState();
        setPaletteOpen(!paletteOpen);
        return;
      }
      if ((e.metaKey || e.ctrlKey) && (e.key === 'i' || e.key === 'I')) {
        e.preventDefault();
        const { inlineChatOpen, openInlineChat, closeInlineChat } =
          useStore.getState();
        if (inlineChatOpen) {
          closeInlineChat();
          return;
        }
        const el = document.activeElement;
        if (el && el !== document.body) {
          const r = (el as HTMLElement).getBoundingClientRect();
          openInlineChat({ x: r.left + r.width / 2, y: r.bottom + 8 });
        } else {
          openInlineChat(null);
        }
        return;
      }
      if (e.key === 'Escape' && useStore.getState().paletteOpen) {
        e.preventDefault();
        setPaletteOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setPaletteOpen]);

  // Per spec §8 the dock-right placement is the only one shipped. We keep
  // chatVisible as the toggle so the user can still hide the panel.
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
          gridTemplateColumns: `auto 1fr ${chatVisible ? '420px' : ''}`,
          minHeight: 0,
        }}
      >
        <Sidebar />

        <div
          style={{
            display: 'grid',
            gridTemplateRows: 'auto 1fr',
            minWidth: 0,
            minHeight: 0,
          }}
        >
          <EditorTabs />
          <div style={{ overflow: 'auto', background: 'var(--bg-0)', minHeight: 0 }}>
            <TabContent />
          </div>
        </div>

        {chatVisible && <ChatPanel />}
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

      <SettingsModal />
      <CommandPalette />
      <InlineChat />
    </div>
  );
};

export default App;
