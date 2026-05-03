/**
 * Sidebar - Commands / Eval runs / History tab strip + content panes.
 *
 * The Files tab was hidden in the P0 trust pass: clicking files opened
 * "coming soon" tabs. FileTree.tsx remains for when /api/files/read gets
 * wired to a real viewer.
 */

import { useStore, type SidebarView } from '../store';
import CliCatalog from './CliCatalog';
import RunsList from './RunsList';
import JobsList from './JobsList';

const ICON_BY_VIEW: Record<SidebarView, string> = {
  files: '▤',
  clis: '$',
  runs: '▶',
  jobs: '◷',
};

const LABEL_BY_VIEW: Record<SidebarView, string> = {
  files: 'Files',
  clis: 'Commands',
  runs: 'Eval runs',
  jobs: 'History',
};

const Sidebar = () => {
  const collapsed = useStore((s) => s.tweaks.sidebarCollapsed);
  const view = useStore((s) => s.sidebarView);
  const setView = useStore((s) => s.setSidebarView);

  if (collapsed) {
    return (
      <div
        style={{
          width: 48,
          background: 'var(--bg-1)',
          borderRight: '1px solid var(--line)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '10px 0',
          gap: 6,
          flexShrink: 0,
        }}
      >
        {(['clis', 'runs', 'jobs'] as const).map((id) => (
          <button
            key={id}
            type="button"
            className="btn ghost icon"
            title={LABEL_BY_VIEW[id]}
            style={{
              width: 34,
              height: 34,
              fontSize: 15,
              color: view === id ? 'var(--accent)' : 'var(--text-2)',
              background: view === id ? 'var(--accent-soft)' : 'transparent',
            }}
            onClick={() => setView(id)}
          >
            {ICON_BY_VIEW[id]}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div
      style={{
        width: 280,
        background: 'var(--bg-1)',
        borderRight: '1px solid var(--line)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', padding: '10px 12px 4px', gap: 4 }}>
        {(['clis', 'runs', 'jobs'] as const).map((id) => (
          <button
            key={id}
            type="button"
            className="btn ghost sm"
            style={{
              background: view === id ? 'var(--bg-3)' : 'transparent',
              color: view === id ? 'var(--text-0)' : 'var(--text-2)',
              fontWeight: view === id ? 500 : 400,
            }}
            onClick={() => setView(id)}
          >
            {LABEL_BY_VIEW[id]}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 0 12px' }}>
        {view === 'clis' && <CliCatalog />}
        {view === 'runs' && <RunsList />}
        {view === 'jobs' && <JobsList />}
      </div>
    </div>
  );
};

export default Sidebar;
