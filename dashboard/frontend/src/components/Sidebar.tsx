/**
 * Sidebar — Files / CLIs / Runs tab strip + content panes.
 *
 * Skeleton ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (lines 57-99).
 * The actual content of each pane (file tree, CLI catalog rows, run rows)
 * is filled in by Lane B2; this lane wires the tab strip + collapsible
 * icon rail and renders empty placeholders.
 */

import { useStore, type SidebarView } from '../store';

const ICON_BY_VIEW: Record<SidebarView | 'search', string> = {
  files: '▤',
  clis: '$',
  runs: '▶',
  search: '⌕',
};

const LABEL_BY_VIEW: Record<SidebarView, string> = {
  files: 'Files',
  clis: 'CLIs',
  runs: 'Runs',
};

const EmptyPane = ({ label }: { label: string }) => (
  <div
    className="mono text-3"
    style={{
      padding: '24px 14px',
      fontSize: 11,
      lineHeight: 1.6,
    }}
  >
    {label}
    <div style={{ marginTop: 6 }}>{'// Phase 2 (Lane B2)'}</div>
  </div>
);

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
        {(['files', 'clis', 'runs'] as const).map((id) => (
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
        {(['files', 'clis', 'runs'] as const).map((id) => (
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
        {view === 'files' && <EmptyPane label="// file tree" />}
        {view === 'clis' && <EmptyPane label="// CLI catalog" />}
        {view === 'runs' && <EmptyPane label="// runs list" />}
      </div>
    </div>
  );
};

export default Sidebar;
