/**
 * EditorTabs — VS-Code-style tab strip above the editor area.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (lines 218-249).
 * Tab data and active id come from the Zustand store.
 */

import { useStore } from '../store';
import type { Tab } from '../types/jobs';

const iconForKind = (kind: Tab['kind']): string => {
  switch (kind) {
    case 'cli':
      return '$';
    case 'run':
      return '▶';
    case 'yaml':
      return '⌬';
    case 'job':
      return '⚙';
    default:
      return '·';
  }
};

const EditorTabs = () => {
  const tabs = useStore((s) => s.tabs);
  const activeTabId = useStore((s) => s.activeTabId);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const closeTab = useStore((s) => s.closeTab);

  if (tabs.length === 0) {
    // Empty tab strip; the welcome view fills the editor area below.
    return (
      <div
        style={{
          display: 'flex',
          background: 'var(--bg-1)',
          borderBottom: '1px solid var(--line)',
          flexShrink: 0,
          height: 33,
        }}
      />
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        background: 'var(--bg-1)',
        borderBottom: '1px solid var(--line)',
        overflowX: 'auto',
        flexShrink: 0,
      }}
    >
      {tabs.map((tab) => {
        const isActive = activeTabId === tab.id;
        return (
          <div
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '8px 12px 8px 14px',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              fontSize: 11,
              fontFamily: 'var(--mono)',
              color: isActive ? 'var(--text-0)' : 'var(--text-2)',
              borderRight: '1px solid var(--line)',
              borderTop: isActive ? '2px solid var(--accent)' : '2px solid transparent',
              borderBottom: isActive ? '1px solid var(--bg-0)' : '1px solid transparent',
              marginBottom: -1,
              background: isActive ? 'var(--bg-0)' : 'transparent',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <span style={{ color: tab.tone ?? 'var(--text-3)', fontSize: 11 }}>
              {iconForKind(tab.kind)}
            </span>
            <span>{tab.title}</span>
            {tab.dirty && (
              <span className="accent" style={{ fontSize: 14, lineHeight: 0 }}>
                ●
              </span>
            )}
            <span
              role="button"
              aria-label={`Close ${tab.title}`}
              onClick={(e) => {
                e.stopPropagation();
                closeTab(tab.id);
              }}
              style={{
                marginLeft: 4,
                color: 'var(--text-3)',
                fontSize: 14,
                lineHeight: 0.7,
                cursor: 'pointer',
              }}
            >
              ×
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default EditorTabs;
