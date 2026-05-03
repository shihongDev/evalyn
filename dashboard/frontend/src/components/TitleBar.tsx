/**
 * TitleBar - friendly top chrome.
 *
 * Audience includes non-coders, so we keep the brand mark + a simple settings
 * button rather than the IDE-style traffic-lights + breadcrumb + ⌘K palette.
 */

import { useStore } from '../store';

const TitleBar = () => {
  const openSettings = useStore((s) => s.openSettings);
  const tabs = useStore((s) => s.tabs);
  const activeTabId = useStore((s) => s.activeTabId);
  const activeTab = tabs.find((t) => t.id === activeTabId) ?? null;

  return (
    <div
      style={{
        height: 44,
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '0 16px',
        background: 'var(--bg-2)',
        borderBottom: '1px solid var(--line)',
        flexShrink: 0,
      }}
    >
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
      <span style={{ fontFamily: 'var(--serif)', fontSize: 18, color: 'var(--text-0)' }}>
        evalyn
      </span>
      {activeTab ? (
        <>
          <span className="text-3" style={{ marginLeft: 6, fontSize: 12 }}>
            {'·'}
          </span>
          <span className="text-2 truncate" style={{ fontSize: 13, maxWidth: 480 }}>
            {activeTab.title}
          </span>
        </>
      ) : null}
      <span className="grow" />
      <span className="chip pass dot" style={{ marginRight: 4 }}>
        Local · 7401
      </span>
      <button
        className="btn ghost sm"
        type="button"
        onClick={() => openSettings()}
      >
        Settings
      </button>
    </div>
  );
};

export default TitleBar;
