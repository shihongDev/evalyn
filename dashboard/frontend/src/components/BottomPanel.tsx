/**
 * BottomPanel — Terminal / Jobs / Problems tab strip + live panes.
 *
 * Tab strip ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (lines 425-518).
 * Inner Terminal + Jobs panes wired to real components in Lane B3 (this
 * lane). Problems stays a stub through Phase 2.
 */

import { useStore, type BottomTab } from '../store';
import Terminal from './Terminal';
import JobsList from './JobsList';

const ProblemsPane = () => (
  <div className="mono text-3" style={{ fontSize: 11, lineHeight: 1.7, padding: '6px 0' }}>
    <div>{'// problems pane — not implemented in v1'}</div>
  </div>
);

const BottomPanel = () => {
  const tab = useStore((s) => s.bottomTab);
  const setTab = useStore((s) => s.setBottomTab);
  const jobs = useStore((s) => s.jobs);
  const jobsCount = jobs.size;
  const runningCount = Array.from(jobs.values()).filter((j) => j.status === 'running').length;
  const todayCount = jobsCount;

  const tabs: Array<[BottomTab, string]> = [
    ['terminal', 'Terminal'],
    ['jobs', `Jobs · ${jobsCount}`],
    ['problems', 'Problems · 0'],
  ];

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-1)',
        borderTop: '1px solid var(--line)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0 14px',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-2)',
          flexShrink: 0,
        }}
      >
        {tabs.map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className="btn ghost sm"
            style={{
              padding: '8px 12px',
              borderRadius: 0,
              color: tab === id ? 'var(--text-0)' : 'var(--text-2)',
              borderBottom: tab === id ? '2px solid var(--accent)' : '2px solid transparent',
              marginBottom: -1,
            }}
          >
            {label}
          </button>
        ))}
        <span className="grow" />
        <span className="text-3 mono" style={{ fontSize: 10, marginRight: 8 }}>
          {runningCount} running · {todayCount} today
        </span>
        <button className="btn ghost icon" title="Clear" type="button">
          {'⊘'}
        </button>
      </div>
      <div style={{ flex: 1, overflow: 'hidden', padding: 14 }}>
        {tab === 'terminal' && <Terminal />}
        {tab === 'jobs' && <JobsList />}
        {tab === 'problems' && <ProblemsPane />}
      </div>
    </div>
  );
};

export default BottomPanel;
