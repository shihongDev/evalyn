/**
 * RunsList - sidebar list of recent eval runs.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (RunsListView).
 * Data lives in `store.runs` and is populated by `loadRuns()` once Lane B1's
 * `/api/runs` endpoint ships. Click a row to open `<id>.run` as a tab.
 */

import { useStore } from '../store';
import type { RunMeta } from '../types/jobs';

const arrowFor = (r: RunMeta): { glyph: string; tone: string } => {
  if (r.regression) return { glyph: '▼', tone: 'var(--fail)' };
  if ((r.delta ?? 0) > 0) return { glyph: '▲', tone: 'var(--pass)' };
  return { glyph: '·', tone: 'var(--text-2)' };
};

const passClass = (pass: number): string => {
  if (pass < 0.6) return 'fail';
  if (pass < 0.8) return 'warn';
  return 'pass';
};

const RunsList = () => {
  const runs = useStore((s) => s.runs);
  const openFile = useStore((s) => s.openFile);

  if (runs.length === 0) {
    return (
      <div
        className="mono text-3"
        style={{ padding: '24px 14px', fontSize: 11, lineHeight: 1.6 }}
      >
        // no runs yet
      </div>
    );
  }

  return (
    <div>
      <div
        className="mono"
        style={{
          padding: '8px 14px 4px',
          fontSize: 10,
          color: 'var(--text-3)',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
        }}
      >
        runs · {runs.length}
      </div>
      {runs.map((r) => {
        const a = arrowFor(r);
        return (
          <div
            key={r.id}
            onClick={() => openFile(`${r.id}.run`)}
            style={{
              padding: '8px 14px',
              borderBottom: '1px solid var(--line)',
              cursor: 'pointer',
            }}
          >
            <div className="row mono" style={{ fontSize: 11 }}>
              <span style={{ color: a.tone }}>{a.glyph}</span>
              <span className="text-1">{r.id}</span>
              <span className="grow" />
              <span className={passClass(r.pass)}>{(r.pass * 100).toFixed(0)}%</span>
            </div>
            <div className="text-3 mono" style={{ fontSize: 10, marginTop: 2 }}>
              {r.dataset.slice(0, 28)}
            </div>
            {r.at && (
              <div className="text-3" style={{ fontSize: 10, marginTop: 2 }}>
                {r.at}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default RunsList;
