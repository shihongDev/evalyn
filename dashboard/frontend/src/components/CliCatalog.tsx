/**
 * CliCatalog - filter + group/alpha toggle list of CLIs.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (CliCatalogView + CliRow).
 * Data lives in `store.catalog` and is populated by `loadCatalog()` at boot.
 * Each row click opens a `cli:<id>` tab via `openCli`.
 *
 * Groups are derived from the catalog (each CliSchema has a `group: string`).
 * Group order is the order of first appearance in the catalog.
 */

import { useMemo, useState } from 'react';
import { useStore } from '../store';
import type { CliSchema } from '../types/catalog';

interface RowProps {
  cli: CliSchema;
  active: boolean;
  onClick: () => void;
}

const CliRow = ({ cli, active, onClick }: RowProps) => (
  <div
    onClick={onClick}
    style={{
      padding: '5px 14px',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      fontFamily: 'var(--mono)',
      fontSize: 12,
      color: 'var(--text-1)',
      cursor: 'pointer',
      background: active ? 'var(--accent-soft)' : 'transparent',
      borderLeft: active ? '2px solid var(--accent)' : '2px solid transparent',
    }}
  >
    <span className="accent" style={{ fontSize: 10 }}>
      $
    </span>
    <span className="grow truncate">{cli.id}</span>
  </div>
);

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
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
    {children}
  </div>
);

const CliCatalog = () => {
  const catalog = useStore((s) => s.catalog);
  const activeTabId = useStore((s) => s.activeTabId);
  const openCli = useStore((s) => s.openCli);
  const [filter, setFilter] = useState('');
  const [alpha, setAlpha] = useState(false);

  const activeCli =
    activeTabId && activeTabId.startsWith('cli:') ? activeTabId.slice(4) : null;

  const filtered = useMemo(() => {
    if (!filter) return catalog;
    const q = filter.toLowerCase();
    return catalog.filter(
      (c) => c.id.toLowerCase().includes(q) || c.blurb.toLowerCase().includes(q),
    );
  }, [catalog, filter]);

  // Derive group order from the catalog (first appearance wins).
  const groupOrder = useMemo(() => {
    const seen: string[] = [];
    for (const c of catalog) {
      if (!seen.includes(c.group)) seen.push(c.group);
    }
    return seen;
  }, [catalog]);

  if (catalog.length === 0) {
    return (
      <div
        className="mono text-3"
        style={{ padding: '24px 14px', fontSize: 11, lineHeight: 1.6 }}
      >
        // catalog loading…
      </div>
    );
  }

  return (
    <div>
      <div style={{ padding: '0 12px 6px', display: 'flex', gap: 6 }}>
        <input
          className="input"
          style={{ padding: '4px 8px', fontSize: 11, height: 26 }}
          placeholder="filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="filter catalog"
        />
        <button
          type="button"
          className="btn ghost sm"
          onClick={() => setAlpha(!alpha)}
          title={alpha ? 'By group' : 'Alphabetical'}
        >
          {alpha ? 'abc' : '▤'}
        </button>
      </div>
      {alpha || filter ? (
        <div>
          <SectionLabel>
            {filter ? `results · ${filtered.length}` : 'all clis · alpha'}
          </SectionLabel>
          {filtered
            .slice()
            .sort((a, b) => a.id.localeCompare(b.id))
            .map((c) => (
              <CliRow
                key={c.id}
                cli={c}
                active={activeCli === c.id}
                onClick={() => openCli(c.id)}
              />
            ))}
        </div>
      ) : (
        groupOrder.map((g) => {
          const items = catalog.filter((c) => c.group === g);
          if (items.length === 0) return null;
          return (
            <div key={g} style={{ marginBottom: 4 }}>
              <SectionLabel>
                {g} · {items.length}
              </SectionLabel>
              {items.map((c) => (
                <CliRow
                  key={c.id}
                  cli={c}
                  active={activeCli === c.id}
                  onClick={() => openCli(c.id)}
                />
              ))}
            </div>
          );
        })
      )}
    </div>
  );
};

export default CliCatalog;
