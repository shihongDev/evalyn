/**
 * CliCatalog - filter + group/alpha toggle list of CLIs.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (CliCatalogView + CliRow).
 * Data lives in `store.catalog` and is populated by `loadCatalog()` at boot.
 * Clicking a row sets the Workspace's active CLI via `selectActiveCli`.
 *
 * Groups are derived from the catalog (each CliSchema has a `group: string`).
 * Group order is the order of first appearance in the catalog.
 *
 * Onboarding (P0):
 * Surfaces a 5-CLI STARTER section above the full catalog so first-time
 * users are not overwhelmed by the 35-command surface area. The full
 * catalog is hidden behind a "Show all 35 commands" link by default.
 * Once the user has run any CLI in the session (runHistory.length > 0),
 * the default flips: starter collapses to a single "Starter (5)" link
 * and the grouped/alpha catalog renders by default. The toggle is
 * persisted across sessions via localStorage key `cli_catalog_show_all`.
 * Searching always queries across the full 35 regardless of toggle state.
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import type { CliSchema } from '../types/catalog';
import Skeleton from './Skeleton';
import { useBootGrace } from '../lib/useBootGrace';

/** Curated starter set surfaced at the top of the catalog for new users. */
const STARTER_IDS = ['quickstart', 'one-click', 'list-calls', 'status', 'workflow'];

const STORAGE_KEY = 'cli_catalog_show_all';

const readShowAll = (): boolean => {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
};

const writeShowAll = (v: boolean): void => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, v ? 'true' : 'false');
  } catch {
    // ignore — non-fatal for an onboarding affordance
  }
};

interface RowProps {
  cli: CliSchema;
  active: boolean;
  starter?: boolean;
  onClick: () => void;
}

const CliRow = ({ cli, active, starter = false, onClick }: RowProps) => (
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
    <span
      className={starter ? 'accent' : 'text-3'}
      style={{ fontSize: 10, width: 10, display: 'inline-block', textAlign: 'center' }}
      aria-hidden
    >
      {starter ? '★' : '$'}
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

const LinkButton = ({
  onClick,
  children,
  ariaExpanded,
}: {
  onClick: () => void;
  children: React.ReactNode;
  ariaExpanded?: boolean;
}) => (
  <button
    type="button"
    onClick={onClick}
    aria-expanded={ariaExpanded}
    style={{
      all: 'unset',
      display: 'block',
      padding: '6px 14px',
      fontFamily: 'var(--mono)',
      fontSize: 11,
      color: 'var(--text-2)',
      cursor: 'pointer',
      width: '100%',
      boxSizing: 'border-box',
    }}
  >
    {children}
  </button>
);

const CliCatalog = () => {
  const catalog = useStore((s) => s.catalog);
  const selectActiveCli = useStore((s) => s.selectActiveCli);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const activeCli = useStore((s) => s.activeCliId);
  const hasRunHistory = useStore((s) => s.runHistory.length > 0);

  const [filter, setFilter] = useState('');
  const [alpha, setAlpha] = useState(false);
  const inBootGrace = useBootGrace(1500);
  // showAll: when true, the grouped/alpha "all 35" catalog is rendered.
  // Persisted across sessions; defaults to `false` for first-time users
  // and flips to `true` once the user has actually run something.
  const [showAll, setShowAll] = useState<boolean>(() => readShowAll());

  // Once the user has run a CLI in the session, default the catalog
  // to expanded — but only if they have not already explicitly chosen.
  // We treat the very first runHistory entry as the trigger and write
  // `true` to localStorage so the choice carries forward.
  useEffect(() => {
    if (hasRunHistory && !readShowAll()) {
      writeShowAll(true);
      setShowAll(true);
    }
  }, [hasRunHistory]);

  const onPick = (id: string) => {
    selectActiveCli(id);
    // Switch to the Workspace so the user sees the form for the chosen
    // CLI (the activeTab might be a job:* or file tab).
    setActiveTab(null);
  };

  const toggleShowAll = () => {
    const next = !showAll;
    setShowAll(next);
    writeShowAll(next);
  };

  const filtered = useMemo(() => {
    if (!filter) return catalog;
    const q = filter.toLowerCase();
    return catalog.filter(
      (c) => c.id.toLowerCase().includes(q) || c.blurb.toLowerCase().includes(q),
    );
  }, [catalog, filter]);

  // Resolve starter CLIs in the curated order, dropping any that are
  // not present in the catalog (defensive — keeps the starter set
  // working if a CLI is renamed or removed).
  const starterClis = useMemo(() => {
    const byId = new Map(catalog.map((c) => [c.id, c]));
    return STARTER_IDS.map((id) => byId.get(id)).filter(
      (c): c is CliSchema => c != null,
    );
  }, [catalog]);

  // Derive group order from the catalog (first appearance wins).
  const groupOrder = useMemo(() => {
    const seen: string[] = [];
    for (const c of catalog) {
      if (!seen.includes(c.group)) seen.push(c.group);
    }
    return seen;
  }, [catalog]);

  if (catalog.length === 0) {
    if (inBootGrace) {
      return <Skeleton rows={6} height={20} testId="cli-catalog-skeleton" />;
    }
    return (
      <div
        className="text-3"
        style={{ padding: '24px 14px', fontSize: 12, lineHeight: 1.6 }}
      >
        Loading commands...
      </div>
    );
  }

  // Searching always queries the full catalog regardless of starter
  // collapse state (Task 1 contract).
  const searching = filter.length > 0;

  // Decide whether the starter section is rendered as a full list or
  // collapsed to a single link. After the user has run something we
  // collapse it by default; before that, we always show the starter set.
  const starterCollapsed = hasRunHistory && showAll;

  const renderStarterSection = () => {
    if (starterClis.length === 0) return null;

    if (starterCollapsed) {
      return (
        <LinkButton onClick={() => setShowAll(false)}>
          {'★ '}Starter ({starterClis.length})
        </LinkButton>
      );
    }

    return (
      <div style={{ marginBottom: 4 }}>
        <SectionLabel>start here · {starterClis.length}</SectionLabel>
        {starterClis.map((c) => (
          <CliRow
            key={c.id}
            cli={c}
            active={activeCli === c.id}
            starter
            onClick={() => onPick(c.id)}
          />
        ))}
      </div>
    );
  };

  const renderFullCatalog = () => {
    if (alpha) {
      return (
        <div>
          <SectionLabel>all clis · alpha</SectionLabel>
          {catalog
            .slice()
            .sort((a, b) => a.id.localeCompare(b.id))
            .map((c) => (
              <CliRow
                key={c.id}
                cli={c}
                active={activeCli === c.id}
                onClick={() => onPick(c.id)}
              />
            ))}
        </div>
      );
    }
    return (
      <>
        {groupOrder.map((g) => {
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
                  onClick={() => onPick(c.id)}
                />
              ))}
            </div>
          );
        })}
      </>
    );
  };

  return (
    <div>
      <div style={{ padding: '0 12px 6px', display: 'flex', gap: 6 }}>
        <input
          className="input"
          style={{ padding: '4px 8px', fontSize: 11, height: 26 }}
          placeholder="Search commands"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Search commands"
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

      {searching ? (
        <div>
          <SectionLabel>results · {filtered.length}</SectionLabel>
          {filtered
            .slice()
            .sort((a, b) => a.id.localeCompare(b.id))
            .map((c) => (
              <CliRow
                key={c.id}
                cli={c}
                active={activeCli === c.id}
                onClick={() => onPick(c.id)}
              />
            ))}
        </div>
      ) : (
        <>
          {renderStarterSection()}
          {showAll ? (
            renderFullCatalog()
          ) : (
            <LinkButton onClick={toggleShowAll} ariaExpanded={false}>
              Show all {catalog.length} commands
            </LinkButton>
          )}
          {showAll && !starterCollapsed && (
            <LinkButton onClick={toggleShowAll} ariaExpanded>
              Hide full catalog
            </LinkButton>
          )}
        </>
      )}
    </div>
  );
};

export default CliCatalog;
