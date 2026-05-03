/**
 * Command palette (Cmd+K).
 *
 * Linear-inspired modal palette with three sections:
 *   1. Run command   - all CLIs from the catalog, grouped by `cli.group`.
 *   2. Open recent   - the 5 newest entries in `store.runs`.
 *   3. Ask agent     - one row that pipes the current query into the chat.
 *
 * Filtering: case-insensitive substring against id + blurb (CLIs) or run id
 * (runs). Exact-prefix matches sort first, then substring matches. The Ask
 * row is always present so the user can fall back to free-text intent.
 *
 * Keyboard: ↑/↓ move highlight (wraps), Enter activates, Esc closes. Click
 * also activates. Focus returns to the previously-focused element on close.
 */

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useStore } from '../store';
import type { CliSchema } from '../types/catalog';
import type { RunMeta } from '../types/jobs';

/** A flat row in the palette list, regardless of section. */
interface PaletteRow {
  /** Stable key for React + aria-activedescendant. */
  key: string;
  section: 'run' | 'open' | 'ask';
  /** Top-line label, monospace. */
  label: string;
  /** Secondary blurb / metadata, sans, dimmed. */
  hint?: string;
  /** Section group header to display before this row (only on the first
   *  row of each visual block). */
  groupHeader?: string;
  onActivate: () => void;
}

/** Sort priority: lower = appears earlier. */
const matchRank = (haystack: string, needle: string): number => {
  if (!needle) return 0;
  const h = haystack.toLowerCase();
  const n = needle.toLowerCase();
  if (h.startsWith(n)) return 0;
  if (h.includes(n)) return 1;
  return 2;
};

const matches = (haystack: string, needle: string): boolean => {
  if (!needle) return true;
  return haystack.toLowerCase().includes(needle.toLowerCase());
};

/** Build the ordered Map<group, CliSchema[]> once per render. */
const groupCatalog = (catalog: CliSchema[]): Map<string, CliSchema[]> => {
  const groups = new Map<string, CliSchema[]>();
  for (const cli of catalog) {
    const g = cli.group || 'Other';
    const bucket = groups.get(g) ?? [];
    bucket.push(cli);
    groups.set(g, bucket);
  }
  return groups;
};

const CommandPalette = () => {
  const paletteOpen = useStore((s) => s.paletteOpen);
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);
  const catalog = useStore((s) => s.catalog);
  const runs = useStore((s) => s.runs);
  const openCli = useStore((s) => s.openCli);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const sendChatMessage = useStore((s) => s.sendChatMessage);
  const chatVisible = useStore((s) => s.chatVisible);
  const setChatVisible = useStore((s) => s.setChatVisible);

  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const listboxId = useId();

  // Reset query + highlight on each open. Capture the previously focused
  // element so we can return focus to it when the palette closes.
  useEffect(() => {
    if (paletteOpen) {
      previousFocusRef.current = (document.activeElement as HTMLElement) ?? null;
      setQuery('');
      setHighlight(0);
      // Defer autofocus to next tick so the ref is attached.
      const id = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(id);
    }
    previousFocusRef.current?.focus?.();
    return undefined;
  }, [paletteOpen]);

  // Memo the row list so arrow-key handling and rendering share one source.
  const rows = useMemo<PaletteRow[]>(() => {
    const out: PaletteRow[] = [];

    // ---- Run command -------------------------------------------------------
    const groups = groupCatalog(catalog);
    const groupNames = Array.from(groups.keys()).sort((a, b) => a.localeCompare(b));
    let runHeaderEmitted = false;
    for (const group of groupNames) {
      const items = (groups.get(group) ?? [])
        .filter((cli) => matches(`${cli.id} ${cli.blurb}`, query))
        .sort((a, b) => {
          const ra = matchRank(a.id, query);
          const rb = matchRank(b.id, query);
          if (ra !== rb) return ra - rb;
          return a.id.localeCompare(b.id);
        });
      if (items.length === 0) continue;
      items.forEach((cli, i) => {
        // First row across the whole section gets the section header; otherwise
        // the first row of each group gets the group header.
        let groupHeader: string | undefined;
        if (!runHeaderEmitted) groupHeader = 'Run command';
        else if (i === 0) groupHeader = group;
        out.push({
          key: `cli:${cli.id}`,
          section: 'run',
          label: `evalyn ${cli.id}`,
          hint: cli.blurb,
          groupHeader,
          onActivate: () => {
            openCli(cli.id);
            setPaletteOpen(false);
          },
        });
        runHeaderEmitted = true;
      });
    }

    // ---- Open recent -------------------------------------------------------
    // Only surface recents when the analyze CLI actually exists in the
    // catalog. Without this, a boot-race (catalog not yet loaded) or a
    // future rename of `analyze` would leave the user staring at a blank
    // workspace after activating the row.
    const hasAnalyze = catalog.some((c) => c.id === 'analyze');
    const filteredRuns = hasAnalyze
      ? runs
          .filter((r) => matches(r.id, query))
          .sort((a, b) => matchRank(a.id, query) - matchRank(b.id, query))
          .slice(0, 5)
      : [];
    filteredRuns.forEach((run, i) => {
      out.push({
        key: `run:${run.id}`,
        section: 'open',
        label: run.id,
        hint: formatRunMeta(run),
        groupHeader: i === 0 ? 'Open recent' : undefined,
        // NOTE: bypasses selectActiveCli's auto-seed-from-history because
        // we're providing an explicit seed. If selectActiveCli ever grows
        // additional side effects (e.g. analytics), this path needs updating.
        onActivate: () => {
          useStore.setState({
            activeCliId: 'analyze',
            activeFormSeed: { cliId: 'analyze', args: { run: run.id } },
          });
          setActiveTab(null);
          setPaletteOpen(false);
        },
      });
    });

    // ---- Ask agent (always last, always present) --------------------------
    const askLabel = query.trim()
      ? `Ask the agent about: ${query.trim()}`
      : 'Ask the agent';
    out.push({
      key: 'ask',
      section: 'ask',
      label: askLabel,
      hint: query.trim() ? undefined : 'Type a question, then press Enter.',
      groupHeader: 'Ask agent',
      onActivate: () => {
        const q = query.trim();
        if (!chatVisible) setChatVisible(true);
        // Empty query: just open chat so the user can type there.
        if (q) void sendChatMessage(q);
        setPaletteOpen(false);
      },
    });

    return out;
    // setPaletteOpen / openCli / setActiveTab / sendChatMessage / setChatVisible
    // are stable Zustand action references; chatVisible is read inside the
    // closure but only matters at click-time, so the deps stay tight.
  }, [query, catalog, runs, openCli, setPaletteOpen, setActiveTab, sendChatMessage, chatVisible, setChatVisible]);

  // Clamp the highlight whenever the row list shrinks (typing narrows results).
  useEffect(() => {
    if (highlight >= rows.length) {
      setHighlight(Math.max(0, rows.length - 1));
    }
  }, [rows.length, highlight]);

  // Scroll the highlighted row into view when navigating with the keyboard.
  // Guarded for jsdom (which doesn't implement scrollIntoView).
  useEffect(() => {
    if (!paletteOpen) return;
    const row = listRef.current?.querySelector(`[data-row-index="${highlight}"]`);
    if (row && typeof (row as HTMLElement).scrollIntoView === 'function') {
      (row as HTMLElement).scrollIntoView({ block: 'nearest' });
    }
  }, [highlight, paletteOpen]);

  if (!paletteOpen) return null;

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const n = rows.length;
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlight((i) => (n === 0 ? 0 : (i + 1) % n));
        return;
      case 'ArrowUp':
        e.preventDefault();
        setHighlight((i) => (n === 0 ? 0 : (i - 1 + n) % n));
        return;
      case 'Enter':
        e.preventDefault();
        rows[highlight]?.onActivate();
        return;
      case 'Escape':
        e.preventDefault();
        setPaletteOpen(false);
    }
  };

  return (
    <div
      onClick={() => setPaletteOpen(false)}
      data-testid="command-palette-scrim"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.45)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '14vh',
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="command-palette"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        style={{
          width: 640,
          maxWidth: 'calc(100vw - 32px)',
          height: 'min(540px, 60vh)',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--bg-1)',
          border: '1px solid var(--line)',
          borderRadius: 12,
          boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            padding: '14px 18px',
            borderBottom: '1px solid var(--line)',
            background: 'var(--bg-1)',
          }}
        >
          <span id={titleId} className="text-3" style={{ display: 'none' }}>
            Command palette
          </span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setHighlight(0);
            }}
            placeholder="Run a command, open a run, or ask the agent..."
            aria-controls={listboxId}
            aria-label="Command palette search"
            data-testid="command-palette-input"
            style={{
              width: '100%',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: 'var(--text-0)',
              fontFamily: 'var(--mono)',
              fontSize: 14,
            }}
          />
        </div>

        <div
          ref={listRef}
          id={listboxId}
          role="listbox"
          aria-label="Command palette results"
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '6px 0',
            minHeight: 0,
          }}
        >
          {rows.map((row, idx) => {
            const isActive = idx === highlight;
            return (
              <div key={row.key}>
                {row.groupHeader && (
                  <div
                    className="text-3 mono"
                    style={{
                      padding: '10px 18px 4px',
                      fontSize: 10,
                      textTransform: 'uppercase',
                      letterSpacing: '0.12em',
                    }}
                  >
                    {row.groupHeader}
                  </div>
                )}
                <div
                  role="option"
                  aria-selected={isActive}
                  data-row-index={idx}
                  data-testid={`palette-row-${row.key}`}
                  onMouseEnter={() => setHighlight(idx)}
                  onClick={row.onActivate}
                  style={{
                    height: 40,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '0 16px',
                    cursor: 'pointer',
                    background: isActive ? 'var(--bg-2)' : 'transparent',
                    borderLeft: isActive
                      ? '2px solid var(--accent)'
                      : '2px solid transparent',
                  }}
                >
                  <span
                    className="mono"
                    style={{
                      fontSize: 13,
                      color: 'var(--text-0)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {row.label}
                  </span>
                  {row.hint && (
                    <span
                      className="text-2"
                      style={{
                        fontSize: 12,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flex: 1,
                        minWidth: 0,
                      }}
                    >
                      {row.hint}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div
          className="text-3"
          style={{
            padding: '8px 16px',
            borderTop: '1px solid var(--line)',
            fontSize: 11,
            display: 'flex',
            gap: 16,
            background: 'var(--bg-1)',
          }}
        >
          <span>
            <kbd className="mono">↑↓</kbd> navigate
          </span>
          <span>
            <kbd className="mono">↵</kbd> select
          </span>
          <span>
            <kbd className="mono">esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  );
};

const formatRunMeta = (run: RunMeta): string => {
  const parts: string[] = [];
  if (run.dataset) parts.push(run.dataset);
  if (typeof run.pass === 'number' && Number.isFinite(run.pass)) {
    parts.push(`pass ${(run.pass * 100).toFixed(1)}%`);
  }
  if (run.at) parts.push(run.at);
  return parts.join(' · ');
};

export default CommandPalette;
