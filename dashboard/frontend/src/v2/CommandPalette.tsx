/**
 * CommandPalette - Cmd/Ctrl+K modal for jumping anywhere in evalyn.
 *
 * Searches across four datasources in parallel on first open and reuses the
 * `useV2Resource` module-level cache so subsequent opens are free:
 *   - CLI commands  (`listCli`)
 *   - Runs          (`v2.experiments`)
 *   - Datasets      (`v2.datasets`)
 *   - Rubrics       (`v2.rubrics`)
 *
 * Each row is a typed `Entry` carrying its own `nav()` action, so Enter just
 * fires `entry.nav()` regardless of kind. Cmd+Enter on a command row keeps
 * the legacy "ask co-pilot about this command" flow.
 *
 * Ranking inside a query is exact-id > prefix > substring; within rank we
 * sort command > run > dataset > rubric and cap visible rows at 50 to keep
 * render cheap. With ~700 entries today an in-memory filter behind a single
 * useMemo is fine - no virtualization needed.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listCli, commandGroup, commandSummary } from './api/cli';
import type { CliSchema } from './api/cli';
import { v2 } from './api/client';
import type { ExperimentRow, DatasetCard, RubricRow } from './api/types';
import { openCliRunner } from './cliRunnerBridge';
import { E } from './tokens';
import { MOD_KEY } from './platform';

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

type EntryKind = 'command' | 'run' | 'dataset' | 'rubric';

interface BaseEntry {
  id: string;
  label: string;
  sublabel: string;
  /** Used to break ties in ranking (command > run > dataset > rubric). */
  kindOrder: number;
}

interface CommandEntry extends BaseEntry {
  kind: 'command';
  cli: CliSchema;
  nav: () => void;
}
interface RunEntry extends BaseEntry {
  kind: 'run';
  nav: () => void;
}
interface DatasetEntry extends BaseEntry {
  kind: 'dataset';
  nav: () => void;
}
interface RubricEntry extends BaseEntry {
  kind: 'rubric';
  nav: () => void;
}

type Entry = CommandEntry | RunEntry | DatasetEntry | RubricEntry;

const KIND_ORDER: Record<EntryKind, number> = {
  command: 0,
  run: 1,
  dataset: 2,
  rubric: 3,
};

const KIND_GLYPH: Record<EntryKind, string> = {
  command: '⌥',
  run: '◆',
  dataset: '◫',
  rubric: '◈',
};

const KIND_HEADER: Record<EntryKind, string> = {
  command: 'COMMANDS',
  run: 'RUNS',
  dataset: 'DATASETS',
  rubric: 'RUBRICS',
};

const MAX_VISIBLE = 50;
const DEFAULT_PER_KIND = 5;

// Module-level cache, parallel to useV2Resource's internal cache. Lives for
// the page session - so the second Cmd+K open is instant.
const _paletteCache: {
  cmds: CliSchema[] | null;
  runs: ExperimentRow[] | null;
  datasets: DatasetCard[] | null;
  rubrics: RubricRow[] | null;
} = { cmds: null, runs: null, datasets: null, rubrics: null };

function formatPass(pass: number | null): string {
  if (pass == null) return '-';
  return `${Math.round(pass)}%`;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [cmds, setCmds] = useState<CliSchema[] | null>(_paletteCache.cmds);
  const [runs, setRuns] = useState<ExperimentRow[] | null>(_paletteCache.runs);
  const [datasets, setDatasets] = useState<DatasetCard[] | null>(_paletteCache.datasets);
  const [rubrics, setRubrics] = useState<RubricRow[] | null>(_paletteCache.rubrics);
  const [errs, setErrs] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Scrollable results container; we steer it via scrollIntoView when
  // the active row leaves the viewport during arrow-key navigation.
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const navigate = useNavigate();

  // Lazy-load all four catalogs in parallel the first time the palette opens.
  // Each fetcher is independent: a failure on one shouldn't block the others.
  useEffect(() => {
    if (!open) return;
    const errors: string[] = [];
    const tasks: Promise<unknown>[] = [];
    if (!cmds) {
      tasks.push(
        listCli()
          .then((d) => {
            _paletteCache.cmds = d;
            setCmds(d);
          })
          .catch((e) => {
            errors.push(`commands: ${String(e)}`);
          }),
      );
    }
    if (!runs) {
      tasks.push(
        v2
          .experiments()
          .then((d) => {
            _paletteCache.runs = d;
            setRuns(d);
          })
          .catch((e) => {
            errors.push(`runs: ${String(e)}`);
          }),
      );
    }
    if (!datasets) {
      tasks.push(
        v2
          .datasets()
          .then((d) => {
            _paletteCache.datasets = d;
            setDatasets(d);
          })
          .catch((e) => {
            errors.push(`datasets: ${String(e)}`);
          }),
      );
    }
    if (!rubrics) {
      tasks.push(
        v2
          .rubrics()
          .then((d) => {
            _paletteCache.rubrics = d;
            setRubrics(d);
          })
          .catch((e) => {
            errors.push(`rubrics: ${String(e)}`);
          }),
      );
    }
    if (tasks.length === 0) return;
    void Promise.all(tasks).then(() => {
      if (errors.length) setErrs(errors);
    });
    // We intentionally only re-run when `open` flips - the cache prevents
    // duplicate fetches on subsequent opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Reset query + autofocus on every open. Also capture the element
  // that had focus before the palette opened so we can hand control
  // back to it on close - keyboard users who Cmd+K'd from a row in
  // ExperimentsList should land back on that row instead of <body>.
  useEffect(() => {
    if (!open) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    setQuery('');
    setActiveIndex(0);
    const id = setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      clearTimeout(id);
      // Skip body - focusing it strips the focus ring without giving
      // the user a meaningful target. Skip the palette's own input
      // which would re-trigger the modal's focus-trap on Safari.
      if (
        prevFocus &&
        prevFocus !== document.body &&
        prevFocus !== inputRef.current &&
        document.contains(prevFocus)
      ) {
        prevFocus.focus();
      }
    };
  }, [open]);

  const loading = !cmds || !runs || !datasets || !rubrics;

  /** Unified, source-agnostic entries list. Stable across query changes so the
   * search useMemo only re-runs when the inputs actually shift. */
  const allEntries = useMemo<Entry[]>(() => {
    const out: Entry[] = [];
    if (cmds) {
      for (const c of cmds) {
        const group = commandGroup(c);
        const summary = commandSummary(c);
        const sublabel = summary ? `${group} - ${summary}` : group;
        out.push({
          kind: 'command',
          kindOrder: KIND_ORDER.command,
          id: c.id,
          label: c.id,
          sublabel,
          cli: c,
          nav: () => {
            onClose();
            openCliRunner(c);
          },
        });
      }
    }
    if (runs) {
      for (const r of runs) {
        const sublabel = `${r.id} - ${formatPass(r.pass)} - ${r.when_iso}`;
        out.push({
          kind: 'run',
          kindOrder: KIND_ORDER.run,
          id: r.id,
          label: r.name,
          sublabel,
          nav: () => {
            onClose();
            navigate(`/experiments/${encodeURIComponent(r.id)}`);
          },
        });
      }
    }
    if (datasets) {
      for (const d of datasets) {
        const tagText = d.tags && d.tags.length ? d.tags.join(', ') : 'no tags';
        const sublabel = `${d.n} items - ${tagText}`;
        out.push({
          kind: 'dataset',
          kindOrder: KIND_ORDER.dataset,
          id: d.name,
          label: d.name,
          sublabel,
          nav: () => {
            onClose();
            navigate(`/datasets/${encodeURIComponent(d.name)}`);
          },
        });
      }
    }
    if (rubrics) {
      for (const ru of rubrics) {
        const sublabel = `${ru.kind} - ${ru.uses}x - ${ru.calibration_label}`;
        out.push({
          kind: 'rubric',
          kindOrder: KIND_ORDER.rubric,
          id: ru.id,
          label: ru.name,
          sublabel,
          nav: () => {
            onClose();
            navigate('/metrics');
          },
        });
      }
    }
    return out;
  }, [cmds, runs, datasets, rubrics, navigate, onClose]);

  /** Default state when the input is empty: a curated "recents" view. */
  const defaultEntries = useMemo<Entry[]>(() => {
    const byKind = (k: EntryKind) => allEntries.filter((e) => e.kind === k);
    const top = (arr: Entry[]) => arr.slice(0, DEFAULT_PER_KIND);

    const cmdsTop = top(byKind('command'));
    // Runs already arrive newest-first from /api/v2/experiments, but be
    // defensive in case the contract ever changes.
    const runsTop = top(
      byKind('run')
        .slice()
        .sort((a, b) => b.sublabel.localeCompare(a.sublabel)),
    );
    // Datasets: most recent eval first. last_used_iso lives only on the raw
    // DatasetCard, so map back through the source list.
    const datasetsTop: Entry[] = (() => {
      if (!datasets) return [];
      const sorted = datasets
        .slice()
        .sort((a, b) => (b.last_used_iso || '').localeCompare(a.last_used_iso || ''));
      const ids = new Set(sorted.slice(0, DEFAULT_PER_KIND).map((d) => d.name));
      return byKind('dataset').filter((e) => ids.has(e.id));
    })();
    // Rubrics: most-used first.
    const rubricsTop: Entry[] = (() => {
      if (!rubrics) return [];
      const sorted = rubrics.slice().sort((a, b) => b.uses - a.uses);
      const ids = new Set(sorted.slice(0, DEFAULT_PER_KIND).map((r) => r.id));
      return byKind('rubric').filter((e) => ids.has(e.id));
    })();
    return [...cmdsTop, ...runsTop, ...datasetsTop, ...rubricsTop];
  }, [allEntries, datasets, rubrics]);

  // Filtered list + total match count. We cap visible rows at MAX_VISIBLE
  // for cheap rendering, but expose the uncapped totalMatches so the
  // header can show "50 of 173" instead of just "50" - users with large
  // command/run libraries otherwise can't tell if they're seeing all
  // matches or just the first slice.
  const { filtered, totalMatches } = useMemo<{
    filtered: Entry[];
    totalMatches: number;
  }>(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return {
        filtered: defaultEntries.slice(0, MAX_VISIBLE),
        totalMatches: defaultEntries.length,
      };
    }

    type Scored = { e: Entry; rank: number };
    const scored: Scored[] = [];
    for (const e of allEntries) {
      const idLower = e.id.toLowerCase();
      const labelLower = e.label.toLowerCase();
      const subLower = e.sublabel.toLowerCase();
      let rank: number;
      if (idLower === needle) {
        rank = 0; // exact id match
      } else if (idLower.startsWith(needle) || labelLower.startsWith(needle)) {
        rank = 1; // prefix match
      } else if (
        idLower.includes(needle) ||
        labelLower.includes(needle) ||
        subLower.includes(needle)
      ) {
        rank = 2; // substring match
      } else {
        continue;
      }
      scored.push({ e, rank });
    }
    scored.sort((a, b) => {
      if (a.rank !== b.rank) return a.rank - b.rank;
      if (a.e.kindOrder !== b.e.kindOrder) return a.e.kindOrder - b.e.kindOrder;
      return a.e.label.localeCompare(b.e.label);
    });
    return {
      filtered: scored.slice(0, MAX_VISIBLE).map((s) => s.e),
      totalMatches: scored.length,
    };
  }, [query, allEntries, defaultEntries]);

  // Clamp the active index when the list changes.
  useEffect(() => {
    if (activeIndex >= filtered.length) setActiveIndex(0);
  }, [filtered.length, activeIndex]);

  // Keep the active row visible during arrow-key navigation. Without
  // this, arrowing past the bottom of the visible window leaves the
  // highlight off-screen and the user can't see which row is selected.
  // `block: 'nearest'` only scrolls when needed - rows already visible
  // don't trigger an unnecessary jump.
  useEffect(() => {
    const container = resultsRef.current;
    if (!container) return;
    const active = container.querySelector<HTMLElement>(
      `[data-palette-index="${activeIndex}"]`,
    );
    if (active) active.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  /** Group filtered results by kind for section headers, preserving the order
   * the search produced (command > run > dataset > rubric within rank). */
  const sections = useMemo(() => {
    const groups = new Map<EntryKind, Entry[]>();
    for (const e of filtered) {
      const list = groups.get(e.kind);
      if (list) list.push(e);
      else groups.set(e.kind, [e]);
    }
    // Render in canonical kind order regardless of search ordering, so
    // sections always appear top-to-bottom command > run > dataset > rubric.
    const order: EntryKind[] = ['command', 'run', 'dataset', 'rubric'];
    return order
      .map((kind) => ({ kind, entries: groups.get(kind) ?? [] }))
      .filter((s) => s.entries.length > 0);
  }, [filtered]);

  /** Flat index -> Entry, used for keyboard navigation across sections. */
  const flat = useMemo<Entry[]>(() => sections.flatMap((s) => s.entries), [sections]);

  /** Build a kind-specific co-pilot prompt for an entry. The co-pilot has
   * read access to all four datasources, so a question about any kind is
   * answerable; what differs is the most-useful default phrasing. */
  function askCoPilotPrompt(entry: Entry): string {
    if (entry.kind === 'command') {
      return `Run the \`${entry.id}\` command and explain the output.`;
    }
    if (entry.kind === 'run') {
      return `Summarize run \`${entry.id}\` (${entry.label}). What changed vs prior runs and where are the failures concentrated?`;
    }
    if (entry.kind === 'dataset') {
      return `Tell me about the \`${entry.id}\` dataset - size, coverage, recent runs, and observed failure patterns.`;
    }
    return `Explain rubric \`${entry.id}\` (${entry.label}) - what it measures, calibration status, and how often it's used.`;
  }

  function askCoPilot(entry: Entry) {
    onClose();
    navigate(`/copilot?prefill=${encodeURIComponent(askCoPilotPrompt(entry))}`);
  }

  function openCommandsPage() {
    onClose();
    navigate('/commands');
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => Math.min(flat.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const entry = flat[activeIndex];
      if (!entry) return;
      // Cmd/Ctrl+Enter routes any entry to the co-pilot pre-filled with a
      // question about it. Plain Enter still navigates / opens the runner.
      if (e.metaKey || e.ctrlKey) {
        askCoPilot(entry);
        return;
      }
      entry.nav();
    }
  }

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(26, 24, 18, 0.45)',
        backdropFilter: 'blur(2px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '12vh',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        style={{
          width: 600,
          maxWidth: 'calc(100vw - 32px)',
          maxHeight: '70vh',
          background: E.panel,
          border: `1px solid ${E.hair2}`,
          borderRadius: 12,
          boxShadow: '0 20px 60px rgba(0,0,0,0.25)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* SEARCH INPUT */}
        <div
          style={{
            padding: '14px 16px',
            borderBottom: `1px solid ${E.hair}`,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <span style={{ fontFamily: E.fMono, fontSize: 12, color: E.text3 }}>{MOD_KEY}K</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Jump to a command, run, dataset, or rubric..."
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: E.text0,
              fontSize: 14,
              fontFamily: E.fSans,
            }}
          />
          <span
            style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}
            title={
              totalMatches > MAX_VISIBLE
                ? `Showing the top ${MAX_VISIBLE} of ${totalMatches} matches. Refine the query to see more.`
                : undefined
            }
          >
            {totalMatches > MAX_VISIBLE
              ? `${flat.length} of ${totalMatches}`
              : flat.length}
          </span>
        </div>

        {/* RESULTS */}
        <div ref={resultsRef} style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {errs.length > 0 && (
            <div style={{ padding: 16, fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
              {errs.join(' / ')}
            </div>
          )}
          {loading && (
            <div style={{ padding: 16, fontSize: 13, color: E.text3 }}>Indexing...</div>
          )}
          {!loading && flat.length === 0 && (
            <div style={{ padding: 16, fontSize: 13, color: E.text3 }}>
              No matches. Try `run-eval`, a dataset name, or a metric id.
            </div>
          )}
          {!loading &&
            sections.map((section, sIdx) => {
              // Compute the running flat-index offset so each row knows whether
              // it's the active one for arrow-key nav.
              let offset = 0;
              for (let k = 0; k < sIdx; k += 1) offset += sections[k].entries.length;
              return (
                <div key={section.kind}>
                  <div
                    style={{
                      padding: '8px 16px 4px',
                      fontFamily: E.fMono,
                      fontSize: 10,
                      letterSpacing: 0.6,
                      color: E.text3,
                      textTransform: 'uppercase',
                      borderTop: sIdx ? `1px solid ${E.hair}` : 'none',
                      background: E.panel,
                    }}
                  >
                    {KIND_HEADER[section.kind]}
                  </div>
                  {section.entries.map((entry, i) => {
                    const flatIdx = offset + i;
                    const isActive = flatIdx === activeIndex;
                    return (
                      <button
                        key={`${entry.kind}:${entry.id}`}
                        type="button"
                        data-palette-index={flatIdx}
                        onClick={() => entry.nav()}
                        onMouseEnter={() => setActiveIndex(flatIdx)}
                        title={`${entry.label}${entry.sublabel ? ` - ${entry.sublabel}` : ''}`}
                        style={{
                          width: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 12,
                          padding: '10px 16px',
                          background: isActive ? E.panel2 : 'transparent',
                          border: 'none',
                          textAlign: 'left',
                          cursor: 'pointer',
                          color: E.text1,
                        }}
                      >
                        <span
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 14,
                            color: isActive ? E.ember : E.text3,
                            width: 16,
                            flexShrink: 0,
                            textAlign: 'center',
                          }}
                          aria-hidden="true"
                        >
                          {KIND_GLYPH[entry.kind]}
                        </span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div
                            style={{
                              fontFamily: entry.kind === 'command' ? E.fMono : E.fSans,
                              fontSize: 12.5,
                              color: E.text0,
                              fontWeight: 500,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {entry.label}
                          </div>
                          {entry.sublabel && (
                            <div
                              style={{
                                fontSize: 11,
                                color: E.text3,
                                marginTop: 2,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                fontFamily: entry.kind === 'run' ? E.fMono : E.fSans,
                              }}
                            >
                              {entry.sublabel}
                            </div>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              );
            })}
        </div>

        {/* FOOTER */}
        <div
          style={{
            padding: '10px 16px',
            borderTop: `1px solid ${E.hair}`,
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            fontSize: 11,
            color: E.text3,
            fontFamily: E.fMono,
          }}
        >
          <span>↑↓ navigate</span>
          <span>↵ open</span>
          <span>{MOD_KEY}↵ ask co-pilot</span>
          <span>esc close</span>
          <span style={{ flex: 1 }} />
          <button
            type="button"
            onClick={openCommandsPage}
            style={{
              background: 'transparent',
              border: 'none',
              color: E.ember,
              fontFamily: E.fMono,
              fontSize: 11,
              cursor: 'pointer',
              padding: 0,
            }}
          >
            Open Commands page →
          </button>
        </div>
      </div>
    </div>
  );
}
