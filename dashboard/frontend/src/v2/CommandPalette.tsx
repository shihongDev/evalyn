/**
 * CommandPalette - Cmd/Ctrl+K modal for jumping into any evalyn CLI command.
 *
 * Loads `listCli()` once on mount, filters by id/group/blurb, and offers two
 * actions per row: "Ask co-pilot" (navigates to /copilot) and "Open Commands
 * page" at the bottom for the full grouped index.
 *
 * This is a plain fixed-position overlay - no portal lib. AppShell owns the
 * open/close state so the same instance persists across route changes.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listCli, commandGroup, commandSummary } from './api/cli';
import type { CliSchema } from './api/cli';
import { E } from './tokens';

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [cmds, setCmds] = useState<CliSchema[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const navigate = useNavigate();

  // Lazy-load the catalog the first time the palette opens (and keep it
  // cached for the rest of the session).
  useEffect(() => {
    if (!open || cmds || err) return;
    listCli()
      .then(setCmds)
      .catch((e) => setErr(String(e)));
  }, [open, cmds, err]);

  // Reset query + autofocus on every open.
  useEffect(() => {
    if (!open) return;
    setQuery('');
    setActiveIndex(0);
    // Defer focus until after the DOM paints the input.
    const id = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(id);
  }, [open]);

  const filtered = useMemo(() => {
    if (!cmds) return [];
    const needle = query.trim().toLowerCase();
    if (!needle) return cmds;
    return cmds.filter((c) => {
      if (c.id.toLowerCase().includes(needle)) return true;
      if (commandGroup(c).toLowerCase().includes(needle)) return true;
      if (commandSummary(c).toLowerCase().includes(needle)) return true;
      return false;
    });
  }, [cmds, query]);

  // Clamp the active index when the list changes.
  useEffect(() => {
    if (activeIndex >= filtered.length) setActiveIndex(0);
  }, [filtered.length, activeIndex]);

  function pickCommand(_id: string) {
    // TODO: prefill the composer once CoPilotThread reads ?prefill=
    // Lane 2 owns CoPilotThread.tsx; coordinate with them OR fall back to
    // anchor + hash. For v2 first cut we just navigate to /copilot.
    onClose();
    navigate('/copilot');
  }

  function openCommandsPage() {
    onClose();
    navigate('/commands');
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const cmd = filtered[activeIndex];
      if (cmd) pickCommand(cmd.id);
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
          <span style={{ fontFamily: E.fMono, fontSize: 12, color: E.text3 }}>⌘K</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands..."
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
          <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
            {filtered.length}
          </span>
        </div>

        {/* RESULTS */}
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {err && (
            <div style={{ padding: 16, fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
              {err}
            </div>
          )}
          {!cmds && !err && (
            <div style={{ padding: 16, fontSize: 13, color: E.text3 }}>Loading commands...</div>
          )}
          {cmds && filtered.length === 0 && (
            <div style={{ padding: 16, fontSize: 13, color: E.text3 }}>
              No commands match "{query}".
            </div>
          )}
          {filtered.map((cmd, i) => {
            const isActive = i === activeIndex;
            const summary = commandSummary(cmd);
            const group = commandGroup(cmd);
            return (
              <button
                key={cmd.id}
                type="button"
                onClick={() => pickCommand(cmd.id)}
                onMouseEnter={() => setActiveIndex(i)}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: '10px 16px',
                  background: isActive ? E.panel2 : 'transparent',
                  border: 'none',
                  borderTop: i ? `1px solid ${E.hair}` : 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  color: E.text1,
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 12.5,
                      color: E.text0,
                      fontWeight: 500,
                    }}
                  >
                    {cmd.id}
                  </div>
                  {summary && (
                    <div
                      style={{
                        fontSize: 11,
                        color: E.text3,
                        marginTop: 2,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {summary}
                    </div>
                  )}
                </div>
                <span
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 10,
                    color: isActive ? E.ember : E.text3,
                    flexShrink: 0,
                  }}
                >
                  {group}
                </span>
              </button>
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
          <span>↵ select</span>
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
