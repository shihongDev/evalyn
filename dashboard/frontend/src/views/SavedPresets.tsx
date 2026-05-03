/**
 * SavedPresets - horizontal strip of starred presets for a given CLI.
 *
 * Sits ABOVE RecentRunsStrip in the form view. Each chip shows the auto-
 * derived label (or user-supplied override). Click a chip to reapply the
 * preset to the active form via `applyPreset`. A small hover-revealed `x`
 * removes the preset.
 *
 * Returns null when no presets exist for this CLI so the strip stays
 * invisible until the user has actually starred something.
 */

import { useState } from 'react';
import { useStore } from '../store';

interface SavedPresetsProps {
  cliId: string;
}

const SavedPresets = ({ cliId }: SavedPresetsProps) => {
  const presets = useStore((s) => s.presets);
  const applyPreset = useStore((s) => s.applyPreset);
  const removePreset = useStore((s) => s.removePreset);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const matching = presets.filter((p) => p.cliId === cliId);
  if (matching.length === 0) return null;

  return (
    <div data-testid="saved-presets-strip" style={{ marginBottom: 10 }}>
      <div
        className="text-3 mono"
        style={{
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        Presets
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {matching.map((p) => {
          const isHover = hoveredId === p.id;
          return (
            <span
              key={p.id}
              data-testid={`preset-chip-${p.id}`}
              onMouseEnter={() => setHoveredId(p.id)}
              onMouseLeave={() => setHoveredId((cur) => (cur === p.id ? null : cur))}
              style={{
                display: 'inline-flex',
                alignItems: 'stretch',
                background: 'var(--bg-2)',
                border: '1px solid var(--line)',
                borderRadius: 6,
                overflow: 'hidden',
              }}
            >
              <button
                type="button"
                onClick={() => applyPreset(p.id)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  removePreset(p.id);
                }}
                title={`Apply preset (right-click to remove): ${p.label}`}
                aria-label={`Apply preset ${p.label}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 10px',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontSize: 11,
                  color: 'var(--text-1)',
                  textAlign: 'left',
                }}
              >
                <span aria-hidden style={{ color: 'var(--accent)', fontSize: 11 }}>
                  ★
                </span>
                <span
                  style={{
                    maxWidth: 200,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {p.label}
                </span>
              </button>
              {isHover && (
                <button
                  type="button"
                  data-testid={`preset-remove-${p.id}`}
                  aria-label={`Remove preset ${p.label}`}
                  title="Remove this preset"
                  onClick={(e) => {
                    e.stopPropagation();
                    removePreset(p.id);
                  }}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    borderLeft: '1px solid var(--line)',
                    padding: '0 8px',
                    cursor: 'pointer',
                    fontSize: 11,
                    color: 'var(--text-3)',
                    fontFamily: 'var(--mono)',
                  }}
                >
                  x
                </button>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
};

export default SavedPresets;
