/**
 * RecentRunsStrip - horizontal strip of the last N runs of a given CLI.
 *
 * Sits above the form body in CliForm. Reads `runHistory` from the store
 * and surfaces 3-5 most recent runs of the SAME `cliId`. Click a card to
 * seed the form with that run's args via `editRunArgs` (which sets the
 * `activeFormSeed` channel that ActiveForm + the new CliForm both watch).
 *
 * Intentionally hidden when no runs exist for this CLI.
 */

import { useStore } from '../store';
import type { Job } from '../types/jobs';

interface RecentRunsStripProps {
  cliId: string;
  /** Cap on how many recent runs to surface. */
  limit?: number;
}

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'var(--text-3)',
  running: 'var(--info)',
  complete: 'var(--ok, #6fbf73)',
  failed: 'var(--fail)',
  cancelled: 'var(--text-3)',
};

/** "3m ago" / "1h ago" / "2d ago" — small, no dep. */
const formatAgo = (epochMs: number): string => {
  const now = Date.now();
  const sec = Math.max(0, Math.floor((now - epochMs) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
};

/**
 * Pick a "key arg" snippet for the card's headline. Path args are the most
 * recognisable (e.g. `evals/spam.jsonl`) so we prefer them; fall back to
 * the first non-empty arg value.
 */
const pickArgSnippet = (args: Record<string, unknown>): string | null => {
  for (const [k, v] of Object.entries(args)) {
    if (typeof v !== 'string' || !v) continue;
    if (k === 'dataset' || k === 'metrics' || v.includes('/') || v.includes('.')) {
      // Trim long paths to the basename for compactness.
      const slash = v.lastIndexOf('/');
      return slash >= 0 ? v.slice(slash + 1) : v;
    }
  }
  for (const v of Object.values(args)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v) && v.length === 0) continue;
    return String(v).slice(0, 32);
  }
  return null;
};

const RecentRunsStrip = ({ cliId, limit = 5 }: RecentRunsStripProps) => {
  const runHistory = useStore((s) => s.runHistory);
  const jobs = useStore((s) => s.jobs);
  const editRunArgs = useStore((s) => s.editRunArgs);
  const attachToChatInput = useStore((s) => s.attachToChatInput);
  // Subscribing to `presets` keeps the star fill-state in sync as the user
  // toggles stars or applies presets from the new SavedPresets strip above.
  // findPresetByArgs is read off the store too so we always reuse the same
  // key-equality logic the action helpers use internally.
  const presets = useStore((s) => s.presets);
  const addPreset = useStore((s) => s.addPreset);
  const removePreset = useStore((s) => s.removePreset);
  const findPresetByArgs = useStore((s) => s.findPresetByArgs);

  // Newest first, scoped to this CLI.
  const matching = runHistory
    .filter((r) => r.cliId === cliId)
    .reverse()
    .slice(0, limit);

  if (matching.length === 0) return null;

  return (
    <div style={{ marginBottom: 14 }}>
      <div
        className="text-3 mono"
        style={{
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          marginBottom: 6,
        }}
      >
        Recent {cliId} runs
      </div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        {matching.map((r) => {
          const job = jobs.get(r.jobId);
          const status = job?.status ?? 'pending';
          const snippet = pickArgSnippet(r.args);
          // Reuses the same sorted-key equality the store action uses, so
          // toggle behaviour stays in sync no matter where args originate.
          // `presets` is referenced so the selector subscription fires on
          // changes; the lookup itself runs through the store helper.
          void presets;
          const matchedPresetId = findPresetByArgs(cliId, r.args);
          const starred = matchedPresetId != null;
          return (
            <span
              key={r.id}
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
                onClick={() => editRunArgs(r.id)}
                title={`Re-fill form with args from ${r.id}`}
                data-recent-run-id={r.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 10px',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontSize: 11,
                  color: 'var(--text-1)',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget.parentElement as HTMLElement).style.background =
                    'var(--bg-3)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget.parentElement as HTMLElement).style.background =
                    'var(--bg-2)';
                }}
              >
                <span
                  aria-hidden
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 999,
                    background: STATUS_COLOR[status],
                    flexShrink: 0,
                  }}
                />
                <span className="text-2" style={{ fontSize: 10 }}>
                  {formatAgo(r.startedAt)}
                </span>
                {snippet && (
                  <span
                    style={{
                      color: 'var(--text-1)',
                      maxWidth: 160,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {snippet}
                  </span>
                )}
                {job?.exitCode !== undefined && status === 'complete' && (
                  <span className="text-3" style={{ fontSize: 10 }}>
                    exit {job.exitCode}
                  </span>
                )}
              </button>
              <button
                type="button"
                data-testid={`attach-recent-run-${r.id}`}
                aria-label="Attach run to chat"
                title="Attach this run to the chat composer"
                onClick={(e) => {
                  e.stopPropagation();
                  attachToChatInput({
                    id: `run:${r.id}`,
                    kind: 'run',
                    label: r.id,
                    ref: r.id,
                  });
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
                + chat
              </button>
              <button
                type="button"
                data-testid={`star-recent-run-${r.id}`}
                aria-label={starred ? 'Unstar preset' : 'Star as preset'}
                aria-pressed={starred}
                title={
                  starred
                    ? 'Remove from saved presets'
                    : 'Save as preset (one-click reapply)'
                }
                onClick={(e) => {
                  e.stopPropagation();
                  if (starred && matchedPresetId) {
                    removePreset(matchedPresetId);
                  } else {
                    addPreset(cliId, r.args);
                  }
                }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  borderLeft: '1px solid var(--line)',
                  padding: '0 8px',
                  cursor: 'pointer',
                  fontSize: 12,
                  color: starred ? 'var(--accent)' : 'var(--text-3)',
                  fontFamily: 'var(--mono)',
                }}
              >
                {starred ? '★' : '☆'}
              </button>
            </span>
          );
        })}
      </div>
    </div>
  );
};

export default RecentRunsStrip;
