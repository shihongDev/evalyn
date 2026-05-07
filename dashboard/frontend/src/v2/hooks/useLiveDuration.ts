/**
 * useLiveDuration - whole-second ticker driven by Date.now() so the
 * counter never drifts when a tab backgrounds (browsers throttle
 * setInterval down to ~1Hz/min). Used by the Recent Jobs drawer rows
 * and the CliRunner output header to show "running 12s" / "running 2m05s"
 * for queued/running jobs.
 *
 * Returns null when not live or when the timestamp does not parse so
 * the caller can render a falsy guard rather than "0s" placeholder.
 */

import { useEffect, useState } from 'react';

export function useLiveDuration(
  startedAtIso: string,
  live: boolean,
): string | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [live]);
  if (!live) return null;
  const t = Date.parse(startedAtIso);
  if (!Number.isFinite(t)) return null;
  const elapsedSec = Math.max(0, Math.floor((now - t) / 1000));
  return formatElapsed(elapsedSec);
}

function formatElapsed(sec: number): string {
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m${s.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h}h${mm.toString().padStart(2, '0')}m`;
}
