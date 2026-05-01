/**
 * Terminal — streaming output panel for one or more jobs.
 *
 * Renders the merged JobLine stream from `store.jobEvents` (optionally
 * filtered to a single job id). Each line passes through the inline
 * ANSI parser so SGR color codes show up as colored spans.
 *
 * Auto-scroll behavior (per spec §8): the panel scrolls to the bottom on
 * each new line UNLESS the user has scrolled up. We track that via the
 * standard scrollTop + clientHeight + scrollHeight check with an 8px
 * tolerance so accidental wheel-jitter doesn't pin the view mid-stream.
 *
 * If no `jobId` is supplied, the Terminal merges every active job's
 * lines in chronological order (the Bottom-panel default view). The
 * default mode also shows a `$ ` prompt cursor at the bottom for
 * mock parity (see /tmp/evalyn-dashboard-mock/wb-app.jsx:454).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../store';
import type { JobLine } from '../types/jobs';
import { parseAnsi } from '../lib/ansi';

const KIND_FALLBACK_COLOR: Record<JobLine['kind'], string> = {
  prompt: 'var(--accent)',
  info: 'var(--text-1)',
  ok: 'var(--pass)',
  warn: 'var(--warn)',
  fail: 'var(--fail)',
  stderr: 'var(--fail)',
  stdout: 'var(--text-1)',
  exit: 'var(--text-2)',
};

interface AnsiLineProps {
  line: JobLine;
}

const AnsiLine = ({ line }: AnsiLineProps) => {
  const fallback = KIND_FALLBACK_COLOR[line.kind] ?? 'var(--text-1)';
  const spans = parseAnsi(line.text);
  return (
    <div style={{ color: fallback, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
      {line.kind === 'prompt' && <span className="text-3">$ </span>}
      {spans.length === 0 ? (
        <span>{line.text || ' '}</span>
      ) : (
        spans.map((s, i) => (
          <span
            key={i}
            style={{
              color: s.color,
              fontWeight: s.bold ? 600 : undefined,
              opacity: s.faint ? 0.7 : undefined,
            }}
          >
            {s.text}
          </span>
        ))
      )}
    </div>
  );
};

interface TerminalProps {
  /** When set, only render this job's lines. Otherwise show all jobs. */
  jobId?: string;
}

const SCROLL_TOLERANCE_PX = 8;

const Terminal = ({ jobId }: TerminalProps) => {
  const jobEvents = useStore((s) => s.jobEvents);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [pinned, setPinned] = useState(true);

  const lines = useMemo<JobLine[]>(() => {
    if (jobId) return jobEvents.get(jobId) ?? [];
    // Merge all jobs' buffers. We don't have global ordering across jobs
    // so we just concatenate by insertion order of the Map. This matches
    // the typical "1 job at a time" usage the bottom panel sees.
    const out: JobLine[] = [];
    for (const buf of jobEvents.values()) {
      for (const line of buf) out.push(line);
    }
    return out;
  }, [jobEvents, jobId]);

  // Scroll to bottom whenever new lines arrive AND the user is pinned.
  useEffect(() => {
    if (!pinned) return;
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines.length, pinned]);

  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const distance = el.scrollHeight - (el.scrollTop + el.clientHeight);
    setPinned(distance < SCROLL_TOLERANCE_PX);
  };

  return (
    <div
      ref={containerRef}
      onScroll={onScroll}
      className="mono"
      data-testid="terminal-scroll"
      data-pinned={pinned ? '1' : '0'}
      style={{
        fontSize: 12,
        lineHeight: 1.7,
        height: '100%',
        overflow: 'auto',
      }}
    >
      {lines.length === 0 ? (
        <div className="text-3" style={{ fontSize: 11 }}>
          {'// no output yet — run a CLI to see streamed stdout/stderr'}
        </div>
      ) : (
        lines.map((line, i) => <AnsiLine key={i} line={line} />)
      )}
      {!jobId && (
        <div style={{ marginTop: 8, color: 'var(--accent)' }}>
          ${' '}
          <span
            style={{
              background: 'var(--text-1)',
              color: 'var(--bg-0)',
              padding: '0 1px',
            }}
          >
            &nbsp;
          </span>
        </div>
      )}
    </div>
  );
};

export default Terminal;
