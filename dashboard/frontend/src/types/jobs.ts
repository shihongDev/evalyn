/**
 * Jobs types.
 *
 * Mirrors the Phase 1 dashboard job-manager contract from spec §4 and §9.
 * Lane A3 owns the backend implementation; this file is the frontend
 * counterpart consumed by the Zustand store.
 */

export type JobStatus = 'pending' | 'running' | 'complete' | 'failed' | 'cancelled';

export type JobLineKind = 'stdout' | 'stderr' | 'info' | 'prompt' | 'ok' | 'warn' | 'fail' | 'exit';

export interface JobLine {
  /** Stream tag for the line. */
  kind: JobLineKind;
  /** Raw text (may include ANSI sequences). */
  text: string;
  /** Server timestamp (ISO 8601). */
  ts?: string;
}

export interface Job {
  /** Server-assigned job id (e.g. `j-9282`). */
  id: string;
  /** Full command string for display. */
  cmd: string;
  /** CLI subcommand id, when known. */
  cliId?: string;
  /** Status updated by WS events. */
  status: JobStatus;
  /** Fractional progress 0..1, when reported. */
  progress?: number;
  /** Wall-clock start time (HH:MM:SS) for display. */
  started?: string;
  /** Epoch ms the job started — used for elapsed-time calc. */
  startedAt?: number;
  /** Final duration string once complete. */
  duration?: string;
  /** Estimated time remaining while running. */
  eta?: string;
  /** Exit code, populated on completion. */
  exitCode?: number;
  /** Captured output lines (cap enforced by backend). */
  lines?: JobLine[];
}

/**
 * Wire format for events streamed over `/ws/jobs/{id}`. The backend
 * (Lane B1) emits one of these per WS frame, JSON-encoded.
 */
export type JobWsEvent =
  | { type: 'stdout' | 'stderr' | 'info' | 'prompt' | 'ok' | 'warn' | 'fail'; line: string; ts?: string; event_id?: number }
  | { type: 'exit'; code: number; duration?: string; ts?: string; event_id?: number }
  | { type: 'progress'; progress: number; eta?: string; ts?: string; event_id?: number }
  | { type: 'truncated'; count: number; ts?: string; event_id?: number };

export interface Tab {
  /** Stable tab id (e.g. `cli:run-eval`, `job:j-9282`, `82dddcc3.run`). */
  id: string;
  /** Display title. */
  title: string;
  /** Tab kind drives the icon and content view. */
  kind: 'cli' | 'run' | 'yaml' | 'file' | 'job' | 'welcome';
  /** Optional CSS color for the icon dot. */
  tone?: string;
  /** Unsaved-state indicator. */
  dirty?: boolean;
}

export interface FileNode {
  /** Display name (filename or folder name). */
  name: string;
  /** Folder vs file. */
  kind: 'folder' | 'file';
  /** Children for folders. */
  children?: FileNode[];
  /** Optional metadata string shown on the right. */
  meta?: string;
  /** Whether this node should start expanded. */
  expand?: boolean;
  /** Indicators ported from mock data. */
  active?: boolean;
  live?: boolean;
  warn?: boolean;
  fail?: boolean;
}

export interface RunMeta {
  /** Run id (8-char prefix). */
  id: string;
  /** Dataset path. */
  dataset: string;
  /** Pass-rate 0..1. */
  pass: number;
  /** Delta vs prior run, in pass-rate units. */
  delta?: number;
  /** Whether this run regressed vs prior. */
  regression?: boolean;
  /** Human-friendly timestamp. */
  at?: string;
}
