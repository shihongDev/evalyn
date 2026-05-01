/**
 * Agent types.
 *
 * Mirrors the Phase 3 agent-runtime contract from spec §8 and §9. Lane C1
 * owns the backend; this is the frontend-side shape consumed by the store
 * and the (future) ChatPanel.
 */

export type AgentStatus = 'idle' | 'streaming' | 'awaiting_confirmation';

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ToolCall {
  /** CLI id or other tool name. */
  tool: string;
  /** Arguments dict. */
  args: Record<string, unknown>;
  /** Equivalent shell command for preview. */
  previewCmd: string;
  /** Status as the agent runtime advances. */
  status: 'proposed' | 'awaiting_confirmation' | 'running' | 'complete' | 'rejected' | 'error';
  /** Captured stdout once complete. */
  output?: string;
  /** Job id when the tool spawned a subprocess. */
  jobId?: string;
}

export interface FinalSuggestion {
  /** CLI id to open as a form on click. */
  cli: string;
  /** Pre-filled args. */
  args: Record<string, unknown>;
  /** Short rationale for the suggestion. */
  rationale: string;
}

export interface ChatMessage {
  /** Message id (server- or client-assigned). */
  id: string;
  /** Author role. */
  role: ChatRole;
  /** Plain-text content for text bubbles. */
  text?: string;
  /** Inline tool-call card payload. */
  toolCall?: ToolCall;
  /** Final-suggestion card payload. */
  finalSuggestion?: FinalSuggestion;
  /** Timestamp for ordering / display. */
  ts?: string;
}

export interface PendingConfirmation {
  /** Tool name awaiting confirmation. */
  tool: string;
  /** Arguments dict. */
  args: Record<string, unknown>;
  /** Equivalent shell command for preview. */
  previewCmd: string;
}

export interface AgentState {
  /** Server-assigned thread id. */
  threadId: string | null;
  /** Ordered history. */
  messages: ChatMessage[];
  /** Loop status. */
  status: AgentStatus;
  /** Set while waiting for the user to approve a non-allowlisted tool call. */
  pendingConfirmation: PendingConfirmation | null;
}

export interface ProviderState {
  /** Display name (e.g. "Anthropic"). */
  name: string;
  /** Whether the API key is set on disk. */
  hasKey: boolean;
  /** Selected model id. */
  model: string | null;
  /** Most recent test status. */
  testStatus?: 'untested' | 'ok' | 'error';
  /** Test error message, when applicable. */
  testError?: string;
}

export interface SettingsState {
  providers: Record<string, ProviderState>;
  active: string | null;
}
