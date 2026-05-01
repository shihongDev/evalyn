/**
 * Agent types.
 *
 * Mirrors the Phase 3 agent-runtime contract from spec §8 and §9. Lane C1
 * owns the backend; this is the frontend-side shape consumed by the store
 * and the ChatPanel / SettingsModal.
 */

export type AgentStatus = 'idle' | 'streaming' | 'awaiting_confirmation' | 'error';

export type ChatRole = 'user' | 'assistant' | 'system';

/** Status of a single tool call as it advances through the agent loop. */
export type ToolCallStatus =
  | 'proposed'
  | 'awaiting_confirmation'
  | 'running'
  | 'complete'
  | 'rejected'
  | 'error';

export interface ToolCall {
  /** Server-side id for this tool invocation; lets later events update the same card. */
  id: string;
  /** CLI id or other tool name. */
  tool: string;
  /** Arguments dict. */
  args: Record<string, unknown>;
  /** Equivalent shell command for preview. */
  previewCmd: string;
  /** Status as the agent runtime advances. */
  status: ToolCallStatus;
  /** Captured stdout once complete. */
  output?: string;
  /** Job id when the tool spawned a subprocess. */
  jobId?: string;
  /** Error message when status === 'error' or 'rejected'. */
  error?: string;
}

export interface FinalSuggestion {
  /** Display label shown on the card. */
  label: string;
  /** CLI id to open as a form on click. */
  cliId: string;
  /** Pre-filled args. */
  args: Record<string, unknown>;
}

/** Agent error payload — typically a provider 401, rate limit, or budget exceeded. */
export interface AgentError {
  /** Short error category: 'auth' | 'rate_limit' | 'budget' | 'network' | 'unknown'. */
  kind: string;
  /** Human-readable message. */
  message: string;
  /** Provider id when relevant (e.g. 'openai'). */
  provider?: string;
}

export interface ChatMessage {
  /** Message id (server- or client-assigned). */
  id: string;
  /** Author role. */
  role: ChatRole;
  /** Plain-text content for text bubbles. May grow as text_delta events arrive. */
  text?: string;
  /** Inline tool-call card payload. */
  toolCall?: ToolCall;
  /** Final-suggestion card payload(s). */
  finalSuggestions?: FinalSuggestion[];
  /** Timestamp for ordering / display. */
  ts?: string;
  /** Set when the assistant message is still being streamed. */
  streaming?: boolean;
}

export interface PendingConfirmation {
  /** Tool-call id (matches the ToolCall.id on the message that produced it). */
  toolCallId: string;
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
  /** Most recent agent / provider error (cleared on next successful turn). */
  error: AgentError | null;
}

/* ---------------------------------------------------------------------------
 * WebSocket wire format. The agent runtime (Lane C1) emits one of these per
 * frame on `/ws/agent/{thread_id}`. Use the `type` field as a tagged-union
 * discriminator on the frontend.
 * ------------------------------------------------------------------------- */

export type AgentWsEvent =
  | AgentTextDeltaEvent
  | AgentToolCallProposalEvent
  | AgentToolCallRunningEvent
  | AgentToolCallCompleteEvent
  | AgentConfirmationRequiredEvent
  | AgentErrorEvent
  | AgentFinalEvent;

export interface AgentTextDeltaEvent {
  type: 'text_delta';
  /** Server-assigned message id; deltas with the same id append to the same bubble. */
  message_id: string;
  /** Newly-appended text fragment. */
  delta: string;
  ts?: string;
}

export interface AgentToolCallProposalEvent {
  type: 'tool_call_proposal';
  tool_call_id: string;
  tool: string;
  args: Record<string, unknown>;
  preview_cmd: string;
  ts?: string;
}

export interface AgentToolCallRunningEvent {
  type: 'tool_call_running';
  tool_call_id: string;
  /** Job id when the tool spawned a subprocess. */
  job_id?: string;
  ts?: string;
}

export interface AgentToolCallCompleteEvent {
  type: 'tool_call_complete';
  tool_call_id: string;
  /** Captured stdout / structured output from the tool. */
  output: string;
  /** Non-zero exit codes flip the card to the error tone. */
  exit_code?: number;
  ts?: string;
}

export interface AgentConfirmationRequiredEvent {
  type: 'confirmation_required';
  tool_call_id: string;
  tool: string;
  args: Record<string, unknown>;
  preview_cmd: string;
  ts?: string;
}

export interface AgentErrorEvent {
  type: 'error';
  kind: string;
  message: string;
  provider?: string;
  ts?: string;
}

export interface AgentFinalEvent {
  type: 'final';
  /** Optional message id of the assistant turn we're closing. */
  message_id?: string;
  /** Final summary text. */
  text: string;
  /** Optional list of suggestion cards. */
  suggestions?: { label: string; cli_id: string; args: Record<string, unknown> }[];
  ts?: string;
}

/* ---------------------------------------------------------------------------
 * Settings / providers
 * ------------------------------------------------------------------------- */

export type ProviderId = 'openai' | 'anthropic' | 'ollama' | string;

export type ProviderTestStatus = 'untested' | 'testing' | 'ok' | 'error';

export interface ProviderState {
  /** Provider id (e.g. 'openai'). */
  id: ProviderId;
  /** Display name (e.g. "OpenAI"). */
  name: string;
  /** Whether the API key is set on disk. Server never returns plaintext. */
  hasKey: boolean;
  /** Whether the API key field is required (false for local providers like Ollama). */
  requiresKey: boolean;
  /** Selected model id. */
  model: string | null;
  /** Most recent test status. */
  testStatus: ProviderTestStatus;
  /** Test error message, when applicable. */
  testError?: string;
  /** Cached model list (loaded on demand from /api/settings/models/{provider}). */
  models?: string[];
  /** Whether the model list is currently being fetched. */
  loadingModels?: boolean;
}

export interface SettingsState {
  /** Provider state keyed by id. */
  providers: Record<string, ProviderState>;
  /** Currently active provider id (server enforces one-active invariant). */
  active: string | null;
}
