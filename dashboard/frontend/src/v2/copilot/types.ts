/**
 * Co-pilot conversation model - synthesized from agent WS events.
 */

export interface ToolBlockEntry {
  tool_call_id: string;
  cmd: string;
  status: 'proposed' | 'awaiting_confirmation' | 'running' | 'complete' | 'error';
  duration_s: number | null;
  output_preview?: string;
}

export interface PendingConfirmation {
  tool_call_id: string;
  tool: string;
  preview_cmd: string;
  args: Record<string, unknown>;
  side_effects?: string[];
}

export interface ConvMessage {
  id: string;
  role: 'you' | 'agent';
  text: string;
  streaming: boolean;
  /** Inline tool blocks captured during this turn. */
  tools: ToolBlockEntry[];
  /** When set, an awaiting-confirmation card renders inside this bubble. */
  pending_confirm?: PendingConfirmation | null;
  ts: number;
}

export type AgentWsEvent =
  | {
      type: 'text_delta';
      thread_id: string;
      message_id: string;
      /** Field name matches the agent.py emitter (line 940). Not 'delta'. */
      text: string;
      ts: number;
    }
  | {
      type: 'tool_call_proposal';
      thread_id: string;
      tool_call_id: string;
      tool: string;
      args: Record<string, unknown>;
      preview_cmd?: string;
      ts: number;
    }
  | {
      type: 'tool_call_running';
      thread_id: string;
      tool_call_id: string;
      tool: string;
      job_id?: string;
      ts: number;
    }
  | {
      type: 'tool_call_complete';
      thread_id: string;
      tool_call_id: string;
      tool: string;
      ok: boolean;
      output?: string;
      stdout?: string;
      exit_code?: number | null;
      ts: number;
    }
  | {
      type: 'confirmation_required';
      thread_id: string;
      tool_call_id: string;
      tool: string;
      args: Record<string, unknown>;
      preview_cmd: string;
      side_effects?: string[];
      ts: number;
    }
  | {
      type: 'final';
      thread_id: string;
      message_id?: string;
      text?: string;
      reason?: string;
      ts: number;
    }
  | {
      type: 'error';
      thread_id: string;
      kind?: string;
      message: string;
      provider?: string;
      ts: number;
    };
