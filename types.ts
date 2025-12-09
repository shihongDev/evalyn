export enum TraceType {
  TOOL_CALL = 'TOOL_CALL',
  CHAIN_OF_THOUGHT = 'CHAIN_OF_THOUGHT',
  FINAL_ANSWER = 'FINAL_ANSWER',
  ERROR = 'ERROR'
}

export interface AgentTrace {
  id: string;
  timestamp: number;
  input: string;
  output: string;
  steps?: {
    type: TraceType;
    name?: string;
    content: string;
    durationMs: number;
  }[];
  metadata?: Record<string, any>;
}

export enum MetricType {
  OBJECTIVE = 'OBJECTIVE', // Code based (e.g., Regex, JSON valid)
  SUBJECTIVE = 'SUBJECTIVE' // LLM Judge based
}

export interface Metric {
  id: string;
  name: string;
  description: string;
  type: MetricType;
  // For Objective
  codeSnippet?: string; 
  // For Subjective
  judgePrompt?: string;
  gradingScale?: string; // e.g. "1-5" or "Pass/Fail"
}

export interface EvaluationResult {
  id: string;
  traceId: string;
  metricId: string;
  score: number | string;
  reasoning: string;
  isHumanLabel?: boolean; // If true, this is the ground truth
  humanNotes?: string;
  timestamp: number;
}

export interface SimulationConfig {
  scenario: string;
  agentSystemPrompt: string;
  turns: number;
}
