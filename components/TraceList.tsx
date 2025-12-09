import React from 'react';
import { AgentTrace, TraceType } from '../types';
import { Clock, Terminal, CheckCircle2, AlertCircle, MessageSquare } from 'lucide-react';

interface TraceListProps {
  traces: AgentTrace[];
  selectedTraceId: string | null;
  onSelect: (id: string) => void;
}

export const TraceList: React.FC<TraceListProps> = ({ traces, selectedTraceId, onSelect }) => {
  return (
    <div className="flex flex-col h-full bg-surface border-r border-slate-700 w-80 overflow-hidden">
      <div className="p-4 border-b border-slate-700 bg-slate-900/50">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Recorded Traces</h2>
        <p className="text-xs text-slate-500 mt-1">@eval capture logs</p>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {traces.length === 0 && (
          <div className="text-center text-slate-500 text-sm mt-10 p-4">
            No traces found. Run a simulation or add quick start data.
          </div>
        )}
        {traces.map((trace) => (
          <div
            key={trace.id}
            onClick={() => onSelect(trace.id)}
            className={`p-3 rounded-lg cursor-pointer border transition-all ${
              selectedTraceId === trace.id
                ? 'bg-primary/20 border-primary text-white'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-750 hover:border-slate-600'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono bg-slate-900 px-1.5 py-0.5 rounded text-slate-500">
                {trace.id.slice(-6)}
              </span>
              <span className="text-xs flex items-center gap-1 opacity-70">
                <Clock size={10} />
                {new Date(trace.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-sm line-clamp-2 font-medium mb-1">
              <MessageSquare size={12} className="inline mr-1 text-sky-400" />
              {trace.input}
            </div>
            <div className="flex gap-2 mt-2">
                {trace.steps?.some(s => s.type === TraceType.TOOL_CALL) && (
                    <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded flex items-center gap-1">
                        <Terminal size={8} /> Tool
                    </span>
                )}
                 {trace.output ? (
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded flex items-center gap-1">
                        <CheckCircle2 size={8} /> Success
                    </span>
                 ) : (
                    <span className="text-[10px] bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded flex items-center gap-1">
                        <AlertCircle size={8} /> Error
                    </span>
                 )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
