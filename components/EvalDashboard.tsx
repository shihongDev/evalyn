import React, { useState } from 'react';
import { AgentTrace, Metric, EvaluationResult, MetricType } from '../types';
import { runLLMJudge, optimizeJudgePrompt } from '../services/geminiService';
import { Play, UserCheck, AlertTriangle, Wand2, BarChart3, Check, X } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface EvalDashboardProps {
  selectedTrace: AgentTrace | null;
  metrics: Metric[];
  results: EvaluationResult[];
  onAddResult: (res: EvaluationResult) => void;
  onUpdateMetric: (m: Metric) => void;
}

export const EvalDashboard: React.FC<EvalDashboardProps> = ({ 
  selectedTrace, 
  metrics, 
  results, 
  onAddResult, 
  onUpdateMetric 
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [humanScoreInput, setHumanScoreInput] = useState<Record<string, string>>({});

  // Filter results for the selected trace
  const traceResults = results.filter(r => r.traceId === selectedTrace?.id);

  // Run automatic evaluation
  const handleRunEval = async () => {
    if (!selectedTrace) return;
    setIsRunning(true);

    for (const metric of metrics) {
      // Skip if result already exists (naive caching for demo)
      // if (traceResults.find(r => r.metricId === metric.id)) continue;

      let score: string | number = 0;
      let reasoning = "";

      if (metric.type === MetricType.SUBJECTIVE) {
        const res = await runLLMJudge(selectedTrace, metric);
        score = res.score;
        reasoning = res.reasoning;
      } else {
        // Simple mock for objective metrics since we can't eval raw JS safely
        if (metric.codeSnippet?.includes("length")) {
           score = selectedTrace.output.length > 50 ? 1 : 0;
           reasoning = "Length check passed";
        } else {
           score = 1;
           reasoning = "Objective check passed (simulated)";
        }
      }

      onAddResult({
        id: `res-${Date.now()}-${metric.id}`,
        traceId: selectedTrace.id,
        metricId: metric.id,
        score,
        reasoning,
        timestamp: Date.now()
      });
    }
    setIsRunning(false);
  };

  // Add Human Ground Truth
  const handleAddHumanLabel = (metricId: string, value: string) => {
    onAddResult({
      id: `human-${Date.now()}`,
      traceId: selectedTrace!.id,
      metricId,
      score: value,
      reasoning: "Human annotation",
      isHumanLabel: true,
      timestamp: Date.now()
    });
    setHumanScoreInput(prev => ({ ...prev, [metricId]: '' }));
  };

  // Optimize Prompt
  const handleOptimizePrompt = async (metric: Metric) => {
    setIsOptimizing(true);
    
    // Find discrepancies
    // In a real app, this would query ALL traces, not just the selected one.
    // We will build a small mock list from the current view for demonstration
    const metricResults = traceResults.filter(r => r.metricId === metric.id);
    const machineResult = metricResults.find(r => !r.isHumanLabel);
    const humanResult = metricResults.find(r => r.isHumanLabel);

    if (machineResult && humanResult && machineResult.score !== humanResult.score) {
       const badExample = {
           input: selectedTrace!.input,
           output: selectedTrace!.output,
           llmScore: String(machineResult.score),
           humanScore: String(humanResult.score),
           humanReasoning: humanResult.reasoning
       };

       const newPrompt = await optimizeJudgePrompt(metric, [badExample]);
       onUpdateMetric({ ...metric, judgePrompt: newPrompt });
       alert(`Prompt Optimized!\n\nNew Prompt:\n${newPrompt.substring(0, 100)}...`);
    } else {
        alert("Need a discrepancy between AI and Human scores to optimize.");
    }

    setIsOptimizing(false);
  };

  if (!selectedTrace) {
      return <div className="flex-1 flex items-center justify-center text-slate-500">Select a trace to view details.</div>
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background">
      {/* Header */}
      <div className="p-6 border-b border-slate-700">
         <div className="flex justify-between items-start">
             <div>
                <h1 className="text-xl font-bold text-white mb-2">Evaluation Workspace</h1>
                <div className="text-slate-400 font-mono text-xs bg-slate-900 p-2 rounded max-w-2xl border border-slate-800">
                    <span className="text-primary font-bold">INPUT:</span> {selectedTrace.input}<br/>
                    <span className="text-success font-bold">OUTPUT:</span> {selectedTrace.output.slice(0, 200)}...
                </div>
             </div>
             <button
                onClick={handleRunEval}
                disabled={isRunning || metrics.length === 0}
                className="flex items-center gap-2 bg-primary hover:bg-primary/90 text-white px-5 py-2.5 rounded-lg font-medium transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
             >
                {isRunning ? <span className="animate-spin">⏳</span> : <Play size={16} fill="currentColor" />}
                Run Evaluation
             </button>
         </div>
      </div>

      {/* Results Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {metrics.map(metric => {
            const mResults = traceResults.filter(r => r.metricId === metric.id);
            const aiScore = mResults.find(r => !r.isHumanLabel);
            const humanScore = mResults.find(r => r.isHumanLabel);

            return (
                <div key={metric.id} className="bg-surface border border-slate-700 rounded-xl overflow-hidden shadow-sm">
                    {/* Metric Header */}
                    <div className="px-4 py-3 bg-slate-800/50 border-b border-slate-700 flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${metric.type === MetricType.OBJECTIVE ? 'bg-blue-400' : 'bg-purple-400'}`}></span>
                            <h3 className="font-semibold text-slate-200">{metric.name}</h3>
                        </div>
                        {metric.type === MetricType.SUBJECTIVE && aiScore && humanScore && aiScore.score !== humanScore.score && (
                             <button 
                                onClick={() => handleOptimizePrompt(metric)}
                                disabled={isOptimizing}
                                className="text-xs bg-amber-500/20 text-amber-300 px-3 py-1 rounded-full border border-amber-500/30 hover:bg-amber-500/30 flex items-center gap-1 transition-colors"
                             >
                                <Wand2 size={12} />
                                {isOptimizing ? 'Optimizing...' : 'Calibrate Judge'}
                             </button>
                        )}
                    </div>

                    {/* Scores Area */}
                    <div className="p-4 grid grid-cols-12 gap-6">
                        {/* AI Section */}
                        <div className="col-span-8">
                            {aiScore ? (
                                <div>
                                    <div className="flex items-center gap-3 mb-2">
                                        <div className="text-sm text-slate-400 uppercase font-bold tracking-wider">AI Score</div>
                                        <div className="text-2xl font-bold text-white">{aiScore.score}</div>
                                    </div>
                                    <div className="text-sm text-slate-400 bg-slate-900/50 p-3 rounded border border-slate-800">
                                        {aiScore.reasoning}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-slate-600 italic text-sm py-4">Not evaluated yet.</div>
                            )}
                        </div>

                        {/* Human Section */}
                        <div className="col-span-4 border-l border-slate-700 pl-6">
                            <div className="text-sm text-slate-400 uppercase font-bold tracking-wider mb-2 flex items-center gap-2">
                                <UserCheck size={14} /> Human Label
                            </div>
                            
                            {humanScore ? (
                                <div>
                                     <div className="text-2xl font-bold text-sky-400 mb-1">{humanScore.score}</div>
                                     <div className="text-xs text-slate-500">Corrected ground truth</div>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <input 
                                        type="text" 
                                        placeholder="Enter score (e.g. 5)"
                                        className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-white focus:border-sky-500 outline-none"
                                        value={humanScoreInput[metric.id] || ''}
                                        onChange={(e) => setHumanScoreInput({...humanScoreInput, [metric.id]: e.target.value})}
                                    />
                                    <button 
                                        onClick={() => handleAddHumanLabel(metric.id, humanScoreInput[metric.id])}
                                        disabled={!humanScoreInput[metric.id]}
                                        className="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs py-1.5 rounded transition-colors"
                                    >
                                        Save Label
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            );
        })}

        {metrics.length > 0 && traceResults.length > 0 && (
             <div className="h-64 bg-surface border border-slate-700 rounded-xl p-4">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">Score Distribution</h4>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metrics.map(m => {
                        const r = traceResults.find(res => res.metricId === m.id && !res.isHumanLabel);
                        return { name: m.name, score: r ? Number(r.score) || 0 : 0 };
                    })}>
                         <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                         <YAxis stroke="#64748b" fontSize={10} />
                         <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                            itemStyle={{ color: '#e2e8f0' }}
                         />
                         <Bar dataKey="score" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                            {metrics.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'][index % 4]} />
                            ))}
                         </Bar>
                    </BarChart>
                </ResponsiveContainer>
             </div>
        )}
      </div>
    </div>
  );
};
