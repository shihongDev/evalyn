import React, { useState } from 'react';
import { Metric, MetricType, AgentTrace } from '../types';
import { Sparkles, Code, Gavel, Save, RefreshCw } from 'lucide-react';
import { suggestMetrics } from '../services/geminiService';

interface MetricBuilderProps {
  selectedTrace: AgentTrace | null;
  metrics: Metric[];
  onAddMetric: (m: Metric) => void;
  onUpdateMetric: (m: Metric) => void;
}

export const MetricBuilder: React.FC<MetricBuilderProps> = ({ selectedTrace, metrics, onAddMetric, onUpdateMetric }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [editingMetric, setEditingMetric] = useState<Metric | null>(null);

  const handleAutoSuggest = async () => {
    if (!selectedTrace) return;
    setIsGenerating(true);
    const suggestions = await suggestMetrics(selectedTrace);
    suggestions.forEach(m => onAddMetric(m));
    setIsGenerating(false);
  };

  return (
    <div className="flex flex-col h-full bg-surface border-r border-slate-700 w-96">
      <div className="p-4 border-b border-slate-700 bg-slate-900/50 flex justify-between items-center">
        <div>
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Metrics Store</h2>
          <p className="text-xs text-slate-500 mt-1">Define logic or prompts</p>
        </div>
        <button
          onClick={handleAutoSuggest}
          disabled={!selectedTrace || isGenerating}
          className="flex items-center gap-1 px-3 py-1.5 bg-accent hover:bg-accent/90 text-white text-xs rounded-md disabled:opacity-50 transition-colors"
        >
            {isGenerating ? <RefreshCw className="animate-spin" size={14} /> : <Sparkles size={14} />}
            Auto-Suggest
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {metrics.length === 0 && (
            <div className="p-6 border border-dashed border-slate-700 rounded-xl text-center">
                <p className="text-slate-500 text-sm mb-2">No metrics defined.</p>
                <p className="text-xs text-slate-600">Select a trace and click Auto-Suggest to let Gemini build your evaluation suite.</p>
            </div>
        )}

        {metrics.map(metric => (
          <div key={metric.id} className="bg-slate-800 rounded-lg p-3 border border-slate-700 hover:border-slate-600 transition-colors">
            <div className="flex justify-between items-start mb-2">
              <div className="font-medium text-slate-200 text-sm flex items-center gap-2">
                {metric.type === MetricType.OBJECTIVE ? <Code size={14} className="text-blue-400"/> : <Gavel size={14} className="text-purple-400"/>}
                {metric.name}
              </div>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 border border-slate-600">
                {metric.type}
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-3">{metric.description}</p>
            
            {metric.type === MetricType.SUBJECTIVE && (
                <div className="bg-slate-900 p-2 rounded text-xs text-slate-300 font-mono mb-2 overflow-hidden text-ellipsis line-clamp-3">
                    <span className="text-purple-400 select-none">PROMPT: </span>
                    {metric.judgePrompt}
                </div>
            )}
            
             {metric.type === MetricType.OBJECTIVE && (
                <div className="bg-slate-900 p-2 rounded text-xs text-blue-300 font-mono mb-2 overflow-hidden text-ellipsis">
                    {metric.codeSnippet || "// No code snippet provided"}
                </div>
            )}

            <div className="flex justify-end">
                <button 
                    onClick={() => setEditingMetric(metric)}
                    className="text-xs text-slate-400 hover:text-white underline decoration-slate-600"
                >
                    Edit / Calibrate
                </button>
            </div>
          </div>
        ))}
      </div>

      {editingMetric && (
        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-8">
            <div className="bg-surface border border-slate-600 rounded-xl p-6 w-full max-w-2xl shadow-2xl">
                <h3 className="text-lg font-bold text-white mb-4">Edit Metric: {editingMetric.name}</h3>
                
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs font-semibold text-slate-400 mb-1">Description</label>
                        <input 
                            className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-white"
                            value={editingMetric.description}
                            onChange={(e) => setEditingMetric({...editingMetric, description: e.target.value})}
                        />
                    </div>
                    {editingMetric.type === MetricType.SUBJECTIVE ? (
                        <div>
                             <label className="block text-xs font-semibold text-slate-400 mb-1">LLM Judge Prompt</label>
                             <textarea 
                                className="w-full h-40 bg-slate-900 border border-slate-700 rounded p-2 text-sm font-mono text-slate-300 focus:border-primary outline-none"
                                value={editingMetric.judgePrompt}
                                onChange={(e) => setEditingMetric({...editingMetric, judgePrompt: e.target.value})}
                             />
                             <p className="text-[10px] text-slate-500 mt-1">Tip: Use placeholders like {'{{input}}'} and {'{{output}}'} in advanced mode.</p>
                        </div>
                    ) : (
                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1">Validation Code (JavaScript)</label>
                             <textarea 
                                className="w-full h-40 bg-slate-900 border border-slate-700 rounded p-2 text-sm font-mono text-blue-300"
                                value={editingMetric.codeSnippet}
                                onChange={(e) => setEditingMetric({...editingMetric, codeSnippet: e.target.value})}
                             />
                        </div>
                    )}
                </div>

                <div className="mt-6 flex justify-end gap-3">
                    <button 
                        onClick={() => setEditingMetric(null)}
                        className="px-4 py-2 rounded text-sm text-slate-400 hover:text-white"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={() => {
                            onUpdateMetric(editingMetric);
                            setEditingMetric(null);
                        }}
                        className="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded text-sm flex items-center gap-2"
                    >
                        <Save size={14} /> Save Changes
                    </button>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};
