import React, { useState } from 'react';
import { TraceList } from './components/TraceList';
import { MetricBuilder } from './components/MetricBuilder';
import { EvalDashboard } from './components/EvalDashboard';
import { SdkPlayground } from './components/SdkPlayground';
import { AgentTrace, Metric, EvaluationResult, MetricType } from './types';
import { Layers, Code2, LayoutDashboard } from 'lucide-react';

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<'playground' | 'dashboard'>('playground');
  
  const [traces, setTraces] = useState<AgentTrace[]>([]);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [results, setResults] = useState<EvaluationResult[]>([]);

  const activeTrace = traces.find(t => t.id === selectedTraceId) || null;

  const handleTraceGenerated = (trace: AgentTrace) => {
    setTraces(prev => [trace, ...prev]);
    setSelectedTraceId(trace.id);
    // Automatically switch to dashboard to see result if it's the first one
    if (traces.length === 0) {
        // Optional: show a toast notification instead of switching
        // setActiveView('dashboard'); 
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background text-slate-200 font-sans selection:bg-primary/30">
      {/* Top Navigation */}
      <div className="h-14 border-b border-slate-700 flex items-center px-6 bg-slate-900/80 backdrop-blur justify-between shrink-0 z-10">
        <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center shadow-lg shadow-primary/20">
                <Layers className="text-white" size={18} />
            </div>
            <span className="font-bold text-lg tracking-tight text-white">AutoEval</span>
            
            <div className="h-6 w-px bg-slate-700 mx-2"></div>

            <nav className="flex bg-slate-800 rounded-lg p-1">
                <button
                    onClick={() => setActiveView('playground')}
                    className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-medium transition-all ${activeView === 'playground' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <Code2 size={14} /> SDK Playground
                </button>
                <button
                    onClick={() => setActiveView('dashboard')}
                    className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-medium transition-all ${activeView === 'dashboard' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <LayoutDashboard size={14} /> Eval Dashboard
                    {traces.length > 0 && <span className="bg-primary text-white text-[10px] px-1.5 rounded-full">{traces.length}</span>}
                </button>
            </nav>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden relative">
        
        {/* View: SDK Playground */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${activeView === 'playground' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <SdkPlayground onTraceGenerated={handleTraceGenerated} />
        </div>

        {/* View: Dashboard */}
        <div className={`absolute inset-0 flex transition-opacity duration-300 ${activeView === 'dashboard' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <TraceList 
                traces={traces} 
                selectedTraceId={selectedTraceId} 
                onSelect={setSelectedTraceId} 
            />
            <MetricBuilder 
                selectedTrace={activeTrace}
                metrics={metrics}
                onAddMetric={(m) => setMetrics(prev => [...prev, m])}
                onUpdateMetric={(m) => setMetrics(prev => prev.map(pm => pm.id === m.id ? m : pm))}
            />
            <EvalDashboard 
                selectedTrace={activeTrace}
                metrics={metrics}
                results={results}
                onAddResult={(res) => setResults(prev => [...prev, res])}
                onUpdateMetric={(m) => setMetrics(prev => prev.map(pm => pm.id === m.id ? m : pm))}
            />
        </div>

      </div>
    </div>
  );
};

export default App;
