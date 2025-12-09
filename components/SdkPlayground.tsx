import React, { useState } from 'react';
import { Play, Terminal, RefreshCw, Layers } from 'lucide-react';
import { AgentTrace } from '../types';
import { simulateAgentInteraction, generateTestInputs } from '../services/geminiService';

interface SdkPlaygroundProps {
  onTraceGenerated: (trace: AgentTrace) => void;
}

const DEFAULT_PYTHON_CODE = `import os
from autoeval import eval
from google.genai import GoogleGenAI

# Initialize Client
# API_KEY is injected from environment
ai = GoogleGenAI(api_key=os.getenv("API_KEY"))

# --- CONFIGURATION ---
SYSTEM_PROMPT = """
You are a concise technical support bot for a cloud database company.
You only answer SQL related questions.
If the user asks about anything else, politely decline.
"""

@eval
def process_user_request(user_input: str):
    """
    Main entry point for the agent.
    The @eval decorator automatically traces inputs, outputs, 
    and latency for this function.
    """
    print(f"Processing request: {user_input}")
    
    response = ai.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_input,
        config={
            'system_instruction': SYSTEM_PROMPT
        }
    )
    
    return response.text
`;

export const SdkPlayground: React.FC<SdkPlaygroundProps> = ({ onTraceGenerated }) => {
  const [code, setCode] = useState(DEFAULT_PYTHON_CODE);
  const [userInput, setUserInput] = useState("How do I perform a left join?");
  const [logs, setLogs] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  // Helper to extract system prompt from the "Python" code to make the simulation real
  const extractSystemPrompt = (codeString: string) => {
    const match = codeString.match(/SYSTEM_PROMPT\s*=\s*"""([\s\S]*?)"""/);
    return match ? match[1].trim() : "You are a helpful assistant.";
  };

  const handleRun = async () => {
    setIsRunning(true);
    setLogs(prev => [...prev, `> process_user_request("${userInput}")`]);
    
    try {
      const systemPrompt = extractSystemPrompt(code);
      setLogs(prev => [...prev, `[System] Loaded prompt: "${systemPrompt.substring(0, 30)}..."`]);
      setLogs(prev => [...prev, `[AutoEval] @eval tracing started...`]);

      const trace = await simulateAgentInteraction(systemPrompt, userInput);
      
      setLogs(prev => [...prev, `[Agent] Output: ${trace.output}`]);
      setLogs(prev => [...prev, `[AutoEval] Trace captured: ${trace.id}`]);
      setLogs(prev => [...prev, `[AutoEval] stored to ./data/traces/${trace.id}.json`]);
      
      onTraceGenerated(trace);
    } catch (e) {
      setLogs(prev => [...prev, `[Error] Execution failed: ${e}`]);
    }
    
    setIsRunning(false);
  };

  const handleBatchRun = async () => {
    setIsRunning(true);
    setLogs(prev => [...prev, `> run_batch_simulation(n=10)`]);

    try {
        const systemPrompt = extractSystemPrompt(code);
        setLogs(prev => [...prev, `[AutoEval] Generating 10 diverse test inputs based on system prompt...`]);

        const inputs = await generateTestInputs(systemPrompt, 10);
        
        for (const [index, input] of inputs.entries()) {
             setLogs(prev => [...prev, `\n[${index + 1}/10] Input: "${input}"`]);
             
             // Short delay for visual pacing
             await new Promise(r => setTimeout(r, 400));
             
             const trace = await simulateAgentInteraction(systemPrompt, input);
             setLogs(prev => [...prev, `[Agent] Output: ${trace.output.substring(0, 50)}...`]);
             onTraceGenerated(trace);
        }

        setLogs(prev => [...prev, `\n[AutoEval] Batch complete. 10 traces captured.`]);

    } catch (e) {
        setLogs(prev => [...prev, `[Error] Batch failed: ${e}`]);
    }

    setIsRunning(false);
  };

  return (
    <div className="flex h-full text-slate-300 font-mono text-sm">
      {/* Code Editor Panel */}
      <div className="flex-1 flex flex-col border-r border-slate-700 bg-[#0d1117]">
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-2">
            <span className="text-blue-400 font-bold">agent.py</span>
            <span className="text-xs text-slate-500">Python 3.10</span>
          </div>
          <button 
            onClick={() => setCode(DEFAULT_PYTHON_CODE)}
            className="text-xs flex items-center gap-1 text-slate-500 hover:text-white"
          >
            <RefreshCw size={12}/> Reset
          </button>
        </div>
        <textarea
          className="flex-1 w-full bg-[#0d1117] text-slate-300 p-4 resize-none focus:outline-none leading-relaxed"
          spellCheck={false}
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />
      </div>

      {/* Execution Terminal */}
      <div className="w-1/3 flex flex-col bg-[#0f172a]">
        {/* Input Area */}
        <div className="p-4 border-b border-slate-700">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Terminal size={14} /> SDK Test Runner
          </h3>
          <div className="space-y-3">
            <div>
                <label className="block text-xs text-slate-500 mb-1">user_input (str)</label>
                <input
                    type="text"
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                />
            </div>
            
            <div className="grid grid-cols-2 gap-2">
                <button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white py-2 rounded font-medium transition-colors disabled:opacity-50 text-xs"
                >
                    {isRunning ? <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"/> : <Play size={14} fill="currentColor" />}
                    Run Single
                </button>
                <button
                    onClick={handleBatchRun}
                    disabled={isRunning}
                    className="flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white py-2 rounded font-medium transition-colors disabled:opacity-50 text-xs"
                >
                    <Layers size={14} />
                    Run Batch (10x)
                </button>
            </div>
          </div>
        </div>

        {/* Console Output */}
        <div className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1">
            {logs.length === 0 && <span className="text-slate-600 italic">Ready to execute...</span>}
            {logs.map((log, i) => (
                <div key={i} className={`${log.startsWith('>') ? 'text-blue-400 font-bold mt-2' : log.startsWith('[Error]') ? 'text-red-400' : 'text-slate-400'}`}>
                    {log}
                </div>
            ))}
            {isRunning && <div className="text-slate-500 animate-pulse">_</div>}
        </div>
      </div>
    </div>
  );
};
