import { GoogleGenAI, Type } from "@google/genai";
import { AgentTrace, Metric, MetricType, TraceType } from "../types";

const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

// 1. Suggest Metrics based on Traces
export const suggestMetrics = async (trace: AgentTrace): Promise<Metric[]> => {
  const ai = getAI();
  const prompt = `
    Analyze the following agent interaction trace and suggest 3 evaluation metrics.
    Some should be objective (checking format, length, specific keywords) and some subjective (tone, helpfulness, safety).
    
    Trace Input: ${trace.input}
    Trace Output: ${trace.output}
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name: { type: Type.STRING },
              description: { type: Type.STRING },
              type: { type: Type.STRING, enum: ["OBJECTIVE", "SUBJECTIVE"] },
              codeSnippet: { type: Type.STRING, description: "Pseudocode or JS for objective metrics" },
              judgePrompt: { type: Type.STRING, description: "Prompt for LLM judge for subjective metrics" },
              gradingScale: { type: Type.STRING, description: "e.g. 1-5 or Binary" }
            },
            required: ["name", "description", "type", "gradingScale"]
          }
        }
      }
    });
    
    const rawMetrics = JSON.parse(response.text || "[]");
    return rawMetrics.map((m: any, idx: number) => ({
      id: `sug-${Date.now()}-${idx}`,
      ...m
    }));
  } catch (error) {
    console.error("Error suggesting metrics:", error);
    return [];
  }
};

// 2. Run LLM Judge
export const runLLMJudge = async (trace: AgentTrace, metric: Metric): Promise<{ score: string | number; reasoning: string }> => {
  const ai = getAI();
  
  const prompt = `
    You are an AI Evaluator.
    
    TASK: Evaluate the following interaction based on the criteria provided.
    
    METRIC NAME: ${metric.name}
    CRITERIA/PROMPT: ${metric.judgePrompt}
    GRADING SCALE: ${metric.gradingScale}
    
    INTERACTION:
    User Input: ${trace.input}
    Agent Output: ${trace.output}
    
    Return your evaluation in JSON format with "score" and "reasoning".
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.STRING }, // Keeping string to handle "Pass"/"Fail" or numbers
            reasoning: { type: Type.STRING }
          },
          required: ["score", "reasoning"]
        }
      }
    });

    return JSON.parse(response.text || '{"score": "0", "reasoning": "Error parsing result"}');
  } catch (e) {
    return { score: "Error", reasoning: "Failed to run judge" };
  }
};

// 3. Optimize Prompt (Auto-Prompt Engineering)
export const optimizeJudgePrompt = async (
  metric: Metric,
  badExamples: { input: string; output: string; llmScore: string; humanScore: string; humanReasoning?: string }[]
): Promise<string> => {
  const ai = getAI();
  
  const examplesText = badExamples.map(ex => `
    [Example]
    Input: ${ex.input}
    Output: ${ex.output}
    Original Judge Score: ${ex.llmScore}
    Human Ground Truth: ${ex.humanScore}
    Human Reasoning: ${ex.humanReasoning || "N/A"}
  `).join("\n");

  const prompt = `
    You are an expert Prompt Engineer. 
    The current evaluation prompt for the metric "${metric.name}" is misaligned with human ratings.
    
    CURRENT PROMPT:
    ${metric.judgePrompt}
    
    MISALIGNMENT EXAMPLES:
    ${examplesText}
    
    TASK:
    Rewrite the "judgePrompt" to better capture the nuance shown in the human ground truth labels. 
    Analyze why the original prompt failed and create a more robust version.
    
    Return ONLY the new prompt text.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3-pro-preview", // Use a smarter model for reasoning about prompts
    contents: prompt
  });

  return response.text || metric.judgePrompt || "";
};

// 4. Simulate Agent (Generate Traces)
export const simulateAgentInteraction = async (systemPrompt: string, userScenario: string): Promise<AgentTrace> => {
  const ai = getAI();
  
  const chat = ai.chats.create({
    model: "gemini-2.5-flash",
    config: { systemInstruction: systemPrompt }
  });

  const startTime = Date.now();
  const response = await chat.sendMessage({ message: userScenario });
  const endTime = Date.now();

  return {
    id: `trace-${Date.now()}`,
    timestamp: Date.now(),
    input: userScenario,
    output: response.text || "",
    steps: [
      {
        type: TraceType.CHAIN_OF_THOUGHT,
        content: "Processing user intent...",
        durationMs: 50
      },
      {
        type: TraceType.FINAL_ANSWER,
        content: response.text || "",
        durationMs: endTime - startTime
      }
    ]
  };
};

// 5. Generate Test Inputs for Batch Simulation
export const generateTestInputs = async (systemPrompt: string, count: number): Promise<string[]> => {
  const ai = getAI();
  const prompt = `
    Generate ${count} diverse user inputs for an AI agent defined by the following system prompt.
    The inputs should vary in intent, complexity, and length.
    Include some edge cases, queries that are slightly out of scope, or adversarial inputs if appropriate for the persona to test robustness.
    
    System Prompt: ${systemPrompt}
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: { type: Type.STRING }
        }
      }
    });
    return JSON.parse(response.text || "[]");
  } catch (e) {
    console.error("Failed to generate test inputs", e);
    // Fallback if generation fails
    return Array(count).fill("Test Input (Fallback)");
  }
};
