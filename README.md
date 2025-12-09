# Evalyn - AutoEval Framework

A comprehensive evaluation framework for LLM Agents, combining a Python SDK simulation with a powerful frontend dashboard for analysis, metric generation, and automatic prompt engineering.

## Features

- **SDK Playground**: 
  - Simulates the developer experience of using the `@eval` Python decorator.
  - Edit the `SYSTEM_PROMPT` and run individual or batch (10x) simulations to generate diverse test traces using Gemini.
  
- **Automated Trace Capture**: 
  - Records inputs, chain-of-thought, outputs, and latency.

- **Metric Builder**: 
  - **Auto-Suggest**: Uses Gemini to analyze traces and suggest bespoke evaluation metrics (Objective & Subjective).
  - **LLM Judge**: Automatically runs evaluation prompts against your agent's traces.

- **Human-in-the-Loop Calibration**:
  - Review LLM Judge scores alongside traces.
  - Provide "Human Ground Truth" labels.
  - **Auto-Optimize**: Uses Gemini to rewrite and improve the Judge's prompt based on discrepancies between the AI score and Human score.

## Quick Start

1. **Generate Data**:
   - Go to the **SDK Playground** tab.
   - Click **Run Batch (10x)** to generate diverse simulation data based on the code's System Prompt.

2. **Define Metrics**:
   - Switch to the **Eval Dashboard** tab.
   - Select a trace from the left sidebar.
   - Click **Auto-Suggest** in the middle panel to generate relevant metrics.

3. **Run Evaluation**:
   - Click **Run Evaluation** in the main dashboard view to score the selected trace.

4. **Calibrate**:
   - If you disagree with an AI score, add a **Human Label**.
   - If there is a disagreement, click **Calibrate Judge** to automatically optimize the evaluation prompt.

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS
- **AI Integration**: Google Gemini API (`@google/genai`)
- **Visualization**: Recharts, Lucide React

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
