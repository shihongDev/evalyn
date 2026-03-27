# Deep Research: Evaluation Frameworks & Security

## DeepEval Deep Dive

**Metric categories (50+):**
- RAG: Faithfulness, Answer Relevancy, Contextual Precision/Recall/Relevancy
- Safety: Toxicity, Bias, PII Leakage
- Quality: Hallucination, Summarization, G-Eval (custom criteria)
- Agent: Tool Correctness, Task Completion (via trace analysis)
- Conversational: Knowledge Retention, Conversation Relevancy
- Multi-modal: Image/audio support in v3.0

**Synthesizer pipeline:** Generate -> Filter -> Evolve (Evol-Instruct) -> Style. Quality scoring on Clarity/Depth/Structure/Relevance.

**DeepTeam (separate framework):** 40+ vulnerability types, 10+ attack strategies. Separate from DeepEval proper.

**pytest plugin:** Auto-registered via `pytest11` entry point. `deepeval test run` discovers test files. Threshold-based pass/fail. CI integration via exit codes.

**Pricing:** Core: Apache 2.0 free. Confident AI cloud: Free tier, Starter $19.99/seat/month, Premium $79.99/seat/month.

## Ragas Deep Dive

**Faithfulness metric formula:** Decompose response into atomic claims -> verify each claim against context via NLI -> score = supported_claims / total_claims.

**Reference-free metrics:** Don't require ground truth. Core 4 (Faithfulness, Answer Relevancy, Context Precision, Context Recall) work without expected answers.

**Agent metrics:**
- ToolCallAccuracy: sequence + argument correctness
- ToolCallF1: unordered matching
- AgentGoalAccuracy: end-state vs expected outcome
- TopicAdherence: domain boundary enforcement

**TestsetGenerator:** Evolutionary generation with configurable question type distribution (simple 40%, reasoning 30%, multi-context 20%, conditional 10%). Uses knowledge graph extraction from source documents.

## PromptFoo Deep Dive

**Acquired by OpenAI March 9, 2026.** Remains open-source MIT. Integrating into OpenAI Frontier.

**Red teaming vulnerability categories:**
- Prompt injection (direct, indirect, encoded)
- Jailbreaks (DAN, role-play, multi-language)
- PII leakage, data exfiltration
- Hallucination, fabrication
- Toxicity, bias, hate speech
- Unauthorized tool use
- System prompt extraction

**Hydra multi-turn strategy:** Adaptive multi-turn jailbreak. Each turn builds on the previous, adjusting approach based on model's responses. Up to 195% higher ASR than single-turn.

**YAML config:**
```yaml
prompts:
  - "Translate {{input}} to French"
providers:
  - openai:gpt-4
  - anthropic:claude-3
tests:
  - vars: { input: "Hello" }
    assert:
      - type: contains
        value: "Bonjour"
      - type: llm-rubric
        value: "Translation is accurate and natural"
```

## Inspect AI Deep Dive

**Pipeline:** Dataset -> Task -> Solver -> Scorer. Each component is composable.

**Sandboxed execution:** Docker (default), Kubernetes, Proxmox, Modal. Critical for agent evals where models execute code.

**100+ evaluations:** Coding (HumanEval, MBPP, SWE-bench), Math (MATH, GSM8K), Safety (AdvBench, HarmBench), Knowledge (MMLU, ARC), Reasoning (Big-Bench, HellaSwag).

**VS Code extension:** Log viewer, task browser, inline result display.

## LLM Security Tools

**NeMo Guardrails - Colang DSL:**
```colang
define flow greeting
  user express greeting
  bot express greeting

define rail input check toxicity
  if input is toxic
    bot refuse and explain
```

5 rail types: input, dialog, retrieval, execution, output. Streaming mode for real-time guardrails.

**Guardrails AI validators:** 50+ built-in. Composable via Guard class. Types: format (JSON, regex), content (profanity, PII), semantic (relevance, faithfulness).

**Garak taxonomy:** 35+ probe modules across: hallucination, toxicity, data leakage, prompt injection, malware generation, XSS. 150+ individual probes, 3000+ test prompts.

## Governance

**EU AI Act Annex IV:** Most prescriptive evaluation documentation requirement globally. Mandates: dated/signed test logs, specific metrics, evaluation methodology documentation, benchmark descriptions, risk assessment.

**No universal model evaluation reporting standard exists yet.** This is a gap and opportunity.

## New Gaps for Evalyn

1. **Sandboxed agent evaluation** - Inspect AI's Docker sandbox is critical for safe agent evals where models execute code
2. **Knowledge graph-based test generation** - Ragas extracts knowledge graphs from source docs to generate evaluation questions
3. **Composable assertion framework** - PromptFoo's assertion types (contains, llm-rubric, similar, etc.) are a clean evaluation primitive
4. **Evaluation reporting standard** - No standard exists; evalyn could define one (JSON schema for evaluation results)

*Sources: DeepEval docs/GitHub, Ragas docs, PromptFoo docs, Inspect AI docs, NeMo Guardrails docs, Guardrails AI docs, Garak GitHub, EU AI Act*
