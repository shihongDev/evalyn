# Competitive Landscape: LLM Evaluation & Metrics

## Evaluation Frameworks

| Framework | Metrics | Unique Feature | Integration |
|---|---|---|---|
| DeepEval | 50+ multi-modal | DAGMetric (decision-tree scoring), red-teaming, arena comparison | Pytest, CI/CD |
| Ragas | RAG-specific (4 core + agent metrics) | Reference-free evaluation, composite score | Library plugin for Phoenix/Langfuse |
| PromptFoo | Security-focused | 50+ vulnerability types, NIST/MITRE presets | CLI, YAML |
| Inspect AI (UK AISI) | Safety benchmarks | Sandboxed execution, Bayesian evaluator reliability | Python, VS Code |
| EleutherAI Harness | 60+ academic benchmarks | De facto standard for model benchmarking | CLI, YAML tasks |
| OpenAI Evals | Model-graded templates | Cloud service with dashboard | API |
| Vertex AI | Rubric-based managed metrics | Pointwise + Pairwise modes | GCP SDK |

## LLM-as-Judge Research

| Approach | Key Technique | Relevance |
|---|---|---|
| MT-Bench | Swap-and-tie for position bias mitigation | Implement in pairwise comparisons |
| AlpacaEval | Length-controlled win rates via GLM | Debias length preference in judge |
| Arena-Hard | BenchBuilder - auto-curate hard prompts from usage | Mine production traces for eval cases |
| G-Eval | Logprob aggregation for continuous scores | Use when provider supports logprobs |

## 12 Known Judge Biases (CALM Framework)

Position bias, length bias, verbosity bias, self-enhancement bias, authority bias, bandwagon bias, compassion bias, attentional bias, format bias, cultural bias, egocentric bias, social desirability bias.

## Prompt Optimization State of Art

| Optimizer | Approach | In Evalyn? |
|---|---|---|
| DSPy MIPROv2 | Bootstrap traces + grounded proposals + discrete search | Yes |
| OPRO | Meta-prompt with solution history | Yes |
| APE | Generate candidates from demos, select by score | Yes |
| TextGrad | Textual backpropagation (Nature 2025) | Yes |
| PromptBreeder | Self-referential prompt evolution | Yes |
| EvoPrompt | Evolutionary operators on prompt population | Yes |
| SAMMO | Structural DAG mutations (not just text rewriting) | **No - gap** |
| PhaseEvo | Two-phase: global mutation + focused semantic refinement | **No - gap** |

## High-Priority Gaps for Evalyn

1. **Judge Calibration & Debiasing** - No framework integrates position-bias mitigation, length normalization, and small-sample human calibration together. Evalyn could be first.
2. **Agent Evaluation Metrics** - ToolCallAccuracy, ToolCallF1, AgentGoalAccuracy (from Ragas). Evalyn's trace infrastructure is ideal for this.
3. **Automatic Test Case Generation** - Anthropic's Bloom generates diverse scenarios for behaviors. Arena-Hard curates from real usage. Evalyn could do both.
4. **Statistical Rigor** - Confidence intervals, power analysis, bootstrap significance testing (Anthropic research).
5. **Red Teaming** - Lightweight adversarial testing (jailbreak resistance, prompt injection detection).
6. **DAGMetric** - Decision-tree evaluation approach from DeepEval.
7. **SAMMO-style structural optimization** - Treating prompts as DAGs with structural mutations.

## Where Evalyn Already Differentiates

- 7+ prompt optimizers in one framework (no competitor matches this breadth)
- Local-first, provider-agnostic (vs cloud-locked alternatives)
- Integrated trace + evaluate + calibrate loop
- 133 metrics (73 objective + 60 subjective)

*Sources: DeepEval docs/GitHub, Ragas docs, PromptFoo docs, Inspect AI docs, EleutherAI GitHub, OpenAI Evals, Vertex AI docs, MT-Bench/AlpacaEval/Arena-Hard/G-Eval papers, LLM-as-Judge survey papers, DSPy docs, TextGrad paper, SAMMO blog*
