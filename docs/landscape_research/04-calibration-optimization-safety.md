# Competitive Landscape: Calibration, Prompt Optimization & Safety

Research date: 2026-03-27

---

## 1. Judge Reliability & Calibration Platforms

### 1.1 Braintrust - Loop & AutoEvals

**Loop (NL Scorer Generation):**
- Loop generates custom scorers from plain-language descriptions of failure modes
- Non-technical teammates can draft scorers by describing what "bad" looks like in natural language
- Loop then generates scorer code, builds evaluation datasets from production logs, and optimizes prompts using real user examples
- Calibration is iterative: after initial runs, teams refine definitions, add new scorers, and rerun on expanded datasets
- Loop also analyzes existing prompts and generates better-performing versions automatically

**AutoEvals (open-source scorer library):**
- Preconfigured LLM-as-a-judge scorers: Factuality, ClosedQA, Battle, Relevance, Safety, Summarization
- Factuality scorer: reads input, output, and optional reference document, assigns 0-1 score with written rationale
- ClosedQA: evaluates answer quality purely based on input and output (no reference needed)
- Battle: pairwise comparison returning 1 (better) or 0 (worse)
- All scorers use consistent API: accept input/output/expected, return score + metadata

**Key insight:** Braintrust's differentiator is the production-to-evaluation feedback loop. Low-scoring production traces automatically become dataset candidates for future evaluation runs.

Sources:
- https://www.braintrust.dev/docs/reference/autoevals
- https://github.com/braintrustdata/autoevals
- https://www.braintrust.dev/articles/how-to-eval

---

### 1.2 LangSmith - Annotation Queues & Self-Improving Evaluators

**Annotation Queue Pipeline:**
1. Production runs get flagged (manually or by low automated scores)
2. Flagged runs enter annotation queues assigned to subject-matter experts
3. Experts review, label, and optionally add comments explaining their judgment
4. Corrections feed back into automated evaluation via few-shot examples

**Align Evals Feature (self-improving judges):**
- When a human corrects an LLM evaluator's score, the correction is stored as a few-shot example
- Future evaluator runs automatically include these corrections in the prompt
- Provides a playground-like interface to iterate on evaluator prompts
- Shows side-by-side comparison of human-graded vs. LLM-generated scores
- Tracks "alignment score" over time to measure judge-human agreement
- Eliminates manual prompt engineering for judge calibration

**Calibration Loop:**
1. Run LLM-as-judge on a batch
2. Human reviews disagreements, makes corrections
3. Corrections stored as few-shot examples in a corrections dataset
4. Next evaluator run includes corrections, improving alignment
5. Track alignment score to measure convergence

**Key insight:** LangSmith's approach is the most systematic for judge calibration. The corrections-as-few-shot-examples pattern is elegant because it requires no retraining - just prompt augmentation.

Sources:
- https://www.langchain.com/articles/llm-as-a-judge
- https://changelog.langchain.com/announcements/self-improving-llm-evaluators-in-langsmith
- https://blog.langchain.com/introducing-align-evals/
- https://docs.smith.langchain.com/evaluation/how_to_guides/create_few_shot_evaluators

---

### 1.3 W&B Weave - HITL Rubric Improvement

- Domain experts review traces and provide feedback directly in the Weave UI
- Feedback is used to iteratively refine automated evaluation rubrics
- Integrates human evaluation seamlessly into the W&B dashboard
- The feedback loop is crucial for iteratively improving LLM outputs
- Rubric formulation is treated as a first-class concern: revisions make rubrics more explicit and easier to follow for both humans and LLMs
- Combines automated metrics with human-in-the-loop review for edge case adjudication

**Key insight:** Weave's approach is less formalized than LangSmith's Align Evals but benefits from tight integration with the broader W&B experiment tracking ecosystem (model registry, leaderboards).

Sources:
- https://docs.wandb.ai/weave
- https://wandb.ai/site/evaluations/
- https://www.zenml.io/llmops-database/building-robust-llm-evaluation-frameworks-w-b-s-evaluation-driven-development-approach

---

### 1.4 Patronus AI - Lynx Hallucination Detection Model

**Architecture:**
- Fine-tuned from Llama-3-70B-Instruct (also available as 8B variant)
- Specialized for hallucination detection in RAG systems
- Open-source, available on HuggingFace

**Training Methodology:**
- Training data: mix of CovidQA, PubmedQA, DROP, RAGTruth datasets
- Combines hand-annotated and synthetic data
- Uses a perturbation process to construct specialized training/evaluation datasets for hallucination identification
- The perturbation approach creates controlled hallucination examples by modifying factual content
- Trained with mixed precision and flash attention on 8x NVIDIA H100 GPUs

**Benchmark: HaluBench:**
- Open-sourced benchmark sourced from real-world domains
- Assesses faithfulness in LLM responses comprehensively
- Lynx-70B achieved 87.4% accuracy on HaluBench
- Outperforms GPT-4o and Claude-3-Sonnet on hallucination detection

**Lynx v1.1 (August 2024):**
- Updated 8B model with improved RAG hallucination detection
- Maintained performance while being more efficient

**Key insight:** Patronus demonstrates that specialized fine-tuned models (even 8B) can outperform general-purpose frontier models on specific evaluation tasks like hallucination detection. This validates the approach of training domain-specific judge models.

Sources:
- https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
- https://www.databricks.com/blog/patronus-ai-lynx
- https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct
- https://www.marktechpost.com/2024/08/01/patronus-ai-releases-lynx-v1-1-an-8b-state-of-the-art-rag-hallucination-detection-model/

---

### 1.5 Academic Papers on Calibrating LLM Judges (2024-2026)

**Major Surveys:**

| Paper | Date | Key Contribution |
|---|---|---|
| "A Survey on LLM-as-a-Judge" (arXiv:2411.15594) | Nov 2024 | Framework for building reliable judge systems: consistency, bias mitigation, adaptation |
| "LLMs-as-Judges: Comprehensive Survey" (arXiv:2412.05579) | Dec 2024 | Five perspectives: Functionality, Methodology, Applications, Meta-evaluation, Limitations |
| "Opportunities and Challenges of LLM-as-a-judge" (EMNLP 2025) | 2025 | Industry-focused analysis of deployment challenges |

**Known Biases in LLM Judges:**

| Bias Type | Description | Mitigation |
|---|---|---|
| Position bias | Preference for first/last option in pairwise comparison | Swap positions and require consistent judgment across both orderings |
| Verbosity bias | Favoring longer responses | Score content quality independently of length; some studies find this is less severe than assumed after controlling for quality gap |
| Self-preference bias | Models prefer their own outputs | Use different model as judge than generator |
| Authority bias | Influenced by claimed credentials | Strip metadata from evaluated content |
| Format bias | Preference for structured/formatted responses | Normalize formatting before evaluation |

**CALM Framework:**
- Automated bias quantification framework
- Identifies 12 key potential biases in LLM judges
- Uses automated and principle-guided modification to measure each bias type

**Calibration Techniques from Literature:**
- Multiple evidence calibration
- Balanced position calibration (swap-and-verify)
- Human-in-the-loop calibration
- Aggregation strategies (majority voting among heterogeneous judge models)
- Probability/prompt-level calibration that penalizes verbosity and unwarranted confidence
- JudgeLM: enhances eval capability via reference support and reference drop training paradigms
- CritiqueLLM: multi-path prompting combining pointwise-to-pairwise and referenced-to-reference-free strategies

Sources:
- https://arxiv.org/abs/2411.15594
- https://arxiv.org/abs/2412.05579
- https://llm-judge-bias.github.io/
- https://aclanthology.org/2025.emnlp-main.138.pdf

---

## 2. Prompt Optimization Services

### 2.1 DSPy MIPROv2

**Architecture - three phases:**

1. **Bootstrap Stage:**
   - Runs program many times across different inputs
   - Collects traces of input/output behavior for each module
   - Filters traces to keep only those in trajectories scored highly by the metric

2. **Grounded Proposal Stage:**
   - Instruction proposer receives: summary of training dataset properties, summary of program code, bootstrapped few-shot examples, and randomly sampled generation tips ("be creative", "be concise") to explore the instruction feature space
   - Produces multiple instruction candidates per module

3. **Search Stage (Bayesian Optimization):**
   - Uses Bayesian optimization to find best combinations of instructions and demonstrations
   - Runs trials evaluating new prompt sets against validation data
   - Optimizes jointly across all predictors in the program

**Configuration Modes:**

| Mode | Candidates | Val Set Size | Compute |
|---|---|---|---|
| auto="light" | fewer | smaller | minimal |
| auto="medium" | 12 | up to 300 | moderate |
| auto="heavy" | more | larger | intensive |

**Key Results:**
- MIPROv2 is currently the most principled prompt optimization approach
- Joint optimization of instructions + demonstrations outperforms optimizing either alone
- Bayesian optimization is more sample-efficient than random search or evolutionary approaches

Sources:
- https://dspy.ai/api/optimizers/MIPROv2/
- https://dspy.ai/learn/optimization/optimizers/
- https://www.langtrace.ai/blog/grokking-miprov2-the-new-optimizer-from-dspy
- https://deepwiki.com/stanfordnlp/dspy/4.4-miprov2:-instruction-and-parameter-optimization

---

### 2.2 Other Prompt Optimization Frameworks

**Promptomatix (Salesforce, July 2025):**
- Transforms natural language task descriptions into high-quality prompts
- Supports lightweight meta-prompt-based optimizer and DSPy-powered compiler
- Analyzes user intent, generates synthetic training data, selects prompting strategies
- Refines prompts using cost-aware objectives (balances quality vs. API cost)
- arXiv:2507.14241

**promptolution (December 2025):**
- Unified modular framework implementing four established prompt optimizers:
  - OPRO (LLM-as-optimizer, Yang et al. 2024)
  - EvoPromptDE (differential evolution)
  - EvoPromptGA (genetic algorithm)
  - CAPO (current SOTA discrete prompt optimizer, Zehle et al. 2025)
- Enables fair comparison across optimization approaches

**Evidently AI (open-source, 2025):**
- Automated prompt optimization built into Evidently Python library
- Specifically targets LLM judge prompt generation
- Process: label dataset with expert annotations, optionally add comments, optimizer generates variants, evaluates on dataset, keeps best performers
- Reported result: raised accuracy from 64% to 96% in seconds vs. manual iteration

**POaaS - Prompt Optimization as a Service (2026):**
- Minimal-edit approach: optimizes prompts with smallest possible changes
- Targets on-device small LLMs specifically
- Aims to lift accuracy while cutting hallucinations

**Key comparison:**

| Framework | Approach | Strengths | Weaknesses |
|---|---|---|---|
| DSPy MIPROv2 | Bayesian optimization | Joint instruction+demo optimization | Requires many LLM calls |
| OPRO | LLM-as-optimizer | Simple, model-agnostic | Single-shot proposals |
| EvoPrompt | Evolutionary algorithms (GA/DE) | Good exploration | Slow convergence |
| CAPO | Center-aware textual gradients | Current SOTA discrete | Newest, less battle-tested |
| Promptomatix | Meta-prompt + DSPy hybrid | Cost-aware, accessible | Salesforce ecosystem |
| Evidently | Self-improvement loop | Targets judge prompts specifically | Narrower scope |

Sources:
- https://arxiv.org/abs/2507.14241
- https://arxiv.org/html/2512.02840v1
- https://www.evidentlyai.com/blog/automated-prompt-optimization
- https://arxiv.org/html/2603.16045

---

### 2.3 Commercial Prompt Management Platforms

**Humanloop (shutting down September 2025):**
- Prompt version control, evaluation workflows, collaborative review
- Strong non-technical collaboration UI
- Users migrating to PromptLayer and others

**PromptLayer:**
- Prompt Registry: version control for all prompts
- Evaluations: automated testing for prompt quality
- A/B Testing: split traffic between prompt versions by percentage or user segment
- Observability: complete logging and analytics for LLM interactions
- No-code interface for non-technical team members

**PromptPerfect (Jina AI):**
- Prompt optimization for GPT-4, ChatGPT, Midjourney
- Generates and refines prompts in seconds
- Consumer-oriented, less suited for production pipelines

**Key insight:** The prompt management market is consolidating. Humanloop's shutdown signals that prompt versioning alone is not enough to sustain a business - evaluation and optimization must be integrated.

Sources:
- https://blog.promptlayer.com/humanloop-shutdown-guide-to-migrating-your-prompts-and-evals-to-promptlayer/
- https://docs.promptlayer.com/why-promptlayer/ab-releases
- https://promptperfect.jina.ai/

---

## 3. Confidence Estimation

### 3.1 Meta AI's DeepConf

**Core Idea:**
- Uses model-internal confidence signals to filter out low-quality reasoning traces during or after generation
- Parallel thinking method: generates many reasoning paths, uses confidence to select the best ones

**Confidence Metrics:**

| Metric | How it Works | Best For |
|---|---|---|
| Group confidence | Calculates confidence across different token segments | Segment-level quality |
| Tail confidence | Focuses on final portion of reasoning trace | Answer-proximal quality |
| Lowest group confidence | Identifies single least-confident segment ("weakest link") | Detecting critical errors |

**Two Operating Modes:**

| Mode | Description | Efficiency |
|---|---|---|
| Offline thinking | Generate all traces first, then filter/weight by confidence before final vote | Higher accuracy |
| Online thinking | Evaluate in real-time; stop trace if confidence drops below threshold | Lower token cost |

**Key Results:**
- 99.9% accuracy on AIME 2025 with GPT-OSS-120B (vs. 97.0% with standard majority voting)
- Reduces generated tokens by up to 84.7% compared to standard thinking approaches
- DeepSeek-8B: +5.8 percentage points accuracy while using 77.9% fewer tokens
- First AI method to achieve 99.9% on AIME 2025 with open-source models

Sources:
- https://venturebeat.com/ai/metas-deepconf-offers-a-dial-to-balance-llm-reasoning-cost-and-accuracy
- https://www.marktechpost.com/2025/08/27/meta-ai-introduces-deepconf-first-ai-method-to-achieve-99-9-on-aime-2025-with-open-source-models-using-gpt-oss-120b/
- https://jiaweizzhao.github.io/deepconf/
- https://arxiv.org/pdf/2508.15260

---

### 3.2 Verbalized Confidence vs. Token Logprobs

**Token Logprobs:**
- Well-calibrated in multiple-choice and yes/no settings
- Requires access to model internals (not available for most black-box APIs)
- Measures token-level fluency, not necessarily answer correctness
- Available with OpenAI API (logprobs parameter), not available with Anthropic

**Verbalized Confidence:**
- Ask the model to state its confidence as a number (e.g., "Rate your confidence 0-100")
- LLMs tend to be highly overconfident when verbalizing confidence
- Performance is highly sensitive to how confidence is elicited (prompt wording matters)
- Small LLMs favor simple prompt formulations; large LLMs use different strategies
- Recent finding (2025): cached representations explain substantial variance in verbal confidence beyond token log-probabilities - suggests verbal confidence is more than just fluency readout
- Demonstrates "sophisticated metacognitive capacity" involving retrieval of internal representations

**Consistency Sampling (Self-Consistency):**
- Sample N responses, approximate confidence via frequency of parsed answers
- CoT-Consistency: use chain-of-thought prompting for each sample
- Does not require model internals
- Most expensive approach (requires N forward passes)
- Generally well-calibrated but costly

**Comparison:**

| Method | Calibration | Cost | API Access Needed | Black-box Compatible |
|---|---|---|---|---|
| Token logprobs | Good (constrained formats) | 1x | Logprobs endpoint | No |
| Verbalized confidence | Poor (overconfident) | 1x | Standard API | Yes |
| Consistency sampling | Good | Nx | Standard API | Yes |
| DeepConf (group confidence) | Excellent | Nx but prunable | Logprobs endpoint | No |

Sources:
- https://arxiv.org/html/2603.17839v1
- https://openreview.net/pdf?id=CVRdNQvFPE
- https://arxiv.org/pdf/2306.13063
- https://ericjinks.com/blog/2025/logprobs/

---

### 3.3 New Confidence Estimation Methods (2025-2026)

**Confidence-Informed Self-Consistency (CISC) - ACL 2025 Findings:**
- Weighted majority vote using confidence scores from the model itself
- Outperforms vanilla self-consistency in nearly all configurations
- Reduces required reasoning paths by over 40% on average
- Introduces within-question confidence evaluation (better predictor than across-question metrics)

**Confidence Enhanced Reasoning (CER) - ACL 2025:**
- Training-free method
- Key insight: "process confidence" during reasoning is more valuable than confidence in the final answer
- Measures confidence at intermediate reasoning steps, not just the conclusion

**Self-Certainty Metric:**
- Measures divergence of predicted token distribution from uniform distribution
- Higher divergence = more peaked/certain prediction
- No additional inference cost, single forward pass

**ReflectiveConf:**
- Incorporates reflect-and-correct procedure to improve reasoning quality
- Surpasses self-consistency in both accuracy and efficiency
- Online self-correction guided by confidence signals

**Confidence-Aware Adaptive Sampling:**
- Dynamically decides whether to continue or terminate generation
- Uses sentence-level numeric and linguistic features capturing confidence, uncertainty, and temporal dynamics
- Accepts paths deemed likely correct early; requires multiple runs only for low-confidence paths

Sources:
- https://arxiv.org/abs/2502.06233
- https://datasciocean.com/en/paper-intro/cer/
- https://openreview.net/pdf?id=29FRqmVQK8
- https://arxiv.org/pdf/2512.18605
- https://arxiv.org/html/2603.08999

---

## 4. Red Teaming & Safety

### 4.1 PromptFoo Vulnerability Scanning

**Overview:** CLI and library for evaluating and red-teaming LLM apps. As of March 2026, PromptFoo has been acquired by OpenAI.

**Vulnerability Categories (50+ types):**

| Category | Specific Types |
|---|---|
| Prompt injection | Direct injection, indirect injection (via RAG/external content) |
| Jailbreaking | DAN-mode, multi-turn escalation, encoding attacks |
| PII exposure | Direct PII disclosure, cross-session PII leaks, social engineering |
| Toxicity | Hate speech, harassment, sexual content, self-harm |
| Data exfiltration | Training data extraction, RAG context leakage |
| Tool/Agent exploits | Unauthorized data access, privilege escalation, hijacking |
| Compliance | OWASP LLM Top 10, custom organizational policies |
| Hallucination | Factual errors, fabricated citations, invented entities |

**Red Team Strategies:**
- Hydra: adaptive multi-turn conversations with persistent scan-wide memory; pivots across conversation branches to uncover stateful vulnerabilities
- Automated generation of adversarial inputs targeting specific vulnerability classes
- CI/CD integration for continuous security assessment

**OWASP Alignment:** Prompt injection is #1 on OWASP 2025 LLM risk list. PromptFoo covers all OWASP LLM Top 10 categories.

Sources:
- https://www.promptfoo.dev/docs/red-team/llm-vulnerability-types/
- https://www.promptfoo.dev/docs/red-team/strategies/
- https://www.promptfoo.dev/docs/red-team/owasp-llm-top-10/
- https://github.com/promptfoo/promptfoo

---

### 4.2 Anthropic's Red Teaming Methodology

**Multi-layered Approach:**

1. **Domain Expert Teaming:** Subject matter experts in CBRN, cybersecurity, autonomous AI evaluate model capabilities within their area. 150+ hours with top biosecurity experts evaluating harmful biological information generation.

2. **Frontier Threats Evaluation:** Focuses on CBRN (Chemical, Biological, Radiological, Nuclear), cybersecurity, and autonomous AI risks.

3. **AI Control Framework:** Tests systems where a red team designs "attack policies" that let an AI model intentionally pursue hidden harmful goals. Studies risks of scheming/deceptive alignment.

4. **Crowdsourced Manual Testing:** Broad coverage through many users attempting adversarial interactions.

5. **Automated Red Teaming:** Simulated adversarial attacks at scale, complementing manual testing.

**Results:**
- 96% prevention rate in tool use scenarios
- 99.4% with additional safeguards (prompt shields)
- 3,000+ collective red teaming hours: no universal jailbreak found against classifier-guarded LLMs
- Safeguards Research Team (launched 2025): focused on jailbreak-resistant training and scalable red teaming tools

**AI Safety Levels (ASL):**
- ASL-3 deployment safeguards report published May 2025
- Structured risk management framework with escalating safety requirements

Sources:
- https://alignment.anthropic.com/2025/strengthening-red-teams/
- https://www.anthropic.com/news/challenges-in-red-teaming-ai-systems
- https://www.anthropic.com/news/frontier-threats-red-teaming-for-ai-safety
- https://www.anthropic.com/asl3-deployment-safeguards

---

### 4.3 NVIDIA NeMo Guardrails

**Architecture - Multi-layered Rail System:**

| Rail Type | Applied To | Purpose |
|---|---|---|
| Input rails | User input | Reject or alter input before processing |
| Dialog rails | LLM prompting | Control how LLM is prompted, determine actions |
| Retrieval rails | RAG chunks | Reject or alter retrieved content |
| Execution rails | Action I/O | Guard custom action inputs/outputs |
| Output rails | LLM output | Reject or alter generated response |

**Colang Language:**
- Python-like DSL for designing dialogue flows
- Two versions: Colang 1.0 (default) and Colang 2.0
- Colang 2.0: complete overhaul with flows engine supporting multiple parallel flows, advanced pattern matching over event streams, core abstractions of flows/events/actions

**Built-in Safety Features:**
- LLM self-checking: input/output moderation, fact-checking, hallucination detection
- NVIDIA safety models: content safety, topic safety
- Jailbreak and injection detection
- Third-party API integrations (Palo Alto Networks, Guardrails AI)

**2025 Capabilities:**
- Streaming mode: decouples response generation from validation, chunked processing with context-aware moderation
- Parallel rails execution: reduces latency when multiple rails are configured
- OpenTelemetry migration: standardized observability for LLM calls, rail execution times, token usage
- Multi-application serving: single microservice instance serves multiple guardrail configurations

Sources:
- https://github.com/NVIDIA-NeMo/Guardrails
- https://docs.nvidia.com/nemo/guardrails/latest/index.html
- https://developer.nvidia.com/blog/stream-smarter-and-safer-learn-how-nvidia-nemo-guardrails-enhance-llm-output-streaming/
- https://docs.nvidia.com/nemo/guardrails/latest/colang-2/getting-started/dialog-rails.html

---

### 4.4 Garak (NVIDIA LLM Vulnerability Scanner)

**Architecture - Three Core Components:**

1. **Generators:** Connect to target LLMs (HuggingFace Hub, Replicate, OpenAI, LiteLLM, REST APIs). Handle authentication, connection management, response processing.

2. **Probes:** 150+ attacks organized into categories, comprising 3,000+ prompts and templates:

| Probe Category | Examples |
|---|---|
| Hallucination | Snowballed hallucination, package hallucination (non-existent packages) |
| Toxicity | RealToxicityPrompts (7 classes of toxic speech) |
| Data leakage | Training data extraction, token repetition attack |
| Prompt injection | DAN-mode, PromptInject framework, AutoDAN, Greedy Coordinate Descent (GCG) |
| Malware | Code generation that produces malicious software |
| Misinformation | Misleading claims generation |
| XSS | Cross-site scripting via LLM output |

3. **Detectors:** Analyze LLM responses using string matching, ML classifiers, or LLM-as-judge to determine if vulnerability was triggered.

**Probe Types:**
- Static: fixed prompt sets
- Dynamic: template-based with parameter variation
- Adaptive: multi-turn probes that adjust based on model responses

**Output:** JSONL report log with per-prompt records: prompt text, probe parameters, model outputs, detector results.

Sources:
- https://github.com/NVIDIA/garak
- https://arxiv.org/html/2406.11036v1
- https://docs.garak.ai/garak
- https://www.helpnetsecurity.com/2025/09/10/garak-open-source-llm-vulnerability-scanner/

---

## Cross-Cutting Themes

### Convergence Patterns

1. **Human-in-the-loop calibration is table stakes.** Every major platform (LangSmith, Braintrust, Weave) now has some form of human annotation feeding back into automated evaluation. LangSmith's Align Evals is the most formalized version.

2. **Specialized judge models outperform general-purpose models.** Patronus Lynx (8B) beats GPT-4o on hallucination detection. This suggests evaluation should use purpose-built models, not the same frontier model that generated the content.

3. **Prompt optimization is maturing rapidly.** DSPy MIPROv2 is the most rigorous approach (Bayesian optimization over instructions+demos). CAPO is emerging as the SOTA for discrete prompt optimization. Commercial services are nascent.

4. **Confidence estimation is moving from output-level to process-level.** CER, DeepConf, and CISC all show that measuring confidence during reasoning (not just at the final answer) yields better calibration and efficiency.

5. **Red teaming is consolidating.** PromptFoo acquired by OpenAI. Garak maintained by NVIDIA. NeMo Guardrails provides the runtime defense layer. The ecosystem is: scan (PromptFoo/Garak) then guard (NeMo Guardrails).

### Gaps and Opportunities

| Gap | Description | Who Partially Addresses It |
|---|---|---|
| Judge calibration without human labels | Current approaches require human annotations. Fully automated calibration (e.g., using inter-model agreement) is underexplored | None fully |
| Cost-aware evaluation | Evaluations can cost more than the inference they assess. Cost-optimized evaluation pipelines are rare | Promptomatix (cost-aware objectives) |
| Confidence for open-ended generation | Most confidence methods target QA/math. Open-ended text generation confidence is harder | Verbal confidence research |
| Safety testing for agents | Multi-step agent safety testing is nascent. Most tools test single-turn interactions | PromptFoo Hydra (multi-turn) |
| Evaluation of evaluation | Meta-evaluation (how good are your evaluators?) lacks standardized benchmarks | CALM framework (bias quantification) |
