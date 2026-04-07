# Competitive Landscape Research: Multi-Agent Evaluation, SDK Design, Cost Optimization

Date: 2026-03-27

---

## 1. Multi-Agent Evaluation

### 1.1 LangGraph + LangSmith

**How evaluation works:**
- LangSmith's `evaluate()` / `aevaluate()` runs an agent over a dataset with evaluators
- Captures full trajectory of steps, tool calls, and reasoning
- Evaluators score intermediate decisions and agent behavior, not just final output
- Supports offline evals (curated datasets, regression testing) and online evals (production traffic scoring)

**API pattern:**
```python
from langsmith import Client
client = Client()
results = await client.aevaluate(
    agent_function,
    data="dataset-name",
    evaluators=[trajectory_evaluator, correctness_evaluator],
    experiment_prefix="v2",
    num_repetitions=3,
    max_concurrency=5,
)
```

**Evaluator types:**
- Human annotation queues
- Heuristic checks (code-based assertions)
- LLM-as-judge evaluators
- Pairwise comparisons
- Custom Python/TypeScript evaluators
- Trajectory evaluators via `agentevals` library (`create_trajectory_llm_as_judge`)

**Key patterns (5 evaluation patterns from LangChain):**
1. Bespoke test logic per datapoint with custom assertions
2. Single-step evaluations for validating specific decision points
3. Full agent turn testing for end-to-end behavior
4. Multi-turn conversations with conditional logic
5. Environment setup with clean, reproducible test conditions

**Integration:** Framework-agnostic - works with LangGraph, custom Python, or any framework via SDK/API.

---

### 1.2 CrewAI

**How evaluation works:**
- Built-in CLI testing: `crewai test -n <n_iterations>`
- Runs crew for specified iterations, displays performance metrics at end
- Shows average total score for each task and crew as a whole

**Key features:**
- Performance metrics per-task and per-crew
- Optional evaluator/reviewer agents that assess outputs and trigger retries
- Lightweight evaluator agents to gate expensive steps
- Integration with third-party observability (Braintrust, LangSmith, etc.)

**Testing infrastructure:**
- Built on pytest with parallel execution, test splitting, network blocking
- VCR (Video Cassette Recorder) for recording/replaying HTTP interactions
- Deterministic test behavior without real API calls

**Performance claims:** 5.76x faster than LangGraph in certain QA task examples.

---

### 1.3 Microsoft AutoGen

**How evaluation works:**
- AutoGen v0.4 (January 2025) redesigned with modular architecture
- Agents communicate via asynchronous messages (event-driven and request/response)
- Built-in metric tracking, message tracing, debugging tools
- OpenTelemetry support for industry-standard observability

**Key metrics for multi-agent patterns:**
- Task completion rate
- Resource utilization
- Agent response time
- Collaborative efficiency

**Conversation patterns supported:**
- Multi-turn conversation handling with context management
- Star, chain, tree, graph topologies
- Message flow visualization for debugging

**Microsoft Agent Framework (2025):** Convergence of AutoGen + Semantic Kernel into unified production-ready framework.

---

### 1.4 Anthropic Claude Agent SDK

**How evaluation works:**
- SDK provides building blocks used in Claude Code
- Agents can return validated JSON matching schemas
- Self-testing and benchmark capabilities added March 2026

**Key features (2026):**
- Skill-creator handles eval writing, benchmark execution, A/B testing
- Natural language interaction for domain experts
- Self-correcting agents that catch mistakes before they compound
- Integration with external eval tools (Promptfoo supports Claude Agent SDK)

**Evaluation philosophy:** Agents that check and improve their own output are fundamentally more reliable.

---

### 1.5 Google ADK (Agent Development Kit)

**How evaluation works:**
- Three evaluation methods: web UI (`adk web`), programmatic (`pytest`), CLI (`adk eval`)
- Uses `.test.json` files for test case definition
- Evaluates both final response quality and step-by-step execution trajectory

**Test case format (`.test.json`):**
```json
{
  "turns": [
    {
      "user_content": "What's the weather?",
      "expected_tool_use": [
        {"tool_name": "get_weather", "tool_input": {"city": "NYC"}}
      ],
      "expected_response": "The weather in NYC is..."
    }
  ]
}
```

**Configuration (`test_config.json`):**
```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.8,
    "response_match_score": 0.5
  }
}
```

**Built-in evaluators:**
- `tool_trajectory_avg_score` - compares actual vs expected tool calls (match=1, mismatch=0, averaged)
- `final_response_match_v2` - LLM-as-judge for semantic equivalence
- `hallucinations_v1` - segments responses, checks each sentence for grounding
- `safety_v1` - delegates to Vertex AI Eval SDK for harmlessness
- Rubric-based criteria for response quality and tool usage

**User Simulation (2025):** LLM-powered user prompt generator integrated into eval framework - evaluates agent's ability to achieve user intent rather than following rigid implementation paths.

**Python API:**
```python
from google.adk.evaluation.agent_evaluator import AgentEvaluator

await AgentEvaluator.evaluate(
    agent_module="my_agent.agent",
    agent_name="root_agent",
    eval_dataset_file_path_or_dir="data/eval_case.test.json",
    num_runs=1,
)
```

---

### 1.6 Multi-Agent Coordination Metrics

**Standard metrics from MultiAgentBench / MARBLE (ACL 2025):**
- Task completion rate (milestone-based KPIs)
- Communication Score: LLM judges rate each inter-agent utterance on 1-5 scale (clarity, relevance, helpfulness)
- Coordination efficiency across topologies (star, chain, tree, graph)
- Plan quality assessment

**CLEAR Framework metrics:**
- Cost
- Latency
- Efficiency
- Assurance
- Reliability
- Key finding: agent performance drops from 60% single-run to 25% when measuring 8-run consistency

**REALM-Bench dimensions:**
- Multi-agent coordination
- Inter-agent dependencies
- Dynamic environmental disruptions
- Real-world planning/scheduling (14 problems, basic to complex)

**General multi-agent metrics:**
- Task delegation accuracy
- Communication efficiency (information exchange effectiveness)
- Decision synchronization (action alignment for optimal outcomes)
- Coordination overhead
- Group-level alignment/fairness

---

## 2. Agentic Evaluation Benchmarks

### 2.1 Established Benchmarks

| Benchmark | Domain | Tasks | Key Metric | Notes |
|-----------|--------|-------|------------|-------|
| **SWE-bench** | Coding agents | Real GitHub issues | % resolved | SWE-bench Verified: human-validated subset |
| **WebArena** | Web agents | 812 web tasks | Task success rate | 4 domains: e-commerce, social, code, CMS |
| **VisualWebArena** | Multimodal web agents | 910 visual tasks | Task success rate | Best models: 16.4% (ACL 2024) |
| **AgentBench** | General agents | 8 environments | Composite score | OS, database, KG, gaming, embodied AI |
| **GAIA** | General assistant | 466 tasks | Task completion | Multi-step reasoning, multimodal |

### 2.2 New Benchmarks (2025-2026)

| Benchmark | Domain | Tasks | Key Metric | Notes |
|-----------|--------|-------|------------|-------|
| **TAU-bench / TAU2-bench** | Customer service agents | Multi-domain | Dual-control success | User+agent coordination; airline, retail, banking |
| **MLE-bench** | ML engineering | 75 Kaggle competitions | Medal threshold | Best: 16.9% bronze (o1-preview + AIDE) |
| **CORE-Bench** | Scientific reproducibility | 270 tasks / 90 papers | Accuracy by difficulty | Best: 19% on hardest level |
| **MultiAgentBench (MARBLE)** | Multi-agent collaboration | Diverse scenarios | Milestone KPIs + Communication Score | ACL 2025; star/chain/tree/graph topologies |
| **REALM-Bench** | Planning/scheduling | 14 problems | Multi-dimensional | Real-world coordination, dynamic disruptions |
| **SWE-bench Pro** | Advanced coding | Harder subset | % resolved | Scale Labs leaderboard |
| **VideoWebArena** | Long-context web agents | Video understanding | Task success rate | ICLR 2025 |

**Best practice:** Combine 2-4 complementary benchmarks - baseline multi-environment (AgentBench) + domain-specific benchmarks matching primary function.

---

## 3. Programmatic SDK Design Patterns

### 3.1 DeepEval

**Core abstraction: `evaluate()` + `LLMTestCase` + metrics**

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
    expected_output="Full refund within 30 days.",
    retrieval_context=["Return policy: 30-day full refund..."]
)

results = evaluate(
    test_cases=[test_case],
    metrics=[metric],
    run_async=True,            # concurrent metric evaluation
    hyperparameters={"model": "gpt-4o", "temperature": 0.7},
)
```

**Key design choices:**
- Pytest-style: `assert_test(test_case, [metrics])` for CI/CD
- `evaluate()` for notebook/programmatic use with parallel execution
- 50+ built-in metrics, all scores 0-1, threshold-based pass/fail
- `EvaluationDataset` wraps list of test cases
- `ConversationalTestCase` for multi-turn
- G-Eval for arbitrary custom criteria via LLM-as-judge

**Input format:** `LLMTestCase(input, actual_output, expected_output, retrieval_context, context)`
**Metric interface:** `BaseMetric` with `threshold`, `model`, `measure(test_case)` returning score 0-1
**Result format:** Per-test-case scores, pass/fail, aggregated metrics

---

### 3.2 Ragas

**Core abstraction: `evaluate()` + `EvaluationDataset` + metrics**

```python
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

sample = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",
)
dataset = EvaluationDataset(samples=[sample])

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
    llm=evaluator_llm,
)
result.to_pandas()  # DataFrame with scores
```

**Key design choices:**
- `SingleTurnSample` / `MultiTurnSample` as data units
- `@experiment()` decorator for running RAG pipelines over datasets
- `DiscreteMetric` for custom pass/fail metrics
- Async scoring via `metric.ascore()`
- `column_map` parameter for dataset column name mapping
- `result.to_pandas()` for analysis

**Input format:** `SingleTurnSample(user_input, response, reference, retrieved_contexts)`
**Metric interface:** Metric classes with `ascore()` method
**Result format:** `EvaluationResult` with `.to_pandas()`, per-sample scores

---

### 3.3 Phoenix (Arize)

**Core abstraction: `px.launch_app()` + evaluator templates + trace scoring**

```python
import phoenix as px
from phoenix.evals import OpenAIModel, llm_classify

session = px.launch_app()  # starts UI at localhost:6006

# Evaluate traces with templates
model = OpenAIModel(model="gpt-4o")
results = llm_classify(
    dataframe=traces_df,
    template=HALLUCINATION_PROMPT_TEMPLATE,
    model=model,
    rails=["hallucinated", "factual"],
)
```

**Key design choices:**
- Observability-first: traces/spans as primary data model
- Score traces/spans with LLM evaluators, code checks, or human labels
- Battle-tested evaluation templates (relevance, toxicity, hallucination)
- Vendor/framework agnostic - supports OpenAI, Anthropic, Google, LangGraph, CrewAI, etc.
- Self-hosted or cloud
- Composable building blocks, not monolithic

**Input format:** DataFrame of traces/spans
**Metric interface:** Evaluator templates + `llm_classify()` function
**Result format:** Scored DataFrame, integrated with trace UI

---

### 3.4 Braintrust

**Core abstraction: `Eval()` + `data` + `task` + `scores`**

```python
from autoevals import Levenshtein, Factuality
from braintrust import Eval

Eval(
    "My Project",
    data=lambda: [
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "Capital of France?", "expected": "Paris"},
    ],
    task=lambda input: call_my_llm(input),
    scores=[Levenshtein, Factuality],
)
```

**Key design choices:**
- Declarative: single `Eval()` call defines everything
- `data` = lambda returning list of dicts (or dataset reference)
- `task` = function(input) -> output
- `scores` = list of scorer classes/functions
- Creates permanent experiment record with inputs, outputs, scores, metadata
- `autoevals` library: pre-built scorers (Levenshtein, Factuality, security, moderation)
- Custom scorers: simple functions returning score dicts

**Custom scorer example:**
```python
def conciseness_scorer(output, expected):
    tokens = len(output.split())
    return {"score": 1.0 if tokens < 200 else 0.5, "name": "conciseness"}
```

**Input format:** `{"input": ..., "expected": ...}` dicts
**Metric interface:** Scorer functions/classes returning `{"score": float, "name": str}`
**Result format:** Experiment with per-case scores, UI dashboard with comparisons

---

### 3.5 W&B Weave

**Core abstraction: `Evaluation()` + `Model` + `scorers`**

```python
import weave
import asyncio

class MyModel(weave.Model):
    @weave.op
    async def predict(self, question: str) -> str:
        return call_llm(question)

@weave.op
def correctness(output: str, expected: str) -> dict:
    return {"correct": output.strip() == expected.strip()}

dataset = [
    {"question": "What is 2+2?", "expected": "4"},
    {"question": "Capital of France?", "expected": "Paris"},
]

model = MyModel()
evaluation = weave.Evaluation(dataset=dataset, scorers=[correctness])
results = asyncio.run(evaluation.evaluate(model))
```

**Key design choices:**
- Model class with `predict()` method (argument names match dataset columns)
- Two scorer types: function-based (`@weave.op` decorated) and class-based (inherit `weave.Scorer`)
- Scorers must return dict; can include multiple metrics, nested metrics, text explanations
- `output` is mandatory keyword arg for scorers; other args taken from dataset columns
- Async-native evaluation
- Full tracing integration

**Input format:** List of dicts (or Weave Dataset); column names match predict() args
**Metric interface:** Functions/classes returning dicts; `@weave.op` decorator for tracing
**Result format:** Traced evaluation results in Weave UI

---

### 3.6 Cross-SDK Pattern Summary

| Aspect | DeepEval | Ragas | Phoenix | Braintrust | Weave |
|--------|----------|-------|---------|------------|-------|
| **Entry point** | `evaluate()` | `evaluate()` | `px.launch_app()` + `llm_classify()` | `Eval()` | `Evaluation().evaluate()` |
| **Data unit** | `LLMTestCase` | `SingleTurnSample` | DataFrame row | `dict` | `dict` |
| **Data collection** | `EvaluationDataset` | `EvaluationDataset` | DataFrame | lambda/list | list/Dataset |
| **Metric interface** | Class with `measure()` | Class with `ascore()` | Template + `llm_classify()` | Function/class -> `{score, name}` | Function/class -> `dict` |
| **Score range** | 0-1 | 0-1 | categorical or numeric | numeric | any dict |
| **Pass/fail** | threshold-based | N/A | N/A | N/A | N/A |
| **CI/CD** | pytest native | pytest possible | N/A | CLI + SDK | SDK |
| **Async** | `run_async` param | native async | N/A | N/A | native async |
| **Tracing** | optional | optional | core feature | core feature | core feature |

**Common patterns across all:**
1. Separation of data, task/model, and scoring
2. Metrics as composable, pluggable units
3. LLM-as-judge as primary complex evaluation method
4. Support for custom metrics via simple function interface
5. Results as structured data (DataFrames, dicts, dashboards)
6. Batch evaluation over datasets

---

## 4. Cost Optimization & Model Routing

### 4.1 LLM Gateway Architecture

**How gateways work:**
- Single API endpoint between app and multiple LLM providers
- Handle API format differences, failovers, cost optimization, monitoring
- Enterprise LLM spending surpassed $8.4B in 2025

**Portkey:**
- Routes to 250+ LLM providers via single API
- Routing strategies: round-robin, weighted, priority-based, conditional
- Conditional routing: "use cheaper model for summarization, premium for reasoning"
- Semantic caching + simple caching
- Automatic retries (up to 5x with exponential backoff)
- Fallback targets are composable (each target can be a load balancer, conditional router, or another fallback)
- 50+ AI guardrails integrated

**LiteLLM:**
- Unified interface across 100+ providers
- Python Router class with model_list and fallback configuration:
```python
from litellm import Router
router = Router(
    model_list=[
        {"model_name": "gpt-3.5-turbo", "litellm_params": {...}},
        {"model_name": "gpt-4", "litellm_params": {...}},
    ],
    fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}],
)
response = router.completion(model="gpt-3.5-turbo", messages=[...])
```
- Priority-based routing: each priority level gets own retries before escalating
- Redis-based caching for exact matches; semantic caching available but secondary
- Open-source, self-hostable

### 4.2 Semantic Caching

**How it works:**
1. Convert query to embedding vector
2. Search vector store for semantically similar cached queries
3. If similarity above threshold, return cached response (skip LLM call)
4. If miss, call LLM, cache result with embedding

**GPTCache architecture (6 components):**
1. Adapter - interfaces with LLM frameworks
2. Pre-processor - normalizes queries
3. Embedding generator - converts to vectors
4. Cache manager - storage/retrieval (supports Redis, FAISS, Milvus)
5. Similarity evaluator - threshold-based matching
6. Post-processor - formats cached responses

**Performance:**
- Reduces API calls by up to 68.8%
- Cache hit rates: 61.6% to 68.8% across categories
- Positive hit accuracy exceeding 97%
- Up to 95% cost savings for repetitive queries

**Redis semantic caching:**
- In-memory storage for fast retrieval
- HNSW for efficient similarity search
- Strong scalability

### 4.3 Cost-Aware Model Selection (Cascade Routing)

**Research patterns (2025):**

**Simple cascade:** Run cheap model first; if confidence below threshold, escalate to expensive model.
- 90% of queries handled by small models (e.g., Mistral 7B)
- Only 10% escalated to premium models
- 87% cost reduction achieved

**Unified cascade routing (ETH Zurich, 2024):**
- Integrates routing + cascading into theoretically optimal strategy
- Iteratively picks best model; can skip models, reorder, run as few as needed
- Outperforms existing strategies by up to 14%
- Key insight: quality estimators are the critical success factor

**Gatekeeper approach:**
- Routes each query to small on-device model or large cloud model
- Based on predicted difficulty + tunable quality threshold
- Up to 40% reduction in expensive model calls without quality degradation

**Router-R1 (2025):**
- Reinforcement learning-based router
- Learns to select optimal model per query

**Practical implementation pattern:**
```
Query -> Confidence Estimator ->
  High confidence -> Small/cheap model -> Response
  Low confidence  -> Large/expensive model -> Response
```

### 4.4 Braintrust Gateway

**Current state:**
- AI proxy deprecated; replaced by "gateway"
- Routes LLM API calls through Braintrust
- Automatic log capture, caching, fallbacks
- Supports OpenAI, Anthropic, Google, AWS, Mistral, 100+ models
- Simple base URL change to enable
- Production-grade reliability focus

**Caching:** Automatically caches results, reuses when possible for cost reduction. Second run significantly faster from cache.

**Note:** Specific "auto-model selection" feature not confirmed in current docs - may refer to the gateway's caching + fallback behavior rather than intelligent routing.

---

## Key Takeaways for Evalyn

### What the market is converging on:
1. **Trajectory evaluation** (not just final output) is the standard for agent evals
2. **LLM-as-judge** is the primary method for complex evaluation
3. **Simple function interface** for custom metrics (function -> dict/score)
4. **Separation of concerns:** data, task, scoring as independent pluggable components
5. **Async-native** evaluation for performance
6. **CI/CD integration** via pytest or similar
7. **Multi-turn / conversational** test cases as first-class citizens

### Gaps in the market:
1. **Multi-agent evaluation** is nascent - only MultiAgentBench/MARBLE has formal metrics
2. **Cost-aware evaluation** (routing during evals) not integrated into any eval SDK
3. **Consistency measurement** rarely built-in (CLEAR framework is the exception: 60% -> 25% drop)
4. **Agent trajectory comparison** across runs is manual in most tools
5. **User simulation** only in Google ADK - others lack synthetic user generation (Note: Evalyn now has `evalyn simulate` for synthetic user generation, closing this gap)

### Design patterns to adopt:
- `evaluate(dataset, metrics, ...)` as the core API (DeepEval, Ragas pattern)
- Scorer as simple function: `def my_scorer(output, expected) -> dict`
- `TestCase` / `Sample` dataclass for structured input
- Threshold-based pass/fail for CI/CD
- `.to_pandas()` / `.to_dict()` for result export
- Async evaluation with concurrency control

---

## Sources

### Multi-Agent Evaluation
- [LangSmith Evaluation Platform](https://www.langchain.com/langsmith/evaluation)
- [LangChain - How to evaluate a graph](https://docs.langchain.com/langsmith/evaluate-graph)
- [LangChain agentevals](https://github.com/langchain-ai/agentevals)
- [CrewAI Testing](https://docs.crewai.com/en/concepts/testing)
- [AutoGen v0.4](https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)
- [AutoGen Multi-Agent Patterns](https://sparkco.ai/blog/deep-dive-into-autogen-multi-agent-patterns-2025)
- [Anthropic Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Building agents with Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Google ADK Docs](https://google.github.io/adk-docs/)
- [Google ADK Evaluation Criteria](https://google.github.io/adk-docs/evaluate/criteria/)
- [ADK User Simulation](https://developers.googleblog.com/announcing-user-simulation-in-adk-evaluation/)
- [Google ADK Python](https://github.com/google/adk-python)

### Agent Benchmarks
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [WebArena](https://webarena.dev/)
- [VisualWebArena](https://jykoh.com/vwa)
- [MultiAgentBench / MARBLE](https://github.com/MultiagentBench/MARBLE)
- [REALM-Bench](https://arxiv.org/abs/2502.18836)
- [MLE-bench](https://github.com/openai/mle-bench)
- [CORE-Bench](https://openreview.net/forum?id=BsMMc4MEGS)
- [TAU2-bench](https://github.com/sierra-research/tau2-bench)
- [Agent Evaluation Framework 2026 - Galileo](https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks)
- [AI Agent Benchmarks Guide](https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide)
- [Agent Benchmark Compendium](https://github.com/philschmid/ai-agent-benchmark-compendium)

### SDK Design
- [DeepEval Getting Started](https://deepeval.com/docs/getting-started)
- [DeepEval Evaluation Introduction](https://deepeval.com/docs/evaluation-introduction)
- [DeepEval Metrics](https://deepeval.com/docs/metrics-introduction)
- [Ragas evaluate() Reference](https://docs.ragas.io/en/stable/references/evaluate/)
- [Ragas Quick Start](https://docs.ragas.io/en/latest/getstarted/quickstart/)
- [Phoenix - Arize AI](https://arize.com/docs/phoenix)
- [Braintrust Evaluation Quickstart](https://www.braintrust.dev/docs/evaluation)
- [Braintrust Scorers](https://www.braintrust.dev/docs/evaluate/write-scorers)
- [Braintrust AutoEvals](https://github.com/braintrustdata/autoevals)
- [W&B Weave Scorers](https://github.com/wandb/weave/blob/master/docs/docs/guides/evaluation/scorers.md)
- [W&B Weave Evaluation Tutorial](https://docs.wandb.ai/weave/tutorial-eval)

### Cost Optimization
- [Top LLM Gateways 2025](https://www.helicone.ai/blog/top-llm-gateways-comparison-2025)
- [Portkey AI Gateway](https://portkey.ai/features/ai-gateway)
- [LiteLLM Router](https://docs.litellm.ai/docs/routing)
- [LiteLLM Fallbacks](https://docs.litellm.ai/docs/proxy/reliability)
- [GPTCache](https://github.com/zilliztech/GPTCache)
- [Semantic Caching for LLMs](https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality/)
- [Redis Semantic Caching](https://redis.io/blog/what-is-semantic-caching/)
- [Unified Routing and Cascading for LLMs](https://arxiv.org/abs/2410.10347)
- [Cost-Aware Contrastive Routing](https://openreview.net/pdf?id=4Qe2Hga43N)
- [LLM Cost Optimization Guide](https://ai.koombea.com/blog/llm-cost-optimization)
- [Braintrust AI Proxy](https://www.braintrust.dev/blog/ai-proxy)
