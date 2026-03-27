# Competitive Landscape: Dataset Management, Annotation, and Data Engineering for LLM Evaluation

Research date: 2026-03-27

---

## 1. Dataset Platforms

### 1.1 HuggingFace Datasets

**Core architecture:**
- Backed by Apache Arrow for zero-copy reads and memory-mapped disk access
- Hub auto-converts first 5GB of every dataset to Parquet for the dataset viewer
- Available builders: json, csv, parquet, arrow, text, xml, webdataset, imagefolder, audiofolder, videofolder
- Features/schema auto-inferred by Arrow, overridable via `Features` argument

**Versioning:**
- Git-based repositories with full commit history, diffs, branches
- Every dataset is a Git repo on the Hub - version-controlled folders with library integrations

**Streaming:**
- Iterate progressively without full download
- No local disk required - only a small portion in memory at a time
- 100x efficiency improvements with prefetching and multi-worker scaling

**Dataset Cards:**
- Rendered on the dataset's main page
- Metadata: license, language, size, tags for discoverability
- Data files configuration in YAML header

**Key takeaways for evalyn:**
- Arrow/Parquet as storage format is the de facto standard - evalyn should read/write Parquet
- Git-based versioning for datasets is proven at scale
- Streaming API pattern (iterate without full download) is important for large eval sets
- Dataset cards (structured metadata) should be a first-class concept

Sources:
- [HuggingFace Datasets GitHub](https://github.com/huggingface/datasets)
- [Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Dataset Cards](https://huggingface.co/docs/hub/en/datasets-cards)
- [Streaming Datasets](https://huggingface.co/blog/streaming-datasets)
- [Datasets and Arrow](https://huggingface.co/docs/datasets/about_arrow)

---

### 1.2 LangSmith Datasets

**Dataset management:**
- Dataset = set of examples, each with input params and expected/reference output
- Split labels and metadata for traceability across updates
- Metadata and text filters for narrowing focus by tags, owners, or free text
- Automatic versions on every edit/deletion - clean audit trail
- Tags to mark significant versions

**Annotation UI:**
- Annotation queues: flag runs for review, assign to subject-matter experts
- Expert feedback used to calibrate automated evaluation, improve prompts, augment datasets
- Inline trace-based annotation - add human labels directly on chain traces

**Evaluation integration:**
- Human evaluation via annotation queues
- Heuristic checks (output validation, code compilation)
- LLM-as-judge evaluators with custom scoring criteria
- Pairwise comparisons
- Offline evaluation (curated datasets, regression testing) + online evaluation (production traffic)

**Synthetic data:**
- Generate candidates with LLM, tag as synthetic, keep creation steps for review

**Key takeaways for evalyn:**
- Auto-versioning on every dataset mutation is a strong pattern
- Annotation queues that feed back into dataset improvement create a flywheel
- Supporting both offline (pre-deployment) and online (production) eval modes is standard
- Trace-to-dataset pipeline (flag production traces, add to eval set) is valuable

Sources:
- [LangSmith Evaluation Platform](https://www.langchain.com/langsmith/evaluation)
- [LangSmith Datasets Managing Evaluation](https://www.statsig.com/perspectives/langsmith-datasets-managing-evaluation)
- [What Is LangSmith 2026 Guide](https://www.trantorinc.com/blog/what-is-langsmith)

---

### 1.3 Argilla

**Core design:**
- Open-source data platform integrating human and machine feedback
- Server + web app for labeling/curation, Python library for building annotation workflows
- Now owned by HuggingFace, deeply integrated with the Hub

**RLHF workflow support:**
- Demonstration data collection for supervised fine-tuning
- Comparison data collection for reward model training (rank multiple responses per prompt)
- LLM monitoring/evaluation with continuous feedback collection

**Custom annotation interfaces:**
- CustomField feature: define own HTML/CSS/JS templates
- Build fully customized annotation interfaces for specialized data types
- Supports code annotation, 3D models, multimodal content

**Planned features:**
- Suggestions from multiple models and rules
- Active learning for SFT and reward modeling
- Vector search integration
- Weak supervision

**Key takeaways for evalyn:**
- Flexible annotation interfaces (not one-size-fits-all) matter for diverse eval tasks
- Comparison/ranking UI is essential for preference-based evaluation
- Integration with model training pipeline (RLHF) shows the full data lifecycle
- Active learning for annotation efficiency is a coming differentiator

Sources:
- [Argilla GitHub](https://github.com/argilla-io/argilla/)
- [Argilla Website](https://argilla.io/)
- [Argilla RLHF Documentation](https://docs.v1.argilla.io/en/v1.16.0/conceptual_guides/llm/rlhf.html)

---

### 1.4 Label Studio

**Core platform:**
- Open-source, multi-type data labeling with standardized output format
- Enterprise version adds LLM-assisted auto-labeling, model evaluation, team coordination

**LLM evaluation features:**
- Built-in templates for chatbot evaluation, response grading, side-by-side model comparison
- Instruction tuning, preference annotation, span-level editing for NER/RE
- Human evaluation layer for AI and agentic systems (Jan 2026 update)

**Quality assurance:**
- Annotator onboarding and evaluation workflows
- Calibration, quality gates, reviewer feedback, dashboards
- Consensus scoring and inter-annotator agreement

**Multi-modal support:**
- Images, audio, text, video, time series
- Speech event detection, multilingual transcription correction
- Custom hotkeys, pixel-perfect annotation

**Key takeaways for evalyn:**
- Standardized output format across annotation types is important
- Quality gates and inter-annotator agreement are mature patterns for annotation QA
- Side-by-side comparison UI is standard for LLM evaluation
- Template-based annotation setup enables rapid workflow creation

Sources:
- [Label Studio](https://labelstud.io/)
- [LLM Evaluation Templates](https://labelstud.io/blog/new-llm-evaluation-templates-for-label-studio/)
- [Label Studio 2025 Review](https://sider.ai/blog/ai-tools/is-label-studio-the-best-open-source-labeling-tool-a-2025-review)
- [Label Studio GitHub](https://github.com/HumanSignal/label-studio)

---

### 1.5 Lilac (now Databricks)

**Core purpose:** Dataset exploration, curation, and quality control for LLM training/fine-tuning

**Key capabilities:**
- Explore, filter, cluster, and annotate data at scale
- LLM-powered insights for automated data quality analysis
- Semantic search across datasets
- PII detection
- Duplicate removal
- Automated data transformations

**Clustering and insights:**
- Cluster any text column for automated dataset insights
- Lilac Garden (hosted platform) achieves 100x clustering speedup
- Analyze model outputs for bias or toxicity

**Acquisition by Databricks:**
- Integrated into Mosaic AI platform
- Simplifies data tailoring for LLM evaluation, RAG preparation, fine-tuning

**Key takeaways for evalyn:**
- Automated clustering of eval datasets reveals patterns humans miss
- PII detection should be built into dataset curation pipelines
- Semantic search over datasets (not just keyword) is a baseline expectation
- Dataset-level insights (distributions, clusters, outliers) are a strong UX pattern

Sources:
- [Lilac GitHub](https://github.com/databricks/lilac)
- [Lilac Website](https://www.lilacml.com/)
- [Databricks Acquires Lilac](https://www.databricks.com/blog/lilac-joins-databricks-simplify-unstructured-data-evaluation-generative-ai)

---

### 1.6 Scale AI

**Current focus (2025-2026):**
- Primary business is RLHF annotation for LLMs and defense contracts
- Meta acquired 49% non-voting stake for $14.8B (June 2025)

**Platform capabilities:**
- Data Engine: end-to-end data collection, curation, annotation, RLHF, evaluation
- Scale Evaluation (April 2025): test LLMs against benchmarks, pinpoint weaknesses, flag where additional training data would improve
- Scale Labs (March 2026): expanded research division for post-training evaluation, enterprise deployment, risk oversight

**Key takeaways for evalyn:**
- Evaluation and annotation are converging - platforms do both
- "Pinpoint weaknesses and recommend training data" is the emerging value prop
- Enterprise-grade evaluation is a growing market segment

Sources:
- [Scale AI Wikipedia](https://en.wikipedia.org/wiki/Scale_AI)
- [Scale AI RLHF](https://scale.com/rlhf)
- [Scale AI Review 2026](https://labelyourdata.com/articles/scale-ai-review)

---

### 1.7 Snorkel AI

**Core approach: Programmatic labeling via weak supervision**

**How it works:**
- Users write labeling functions (simple programs) to label data
- Statistical techniques model agreement/disagreement between labeling functions
- System learns when, where, and how much to trust each function
- Both no-code UI and Python SDK

**Weak supervision methodology:**
- High-level, noisier supervision sources create large training sets
- Observes where labeling functions agree/disagree to learn expertise areas
- Subject-matter experts collaborate with data scientists on labeling functions

**Recent products (2025-2026):**
- Snorkel Evaluate: specialized evaluation for agentic AI systems
- Expert Data-as-a-Service: expert data development and labeling
- Alfred: open-source package for programmatic weak supervision with foundation models

**Key takeaways for evalyn:**
- Programmatic labeling (code-defined annotation rules) scales better than manual annotation
- Weak supervision with multiple noisy signals is a proven approach
- Labeling functions as first-class objects (versionable, composable) is a strong pattern
- SME + data scientist collaboration model is the right abstraction level

Sources:
- [Snorkel AI](https://snorkel.ai/)
- [Snorkel Weak Supervision](https://snorkel.ai/data-centric-ai/weak-supervision/)
- [Alfred Package](https://snorkel.ai/blog/alfred-data-labeling-with-foundation-models-and-weak-supervision/)

---

## 2. Evaluation Dataset Standards

### 2.1 HELM (Stanford)

**Design principles:**
- Holistic: 7 metrics per scenario (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency)
- 42 scenarios across diverse tasks, 16+ models evaluated
- Open-source Python framework

**Architecture:**
- Standardized dataset format with unified model interface
- Web UI for inspecting individual prompts and responses
- Web leaderboard for cross-model comparison
- Full transparency: all raw prompts and completions released publicly

**Extensibility:**
- IBM extended HELM for enterprise benchmarks (finance, legal, climate, cybersecurity)
- Modular toolkit enables custom scenarios and metrics

**Key takeaways for evalyn:**
- Multi-metric evaluation per scenario (not just accuracy) is the gold standard
- Releasing raw prompts/completions enables community analysis
- Modular scenario + metric architecture enables domain-specific extension
- Web UI for prompt-level inspection is expected infrastructure

Sources:
- [HELM Website](https://crfm.stanford.edu/helm/)
- [HELM GitHub](https://github.com/stanford-crfm/helm)
- [HELM Overview](https://www.statsig.com/perspectives/helm-benchmark-llm-eval)

---

### 2.2 BIG-bench

**Design patterns:**
- 200+ tasks from 450 authors across 132 institutions
- Two task types: JSON (~80%) and programmatic (code-defined)
- JSON tasks: list of input/target examples for few-shot evaluation
- Programmatic tasks: code-based, allow sophisticated model interaction

**Task diversity:**
- Formal logic and symbolic reasoning
- Mathematical and algorithmic reasoning
- Linguistic and meta-linguistic competence
- Multi-step compositional reasoning

**Evaluation methodology:**
- Standardized automatic answer extraction
- Deterministic scoring at scale
- Primary metric: accuracy; BBEH introduces harmonic mean accuracy

**BIG-Bench Extra Hard (BBEH, 2025):**
- Semi-adversarial protocol: 6x context length, 7x reasoning depth
- Distractors and adversarial elements added
- Automatic answer grading preserved
- Replaces each BBH task with harder counterpart probing same skill

**Key takeaways for evalyn:**
- Supporting both JSON-defined and code-defined tasks is important
- Community contribution model (450 authors) shows value of open task submission
- Increasing difficulty via adversarial augmentation is a standard technique
- Harmonic mean accuracy penalizes models that only excel in narrow areas

Sources:
- [BIG-bench GitHub](https://github.com/google/BIG-bench)
- [BIG-Bench Extra Hard Paper](https://arxiv.org/pdf/2502.19187)
- [BIG-bench Overview](https://www.emergentmind.com/topics/big-bench)

---

### 2.3 MMLU

**Format:**
- 15,908 multiple-choice questions across 57 subjects
- Four answer choices per question, one correct
- Difficulty levels from high school to professional

**Contamination and versioning evolution:**
- Original MMLU suffered from benchmark contamination (models memorized test data)
- MMLU-Pro: harder variant with more answer choices and reasoning requirements
- MMLU-CF (Contamination-Free, ACL 2025): statement rephrasing, option shuffling, "None of the other choices" injection
- MMLU-CF maintains closed-source test set + open validation set

**Best practices (as of 2025):**
- Expert multi-stage review
- Prompt set averaging
- Provenance tracking
- Closed and versioned test splits
- Subject- and language-specific calibration

**Key takeaways for evalyn:**
- Closed test splits are necessary for contamination resistance
- Versioned benchmarks with known fidelity scores are a best practice
- Statement rephrasing and option shuffling are concrete decontamination techniques
- MMLU is being phased out in favor of harder alternatives - difficulty must evolve

Sources:
- [MMLU Wikipedia](https://en.wikipedia.org/wiki/MMLU)
- [MMLU-CF GitHub](https://github.com/microsoft/MMLU-CF)
- [MMLU-CF ACL 2025](https://aclanthology.org/2025.acl-long.656/)
- [MMLU Benchmark Overview](https://www.emergentmind.com/topics/mmlu-benchmark)

---

### 2.4 HumanEval

**Format:**
- 164 programming problems with docstrings
- Zero-shot prompting: model generates complete Python function body
- Unit tests intentionally hidden from model
- Primary metric: pass@k (probability at least 1 of k samples passes)

**Evolution and versioning:**
- HumanEval+ (2024): expands to median 764 test cases per problem via LLM-generated corner cases + mutation
- HumanEvalNext: fixes docstring ambiguities, increases difficulty spectrum, 20-31% pass@1 drop
- HumanEval-T: template-based variants preventing memorization, combinatorial lexical transformations

**Contamination countermeasures:**
- HumanEval-T generates lexically distinct variants ensuring semantic equivalence but preventing memorization
- Template-based abstraction + pairwise covering arrays

**Key takeaways for evalyn:**
- pass@k metric (sample multiple, check if any pass) is standard for code eval
- Hidden test suites prevent gaming
- Expanding test coverage programmatically (mutation testing) catches more edge cases
- Template-based variant generation is a concrete anti-contamination technique

Sources:
- [HumanEval Overview](https://www.emergentmind.com/topics/humaneval)
- [HumanEval DataCamp](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities)
- [HumanEval+ Enhanced](https://www.emergentmind.com/topics/humaneval-184d0fb5-b481-4681-aca4-8f5a7f000fca)
- [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench)

---

### 2.5 Benchmark Versioning, Contamination, and Splits - Cross-Cutting Patterns

**Contamination detection approaches:**
1. String matching and embedding similarity
2. Likelihood-based techniques (perplexity analysis)
3. CoDeC: measures how in-context learning affects performance on suspected contaminated data
4. Watermarking: embed cryptographic watermarks via question reformulation before release
5. Data Contamination Quiz (DCQ): present perturbed versions, check if model identifies original

**Key finding:** No single detection technique reliably distinguishes contaminated from uncontaminated items. Combining methods does not significantly improve detection for subtle contamination.

**Mitigation strategies:**
- Dynamic benchmarking: continuously update datasets based on LLM training timestamps
- Closed test sets with open validation sets
- Periodic test set rotation
- Proprietary test sets from production data (immune to contamination by definition)
- High-fidelity paraphrasing constrained by semantic similarity

**Best practices for dataset splits:**
- Version test sets, record scores for every model update
- Rotate test sets periodically
- Treat evaluation baselines as products requiring maintenance
- Build proprietary test sets of 100-500 examples from production data

Sources:
- [Data Contamination Survey](https://arxiv.org/html/2406.04244v1)
- [Static to Dynamic Evaluation Survey](https://aclanthology.org/2025.emnlp-main.511/)
- [ICML 2025 Contamination Mitigation](https://github.com/ASTRAL-Group/BDC-mitigation-assessment)
- [Contamination Detection via CoDeC](https://openreview.net/forum?id=YlpaaYxx4t)

---

## 3. Data Quality and Curation

### 3.1 Data Contamination Detection

**Current approaches (2025):**

| Method | How it works | Strength | Weakness |
|--------|-------------|----------|----------|
| String matching | Exact/near-exact overlap between train and test | Simple, fast | Misses paraphrases |
| Embedding similarity | Semantic similarity in vector space | Catches paraphrases | High compute cost |
| Likelihood-based | Perplexity analysis on test items | No test data access needed | High false positive rate |
| CoDeC (in-context) | Measures performance delta with/without context | Works on black-box models | Requires careful calibration |
| Watermarking | Cryptographic marks via reformulation | Proactive, high confidence | Only works on new benchmarks |
| DCQ (quiz-based) | Present perturbed versions, check if model picks original | Intuitive, low cost | Limited to memorization detection |

**Key limitation:** Current methods struggle with contamination from instruction fine-tuning with answer augmentation, and show limited consistency between techniques.

Sources:
- [Contamination Detection Limitations ACL 2025](https://aclanthology.org/2025.coling-main.338/)
- [Watermarking Approach](https://openreview.net/forum?id=WFGxFzFDmQ)
- [Awesome Data Contamination List](https://github.com/lyy1994/awesome-data-contamination)

---

### 3.2 Dataset Deduplication

**Three strategies:**

| Strategy | Technique | Precision | Cost | Best for |
|----------|-----------|-----------|------|----------|
| Exact | Cryptographic hashing (MD5/SHA) | Perfect | Low | Identical duplicates |
| Approximate | MinHash + LSH (Jaccard similarity) | High | Medium | Near-duplicates at scale |
| Semantic | Embedding models + ANN search | Highest | High | Conceptual duplicates |

**MinHash + LSH** is the dominant technique for LLM data pipelines:
- MinHash compresses documents into compact signatures
- LSH groups likely matches, narrowing search space
- Milvus 2.6 added native MinHash LSH indexing
- LSHBloom improves computational cost/performance ratio

**Semantic deduplication:**
- SemHash: lightweight multimodal library using Model2Vec embeddings + Vicinity ANN search
- Works for text out of the box, supports images/audio/other modalities with custom encoders

**Key takeaways for evalyn:**
- Exact dedup is table stakes, approximate dedup is important, semantic dedup is the frontier
- MinHash + LSH should be the default for text deduplication
- SemHash pattern (lightweight embeddings + ANN) is practical for eval datasets
- Dedup should run at dataset creation time, not evaluation time

Sources:
- [Dedup at Trillion Scale](https://zilliz.com/blog/data-deduplication-at-trillion-scale-solve-the-biggest-bottleneck-of-llm-training)
- [SemHash GitHub](https://github.com/MinishLab/semhash)
- [MinHash LSH in Milvus](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)
- [Duplodocus (Allen AI)](https://github.com/allenai/duplodocus)

---

### 3.3 Data Quality Scoring

**LLM-based scoring dimensions:**
- Deita Complexity: prompt difficulty prediction
- Thinking Probability: likelihood of multi-step reasoning
- Deita Quality: automated reward model score
- Instruction Following Difficulty
- Fail Rate

**LLM-as-Judge scoring dimensions:**
- Difficulty, Relevance, Clarity, Coherence, Completeness, Complexity, Correctness, Meaningfulness

**Score correction approaches:**
- DS2: models error patterns via score transition matrix to correct LLM-based scores
- Promotes diversity alongside quality

**Modern curation pipeline:**
1. Normalization and language identification
2. Near-duplicate removal (Jaccard/MinHash)
3. Document-level filtering based on metadata descriptors
4. Quality scoring (LLM-based or heuristic)
5. Diversity sampling

**Key limitation:** LLM-based scoring introduces model-specific biases. Distilling judgments into lightweight classifiers is the mitigation pattern.

**Key takeaways for evalyn:**
- Multi-dimensional quality scoring (not a single number) is the standard
- LLM-as-judge for quality scoring is practical but needs bias correction
- Pipeline: dedup -> filter -> score -> sample is the canonical flow
- Lightweight classifiers trained on LLM judgments scale better than calling LLMs repeatedly

Sources:
- [LLM Data Auditor Survey](https://arxiv.org/html/2601.17717v1)
- [Quality Over Quantity (Microsoft)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/2503.09205v4.pdf)
- [DS2 Score Correction](https://openreview.net/forum?id=DKkQtRMowq)

---

### 3.4 Active Learning for Annotation Efficiency

**Core query strategies:**
1. Uncertainty sampling (QBC): select examples the model is least confident about
2. Entropy sampling with diversity: combine uncertainty with coverage
3. Core-set sampling: maximize geometric coverage of feature space
4. Information density sampling: weight by representativeness

**LLM-specific uncertainty estimation (2025):**
- Verbalization-based: prompt LLMs for explicit confidence judgments
- Consistency-based: generate multiple responses, analyze variability
- Logit-based: entropy or margin metrics from internal probability distribution

**Efficiency gains:**
- Active learning cuts labeling effort by 30-70% depending on domain
- Uncertainty sampling reaches 90% of final performance using only 40% of labeled data
- LLM-bootstrapped active learning retains 93% performance at 6% of compute cost

**Hybrid approach:**
- LLM preprocesses and clusters documents
- Humans label a few per cluster (diversity) + a few the model is uncertain about
- Example: 1,000 labels out of 100,000 achieved near-full accuracy

**Key takeaways for evalyn:**
- Active learning should guide which items to annotate next
- Clustering + uncertainty sampling hybrid is the most practical approach
- LLM-bootstrapped annotation (LLM labels first, human corrects) is proven efficient
- Even simple uncertainty sampling provides 2-3x annotation efficiency

Sources:
- [Survey of LLM-based Active Learning](https://aclanthology.org/2025.acl-long.708.pdf)
- [Active Learning Guide 2025](https://encord.com/blog/active-learning-machine-learning-guide/)
- [Active Learning Reduces Labeling Costs](https://labelyourdata.com/articles/active-learning-machine-learning)

---

## 4. Synthetic Data Generation

### 4.1 Persona-Based Simulation

**Approaches:**
- Two-stage: LLM completes personality questionnaire, then generates biography reflecting traits
- PolyPersona: persona-conditioned survey response generation across domains
- Define user profiles that shape tone, intent, knowledge level, and "quirks"

**Evidently AI pattern:**
- Define user profiles and goals
- Select LLMs for generation
- Build customizable pipelines
- Set specific user profiles: knowledge level, emotional state, communication style

**Challenges:**
- LLM personas produce biased and overly homogeneous opinions
- Systematic biases in downstream tasks when predicting real-world outcomes
- Ad hoc generation without methodological rigor

**Validation requirements:**
- Expert and crowd validation with independent annotators
- Assess realism, coverage, and correctness
- Compare synthetic distribution to known real-world distribution

**Key takeaways for evalyn:**
- Persona definitions should be structured, versionable objects
- Diversity controls (not just quality) are essential to avoid homogeneity
- Validation pipeline comparing synthetic vs. real distributions is needed
- Profile-based generation (tone, intent, knowledge level) is the standard abstraction

Sources:
- [PolyPersona Paper](https://arxiv.org/pdf/2512.14562)
- [Evidently Synthetic Data Generator](https://www.evidentlyai.com/blog/synthetic-data-generator-python)
- [LLM Synthetic Data Reading List](https://github.com/pengr/LLM-Synthetic-Data)

---

### 4.2 Adversarial Data Generation

**Key frameworks:**
- Rainbow Teaming (NeurIPS 2024): open-ended generation of diverse adversarial prompts
- TRIDENT (ACL 2025): tri-dimensional diversified red-teaming data synthesis
- MALLM-GAN: multi-agent GAN framework for realistic synthetic data

**Red teaming methodology (OpenAI):**
- Manual: humans craft prompts simulating adversarial scenarios
- Automated: AI models or templates generate adversarial prompts
- Findings documented in specific format for addition to safety evaluations

**Adversarial evaluation focus areas:**
- Forbidden topic probing
- Prompt injection design
- Rare/difficult test case generation
- Jailbreak attempt simulation

**Key takeaways for evalyn:**
- Adversarial dataset generation should be a built-in capability
- Red teaming data should feed back into eval datasets
- Structured finding format enables systematic safety evaluation
- Both manual and automated adversarial approaches complement each other

Sources:
- [OpenAI Red Teaming Paper](https://cdn.openai.com/papers/openais-approach-to-external-red-teaming.pdf)
- [OpenAI Evals GitHub](https://github.com/openai/evals)
- [LLM Synthetic Data Survey](https://arxiv.org/html/2503.14023v1)

---

### 4.3 Multi-Turn Conversation Generation

**Framework approaches:**
- Modular framework supporting SFT, DPO, GRPO training objectives
- Four generation paradigms: multi-turn dialogues, document-grounded pairs, verifiable instruction-response tasks, long-context reasoning

**Composition strategies:**
- Bottom-Up: generate atomic tasks from real-world scenarios
- Sequential Composition: chain atomic tasks
- Parallel-then-Sequential: combine parallel + sequential composition
- ToolDial: 11,111 dialogues averaging 8.95 turns with 16 user/system actions

**Prompt-based generation:**
- Provide article summaries + emotional tone as input
- LLM produces multi-turn conversations
- Multilingual support

**Key challenge:** All top LLMs exhibit 39% average performance drop in multi-turn vs. single-turn conversations, making multi-turn eval data especially important.

**Key takeaways for evalyn:**
- Multi-turn generation should support composition of atomic tasks
- Both sequential and parallel composition patterns are needed
- 8-10 turns per dialogue is a realistic target for richness
- Multi-turn eval data is critical because models degrade significantly in this setting

Sources:
- [Modular Long-Context Generation](https://arxiv.org/html/2509.01185)
- [Langfuse Multi-Turn Simulation](https://langfuse.com/guides/cookbook/example_simulated_multi_turn_conversations)
- [ToolDial](https://openreview.net/forum?id=J1J5eGJsKZ)
- [LLMs Get Lost in Multi-Turn](https://arxiv.org/pdf/2505.06120)

---

### 4.4 How Major Companies Generate Eval Data

**Anthropic - Bloom framework:**
- Four-stage agentic pipeline: Understanding -> Ideation -> Rollout -> Judgment
- Generates detailed scenario descriptions (situation, simulated user, system prompt, environment)
- Simulates both user and tool responses dynamically
- Judge model scores transcripts for target behavior + secondary qualities
- Correlates 0.86 Spearman with human scores (Claude Opus 4.1)
- Evaluates: sycophancy, sabotage, self-preservation, self-preferential bias
- Open-source at github.com/safety-research/bloom

**Anthropic - Model-Written Evaluations:**
- LLMs generate evaluation datasets for evaluating other LLMs
- Interactive visualization for exploring generated datasets
- Focused on behavioral trait discovery

**OpenAI:**
- GPT-4 generates synthetic eval data with custom YAML configuration
- Red teaming: manual + automated, documented in structured format
- Evals framework: registry of standard evals + custom eval support
- Private evals without exposing data publicly

**Cross-company collaboration:**
- Anthropic-OpenAI joint evaluation exercise (June-July 2025)
- Ran strongest internal alignment evals on each other's models
- Evaluated GPT-4o, GPT-4.1, o3, o4-mini vs. Claude Opus 4, Claude Sonnet 4

**DeepEval (Confident AI):**
- Synthetic dataset generation with evolution techniques
- Multi-turn goldens, not just single-turn
- Tree-based DAG for multi-step conditional scoring
- Agent evaluation via trace analysis

**Key takeaways for evalyn:**
- Agentic eval generation (Bloom pattern) is the frontier
- Four-stage pipeline (understand -> ideate -> rollout -> judge) is a strong architecture
- Model-written evals (LLMs generating eval data) is proven by Anthropic
- YAML-based eval configuration (OpenAI pattern) enables reproducibility
- Evolution techniques for synthetic data improve diversity and difficulty

Sources:
- [Bloom Framework](https://alignment.anthropic.com/2025/bloom-auto-evals/)
- [Bloom GitHub](https://github.com/safety-research/bloom)
- [Anthropic-OpenAI Joint Evaluation](https://alignment.anthropic.com/2025/openai-findings/)
- [OpenAI Evals](https://github.com/openai/evals)
- [DeepEval](https://deepeval.com/)

---

## 5. Data Format Standards Summary

### Common formats across the ecosystem:

| Format | Used by | Best for |
|--------|---------|----------|
| JSONL | OpenAI, DeepEval, most eval frameworks | Eval datasets, streaming, append-only logs |
| Parquet | HuggingFace, Databricks/Lilac | Large datasets, columnar queries, storage efficiency |
| Arrow | HuggingFace (internal) | In-memory processing, zero-copy reads |
| CSV | Legacy, simple datasets | Small datasets, human readability |
| ShareGPT JSON | Multi-turn conversations | Chat/conversation training data |
| Alpaca JSON | Instruction datasets | Prompt-response pairs |
| YAML | OpenAI evals, HELM | Eval configuration and metadata |

### Conversation format patterns:

**Chat format (dominant):**
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**ShareGPT format:**
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**Eval format (typical):**
```json
{"input": "...", "expected_output": "...", "metadata": {...}, "tags": [...]}
```

---

## 6. Key Patterns and Recommendations for Evalyn

### Must-have capabilities (table stakes):

1. **JSONL + Parquet support** - read/write both formats natively
2. **Dataset versioning** - auto-version on every mutation, tag significant versions
3. **Structured metadata** - dataset cards with license, source, description, schema
4. **Train/test/validation splits** - with contamination-aware split management
5. **Multi-metric evaluation** - not just accuracy; calibration, robustness, fairness
6. **Streaming/lazy loading** - iterate without full download for large datasets

### High-value differentiators:

1. **Contamination detection** - built-in checks for overlap between eval data and training data
2. **Active learning annotation** - suggest which items to annotate next based on uncertainty
3. **Programmatic labeling** - define annotation rules as code (Snorkel pattern)
4. **Persona-based synthetic generation** - structured persona definitions driving diverse test data
5. **Adversarial augmentation** - systematically increase dataset difficulty
6. **Quality scoring pipeline** - multi-dimensional quality assessment of dataset items

### Architectural patterns to adopt:

1. **Dataset-as-Git-repo** (HuggingFace) - version control for datasets
2. **Auto-versioning on mutation** (LangSmith) - every edit creates a version
3. **Trace-to-dataset pipeline** (LangSmith) - flag production traces, add to eval set
4. **Four-stage generation** (Bloom) - understand, ideate, rollout, judge
5. **Annotation queue flywheel** (LangSmith/Argilla) - human feedback improves both model and dataset
6. **Dynamic benchmarking** (MMLU-CF) - evolve benchmarks to resist contamination

### Integration priorities:

1. HuggingFace Datasets - load/push datasets from/to Hub
2. JSONL/Parquet - standard interchange formats
3. OpenAI evals format - compatibility with existing eval infrastructure
4. ShareGPT conversation format - standard for multi-turn data
