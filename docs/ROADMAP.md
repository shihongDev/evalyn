# Evalyn Roadmap: Human-in-the-Loop Calibration Pipeline

## Vision

Build an end-to-end evaluation system where:
1. LLM judges evaluate agent outputs automatically
2. Humans annotate and validate judgements
3. System automatically tunes judge prompts to align with human preferences
4. Continuous improvement through iterative calibration

---

## Current State (v0.1) ✅

### Data Model
```
DatasetItem:
  - input        # User input to agent
  - output       # Agent response (from trace)
  - human_label  # Human annotation (optional)
  - metadata     # call_id, trace info
```

### Implemented Features
- [x] `@eval` decorator for tracing
- [x] Dataset building from traces (`build-dataset`)
- [x] Objective metrics (20+ trace-compatible)
- [x] Subjective metrics with GeminiJudge (7 templates)
- [x] Evaluation runner with progress bar (`run-eval`)
- [x] Basic annotation import (`import-annotations`)

---

## Phase 1: Human Annotation UI (v0.2) ✅

### Goal
Enable humans to easily review and annotate (input, output, eval_result) pairs.

### Features

#### 1.1 Annotation Export Format
```json
{
  "id": "item-123",
  "input": {"query": "What is ML?"},
  "output": "Machine learning is...",
  "eval_results": {
    "toxicity_safety": {"passed": true, "score": 1.0, "reason": "Safe"},
    "helpfulness": {"passed": true, "score": 0.9, "reason": "Helpful"}
  },
  "human_label": null  // To be filled by annotator
}
```

#### 1.2 Annotation Schema
```json
{
  "human_label": {
    "passed": true,           // Overall pass/fail
    "scores": {               // Per-metric human scores
      "toxicity_safety": 1.0,
      "helpfulness": 0.8
    },
    "notes": "Good response but could be more detailed",
    "annotator": "alice@company.com",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 1.3 CLI Commands
```bash
# Export dataset for annotation (with eval results)
evalyn export-for-annotation --dataset data/myproj --output annotations.jsonl

# Import completed annotations
evalyn import-annotations --path annotations_completed.jsonl

# View annotation coverage
evalyn annotation-stats --dataset data/myproj
```

### Tasks
- [x] Add `export-for-annotation` command
- [x] Define annotation JSON schema (`HumanLabel`, `AnnotationItem` in models.py)
- [x] Add annotation stats/coverage reporting (`annotation-stats` command)
- [ ] (Optional) Simple web UI for annotation

---

## Phase 2: Calibration Analysis (v0.3)

### Goal
Identify where LLM judges disagree with human judgement.

### Features

#### 2.1 Disagreement Detection
```bash
evalyn calibrate-analyze --dataset data/myproj --metric toxicity_safety

# Output:
# Calibration Report: toxicity_safety
# ────────────────────────────────────
# Total items with human labels: 150
# Agreement rate: 82.0%
#
# Disagreements by type:
#   False Positives (LLM=FAIL, Human=PASS): 12 (8%)
#   False Negatives (LLM=PASS, Human=FAIL): 15 (10%)
#
# Top disagreement patterns:
#   - "Technical jargon flagged as unclear" (5 cases)
#   - "Concise answers marked incomplete" (4 cases)
```

#### 2.2 Disagreement Export
```json
{
  "item_id": "item-123",
  "input": {"query": "..."},
  "output": "...",
  "llm_judgement": {"passed": false, "reason": "Too brief"},
  "human_judgement": {"passed": true, "notes": "Concise is fine here"},
  "disagreement_type": "false_negative"
}
```

#### 2.3 Calibration Metrics
- **Agreement Rate**: % of items where LLM and human agree
- **Cohen's Kappa**: Agreement adjusted for chance
- **Precision/Recall**: For each pass/fail class
- **Confidence Calibration**: Score distribution vs actual pass rate

### Tasks
- [ ] Implement `calibrate-analyze` command
- [ ] Add disagreement detection logic
- [ ] Generate calibration reports
- [ ] Export disagreement cases for review

---

## Phase 3: GEPA - Genetic/Evolutionary Prompt Alignment (v0.4)

### Goal
Automatically tune LLM judge prompts to better align with human judgement.

### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│  GEPA: Genetic Evolutionary Prompt Alignment                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Initial Population                                          │
│     Generate N prompt variants from base prompt                 │
│                                                                 │
│  2. Fitness Evaluation                                          │
│     Run each prompt on human-labeled dataset                    │
│     Score = agreement with human judgement                      │
│                                                                 │
│  3. Selection                                                   │
│     Keep top K prompts with highest alignment                   │
│                                                                 │
│  4. Mutation/Crossover                                          │
│     LLM generates new variants from top prompts                 │
│     Incorporate disagreement patterns as feedback               │
│                                                                 │
│  5. Iterate until convergence                                   │
│     Target: >90% agreement with human labels                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Algorithm

```python
def gepa_optimize(
    base_prompt: str,
    human_labeled_dataset: List[DatasetItem],
    metric_id: str,
    generations: int = 10,
    population_size: int = 8,
    top_k: int = 3,
) -> str:
    """
    Evolve judge prompt to maximize alignment with human labels.
    """
    # 1. Initialize population
    population = generate_initial_variants(base_prompt, population_size)

    for gen in range(generations):
        # 2. Evaluate fitness (agreement with humans)
        scores = []
        for prompt in population:
            judge = GeminiJudge(prompt=prompt)
            agreement = evaluate_agreement(judge, human_labeled_dataset)
            scores.append((prompt, agreement))

        # 3. Select top performers
        top_prompts = sorted(scores, key=lambda x: -x[1])[:top_k]

        # 4. Check convergence
        if top_prompts[0][1] >= 0.95:
            return top_prompts[0][0]  # 95% agreement reached

        # 5. Generate next generation
        # - Analyze disagreement patterns
        # - Ask LLM to create improved variants
        disagreements = get_disagreement_patterns(top_prompts[0][0], human_labeled_dataset)
        population = evolve_prompts(top_prompts, disagreements, population_size)

    return top_prompts[0][0]
```

### Prompt Evolution Strategies

#### 3.1 Mutation via LLM
```
Given this judge prompt that achieves 78% agreement with human labels:
{current_prompt}

These are cases where the judge disagreed with humans:
{disagreement_examples}

Generate an improved prompt that would correctly handle these cases
while maintaining accuracy on other cases.
```

#### 3.2 Crossover
```
Combine the best aspects of these two high-performing prompts:

Prompt A (82% agreement): {prompt_a}
Prompt B (80% agreement): {prompt_b}

Create a new prompt that incorporates strengths from both.
```

#### 3.3 Targeted Refinement
```
The judge is making these systematic errors:
- False positives on: {fp_patterns}
- False negatives on: {fn_patterns}

Modify the rubric to address these specific issues.
```

### CLI Commands
```bash
# Run GEPA optimization
evalyn gepa-optimize \
  --metric toxicity_safety \
  --dataset data/myproj \
  --generations 10 \
  --population-size 8 \
  --target-agreement 0.90

# Output:
# GEPA Optimization: toxicity_safety
# ──────────────────────────────────
# Generation 1: Best agreement = 0.78
# Generation 2: Best agreement = 0.82
# Generation 3: Best agreement = 0.85
# Generation 4: Best agreement = 0.89
# Generation 5: Best agreement = 0.92 ✓ Target reached!
#
# Optimized prompt saved to: data/myproj/prompts/toxicity_safety_v2.txt

# Compare original vs optimized
evalyn gepa-compare \
  --metric toxicity_safety \
  --original-prompt prompts/v1.txt \
  --optimized-prompt prompts/v2.txt \
  --dataset data/myproj
```

### Tasks
- [ ] Design GEPA algorithm
- [ ] Implement prompt mutation via LLM
- [ ] Implement fitness evaluation
- [ ] Add convergence detection
- [ ] Create `gepa-optimize` command
- [ ] Add prompt versioning/history

---

## Phase 4: Continuous Calibration Loop (v0.5)

### Goal
Enable ongoing improvement through iterative human feedback.

### Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CONTINUOUS CALIBRATION LOOP                                             │
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│  │  Trace  │───>│  Eval   │───>│  Human  │───>│  GEPA   │──┐            │
│  │ Collect │    │  Run    │    │ Review  │    │  Tune   │  │            │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │            │
│       ^                                                     │            │
│       └─────────────────────────────────────────────────────┘            │
│                                                                          │
│  After GEPA tuning:                                                      │
│  - New judge prompt deployed                                             │
│  - Re-evaluate existing dataset                                          │
│  - Human reviews edge cases                                              │
│  - Iterate until alignment is stable                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Features

#### 4.1 Prompt Versioning
```bash
evalyn prompt-history --metric toxicity_safety

# Version  Agreement  Created      Status
# v1       0.78       2024-01-10   archived
# v2       0.85       2024-01-15   archived
# v3       0.92       2024-01-20   active
```

#### 4.2 Drift Detection
```bash
evalyn calibration-drift --metric toxicity_safety --window 7d

# Alert: Agreement dropped from 0.92 to 0.84 in last 7 days
# Possible causes:
# - Distribution shift in inputs
# - New edge cases not covered by prompt
# Recommendation: Run GEPA with recent human labels
```

#### 4.3 Active Learning
```bash
# Suggest items most valuable for human annotation
evalyn suggest-annotations --dataset data/myproj --strategy uncertainty

# Items with highest annotation value:
# 1. item-456: Judge confidence = 0.52 (borderline)
# 2. item-789: Similar to past disagreements
# 3. item-012: Novel input pattern
```

### Tasks
- [ ] Implement prompt versioning
- [ ] Add drift detection
- [ ] Create active learning sampler
- [ ] Build calibration dashboard (optional)

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Production Agent                                                       │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────┐     ┌─────────────┐     ┌──────────────┐                  │
│  │ @eval   │────>│   SQLite    │────>│  Dataset     │                  │
│  │ Traces  │     │   Storage   │     │  (JSONL)     │                  │
│  └─────────┘     └─────────────┘     └──────────────┘                  │
│                                             │                           │
│                                             ▼                           │
│                                      ┌──────────────┐                  │
│                                      │   run-eval   │                  │
│                                      │  (metrics)   │                  │
│                                      └──────────────┘                  │
│                                             │                           │
│                        ┌────────────────────┼────────────────────┐     │
│                        ▼                    ▼                    ▼     │
│                  ┌──────────┐        ┌──────────┐         ┌──────────┐│
│                  │Objective │        │Subjective│         │  Human   ││
│                  │ Metrics  │        │ (Judge)  │         │  Review  ││
│                  └──────────┘        └──────────┘         └──────────┘│
│                        │                    │                    │     │
│                        └────────────────────┼────────────────────┘     │
│                                             ▼                           │
│                                      ┌──────────────┐                  │
│                                      │  Calibrate   │                  │
│                                      │  (Compare)   │                  │
│                                      └──────────────┘                  │
│                                             │                           │
│                                             ▼                           │
│                                      ┌──────────────┐                  │
│                                      │    GEPA      │                  │
│                                      │ (Tune Prompt)│                  │
│                                      └──────────────┘                  │
│                                             │                           │
│                                             ▼                           │
│                                      ┌──────────────┐                  │
│                                      │  Improved    │                  │
│                                      │   Judge      │──────> Loop back │
│                                      └──────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

| Phase | Feature | Priority | Effort |
|-------|---------|----------|--------|
| 1.1 | Export for annotation | High | Low |
| 1.2 | Annotation schema | High | Low |
| 1.3 | Import annotations | Done | - |
| 2.1 | Disagreement detection | High | Medium |
| 2.2 | Calibration reports | High | Medium |
| 3.1 | GEPA algorithm | High | High |
| 3.2 | Prompt mutation | Medium | Medium |
| 3.3 | CLI commands | Medium | Low |
| 4.1 | Prompt versioning | Low | Medium |
| 4.2 | Drift detection | Low | Medium |
| 4.3 | Active learning | Low | High |

---

## Open Questions

1. **GEPA Population Size**: How many prompt variants per generation?
2. **Convergence Criteria**: When to stop? 90% agreement? Plateau detection?
3. **Human Annotation Budget**: How many labels needed for reliable calibration?
4. **Prompt Storage**: Version control? Database? Files?
5. **Multi-metric Calibration**: Optimize prompts independently or jointly?

---

## Next Steps

1. [ ] Review and finalize Phase 1 annotation format
2. [ ] Implement `export-for-annotation` command
3. [ ] Design GEPA fitness function in detail
4. [ ] Create prototype of GEPA optimization loop
5. [ ] Test with real human annotations
