# Evalyn Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 4 Claude Code skills that guide developers through the evalyn evaluation workflow, with data-driven recommendations.

**Architecture:** Sequential skill chain (evalyn-setup -> evalyn-eval -> evalyn-analyze -> evalyn-calibrate). Each skill is a self-contained SKILL.md with pre-flight checks and hand-off logic. Skills are technique/reference type - tested with application scenarios via subagents.

**Tech Stack:** Claude Code skills (Markdown with YAML frontmatter), tested with Agent tool subagents.

---

### Task 1: Create directory structure

**Files:**
- Create: `sdk/skills/evalyn-setup/SKILL.md` (placeholder)
- Create: `sdk/skills/evalyn-eval/SKILL.md` (placeholder)
- Create: `sdk/skills/evalyn-analyze/SKILL.md` (placeholder)
- Create: `sdk/skills/evalyn-calibrate/SKILL.md` (placeholder)

**Step 1: Create skill directories and placeholder files**

```bash
mkdir -p sdk/skills/evalyn-setup sdk/skills/evalyn-eval sdk/skills/evalyn-analyze sdk/skills/evalyn-calibrate
```

Create each placeholder with just the frontmatter:

`sdk/skills/evalyn-setup/SKILL.md`:
```markdown
---
name: evalyn-setup
description: Use when setting up evalyn evaluation for an LLM agent project, instrumenting agent code, or adding the evalyn decorator
---

# evalyn-setup

TODO
```

`sdk/skills/evalyn-eval/SKILL.md`:
```markdown
---
name: evalyn-eval
description: Use when building evaluation datasets, selecting metrics, or running evaluations on an LLM agent project with evalyn
---

# evalyn-eval

TODO
```

`sdk/skills/evalyn-analyze/SKILL.md`:
```markdown
---
name: evalyn-analyze
description: Use when analyzing evalyn evaluation results, investigating failures, comparing runs, or understanding agent performance
---

# evalyn-analyze

TODO
```

`sdk/skills/evalyn-calibrate/SKILL.md`:
```markdown
---
name: evalyn-calibrate
description: Use when LLM judges need calibration, evaluation metrics seem misaligned with expectations, or annotation and judge tuning is needed
---

# evalyn-calibrate

TODO
```

**Step 2: Commit**

```bash
git add sdk/skills/
git commit -m "chore: scaffold evalyn skill directories"
```

---

### Task 2: RED baseline for evalyn-setup

**Goal:** Run a subagent WITHOUT the skill to see how Claude handles "help me evaluate my agent" with no guidance.

**Step 1: Run baseline subagent test**

Use the Agent tool to spawn a subagent with this prompt (no skill loaded):

```
You are helping a developer set up evalyn to evaluate their LLM agent.
The project is at /mnt/c/Users/shiho/Desktop/projects/evalyn and has
an example agent at example_agents/.

The user says: "I want to evaluate my langchain deep research agent.
Help me set up evalyn."

DO NOT write any code. Instead, describe step by step what you would do,
including exact commands you would run and what code changes you would make.
Think through: what do you check first? How do you detect the framework?
What decorator do you add? How do you verify it works?
```

**Step 2: Document baseline failures**

Record what the subagent gets wrong or misses:
- Does it check if evalyn is installed?
- Does it detect the framework correctly?
- Does it know the decorator syntax?
- Does it know to verify traces with `evalyn list-calls`?
- Does it suggest the right next step?

Save findings as notes for writing the skill.

---

### Task 3: GREEN - write evalyn-setup skill

**Files:**
- Modify: `sdk/skills/evalyn-setup/SKILL.md`

**Step 1: Write the skill based on baseline failures**

The skill should cover:
- Pre-flight: check evalyn installed, check evalyn.yaml exists
- Detect agent framework by scanning imports (langchain, anthropic, openai, google ADK, claude agent SDK)
- Framework detection table mapping imports to instrumentation approach
- Add evalyn decorator to agent entry function with project/version params
- User runs agent, verify traces with `evalyn list-calls`
- Inspect trace with `evalyn show-trace --last -v`
- Hand-off to evalyn-eval

Key details:
- Decorator import: `from evalyn_sdk import eval`
- Decorator usage: `@eval(project="<name>", version="v1")`
- All supported frameworks are auto-instrumented - decorator is sufficient
- project should be kebab-case, version starts at "v1"
- Wrap outermost function only, not internal helpers

**Step 2: Run subagent WITH skill to verify improvement**

Use Agent tool - same prompt as Task 2 baseline, but include the skill content in the system prompt. Verify the subagent now follows the correct flow.

**Step 3: Commit**

```bash
git add sdk/skills/evalyn-setup/SKILL.md
git commit -m "feat: add evalyn-setup skill for agent instrumentation"
```

---

### Task 4: RED baseline for evalyn-eval

**Step 1: Run baseline subagent test**

Agent prompt (no skill):
```
You are helping a developer who has evalyn traces for their "research-agent" project.
They ran `evalyn list-calls` and see 15 traces.

The user says: "I have traces now. Help me evaluate my agent."

DO NOT write any code. Describe step by step what you would do.
Include exact evalyn commands with flags. How do you pick the right
metrics? What mode do you use for suggest-metrics? How do you decide
which bundle fits?
```

**Step 2: Document baseline failures**

Record what the subagent misses:
- Does it build the dataset first?
- Does it inspect traces to recommend a bundle?
- Does it use the right suggest-metrics mode?
- Does it know to append with llm-registry?

---

### Task 5: GREEN - write evalyn-eval skill

**Files:**
- Modify: `sdk/skills/evalyn-eval/SKILL.md`

**Step 1: Write the skill**

The skill should cover:
- Pre-flight: verify traces exist with `evalyn list-calls`, check for existing dataset
- Build dataset: `evalyn build-dataset --project <name>`
- Auto-recommend metrics by analyzing trace structure:
  - Tool calls -> orchestrator or multi-step-agent bundle
  - URLs/citations -> research-agent or rag-qa bundle
  - Multi-turn -> chatbot bundle
  - Code output -> code-assistant bundle
  - Short outputs -> summarization bundle
  - Educational content -> tutor bundle
  - Content generation -> content-writer bundle
- Two-pass metric selection:
  1. `evalyn suggest-metrics --dataset <path> --mode bundle --bundle <recommended>`
  2. `evalyn suggest-metrics --dataset <path> --mode llm-registry --append`
- Run evaluation: `evalyn run-eval --dataset <path>`
- Hand-off to evalyn-analyze

Key principle: data-driven recommendations from trace inspection, not user questions.

**Step 2: Test with subagent, verify improvement**

**Step 3: Commit**

```bash
git add sdk/skills/evalyn-eval/SKILL.md
git commit -m "feat: add evalyn-eval skill for dataset building and evaluation"
```

---

### Task 6: RED baseline for evalyn-analyze

**Step 1: Run baseline subagent test**

Agent prompt (no skill):
```
You are helping a developer who just ran evalyn evaluation on their agent.
The run ID is "220e8590". Some metrics passed, some failed:
- factual_accuracy: 80% pass rate
- output_nonempty: 100%
- helpfulness_accuracy: 96%
- latency_ms: N/A (numeric metric)

The user says: "What do my eval results mean? What should I do next?"

DO NOT write any code. Describe what evalyn commands you would run
and how you would interpret the results. What thresholds matter?
When should the user calibrate vs fix their agent?
```

**Step 2: Document baseline failures**

---

### Task 7: GREEN - write evalyn-analyze skill

**Files:**
- Modify: `sdk/skills/evalyn-analyze/SKILL.md`

**Step 1: Write the skill**

The skill should cover:
- Pre-flight: verify eval runs exist with `evalyn list-runs`
- Analysis cascade:
  1. `evalyn analyze --run <id>` for metric summary
  2. `evalyn insights --run <id>` for diagnostic analysis
  3. `evalyn compare` if multiple runs exist
  4. `evalyn trend` for longer history
  5. `evalyn cluster-failures --run <id>` if failure rate warrants
- Interpretation table:
  - Above 95%: agent performing well, suggest simulate for edge cases
  - 80-95%: moderate issues, could be agent OR judge misalignment, consider calibrate
  - Below 80%: significant issues, invoke evalyn-calibrate
- Key distinction: low pass rates can mean agent is bad OR judges are misaligned
- Export option: `evalyn export --run <id> --format html`
- Conditional hand-off based on results

**Step 2: Test with subagent, verify improvement**

**Step 3: Commit**

```bash
git add sdk/skills/evalyn-analyze/SKILL.md
git commit -m "feat: add evalyn-analyze skill for result analysis and interpretation"
```

---

### Task 8: RED baseline for evalyn-calibrate

**Step 1: Run baseline subagent test**

Agent prompt (no skill):
```
You are helping a developer whose evalyn evaluation shows factual_accuracy
at 80% pass rate. They suspect the LLM judge is too strict.

The user says: "I think the judge is wrong on some of these. How do I fix it?"

DO NOT write any code. Describe what evalyn commands you would run.
How does annotation work? What optimizer should they use? How do they
verify calibration helped?
```

**Step 2: Document baseline failures**

---

### Task 9: GREEN - write evalyn-calibrate skill

**Files:**
- Modify: `sdk/skills/evalyn-calibrate/SKILL.md`

**Step 1: Write the skill**

The skill should cover:
- Pre-flight: verify eval runs exist, identify calibration targets (subjective metrics with low pass rates)
- Annotation guidance:
  - `evalyn annotate --run <id> --per-metric`
  - Interactive terminal session, explain commands (y/n/s/v/q)
  - Recommend 20-30 items minimum
- Calibration:
  - Start with `evalyn calibrate --metric-id <id> --optimizer basic`
  - Optimizer comparison table (basic/ape/opro/gepa)
  - If basic insufficient, escalate to gepa
- Re-evaluate: `evalyn run-eval --dataset <path> --use-calibrated`
- Compare: `evalyn compare --run1 <original> --run2 <calibrated>`
- Troubleshooting if still poor: more annotations, different optimizer, manual rubric editing, cluster-misalignments
- Terminal skill - user decides next steps (iterate, fix agent, expand testing, ship)

**Step 2: Test with subagent, verify improvement**

**Step 3: Commit**

```bash
git add sdk/skills/evalyn-calibrate/SKILL.md
git commit -m "feat: add evalyn-calibrate skill for annotation and judge calibration"
```

---

### Task 10: Integration test across the full chain

**Step 1: Run end-to-end subagent test**

Spawn a subagent with ALL four skills loaded, prompt:
```
You have access to evalyn skills: evalyn-setup, evalyn-eval, evalyn-analyze, evalyn-calibrate.

A developer says: "I just built a LangChain research agent that searches
the web and summarizes findings. I want to evaluate how good it is.
Walk me through the entire process."

Describe the full workflow you would follow, referencing specific skills
at each stage. Include exact commands.
```

Verify:
- Correct skill ordering (setup -> eval -> analyze -> calibrate)
- Correct hand-off points
- Correct framework detection (LangChain -> auto-instrumented)
- Correct bundle recommendation (research-agent)
- Correct interpretation of results
- Correct calibration guidance

**Step 2: Fix any gaps found, commit**

---

### Task 11: Deploy to personal skills and update README

**Step 1: Copy skills to personal directory**

```bash
cp -r sdk/skills/evalyn-setup ~/.claude/skills/
cp -r sdk/skills/evalyn-eval ~/.claude/skills/
cp -r sdk/skills/evalyn-analyze ~/.claude/skills/
cp -r sdk/skills/evalyn-calibrate ~/.claude/skills/
```

**Step 2: Add install instructions to README.md**

Add a section under Documentation with a table of skills and copy instructions.

**Step 3: Final commit**

```bash
git add sdk/skills/ README.md
git commit -m "feat: add evalyn Claude Code skills for guided evaluation workflow"
```
