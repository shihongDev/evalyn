# Evalyn Skill Design

## Goal

A set of Claude Code skills that guide developers through the evalyn evaluation workflow: from instrumenting their agent to analyzing results and calibrating judges. Skills auto-recommend parameters by inspecting the user's actual data.

## Target User

Developer using Claude Code who wants help evaluating their LLM agent project.

## Architecture: Sequential Skill Chain

Four self-contained skills, each with pre-flight checks and hand-off to the next stage.

```
evalyn-setup -> evalyn-eval -> evalyn-analyze -> evalyn-calibrate
```

No parent dispatcher. Each skill ends with a conditional recommendation for what to invoke next.

## File Layout

```
sdk/skills/
  evalyn-setup/SKILL.md
  evalyn-eval/SKILL.md
  evalyn-analyze/SKILL.md
  evalyn-calibrate/SKILL.md
```

Users copy to `~/.claude/skills/` for personal use across projects.

## Skill Designs

### evalyn-setup

**Trigger:** "evaluate my agent", "add evalyn", "set up evaluation", "instrument my code"

**Flow:**
1. Pre-flight: check evalyn_sdk installed (`uv pip show evalyn_sdk`)
2. Detect agent framework by scanning imports (langchain, anthropic SDK, openai, google ADK, etc.)
3. Add `@eval` decorator to agent's entry function
4. User runs agent, skill verifies traces with `evalyn list-calls`
5. Inspect traces with `evalyn show-trace --last -v`
6. Hand-off: "invoke evalyn-eval"

**Framework detection logic:**
- LangChain -> auto-instrumentation via callback handler
- Claude Agent SDK -> hook-based integration
- Google ADK -> automatic callback injection
- Raw API calls -> `@eval` decorator sufficient

### evalyn-eval

**Trigger:** "run evaluation", "evaluate my dataset", "select metrics", "build dataset"

**Flow:**
1. Pre-flight: `evalyn list-calls --limit 5` to verify traces exist. No traces -> "invoke evalyn-setup"
2. Build dataset: `evalyn build-dataset --project <detected-project>`
3. Auto-recommend metrics by analyzing trace structure:
   - Tool calls -> `orchestrator` or `multi-step-agent` bundle
   - Citations/URLs -> `research-agent` or `rag-qa` bundle
   - Multi-turn -> `chatbot` bundle
   - Code output -> `code-assistant` bundle
   - Short outputs -> `summarization` bundle
4. Run `evalyn suggest-metrics --dataset <path> --mode bundle --bundle <recommended>`
5. Append: `evalyn suggest-metrics --dataset <path> --mode llm-registry --append`
6. Run `evalyn run-eval --dataset <path>`
7. Hand-off: "invoke evalyn-analyze"

**Key principle:** data-driven recommendations, not questions.

### evalyn-analyze

**Trigger:** "analyze results", "show evaluation results", "what failed", "how did my agent do"

**Flow:**
1. Pre-flight: `evalyn list-runs --limit 1`. No runs -> "invoke evalyn-eval"
2. Run `evalyn analyze --run <latest>`
3. Run `evalyn insights --run <latest>`
4. If multiple runs: `evalyn compare` and `evalyn trend`
5. If failure rate warrants: `evalyn cluster-failures --run <latest>`
6. Interpret results, highlight low pass rates
7. Conditional hand-off:
   - Above 95%: "your agent is performing well"
   - 80-95%: "consider invoking evalyn-calibrate"
   - Below 80%: "invoke evalyn-calibrate"
8. Offer `evalyn export --run <latest> --format html`

### evalyn-calibrate

**Trigger:** "calibrate judges", "annotate results", "improve evaluation", "judges are wrong"

**Flow:**
1. Pre-flight: verify eval runs exist
2. Identify subjective metrics with low pass rates as calibration targets
3. Guide annotation: `evalyn annotate --run <latest> --per-metric` (interactive - user responds in terminal)
4. Run calibration: `evalyn calibrate --metric-id <target> --optimizer basic`
5. If basic insufficient, suggest `--optimizer gepa` or `--optimizer opro`
6. Re-evaluate: `evalyn run-eval --dataset <path> --use-calibrated`
7. Compare: `evalyn compare --run1 <original> --run2 <calibrated>`
8. Terminal skill - user decides whether to iterate

## Design Principles

- Skills read user's actual data to make recommendations (no abstract questions)
- Each skill is self-contained with its own pre-flight check
- No config management - just workflow commands
- Progressive depth: setup -> eval -> analyze -> calibrate
- Skill descriptions focus on triggering conditions only (per CSO best practices)

## Testing Plan (TDD)

For each skill:
1. RED: run pressure scenario with subagent WITHOUT skill, document baseline failures
2. GREEN: write skill addressing those failures, verify compliance
3. REFACTOR: close loopholes found during testing

Pressure scenarios to test:
- User has no traces (should route to setup)
- User has traces but no dataset (should build dataset)
- User has eval runs with mixed results (should give specific recommendations)
- User's agent uses multiple frameworks (should detect correctly)
- User wants to skip steps (skill should enforce pre-flight checks)
