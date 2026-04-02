# workflow

Show the evaluation workflow and context-aware next steps.

## Usage

```bash
evalyn workflow
```

## Options

No options. The command automatically detects your current state and suggests next steps.

## Description

The `workflow` command prints the full evaluation pipeline and suggests what to do next based on your current progress:

**Phase 1: COLLECT**
1. Add `@eval` decorator to your agent function
2. Run your agent to collect traces
3. Build a dataset from traces

**Phase 2: EVALUATE**
4. Select metrics for evaluation
5. Run evaluation
6. Analyze results

**Phase 3: CALIBRATE** (optional)
7. Annotate results (human feedback)
8. Calibrate LLM judges
9. Re-evaluate with calibrated prompts

The command checks your trace storage and shows context-aware suggestions - for example, if you have traces for specific projects, it suggests the `build-dataset` command with the right project name.

## Examples

```bash
# Show workflow and next steps
evalyn workflow
```

## See Also

- [quickstart](quickstart.md) - Guided first-run setup
- [one-click](one-click.md) - Run the full pipeline automatically
- [status](status.md) - Show dataset status overview
