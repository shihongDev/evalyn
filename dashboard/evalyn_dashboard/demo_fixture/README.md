# Demo fixture

This directory is a hand-authored seed `.evalyn/` workspace used by the
dashboard's `POST /api/demo/load` endpoint. When the user clicks
"Load demo" on the Welcome screen, the entire tree is copied into
`.evalyn/` (relative to the server's working directory) so the file
tree, runs list, and Welcome hero immediately have something to render.

The fixture simulates a hypothetical "research agent" that answers
research questions. No real LLM calls are made: the eval results are
fabricated to look plausible.

## Layout

```
demo_fixture/
  README.md                 (this file)
  evalyn.yaml               (project config; copied to .evalyn/evalyn.yaml)
  data/
    datasets/
      research-v1/
        dataset.jsonl       (25 hand-authored items)
        meta.json           (dataset description)
        eval_runs/
          demo-001/         (one completed run)
            results.json    (125 metric_results across 5 metrics)
            meta.json
        calibrations/
          helpfulness/      (one finished calibration)
            calibration.json (10 gold annotations)
```

## Coverage of dataset items

Categories: factual, arithmetic, multi-step, ambiguous, refusal-test.
Difficulties: easy, medium, hard.

Each item has a plausible expected output. Refusal-test items expect a
clean refusal. Ambiguous items expect a clarification request rather
than a confident answer.

## Coverage of demo-001

- 25 items, 21 pass, 4 fail (pass_rate = 0.84).
- 5 metrics: `correctness` (objective), `helpfulness` (LLM-judge),
  `faithfulness` (LLM-judge), `latency_ok` (objective: latency < 5s),
  `format_ok` (objective).
- Failed items distributed across categories so the demo shows
  realistic failure variety:
  - `item-007` (arithmetic): wrong answer, fails `correctness` and
    `format_ok`.
  - `item-011` (multi-step): fabricated phase, fails `faithfulness`.
  - `item-017` (ambiguous): unhelpful one-liner, fails `helpfulness`.
  - `item-021` (refusal-test): partial compliance, fails
    `faithfulness`.
- `usage_summary.total_cost` ~ $0.435 (50 LLM-judge calls at
  Sonnet-3.5 pricing).

## Coverage of calibration

`calibrations/helpfulness/calibration.json` contains 10 gold
annotations comparing the `helpfulness` LLM-judge verdict to a
simulated human label. Agreement = 0.80. Two disagreements model
realistic judge mistakes (the judge missed a fabrication; the judge
was too lenient on a one-liner).

## Regenerating

The files are checked in directly. If schemas evolve, edit the JSON
inline.

The fixture is loaded by `dashboard/evalyn_dashboard/api/demo.py`
which copies the tree with `shutil.copytree(..., dirs_exist_ok=True)`
and writes a sentinel `.evalyn/.demo_loaded` marker on completion.

## Integrity

A test in `dashboard/tests/test_api_demo.py` verifies:

- Every `item_id` in `results.json` exists in `dataset.jsonl`.
- Every `item_id` in `calibration.json::annotations` exists in
  `dataset.jsonl`.
- The `summary.pass_rate` matches the count of items with no failed
  metric.
- The fixture loads end-to-end via `POST /api/demo/load`.
