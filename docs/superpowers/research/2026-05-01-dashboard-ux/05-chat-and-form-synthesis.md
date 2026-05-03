# Chat and CLI Form Redesign

## Executive summary

The two highest-touch surfaces of the Evalyn Dashboard - ChatPanel and CliForm - sit at opposite ends of the same user need: "I want to run an evalyn CLI and understand the result." Today both surfaces are visually competent (ported from a high-fidelity mock) but operationally weak. Chat is a one-shot text exchange with brittle confirmations; the form is a tall stack of fields with a ceremonial preview pane and no recall of what the user just did 30 seconds ago.

**Top 3 chat changes**

1. Replace text bubbles with a flat thread (Claude.ai-style) and make tool-call cards collapsible with a one-line `$ evalyn analyze --run abc` summary; today every tool result blasts up to 320px of pre-formatted text into the panel.
2. Rebuild the confirmation card around three concrete affordances: **(a)** show the exact argv the agent will run; **(b)** let the user edit args inline before approving; **(c)** offer "Approve once / Approve for session / Reject". Today the only choice is Approve / Reject and the args are not editable.
3. Persistent thread sidebar (collapsed by default), per-message actions (copy, retry from here, branch), and a "@-mention a run" affordance the composer placeholder already hints at but never delivers.

**Top 3 form changes**

1. Kill the 3-mode toggle. Default to a single "smart form" view that always shows the live command at the bottom (sticky, copyable, executable as raw). Promote the `raw` mode into a "Paste a CLI" reverse-parse import. The current `preview` mode wastes a 380px column on a fake cost/duration heuristic (`0.4 + filledCount * 0.18`).
2. Replace the bare path `<input>` with a real `.evalyn/` file picker for any field whose `kind === "path"`. The single most common error mode (per the heuristics in `introspect.py`) is a typo in `--dataset` or `--metrics`; the file picker eliminates it entirely.
3. First-class history surface: a "Recent runs of this CLI" strip above the form with one-click "Re-run with same args / one tweak / compare". The store already has `runHistory` and `editRunArgs` - the form just doesn't surface them.

---

## Part A: ChatPanel

### Today (read of ChatPanel.tsx)

`ChatPanel.tsx` is 585 lines, single file, all subcomponents inlined.

**Header** (lines 328-394). A serif "Ask agent" title, plus three icon buttons: New conversation (`＋` at L372), Settings gear (`⚙` at L381), and Hide chat (`×` at L390). The decorative double-circle SVG at L343-360 is pure visual ornament.

**Error banner** (lines 396-436). Shows `agent.error.message` with a "Open settings" button when the error kind is `auth` or `rate_limit` (L401).

**Message list** (lines 545-559). Renders each `ChatMessage` via `ChatTurn` (L311-326). Empty state is two grey monospace comments (L549-558): `// Ask anything about your evals.` and `// Tools call CLIs; writes require your approval.` There are no example prompts, no chips, nothing data-grounded.

**ChatTurn** (lines 311-326). 26x26 avatar (`YOU` text or accent `e` letter at L52-95) plus a flex column with optional text, optional `ToolCallCard`, optional `SuggestionCard[]`. There is no per-message timestamp, no copy button, no edit, no retry. Spacing is hardcoded `marginBottom: 18`.

**Markdownish renderer** (lines 98-144). Hand-rolled, supports only `**bold**` and `` `inline code` ``. No fenced code blocks, no lists, no links. Anything multiline pre-formatted is rendered as a single paragraph with `whiteSpace: pre-wrap` at L122.

**ToolCallCard** (lines 146-256). One bordered box per tool call with three regions:

- Header (L162-189): status icon (`✓ ✗ ▸ ·`), monospace command preview, status pill chip.
- Awaiting-confirmation strip (L191-221): "This command writes to disk. Approve to run?" with two buttons.
- Output pre block (L223-239): up to `maxHeight: 320`, scrollable, monospace 11px.
- Error block (L241-253): red text on `--fail-soft`.

The output block is **never collapsed**. Every completed tool call dumps up to 320px of stdout into the chat thread. With the 8-call tool budget, a single agent turn can fill the entire panel before the final summary is even visible.

**SuggestionCard** (lines 258-309). Accent-bordered button. Click writes args to `sessionStorage[cli:prefill:<cliId>]` (L266) and calls `openCli(suggestion.cliId)`. The "open form" copy is in monospace at L301: `open form: <cliId>`. Nothing about the click destination is visually obvious - it could equally be a link to docs or a code snippet.

**ChatComposer** (lines 438-515). One textarea (`rows={2}`), Enter sends (L451-456), Shift+Enter newlines, send button is an up-arrow at L501-510. The placeholder at L493 is `Ask anything · paste a CLI · @-mention a run` - both promises are aspirational; neither is implemented. The hint at L498-500 says `↵ send · ⌘K palette` but there is no `⌘K` handler in this file.

**Auto-scroll** (lines 525-528). Brute-force `scrollTop = scrollHeight` on every messages or status change. No "user scrolled up, pin in place" behavior - if you scroll back to read a tool result and a `text_delta` arrives, you snap to the bottom.

### Friction list

1. **Tool result avalanche.** A 320px-tall `<pre>` per call. After 4-5 calls the panel is unreadable. (L223-239.)
2. **No collapse / expand on tool cards.** Once rendered, output is always visible.
3. **No editable args before confirmation.** User can only Approve or Reject the agent's exact argv. If the agent picked the wrong `--dataset`, you reject, retype, hope. (L191-221.)
4. **Confirmation copy is generic.** "This command writes to disk. Approve to run?" is the same for `delete-traces` and `run-eval`. The latter takes 5 minutes and costs $$; the former is irreversible.
5. **No timestamps, no copy buttons, no retry.** Per-message actions are absent. (L311-326.)
6. **Markdown is too thin.** No fenced code blocks for the agent to render `results.json` snippets cleanly.
7. **Scroll behavior is hostile.** Auto-scroll on every event regardless of user intent. (L525-528.)
8. **Empty state is dead air.** Two grey comments, zero suggestions, zero example prompts. (L549-558.)
9. **Suggestion-card click destination is opaque.** "open form: run-eval" reads like a code identifier, not a CTA. (L301.)
10. **Composer claims "paste a CLI · @-mention a run" but neither works.** (L493.)
11. **No thread persistence, no thread list.** New conversation (`＋`) wipes everything. The agent has no recollection of last week's debugging session.
12. **Streaming cursor: there isn't one.** `text_delta` events update text inline; the only "thinking" indicator is a flat `thinking …` row at L563-578.
13. **Provider error banner has no retry.** "Open settings" is the only escape; there is no "Retry last message" button.
14. **No keyboard shortcuts on confirmation.** Approve/Reject require mouse clicks even though the panel is focused.

### Redesign proposal

#### A1. Message list - flat thread with per-message rail

**Why this hurts today.** Bubbles + 26px avatars create visual weight that does not map to information value. Claude.ai pioneered flat threads with subtle role markers, and the Evalyn dashboard's editorial design (cream/serif, dark mono) wants the same restraint.

**Current.**

```
[YOU]  user text in flat box, 13.5px sans
       (18px gap)
[ e ]  assistant text
       [tool-card: 320px stdout dump]
       [suggestion-card: orange button]
       (18px gap)
```

**Proposed.**

```
You · 14:32                                          [⎘ copy]
  Compare last 3 run-eval runs and find the regression.

Agent · 14:32 · gpt-5.1                              [⎘] [↻] [⤴︎ branch]
  I'll list recent runs, then analyze the regression.

  ▸ $ evalyn list-runs --limit 3                        complete · 0.4s
  ▸ $ evalyn analyze --run abc123 --compare prev       complete · 2.1s
  ▾ $ evalyn cluster-failures --run abc123              complete · 8.3s
    Cluster 1 (12 items): tool-use timeout in browser_navigate
    Cluster 2 (5 items):  off-topic refusals
    [show 47 more lines]

  Pass rate dropped 8pp on multi-turn items. Likely cause:
  the new system prompt added a refusal heuristic that fires
  on harmless follow-ups.

  [Open analyze report →]   [Open cluster report →]
```

**Microcopy.** Role label + timestamp + (assistant only) model id. Per-message rail: copy (`⎘`), retry-from-here (`↻`), branch (`⤴︎`). Tool calls collapsed by default, one-line `$ <cmd>` + status + duration. Click expands to first 200 lines, with `[show N more lines]` to fully open.

**Spacing.** Single 24px gap between turns; tool cards indented 12px under the assistant message. No avatars at all - role + time is enough.

#### A2. Tool-call rendering - collapsible by default

**Why this hurts today.** The 320px `<pre>` block on every completed tool call (ChatPanel.tsx L223-239) destroys scannability. The agent's actual reasoning gets buried.

**Current ASCII.**

```
+------------------------------------------------------+
| · $ list-runs --limit 3              complete   ✓  |
+------------------------------------------------------+
| run-2026-04-28-abc  pass=0.84  cost=$0.42  multi   |
| run-2026-04-29-def  pass=0.78  cost=$0.39  multi   |
| run-2026-04-30-ghi  pass=0.76  cost=$0.41  multi   |
| (47 more lines)                                      |
+------------------------------------------------------+
```

**Proposed (collapsed default).**

```
▸ $ evalyn list-runs --limit 3                  complete · 0.4s · 50 lines
```

**Proposed (expanded).**

```
▾ $ evalyn list-runs --limit 3                  complete · 0.4s · 50 lines
  ┌────────────────────────────────────────────────────┐
  │ run-2026-04-28-abc  pass=0.84  cost=$0.42  multi  │
  │ run-2026-04-29-def  pass=0.78  cost=$0.39  multi  │
  │ run-2026-04-30-ghi  pass=0.76  cost=$0.41  multi  │
  │ ... 47 more lines  [show all] [open in terminal]  │
  └────────────────────────────────────────────────────┘
  [⎘ copy output]   [open job:xyz123 in tab →]
```

**Per-card affordances.**

- Click chevron to toggle.
- Show first 8 lines as a peek (a "preview" of the output) when collapsed, in `text-2` color, monospace 10.5px - so the user knows whether to expand.
- "open job in tab" deep-links to the existing job-tab infrastructure already in `store.openJobTab`.
- Errored tool calls (red `✗`) auto-expand and stay open.

**Anthropic-pattern parallel.** Claude.ai's tool cards have exactly this shape: one-line `<tool>` name + status + duration, expandable with full input/output. Current Evalyn renders the input as a header (`previewCmd`) but never separates it from the output, and the output is always-on.

#### A3. Streaming - cursor + smart scroll

**Why this hurts today.** No visible cursor during streaming - text just appears. The "thinking" row (L563-578) is a separate component below the messages, which means the layout reflows when streaming starts.

**Proposed.**

- Streaming text gets a blinking caret (`▌`) at the end while `streaming: true` (the field already exists at `ChatMessage.streaming` per store.ts L988). One CSS animation, no React work.
- Auto-scroll only when the user is within 80px of the bottom. If they have scrolled up, show a small "↓ 3 new" pill at the bottom-right of the scroll area.
- The "thinking" row goes inside the assistant message slot instead of below it, so layout is stable.

```
Agent · 14:32 · gpt-5.1
  I'll list recent runs, then analyze the regress▌

[user has scrolled up]
                                          ┌──────────┐
                                          │ ↓ 3 new  │
                                          └──────────┘
```

**Backend implication.** None. `text_delta` events already exist (agent.py L770-773) and `streaming: true` is set in store.ts L987.

#### A4. Confirmations - the most consequential redesign

**Why this hurts today.** "This command writes to disk. Approve to run?" (ChatPanel.tsx L201) is generic and uneditable. For `run-eval` the user's actual decision involves: the dataset path, the metrics file, the model, the worker count, the cost. They cannot inspect or change any of these in the chat UI - they must reject, switch context to the form, retype.

The agent runtime (agent.py L876-895) only listens for an approve/reject boolean. Any args change requires the agent to re-propose.

**Current ASCII.**

```
+------------------------------------------------------+
| · $ run-eval --dataset evals/spam.jsonl  awaiting   |
+------------------------------------------------------+
| This command writes to disk. Approve to run?         |
|                              [Approve] [Reject]      |
+------------------------------------------------------+
```

**Proposed ASCII.**

```
▾ $ evalyn run-eval                          ⚠ awaiting confirmation
  ┌──────────────────────────────────────────────────────┐
  │ COMMAND                              [edit args] [⎘]  │
  │   evalyn run-eval                                     │
  │     --dataset    evals/spam.jsonl    [📁 pick]        │
  │     --metrics    metrics/v3.json     [📁 pick]        │
  │     --workers    8                                    │
  │     --provider   gemini                               │
  │                                                       │
  │ THIS WILL                                             │
  │   • Spawn an evalyn run-eval subprocess               │
  │   • Write to .evalyn/runs/<new-id>/                   │
  │   • Stream stdout to a job: tab                       │
  │   • Estimated cost: ~$0.42  (based on 240 items)     │
  │   • Estimated time: ~3-5 min                          │
  │                                                       │
  │ [Reject] [Approve once]  [Approve · don't ask again   │
  │                            this session for run-eval] │
  └──────────────────────────────────────────────────────┘
```

**Microcopy.** "THIS WILL" header, bulleted side-effects in plain English. Buttons are explicit: "Approve once" vs. "Approve · don't ask again this session for run-eval" - per-tool, per-session whitelist (not persisted to disk; cleared on reset).

**Args edit flow.** Click [edit args] to inline-mount a tiny form that reuses `ParamField` for each arg the agent proposed. On approve, send the edited args to `/api/agent/chat/{tid}/confirm`.

**Backend implications.**

1. `confirm` endpoint must accept an optional `args_override: dict` payload. Today it's `{approve: bool}` (spec §7).
2. New event type `auto_approve_session` so the runtime can skip confirmation for the current session for this tool. Stored in `_Thread` as a set of tool names.
3. `confirmation_required` event must include enough context to render the bullets. The runtime already has `args` and `preview_cmd`; we add a server-side `side_effects: list[str]` field per CLI (manually curated for the ~15 write CLIs in `agent.py`'s non-allowlist).

**Three-second-undo.** Skip. The subprocess starts immediately after approve, and "undo" maps to the existing cancel button on the job tab, which already does SIGTERM + 3s grace. Ship the cancel surface as part of the approval card instead:

```
  [Running · 0:12 elapsed]                  [Cancel]
```

#### A5. Errors - typed banners, recoverable inline

**Why this hurts today.** All errors collapse to the single banner at the top of the panel (L396-436), with one CTA (Open settings). Tool failures, provider 401s, rate limits, and timeouts all look the same.

**Proposed taxonomy.**

| Kind | Where it lands | Recovery |
|---|---|---|
| `auth` (provider 401) | Top banner, persistent until dismissed | "Open settings" + "Retry last message" |
| `rate_limit` | Top banner, with countdown | "Retry in 0:38" auto-counts; on retry, replays last message |
| `tool_failure` (subprocess exit != 0) | Inline in the tool card (red, expanded) | "Show full output", "Re-run with same args", "Open in form to tweak" |
| `timeout` (5min confirmation lapse) | Inline in the confirmation card | "Approve now" (still allowed if within 30s grace) |
| `network` (WS disconnect) | Bottom toast, transient | Auto-reconnects with exponential backoff (already in store.ts L585-596 for jobs; agent socket has none today) |
| `budget_exceeded` (8 tool calls) | Final-message slot | "Continue anyway (8 more)" button, sends a sentinel message that resets `tool_calls_used` |

**Backend implication.** Agent socket needs a `reconnect` path. Today `_attachAgentSocket` (store.ts L933) has no retry on close. Add the same `RECONNECT_DELAYS_MS` pattern jobs use. New event `tool_call_error` (vs. reusing `tool_call_complete` with `ok: false`) so the frontend can render inline-error treatment cleanly.

#### A6. Final-suggestion cards

**Why this hurts today.** The orange "Open →" button (L304) opens a tab and quietly stuffs args in `sessionStorage` (L266). The user has no preview of what they're about to land in.

**Proposed.**

```
[Open analyze report →]
  ╭────────────────────────────────────────────────╮
  │ Will open: cli:analyze                         │
  │   --run abc123                                 │
  │   --output html                                │
  │ Click to open the form pre-filled.             │
  ╰────────────────────────────────────────────────╯
```

The card body shows the args as a mini-list. Hover/focus reveals a tooltip-style mini-preview. The button's accessible label becomes "Open analyze form pre-filled with run abc123".

When the form opens, it should land in a "this came from chat" state with a dismissible chip at the top: `From chat · args from agent (abc12)` - so the user knows where they are and can revert to defaults with one click.

**Backend implication.** The `final` event already carries suggestions (agent.py emits these only via the model's structured output - see store.ts L1119-1156 for the dispatch). Today the suggestions are formless `{label, cliId, args}`. We just need a frontend-only landing-state convention; no backend change required.

#### A7. Composer

**Why this hurts today.** Placeholder claims "paste a CLI · @-mention a run" (L493). Neither works. Hint says `⌘K palette` - no handler.

**Proposed.**

```
┌──────────────────────────────────────────────────────┐
│ Ask anything · /run · /compare · @run · @file        │
│                                                      │
│                                                      │
│                                                      │
│ /run @run-2026-04-30-ghi                  [@] [⏎ Send│
│                                                      │
│ ↵ send · ⇧↵ newline · / commands · @ mentions       │
└──────────────────────────────────────────────────────┘
```

**Slash commands** (client-side, expand to text on send):

- `/run <cli-id>` - inserts "please run `<cli-id>` with these args: ..." prompt scaffold
- `/compare <runA> <runB>` - inserts a comparison-shaped prompt
- `/explain <run-id>` - "Walk through what happened in this run"
- `/clear` - reset thread (same as `＋` button)
- `/settings` - open settings modal

**@-mentions** via a popover triggered by `@`:

- `@run-...` shows recent run ids from `store.runs`
- `@file:...` shows files from `store.fileTree`
- selecting an option inserts a chip into the textarea, which serializes on send as `[run-id: abc123]` - the agent's system prompt teaches it this convention.

**Paste detection.** If the user pastes a string starting with `evalyn ` and ending with a newline, show a contextual button: `[Convert to slash command?]` that turns it into `/run <id> --foo bar`.

**Multiline.** Current textarea is `rows={2}` (L475). Make it auto-grow up to 8 rows, then internal scroll.

**Token count.** Bottom-right counter: `1234 / 100k tokens`. Soft-warn at 80%, hard-block at 100%.

#### A8. Conversation hygiene

**Why this hurts today.** `＋` wipes the thread (L370 / store.ts L883-894). No recovery. No history.

**Proposed.**

- Persistent threads (`localStorage` per origin, cap last 20). Each thread: id, title (auto-derived from first user message, truncated), last-modified timestamp, message count.
- Header gets a "thread switcher" caret next to the title:

```
Ask agent  ▾                              ＋  ⚙  ×
─────────────────────────────────────────────────
[dropdown opens:]
  ● Compare last 3 run-eval runs            14:32 · 12 msg
  ○ Why did spam dataset pass rate drop     11:08 · 28 msg
  ○ How do I add a custom metric            yesterday · 4
  [+ New thread]   [Manage threads]
```

- Per-message "branch from here" (`⤴︎` icon) creates a new thread that copies messages up to and including the current one, then re-runs the last user message in the new thread.
- "Save thread" - export to `.evalyn/threads/<id>.md` with timestamps and tool outputs. Useful for sharing debugging sessions.

**Backend implication.** None for v1 - store on the client. v2 could add `/api/agent/threads` endpoints with disk persistence under `~/.evalyn/threads/`.

#### A9. Empty state

**Why this hurts today.** Two grey comments (L549-558). New users have no idea what's possible. Returning users have no shortcuts.

**Proposed (cold start - no runs in `.evalyn/`).**

```
              ╭─────────────────╮
              │   ●     ●       │
              │       ●         │
              │   ●     ●       │
              ╰─────────────────╯
              Agent is ready

  Try one of these to get started:

  ┌─────────────────────────────────────────────┐
  │ "Walk me through evalyn quickstart"         │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ "Show me how to evaluate my agent"          │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ "What metrics should I use for a chatbot?"  │
  └─────────────────────────────────────────────┘
```

**Proposed (warm start - runs exist).**

Suggestions are computed client-side from `store.runs` and `store.runHistory`:

```
              Pick up where you left off:

  ┌─────────────────────────────────────────────┐
  │ "Why did pass rate drop on run-...-ghi?"    │
  │ Latest run: 8pp regression vs. previous     │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ "Compare my last 3 run-eval runs"           │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ "Cluster failures in run-...-ghi"           │
  │ 47 failed items                             │
  └─────────────────────────────────────────────┘
```

Click sends the suggestion as a user message immediately.

### Backend implications (chat, summary)

| Need | Where |
|---|---|
| Args override on confirm | New optional field on `POST /api/agent/chat/{tid}/confirm` |
| Per-session approval whitelist | `_Thread.session_auto_approve: set[str]` in agent.py |
| `side_effects` per write CLI | Static dict in agent.py keyed by tool name; emitted with `confirmation_required` |
| Agent WS reconnect | Mirror `RECONNECT_DELAYS_MS` from job WS |
| `tool_call_error` event | Distinct from `tool_call_complete`; renders inline-error |
| Thread persistence | Defer to v2; localStorage in v1 |
| Slash-command catalog | None - purely client-side; the `/run` etc. expand into prose before send |

---

## Part B: CliForm

### Today (read of CliForm + CliFormBody + ParamField)

**CliForm.tsx** (158 lines). Header strip with group label + `$ evalyn <id>` + h1 (L66-80), blurb (L82-91), the `CliFormBody` in a bordered box (L93-102), client-side `missing` required validation (L31-36), error panel (L104-118), footer with Run button + Cancel + a 560px-wide truncated preview + copy button (L120-152).

**CliFormBody.tsx** (218 lines). Three modes:

- `raw` (L55-83): a single textarea showing the assembled command, with `onChange` that does nothing (L73-75) - the textarea is read-only despite looking editable. Hint says "Edit this string directly" (L78-80) but the change is dropped. **This is broken.**
- `form` (L214-216): just the field grid.
- `preview` (L151-212): two-column - form on left, sticky right column with `live command` block + a "predicted" panel (L178-208) showing fake cost + fake duration computed as `0.4 + filledCount * 0.18` (L201) and `3 + Math.round(filledCount * 1.4)` (L206). **This is invented data.** It will mislead users about real cost.

**Disclosure.** Default visible = required + `essential` (L51-53). Everything else lives behind a "Show all options (N)" button (L107-115). The `advanced` flag is rendered as a chip but does not partition (L132-145).

**ParamField.tsx** (184 lines).

- `bool` (L51-77): two buttons "true" / "false". No third "default/unset" state, so a user cannot un-pick. Default values are visually identical to user picks.
- `select` (L80-99): native `<select>`. The empty option renders as `(any)` (L92).
- `multiselect` (L101-129): chip toggles, each labeled `on opt` or `off opt` (L122). Wraps. No "select all" / "clear".
- `number` (L132-148): `<input type=number>`. No min/max, no step, no slider. No display of unit (workers? seconds? tokens?).
- `long-text` (L150-164): plain `<textarea rows={4}>`. No resize handle, no token count, no markdown preview.
- `path` (L166-180): plain `<input>` with `./...` placeholder (L167). No file picker. **The user must remember the exact path.**
- `string` (L166-180): same as path, no placeholder.

The `Label` component (L23-45) shows the param name with underscores replaced by spaces, plus `*` for required, plus an "advanced" chip. No type indicator (`number`, `path`, `multiselect`).

### Friction list

1. **`raw` mode's textarea pretends to be editable but discards changes** (CliFormBody L73-75). Either ship reverse-parse or remove the textarea.
2. **`preview` mode's cost/duration are fabricated** (L201, L206). Users will trust them. **High risk - must remove or replace with real estimates.**
3. **Path fields are bare `<input>`s.** No file tree integration despite `store.fileTree` existing.
4. **`bool` has no "unset" state.** Once you click "false", you can never go back to "use the argparse default" without remembering it. Defaults displayed identically to user choices.
5. **Number fields lack unit + range.** `--workers 8`, `--confidence-samples 3`, `--num-traces 5` all look the same. The `help` text contains hints (`max: 16`) but they are not enforced.
6. **`select` empty-option is `(any)`** (L92). For `--format` (`table` / `json`), that label is wrong; argparse defaults are not "any".
7. **No essential vs. advanced visual distinction beyond a chip.** "Show all options (12)" button is a single disclosure - no grouping by purpose (e.g., output config vs. provider config vs. tuning).
8. **No inline doc.** `param.help` is rendered in 10.5px grey at the bottom (`hint` class). Long help strings wrap awkwardly. No "show full description" affordance.
9. **No validation feedback per-field.** All errors land in the top banner (CliForm L104-118): "missing required: dataset, metrics". Field-level red borders / inline messages absent.
10. **Run button label is a static `▶ Run`** (CliForm L127). No keyboard shortcut, no `Run run-eval` clarity.
11. **No re-run / history.** Workspace.tsx has `runHistory` and `editRunArgs` but the form itself doesn't surface them. Users open the CLI fresh every time.
12. **No saved presets.** `run-eval` is run dozens of times with the same dataset; users can only re-build the form by hand or by clicking "Edit" on a run card in Workspace.
13. **No "open last result" shortcut after a run.** Form closes (L47), tab swaps, but the user loses the form context.
14. **Multiselect with no options shown** (e.g., `--unit-types`) renders as zero chips. The field is invisible if `param.options` is empty - which happens for free-form comma-separated args. Today `--unit-types` is `kind: string`, not multiselect, but the field is documented as comma-separated; the user has no UI hint.
15. **Empty state.** A user opens `cli:run-eval` with nothing filled - they see required fields with `*` but no clear "this is what you do next."

### Redesign proposal

#### B1. Field rendering - per-kind

##### bool

**Today.** Two-button toggle, no unset.

**Proposed.** Three-state segmented control:

```
  --use-calibrated    [ default ] [ true ] [ false ]    ⓘ
                       ──────────
                       (default = false, from argparse)
```

When `default` is selected, the flag is omitted from the assembled command (matching `buildCli.ts` L52-56 logic). Show the resolved value subtly: `(default = false)`. Hover ⓘ shows full `help` text.

**Microcopy.** Replace "true / false" with the semantic flag where possible: for `--use-calibrated`, render as `[ off ] [ on ]` with the underlying mapping inferred from `argparse._StoreTrueAction`.

##### number

**Today.** Bare `<input type=number>`.

**Proposed.** Number + unit + slider where range is known:

```
  --workers           [  4  ]  workers   ◯─────────●─────  (1-16)
  --confidence-samples [ 3  ]  samples   ◯●─────────────  (1-32)
```

The unit string is parsed from `help` text via a small regex (`/(\b\w+s?\b)$/` after the colon). The slider range is read from `help` substrings (`max: 16`, `>= 1`); fall back to no slider when no range is parseable.

**Backend implication.** Add an optional `range: {min, max, step}` and `unit: string` to `ParamSchema`. Curated per-CLI in command modules (`RANGES = {"workers": (1, 16, 1), ...}`).

##### path

**Today.** Bare `<input>` with `./...` placeholder.

**Proposed.** Combobox: typeable input + dropdown of matching `.evalyn/` paths from `store.fileTree`, plus a [📁] button that opens a file picker modal scoped to `.evalyn/`.

```
  --dataset           [ evals/spam.jsonl                ▼] [📁]
                        ┌─────────────────────────────┐
                        │ ▸ evals/                    │
                        │   spam.jsonl                │
                        │   refusals.jsonl            │
                        │ ▸ runs/                     │
                        └─────────────────────────────┘
                      ⓘ JSON/JSONL dataset or directory with dataset.jsonl
```

**Why.** `--dataset` and `--metrics` typos are the dominant first-time error. The file tree is already in the store. Combobox is non-blocking (typing still works for paths outside `.evalyn/`).

##### long-text

**Today.** `<textarea rows={4}>`, no help.

**Proposed.**

```
  --prompt            [Edit ▾] [Preview ▾]  Markdown · 312 tokens · 4 lines
  ┌──────────────────────────────────────────────────────────┐
  │ You are a helpful assistant. When asked about            │
  │ refunds, always check the policy first.                  │
  │                                                          │
  │ ▒ ← drag handle (CSS resize)                            │
  └──────────────────────────────────────────────────────────┘
  ⓘ System prompt for the judge LLM
```

Edit/Preview toggle (preview renders Markdown with our existing `Markdownish` style). Token count via `~text.length / 4` heuristic with `[≈]` prefix to mark it as approximate. Resize handle visible (CSS `resize: vertical` is already on `.textarea` per index.css L252).

##### multiselect

**Today.** Inline chip wall.

**Proposed.** Two patterns by cardinality:

- ≤ 6 options: keep inline chips, but add a `[Clear]` link.
- > 6 options: collapse to a chip-input with popover:

```
  --unit-types        [outcome × multi_turn × ] [+ add]    ⓘ
                                                  popover:
                                                  ┌─────────────┐
                                                  │ ☐ outcome   │
                                                  │ ☐ single    │
                                                  │ ☐ tool_use  │
                                                  │ ☐ multi_turn│
                                                  │ ☐ custom    │
                                                  │ [Select all]│
                                                  └─────────────┘
```

##### select

Today's `(any)` empty option is wrong. Replace with the literal default value rendered with `(default)` annotation:

```
  --format            [ table (default) ▾ ]
                        ┌─────────┐
                        │ table   │
                        │ json    │
                        └─────────┘
```

##### Label, defaults visibility, validation

**Today.** Underscored→spaced label, optional `*`, optional advanced chip.

**Proposed.**

```
  dataset *  path                                         ⓘ
  ────────   ────                                         (icon hover
   field      kind chip                                   shows full help)
   name
```

A small `kind` chip after the name (`path`, `bool`, `select(3)`, `int`). On hover/focus the ⓘ icon expands the help to a tooltip OR pins it inline below.

**Required indication.** Replace the orange `*` with a left border on the field row when required:

```
│ dataset  path                  [ evals/...      ▼] [📁]
│ metrics  path                  [ metrics/...    ▼] [📁]
```

Field-level error rendering on submit:

```
│ dataset  path                  [                ▼] [📁]
│  required                                            
```

Replace the top error banner with field-scoped errors + a single "3 fields need attention" anchor link that focus-scrolls to the first.

#### B2. Mode toggle - kill it, but absorb its useful parts

**Why this hurts today.** The mode toggle adds cognitive overhead, the `raw` mode is broken (read-only despite looking editable), and the `preview` mode's "predicted" panel fabricates data.

**Proposed.** Single canonical view, always visible:

```
┌─ Form ─────────────────────────────────────┬─ Live command ──────────┐
│ dataset *  path        [...]               │ $ evalyn run-eval       │
│ metrics    path        [...]               │   --dataset evals/...   │
│ workers    int   8     ◯────●───  (1-16)   │   --metrics metrics/... │
│ ...                                        │   --workers 8           │
│ [▸ 12 more options]                        │                         │
│                                            │ [⎘ copy]  [✎ paste raw] │
└────────────────────────────────────────────┴─────────────────────────┘
[▶ Run run-eval]   [Cancel]                           ⌘↵ to run
```

**Sticky right column** with the live command (carrying the existing `CliHighlighted` syntax-coloring at CliFormBody L20-35). **Two new affordances** below the command:

- `[⎘ copy]` - same as today.
- `[✎ paste raw]` - opens an inline modal to paste a CLI string (e.g., from a Slack message or a teammate). Reverse-parses it via a tiny client-side parser and merges values into the form. Errors render below the modal.

**Replace the "predicted" panel.** The fake cost/duration must go. Either:

- (a) Remove entirely (cheapest).
- (b) Replace with a real estimate sourced from `runHistory`: "Last similar run: 240 items · 3m12s · $0.42". Computed by finding the most recent `RunRecord` for this `cliId` whose args match within tolerance (e.g., same `--dataset`).

I recommend (b). It's grounded in real data and reuses `runHistory` already in the store (store.ts L92).

#### B3. Run button + post-run

**Today.** `▶ Run` (CliForm L127), no shortcut, on submit closes the tab and opens a job tab.

**Proposed.**

- Label: `▶ Run run-eval` (mirror the actual id - matches the tab title).
- Shortcut: `⌘↵` everywhere on the form (focus anywhere). Visible kbd hint to the right.
- Loading state: `▶ Running run-eval... 0:08` with elapsed counter; clicking transforms to `[■ Cancel]`.
- On success **don't close the tab** - swap the form-body region with a "Run completed" header, keep the form values intact, and append:

```
┌── Run completed · 3m 12s · pass=0.84 · $0.42 ────────────┐
│ run-2026-04-30-ghi                                       │
│                                                          │
│ [Open in Run viewer →]   [Re-run with same args]         │
│ [Re-run · tweak…    ]   [Compare to previous]            │
└──────────────────────────────────────────────────────────┘
```

The user stays on the same tab, with the run summary plus actions for the natural next step. No tab churn.

#### B4. Recent runs strip - the missing surface

**Why this hurts today.** A power user's main job is iterating. They run `run-eval` 30 times. Today the Workspace shows runs, but the form does not.

**Proposed.** Above the form, a horizontal strip of the last N runs of THIS cli:

```
Recent run-eval runs:
[●  3m ago  pass=0.84  spam.jsonl   ]
[○  1h ago  pass=0.82  spam.jsonl   ]
[○  2h ago  pass=0.78  refusals.jsonl]
[+ all 12 runs of run-eval →]
```

Click a card → seeds the form with that run's args (using existing `editRunArgs` logic, store.ts L790-797). Right-click → "Pin" / "Compare" / "Open in Run viewer".

A diff badge on the active form: `Form differs from last run by: workers (4 → 8)` - reusing `diffArgs` (diffArgs.ts).

#### B5. Saved presets (v1.5)

**Why this hurts today.** Iteration patterns repeat across sessions, but `runHistory` is in-memory only. A user who closes the dashboard loses everything.

**Proposed.** Star a run's args from the recent-runs strip → persists to `localStorage` keyed by `cliId`. New section in the strip:

```
Presets:
[★ smoke-test       spam.jsonl, workers=4]
[★ full-evaluation  all metrics, workers=16]
```

v2 could persist to `.evalyn/dashboard/presets.json` for cross-session and cross-machine sharing.

#### B6. Empty state

**Today.** The form just renders. No CTA.

**Proposed.** When required fields are unfilled and `runHistory.filter(r => r.cliId === cli.id)` is empty:

```
Run run-eval for the first time
─────────────────────────────────

  This will run your evaluations against the dataset
  you choose, judging each item with the metrics you
  pick. Output streams live to a job tab.

  To get started:
  1. Pick a dataset      ← required
  2. Pick metrics        ← optional (auto-detected)
  3. Click Run
```

When `runHistory` is non-empty for this CLI but the form is fresh, surface "Pre-fill from your last run?":

```
[ Pre-fill with last run's args (3m ago) ]   [ Start blank ]
```

### Worked example: redesigned `run-eval` form

```
┌──────────────────────────────────────────────────────────────────────┐
│ EVAL  /  $ evalyn run-eval                                            │
│ Run evaluation on dataset using specified metrics                     │
│                                                                       │
│ Recent run-eval runs:                                                 │
│ [●  3m ago  pass=0.84  spam.jsonl  4w   ]                            │
│ [○  1h ago  pass=0.82  spam.jsonl  4w   ]                            │
│ [○  2h ago  pass=0.78  refus.json  4w   ]    [+ 12 runs]             │
│                                                                       │
│ Form differs from last run by: workers (4 → 8)        [Reset to last] │
│                                                                       │
│ ┌─ Form ─────────────────────────────┬─ Live command ──────────────┐ │
│ │ dataset *    path                   │ $ evalyn run-eval           │ │
│ │   [evals/spam.jsonl    ▼] [📁]      │   --dataset evals/spam.jso. │ │
│ │   ⓘ JSON/JSONL or directory         │   --workers 8               │ │
│ │                                     │                             │ │
│ │ latest        bool                  │ [⎘ copy] [✎ paste raw]      │ │
│ │   [default] [on] [off]              │                             │ │
│ │                                     │ Last similar run:           │ │
│ │ metrics       path                  │   240 items · 3m 12s · $0.42│ │
│ │   [auto-detect from meta.json    ▼] │                             │ │
│ │                                     │                             │ │
│ │ workers       int  ◯───●───  (1-16) │                             │ │
│ │   [8] workers                       │                             │ │
│ │                                     │                             │ │
│ │ provider      select                │                             │ │
│ │   [gemini (default) ▾]              │                             │ │
│ │                                     │                             │ │
│ │ ▸ 9 more options                    │                             │ │
│ └─────────────────────────────────────┴─────────────────────────────┘ │
│                                                                       │
│ [▶ Run run-eval]  [Cancel]                              ⌘↵ to run     │
└──────────────────────────────────────────────────────────────────────┘
```

After run completes, the form area is replaced by:

```
┌──────────────────────────────────────────────────────────────────────┐
│ ✓ Run completed · 3m 12s · pass=0.84 · $0.42                         │
│   run-2026-04-30-ghi                                                 │
│                                                                       │
│   Pass rate dropped 8pp vs. previous run. [Why? · ask agent]         │
│                                                                       │
│   [Open Run viewer →]  [Re-run same args]  [Re-run · tweak…]         │
│   [Compare to previous]                                              │
└──────────────────────────────────────────────────────────────────────┘
```

### Backend implications (form, summary)

| Need | Where |
|---|---|
| `range: {min, max, step}` per number param | New optional field on `ParamSchema` |
| `unit: string` per number param | New optional field on `ParamSchema` (defaults parseable from `help`) |
| `examples: string[]` per param | New optional field on `ParamSchema`, used by combobox / multiselect for non-`choices` strings (e.g., `--unit-types`) |
| `side_effects: list[str]` per CLI | New optional field on `CliSchema` |
| Cost-history endpoint | New `/api/runs/cost-history?cli_id=run-eval` returning recent (cost, duration, item-count) tuples for the heuristic |
| File picker | Reuses existing `/api/files/tree`; add `/api/files/glob?pattern=*.jsonl` for filtering |
| Reverse-parse raw CLI | Pure frontend (small parser); no backend change |

The single most leverageable addition is `range` + `unit` per `ParamSchema`. Per-module `RANGES` constants in argparse modules are cheap to add and unblock sliders, validation, and clearer rendering across all 35 CLIs.

---

## Cross-surface synergies

### Chat → Form handoff

Today: SuggestionCard click writes to `sessionStorage` (ChatPanel.tsx L266) then opens the tab. The form has no idea it's "from chat."

Proposed shared convention:

```ts
type FormLandingState = {
  source: 'fresh' | 'chat' | 'editRun' | 'preset';
  sourceLabel?: string;       // "from agent · thread abc"
  args: Record<string, unknown>;
};
```

Routed via a transient store field (`activeFormSeed` already exists - just add `source` and `sourceLabel`). The form renders a dismissible chip at the top:

```
ⓘ Pre-filled from agent · thread abc      [Reset to defaults]
```

This makes the handoff visible and undoable.

### Form → Chat handoff

Inverse direction: a user in a CliForm can ask the agent about it without context-switching:

```
Form footer: [▶ Run run-eval]  [Cancel]   [💬 Ask agent about this form]
```

Click sends a synthetic user message to the chat: "Help me configure run-eval. My current args: --dataset ... --workers 8". The chat opens (if hidden) and the agent responds with suggestions.

### Run completion → both surfaces

When a run completes, both surfaces should react:

- Chat: insert a passive notification card "Run-eval finished · pass=0.84 · open" (only if the chat has been used in this session, to avoid noise).
- Form: swap to the post-run summary (B3 above).

This is purely a frontend coordination concern - `runHistory` updates already happen in `runActive` (store.ts L737-746).

### Shared visual atoms

Both surfaces should share a "command pill" component:

```
$ evalyn run-eval --dataset evals/spam.jsonl
```

Currently rendered three different ways: `ToolCallCard` header (ChatPanel.tsx L181-184), `CliHighlighted` (CliFormBody L20-35), `cmd` truncated text (CliForm L137-148). Extract a single `<CliPreview cmd={...} />` with consistent syntax coloring, hover-to-expand if truncated, click-to-copy.

---

## Implementation phasing

### P0 - Ship in v1 (high-impact, low-cost)

1. **Kill `preview` mode's fake cost/duration panel.** Either delete (1-line change at CliFormBody L178-208) or replace with `runHistory`-based heuristic. The fabricated numbers are an active risk.
2. **Fix `raw` mode** - either implement reverse-parse OR remove the textarea and replace with a single read-only command line + copy button. Today it pretends to be editable.
3. **Collapsible tool cards** in chat. Default collapsed with one-line summary + duration. Click to expand. (~80 lines in ChatPanel.tsx around L146-256.)
4. **Smart auto-scroll** in chat. Pin if user scrolled up; "↓ N new" pill. (~30 lines.)
5. **Streaming caret** (`▌`). One CSS animation on `streaming: true`. (<10 lines.)
6. **Per-message timestamp** + copy button. (~20 lines per ChatTurn.)
7. **File picker for path fields.** Combobox with `store.fileTree`. (~60 lines in ParamField.tsx; new component.)
8. **Run button label `Run run-eval` + ⌘↵ shortcut.** (~10 lines.)
9. **Empty-state suggestion chips** in chat (cold + warm variants, sourced from `store.runs`). (~50 lines.)
10. **Field-scoped validation errors** instead of top banner.

### P1 - Ship in v1.1 (week 2-4)

1. **Editable args before confirmation** in chat. Inline mini-form using `ParamField`. New field on `confirm` payload. (~150 lines + backend.)
2. **"Approve · don't ask again this session"** per-tool whitelist. (~50 lines + backend.)
3. **`side_effects` curation** for ~15 write CLIs. Feed into the confirmation card "THIS WILL" bullets.
4. **Recent runs strip** above the CliForm. (~80 lines reusing `runHistory` + `editRunArgs`.)
5. **Post-run summary** in CliForm (don't close tab). (~100 lines, rework of submit handler.)
6. **Slash commands + @-mentions** in composer. (~120 lines.)
7. **Three-state bool toggle** (default/on/off) + visible defaults annotation. (~40 lines.)
8. **Number sliders + units** for params with parseable ranges. (~60 lines + per-CLI `RANGES` constants.)
9. **Agent WS reconnect** with backoff. (~40 lines, port the job WS pattern.)
10. **Long-text token count + edit/preview toggle.** (~50 lines.)

### P2 - Defer to v1.2+

1. Persistent threads with switcher dropdown (localStorage v1, disk v2).
2. Per-message branch-from-here.
3. Saved presets per-CLI with localStorage.
4. Reverse-parse paste-a-CLI modal in form.
5. Save-thread to `.evalyn/threads/<id>.md`.
6. Token budget counter in composer.
7. Final-suggestion card mini-preview / hover state.
8. Form → Chat handoff button ("Ask agent about this form").
9. Cost-history endpoint + form's "Last similar run" panel.
10. Multiselect popover for high-cardinality options.

### Critical path

P0 ships in roughly 1 engineering week. The two highest-value items are **#1 (kill the fake cost panel)** and **#7 (file picker for paths)** - the first removes a credibility risk, the second removes the most common form error. P1 takes another 2-3 weeks and lifts both surfaces from "competent demo" to "tool I want to use every day." P2 is polish and v2 territory.

### Open questions

1. **Confirmation editability** - if the user edits args on a `run-eval` confirm card, does the agent get told the args were edited (so it can adjust its narrative)? Recommended yes - send `{approve: true, args: {...}, edited: true}` and have the agent receive a system note "user edited args before approving."
2. **Persistent threads** - localStorage caps at ~5MB per origin. With tool outputs, a busy session could hit this fast. Recommended: store thread metadata in localStorage and last 50 messages per thread; drop tool outputs to a `output_truncated: true` flag and replay-on-demand from `/api/jobs/{id}` if the job still exists.
3. **Slash-command UI** - inline expand vs. modal? Recommended inline (lighter, faster, matches Linear / Notion patterns).
4. **Cost-history accuracy** - the heuristic "most recent matching args" can drift if args differ. How fuzzy is acceptable? Recommended: match on `--dataset` exact path + `--metrics` exact path; everything else free.
5. **Preset sharing** - localStorage is per-machine. v2 should write to `.evalyn/dashboard/presets.json` so teams can commit presets. Out of scope for v1.
