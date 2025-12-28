from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Iterable, List, Optional

from ..models import FunctionCall, MetricSpec
from ..metrics.registry import MetricRegistry

Suggestion = MetricSpec


class MetricSuggester:
    """Interface for suggesters that return MetricSpec objects."""

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
        desired_count: Optional[int] = None,
    ) -> List[Suggestion]:
        raise NotImplementedError


class HeuristicSuggester(MetricSuggester):
    """
    Fast, offline suggester that proposes a starter set of metrics:
    - latency
    - text overlap (if outputs look like text AND has_reference=True)
    - JSON validity (if outputs look structured)
    - tool call count (if traces show tool usage)

    Args:
        has_reference: If True, reference-based metrics (ROUGE, BLEU, etc.) can be suggested.
                      If False, only trace-compatible metrics are suggested.
    """

    def __init__(self, has_reference: bool = False):
        self.has_reference = has_reference

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
        desired_count: Optional[int] = None,
    ) -> List[Suggestion]:
        suggestions: List[MetricSpec] = []
        name = getattr(target_fn, "__name__", "agent")

        suggestions.append(
            MetricSpec(
                id="latency_ms",
                name="Latency (ms)",
                type="objective",
                description=f"Measure execution time for {name}.",
                config={},
            )
        )

        outputs = [call.output for call in traces] if traces else []

        # Only suggest reference-based metrics if has_reference is True
        if self.has_reference and outputs and any(isinstance(o, str) for o in outputs):
            suggestions.append(
                MetricSpec(
                    id="token_overlap_f1",
                    name="Token Overlap F1",
                    type="objective",
                    description="Measure token overlap with expected text.",
                    config={"expected_field": "expected"},
                )
            )

        if outputs and any(isinstance(o, (dict, list)) for o in outputs):
            suggestions.append(
                MetricSpec(
                    id="json_valid",
                    name="JSON Valid",
                    type="objective",
                    description="Check output parses as JSON or structured data.",
                    config={},
                )
            )
        if traces and any("tool" in ev.kind.lower() for call in traces for ev in call.trace):
            suggestions.append(
                MetricSpec(
                    id="tool_call_count",
                    name="Tool Call Count",
                    type="objective",
                    description="Count tool-related events in traces.",
                    config={},
                )
            )

        # output_nonempty is always useful
        suggestions.append(
            MetricSpec(
                id="output_nonempty",
                name="Output Not Empty",
                type="objective",
                description="Check that output is not empty/None.",
                config={},
            )
        )

        suggestions.append(
            MetricSpec(
                id="helpfulness_accuracy",
                name="Helpfulness and Accuracy",
                type="subjective",
                description="LLM judge scoring of helpfulness/accuracy.",
                config={
                    "rubric": [
                        "Addresses the user's request directly.",
                        "No major factual errors; if unsure, states uncertainty.",
                        "Does not contradict provided expected/context (if any).",
                    ]
                },
            )
        )
        return suggestions


class LLMSuggester(MetricSuggester):
    """
    Thin wrapper that delegates to a callable that hits an LLM.
    The callable receives a prompt string and returns structured MetricSpec-like dicts.
    """

    def __init__(self, caller: Callable[[str], List[dict]]):
        self.caller = caller

    @staticmethod
    def _parse_json_array(text: str) -> List[dict]:
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
        desired_count: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> List[Suggestion]:
        signature = inspect.signature(target_fn)
        prompt = render_suggestion_prompt(target_fn.__name__, signature, traces or [], desired_count=desired_count, scope=scope)
        raw = self.caller(prompt)
        raw_copy = raw
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = self._parse_json_array(raw)
        if not isinstance(raw, list):
            preview = str(raw_copy)
            if isinstance(raw_copy, str) and len(preview) > 1000:
                preview = preview[:1000] + "..."
            print("LLM returned no metrics. Raw response:", preview)
            return []

        specs: List[MetricSpec] = []
        for item in raw:
            try:
                name = item.get("name") or item.get("id")
                if not name:
                    continue
                specs.append(
                    MetricSpec(
                        id=item.get("id") or str(name).lower().replace(" ", "_"),
                        name=name,
                        type=item.get("type", "subjective"),  # default subjective
                        description=item.get("description", ""),
                        config=item.get("config", {}),
                        why=item.get("why", ""),
                    )
                )
            except KeyError:
                continue
        return specs


def render_selection_prompt_with_templates(
    target_fn: Callable,
    traces: Iterable[FunctionCall],
    templates: Iterable[dict],
    code_meta: Optional[dict] = None,
    desired_count: Optional[int] = None,
    has_reference: bool = False,
) -> str:
    """
    Render prompt for LLM to select metrics from templates.

    Args:
        has_reference: If True, reference-based metrics are available. If False,
                      metrics requiring `expected`/`human_label.reference` are excluded.
    """
    name = getattr(target_fn, "__name__", "function")
    sig = None
    try:
        sig = str(inspect.signature(target_fn))
    except Exception:
        sig = "()"
    code_block = ""
    if code_meta and code_meta.get("source"):
        code_block = f"\nFunction source:\n```\n{code_meta['source']}\n```\n"
    example_lines = []
    for call in list(traces)[:5]:
        example_lines.append(
            f"- id={call.id}, status={'error' if call.error else 'ok'}, duration={call.duration_ms} ms, "
            f"inputs={call.inputs}, output_excerpt={repr(call.output)[:200]}, error={call.error}"
        )
    examples = "\n".join(example_lines) if example_lines else "No traces yet."

    # Filter templates based on reference availability
    filtered_templates = []
    for tpl in templates:
        requires_ref = tpl.get("requires_reference", False)
        if requires_ref and not has_reference:
            continue  # Skip reference-based metrics when no reference available
        filtered_templates.append(tpl)

    tpl_lines = []
    for tpl in filtered_templates:
        inputs = tpl.get("inputs") or tpl.get("signals") or []
        ref_note = " [REQUIRES REFERENCE]" if tpl.get("requires_reference") else ""
        tpl_lines.append(
            f"{tpl['id']} [{tpl['type']}]: {tpl['description']}{ref_note}; "
            f"category={tpl.get('category', '')}; "
            f"inputs={inputs}; "
            f"config={tpl.get('config', {})}"
        )
    registry_desc = "\n".join(tpl_lines)
    count_hint = (
        f"Return JSON array with exactly {desired_count} entries (or as close as possible)."
        if desired_count
        else "Return JSON array of the best metrics."
    )

    # Add reference availability note
    ref_note = ""
    if not has_reference:
        ref_note = (
            "\nIMPORTANT: This dataset does NOT have reference/expected values (no golden standard). "
            "Only select metrics that work with output alone (no ROUGE, BLEU, token_overlap, etc.).\n"
        )

    return (
        "You are selecting evaluation metrics for an LLM function based on its code and behavior.\n"
        f"Function: {name}{sig}"
        f"{code_block}\n"
        f"Recent traces:\n{examples}\n"
        f"{ref_note}"
        "Available metrics (objective + subjective):\n"
        f"{registry_desc}\n"
        "Pick a diverse subset that best evaluates correctness, safety, structure, and efficiency. "
        f"{count_hint} Entries like {{\"id\": \"metric_id\", \"config\": {{...}}, \"why\": \"short reason\"}}.\n"
        "For subjective metrics, you may override config (e.g., rubric/policy/desired_tone) if needed. "
        "Do not include prose outside JSON."
    )


class TemplateSelector:
    """
    LLM-driven selector over provided templates. The caller supplies a callable that accepts a prompt string
    and returns a JSON string or parsed list.

    Args:
        caller: Callable that takes a prompt string and returns JSON response.
        templates: Iterable of metric template dicts.
        has_reference: If True, reference-based metrics can be selected.
                      If False, metrics with requires_reference=True are filtered out.
    """

    def __init__(self, caller: Callable[[str], Any], templates: Iterable[dict], has_reference: bool = False):
        self.caller = caller
        self.templates = list(templates)
        self.has_reference = has_reference
        self._tpl_by_id = {tpl["id"]: tpl for tpl in self.templates}

    def select(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
        code_meta: Optional[dict] = None,
        desired_count: Optional[int] = None,
    ) -> List[MetricSpec]:
        prompt = render_selection_prompt_with_templates(
            target_fn, traces or [], self.templates, code_meta,
            desired_count=desired_count, has_reference=self.has_reference
        )
        raw = self.caller(prompt)
        # raw may be JSON string or already parsed
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []
        else:
            parsed = raw
        specs: List[MetricSpec] = []
        for entry in parsed or []:
            tpl_id = entry.get("id") if isinstance(entry, dict) else None
            if tpl_id and tpl_id in self._tpl_by_id:
                tpl = self._tpl_by_id[tpl_id]
                # Double-check: skip reference-based metrics if no reference available
                if tpl.get("requires_reference", False) and not self.has_reference:
                    continue
                cfg = entry.get("config") if isinstance(entry, dict) else {}
                specs.append(
                    MetricSpec(
                        id=tpl["id"],
                        name=tpl["id"],
                        type=tpl["type"],
                        description=tpl["description"],
                        config=cfg or tpl.get("config", {}),
                        why=entry.get("why", "") if isinstance(entry, dict) else "",
                    )
                )
        return specs


class LLMRegistrySelector:
    """
    Ask an LLM to choose the best metrics from a registry based on function code/metadata.
    The callable should accept a prompt string and return either a list of metric ids or objects with an `id` field.
    """

    def __init__(self, caller: Callable[[str], List[dict] | List[str]]):
        self.caller = caller

    def select(
        self,
        target_fn: Callable,
        registry: MetricRegistry,
        traces: Optional[Iterable[FunctionCall]] = None,
        code_meta: Optional[dict] = None,
    ) -> List[MetricSpec]:
        specs = registry.list()
        prompt = render_selection_prompt(target_fn, specs, traces or [], code_meta)
        raw = self.caller(prompt)
        ids: List[str] = []
        if all(isinstance(item, str) for item in raw):
            ids = [str(item) for item in raw]  # type: ignore[list-item]
        else:
            for item in raw:  # type: ignore[assignment]
                if isinstance(item, dict) and "id" in item:
                    ids.append(str(item["id"]))
        selected = [m.spec for m in specs if m.spec.id in ids] if hasattr(specs[0], "spec") else [
            s for s in specs if s.id in ids  # type: ignore[attr-defined]
        ]
        if selected:
            return selected  # type: ignore[return-value]
        return [m.spec for m in specs] if hasattr(specs[0], "spec") else specs  # type: ignore[return-value]


DEFAULT_JUDGE_PROMPT = (
    "You are an expert evaluator for an LLM agent. Score the output for correctness, safety, and adherence "
    "to instructions on a 0-1 scale. Return JSON with keys: score (0-1), reason (string)."
)


def render_suggestion_prompt(
    fn_name: str,
    signature,
    traces: Iterable[FunctionCall],
    desired_count: Optional[int] = None,
    scope: Optional[str] = None,
) -> str:
    sig_str = f"{fn_name}{signature}"
    example_lines = []
    for call in list(traces)[:3]:
        output_str = str(call.output) if call.output else ""
        # Show more context for long outputs, with length indicator
        if len(output_str) > 800:
            output_preview = repr(output_str[:400]) + f"... [{len(output_str)} chars]"
        else:
            output_preview = repr(call.output)[:500]
        example_lines.append(
            f"- inputs={call.inputs}, output={output_preview}, error={call.error}"
        )
    examples = "\n".join(example_lines) if example_lines else "No traces yet."
    count_hint = (
        f"Return JSON array with exactly {desired_count} entries (or as close as possible)."
        if desired_count
        else "Return JSON array."
    )

    # Scope-specific guidance
    scope_guidance = ""
    if scope == "overall":
        scope_guidance = "\nFOCUS: Generate metrics that evaluate the FINAL OUTPUT of the function (helpfulness, correctness, format).\n"
    elif scope == "llm_call":
        scope_guidance = "\nFOCUS: Generate metrics for evaluating INDIVIDUAL LLM CALLS (safety, hallucination, tone, coherence).\n"
    elif scope == "tool_call":
        scope_guidance = "\nFOCUS: Generate metrics for evaluating TOOL CALLS (correct tool selection, argument correctness, success).\n"
    elif scope == "trace":
        scope_guidance = "\nFOCUS: Generate metrics that AGGREGATE over the trace (counts, ratios, total cost, error rates).\n"

    return (
        "You are designing evaluation metrics for an LLM-powered function.\n"
        f"Function: {sig_str}\n"
        f"Recent traces:\n{examples}\n"
        f"{scope_guidance}"
        "Generate SUBJECTIVE metrics that an LLM judge can evaluate using custom rubrics.\n"
        f"{count_hint}\n"
        "Respond as JSON array ONLY (no prose) with fields:\n"
        "- id: unique identifier (snake_case)\n"
        "- type: always 'subjective'\n"
        "- description: what this metric evaluates\n"
        "- config: MUST include:\n"
        "    - rubric: list of 2-4 criteria for PASS/FAIL judgment\n"
        "- why: short reason for including this metric\n\n"
        "Example:\n"
        '{"id": "factual_accuracy", "type": "subjective", "description": "Checks factual correctness", '
        '"config": {"rubric": ["No false claims", "Cites sources when appropriate", "Admits uncertainty when unsure"]}, '
        '"why": "Ensures reliable information"}\n\n'
        "Focus on quality aspects specific to this function's purpose."
    )


def render_selection_prompt(
    target_fn: Callable,
    metrics: Iterable[MetricSpec | object],
    traces: Iterable[FunctionCall],
    code_meta: Optional[dict] = None,
) -> str:
    name = getattr(target_fn, "__name__", "function")
    sig = None
    try:
        sig = str(inspect.signature(target_fn))
    except Exception:
        sig = "()"
    code_block = ""
    if code_meta and code_meta.get("source"):
        code_block = f"\nFunction source:\n```\n{code_meta['source']}\n```\n"
    example_lines = []
    for call in list(traces)[:3]:
        example_lines.append(
            f"- inputs={call.inputs}, output={repr(call.output)[:200]}, error={call.error}"
        )
    examples = "\n".join(example_lines) if example_lines else "No traces yet."
    available_metrics = []
    for metric in metrics:
        if hasattr(metric, "spec"):
            spec = metric.spec  # type: ignore[attr-defined]
        else:
            spec = metric  # MetricSpec
        available_metrics.append(
            f"{spec.id}: {spec.name} [{spec.type}] - {spec.description}"
        )
    metrics_list = "\n".join(available_metrics)
    return (
        "You are selecting the best evaluation metrics for a function based on its code and behavior.\n"
        f"Function: {name}{sig}"
        f"{code_block}\n"
        f"Recent traces:\n{examples}\n"
        "Available metrics (choose a subset, return JSON array of metric ids):\n"
        f"{metrics_list}\n"
        "Respond as a JSON array of metric ids to use (e.g., [\"latency_ms\", \"exact_match\"])."
    )
