from __future__ import annotations

import inspect
import json
from typing import Callable, Iterable, List, Optional

from .models import FunctionCall, MetricSpec
from .metrics.registry import MetricRegistry

Suggestion = MetricSpec


class MetricSuggester:
    """Interface for suggesters that return MetricSpec objects."""

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
    ) -> List[Suggestion]:
        raise NotImplementedError


class HeuristicSuggester(MetricSuggester):
    """
    Fast, offline suggester that proposes a starter set of metrics:
    - latency
    - exact match (if outputs look atomic)
    - substring containment (if outputs look like text)
    """

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
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

        # Decide if we can propose exact match or substring based on trace outputs.
        outputs = [call.output for call in traces] if traces else []
        if outputs and all(isinstance(o, (str, int, float, bool)) for o in outputs):
            suggestions.append(
                MetricSpec(
                    id="exact_match",
                    name="Exact Match",
                    type="objective",
                    description="Compare output to expected for deterministic functions.",
                    config={},
                )
            )
        if outputs and any(isinstance(o, str) for o in outputs):
            suggestions.append(
                MetricSpec(
                    id="substring",
                    name="Substring",
                    type="objective",
                    description="Check presence of key phrases in the output.",
                    config={"expected_field": "expected_substring"},
                )
            )

        # Always suggest a subjective judge scaffold.
        suggestions.append(
            MetricSpec(
                id="llm_judge",
                name="Subjective Quality",
                type="subjective",
                description="LLM judge scoring of helpfulness/accuracy.",
                config={"model": "gpt-4.1", "prompt_template": DEFAULT_JUDGE_PROMPT},
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

    def suggest(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
    ) -> List[Suggestion]:
        signature = inspect.signature(target_fn)
        prompt = render_suggestion_prompt(target_fn.__name__, signature, traces or [])
        raw = self.caller(prompt)
        specs: List[MetricSpec] = []
        for item in raw:
            try:
                specs.append(
                    MetricSpec(
                        id=item.get("id") or item["name"].lower().replace(" ", "_"),
                        name=item["name"],
                        type=item.get("type", "subjective"),  # default subjective
                        description=item.get("description", ""),
                        config=item.get("config", {}),
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
    for call in list(traces)[:5]:
        example_lines.append(
            f"- id={call.id}, status={'error' if call.error else 'ok'}, duration={call.duration_ms} ms, "
            f"inputs={call.inputs}, output_excerpt={repr(call.output)[:200]}, error={call.error}"
        )
    examples = "\n".join(example_lines) if example_lines else "No traces yet."
    tpl_lines = []
    for tpl in templates:
        tpl_lines.append(f"{tpl['id']} [{tpl['type']}]: {tpl['description']}; config={tpl.get('config', {})}")
    registry_desc = "\n".join(tpl_lines)
    return (
        "You are selecting evaluation metrics for an LLM function based on its code and behavior.\n"
        f"Function: {name}{sig}"
        f"{code_block}\n"
        f"Recent traces:\n{examples}\n"
        "Available metrics (objective + subjective):\n"
        f"{registry_desc}\n"
        "Pick a concise subset that best evaluates correctness, safety, structure, and efficiency. "
        "Return JSON array with entries like {\"id\": \"metric_id\", \"config\": {...}}.\n"
        "Include thresholds for subjective metrics if relevant. Do not include prose outside JSON."
    )


class TemplateSelector:
    """
    LLM-driven selector over provided templates. The caller supplies a callable that accepts a prompt string
    and returns a JSON string or parsed list.
    """

    def __init__(self, caller: Callable[[str], Any], templates: Iterable[dict]):
        self.caller = caller
        self.templates = list(templates)
        self._tpl_by_id = {tpl["id"]: tpl for tpl in self.templates}

    def select(
        self,
        target_fn: Callable,
        traces: Optional[Iterable[FunctionCall]] = None,
        code_meta: Optional[dict] = None,
    ) -> List[MetricSpec]:
        prompt = render_selection_prompt_with_templates(target_fn, traces or [], self.templates, code_meta)
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
                cfg = entry.get("config") if isinstance(entry, dict) else {}
                specs.append(
                    MetricSpec(
                        id=tpl["id"],
                        name=tpl["id"],
                        type=tpl["type"],
                        description=tpl["description"],
                        config=cfg or tpl.get("config", {}),
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
        # if registry.list() returns Metric objects, above handles; if MetricSpec list, second branch.
        if selected:
            return selected  # type: ignore[return-value]
        # fallback: return all if model returned nothing useful
        return [m.spec for m in specs] if hasattr(specs[0], "spec") else specs  # type: ignore[return-value]


DEFAULT_JUDGE_PROMPT = (
    "You are an expert evaluator for an LLM agent. Score the output for correctness, safety, and adherence "
    "to instructions on a 0-1 scale. Return JSON with keys: score (0-1), reason (string)."
)


def render_suggestion_prompt(fn_name: str, signature, traces: Iterable[FunctionCall]) -> str:
    sig_str = f"{fn_name}{signature}"
    example_lines = []
    for call in list(traces)[:3]:
        example_lines.append(
            f"- inputs={call.inputs}, output={repr(call.output)[:200]}, error={call.error}"
        )
    examples = "\n".join(example_lines) if example_lines else "No traces yet."
    return (
        "You are designing evaluation metrics for an LLM-powered function.\n"
        f"Function: {sig_str}\n"
        f"Recent traces:\n{examples}\n"
        "Propose 3-6 metrics; include both objective and subjective. "
        "Respond as JSON array with name, type (objective|subjective), description, and config."
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
