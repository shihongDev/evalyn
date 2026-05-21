"""Argparse-to-CliSchema introspection for the evalyn dashboard.

Walks every ``register_commands`` entry under ``evalyn_sdk.cli.commands.*`` and
produces a list of :class:`CliSchema` dataclasses that match the JSON Schema
at ``dashboard/schema/catalog.schema.json`` and the TypeScript types at
``dashboard/frontend/src/types/catalog.ts`` (all three are sources of truth
and must agree).

Form-field discriminator (``ParamKind``) is derived from each
:class:`argparse.Action` using these rules in order:

1. ``store_true`` / ``store_false`` -> ``bool``.
2. ``choices`` set with ``nargs in {"*", "+"}`` -> ``multiselect``.
3. ``choices`` set otherwise -> ``select``.
4. ``type in {int, float}`` -> ``number``.
5. ``dest`` exactly matches one of ``LONG_TEXT_NAMES`` -> ``long-text``.
6. ``dest`` (with ``_`` -> ``-``) matches ``PATH_NAME_RE`` -> ``path``.
7. Otherwise ``string``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)

PATH_NAME_RE = re.compile(r"\b(path|file|dir|output|out)\b", re.IGNORECASE)
LONG_TEXT_NAMES = {
    "prompt",
    "template",
    "description",
    "system_prompt",
    "instructions",
}

# Range / unit extraction from argparse ``help=`` strings. Best-effort; both
# fields are optional in the public schema and any failure to extract just
# leaves the field None (CLIs without recognizable hints render the bare
# numeric input).

# Match patterns like "(>= 1)", "<=16", "min: 0", "max: 100", "range: 1-32",
# "1-16", "1..16". Numbers may be ints or floats. We deliberately anchor on
# explicit prefix tokens so a stray "...max 16 retries..." in prose doesn't
# accidentally match — except for the bare "N-M" pattern which is parenthesised
# below to reduce false positives.
_RANGE_MIN_MAX_RE = re.compile(
    r"\brange[:=]?\s*(-?\d+(?:\.\d+)?)\s*(?:-|to|\.\.|,)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RANGE_PAREN_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*(?:-|to|\.\.)\s*(-?\d+(?:\.\d+)?)\s*\)")
_RANGE_GE_LE_RE = re.compile(
    r"(>=|>|min[:=]?\s*)\s*(-?\d+(?:\.\d+)?)\b.*?(<=|<|max[:=]?\s*)\s*(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE | re.DOTALL,
)

# Unit extraction: "default: 4 workers" -> "workers". Trailing word right
# after the default literal. Restricted to a-z so we don't pick up trailing
# punctuation, model names, or paths.
_UNIT_AFTER_DEFAULT_RE = re.compile(
    r"default[:=]?\s*-?\d+(?:\.\d+)?\s+([a-z][a-z\-]{1,16})\b",
    re.IGNORECASE,
)


def _extract_range_from_help(help_text: str) -> NumberRange | None:
    """Best-effort range extraction from an argparse help string."""
    if not help_text:
        return None
    # Prefer the explicit "range:" form, then parenthesised "(N-M)", then the
    # min/max pair.
    for regex in (_RANGE_MIN_MAX_RE, _RANGE_PAREN_RE):
        m = regex.search(help_text)
        if m:
            try:
                lo = float(m.group(1))
                hi = float(m.group(2))
            except ValueError:
                continue
            if hi > lo:
                return NumberRange(min=lo, max=hi)
    m = _RANGE_GE_LE_RE.search(help_text)
    if m:
        try:
            lo = float(m.group(2))
            hi = float(m.group(4))
        except ValueError:
            return None
        if hi > lo:
            return NumberRange(min=lo, max=hi)
    return None


def _extract_unit_from_help(help_text: str) -> str | None:
    """Best-effort unit extraction from an argparse help string."""
    if not help_text:
        return None
    m = _UNIT_AFTER_DEFAULT_RE.search(help_text)
    if not m:
        return None
    unit = m.group(1).strip()
    # Drop the obvious junk: "the", "a", "an" tend to slip in if a help string
    # reads "default: 4 the something". Empty / single-char picks are dropped.
    if not unit or len(unit) < 2:
        return None
    if unit.lower() in {"the", "and", "for", "of", "to", "with"}:
        return None
    return unit


def _normalise_range(rng: NumberRange | tuple) -> NumberRange | None:
    """Coerce a tuple-or-NumberRange to a validated NumberRange.

    Tolerates 2-tuples ``(min, max)`` and 3-tuples ``(min, max, step)``.
    Returns None if the result is invalid (min >= max, negative step, ...).
    """
    if isinstance(rng, NumberRange):
        candidate = rng
    elif isinstance(rng, (tuple, list)) and len(rng) in (2, 3):
        try:
            lo = float(rng[0])
            hi = float(rng[1])
        except (TypeError, ValueError):
            return None
        step: float | None = None
        if len(rng) == 3 and rng[2] is not None:
            try:
                step = float(rng[2])
            except (TypeError, ValueError):
                step = None
        candidate = NumberRange(min=lo, max=hi, step=step)
    else:
        return None
    # Validation: reject min > max, NaN, and non-positive step.
    if not (candidate.max > candidate.min):
        return None
    if candidate.step is not None and candidate.step <= 0:
        return None
    return candidate


@dataclass
class NumberRange:
    """Optional numeric range hint for a number-kind parameter.

    Mirrors the ``range`` field on ``ParamSchema`` in
    ``dashboard/frontend/src/types/catalog.ts``. ``step`` is optional; the
    frontend infers a sensible step from ``(max - min) / 100`` when absent.
    """

    min: float
    max: float
    step: float | None = None


@dataclass
class ParamSchema:
    """One CLI argument, ready to render as a form field.

    Mirrors the ``ParamSchema`` interface in
    ``dashboard/frontend/src/types/catalog.ts``.
    """

    name: str
    kind: str
    default: Any = None
    options: list[str] | None = None
    help: str | None = None
    required: bool = False
    advanced: bool = False
    essential: bool = False
    range: NumberRange | None = None
    unit: str | None = None


@dataclass
class CliSchema:
    """One CLI subcommand, ready to render as a form.

    Mirrors the ``CliSchema`` interface in
    ``dashboard/frontend/src/types/catalog.ts``.
    """

    id: str
    name: str
    group: str
    blurb: str
    params: list[ParamSchema] = field(default_factory=list)


def _classify_kind(action: argparse.Action) -> str:
    """Map an :class:`argparse.Action` to a ``ParamKind`` enum value."""
    # 1. boolean flags first - argparse leaves their ``choices`` and ``type``
    # at None so this rule must run before the choices/type rules.
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"

    # 2/3. choices distinguish between select and multiselect via nargs.
    if action.choices is not None:
        if action.nargs in ("*", "+"):
            return "multiselect"
        return "select"

    # 4. numeric types. ``type`` may also be a callable (e.g. _positive_int);
    # we restrict to int/float to avoid misclassifying validators.
    if action.type in (int, float):
        return "number"

    # 5/6. Name-based heuristics. ``dest`` is snake_case; the long-text set
    # uses snake_case directly while the path regex normalises back to dashes.
    name_norm = action.dest
    if name_norm in LONG_TEXT_NAMES:
        return "long-text"
    if PATH_NAME_RE.search(name_norm.replace("_", "-")):
        return "path"

    # 7. fall-through.
    return "string"


def introspect_parser(
    parser: argparse.ArgumentParser,
    *,
    group: str = "Misc",
    advanced: Iterable[str] | None = None,
    essential: Iterable[str] | None = None,
    ranges: dict[str, Any] | None = None,
    units: dict[str, str] | None = None,
) -> CliSchema:
    """Convert an ``argparse.ArgumentParser`` into a :class:`CliSchema`.

    The parser's ``prog`` is used as both id and display name; callers that
    want the bare subcommand id (e.g. ``run-eval`` instead of
    ``evalyn run-eval``) should override after the call. ``build_catalog``
    does that.

    ``ranges`` / ``units`` are optional per-module overrides keyed by argparse
    ``dest``. When omitted, the introspector falls back to a help-string regex
    sweep. Both fields are optional in the output schema; CLIs without
    recognizable hints just render the bare numeric input.
    """
    advanced_set: set[str] = set(advanced or ())
    essential_set: set[str] = set(essential or ())
    ranges = ranges or {}
    units = units or {}
    params: list[ParamSchema] = []
    for action in parser._actions:
        # Skip auto-generated --help / --version actions that argparse adds
        # implicitly; they have no place in the form.
        if action.dest in ("help", "version"):
            continue
        if isinstance(action, argparse._SubParsersAction):
            # Nested subcommands are walked separately by build_catalog.
            continue

        kind = _classify_kind(action)
        opts = list(action.choices) if action.choices is not None else None
        default = action.default if action.default is not argparse.SUPPRESS else None
        # Positional args have empty option_strings and are always required.
        is_positional = not action.option_strings
        required = bool(getattr(action, "required", False)) or is_positional

        # Range + unit are number-only annotations. Per-module dicts win over
        # help-string regex; both are best-effort and may stay None.
        param_range: NumberRange | None = None
        param_unit: str | None = None
        if kind == "number":
            if action.dest in ranges:
                param_range = _normalise_range(ranges[action.dest])
            if param_range is None:
                param_range = _extract_range_from_help(action.help or "")
            if action.dest in units:
                raw_unit = units[action.dest]
                if isinstance(raw_unit, str) and raw_unit.strip():
                    param_unit = raw_unit.strip()
            if param_unit is None:
                param_unit = _extract_unit_from_help(action.help or "")

        params.append(
            ParamSchema(
                name=action.dest,
                kind=kind,
                default=default,
                options=opts,
                help=action.help,
                required=required,
                advanced=action.dest in advanced_set,
                essential=action.dest in essential_set,
                range=param_range,
                unit=param_unit,
            )
        )

    return CliSchema(
        id=parser.prog,
        name=parser.prog,
        group=group,
        blurb=parser.description or "",
        params=params,
    )


# Module path under ``evalyn_sdk.cli.commands`` -> CLI category. The CLI
# command map (`_COMMAND_MODULE_MAP`) maps subcommand names to module names;
# we want the inverse: per-module category. Modules without an entry default
# to "Misc" (and the test suite asserts no module relies on that fallback).
_DEFAULT_GROUPS: dict[str, str] = {
    "traces": "Tracing",
    "runs": "Eval",
    "dataset": "Dataset",
    "simulate": "Simulation",
    "export": "Export",
    "analysis": "Analysis",
    "annotation": "Annotation",
    "calibration": "Annotation",
    "clustering": "Analysis",
    "evaluation": "Eval",
    "insights": "Insights",
    "infrastructure": "Infrastructure",
    "report": "Insights",
    "dashboard_alias": "Insights",
    "quickstart": "Quickstart",
}


def _collect_command_modules() -> list[str]:
    """Return module paths for every command module under
    ``evalyn_sdk.cli.commands.*`` based on the central command map.

    We deliberately use ``_COMMAND_MODULE_MAP`` rather than a directory walk
    so the dashboard catalog stays in lockstep with the CLI's own command
    routing: any subcommand the CLI cannot dispatch must not appear in the
    dashboard catalog either.
    """
    from evalyn_sdk.cli.main import _COMMAND_MODULE_MAP

    package = "evalyn_sdk.cli.commands"
    seen: set[str] = set()
    modules: list[str] = []
    for module_name in _COMMAND_MODULE_MAP.values():
        full = f"{package}.{module_name}"
        if full in seen:
            continue
        seen.add(full)
        modules.append(full)
    return modules


def build_catalog() -> list[CliSchema]:
    """Walk every evalyn command module, register its subparsers against a
    fresh ``ArgumentParser``, and introspect each one.

    The dashboard's REST + WS layer caches this catalog at startup. Callers
    that want a different group/advanced set per module should set ``GROUP``
    and ``ADVANCED`` constants on the command module itself; this function
    reads them via ``getattr`` so existing modules keep working without
    changes.
    """
    result: list[CliSchema] = []
    seen_ids: set[str] = set()

    for module_path in _collect_command_modules():
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            # A broken plugin should never bring down the dashboard
            # catalog, but the operator deserves a breadcrumb -
            # otherwise a third-party plugin silently vanishing from
            # the command list is a debugging dead end. WARN level
            # so the EVALYN_LOG_LEVEL=INFO default surfaces it.
            logger.warning(
                "CLI plugin import failed: %s: %s: %s",
                module_path,
                type(exc).__name__,
                exc,
            )
            continue
        if not hasattr(mod, "register_commands"):
            continue

        # Resolve group: per-module override wins over the default map.
        module_short = module_path.rsplit(".", 1)[-1]
        group = getattr(mod, "GROUP", None) or _DEFAULT_GROUPS.get(module_short, "Misc")
        advanced: Iterable[str] = getattr(mod, "ADVANCED", set())
        essential: Iterable[str] = getattr(mod, "ESSENTIAL", set())
        ranges: dict[str, Any] = getattr(mod, "RANGES", {}) or {}
        units: dict[str, str] = getattr(mod, "UNITS", {}) or {}

        # Build a throwaway parser to capture whatever subparsers the module
        # registers. Multiple modules can register against the same root
        # without colliding because we recreate it per module.
        root = argparse.ArgumentParser(prog="evalyn")
        subs = root.add_subparsers()
        try:
            mod.register_commands(subs)
        except Exception as exc:
            # Same rationale as the import-failure case above: log
            # a breadcrumb so a broken plugin's commands don't
            # vanish silently from the catalog.
            logger.warning(
                "CLI plugin register_commands failed: %s: %s: %s",
                module_path,
                type(exc).__name__,
                exc,
            )
            continue

        for sub_name, sub_parser in subs.choices.items():
            if sub_name in seen_ids:
                # Plugin entry-points may overlay a subcommand; first wins.
                continue
            seen_ids.add(sub_name)

            # subparser.prog is "evalyn <sub_name>"; we want bare ``sub_name``
            # as the id (matches the `cli_id` used by /api/cli/run).
            help_text = _subparser_help(subs, sub_name)
            schema = introspect_parser(
                sub_parser,
                group=group,
                advanced=advanced,
                essential=essential,
                ranges=ranges,
                units=units,
            )
            schema.id = sub_name
            schema.name = sub_name
            if not schema.blurb and help_text:
                schema.blurb = help_text
            result.append(schema)

    return result


def schema_to_dict(item: CliSchema | ParamSchema) -> dict[str, Any]:
    """Serialise a :class:`CliSchema` or :class:`ParamSchema` to a JSON-ready
    dict.

    Drops optional fields whose value is ``None`` so the output validates
    against ``catalog.schema.json``. Keeps ``help: null`` (the schema accepts
    null for that field) but omits ``options``, ``default``, ``range``, and
    ``unit`` when they are ``None``. ``range`` (a nested dataclass) is
    serialised by ``asdict`` automatically, then drops its own ``step`` if
    None so the JSON Schema's ``additionalProperties: false`` can stay tight.
    """
    raw = asdict(item)
    # ``options`` schema is ``type: array``; null is invalid -> drop.
    if "options" in raw and raw["options"] is None:
        del raw["options"]
    # ``default`` schema accepts any JSON type but we drop null defaults to
    # match the optional TS field.
    if "default" in raw and raw["default"] is None:
        del raw["default"]
    # range / unit are optional ParamSchema annotations; drop when absent.
    if "range" in raw:
        if raw["range"] is None:
            del raw["range"]
        elif isinstance(raw["range"], dict) and raw["range"].get("step") is None:
            raw["range"] = {k: v for k, v in raw["range"].items() if k != "step"}
    if "unit" in raw and raw["unit"] is None:
        del raw["unit"]
    if "params" in raw:
        raw["params"] = [
            schema_to_dict(p) if is_dataclass(p) else _drop_nones(p)
            for p in item.params  # type: ignore[union-attr]
        ]
    return raw


def _drop_nones(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    if out.get("options") is None:
        out.pop("options", None)
    if out.get("default") is None:
        out.pop("default", None)
    if out.get("range") is None:
        out.pop("range", None)
    elif isinstance(out.get("range"), dict) and out["range"].get("step") is None:
        out["range"] = {k: v for k, v in out["range"].items() if k != "step"}
    if out.get("unit") is None:
        out.pop("unit", None)
    return out


def catalog_to_payload(catalog: list[CliSchema]) -> list[dict[str, Any]]:
    """Serialise the entire catalog list. Convenience wrapper used by both
    the API layer and the schema-validation test.
    """
    return [schema_to_dict(item) for item in catalog]


def _subparser_help(subs: argparse._SubParsersAction, name: str) -> str:
    """Return the ``help=`` string passed to ``subparsers.add_parser(name, ...)``.

    Argparse stores it on a parallel ``_choices_actions`` list keyed by
    ``dest``; iterate to find the matching entry.
    """
    for choice_action in subs._choices_actions:
        if choice_action.dest == name:
            return choice_action.help or ""
    return ""
