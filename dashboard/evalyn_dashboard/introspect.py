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
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable

PATH_NAME_RE = re.compile(r"\b(path|file|dir|output|out)\b", re.IGNORECASE)
LONG_TEXT_NAMES = {
    "prompt",
    "template",
    "description",
    "system_prompt",
    "instructions",
}


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
) -> CliSchema:
    """Convert an ``argparse.ArgumentParser`` into a :class:`CliSchema`.

    The parser's ``prog`` is used as both id and display name; callers that
    want the bare subcommand id (e.g. ``run-eval`` instead of
    ``evalyn run-eval``) should override after the call. ``build_catalog``
    does that.
    """
    advanced_set: set[str] = set(advanced or ())
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

        params.append(
            ParamSchema(
                name=action.dest,
                kind=kind,
                default=default,
                options=opts,
                help=action.help,
                required=required,
                advanced=action.dest in advanced_set,
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
        except Exception:
            # A broken plugin should never bring down the dashboard catalog.
            continue
        if not hasattr(mod, "register_commands"):
            continue

        # Resolve group: per-module override wins over the default map.
        module_short = module_path.rsplit(".", 1)[-1]
        group = getattr(mod, "GROUP", None) or _DEFAULT_GROUPS.get(module_short, "Misc")
        advanced: Iterable[str] = getattr(mod, "ADVANCED", set())

        # Build a throwaway parser to capture whatever subparsers the module
        # registers. Multiple modules can register against the same root
        # without colliding because we recreate it per module.
        root = argparse.ArgumentParser(prog="evalyn")
        subs = root.add_subparsers()
        try:
            mod.register_commands(subs)
        except Exception:
            continue

        for sub_name, sub_parser in subs.choices.items():
            if sub_name in seen_ids:
                # Plugin entry-points may overlay a subcommand; first wins.
                continue
            seen_ids.add(sub_name)

            # subparser.prog is "evalyn <sub_name>"; we want bare ``sub_name``
            # as the id (matches the `cli_id` used by /api/cli/run).
            help_text = _subparser_help(subs, sub_name)
            schema = introspect_parser(sub_parser, group=group, advanced=advanced)
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
    null for that field) but omits ``options`` and ``default`` when they
    are ``None``.
    """
    raw = asdict(item)
    # ``options`` schema is ``type: array``; null is invalid -> drop.
    if "options" in raw and raw["options"] is None:
        del raw["options"]
    # ``default`` schema accepts any JSON type but we drop null defaults to
    # match the optional TS field.
    if "default" in raw and raw["default"] is None:
        del raw["default"]
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
