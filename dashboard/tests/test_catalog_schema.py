"""Phase 0 stub: validate ``build_catalog()`` output against the JSON schema.

The schema lives at ``dashboard/schema/catalog.schema.json`` and the
hand-written TypeScript counterpart at
``dashboard/frontend/src/types/catalog.ts``. Both files are sources of truth
and must agree.

This test is a stub for Phase 1 lane A2. ``build_catalog`` does not exist
yet (it lands in A2.4), so the assertion is wrapped in a skip until then.
The schema-loading half is exercised eagerly so a malformed schema is
caught immediately at the Phase 0 sync gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schema" / "catalog.schema.json"
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_is_valid_jsonschema() -> None:
    """The catalog schema itself parses as valid JSON Schema 2020-12."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    # Will raise if the schema document is malformed.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_defines_expected_param_kinds() -> None:
    """Schema enum must match the TypeScript ``ParamKind`` union exactly."""
    schema = _load_schema()
    kinds = schema["$defs"]["ParamKind"]["enum"]
    assert set(kinds) == {
        "bool",
        "string",
        "number",
        "select",
        "multiselect",
        "path",
        "long-text",
    }


def test_build_catalog_validates_against_schema() -> None:
    """A2.4 deliverable: ``build_catalog()`` must produce schema-valid output."""
    pytest.importorskip("jsonschema")
    try:
        from evalyn_dashboard.introspect import (  # type: ignore[attr-defined]
            build_catalog,
            catalog_to_payload,
        )
    except (ImportError, AttributeError):
        pytest.skip("build_catalog lands in Phase 1 lane A2.4")

    import jsonschema

    schema = _load_schema()
    catalog = build_catalog()
    payload = catalog_to_payload(catalog)
    jsonschema.validate(instance=payload, schema=schema)
