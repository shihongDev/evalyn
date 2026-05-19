"""
Schema evolution: handle format changes across dataset versions.

Provides dataclasses for schema definitions and migrations, plus pure
functions to detect schemas, compare versions, apply migrations, and
validate records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SchemaField:
    """A single field in a dataset schema."""

    name: str
    field_type: str = "str"
    required: bool = True
    default: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field_type": self.field_type,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaField:
        return cls(
            name=data["name"],
            field_type=data.get("field_type", "str"),
            required=data.get("required", True),
            default=data.get("default"),
        )


@dataclass
class SchemaVersion:
    """A versioned dataset schema containing a list of fields."""

    version: int
    fields: list[SchemaField] = field(default_factory=list)
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fields": [f.as_dict() for f in self.fields],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaVersion:
        return cls(
            version=data["version"],
            fields=[SchemaField.from_dict(f) for f in data.get("fields", [])],
            description=data.get("description", ""),
        )


@dataclass
class SchemaMigration:
    """Describes how to migrate records from one schema version to another."""

    from_version: int
    to_version: int
    add_fields: list[SchemaField] = field(default_factory=list)
    remove_fields: list[str] = field(default_factory=list)
    rename_fields: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "add_fields": [f.as_dict() for f in self.add_fields],
            "remove_fields": list(self.remove_fields),
            "rename_fields": dict(self.rename_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaMigration:
        return cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            add_fields=[
                SchemaField.from_dict(f) for f in data.get("add_fields", [])
            ],
            remove_fields=data.get("remove_fields", []),
            rename_fields=data.get("rename_fields", {}),
        )


@dataclass
class MigrationResult:
    """Summary of a migration operation."""

    migrated_count: int = 0
    errors: list[str] = field(default_factory=list)
    from_version: int = 0
    to_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "migrated_count": self.migrated_count,
            "errors": list(self.errors),
            "from_version": self.from_version,
            "to_version": self.to_version,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Migration v{self.from_version} -> v{self.to_version}: "
            f"{self.migrated_count} records migrated"
        )
        if self.errors:
            lines.append(f"  {len(self.errors)} error(s):")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Type mapping helpers
# ---------------------------------------------------------------------------

_PYTHON_TYPE_TO_SCHEMA: list[tuple] = [
    (bool, "bool"),   # bool before int (bool is subclass of int)
    (int, "int"),
    (float, "float"),
    (str, "str"),
    (list, "list"),
    (dict, "dict"),
]


def _infer_type(value: Any) -> str:
    """Map a Python value to a schema type string."""
    for py_type, schema_type in _PYTHON_TYPE_TO_SCHEMA:
        if isinstance(value, py_type):
            return schema_type
    return "str"


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def detect_schema(records: list[dict[str, Any]]) -> SchemaVersion:
    """Infer a schema from records by inspecting field names and types.

    All fields found are marked required if they appear in every record.
    The inferred version is always 1.
    """
    if not records:
        return SchemaVersion(version=1)

    # Collect all field names and track presence counts
    field_counts: dict[str, int] = {}
    field_types: dict[str, str] = {}
    for record in records:
        for key, value in record.items():
            field_counts[key] = field_counts.get(key, 0) + 1
            if key not in field_types:
                field_types[key] = _infer_type(value)

    total = len(records)
    fields: list[SchemaField] = []
    for name in sorted(field_counts):
        fields.append(
            SchemaField(
                name=name,
                field_type=field_types[name],
                required=(field_counts[name] == total),
            )
        )

    return SchemaVersion(version=1, fields=fields)


def compare_schemas(
    old: SchemaVersion, new: SchemaVersion
) -> SchemaMigration:
    """Compute a migration that transforms old schema into new schema.

    Fields present in new but not old are additions. Fields present in old
    but not new are removals. Rename detection is based on matching field
    types: if exactly one field was removed and one was added with the same
    type, they are treated as a rename.
    """
    old_names = {f.name for f in old.fields}
    new_names = {f.name for f in new.fields}
    new_field_map = {f.name: f for f in new.fields}
    old_field_map = {f.name: f for f in old.fields}

    added_names = new_names - old_names
    removed_names = old_names - new_names

    # Detect renames: pair removed/added fields that share the same type
    rename_fields: dict[str, str] = {}
    remaining_added: set[str] = set(added_names)
    remaining_removed: set[str] = set(removed_names)

    # Group by type for matching
    removed_by_type: dict[str, list[str]] = {}
    for rname in list(remaining_removed):
        ftype = old_field_map[rname].field_type
        removed_by_type.setdefault(ftype, []).append(rname)

    added_by_type: dict[str, list[str]] = {}
    for aname in list(remaining_added):
        ftype = new_field_map[aname].field_type
        added_by_type.setdefault(ftype, []).append(aname)

    for ftype in removed_by_type:
        r_list = removed_by_type[ftype]
        a_list = added_by_type.get(ftype, [])
        # Only rename when there is an unambiguous 1:1 match
        if len(r_list) == 1 and len(a_list) == 1:
            old_name = r_list[0]
            new_name = a_list[0]
            rename_fields[old_name] = new_name
            remaining_removed.discard(old_name)
            remaining_added.discard(new_name)

    add_fields = [new_field_map[n] for n in sorted(remaining_added)]
    remove_fields = sorted(remaining_removed)

    return SchemaMigration(
        from_version=old.version,
        to_version=new.version,
        add_fields=add_fields,
        remove_fields=remove_fields,
        rename_fields=rename_fields,
    )


def apply_migration(
    records: list[dict[str, Any]], migration: SchemaMigration
) -> tuple[list[dict[str, Any]], MigrationResult]:
    """Apply a migration to records, returning migrated records and a result.

    Operations are applied in order: rename, remove, add defaults.
    """
    result = MigrationResult(
        from_version=migration.from_version,
        to_version=migration.to_version,
    )
    migrated: list[dict[str, Any]] = []

    for i, record in enumerate(records):
        try:
            new_record = dict(record)
            # Rename fields
            for old_name, new_name in migration.rename_fields.items():
                if old_name in new_record:
                    new_record[new_name] = new_record.pop(old_name)
            # Remove fields
            for fname in migration.remove_fields:
                new_record.pop(fname, None)
            # Add fields with defaults
            for sf in migration.add_fields:
                if sf.name not in new_record:
                    new_record[sf.name] = sf.default
            migrated.append(new_record)
            result.migrated_count += 1
        except Exception as exc:
            result.errors.append(f"Record {i}: {exc}")

    return migrated, result


def validate_against_schema(
    records: list[dict[str, Any]], schema: SchemaVersion
) -> tuple[bool, list[str]]:
    """Check that records conform to a schema.

    Returns (valid, errors) where valid is True only when errors is empty.
    """
    errors: list[str] = []
    required_names = {f.name for f in schema.fields if f.required}

    for i, record in enumerate(records):
        for req in sorted(required_names):
            if req not in record:
                errors.append(f"Record {i}: missing required field '{req}'")

    return (len(errors) == 0, errors)


def build_migration_chain(
    versions: list[SchemaVersion],
) -> list[SchemaMigration]:
    """Build sequential migrations between consecutive schema versions.

    Versions are sorted by version number. A migration is created for each
    adjacent pair.
    """
    if len(versions) < 2:
        return []

    sorted_versions = sorted(versions, key=lambda v: v.version)
    chain: list[SchemaMigration] = []
    for i in range(len(sorted_versions) - 1):
        chain.append(compare_schemas(sorted_versions[i], sorted_versions[i + 1]))
    return chain
