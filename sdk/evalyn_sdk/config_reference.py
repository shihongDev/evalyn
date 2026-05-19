"""Auto-generated documentation for evalyn.yaml configuration options.

Pure Python, no external dependencies. Provides a structured reference
for all configuration options with validation, filtering, and formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ConfigOption:
    """Single configuration option."""

    key: str
    type_name: str
    default: Any
    description: str
    env_var: str = ""
    cli_flag: str = ""
    valid_values: list[Any] = field(default_factory=list)
    required: bool = False
    section: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type_name": self.type_name,
            "default": self.default,
            "description": self.description,
            "env_var": self.env_var,
            "cli_flag": self.cli_flag,
            "valid_values": list(self.valid_values),
            "required": self.required,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigOption:
        return cls(
            key=data["key"],
            type_name=data.get("type_name", "str"),
            default=data.get("default"),
            description=data.get("description", ""),
            env_var=data.get("env_var", ""),
            cli_flag=data.get("cli_flag", ""),
            valid_values=data.get("valid_values", []),
            required=data.get("required", False),
            section=data.get("section", ""),
        )


@dataclass
class ConfigReference:
    """Reference documentation for configuration options."""

    options: list[ConfigOption] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    generated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "options": [o.as_dict() for o in self.options],
            "sections": list(self.sections),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigReference:
        return cls(
            options=[ConfigOption.from_dict(o) for o in data.get("options", [])],
            sections=data.get("sections", []),
            generated_at=data.get("generated_at", ""),
        )


# Hardcoded list of common evalyn config options
KNOWN_OPTIONS: list[ConfigOption] = [
    ConfigOption(
        key="provider",
        type_name="str",
        default="openai",
        description="LLM provider for judge evaluations",
        env_var="EVALYN_PROVIDER",
        cli_flag="--provider",
        valid_values=["openai", "anthropic", "google", "azure", "local"],
        required=False,
        section="provider",
    ),
    ConfigOption(
        key="model",
        type_name="str",
        default="gpt-4o",
        description="Model name for LLM judge",
        env_var="EVALYN_MODEL",
        cli_flag="--model",
        valid_values=[],
        required=False,
        section="provider",
    ),
    ConfigOption(
        key="api_key",
        type_name="str",
        default="",
        description="API key for the LLM provider",
        env_var="EVALYN_API_KEY",
        cli_flag="--api-key",
        valid_values=[],
        required=True,
        section="provider",
    ),
    ConfigOption(
        key="dataset_path",
        type_name="str",
        default="./data",
        description="Path to the evaluation dataset directory",
        env_var="EVALYN_DATASET_PATH",
        cli_flag="--dataset",
        valid_values=[],
        required=True,
        section="data",
    ),
    ConfigOption(
        key="metrics_path",
        type_name="str",
        default="./metrics",
        description="Path to custom metric definitions",
        env_var="EVALYN_METRICS_PATH",
        cli_flag="--metrics-path",
        valid_values=[],
        required=False,
        section="data",
    ),
    ConfigOption(
        key="output_dir",
        type_name="str",
        default="./output",
        description="Directory for evaluation results and reports",
        env_var="EVALYN_OUTPUT_DIR",
        cli_flag="--output-dir",
        valid_values=[],
        required=False,
        section="data",
    ),
    ConfigOption(
        key="log_level",
        type_name="str",
        default="info",
        description="Logging verbosity level",
        env_var="EVALYN_LOG_LEVEL",
        cli_flag="--log-level",
        valid_values=["debug", "info", "warning", "error"],
        required=False,
        section="runtime",
    ),
    ConfigOption(
        key="parallel_workers",
        type_name="int",
        default=4,
        description="Number of parallel evaluation workers",
        env_var="EVALYN_PARALLEL_WORKERS",
        cli_flag="--workers",
        valid_values=[],
        required=False,
        section="runtime",
    ),
    ConfigOption(
        key="timeout",
        type_name="int",
        default=300,
        description="Timeout in seconds for each evaluation item",
        env_var="EVALYN_TIMEOUT",
        cli_flag="--timeout",
        valid_values=[],
        required=False,
        section="runtime",
    ),
    ConfigOption(
        key="retry_limit",
        type_name="int",
        default=3,
        description="Maximum retries for failed LLM calls",
        env_var="EVALYN_RETRY_LIMIT",
        cli_flag="--retry-limit",
        valid_values=[],
        required=False,
        section="runtime",
    ),
    ConfigOption(
        key="judge_model",
        type_name="str",
        default="gpt-4o",
        description="Model to use as LLM judge (can differ from main model)",
        env_var="EVALYN_JUDGE_MODEL",
        cli_flag="--judge-model",
        valid_values=[],
        required=False,
        section="judge",
    ),
    ConfigOption(
        key="confidence_threshold",
        type_name="float",
        default=0.7,
        description="Minimum confidence for judge decisions",
        env_var="EVALYN_CONFIDENCE_THRESHOLD",
        cli_flag="--confidence-threshold",
        valid_values=[],
        required=False,
        section="judge",
    ),
    ConfigOption(
        key="cache_enabled",
        type_name="bool",
        default=True,
        description="Enable caching of LLM judge responses",
        env_var="EVALYN_CACHE_ENABLED",
        cli_flag="--cache/--no-cache",
        valid_values=[],
        required=False,
        section="cache",
    ),
    ConfigOption(
        key="results_format",
        type_name="str",
        default="json",
        description="Output format for evaluation results",
        env_var="EVALYN_RESULTS_FORMAT",
        cli_flag="--format",
        valid_values=["json", "csv", "html", "markdown"],
        required=False,
        section="output",
    ),
    ConfigOption(
        key="max_samples",
        type_name="int",
        default=0,
        description="Maximum number of dataset items to evaluate (0 for all)",
        env_var="EVALYN_MAX_SAMPLES",
        cli_flag="--max-samples",
        valid_values=[],
        required=False,
        section="data",
    ),
]

# Type checking helpers
_TYPE_CHECKERS: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def build_reference(options: list[ConfigOption] | None = None) -> ConfigReference:
    """Build reference from options list.

    Args:
        options: List of ConfigOption. Defaults to KNOWN_OPTIONS.

    Returns:
        Populated ConfigReference.
    """
    opts = options if options is not None else list(KNOWN_OPTIONS)
    sections = sorted(set(o.section for o in opts if o.section))

    return ConfigReference(
        options=opts,
        sections=sections,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def filter_reference(
    ref: ConfigReference,
    section: str | None = None,
    search: str | None = None,
) -> ConfigReference:
    """Filter reference by section or text search.

    Args:
        ref: Source reference.
        section: Filter by section (exact match).
        search: Case-insensitive substring search in key and description.

    Returns:
        New ConfigReference with matching options.
    """
    filtered = list(ref.options)

    if section:
        filtered = [o for o in filtered if o.section == section]

    if search:
        term = search.lower()
        filtered = [
            o
            for o in filtered
            if term in o.key.lower() or term in o.description.lower()
        ]

    sections = sorted(set(o.section for o in filtered if o.section))

    return ConfigReference(
        options=filtered,
        sections=sections,
        generated_at=ref.generated_at,
    )


def format_reference_markdown(ref: ConfigReference) -> str:
    """Format reference as markdown with sections.

    Each option shows type, default, env var, CLI flag, and valid values.
    """
    lines: list[str] = []
    lines.append("# Configuration Reference")
    lines.append("")
    lines.append(f"Generated: {ref.generated_at}")
    lines.append("")

    # Group by section
    by_section: dict[str, list[ConfigOption]] = {}
    for opt in ref.options:
        sec = opt.section or "general"
        by_section.setdefault(sec, []).append(opt)

    for sec in sorted(by_section.keys()):
        lines.append(f"## {sec}")
        lines.append("")

        for opt in sorted(by_section[sec], key=lambda o: o.key):
            req = " **(required)**" if opt.required else ""
            lines.append(f"### `{opt.key}`{req}")
            lines.append(f"- **Type**: `{opt.type_name}`")
            lines.append(f"- **Default**: `{opt.default}`")
            lines.append(f"- **Description**: {opt.description}")
            if opt.env_var:
                lines.append(f"- **Env var**: `{opt.env_var}`")
            if opt.cli_flag:
                lines.append(f"- **CLI flag**: `{opt.cli_flag}`")
            if opt.valid_values:
                vals = ", ".join(f"`{v}`" for v in opt.valid_values)
                lines.append(f"- **Valid values**: {vals}")
            lines.append("")

    return "\n".join(lines)


def format_reference_plain(ref: ConfigReference) -> str:
    """Format reference as plain text."""
    lines: list[str] = []
    lines.append("Configuration Reference")
    lines.append(f"Generated: {ref.generated_at}")
    lines.append("")

    # Group by section
    by_section: dict[str, list[ConfigOption]] = {}
    for opt in ref.options:
        sec = opt.section or "general"
        by_section.setdefault(sec, []).append(opt)

    for sec in sorted(by_section.keys()):
        lines.append(f"[{sec}]")
        lines.append("")

        for opt in sorted(by_section[sec], key=lambda o: o.key):
            req_marker = " (required)" if opt.required else ""
            lines.append(f"  {opt.key}{req_marker}")
            lines.append(f"    Type:    {opt.type_name}")
            lines.append(f"    Default: {opt.default}")
            lines.append(f"    {opt.description}")
            if opt.env_var:
                lines.append(f"    Env:     {opt.env_var}")
            if opt.cli_flag:
                lines.append(f"    CLI:     {opt.cli_flag}")
            if opt.valid_values:
                lines.append(f"    Values:  {', '.join(str(v) for v in opt.valid_values)}")
            lines.append("")

    return "\n".join(lines)


def validate_config(config: dict[str, Any], reference: ConfigReference) -> list[str]:
    """Validate a config dict against the reference.

    Returns list of error strings for:
    - Unknown keys (not in reference)
    - Wrong types (value type does not match type_name)
    - Invalid values (value not in valid_values when constrained)
    - Missing required keys

    Args:
        config: User config dict to validate.
        reference: ConfigReference to validate against.

    Returns:
        List of error message strings. Empty list means valid.
    """
    errors: list[str] = []
    known_keys = {o.key for o in reference.options}
    option_map = {o.key: o for o in reference.options}

    # Check for unknown keys
    for key in config:
        if key not in known_keys:
            errors.append(f"Unknown config key: '{key}'")

    # Check required keys
    for opt in reference.options:
        if opt.required and opt.key not in config:
            errors.append(f"Missing required key: '{opt.key}'")

    # Check types and valid values
    for key, value in config.items():
        if key not in option_map:
            continue
        opt = option_map[key]

        # Type check
        expected_type = _TYPE_CHECKERS.get(opt.type_name)
        if expected_type is not None and value is not None:
            # bool is subclass of int in Python, so check bool first
            if opt.type_name == "bool" and not isinstance(value, bool):
                errors.append(
                    f"Wrong type for '{key}': expected {opt.type_name}, got {type(value).__name__}"
                )
            elif opt.type_name == "int" and isinstance(value, bool) or opt.type_name == "float" and isinstance(value, bool):
                errors.append(
                    f"Wrong type for '{key}': expected {opt.type_name}, got bool"
                )
            elif opt.type_name not in ("bool",) and not isinstance(value, expected_type):
                # For int, also accept float that is whole number? No, keep strict.
                # For float, also accept int
                if opt.type_name == "float" and isinstance(value, int):
                    pass  # int is acceptable for float fields
                elif opt.type_name == "int" and isinstance(value, float):
                    errors.append(
                        f"Wrong type for '{key}': expected {opt.type_name}, got {type(value).__name__}"
                    )
                else:
                    errors.append(
                        f"Wrong type for '{key}': expected {opt.type_name}, got {type(value).__name__}"
                    )

        # Valid values check
        if opt.valid_values and value is not None and value not in opt.valid_values:
            errors.append(
                f"Invalid value for '{key}': '{value}' not in {opt.valid_values}"
            )

    return errors
