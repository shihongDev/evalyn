"""Framework detection and instrumentation registry.

Pure Python registry for AI/ML framework specs - no external deps, no actual framework calls.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class FrameworkSpec:
    """Specification for an AI/ML framework that can be instrumented."""

    framework_id: str
    name: str
    module_path: str
    detect_fn_hint: str
    entry_points: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "framework_id": self.framework_id,
            "name": self.name,
            "module_path": self.module_path,
            "detect_fn_hint": self.detect_fn_hint,
            "entry_points": list(self.entry_points),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FrameworkSpec:
        return cls(
            framework_id=str(data["framework_id"]),
            name=str(data["name"]),
            module_path=str(data["module_path"]),
            detect_fn_hint=str(data.get("detect_fn_hint", "")),
            entry_points=list(data.get("entry_points", [])),
        )


@dataclass
class FrameworkDetectionResult:
    """Result of attempting to detect whether a framework is installed."""

    framework_id: str
    detected: bool
    version: str = ""
    confidence: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "framework_id": self.framework_id,
            "detected": self.detected,
            "version": self.version,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FrameworkDetectionResult:
        return cls(
            framework_id=str(data["framework_id"]),
            detected=bool(data["detected"]),
            version=str(data.get("version", "")),
            confidence=float(data.get("confidence", 1.0)),
        )


class FrameworkRegistry:
    """Registry of AI/ML framework specs for instrumentation."""

    def __init__(self) -> None:
        self._frameworks: dict[str, FrameworkSpec] = {}

    def register(self, spec: FrameworkSpec) -> None:
        """Register a framework spec. Overwrites if framework_id already exists."""
        self._frameworks[spec.framework_id] = spec

    def get(self, framework_id: str) -> FrameworkSpec | None:
        """Return the spec for a framework, or None if not registered."""
        return self._frameworks.get(framework_id)

    def list_frameworks(self) -> list[FrameworkSpec]:
        """Return all registered framework specs in insertion order."""
        return list(self._frameworks.values())

    def detect_installed(self) -> list[FrameworkDetectionResult]:
        """Try importing each framework module and return detection results.

        Uses importlib.util.find_spec to check without actually importing.
        If found, attempts to read __version__ from the module.
        """
        results: list[FrameworkDetectionResult] = []
        for spec in self._frameworks.values():
            found = importlib.util.find_spec(spec.module_path) is not None
            version = ""
            if found:
                try:
                    mod = importlib.import_module(spec.module_path)
                    version = getattr(mod, "__version__", "")
                except Exception:
                    # Module found but failed to import - still count as detected
                    pass
            results.append(
                FrameworkDetectionResult(
                    framework_id=spec.framework_id,
                    detected=found,
                    version=version,
                    confidence=1.0 if found else 0.0,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Built-in framework definitions
# ---------------------------------------------------------------------------

BUILTIN_FRAMEWORKS: list[FrameworkSpec] = [
    FrameworkSpec(
        framework_id="crewai",
        name="CrewAI",
        module_path="crewai",
        detect_fn_hint="import crewai; crewai.Crew",
        entry_points=["Crew", "Agent", "Task"],
    ),
    FrameworkSpec(
        framework_id="autogen",
        name="AutoGen",
        module_path="autogen",
        detect_fn_hint="import autogen; autogen.ConversableAgent",
        entry_points=["ConversableAgent", "GroupChat", "AssistantAgent"],
    ),
    FrameworkSpec(
        framework_id="dspy",
        name="DSPy",
        module_path="dspy",
        detect_fn_hint="import dspy; dspy.Module",
        entry_points=["Module", "Predict", "ChainOfThought"],
    ),
    FrameworkSpec(
        framework_id="haystack",
        name="Haystack",
        module_path="haystack",
        detect_fn_hint="import haystack; haystack.Pipeline",
        entry_points=["Pipeline", "Document", "component"],
    ),
    FrameworkSpec(
        framework_id="llamaindex",
        name="LlamaIndex",
        module_path="llama_index",
        detect_fn_hint="import llama_index; llama_index.VectorStoreIndex",
        entry_points=["VectorStoreIndex", "QueryEngine", "ServiceContext"],
    ),
    FrameworkSpec(
        framework_id="semantic_kernel",
        name="Semantic Kernel",
        module_path="semantic_kernel",
        detect_fn_hint="import semantic_kernel; semantic_kernel.Kernel",
        entry_points=["Kernel", "KernelFunction", "ChatCompletionClientBase"],
    ),
]


def create_default_registry() -> FrameworkRegistry:
    """Create a registry pre-populated with all built-in frameworks."""
    registry = FrameworkRegistry()
    for spec in BUILTIN_FRAMEWORKS:
        registry.register(spec)
    return registry


def format_framework_status(results: list[FrameworkDetectionResult]) -> str:
    """Format a text table showing detected frameworks.

    Columns: Framework | Detected | Version | Confidence
    """
    if not results:
        return "No frameworks checked."

    fw_w = max(len(r.framework_id) for r in results)
    fw_w = max(fw_w, len("Framework"))
    det_w = len("Detected")
    ver_w = max((len(r.version) for r in results), default=0)
    ver_w = max(ver_w, len("Version"))
    conf_w = len("Confidence")

    header = (
        f"{'Framework':<{fw_w}}  {'Detected':<{det_w}}  "
        f"{'Version':<{ver_w}}  {'Confidence':<{conf_w}}"
    )
    sep = f"{'-' * fw_w}  {'-' * det_w}  {'-' * ver_w}  {'-' * conf_w}"
    lines = [header, sep]

    for r in results:
        det_str = "yes" if r.detected else "no"
        conf_str = f"{r.confidence:.1f}"
        lines.append(
            f"{r.framework_id:<{fw_w}}  {det_str:<{det_w}}  "
            f"{r.version:<{ver_w}}  {conf_str:<{conf_w}}"
        )

    return "\n".join(lines)
