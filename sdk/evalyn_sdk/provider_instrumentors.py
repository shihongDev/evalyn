"""Provider instrumentation framework for additional LLM providers.

Pure Python registry and spec definitions - no external deps, no actual API calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProviderSpec:
    """Specification for an LLM provider that can be instrumented."""

    provider_id: str
    name: str
    module_path: str
    client_class: str
    methods_to_wrap: List[str]
    env_key: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "module_path": self.module_path,
            "client_class": self.client_class,
            "methods_to_wrap": list(self.methods_to_wrap),
            "env_key": self.env_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ProviderSpec:
        return cls(
            provider_id=str(data["provider_id"]),
            name=str(data["name"]),
            module_path=str(data["module_path"]),
            client_class=str(data["client_class"]),
            methods_to_wrap=list(data.get("methods_to_wrap", [])),
            env_key=str(data["env_key"]),
        )


@dataclass
class InstrumentationHook:
    """A hook attached to a specific provider method for instrumentation."""

    hook_id: str
    provider_id: str
    method_name: str
    before_fn_name: str = ""
    after_fn_name: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "hook_id": self.hook_id,
            "provider_id": self.provider_id,
            "method_name": self.method_name,
            "before_fn_name": self.before_fn_name,
            "after_fn_name": self.after_fn_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> InstrumentationHook:
        return cls(
            hook_id=str(data["hook_id"]),
            provider_id=str(data["provider_id"]),
            method_name=str(data["method_name"]),
            before_fn_name=str(data.get("before_fn_name", "")),
            after_fn_name=str(data.get("after_fn_name", "")),
        )


class ProviderRegistry:
    """Registry of LLM provider specs for instrumentation."""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderSpec] = {}

    def register(self, spec: ProviderSpec) -> None:
        """Register a provider spec. Overwrites if provider_id already exists."""
        self._providers[spec.provider_id] = spec

    def get(self, provider_id: str) -> Optional[ProviderSpec]:
        """Return the spec for a provider, or None if not registered."""
        return self._providers.get(provider_id)

    def list_providers(self) -> List[ProviderSpec]:
        """Return all registered provider specs in insertion order."""
        return list(self._providers.values())

    def is_available(self, provider_id: str) -> bool:
        """Check whether the provider's env key is set (without revealing value)."""
        spec = self._providers.get(provider_id)
        if spec is None:
            return False
        return bool(os.environ.get(spec.env_key))

    def get_available(self) -> List[ProviderSpec]:
        """Return provider specs whose env keys are set."""
        return [s for s in self._providers.values() if self.is_available(s.provider_id)]


# ---------------------------------------------------------------------------
# Built-in provider definitions
# ---------------------------------------------------------------------------

BUILTIN_PROVIDERS: List[ProviderSpec] = [
    ProviderSpec(
        provider_id="cohere",
        name="Cohere",
        module_path="cohere",
        client_class="Client",
        methods_to_wrap=["chat", "generate"],
        env_key="COHERE_API_KEY",
    ),
    ProviderSpec(
        provider_id="mistral",
        name="Mistral",
        module_path="mistralai",
        client_class="Mistral",
        methods_to_wrap=["chat.complete"],
        env_key="MISTRAL_API_KEY",
    ),
    ProviderSpec(
        provider_id="bedrock",
        name="Amazon Bedrock",
        module_path="botocore",
        client_class="BedrockRuntime",
        methods_to_wrap=["invoke_model"],
        env_key="AWS_ACCESS_KEY_ID",
    ),
    ProviderSpec(
        provider_id="azure_openai",
        name="Azure OpenAI",
        module_path="openai",
        client_class="AzureOpenAI",
        methods_to_wrap=["chat.completions.create"],
        env_key="AZURE_OPENAI_API_KEY",
    ),
    ProviderSpec(
        provider_id="groq",
        name="Groq",
        module_path="groq",
        client_class="Groq",
        methods_to_wrap=["chat.completions.create"],
        env_key="GROQ_API_KEY",
    ),
    ProviderSpec(
        provider_id="together",
        name="Together",
        module_path="together",
        client_class="Together",
        methods_to_wrap=["chat.completions.create"],
        env_key="TOGETHER_API_KEY",
    ),
    ProviderSpec(
        provider_id="replicate",
        name="Replicate",
        module_path="replicate",
        client_class="Client",
        methods_to_wrap=["run"],
        env_key="REPLICATE_API_TOKEN",
    ),
]


def create_default_registry() -> ProviderRegistry:
    """Create a registry pre-populated with all built-in providers."""
    registry = ProviderRegistry()
    for spec in BUILTIN_PROVIDERS:
        registry.register(spec)
    return registry


def format_provider_status(registry: ProviderRegistry) -> str:
    """Format a text table showing each provider's availability status.

    Columns: Provider | Status | Env Var
    """
    providers = registry.list_providers()
    if not providers:
        return "No providers registered."

    # Column widths
    name_w = max(len(s.name) for s in providers)
    name_w = max(name_w, len("Provider"))
    status_w = len("not configured")
    env_w = max(len(s.env_key) for s in providers)
    env_w = max(env_w, len("Env Var"))

    header = (
        f"{'Provider':<{name_w}}  {'Status':<{status_w}}  {'Env Var':<{env_w}}"
    )
    sep = f"{'-' * name_w}  {'-' * status_w}  {'-' * env_w}"
    lines = [header, sep]

    for spec in providers:
        available = registry.is_available(spec.provider_id)
        status = "available" if available else "not configured"
        lines.append(
            f"{spec.name:<{name_w}}  {status:<{status_w}}  {spec.env_key:<{env_w}}"
        )

    return "\n".join(lines)
