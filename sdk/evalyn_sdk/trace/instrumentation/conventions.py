"""
OpenInference semantic conventions for span attributes.

Based on: https://github.com/Arize-ai/openinference

These conventions enable interoperability with Phoenix, Langfuse, and other
observability tools that support OpenInference.
"""

from __future__ import annotations

# Span kinds (mapped to Evalyn span_type)
class SpanKind:
    LLM = "LLM"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    AGENT = "AGENT"
    EMBEDDING = "EMBEDDING"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    GUARDRAIL = "GUARDRAIL"


# LLM attributes
class LLMAttributes:
    # Model info
    MODEL_NAME = "llm.model_name"
    PROVIDER = "llm.provider"

    # Token usage
    TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    TOKEN_COUNT_TOTAL = "llm.token_count.total"

    # Input/output
    INPUT_MESSAGES = "llm.input_messages"
    OUTPUT_MESSAGES = "llm.output_messages"
    INVOCATION_PARAMETERS = "llm.invocation_parameters"

    # Prompts (alternative to messages)
    PROMPTS = "llm.prompts"


# Tool attributes
class ToolAttributes:
    NAME = "tool.name"
    DESCRIPTION = "tool.description"
    PARAMETERS = "tool.parameters"


# Input/output attributes
class IOAttributes:
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"


# Message attributes (for LLM messages)
class MessageAttributes:
    ROLE = "message.role"
    CONTENT = "message.content"
    FUNCTION_CALL_NAME = "message.function_call_name"
    FUNCTION_CALL_ARGUMENTS = "message.function_call_arguments"
    TOOL_CALLS = "message.tool_calls"


# Retrieval attributes
class RetrievalAttributes:
    DOCUMENTS = "retrieval.documents"


# Document attributes
class DocumentAttributes:
    ID = "document.id"
    CONTENT = "document.content"
    SCORE = "document.score"
    METADATA = "document.metadata"


# Embedding attributes
class EmbeddingAttributes:
    MODEL_NAME = "embedding.model_name"
    EMBEDDINGS = "embedding.embeddings"
    TEXT = "embedding.text"


# Session/trace attributes
class SessionAttributes:
    ID = "session.id"
    USER_ID = "user.id"


# Metadata attributes
class MetadataAttributes:
    ATTRIBUTES = "metadata"


# Map OpenInference span kinds to Evalyn span types
SPAN_KIND_TO_TYPE = {
    SpanKind.LLM: "llm_call",
    SpanKind.CHAIN: "custom",  # No direct equivalent
    SpanKind.TOOL: "tool_call",
    SpanKind.AGENT: "agent",
    SpanKind.EMBEDDING: "custom",  # Future: add "embedding" type
    SpanKind.RETRIEVER: "retrieval",
    SpanKind.RERANKER: "custom",  # Future: add "reranker" type
    SpanKind.GUARDRAIL: "custom",  # Future: add "guardrail" type
}


def get_span_type(openinference_kind: str) -> str:
    """Convert OpenInference span kind to Evalyn span type."""
    return SPAN_KIND_TO_TYPE.get(openinference_kind, "custom")
