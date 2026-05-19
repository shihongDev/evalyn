"""
LlamaIndex framework instrumentor (v0.10+).

Uses LlamaIndex's instrumentation system (EventHandler + SpanHandler) to capture:
- Query engine execution
- LLM calls (chat, complete, predict) with messages and responses
- Retrieval operations with retrieved nodes
- Agent steps and tool calls
- Embedding operations
- Response synthesis

Span hierarchy:
    query:{query}
      -> retrieval
        -> embedding
      -> synthesis
        -> llm_call:{model}
    agent:step
      -> llm_call:{model}
      -> tool_call:{tool_name}
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

from ....models import Span
from ... import context as span_context
from ..base import Instrumentor, InstrumentorType


class LlamaIndexInstrumentor(Instrumentor):
    """Instrumentor for LlamaIndex via its instrumentation system."""

    _instrumented = False

    @property
    def name(self) -> str:
        return "llamaindex"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        return importlib.util.find_spec("llama_index") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            import llama_index.core.instrumentation as instrument

            dispatcher = instrument.get_dispatcher()
            self._event_handler = EvalynLlamaIndexEventHandler()
            self._span_handler = EvalynLlamaIndexSpanHandler()
            dispatcher.add_event_handler(self._event_handler)
            dispatcher.add_span_handler(self._span_handler)
            self._instrumented = True
            return True
        except (ImportError, AttributeError):
            return False

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        # LlamaIndex does not provide a clean remove handler API,
        # but we mark as uninstrumented
        self._instrumented = False
        return True


class EvalynLlamaIndexSpan:
    """Span data tracked during LlamaIndex operations."""

    def __init__(
        self,
        id_: str,
        parent_id: str | None,
        span: Span,
        prev_stack: list,
        start_time: float,
    ) -> None:
        self.id_ = id_
        self.parent_id = parent_id
        self.span = span
        self.prev_stack = prev_stack
        self.start_time = start_time


class EvalynLlamaIndexSpanHandler:
    """
    SpanHandler that creates evalyn spans for LlamaIndex operations.

    Implements the BaseSpanHandler protocol for LlamaIndex instrumentation.
    """

    def __init__(self) -> None:
        self.open_spans: dict[str, EvalynLlamaIndexSpan] = {}
        self.completed_spans: list[EvalynLlamaIndexSpan] = []
        self.dropped_spans: list[EvalynLlamaIndexSpan] = []

    @classmethod
    def class_name(cls) -> str:
        return "EvalynLlamaIndexSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvalynLlamaIndexSpan | None:
        """Create a new span for a LlamaIndex operation."""
        name, span_type = _infer_span_info(instance, bound_args, tags)

        span = Span.new(
            name=name,
            span_type=span_type,
            parent_id=span_context.get_current_span_id(),
        )

        if tags:
            span.attributes["llamaindex.tags"] = {
                k: str(v)[:200] for k, v in tags.items()
            }

        if instance:
            span.attributes["llamaindex.class"] = type(instance).__name__
            module = type(instance).__module__
            if module:
                span.attributes["llamaindex.module"] = module

        prev_stack = span_context._span_stack.get()
        span_context._span_stack.set(prev_stack + [span.id])

        tracker = EvalynLlamaIndexSpan(
            id_=id_,
            parent_id=parent_span_id,
            span=span,
            prev_stack=prev_stack,
            start_time=time.time(),
        )

        self.open_spans[id_] = tracker
        return tracker

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Finalize a span on successful completion."""
        tracker = self.open_spans.pop(id_, None)
        if tracker is None:
            return None

        if result is not None:
            tracker.span.attributes["llamaindex.result_type"] = type(result).__name__
            _enrich_span_with_result(tracker.span, result)

        span_context._span_stack.set(tracker.prev_stack)
        tracker.span.finish(status="ok")
        span_context._add_span_to_collector(tracker.span)

        self.completed_spans.append(tracker)
        return None

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> Any:
        """Finalize a span on error."""
        tracker = self.open_spans.pop(id_, None)
        if tracker is None:
            return None

        if err:
            tracker.span.attributes["error"] = str(err)

        span_context._span_stack.set(tracker.prev_stack)
        tracker.span.finish(status="error")
        span_context._add_span_to_collector(tracker.span)

        self.dropped_spans.append(tracker)
        return None


class EvalynLlamaIndexEventHandler:
    """
    EventHandler that enriches evalyn spans with LlamaIndex event data.

    Events provide semantic-level information (prompts, completions, retrieved
    nodes, tool calls) that the SpanHandler's bound_args may not expose directly.
    """

    @classmethod
    def class_name(cls) -> str:
        return "EvalynLlamaIndexEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        """Handle a LlamaIndex event by enriching the current span."""
        event_type = type(event).__name__

        # LLM events
        if event_type == "LLMChatStartEvent":
            self._handle_llm_chat_start(event)
        elif event_type == "LLMChatEndEvent":
            self._handle_llm_chat_end(event)
        elif event_type == "LLMCompletionStartEvent":
            self._handle_llm_completion_start(event)
        elif event_type == "LLMCompletionEndEvent":
            self._handle_llm_completion_end(event)
        elif event_type == "LLMPredictStartEvent":
            self._handle_llm_predict_start(event)
        elif event_type == "LLMPredictEndEvent":
            self._handle_llm_predict_end(event)

        # Retrieval events
        elif event_type == "RetrievalStartEvent":
            self._handle_retrieval_start(event)
        elif event_type == "RetrievalEndEvent":
            self._handle_retrieval_end(event)

        # Embedding events
        elif event_type == "EmbeddingStartEvent":
            self._handle_embedding_start(event)
        elif event_type == "EmbeddingEndEvent":
            self._handle_embedding_end(event)

        # Agent events
        elif event_type == "AgentToolCallEvent":
            self._handle_agent_tool_call(event)

        # Query events
        elif event_type == "QueryStartEvent":
            self._handle_query_start(event)

        return None

    def _handle_llm_chat_start(self, event: Any) -> None:
        messages = getattr(event, "messages", None)
        model_dict = getattr(event, "model_dict", None)
        if messages:
            _set_current_span_attr(
                "llamaindex.llm.messages",
                _format_llm_messages(messages),
            )
        if model_dict:
            _set_current_span_attr(
                "llamaindex.llm.model_config", str(model_dict)[:500]
            )

    def _handle_llm_chat_end(self, event: Any) -> None:
        response = getattr(event, "response", None)
        if not response:
            return

        message = getattr(response, "message", None)
        if message:
            content = getattr(message, "content", None)
            if content:
                _set_current_span_attr(
                    "llamaindex.llm.response", str(content)[:1000]
                )
            role = getattr(message, "role", None)
            if role:
                _set_current_span_attr("llamaindex.llm.response_role", str(role))

        raw = getattr(response, "raw", None)
        if raw:
            usage = getattr(raw, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                _set_current_span_attr("input_tokens", input_tokens)
                _set_current_span_attr("output_tokens", output_tokens)

    def _handle_llm_completion_start(self, event: Any) -> None:
        prompt = getattr(event, "prompt", None)
        if prompt:
            _set_current_span_attr("llamaindex.llm.prompt", str(prompt)[:1000])

    def _handle_llm_completion_end(self, event: Any) -> None:
        response = getattr(event, "response", None)
        if response:
            text = getattr(response, "text", None)
            if text:
                _set_current_span_attr(
                    "llamaindex.llm.completion", str(text)[:1000]
                )

    def _handle_llm_predict_start(self, event: Any) -> None:
        template = getattr(event, "template", None)
        if template:
            _set_current_span_attr(
                "llamaindex.llm.template", str(template)[:500]
            )

    def _handle_llm_predict_end(self, event: Any) -> None:
        output = getattr(event, "output", None)
        if output:
            _set_current_span_attr("llamaindex.llm.output", str(output)[:1000])

    def _handle_retrieval_start(self, event: Any) -> None:
        query = getattr(event, "str_or_query_bundle", None)
        if query:
            _set_current_span_attr("llamaindex.retrieval.query", str(query)[:500])

    def _handle_retrieval_end(self, event: Any) -> None:
        nodes = getattr(event, "nodes", None)
        if nodes:
            _set_current_span_attr("llamaindex.retrieval.node_count", len(nodes))
            scores = [getattr(node, "score", None) for node in nodes[:10]]
            if scores:
                _set_current_span_attr("llamaindex.retrieval.scores", scores)

    def _handle_embedding_start(self, event: Any) -> None:
        model_dict = getattr(event, "model_dict", None)
        if model_dict:
            _set_current_span_attr(
                "llamaindex.embedding.model", str(model_dict)[:200]
            )

    def _handle_embedding_end(self, event: Any) -> None:
        chunks = getattr(event, "chunks", None)
        if chunks:
            _set_current_span_attr("llamaindex.embedding.chunk_count", len(chunks))

    def _handle_agent_tool_call(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        arguments = getattr(event, "arguments", None)
        if tool:
            tool_name = getattr(tool, "name", None) or str(tool)
            _set_current_span_attr("llamaindex.tool.name", str(tool_name))
        if arguments:
            _set_current_span_attr(
                "llamaindex.tool.arguments", str(arguments)[:500]
            )

    def _handle_query_start(self, event: Any) -> None:
        query = getattr(event, "query", None)
        if query:
            _set_current_span_attr("llamaindex.query", str(query)[:500])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_span_info(
    instance: Any | None,
    bound_args: Any,
    tags: dict[str, Any] | None,
) -> tuple:
    """Infer span name and type from the instance class."""
    if instance is None:
        return ("llamaindex:operation", "custom")

    class_name = type(instance).__name__
    class_lower = class_name.lower()

    # LLM classes
    if "llm" in class_lower or class_name in ("OpenAI", "Anthropic", "Gemini"):
        model = getattr(instance, "model", None) or class_name
        return (f"llamaindex:{model}", "llm_call")

    # Query engines
    if "queryengine" in class_lower or "query_engine" in class_lower:
        return (f"query:{class_name}", "agent")

    # Retrievers
    if "retriever" in class_lower:
        return (f"retrieval:{class_name}", "retrieval")

    # Agents
    if "agent" in class_lower:
        return (f"agent:{class_name}", "agent")

    # Embedders
    if "embed" in class_lower:
        return (f"embedding:{class_name}", "custom")

    # Synthesizers
    if "synth" in class_lower:
        return (f"synthesis:{class_name}", "custom")

    return (f"llamaindex:{class_name}", "custom")


def _set_current_span_attr(key: str, value: Any) -> None:
    """Set a pending attribute on the current span in the stack.

    The EventHandler fires while the SpanHandler's span is still open,
    but we only have the span ID - not the object. Pending attributes
    are applied when the SpanHandler calls _enrich_span_with_result.
    """
    stack = span_context._span_stack.get()
    if not stack:
        return

    _pending_attrs.setdefault(stack[-1], {})[key] = value


_pending_attrs: dict[str, dict[str, Any]] = {}


def _enrich_span_with_result(span: Span, result: Any) -> None:
    """Apply pending event attributes and extract common result patterns."""
    pending = _pending_attrs.pop(span.id, {})
    span.attributes.update(pending)

    if hasattr(result, "response"):
        span.attributes["llamaindex.response"] = str(result.response)[:1000]

    if hasattr(result, "source_nodes"):
        nodes = result.source_nodes
        if nodes:
            span.attributes["llamaindex.source_node_count"] = len(nodes)


def _format_llm_messages(messages: Any) -> list:
    """Format LLM messages for storage."""
    if not messages:
        return []

    formatted = []
    for msg in list(messages)[:20]:
        role = getattr(msg, "role", None) or "unknown"
        content = getattr(msg, "content", None) or ""
        formatted.append(
            {
                "role": str(role),
                "content": str(content)[:500],
            }
        )
    return formatted
