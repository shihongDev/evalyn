"""
Google ADK (Agent Development Kit) instrumentor.

Uses hybrid OTEL+Callbacks approach:
- OTEL via openinference-instrumentation-google-adk for automatic span structure
- Callbacks for rich content capture (LlmRequest/LlmResponse, tool args/results)

Captures:
- Agent execution via before/after_agent_callback
- LLM calls via before/after_model_callback with full request/response
- Tool calls via before/after_tool_callback with args and results
- User input and agent output
- Token usage with cache metrics
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
)

from ... import context as span_context
from ....models import Span
from ..base import Instrumentor, InstrumentorType
from ..span_processor import get_or_create_tracer_provider

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from google.adk.events import Event
    from google.genai.types import Content


@dataclass
class SpanState:
    """Track span state across callback pairs."""

    span: Span
    start_time: float


@dataclass
class AgentContext:
    """Track agent context for hierarchy reconstruction."""

    span_id: str
    agent_name: str
    invocation_id: str
    start_time: float
    parent_agent_name: Optional[str] = None


class TokenUsage(TypedDict, total=False):
    """Token usage with cache information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    thoughts_tokens: int


class EvalynADKCallbacks:
    """
    Comprehensive callback handler for Google ADK tracing.

    Captures agent execution, LLM calls, and tool usage as spans.
    Works with the @eval decorator for automatic span collection.

    Usage (automatic via instrumentor):
        from evalyn_sdk import eval

        @eval
        async def run_research(query: str):
            runner = InMemoryRunner(agent=my_agent)
            # All spans captured automatically
            return await runner.run_async(...)

    Usage (manual with callbacks):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            create_adk_callbacks
        )

        callbacks = create_adk_callbacks()
        agent = LlmAgent(
            name="my_agent",
            before_model_callback=callbacks.before_model_callback,
            after_model_callback=callbacks.after_model_callback,
            before_tool_callback=callbacks.before_tool_callback,
            after_tool_callback=callbacks.after_tool_callback,
            before_agent_callback=callbacks.before_agent_callback,
            after_agent_callback=callbacks.after_agent_callback,
        )
    """

    def __init__(self) -> None:
        """Initialize the callback handler."""
        # Track spans by invocation_id for correlation
        self._agent_spans: Dict[str, SpanState] = {}
        self._llm_spans: Dict[str, SpanState] = {}
        self._tool_spans: Dict[str, SpanState] = {}

        # Track agent hierarchy
        self._agent_contexts: Dict[str, AgentContext] = {}

        # Track LLM call count per agent for naming
        self._llm_call_counts: Dict[str, int] = {}

        # Track tool call count per agent for naming
        self._tool_call_counts: Dict[str, int] = {}

        # Accumulated output text
        self._output_text: List[str] = []

        # Run metadata
        self._run_start_time: Optional[float] = None

    def _extract_usage(self, usage_metadata: Any) -> TokenUsage:
        """Extract token usage from GenerateContentResponseUsageMetadata."""
        if usage_metadata is None:
            return {}

        def get_count(attr: str) -> int:
            return getattr(usage_metadata, attr, 0) or 0

        return {
            "prompt_tokens": get_count("prompt_token_count"),
            "completion_tokens": get_count("candidates_token_count"),
            "total_tokens": get_count("total_token_count"),
            "cached_tokens": get_count("cached_content_token_count"),
            "thoughts_tokens": get_count("thoughts_token_count"),
        }

    def _get_agent_span_key(self, ctx: "CallbackContext") -> str:
        """Generate unique key for agent span tracking."""
        return f"{ctx.agent_name}:{ctx.invocation_id}"

    def _get_llm_span_key(self, ctx: "CallbackContext") -> str:
        """Generate unique key for LLM span tracking."""
        agent_name = ctx.agent_name
        count = self._llm_call_counts.get(agent_name, 0) + 1
        return f"{agent_name}:llm:{count}"

    def _get_tool_span_key(
        self, tool: "BaseTool", ctx: "ToolContext", args: Dict[str, Any]
    ) -> str:
        """Generate unique key for tool span tracking."""
        tool_name = getattr(tool, "name", type(tool).__name__)
        args_hash = hash(str(sorted(args.items()))) if args else 0
        return f"{ctx.agent_name}:{tool_name}:{ctx.invocation_id}:{args_hash}"

    def _extract_text_from_content(self, content: Optional["Content"]) -> str:
        """Extract text from google.genai.types.Content."""
        if content is None:
            return ""

        parts = getattr(content, "parts", []) or []
        texts = [part.text for part in parts if hasattr(part, "text") and part.text]
        return "\n".join(texts)

    def _extract_tool_calls_from_content(
        self, content: Optional["Content"]
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from content."""
        if content is None:
            return []

        parts = getattr(content, "parts", []) or []
        tool_calls = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(
                    {
                        "name": getattr(fc, "name", "unknown"),
                        "args": dict(getattr(fc, "args", {}) or {}),
                    }
                )
        return tool_calls

    def before_agent_callback(self, ctx: "CallbackContext") -> Optional["Content"]:
        """Called before agent execution begins. Creates agent span and tracks hierarchy."""
        if self._run_start_time is None:
            self._run_start_time = time.time()

        agent_name = ctx.agent_name
        invocation_id = ctx.invocation_id
        span_key = self._get_agent_span_key(ctx)
        parent_id = span_context.get_current_span_id()

        # Check if this is a sub-agent call
        parent_agent_name = None
        for agent_ctx in self._agent_contexts.values():
            if agent_ctx.invocation_id != invocation_id:
                parent_agent_name = agent_ctx.agent_name
                break

        span = Span.new(
            name=f"agent:{agent_name}",
            span_type="agent",
            parent_id=parent_id,
            agent_name=agent_name,
            invocation_id=invocation_id,
        )

        if parent_agent_name:
            span.attributes["parent_agent"] = parent_agent_name

        if ctx.session:
            session_id = getattr(ctx.session, "id", None)
            if session_id:
                span.attributes["session_id"] = session_id

        now = time.time()
        self._agent_spans[span_key] = SpanState(span=span, start_time=now)
        self._agent_contexts[span_key] = AgentContext(
            span_id=span.id,
            agent_name=agent_name,
            invocation_id=invocation_id,
            start_time=now,
            parent_agent_name=parent_agent_name,
        )

        # Push span onto stack for hierarchy
        stack = span_context._span_stack.get()
        span_context._span_stack.set(stack + [span.id])

        return None

    def after_agent_callback(self, ctx: "CallbackContext") -> Optional["Content"]:
        """Called after agent execution completes. Finishes the agent span."""
        span_key = self._get_agent_span_key(ctx)
        state = self._agent_spans.pop(span_key, None)

        if state:
            state.span.attributes["duration_ms"] = (
                time.time() - state.start_time
            ) * 1000

            if self._output_text:
                output = "\n".join(self._output_text)
                state.span.attributes["output_preview"] = output[:2000]
                state.span.attributes["output_size"] = len(output)

            state.span.finish(status="ok")
            span_context._add_span_to_collector(state.span)

            stack = span_context._span_stack.get()
            if stack and stack[-1] == state.span.id:
                span_context._span_stack.set(stack[:-1])

        self._agent_contexts.pop(span_key, None)
        return None

    def before_model_callback(
        self, ctx: "CallbackContext", llm_request: "LlmRequest"
    ) -> Optional["LlmResponse"]:
        """Called before LLM model is invoked. Creates LLM call span."""
        agent_name = ctx.agent_name
        count = self._llm_call_counts.get(agent_name, 0) + 1
        self._llm_call_counts[agent_name] = count

        span_key = self._get_llm_span_key(ctx)
        parent_id = span_context.get_current_span_id()
        model = getattr(llm_request, "model", None) or "unknown"

        # Extract prompt content from last 3 messages
        contents = getattr(llm_request, "contents", []) or []
        prompt_parts = [
            text[:1000]
            for content in contents[-3:]
            if (text := self._extract_text_from_content(content))
        ]
        prompt_preview = "\n---\n".join(prompt_parts)[:3000]

        span = Span.new(
            name=f"llm:{model}",
            span_type="llm_call",
            parent_id=parent_id,
            model=model,
            provider="google",
            agent_name=agent_name,
            turn=count,
            invocation_id=ctx.invocation_id,
        )

        if prompt_preview:
            span.attributes["prompt_preview"] = prompt_preview

        config = getattr(llm_request, "config", None)
        if config:
            temp = getattr(config, "temperature", None)
            if temp is not None:
                span.attributes["temperature"] = temp
            max_tokens = getattr(config, "max_output_tokens", None)
            if max_tokens is not None:
                span.attributes["max_output_tokens"] = max_tokens

        self._llm_spans[span_key] = SpanState(span=span, start_time=time.time())
        return None

    def after_model_callback(
        self, ctx: "CallbackContext", llm_response: "LlmResponse"
    ) -> Optional["LlmResponse"]:
        """Called after LLM model returns response. Finishes LLM span."""
        span_key = self._get_llm_span_key(ctx)
        state = self._llm_spans.pop(span_key, None)

        if state:
            state.span.attributes["duration_ms"] = (
                time.time() - state.start_time
            ) * 1000
            content = getattr(llm_response, "content", None)
            response_text = self._extract_text_from_content(content)

            if response_text:
                state.span.attributes["output_preview"] = response_text[:2000]
                state.span.attributes["output"] = response_text[:4000]
                state.span.attributes["output_size"] = len(response_text)
                self._output_text.append(response_text)

            tool_calls = self._extract_tool_calls_from_content(content)
            if tool_calls:
                state.span.attributes["tool_calls"] = [tc["name"] for tc in tool_calls]
                state.span.attributes["tool_call_count"] = len(tool_calls)

            usage = getattr(llm_response, "usage_metadata", None)
            if usage:
                token_usage = self._extract_usage(usage)
                state.span.attributes["prompt_tokens"] = token_usage.get(
                    "prompt_tokens", 0
                )
                state.span.attributes["completion_tokens"] = token_usage.get(
                    "completion_tokens", 0
                )
                state.span.attributes["total_tokens"] = token_usage.get(
                    "total_tokens", 0
                )
                if token_usage.get("cached_tokens"):
                    state.span.attributes["cached_tokens"] = token_usage[
                        "cached_tokens"
                    ]
                if token_usage.get("thoughts_tokens"):
                    state.span.attributes["thoughts_tokens"] = token_usage[
                        "thoughts_tokens"
                    ]

            model_version = getattr(llm_response, "model_version", None)
            if model_version:
                state.span.attributes["model_version"] = model_version

            error_code = getattr(llm_response, "error_code", None)
            error_message = getattr(llm_response, "error_message", None)
            if error_code or error_message:
                state.span.finish(
                    status="error", error_code=error_code, error_message=error_message
                )
            else:
                state.span.finish(status="ok")

            span_context._add_span_to_collector(state.span)

        return None

    def before_tool_callback(
        self,
        tool: "BaseTool",
        args: Dict[str, Any],
        ctx: "ToolContext",
    ) -> Optional[Dict[str, Any]]:
        """Called before tool execution. Creates tool call span."""
        tool_name = getattr(tool, "name", None) or type(tool).__name__
        agent_name = ctx.agent_name

        count = self._tool_call_counts.get(agent_name, 0) + 1
        self._tool_call_counts[agent_name] = count

        span_key = self._get_tool_span_key(tool, ctx, args)
        parent_id = span_context.get_current_span_id()

        args_str = str(args)
        input_size = len(args_str)

        extra_attrs: Dict[str, Any] = {}
        if type(tool).__name__ == "AgentTool":
            sub_agent = getattr(tool, "agent", None)
            if sub_agent:
                sub_agent_name = getattr(sub_agent, "name", None)
                if sub_agent_name:
                    extra_attrs["sub_agent_name"] = sub_agent_name
            extra_attrs["is_agent_tool"] = True

        span = Span.new(
            name=tool_name,
            span_type="tool_call",
            parent_id=parent_id,
            tool_name=tool_name,
            agent_name=agent_name,
            invocation_id=ctx.invocation_id,
            input=args_str[:4000],
            input_size=input_size,
            input_truncated=input_size > 4000,
            **extra_attrs,
        )

        self._tool_spans[span_key] = SpanState(span=span, start_time=time.time())
        return None

    def after_tool_callback(
        self,
        tool: "BaseTool",
        args: Dict[str, Any],
        ctx: "ToolContext",
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Called after tool execution completes. Finishes tool span."""
        span_key = self._get_tool_span_key(tool, ctx, args)
        state = self._tool_spans.pop(span_key, None)

        if state:
            state.span.attributes["duration_ms"] = (
                time.time() - state.start_time
            ) * 1000

            result_str = str(result) if result else ""
            output_size = len(result_str)

            state.span.attributes["output_size"] = output_size
            state.span.attributes["output_truncated"] = output_size > 4000
            state.span.finish(status="ok", output=result_str[:4000])
            span_context._add_span_to_collector(state.span)

        return None

    def capture_user_input(
        self, message: str, session_id: Optional[str] = None
    ) -> None:
        """Capture user input message as a span."""
        span = Span.new(
            name="user_input",
            span_type="user_message",
            parent_id=span_context.get_current_span_id(),
        )
        span.attributes["content"] = message
        span.attributes["content_length"] = len(message)
        if len(message) > 500:
            span.attributes["content_preview"] = message[:500]
            span.attributes["content_truncated"] = True
        if session_id:
            span.attributes["session_id"] = session_id

        span.finish(status="ok")
        span_context._add_span_to_collector(span)

    def capture_final_output(self, output: str) -> None:
        """Capture final agent output as a span."""
        span = Span.new(
            name="agent_output",
            span_type="agent_message",
            parent_id=span_context.get_current_span_id(),
        )
        output_len = len(output)
        span.attributes["content"] = output[:4000]
        span.attributes["content_length"] = output_len
        if output_len > 500:
            span.attributes["content_preview"] = output[:500]
        if output_len > 4000:
            span.attributes["content_truncated"] = True

        span.finish(status="ok")
        span_context._add_span_to_collector(span)

    def reset(self) -> None:
        """Reset state for a new run."""
        self._agent_spans.clear()
        self._llm_spans.clear()
        self._tool_spans.clear()
        self._agent_contexts.clear()
        self._llm_call_counts.clear()
        self._tool_call_counts.clear()
        self._output_text.clear()
        self._run_start_time = None


class ADKStreamAdapter:
    """
    Wraps ADK event stream to capture agent output.

    Use this to process the run_async() event stream and automatically
    capture final output and metrics.
    """

    def __init__(self, callbacks: EvalynADKCallbacks) -> None:
        self._callbacks = callbacks
        self._accumulated_text: List[str] = []
        self._last_usage: Optional[Dict[str, int]] = None

    async def wrap_stream(
        self, events: AsyncIterator["Event"]
    ) -> AsyncIterator["Event"]:
        """Wrap an ADK event stream to intercept and instrument events."""
        async for event in events:
            content = getattr(event, "content", None)
            if content:
                text = self._callbacks._extract_text_from_content(content)
                if text:
                    self._accumulated_text.append(text)

            usage = getattr(event, "usage_metadata", None)
            if usage:
                self._last_usage = self._callbacks._extract_usage(usage)

            is_final = getattr(event, "is_final_response", lambda: False)
            if callable(is_final) and is_final() and self._accumulated_text:
                self._callbacks.capture_final_output("\n".join(self._accumulated_text))

            yield event

    def get_accumulated_output(self) -> str:
        """Get all accumulated text output."""
        return "\n".join(self._accumulated_text)

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Get the last captured usage metadata."""
        return self._last_usage


class GoogleADKInstrumentor(Instrumentor):
    """
    Instrumentor for Google ADK (Agent Development Kit).

    Uses hybrid approach:
    1. OTEL via openinference-instrumentation-google-adk for automatic spans
    2. Callbacks for rich content capture (LlmRequest/LlmResponse, tool I/O)
    3. Automatic callback injection via canonical_*_callbacks property patching
    """

    _instrumented: bool = False
    _otel_instrumentor: Optional[Any] = None
    _callbacks: Optional[EvalynADKCallbacks] = None
    _original_run: Optional[Callable[..., Any]] = None
    _original_run_async: Optional[Callable[..., Any]] = None
    _patched: bool = False
    _callbacks_injected: bool = False
    _original_properties: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "google_adk"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.OTEL_NATIVE

    def is_available(self) -> bool:
        try:
            return importlib.util.find_spec("google.adk") is not None
        except (ModuleNotFoundError, ImportError):
            return False

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        if not self.is_available():
            return False

        try:
            get_or_create_tracer_provider()
            self._callbacks = EvalynADKCallbacks()

            try:
                from openinference.instrumentation.google_adk import (
                    GoogleADKInstrumentor as OIInstrumentor,
                )

                self._otel_instrumentor = OIInstrumentor()
                self._otel_instrumentor.instrument()
            except ImportError:
                self._instrument_manually()

            self._patch_runners()
            self._inject_callbacks()
            self._instrumented = True
            return True
        except Exception:
            return False

    def _instrument_manually(self) -> None:
        """Manual instrumentation when openinference is not available."""
        try:
            from opentelemetry import trace
            from google.adk import Agent

            tracer = trace.get_tracer("evalyn.google_adk")
            original_run = Agent.run

            def patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
                agent_name = getattr(self, "name", "agent")
                with tracer.start_as_current_span(
                    f"agent:{agent_name}",
                    attributes={
                        "openinference.span.kind": "AGENT",
                        "agent.name": agent_name,
                    },
                ):
                    return original_run(self, *args, **kwargs)

            Agent.run = patched_run
            self._original_run = original_run
        except ImportError:
            pass

    def _patch_runners(self) -> None:
        """Patch Runner classes to capture user input."""
        if self._patched:
            return

        try:
            from google.adk.runners import InMemoryRunner
        except ImportError:
            return

        if not hasattr(InMemoryRunner, "run_async"):
            return

        callbacks = self._callbacks
        original_run_async = InMemoryRunner.run_async

        async def patched_run_async(
            self_runner: Any,
            user_id: str,
            session_id: str,
            new_message: Any,
            **kwargs: Any,
        ) -> Any:
            if callbacks is not None:
                if isinstance(new_message, str):
                    message_text = new_message
                else:
                    message_text = callbacks._extract_text_from_content(new_message)
                if message_text:
                    callbacks.capture_user_input(message_text, session_id)

            return await original_run_async(
                self_runner, user_id, session_id, new_message, **kwargs
            )

        InMemoryRunner.run_async = patched_run_async
        self._original_run_async = original_run_async
        self._patched = True

    def _unpatch_runners(self) -> None:
        """Restore original runner methods."""
        if not self._patched:
            return

        try:
            from google.adk.runners import InMemoryRunner

            if self._original_run_async:
                InMemoryRunner.run_async = self._original_run_async
                self._original_run_async = None

            self._patched = False
        except ImportError:
            pass

    def _inject_callbacks(self) -> None:
        """Inject Evalyn callbacks into all LlmAgent instances via property patching."""
        if self._callbacks_injected or self._callbacks is None:
            return

        try:
            from google.adk.agents import LlmAgent
            from google.adk.agents.base_agent import BaseAgent
        except ImportError:
            return

        callbacks = self._callbacks

        def make_patched_getter(
            evalyn_callback: Callable[..., Any],
            original_getter: Callable[..., List[Any]],
        ) -> Callable[[Any], List[Any]]:
            """Create a patched property getter that prepends the Evalyn callback."""

            def patched(instance: Any) -> List[Any]:
                return [evalyn_callback] + original_getter(instance)

            return patched

        # Define callback mappings: (class, property_name, evalyn_callback)
        llm_callbacks = [
            (
                "before_model",
                "canonical_before_model_callbacks",
                callbacks.before_model_callback,
            ),
            (
                "after_model",
                "canonical_after_model_callbacks",
                callbacks.after_model_callback,
            ),
            (
                "before_tool",
                "canonical_before_tool_callbacks",
                callbacks.before_tool_callback,
            ),
            (
                "after_tool",
                "canonical_after_tool_callbacks",
                callbacks.after_tool_callback,
            ),
        ]
        agent_callbacks = [
            (
                "before_agent",
                "canonical_before_agent_callbacks",
                callbacks.before_agent_callback,
            ),
            (
                "after_agent",
                "canonical_after_agent_callbacks",
                callbacks.after_agent_callback,
            ),
        ]

        # Store originals and patch LlmAgent properties
        for key, prop_name, evalyn_cb in llm_callbacks:
            original_getter = getattr(LlmAgent, prop_name).fget
            self._original_properties[key] = original_getter
            setattr(
                LlmAgent,
                prop_name,
                property(make_patched_getter(evalyn_cb, original_getter)),
            )

        # Store originals and patch BaseAgent properties
        for key, prop_name, evalyn_cb in agent_callbacks:
            original_getter = getattr(BaseAgent, prop_name).fget
            self._original_properties[key] = original_getter
            setattr(
                BaseAgent,
                prop_name,
                property(make_patched_getter(evalyn_cb, original_getter)),
            )

        self._callbacks_injected = True

    def _uninject_callbacks(self) -> None:
        """Restore original canonical callback properties."""
        if not self._callbacks_injected or not self._original_properties:
            return

        try:
            from google.adk.agents import LlmAgent
            from google.adk.agents.base_agent import BaseAgent
        except ImportError:
            return

        # Mapping of keys to (class, property_name)
        property_map = {
            "before_model": (LlmAgent, "canonical_before_model_callbacks"),
            "after_model": (LlmAgent, "canonical_after_model_callbacks"),
            "before_tool": (LlmAgent, "canonical_before_tool_callbacks"),
            "after_tool": (LlmAgent, "canonical_after_tool_callbacks"),
            "before_agent": (BaseAgent, "canonical_before_agent_callbacks"),
            "after_agent": (BaseAgent, "canonical_after_agent_callbacks"),
        }

        for key, (cls, prop_name) in property_map.items():
            if original_getter := self._original_properties.get(key):
                setattr(cls, prop_name, property(original_getter))

        self._original_properties = {}
        self._callbacks_injected = False

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            if self._otel_instrumentor:
                self._otel_instrumentor.uninstrument()
                self._otel_instrumentor = None
            elif hasattr(self, "_original_run") and self._original_run:
                from google.adk import Agent

                Agent.run = self._original_run
                del self._original_run

            self._uninject_callbacks()
            self._unpatch_runners()
            self._callbacks = None
            self._instrumented = False
            return True
        except Exception:
            return False

    def get_callbacks(self) -> Optional[EvalynADKCallbacks]:
        """Get the callbacks instance for manual integration."""
        if self._callbacks is None:
            self.instrument()
        return self._callbacks


def create_adk_callbacks() -> EvalynADKCallbacks:
    """
    Create Evalyn ADK callbacks for manual integration.

    Use when passing callbacks directly to LlmAgent. For automatic
    instrumentation, use the @eval decorator instead.
    """
    return EvalynADKCallbacks()


def create_stream_adapter(callbacks: EvalynADKCallbacks) -> ADKStreamAdapter:
    """
    Create a stream adapter for additional instrumentation.

    Wraps the event stream from runner.run_async() to capture final output.
    """
    return ADKStreamAdapter(callbacks)
