"""
LangGraph SDK instrumentor.

Patches LangGraph to capture graph and node execution spans.
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any, Callable, Optional

from ..base import Instrumentor, InstrumentorType
from ....models import Span
from ... import context as span_context


class LangGraphInstrumentor(Instrumentor):
    """Instrumentor for LangGraph SDK."""

    _instrumented = False
    _original_compile: Optional[Any] = None
    _original_init: Optional[Any] = None

    @property
    def name(self) -> str:
        return "langgraph"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("langgraph") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            from langgraph.graph import StateGraph
            from langgraph.graph.graph import CompiledGraph
        except ImportError:
            return False

        # Patch StateGraph.compile to wrap the resulting graph
        self._original_compile = StateGraph.compile

        @functools.wraps(self._original_compile)
        def patched_compile(inst, *args, **kwargs):
            graph = self._original_compile(inst, *args, **kwargs)
            return self._wrap_compiled_graph(graph)

        StateGraph.compile = patched_compile

        # Also patch CompiledGraph directly for already-compiled graphs
        self._patch_compiled_graph_class(CompiledGraph)

        self._instrumented = True
        return True

    def _wrap_compiled_graph(self, graph: Any) -> Any:
        """Wrap a compiled graph to add instrumentation."""
        # Wrap invoke method
        original_invoke = graph.invoke

        @functools.wraps(original_invoke)
        def wrapped_invoke(input_data, config=None, **kwargs):
            graph_name = getattr(graph, "name", "graph")

            # Create graph execution span
            parent_span_id = span_context.get_current_span_id()
            graph_span = Span.new(
                name=f"graph:{graph_name}",
                span_type="graph",
                parent_id=parent_span_id,
            )

            # Push graph span onto stack so node spans are children
            stack = span_context._span_stack.get()
            new_stack = stack + [graph_span.id]
            _token = span_context._span_stack.set(new_stack)  # noqa: F841

            try:
                result = original_invoke(input_data, config=config, **kwargs)
                graph_span.finish(status="ok")
                return result
            except Exception as e:
                graph_span.finish(status="error", error=str(e))
                raise
            finally:
                span_context._span_stack.set(stack)
                span_context._add_span_to_collector(graph_span)

        graph.invoke = wrapped_invoke

        # Wrap ainvoke for async
        if hasattr(graph, "ainvoke"):
            original_ainvoke = graph.ainvoke

            @functools.wraps(original_ainvoke)
            async def wrapped_ainvoke(input_data, config=None, **kwargs):
                graph_name = getattr(graph, "name", "graph")

                parent_span_id = span_context.get_current_span_id()
                graph_span = Span.new(
                    name=f"graph:{graph_name}",
                    span_type="graph",
                    parent_id=parent_span_id,
                )

                stack = span_context._span_stack.get()
                new_stack = stack + [graph_span.id]
                _token = span_context._span_stack.set(new_stack)  # noqa: F841

                try:
                    result = await original_ainvoke(input_data, config=config, **kwargs)
                    graph_span.finish(status="ok")
                    return result
                except Exception as e:
                    graph_span.finish(status="error", error=str(e))
                    raise
                finally:
                    span_context._span_stack.set(stack)
                    span_context._add_span_to_collector(graph_span)

            graph.ainvoke = wrapped_ainvoke

        return graph

    def _patch_compiled_graph_class(self, CompiledGraph: type) -> None:
        """Patch the CompiledGraph class to instrument node execution."""
        # Patch _execute_node if it exists (internal method)
        if hasattr(CompiledGraph, "_execute_node"):
            original_execute = CompiledGraph._execute_node

            @functools.wraps(original_execute)
            def patched_execute(inst, node_name, *args, **kwargs):
                parent_span_id = span_context.get_current_span_id()

                # Create node span
                node_span = Span.new(
                    name=f"node:{node_name}",
                    span_type="node",
                    parent_id=parent_span_id,
                    node_name=node_name,
                )

                # Push node span so LLM/tool calls are children
                stack = span_context._span_stack.get()
                new_stack = stack + [node_span.id]
                _token = span_context._span_stack.set(new_stack)  # noqa: F841

                try:
                    result = original_execute(inst, node_name, *args, **kwargs)
                    node_span.finish(status="ok")
                    return result
                except Exception as e:
                    node_span.finish(status="error", error=str(e))
                    raise
                finally:
                    span_context._span_stack.set(stack)
                    span_context._add_span_to_collector(node_span)

            CompiledGraph._execute_node = patched_execute

        # Also try to patch via nodes dict iteration
        if hasattr(CompiledGraph, "nodes"):
            self._patch_node_runners(CompiledGraph)

    def _patch_node_runners(self, CompiledGraph: type) -> None:
        """Patch node execution by wrapping the node functions."""
        self._original_init = CompiledGraph.__init__

        @functools.wraps(self._original_init)
        def patched_init(inst, *args, **kwargs):
            self._original_init(inst, *args, **kwargs)

            # Wrap each node's function
            if hasattr(inst, "nodes") and isinstance(inst.nodes, dict):
                wrapped_nodes = {}
                for node_name, node_func in inst.nodes.items():
                    wrapped_nodes[node_name] = self._wrap_node_function(node_name, node_func)
                inst.nodes = wrapped_nodes

        CompiledGraph.__init__ = patched_init

    def _wrap_node_function(self, node_name: str, node_func: Callable) -> Callable:
        """Wrap a node function to create a span around its execution."""
        import asyncio

        if asyncio.iscoroutinefunction(node_func):

            @functools.wraps(node_func)
            async def async_wrapper(*args, **kwargs):
                parent_span_id = span_context.get_current_span_id()
                node_span = Span.new(
                    name=f"node:{node_name}",
                    span_type="node",
                    parent_id=parent_span_id,
                    node_name=node_name,
                )

                stack = span_context._span_stack.get()
                new_stack = stack + [node_span.id]
                _token = span_context._span_stack.set(new_stack)  # noqa: F841

                try:
                    result = await node_func(*args, **kwargs)
                    node_span.finish(status="ok")
                    return result
                except Exception as e:
                    node_span.finish(status="error", error=str(e))
                    raise
                finally:
                    span_context._span_stack.set(stack)
                    span_context._add_span_to_collector(node_span)

            return async_wrapper
        else:

            @functools.wraps(node_func)
            def sync_wrapper(*args, **kwargs):
                parent_span_id = span_context.get_current_span_id()
                node_span = Span.new(
                    name=f"node:{node_name}",
                    span_type="node",
                    parent_id=parent_span_id,
                    node_name=node_name,
                )

                stack = span_context._span_stack.get()
                new_stack = stack + [node_span.id]
                _token = span_context._span_stack.set(new_stack)  # noqa: F841

                try:
                    result = node_func(*args, **kwargs)
                    node_span.finish(status="ok")
                    return result
                except Exception as e:
                    node_span.finish(status="error", error=str(e))
                    raise
                finally:
                    span_context._span_stack.set(stack)
                    span_context._add_span_to_collector(node_span)

            return sync_wrapper

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            from langgraph.graph import StateGraph
            from langgraph.graph.graph import CompiledGraph

            if self._original_compile:
                StateGraph.compile = self._original_compile
            if self._original_init:
                CompiledGraph.__init__ = self._original_init

            self._instrumented = False
            self._original_compile = None
            self._original_init = None
            return True
        except ImportError:
            return False
