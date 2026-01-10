"""
LangGraph auto-instrumentation for hierarchical span tracing.

Automatically captures:
- Graph execution spans (graph.invoke)
- Node execution spans with timing
- Conditional edge decisions

This creates a Phoenix/LangSmith-style trace tree:

Trace (run_agent)
 └── graph.execution
     ├── node (generate_query)
     │    └── llm_call (gemini-2.5-flash)
     ├── node (web_research)
     │    └── tool_call (google_search)
     └── node (finalize_answer)
          └── llm_call (gemini-2.5-flash)

Usage:
    # Auto-enabled when evalyn_sdk is imported
    # Or manually:
    from evalyn_sdk.trace.langgraph import patch_langgraph
    patch_langgraph()
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional
from datetime import timedelta

from . import context as span_context
from ..models import Span

_patched = False


def patch_langgraph() -> bool:
    """
    Patch LangGraph to emit spans for graph and node execution.

    Returns True if patching succeeded, False if LangGraph is not installed.
    """
    global _patched
    if _patched:
        return True

    try:
        from langgraph.graph import StateGraph
        from langgraph.graph.graph import CompiledGraph
    except ImportError:
        return False

    # Patch StateGraph.compile to wrap the resulting graph
    original_compile = StateGraph.compile

    @functools.wraps(original_compile)
    def patched_compile(self, *args, **kwargs):
        graph = original_compile(self, *args, **kwargs)
        return _wrap_compiled_graph(graph)

    StateGraph.compile = patched_compile

    # Also patch CompiledGraph directly for already-compiled graphs
    _patch_compiled_graph_class(CompiledGraph)

    _patched = True
    return True


def _wrap_compiled_graph(graph: Any) -> Any:
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
        token = span_context._span_stack.set(new_stack)

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
            token = span_context._span_stack.set(new_stack)

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


def _patch_compiled_graph_class(CompiledGraph: type) -> None:
    """Patch the CompiledGraph class to instrument node execution."""
    # Patch _execute_node if it exists (internal method)
    if hasattr(CompiledGraph, "_execute_node"):
        original_execute = CompiledGraph._execute_node

        @functools.wraps(original_execute)
        def patched_execute(self, node_name, *args, **kwargs):
            start_time = time.time()
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
            token = span_context._span_stack.set(new_stack)

            try:
                result = original_execute(self, node_name, *args, **kwargs)
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
    # LangGraph calls nodes via the nodes attribute
    if hasattr(CompiledGraph, "nodes"):
        _patch_node_runners(CompiledGraph)


def _patch_node_runners(CompiledGraph: type) -> None:
    """
    Patch node execution by wrapping the node functions.

    LangGraph stores nodes as a dict of name -> callable.
    We wrap each callable to add span tracking.
    """
    original_init = CompiledGraph.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Wrap each node's function
        if hasattr(self, "nodes") and isinstance(self.nodes, dict):
            wrapped_nodes = {}
            for node_name, node_func in self.nodes.items():
                wrapped_nodes[node_name] = _wrap_node_function(node_name, node_func)
            self.nodes = wrapped_nodes

    CompiledGraph.__init__ = patched_init


def _wrap_node_function(node_name: str, node_func: Callable) -> Callable:
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
            token = span_context._span_stack.set(new_stack)

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
            token = span_context._span_stack.set(new_stack)

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


def is_patched() -> bool:
    """Check if LangGraph has been patched."""
    return _patched
