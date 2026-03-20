"""
Import tests for evalyn_sdk.

These tests verify that all public APIs can be imported correctly.
Run with: pytest tests/test_imports.py -v
"""
from __future__ import annotations

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_main_module_import(self):
        """Test main evalyn_sdk import."""
        import evalyn_sdk
        assert hasattr(evalyn_sdk, "__version__")

    def test_eval_decorator(self):
        """Test @eval decorator import."""
        from evalyn_sdk import eval, configure_tracer, get_default_tracer
        assert callable(eval)
        assert callable(configure_tracer)
        assert callable(get_default_tracer)

    def test_tracer_import(self):
        """Test EvalTracer import."""
        from evalyn_sdk import EvalTracer, eval_session
        assert EvalTracer is not None
        assert callable(eval_session)

    def test_runner_import(self):
        """Test EvalRunner import."""
        from evalyn_sdk import EvalRunner
        assert EvalRunner is not None


class TestTraceModuleImports:
    """Test trace module imports."""

    def test_trace_module(self):
        """Test trace module import."""
        from evalyn_sdk import trace
        assert hasattr(trace, "span")
        assert hasattr(trace, "EvalTracer")

    def test_trace_submodule_imports(self):
        """Test trace submodule imports."""
        from evalyn_sdk.trace import (
            span,
            get_current_span_id,
            EvalTracer,
            eval_session,
            patch_all,
            is_patched,
        )
        assert callable(span)
        assert callable(get_current_span_id)

    def test_auto_instrument(self):
        """Test auto-instrumentation imports."""
        from evalyn_sdk import auto_instrument, patch_all, is_patched, calculate_cost
        assert auto_instrument is not None
        assert callable(patch_all)
        assert callable(is_patched)
        assert callable(calculate_cost)

    def test_trace_decorator(self):
        """Test trace decorator import."""
        from evalyn_sdk import trace_decorator
        assert callable(trace_decorator)

    def test_otel_imports(self):
        """Test OpenTelemetry imports."""
        from evalyn_sdk import configure_otel, configure_default_otel
        assert callable(configure_otel)
        assert callable(configure_default_otel)


class TestModelImports:
    """Test model imports."""

    def test_core_models(self):
        """Test core model imports."""
        from evalyn_sdk import (
            Annotation,
            CalibrationRecord,
            DatasetItem,
            EvalRun,
            FunctionCall,
            MetricResult,
            MetricSpec,
            MetricType,
        )
        assert all([
            Annotation, CalibrationRecord, DatasetItem,
            EvalRun, FunctionCall, MetricResult, MetricSpec, MetricType
        ])


class TestMetricImports:
    """Test metric imports."""

    def test_registry(self):
        """Test MetricRegistry import."""
        from evalyn_sdk import MetricRegistry, Metric
        assert MetricRegistry is not None
        assert Metric is not None

    def test_judges(self):
        """Test judge imports."""
        from evalyn_sdk import LLMJudge, EchoJudge, JUDGE_TEMPLATES
        assert all([LLMJudge, EchoJudge])
        assert isinstance(JUDGE_TEMPLATES, dict)

    def test_suggesters(self):
        """Test suggester imports."""
        from evalyn_sdk import (
            MetricSuggester,
            HeuristicSuggester,
            LLMSuggester,
            LLMRegistrySelector,
            DEFAULT_JUDGE_PROMPT,
        )
        assert all([MetricSuggester, HeuristicSuggester, LLMSuggester, LLMRegistrySelector])

    def test_objective_metrics(self):
        """Test objective metric imports."""
        from evalyn_sdk import (
            exact_match_metric,
            latency_metric,
            cost_metric,
            bleu_metric,
            pass_at_k_metric,
            json_valid_metric,
            regex_match_metric,
            token_length_metric,
            tool_call_count_metric,
            register_builtin_metrics,
        )
        assert all([
            callable(exact_match_metric),
            callable(latency_metric),
            callable(register_builtin_metrics),
        ])

    def test_subjective_metrics(self):
        """Test subjective metric imports via LLMJudge templates."""
        from evalyn_sdk import LLMJudge, SUBJECTIVE_REGISTRY
        # Templates are now accessed via LLMJudge.from_template()
        assert hasattr(LLMJudge, "from_template")
        assert hasattr(LLMJudge, "list_templates")
        assert len(LLMJudge.list_templates()) > 0
        # SUBJECTIVE_REGISTRY is the list of all subjective metric templates
        assert isinstance(SUBJECTIVE_REGISTRY, list)
        assert len(SUBJECTIVE_REGISTRY) > 0

    def test_templates(self):
        """Test template imports."""
        from evalyn_sdk import (
            OBJECTIVE_REGISTRY,
            SUBJECTIVE_REGISTRY,
            build_objective_metric,
            build_subjective_metric,
            list_template_ids,
        )
        # Templates are lists of template dicts
        assert isinstance(OBJECTIVE_REGISTRY, list)
        assert isinstance(SUBJECTIVE_REGISTRY, list)
        assert len(OBJECTIVE_REGISTRY) > 0
        assert len(SUBJECTIVE_REGISTRY) > 0
        assert callable(build_objective_metric)
        assert callable(build_subjective_metric)
        assert callable(list_template_ids)


class TestAnnotationModuleImports:
    """Test annotation module imports."""

    def test_annotation_module(self):
        """Test annotation module import."""
        from evalyn_sdk import annotation
        assert annotation is not None

    def test_calibration_imports(self):
        """Test calibration imports."""
        from evalyn_sdk import (
            CalibrationEngine,
            AlignmentMetrics,
            BasicOptimizer,
            PromptOptimizationResult,
            save_calibration,
            load_optimized_prompt,
        )
        assert all([
            CalibrationEngine, AlignmentMetrics,
            BasicOptimizer, PromptOptimizationResult,
        ])

    def test_gepa_imports(self):
        """Test GEPA imports."""
        from evalyn_sdk import GEPAConfig, GEPAOptimizer, GEPA_AVAILABLE
        assert GEPAConfig is not None
        assert GEPAOptimizer is not None
        assert isinstance(GEPA_AVAILABLE, bool)

    def test_ape_imports(self):
        """Test APE imports."""
        from evalyn_sdk import APEConfig, APEOptimizer
        assert APEConfig is not None
        assert APEOptimizer is not None

    def test_span_annotation_imports(self):
        """Test span annotation imports."""
        from evalyn_sdk import (
            SpanAnnotation,
            AnnotationSpanType,
            LLMCallAnnotation,
            ToolCallAnnotation,
            ReasoningAnnotation,
            RetrievalAnnotation,
            OverallAnnotation,
            extract_spans_from_trace,
            get_annotation_prompts,
            ANNOTATION_SCHEMAS,
        )
        assert all([
            SpanAnnotation, AnnotationSpanType, LLMCallAnnotation, ToolCallAnnotation,
            ReasoningAnnotation, RetrievalAnnotation, OverallAnnotation,
        ])
        assert callable(extract_spans_from_trace)
        assert callable(get_annotation_prompts)
        assert isinstance(ANNOTATION_SCHEMAS, dict)


class TestSimulationModuleImports:
    """Test simulation module imports."""

    def test_simulation_module(self):
        """Test simulation module import."""
        from evalyn_sdk import simulation
        assert simulation is not None

    def test_simulator_imports(self):
        """Test simulator imports."""
        from evalyn_sdk import (
            UserSimulator,
            AgentSimulator,
            SimulationConfig,
            GeneratedQuery,
        )
        assert all([UserSimulator, AgentSimulator, SimulationConfig, GeneratedQuery])

    def test_simulation_functions(self):
        """Test simulation function imports."""
        from evalyn_sdk import (
            synthetic_dataset,
            simulate_agent,
        )
        assert callable(synthetic_dataset)
        assert callable(simulate_agent)


class TestDatasetImports:
    """Test dataset imports."""

    def test_dataset_functions(self):
        """Test dataset function imports."""
        from evalyn_sdk import (
            load_dataset,
            save_dataset,
            hash_inputs,
            dataset_from_calls,
            build_dataset_from_storage,
        )
        assert all([
            callable(load_dataset),
            callable(save_dataset),
            callable(hash_inputs),
            callable(dataset_from_calls),
            callable(build_dataset_from_storage),
        ])


class TestCLIImports:
    """Test CLI imports."""

    def test_cli_main(self):
        """Test CLI main function import."""
        from evalyn_sdk.cli import main
        assert callable(main)


class TestAllExports:
    """Test __all__ exports are complete."""

    def test_all_exports_importable(self):
        """Test all items in __all__ can be imported."""
        import evalyn_sdk

        missing = []
        for name in evalyn_sdk.__all__:
            if not hasattr(evalyn_sdk, name):
                missing.append(name)

        if missing:
            pytest.fail(f"Missing exports in __all__: {missing}")

    def test_no_private_in_all(self):
        """Test no private items in __all__."""
        import evalyn_sdk

        private = [name for name in evalyn_sdk.__all__ if name.startswith("_")]
        if private:
            pytest.fail(f"Private items in __all__: {private}")
