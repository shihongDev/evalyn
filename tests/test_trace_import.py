"""Tests for trace import from external platforms."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.integration.trace_import import (
    ImportedTrace,
    TraceImportConfig,
    TraceImportResult,
    import_from_generic,
    import_from_langfuse,
    import_from_phoenix,
    import_traces,
    map_span_type,
    validate_imported_traces,
)


# ---------------------------------------------------------------------------
# map_span_type
# ---------------------------------------------------------------------------


class TestMapSpanType:
    def test_phoenix_llm(self):
        assert map_span_type("LLM", "phoenix") == "llm_call"

    def test_phoenix_tool(self):
        assert map_span_type("TOOL", "phoenix") == "tool_call"

    def test_phoenix_retriever(self):
        assert map_span_type("RETRIEVER", "phoenix") == "retrieval"

    def test_phoenix_unknown(self):
        assert map_span_type("UNKNOWN", "phoenix") == "custom"

    def test_langfuse_generation(self):
        assert map_span_type("GENERATION", "langfuse") == "llm_call"

    def test_langfuse_tool(self):
        assert map_span_type("TOOL", "langfuse") == "tool_call"

    def test_langfuse_retrieval(self):
        assert map_span_type("RETRIEVAL", "langfuse") == "retrieval"

    def test_langsmith_llm(self):
        assert map_span_type("llm", "langsmith") == "llm_call"

    def test_langsmith_tool(self):
        assert map_span_type("tool", "langsmith") == "tool_call"

    def test_langsmith_retriever(self):
        assert map_span_type("retriever", "langsmith") == "retrieval"

    def test_generic_returns_custom(self):
        assert map_span_type("anything", "generic") == "custom"

    def test_unknown_platform_returns_custom(self):
        assert map_span_type("LLM", "unknown_platform") == "custom"


# ---------------------------------------------------------------------------
# import_from_phoenix
# ---------------------------------------------------------------------------


class TestImportFromPhoenix:
    def test_basic_import(self):
        data = [
            {
                "trace_id": "t1",
                "spans": [{"name": "call", "span_kind": "LLM"}],
            }
        ]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.total_imported == 1
        assert result.skipped == 0
        assert len(result.traces) == 1
        assert result.traces[0].trace_id == "t1"
        assert result.traces[0].source_platform == "phoenix"
        assert result.traces[0].spans[0]["span_type"] == "llm_call"

    def test_missing_trace_id_skipped(self):
        data = [{"spans": []}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.total_imported == 0
        assert result.skipped == 1
        assert len(result.errors) == 1

    def test_context_trace_id(self):
        data = [{"context": {"trace_id": "ctx-1"}, "spans": []}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.total_imported == 1
        assert result.traces[0].trace_id == "ctx-1"

    def test_top_level_span(self):
        """When no spans list, a top-level item with 'name' is treated as a single span."""
        data = [{"trace_id": "t2", "name": "root", "span_kind": "TOOL"}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.total_imported == 1
        assert result.traces[0].spans[0]["span_type"] == "tool_call"

    def test_no_span_type_mapping(self):
        data = [
            {"trace_id": "t3", "spans": [{"name": "s", "span_kind": "LLM"}]}
        ]
        config = TraceImportConfig(source_platform="phoenix", map_span_types=False)
        result = import_from_phoenix(data, config)
        assert "span_type" not in result.traces[0].spans[0]

    def test_preserve_ids_false(self):
        data = [
            {"trace_id": "t4", "spans": [{"span_id": "s1", "name": "s"}]}
        ]
        config = TraceImportConfig(source_platform="phoenix", preserve_ids=False)
        result = import_from_phoenix(data, config)
        assert "span_id" not in result.traces[0].spans[0]


# ---------------------------------------------------------------------------
# import_from_langfuse
# ---------------------------------------------------------------------------


class TestImportFromLangfuse:
    def test_basic_import(self):
        data = [
            {
                "id": "lf-1",
                "observations": [
                    {"type": "GENERATION", "name": "gen1"},
                ],
            }
        ]
        config = TraceImportConfig(source_platform="langfuse")
        result = import_from_langfuse(data, config)
        assert result.total_imported == 1
        assert result.traces[0].trace_id == "lf-1"
        assert result.traces[0].source_platform == "langfuse"
        assert result.traces[0].spans[0]["span_type"] == "llm_call"

    def test_trace_id_field(self):
        data = [{"trace_id": "lf-2", "observations": []}]
        config = TraceImportConfig(source_platform="langfuse")
        result = import_from_langfuse(data, config)
        assert result.traces[0].trace_id == "lf-2"

    def test_missing_id_skipped(self):
        data = [{"observations": []}]
        config = TraceImportConfig(source_platform="langfuse")
        result = import_from_langfuse(data, config)
        assert result.total_imported == 0
        assert result.skipped == 1


# ---------------------------------------------------------------------------
# import_from_generic
# ---------------------------------------------------------------------------


class TestImportFromGeneric:
    def test_basic_import(self):
        data = [
            {
                "trace_id": "g1",
                "spans": [{"name": "span1"}],
            }
        ]
        config = TraceImportConfig(source_platform="generic")
        result = import_from_generic(data, config)
        assert result.total_imported == 1
        assert result.traces[0].spans[0]["span_type"] == "custom"

    def test_missing_trace_id(self):
        data = [{"spans": []}]
        config = TraceImportConfig(source_platform="generic")
        result = import_from_generic(data, config)
        assert result.total_imported == 0
        assert result.skipped == 1

    def test_preserves_existing_span_type(self):
        data = [
            {
                "trace_id": "g2",
                "spans": [{"span_type": "llm_call", "name": "s"}],
            }
        ]
        config = TraceImportConfig(source_platform="generic")
        result = import_from_generic(data, config)
        assert result.traces[0].spans[0]["span_type"] == "llm_call"


# ---------------------------------------------------------------------------
# import_traces (router)
# ---------------------------------------------------------------------------


class TestImportTraces:
    def test_routes_to_phoenix(self):
        data = [{"trace_id": "p1", "spans": [{"span_kind": "LLM"}]}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_traces(data, config)
        assert result.total_imported == 1
        assert result.traces[0].source_platform == "phoenix"

    def test_routes_to_langfuse(self):
        data = [{"id": "lf1", "observations": []}]
        config = TraceImportConfig(source_platform="langfuse")
        result = import_traces(data, config)
        assert result.total_imported == 1
        assert result.traces[0].source_platform == "langfuse"

    def test_routes_to_generic(self):
        data = [{"trace_id": "g1", "spans": []}]
        config = TraceImportConfig(source_platform="generic")
        result = import_traces(data, config)
        assert result.total_imported == 1
        assert result.traces[0].source_platform == "generic"

    def test_unknown_platform_uses_generic(self):
        data = [{"trace_id": "u1", "spans": []}]
        config = TraceImportConfig(source_platform="custom_platform")
        result = import_traces(data, config)
        assert result.total_imported == 1

    def test_langsmith_uses_generic(self):
        data = [{"trace_id": "ls1", "spans": []}]
        config = TraceImportConfig(source_platform="langsmith")
        result = import_traces(data, config)
        assert result.total_imported == 1


# ---------------------------------------------------------------------------
# validate_imported_traces
# ---------------------------------------------------------------------------


class TestValidateImportedTraces:
    def test_valid_result(self):
        result = TraceImportResult(
            traces=[
                ImportedTrace(
                    trace_id="t1",
                    spans=[{"name": "s1"}],
                    source_platform="phoenix",
                )
            ],
            total_imported=1,
        )
        valid, issues = validate_imported_traces(result)
        assert valid is True
        assert issues == []

    def test_empty_traces(self):
        result = TraceImportResult()
        valid, issues = validate_imported_traces(result)
        assert valid is False
        assert any("No traces" in i for i in issues)

    def test_empty_trace_id(self):
        result = TraceImportResult(
            traces=[ImportedTrace(trace_id="", spans=[])],
            total_imported=1,
        )
        valid, issues = validate_imported_traces(result)
        assert valid is False
        assert any("empty trace_id" in i for i in issues)


# ---------------------------------------------------------------------------
# Dataclass as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestImportedTraceRoundtrip:
    def test_roundtrip(self):
        trace = ImportedTrace(
            trace_id="rt1",
            spans=[{"name": "s"}],
            source_platform="phoenix",
            imported_at="2026-01-01T00:00:00+00:00",
            metadata={"key": "value"},
        )
        d = trace.as_dict()
        restored = ImportedTrace.from_dict(d)
        assert restored.trace_id == trace.trace_id
        assert restored.spans == trace.spans
        assert restored.source_platform == trace.source_platform
        assert restored.imported_at == trace.imported_at
        assert restored.metadata == trace.metadata

    def test_from_dict_defaults(self):
        trace = ImportedTrace.from_dict({})
        assert trace.trace_id == ""
        assert trace.spans == []
        assert trace.source_platform == ""


class TestTraceImportConfigRoundtrip:
    def test_roundtrip(self):
        config = TraceImportConfig(
            source_platform="langfuse",
            map_span_types=False,
            preserve_ids=False,
        )
        d = config.as_dict()
        restored = TraceImportConfig.from_dict(d)
        assert restored.source_platform == config.source_platform
        assert restored.map_span_types == config.map_span_types
        assert restored.preserve_ids == config.preserve_ids

    def test_from_dict_defaults(self):
        config = TraceImportConfig.from_dict({})
        assert config.source_platform == "phoenix"
        assert config.map_span_types is True
        assert config.preserve_ids is True


class TestTraceImportResultRoundtrip:
    def test_roundtrip(self):
        result = TraceImportResult(
            traces=[ImportedTrace(trace_id="t1", spans=[])],
            total_imported=1,
            skipped=2,
            errors=["err1"],
        )
        d = result.as_dict()
        restored = TraceImportResult.from_dict(d)
        assert restored.total_imported == 1
        assert restored.skipped == 2
        assert restored.errors == ["err1"]
        assert len(restored.traces) == 1
        assert restored.traces[0].trace_id == "t1"

    def test_from_dict_defaults(self):
        result = TraceImportResult.from_dict({})
        assert result.total_imported == 0
        assert result.traces == []


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestTraceImportResultFormatText:
    def test_basic_format(self):
        result = TraceImportResult(total_imported=3, skipped=1)
        text = result.format_text()
        assert "Imported: 3" in text
        assert "Skipped: 1" in text

    def test_format_with_errors(self):
        result = TraceImportResult(
            total_imported=0, skipped=1, errors=["bad data"]
        )
        text = result.format_text()
        assert "Errors: 1" in text
        assert "bad data" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_data_list(self):
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix([], config)
        assert result.total_imported == 0
        assert result.skipped == 0
        assert result.traces == []

    def test_empty_data_langfuse(self):
        config = TraceImportConfig(source_platform="langfuse")
        result = import_from_langfuse([], config)
        assert result.total_imported == 0

    def test_empty_data_generic(self):
        config = TraceImportConfig(source_platform="generic")
        result = import_from_generic([], config)
        assert result.total_imported == 0

    def test_multiple_traces(self):
        data = [
            {"trace_id": f"t{i}", "spans": []} for i in range(5)
        ]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.total_imported == 5

    def test_imported_at_populated(self):
        data = [{"trace_id": "t1", "spans": []}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.traces[0].imported_at != ""

    def test_metadata_preserved(self):
        data = [{"trace_id": "t1", "spans": [], "metadata": {"env": "prod"}}]
        config = TraceImportConfig(source_platform="phoenix")
        result = import_from_phoenix(data, config)
        assert result.traces[0].metadata == {"env": "prod"}
