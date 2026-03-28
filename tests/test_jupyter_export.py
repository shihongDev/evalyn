"""Tests for Jupyter notebook export."""

import json
import pytest
from evalyn_sdk.analysis.jupyter_export import (
    NotebookCell,
    NotebookConfig,
    ExportedNotebook,
    create_markdown_cell,
    create_code_cell,
    build_summary_cells,
    build_chart_cells,
    build_notebook,
    export_notebook,
)


# =============================================================================
# NotebookCell
# =============================================================================


class TestNotebookCell:
    def test_default_type_is_code(self):
        c = NotebookCell()
        assert c.cell_type == "code"

    def test_default_source_empty(self):
        c = NotebookCell()
        assert c.source == ""

    def test_default_outputs_empty(self):
        c = NotebookCell()
        assert c.outputs == []

    def test_as_dict(self):
        c = NotebookCell(cell_type="markdown", source="hello", outputs=[1])
        d = c.as_dict()
        assert d["cell_type"] == "markdown"
        assert d["source"] == "hello"
        assert d["outputs"] == [1]

    def test_as_dict_returns_new_list(self):
        c = NotebookCell(outputs=[1, 2])
        d = c.as_dict()
        d["outputs"].append(3)
        assert len(c.outputs) == 2


# =============================================================================
# NotebookConfig
# =============================================================================


class TestNotebookConfig:
    def test_defaults(self):
        cfg = NotebookConfig()
        assert cfg.title == "Evalyn Analysis"
        assert cfg.include_charts is True
        assert cfg.include_summary is True
        assert cfg.include_raw_data is False

    def test_as_dict(self):
        cfg = NotebookConfig(title="My Run")
        d = cfg.as_dict()
        assert d["title"] == "My Run"
        assert d["include_charts"] is True

    def test_from_dict(self):
        cfg = NotebookConfig.from_dict({
            "title": "Restored",
            "include_charts": False,
        })
        assert cfg.title == "Restored"
        assert cfg.include_charts is False
        assert cfg.include_summary is True

    def test_from_dict_empty(self):
        cfg = NotebookConfig.from_dict({})
        assert cfg.title == "Evalyn Analysis"

    def test_roundtrip(self):
        original = NotebookConfig(title="RT", include_raw_data=True)
        restored = NotebookConfig.from_dict(original.as_dict())
        assert restored.title == original.title
        assert restored.include_raw_data == original.include_raw_data


# =============================================================================
# ExportedNotebook
# =============================================================================


class TestExportedNotebook:
    def test_empty_notebook(self):
        nb = ExportedNotebook()
        assert nb.cells == []
        assert nb.metadata == {}

    def test_as_dict(self):
        nb = ExportedNotebook(
            cells=[NotebookCell(source="x")],
            metadata={"k": "v"},
        )
        d = nb.as_dict()
        assert len(d["cells"]) == 1
        assert d["metadata"]["k"] == "v"

    def test_to_ipynb_dict_has_nbformat(self):
        nb = ExportedNotebook()
        ipynb = nb.to_ipynb_dict()
        assert ipynb["nbformat"] == 4
        assert ipynb["nbformat_minor"] == 5
        assert "cells" in ipynb
        assert "metadata" in ipynb

    def test_to_ipynb_dict_code_cell_has_outputs(self):
        nb = ExportedNotebook(cells=[NotebookCell(cell_type="code", source="1+1")])
        ipynb = nb.to_ipynb_dict()
        cell = ipynb["cells"][0]
        assert "outputs" in cell
        assert "execution_count" in cell

    def test_to_ipynb_dict_markdown_cell_no_outputs(self):
        nb = ExportedNotebook(cells=[NotebookCell(cell_type="markdown", source="# Hi")])
        ipynb = nb.to_ipynb_dict()
        cell = ipynb["cells"][0]
        assert "outputs" not in cell

    def test_to_ipynb_dict_source_is_list(self):
        nb = ExportedNotebook(cells=[NotebookCell(source="line1\nline2")])
        ipynb = nb.to_ipynb_dict()
        src = ipynb["cells"][0]["source"]
        assert isinstance(src, list)

    def test_to_json_valid(self):
        nb = ExportedNotebook(cells=[NotebookCell(source="x")])
        j = nb.to_json()
        parsed = json.loads(j)
        assert parsed["nbformat"] == 4

    def test_to_json_roundtrip(self):
        nb = ExportedNotebook(
            cells=[NotebookCell(source="test")],
            metadata={"key": "val"},
        )
        parsed = json.loads(nb.to_json())
        assert len(parsed["cells"]) == 1


# =============================================================================
# Factory functions
# =============================================================================


class TestFactories:
    def test_create_markdown_cell_type(self):
        c = create_markdown_cell("# Title")
        assert c.cell_type == "markdown"
        assert c.source == "# Title"

    def test_create_code_cell_type(self):
        c = create_code_cell("print(1)")
        assert c.cell_type == "code"
        assert c.source == "print(1)"


# =============================================================================
# build_summary_cells
# =============================================================================


class TestBuildSummaryCells:
    def test_has_cells(self):
        data = {"run_id": "r1", "pass_rate": 0.8, "total_items": 10, "metrics": {"a": 0.9}}
        cells = build_summary_cells(data)
        assert len(cells) >= 2

    def test_contains_run_id(self):
        data = {"run_id": "abc123", "pass_rate": 1.0, "total_items": 5, "metrics": {}}
        cells = build_summary_cells(data)
        combined = " ".join(c.source for c in cells)
        assert "abc123" in combined

    def test_empty_data(self):
        cells = build_summary_cells({})
        assert len(cells) >= 1

    def test_has_markdown_and_code(self):
        data = {"run_id": "r", "pass_rate": 0.5, "total_items": 1, "metrics": {"m": 1}}
        cells = build_summary_cells(data)
        types = {c.cell_type for c in cells}
        assert "markdown" in types
        assert "code" in types


# =============================================================================
# build_chart_cells
# =============================================================================


class TestBuildChartCells:
    def test_has_code(self):
        cells = build_chart_cells({"accuracy": [0.8, 0.9, 0.7]})
        assert len(cells) >= 1
        assert all(c.cell_type == "code" for c in cells)

    def test_empty_scores(self):
        cells = build_chart_cells({})
        assert cells == []

    def test_contains_matplotlib_import(self):
        cells = build_chart_cells({"m": [1.0]})
        combined = " ".join(c.source for c in cells)
        assert "matplotlib" in combined

    def test_multiple_metrics(self):
        cells = build_chart_cells({"a": [0.1], "b": [0.2], "c": [0.3]})
        # One import cell + 3 metric cells
        assert len(cells) == 4


# =============================================================================
# build_notebook
# =============================================================================


class TestBuildNotebook:
    def test_complete_build(self):
        data = {
            "run_id": "run1",
            "pass_rate": 0.9,
            "total_items": 20,
            "metrics": {"accuracy": 0.95},
            "metric_scores": {"accuracy": [0.8, 0.9, 1.0]},
        }
        nb = build_notebook(data)
        assert isinstance(nb, ExportedNotebook)
        assert len(nb.cells) >= 3

    def test_custom_title(self):
        nb = build_notebook({}, config=NotebookConfig(title="Custom"))
        assert any("Custom" in c.source for c in nb.cells)

    def test_no_charts(self):
        data = {"metric_scores": {"m": [1.0]}}
        nb = build_notebook(data, config=NotebookConfig(include_charts=False))
        combined = " ".join(c.source for c in nb.cells)
        assert "matplotlib" not in combined

    def test_no_summary(self):
        nb = build_notebook(
            {"run_id": "x"},
            config=NotebookConfig(include_summary=False),
        )
        combined = " ".join(c.source for c in nb.cells)
        assert "Run Summary" not in combined

    def test_include_raw_data(self):
        nb = build_notebook(
            {"run_id": "r"},
            config=NotebookConfig(include_raw_data=True, include_charts=False),
        )
        combined = " ".join(c.source for c in nb.cells)
        assert "raw" in combined.lower()

    def test_empty_data_no_crash(self):
        nb = build_notebook({})
        assert len(nb.cells) >= 1


# =============================================================================
# export_notebook
# =============================================================================


class TestExportNotebook:
    def test_writes_file(self, tmp_path):
        nb = ExportedNotebook(cells=[NotebookCell(source="1+1")])
        path = str(tmp_path / "test.ipynb")
        result = export_notebook(nb, path)
        assert result == path
        with open(path) as f:
            data = json.load(f)
        assert data["nbformat"] == 4

    def test_file_is_valid_json(self, tmp_path):
        nb = build_notebook({"run_id": "r", "pass_rate": 0.5, "total_items": 1, "metrics": {}})
        path = str(tmp_path / "out.ipynb")
        export_notebook(nb, path)
        with open(path) as f:
            parsed = json.load(f)
        assert "cells" in parsed

    def test_empty_notebook_export(self, tmp_path):
        nb = ExportedNotebook()
        path = str(tmp_path / "empty.ipynb")
        export_notebook(nb, path)
        with open(path) as f:
            parsed = json.load(f)
        assert parsed["cells"] == []
