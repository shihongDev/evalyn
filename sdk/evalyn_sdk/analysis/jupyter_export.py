"""Jupyter notebook export for evaluation analysis.

Generates .ipynb files with pre-built charts and analysis cells.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotebookCell:
    """A single cell in a Jupyter notebook."""

    cell_type: str = "code"
    source: str = ""
    outputs: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "source": self.source,
            "outputs": list(self.outputs),
        }


@dataclass
class NotebookConfig:
    """Configuration for notebook generation."""

    title: str = "Evalyn Analysis"
    include_charts: bool = True
    include_summary: bool = True
    include_raw_data: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "include_charts": self.include_charts,
            "include_summary": self.include_summary,
            "include_raw_data": self.include_raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotebookConfig:
        return cls(
            title=data.get("title", "Evalyn Analysis"),
            include_charts=data.get("include_charts", True),
            include_summary=data.get("include_summary", True),
            include_raw_data=data.get("include_raw_data", False),
        )


@dataclass
class ExportedNotebook:
    """A complete Jupyter notebook ready for export."""

    cells: list[NotebookCell] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "metadata": dict(self.metadata),
        }

    def to_ipynb_dict(self) -> dict[str, Any]:
        """Full ipynb format with nbformat metadata."""
        ipynb_cells = []
        for c in self.cells:
            cell = {
                "cell_type": c.cell_type,
                "source": c.source.splitlines(True),
                "metadata": {},
            }
            if c.cell_type == "code":
                cell["outputs"] = list(c.outputs)
                cell["execution_count"] = None
            ipynb_cells.append(cell)
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": self.metadata or {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            },
            "cells": ipynb_cells,
        }

    def to_json(self) -> str:
        """JSON string of ipynb."""
        return json.dumps(self.to_ipynb_dict(), indent=1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_markdown_cell(source: str) -> NotebookCell:
    """Create a markdown cell."""
    return NotebookCell(cell_type="markdown", source=source)


def create_code_cell(source: str) -> NotebookCell:
    """Create a code cell."""
    return NotebookCell(cell_type="code", source=source)


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def build_summary_cells(data: dict[str, Any]) -> list[NotebookCell]:
    """Generate markdown and code cells for run summary.

    data keys: run_id, pass_rate, total_items, metrics.
    """
    cells: list[NotebookCell] = []

    run_id = data.get("run_id", "unknown")
    pass_rate = data.get("pass_rate", 0.0)
    total_items = data.get("total_items", 0)
    metrics = data.get("metrics", {})

    cells.append(create_markdown_cell(
        f"## Run Summary\n\n"
        f"- **Run ID**: {run_id}\n"
        f"- **Pass Rate**: {pass_rate:.1%}\n"
        f"- **Total Items**: {total_items}\n"
        f"- **Metrics**: {len(metrics)}"
    ))

    metric_lines = []
    for name, value in metrics.items():
        metric_lines.append(f"  '{name}': {value!r},")
    metric_dict_str = "{\n" + "\n".join(metric_lines) + "\n}" if metric_lines else "{}"

    cells.append(create_code_cell(
        f"# Run metrics summary\n"
        f"metrics = {metric_dict_str}\n"
        f"print(f'Run {run_id}: {{len(metrics)}} metrics, pass rate = {pass_rate:.1%}')"
    ))

    return cells


def build_chart_cells(metric_scores: dict[str, list[float]]) -> list[NotebookCell]:
    """Generate code cells for matplotlib-style charts.

    Produces code strings - does not execute charts.
    """
    cells: list[NotebookCell] = []

    if not metric_scores:
        return cells

    cells.append(create_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np"
    ))

    for metric_name, scores in metric_scores.items():
        scores_repr = repr(scores)
        cells.append(create_code_cell(
            f"# Distribution: {metric_name}\n"
            f"scores = {scores_repr}\n"
            f"plt.figure(figsize=(8, 4))\n"
            f"plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)\n"
            f"plt.title('{metric_name} Score Distribution')\n"
            f"plt.xlabel('Score')\n"
            f"plt.ylabel('Count')\n"
            f"plt.axvline(np.mean(scores), color='red', linestyle='--', label='Mean')\n"
            f"plt.legend()\n"
            f"plt.tight_layout()\n"
            f"plt.show()"
        ))

    return cells


# ---------------------------------------------------------------------------
# Notebook builder and exporter
# ---------------------------------------------------------------------------


def build_notebook(
    data: dict[str, Any],
    config: NotebookConfig | None = None,
) -> ExportedNotebook:
    """Build a full notebook from analysis data."""
    cfg = config or NotebookConfig()
    cells: list[NotebookCell] = []

    cells.append(create_markdown_cell(f"# {cfg.title}"))

    if cfg.include_summary:
        cells.extend(build_summary_cells(data))

    if cfg.include_charts:
        metric_scores = data.get("metric_scores", {})
        cells.extend(build_chart_cells(metric_scores))

    if cfg.include_raw_data:
        raw = json.dumps(data, indent=2, default=str)
        cells.append(create_code_cell(
            f"# Raw analysis data\nimport json\nraw = json.loads('''{raw}''')\nraw"
        ))

    return ExportedNotebook(cells=cells)


def export_notebook(notebook: ExportedNotebook, file_path: str) -> str:
    """Write .ipynb file. Return path."""
    content = notebook.to_json()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path
