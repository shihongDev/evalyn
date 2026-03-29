"""Tests for pipeline visualization module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.pipeline_visualization import (
    Pipeline,
    PipelineStep,
    create_evalyn_pipeline,
    get_status_marker,
    render_box,
    render_horizontal,
    render_pipeline,
    render_vertical,
)


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_default_values(self):
        step = PipelineStep(step_id="s1", name="build")
        assert step.description == ""
        assert step.status == "pending"

    def test_custom_values(self):
        step = PipelineStep(
            step_id="s1", name="build", description="Build step", status="running"
        )
        assert step.step_id == "s1"
        assert step.name == "build"
        assert step.description == "Build step"
        assert step.status == "running"

    def test_as_dict(self):
        step = PipelineStep(step_id="s1", name="build", status="complete")
        d = step.as_dict()
        assert d == {
            "step_id": "s1",
            "name": "build",
            "description": "",
            "status": "complete",
        }

    def test_from_dict_full(self):
        d = {
            "step_id": "s2",
            "name": "test",
            "description": "Run tests",
            "status": "failed",
        }
        step = PipelineStep.from_dict(d)
        assert step.step_id == "s2"
        assert step.name == "test"
        assert step.description == "Run tests"
        assert step.status == "failed"

    def test_from_dict_defaults(self):
        d = {"step_id": "s3", "name": "deploy"}
        step = PipelineStep.from_dict(d)
        assert step.description == ""
        assert step.status == "pending"

    def test_roundtrip(self):
        original = PipelineStep(
            step_id="rt", name="roundtrip", description="test", status="skipped"
        )
        rebuilt = PipelineStep.from_dict(original.as_dict())
        assert rebuilt.step_id == original.step_id
        assert rebuilt.name == original.name
        assert rebuilt.description == original.description
        assert rebuilt.status == original.status


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_empty_pipeline(self):
        p = Pipeline(name="empty")
        assert p.name == "empty"
        assert p.steps == []

    def test_pipeline_with_steps(self):
        steps = [PipelineStep(step_id="1", name="a"), PipelineStep(step_id="2", name="b")]
        p = Pipeline(name="test", steps=steps)
        assert len(p.steps) == 2

    def test_as_dict(self):
        steps = [PipelineStep(step_id="1", name="a")]
        p = Pipeline(name="test", steps=steps)
        d = p.as_dict()
        assert d["name"] == "test"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_id"] == "1"

    def test_from_dict(self):
        d = {
            "name": "my_pipeline",
            "steps": [
                {"step_id": "1", "name": "a", "description": "", "status": "pending"},
            ],
        }
        p = Pipeline.from_dict(d)
        assert p.name == "my_pipeline"
        assert len(p.steps) == 1
        assert p.steps[0].name == "a"

    def test_from_dict_empty_steps(self):
        d = {"name": "empty"}
        p = Pipeline.from_dict(d)
        assert p.steps == []

    def test_roundtrip(self):
        steps = [
            PipelineStep(step_id="1", name="trace", status="complete"),
            PipelineStep(step_id="2", name="eval", status="running"),
        ]
        original = Pipeline(name="roundtrip", steps=steps)
        rebuilt = Pipeline.from_dict(original.as_dict())
        assert rebuilt.name == original.name
        assert len(rebuilt.steps) == len(original.steps)
        for o, r in zip(original.steps, rebuilt.steps):
            assert o.name == r.name
            assert o.status == r.status


# ---------------------------------------------------------------------------
# get_status_marker
# ---------------------------------------------------------------------------


class TestGetStatusMarker:
    def test_complete(self):
        assert get_status_marker("complete") == "[OK]"

    def test_failed(self):
        assert get_status_marker("failed") == "[!!]"

    def test_running(self):
        assert get_status_marker("running") == "[>>]"

    def test_skipped(self):
        assert get_status_marker("skipped") == "[--]"

    def test_pending(self):
        assert get_status_marker("pending") == "[  ]"

    def test_unknown_status(self):
        assert get_status_marker("unknown") == "[  ]"

    def test_empty_string(self):
        assert get_status_marker("") == "[  ]"


# ---------------------------------------------------------------------------
# render_box
# ---------------------------------------------------------------------------


class TestRenderBox:
    def test_basic_box(self):
        lines = render_box("hello", width=10)
        assert len(lines) == 3
        assert lines[0].startswith("+")
        assert lines[0].endswith("+")
        assert lines[2] == lines[0]
        assert "hello" in lines[1]

    def test_box_content_centered(self):
        lines = render_box("hi", width=10)
        content = lines[1]
        # Strip borders
        inner = content[2:-2]
        assert inner == "hi".center(10)

    def test_text_wider_than_width(self):
        lines = render_box("a very long step name", width=5)
        # Width should expand to fit text
        assert "a very long step name" in lines[1]

    def test_box_border_consistency(self):
        lines = render_box("test", width=20)
        assert len(lines[0]) == len(lines[1]) == len(lines[2])

    def test_empty_text(self):
        lines = render_box("", width=10)
        assert len(lines) == 3
        # Content line should have spaces
        assert lines[1].startswith("| ")
        assert lines[1].endswith(" |")

    def test_default_width(self):
        lines = render_box("short")
        assert len(lines) == 3
        # Default width is 20, so border is 20 + 4 = 24
        assert len(lines[0]) == 24


# ---------------------------------------------------------------------------
# render_horizontal
# ---------------------------------------------------------------------------


class TestRenderHorizontal:
    def test_empty_pipeline(self):
        p = Pipeline(name="empty")
        result = render_horizontal(p)
        assert "empty" in result.lower()

    def test_single_step(self):
        p = Pipeline(
            name="one",
            steps=[PipelineStep(step_id="1", name="build", status="complete")],
        )
        result = render_horizontal(p)
        assert "[OK]" in result
        assert "[build]" in result
        assert "-->" not in result

    def test_two_steps_with_arrow(self):
        p = Pipeline(
            name="two",
            steps=[
                PipelineStep(step_id="1", name="build", status="complete"),
                PipelineStep(step_id="2", name="test", status="running"),
            ],
        )
        result = render_horizontal(p)
        assert "-->" in result
        assert "[OK]" in result
        assert "[>>]" in result

    def test_all_statuses(self):
        statuses = ["complete", "failed", "running", "skipped", "pending"]
        steps = [
            PipelineStep(step_id=str(i), name=f"s{i}", status=s)
            for i, s in enumerate(statuses)
        ]
        p = Pipeline(name="all", steps=steps)
        result = render_horizontal(p)
        assert "[OK]" in result
        assert "[!!]" in result
        assert "[>>]" in result
        assert "[--]" in result
        assert "[  ]" in result

    def test_arrow_count(self):
        steps = [PipelineStep(step_id=str(i), name=f"s{i}") for i in range(4)]
        p = Pipeline(name="arrows", steps=steps)
        result = render_horizontal(p)
        assert result.count("-->") == 3

    def test_step_order_preserved(self):
        steps = [
            PipelineStep(step_id="1", name="alpha"),
            PipelineStep(step_id="2", name="beta"),
            PipelineStep(step_id="3", name="gamma"),
        ]
        p = Pipeline(name="order", steps=steps)
        result = render_horizontal(p)
        assert result.index("alpha") < result.index("beta") < result.index("gamma")


# ---------------------------------------------------------------------------
# render_vertical
# ---------------------------------------------------------------------------


class TestRenderVertical:
    def test_empty_pipeline(self):
        p = Pipeline(name="empty")
        result = render_vertical(p)
        assert "empty" in result.lower()

    def test_single_step_no_connector(self):
        p = Pipeline(
            name="one",
            steps=[PipelineStep(step_id="1", name="build", status="complete")],
        )
        result = render_vertical(p)
        assert "[OK]" in result
        assert "build" in result
        # Single step should have no pipe connector between steps
        lines = result.split("\n")
        # Only 3 lines for one box: top border, content, bottom border
        assert len(lines) == 3

    def test_two_steps_with_connector(self):
        p = Pipeline(
            name="two",
            steps=[
                PipelineStep(step_id="1", name="build", status="complete"),
                PipelineStep(step_id="2", name="test", status="pending"),
            ],
        )
        result = render_vertical(p)
        assert "|" in result
        assert "[OK]" in result
        assert "[  ]" in result

    def test_connector_count(self):
        steps = [PipelineStep(step_id=str(i), name=f"s{i}") for i in range(3)]
        p = Pipeline(name="conn", steps=steps)
        result = render_vertical(p)
        lines = result.split("\n")
        # Connector lines are standalone lines with just whitespace and |
        connector_lines = [l for l in lines if l.strip() == "|"]
        assert len(connector_lines) == 2

    def test_status_markers_present(self):
        steps = [
            PipelineStep(step_id="1", name="a", status="failed"),
            PipelineStep(step_id="2", name="b", status="running"),
        ]
        p = Pipeline(name="markers", steps=steps)
        result = render_vertical(p)
        assert "[!!]" in result
        assert "[>>]" in result

    def test_multiline_output(self):
        steps = [PipelineStep(step_id=str(i), name=f"step_{i}") for i in range(3)]
        p = Pipeline(name="multi", steps=steps)
        result = render_vertical(p)
        lines = result.split("\n")
        # 3 steps * 3 lines per box + 2 connectors = 11
        assert len(lines) == 11

    def test_box_alignment(self):
        steps = [
            PipelineStep(step_id="1", name="short"),
            PipelineStep(step_id="2", name="a-longer-name"),
        ]
        p = Pipeline(name="align", steps=steps)
        result = render_vertical(p)
        lines = result.split("\n")
        # All border lines should have the same length
        border_lines = [l for l in lines if l.startswith("+")]
        widths = {len(l) for l in border_lines}
        assert len(widths) == 1


# ---------------------------------------------------------------------------
# render_pipeline (dispatch)
# ---------------------------------------------------------------------------


class TestRenderPipeline:
    def test_default_is_vertical(self):
        p = Pipeline(
            name="def",
            steps=[PipelineStep(step_id="1", name="build")],
        )
        result = render_pipeline(p)
        # Vertical output has box borders
        assert "+" in result

    def test_horizontal_dispatch(self):
        p = Pipeline(
            name="horiz",
            steps=[
                PipelineStep(step_id="1", name="a"),
                PipelineStep(step_id="2", name="b"),
            ],
        )
        result = render_pipeline(p, orientation="horizontal")
        assert "-->" in result

    def test_vertical_dispatch(self):
        p = Pipeline(
            name="vert",
            steps=[
                PipelineStep(step_id="1", name="a"),
                PipelineStep(step_id="2", name="b"),
            ],
        )
        result = render_pipeline(p, orientation="vertical")
        assert "+" in result

    def test_unknown_orientation_falls_back_to_vertical(self):
        p = Pipeline(
            name="fallback",
            steps=[PipelineStep(step_id="1", name="a")],
        )
        result = render_pipeline(p, orientation="diagonal")
        # Should fall back to vertical (has boxes)
        assert "+" in result


# ---------------------------------------------------------------------------
# create_evalyn_pipeline
# ---------------------------------------------------------------------------


class TestCreateEvalynPipeline:
    def test_creates_pipeline(self):
        p = create_evalyn_pipeline(["trace", "build-dataset", "run-eval", "analyze"])
        assert p.name == "evalyn"
        assert len(p.steps) == 4

    def test_step_ids_sequential(self):
        p = create_evalyn_pipeline(["a", "b", "c"])
        assert p.steps[0].step_id == "step_0"
        assert p.steps[1].step_id == "step_1"
        assert p.steps[2].step_id == "step_2"

    def test_step_names_preserved(self):
        names = ["trace", "build-dataset", "run-eval", "analyze"]
        p = create_evalyn_pipeline(names)
        for step, name in zip(p.steps, names):
            assert step.name == name

    def test_all_steps_pending(self):
        p = create_evalyn_pipeline(["a", "b"])
        for step in p.steps:
            assert step.status == "pending"

    def test_empty_list(self):
        p = create_evalyn_pipeline([])
        assert p.steps == []

    def test_single_step(self):
        p = create_evalyn_pipeline(["only"])
        assert len(p.steps) == 1
        assert p.steps[0].name == "only"
