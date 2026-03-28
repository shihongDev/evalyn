"""Tests for the interactive tutorial module."""

from __future__ import annotations

from evalyn_sdk.tutorial import (
    EVALYN_TUTORIAL,
    Tutorial,
    TutorialStep,
    advance,
    format_progress,
    format_step,
    get_current_step,
    go_to_step,
    list_steps,
    reset_tutorial,
)


# ---------------------------------------------------------------------------
# TutorialStep dataclass
# ---------------------------------------------------------------------------


class TestTutorialStep:
    def test_create_step(self):
        step = TutorialStep(
            step_id="s1",
            title="First",
            explanation="Explain",
            command_hint="cmd",
            sample_output="out",
            next_step="s2",
        )
        assert step.step_id == "s1"
        assert step.title == "First"
        assert step.next_step == "s2"

    def test_as_dict(self):
        step = TutorialStep("a", "B", "C", "D", "E", "F")
        d = step.as_dict()
        assert d == {
            "step_id": "a",
            "title": "B",
            "explanation": "C",
            "command_hint": "D",
            "sample_output": "E",
            "next_step": "F",
        }

    def test_from_dict(self):
        data = {
            "step_id": "x",
            "title": "Y",
            "explanation": "Z",
            "command_hint": "c",
            "sample_output": "o",
            "next_step": "n",
        }
        step = TutorialStep.from_dict(data)
        assert step.step_id == "x"
        assert step.title == "Y"
        assert step.next_step == "n"

    def test_from_dict_defaults(self):
        step = TutorialStep.from_dict({})
        assert step.step_id == ""
        assert step.title == ""
        assert step.next_step == ""

    def test_roundtrip(self):
        step = TutorialStep("id1", "T", "E", "H", "S", "next")
        rebuilt = TutorialStep.from_dict(step.as_dict())
        assert rebuilt.step_id == step.step_id
        assert rebuilt.title == step.title
        assert rebuilt.explanation == step.explanation
        assert rebuilt.command_hint == step.command_hint
        assert rebuilt.sample_output == step.sample_output
        assert rebuilt.next_step == step.next_step


# ---------------------------------------------------------------------------
# Tutorial dataclass
# ---------------------------------------------------------------------------


class TestTutorial:
    def test_create_empty(self):
        t = Tutorial(name="empty")
        assert t.name == "empty"
        assert t.steps == []
        assert t.current_step_index == 0
        assert t.completed is False

    def test_create_with_steps(self):
        s = TutorialStep("a", "A", "e", "h", "o", "")
        t = Tutorial(name="t", steps=[s])
        assert len(t.steps) == 1

    def test_as_dict(self):
        s = TutorialStep("a", "A", "e", "h", "o", "")
        t = Tutorial(name="t", steps=[s], current_step_index=0, completed=True)
        d = t.as_dict()
        assert d["name"] == "t"
        assert len(d["steps"]) == 1
        assert d["completed"] is True
        assert d["current_step_index"] == 0

    def test_from_dict(self):
        data = {
            "name": "tut",
            "steps": [{"step_id": "x", "title": "X"}],
            "current_step_index": 0,
            "completed": False,
        }
        t = Tutorial.from_dict(data)
        assert t.name == "tut"
        assert len(t.steps) == 1
        assert t.steps[0].step_id == "x"

    def test_from_dict_defaults(self):
        t = Tutorial.from_dict({})
        assert t.name == ""
        assert t.steps == []
        assert t.current_step_index == 0
        assert t.completed is False

    def test_roundtrip(self):
        s1 = TutorialStep("a", "A", "e1", "h1", "o1", "b")
        s2 = TutorialStep("b", "B", "e2", "h2", "o2", "")
        t = Tutorial(name="round", steps=[s1, s2], current_step_index=1)
        rebuilt = Tutorial.from_dict(t.as_dict())
        assert rebuilt.name == t.name
        assert rebuilt.current_step_index == 1
        assert len(rebuilt.steps) == 2
        assert rebuilt.steps[0].step_id == "a"
        assert rebuilt.steps[1].step_id == "b"


# ---------------------------------------------------------------------------
# EVALYN_TUTORIAL constant
# ---------------------------------------------------------------------------


class TestEvalynTutorial:
    def test_has_8_steps(self):
        assert len(EVALYN_TUTORIAL.steps) == 8

    def test_step_ids(self):
        ids = [s.step_id for s in EVALYN_TUTORIAL.steps]
        assert ids == [
            "welcome",
            "tracing",
            "build_dataset",
            "select_metrics",
            "run_eval",
            "analyze",
            "compare",
            "next_steps",
        ]

    def test_last_step_has_empty_next(self):
        assert EVALYN_TUTORIAL.steps[-1].next_step == ""

    def test_chain_is_connected(self):
        steps = EVALYN_TUTORIAL.steps
        for i in range(len(steps) - 1):
            assert steps[i].next_step == steps[i + 1].step_id

    def test_all_steps_have_content(self):
        for step in EVALYN_TUTORIAL.steps:
            assert step.title
            assert step.explanation
            assert step.command_hint
            assert step.sample_output

    def test_starts_at_zero(self):
        assert EVALYN_TUTORIAL.current_step_index == 0
        assert EVALYN_TUTORIAL.completed is False


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------


class TestAdvance:
    def test_advance_increments(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        t2 = advance(t)
        assert t2.current_step_index == 1
        assert t2.completed is False

    def test_advance_at_end_marks_complete(self):
        t = Tutorial(
            name="t",
            steps=EVALYN_TUTORIAL.steps,
            current_step_index=len(EVALYN_TUTORIAL.steps) - 1,
        )
        t2 = advance(t)
        assert t2.completed is True
        assert t2.current_step_index == len(EVALYN_TUTORIAL.steps) - 1

    def test_advance_does_not_mutate(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        advance(t)
        assert t.current_step_index == 0

    def test_advance_multiple_times(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        for i in range(4):
            t = advance(t)
        assert t.current_step_index == 4


# ---------------------------------------------------------------------------
# go_to_step
# ---------------------------------------------------------------------------


class TestGoToStep:
    def test_go_to_existing(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        t2 = go_to_step(t, "analyze")
        assert t2.current_step_index == 5

    def test_go_to_first(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=5)
        t2 = go_to_step(t, "welcome")
        assert t2.current_step_index == 0

    def test_go_to_nonexistent_returns_unchanged(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=3)
        t2 = go_to_step(t, "nonexistent")
        assert t2.current_step_index == 3

    def test_go_to_clears_completed(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, completed=True)
        t2 = go_to_step(t, "tracing")
        assert t2.completed is False

    def test_go_to_does_not_mutate(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        go_to_step(t, "compare")
        assert t.current_step_index == 0


# ---------------------------------------------------------------------------
# get_current_step
# ---------------------------------------------------------------------------


class TestGetCurrentStep:
    def test_returns_first_by_default(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        step = get_current_step(t)
        assert step is not None
        assert step.step_id == "welcome"

    def test_returns_none_for_empty(self):
        t = Tutorial(name="empty")
        assert get_current_step(t) is None

    def test_returns_correct_after_advance(self):
        t = advance(Tutorial(name="t", steps=EVALYN_TUTORIAL.steps))
        step = get_current_step(t)
        assert step is not None
        assert step.step_id == "tracing"

    def test_returns_none_for_out_of_range(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=99)
        assert get_current_step(t) is None

    def test_returns_none_for_negative_index(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=-1)
        assert get_current_step(t) is None


# ---------------------------------------------------------------------------
# format_step
# ---------------------------------------------------------------------------


class TestFormatStep:
    def test_contains_title(self):
        step = EVALYN_TUTORIAL.steps[0]
        text = format_step(step, 1, 8)
        assert "Welcome to Evalyn" in text

    def test_contains_step_number(self):
        step = EVALYN_TUTORIAL.steps[2]
        text = format_step(step, 3, 8)
        assert "Step 3/8" in text

    def test_contains_command_hint(self):
        step = EVALYN_TUTORIAL.steps[1]
        text = format_step(step, 2, 8)
        assert "Try:" in text
        assert step.command_hint in text

    def test_contains_sample_output(self):
        step = EVALYN_TUTORIAL.steps[0]
        text = format_step(step, 1, 8)
        assert "Sample output:" in text

    def test_contains_explanation(self):
        step = EVALYN_TUTORIAL.steps[3]
        text = format_step(step, 4, 8)
        assert step.explanation in text


# ---------------------------------------------------------------------------
# format_progress
# ---------------------------------------------------------------------------


class TestFormatProgress:
    def test_first_step(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps)
        p = format_progress(t)
        assert "Step 1/8" in p
        assert "[" in p and "]" in p

    def test_middle_step(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=3)
        p = format_progress(t)
        assert "Step 4/8" in p

    def test_last_step(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=7)
        p = format_progress(t)
        assert "Step 8/8" in p
        assert ">" not in p  # fully filled

    def test_empty_tutorial(self):
        t = Tutorial(name="empty")
        p = format_progress(t)
        assert "Step 0/0" in p

    def test_progress_bar_has_arrow(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=0)
        p = format_progress(t)
        assert ">" in p


# ---------------------------------------------------------------------------
# list_steps
# ---------------------------------------------------------------------------


class TestListSteps:
    def test_returns_all(self):
        result = list_steps(EVALYN_TUTORIAL)
        assert len(result) == 8

    def test_format(self):
        result = list_steps(EVALYN_TUTORIAL)
        assert result[0] == "welcome: Welcome to Evalyn"

    def test_empty_tutorial(self):
        t = Tutorial(name="empty")
        assert list_steps(t) == []

    def test_all_entries_have_colon(self):
        for entry in list_steps(EVALYN_TUTORIAL):
            assert ": " in entry


# ---------------------------------------------------------------------------
# reset_tutorial
# ---------------------------------------------------------------------------


class TestResetTutorial:
    def test_resets_index(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=5)
        r = reset_tutorial(t)
        assert r.current_step_index == 0

    def test_clears_completed(self):
        t = Tutorial(
            name="t",
            steps=EVALYN_TUTORIAL.steps,
            current_step_index=7,
            completed=True,
        )
        r = reset_tutorial(t)
        assert r.completed is False
        assert r.current_step_index == 0

    def test_does_not_mutate(self):
        t = Tutorial(name="t", steps=EVALYN_TUTORIAL.steps, current_step_index=3)
        reset_tutorial(t)
        assert t.current_step_index == 3

    def test_preserves_name_and_steps(self):
        t = Tutorial(name="myname", steps=EVALYN_TUTORIAL.steps, current_step_index=4)
        r = reset_tutorial(t)
        assert r.name == "myname"
        assert len(r.steps) == 8
