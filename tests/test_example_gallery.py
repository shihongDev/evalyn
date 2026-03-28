"""Tests for the example agent gallery module."""

from __future__ import annotations

import json
import os

import pytest

from evalyn_sdk.example_gallery import (
    BUILTIN_EXAMPLES,
    ExampleAgent,
    ExampleGallery,
    format_gallery_listing,
    get_example,
    list_examples,
    scaffold_example,
    search_examples,
)


# ---------------------------------------------------------------------------
# ExampleAgent dataclass
# ---------------------------------------------------------------------------


class TestExampleAgent:
    def test_create(self):
        a = ExampleAgent(
            framework="test",
            name="Test Agent",
            description="desc",
            code_template="print('hi')",
            dataset_template='{"input": "x"}',
        )
        assert a.framework == "test"
        assert a.name == "Test Agent"

    def test_as_dict(self):
        a = ExampleAgent(
            framework="f",
            name="n",
            description="d",
            code_template="c",
            dataset_template="ds",
            expected_metrics=["m1"],
            tags=["t1"],
        )
        d = a.as_dict()
        assert d["framework"] == "f"
        assert d["expected_metrics"] == ["m1"]
        assert d["tags"] == ["t1"]

    def test_from_dict(self):
        data = {
            "framework": "fw",
            "name": "N",
            "description": "D",
            "code_template": "C",
            "dataset_template": "DS",
            "expected_metrics": ["a", "b"],
            "tags": ["x"],
        }
        a = ExampleAgent.from_dict(data)
        assert a.framework == "fw"
        assert a.expected_metrics == ["a", "b"]

    def test_from_dict_defaults(self):
        a = ExampleAgent.from_dict({})
        assert a.framework == ""
        assert a.name == ""
        assert a.expected_metrics == []
        assert a.tags == []

    def test_roundtrip(self):
        a = ExampleAgent(
            framework="rt",
            name="Round",
            description="Trip",
            code_template="code",
            dataset_template="data",
            expected_metrics=["m"],
            tags=["t"],
        )
        rebuilt = ExampleAgent.from_dict(a.as_dict())
        assert rebuilt.framework == a.framework
        assert rebuilt.name == a.name
        assert rebuilt.expected_metrics == a.expected_metrics

    def test_as_dict_copies_lists(self):
        metrics = ["a"]
        a = ExampleAgent("f", "n", "d", "c", "ds", metrics, [])
        d = a.as_dict()
        d["expected_metrics"].append("b")
        assert a.expected_metrics == ["a"]


# ---------------------------------------------------------------------------
# ExampleGallery dataclass
# ---------------------------------------------------------------------------


class TestExampleGallery:
    def test_create_empty(self):
        g = ExampleGallery()
        assert g.agents == []
        assert g.frameworks == []

    def test_as_dict(self):
        a = ExampleAgent("f", "n", "d", "c", "ds")
        g = ExampleGallery(agents=[a], frameworks=["f"])
        d = g.as_dict()
        assert len(d["agents"]) == 1
        assert d["frameworks"] == ["f"]

    def test_from_dict(self):
        data = {
            "agents": [{"framework": "x", "name": "X"}],
            "frameworks": ["x"],
        }
        g = ExampleGallery.from_dict(data)
        assert len(g.agents) == 1
        assert g.frameworks == ["x"]

    def test_from_dict_defaults(self):
        g = ExampleGallery.from_dict({})
        assert g.agents == []
        assert g.frameworks == []

    def test_roundtrip(self):
        a = ExampleAgent("f", "n", "d", "c", "ds", ["m"], ["t"])
        g = ExampleGallery(agents=[a], frameworks=["f"])
        rebuilt = ExampleGallery.from_dict(g.as_dict())
        assert len(rebuilt.agents) == 1
        assert rebuilt.agents[0].framework == "f"


# ---------------------------------------------------------------------------
# BUILTIN_EXAMPLES constant
# ---------------------------------------------------------------------------


class TestBuiltinExamples:
    def test_has_5_agents(self):
        assert len(BUILTIN_EXAMPLES.agents) == 5

    def test_has_5_frameworks(self):
        assert len(BUILTIN_EXAMPLES.frameworks) == 5

    def test_framework_names(self):
        names = [a.framework for a in BUILTIN_EXAMPLES.agents]
        assert "openai" in names
        assert "anthropic" in names
        assert "langchain" in names
        assert "custom" in names
        assert "multi_turn" in names

    def test_all_have_code_template(self):
        for a in BUILTIN_EXAMPLES.agents:
            assert len(a.code_template) > 50

    def test_all_have_dataset_template(self):
        for a in BUILTIN_EXAMPLES.agents:
            assert len(a.dataset_template) > 10

    def test_all_have_expected_metrics(self):
        for a in BUILTIN_EXAMPLES.agents:
            assert len(a.expected_metrics) >= 1

    def test_all_have_tags(self):
        for a in BUILTIN_EXAMPLES.agents:
            assert len(a.tags) >= 1

    def test_code_uses_env_vars(self):
        for a in BUILTIN_EXAMPLES.agents:
            if a.framework in ("openai", "anthropic", "langchain", "multi_turn"):
                assert "os.environ.get" in a.code_template

    def test_dataset_lines_are_valid_json(self):
        for a in BUILTIN_EXAMPLES.agents:
            for line in a.dataset_template.strip().splitlines():
                parsed = json.loads(line)
                assert "input" in parsed

    def test_no_hardcoded_keys(self):
        for a in BUILTIN_EXAMPLES.agents:
            assert "sk-" not in a.code_template
            assert "key-" not in a.code_template


# ---------------------------------------------------------------------------
# list_examples
# ---------------------------------------------------------------------------


class TestListExamples:
    def test_all(self):
        result = list_examples(BUILTIN_EXAMPLES)
        assert len(result) == 5

    def test_filter_by_framework(self):
        result = list_examples(BUILTIN_EXAMPLES, framework="openai")
        assert len(result) == 1
        assert result[0].framework == "openai"

    def test_filter_by_tag(self):
        result = list_examples(BUILTIN_EXAMPLES, tag="beginner")
        assert len(result) >= 2

    def test_filter_by_both(self):
        result = list_examples(BUILTIN_EXAMPLES, framework="openai", tag="chat")
        assert len(result) == 1

    def test_no_match(self):
        result = list_examples(BUILTIN_EXAMPLES, framework="nonexistent")
        assert result == []

    def test_tag_no_match(self):
        result = list_examples(BUILTIN_EXAMPLES, tag="nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# get_example
# ---------------------------------------------------------------------------


class TestGetExample:
    def test_existing(self):
        a = get_example(BUILTIN_EXAMPLES, "anthropic")
        assert a is not None
        assert a.framework == "anthropic"

    def test_nonexistent(self):
        assert get_example(BUILTIN_EXAMPLES, "nope") is None

    def test_all_frameworks(self):
        for fw in BUILTIN_EXAMPLES.frameworks:
            assert get_example(BUILTIN_EXAMPLES, fw) is not None


# ---------------------------------------------------------------------------
# scaffold_example
# ---------------------------------------------------------------------------


class TestScaffoldExample:
    def test_creates_files(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "project")
        paths = scaffold_example(agent, out)
        assert len(paths) == 3
        for p in paths:
            assert os.path.isfile(p)

    def test_agent_py_content(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "proj")
        scaffold_example(agent, out)
        content = open(os.path.join(out, "agent.py")).read()
        assert "run_agent" in content

    def test_dataset_jsonl_content(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "proj")
        scaffold_example(agent, out)
        content = open(os.path.join(out, "dataset.jsonl")).read()
        for line in content.strip().splitlines():
            json.loads(line)  # must parse

    def test_readme_content(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "proj")
        scaffold_example(agent, out)
        content = open(os.path.join(out, "README.txt")).read()
        assert agent.name in content
        assert agent.framework in content

    def test_creates_output_dir(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "deep" / "nested" / "dir")
        scaffold_example(agent, out)
        assert os.path.isdir(out)

    def test_returns_absolute_paths(self, tmp_path):
        agent = BUILTIN_EXAMPLES.agents[0]
        out = str(tmp_path / "proj")
        paths = scaffold_example(agent, out)
        for p in paths:
            assert os.path.isabs(p)

    def test_scaffold_custom_agent(self, tmp_path):
        agent = get_example(BUILTIN_EXAMPLES, "custom")
        assert agent is not None
        out = str(tmp_path / "custom_proj")
        paths = scaffold_example(agent, out)
        assert len(paths) == 3
        content = open(os.path.join(out, "agent.py")).read()
        assert "rule-based" in content.lower() or "Rule" in content or "rule" in content


# ---------------------------------------------------------------------------
# format_gallery_listing
# ---------------------------------------------------------------------------


class TestFormatGalleryListing:
    def test_contains_all_names(self):
        text = format_gallery_listing(BUILTIN_EXAMPLES)
        for agent in BUILTIN_EXAMPLES.agents:
            assert agent.name in text

    def test_contains_frameworks(self):
        text = format_gallery_listing(BUILTIN_EXAMPLES)
        for agent in BUILTIN_EXAMPLES.agents:
            assert agent.framework in text

    def test_contains_header(self):
        text = format_gallery_listing(BUILTIN_EXAMPLES)
        assert "Example Agent Gallery" in text

    def test_empty_gallery(self):
        g = ExampleGallery()
        text = format_gallery_listing(g)
        assert "Example Agent Gallery" in text


# ---------------------------------------------------------------------------
# search_examples
# ---------------------------------------------------------------------------


class TestSearchExamples:
    def test_search_by_name(self):
        results = search_examples(BUILTIN_EXAMPLES, "OpenAI")
        assert len(results) >= 1

    def test_search_by_tag(self):
        results = search_examples(BUILTIN_EXAMPLES, "beginner")
        assert len(results) >= 2

    def test_search_case_insensitive(self):
        r1 = search_examples(BUILTIN_EXAMPLES, "openai")
        r2 = search_examples(BUILTIN_EXAMPLES, "OPENAI")
        assert len(r1) == len(r2)

    def test_search_no_match(self):
        results = search_examples(BUILTIN_EXAMPLES, "zzzznotfound")
        assert results == []

    def test_search_by_description_word(self):
        results = search_examples(BUILTIN_EXAMPLES, "rule-based")
        assert len(results) >= 1
        assert results[0].framework == "custom"

    def test_search_multi_turn(self):
        results = search_examples(BUILTIN_EXAMPLES, "multi-turn")
        assert len(results) >= 1
