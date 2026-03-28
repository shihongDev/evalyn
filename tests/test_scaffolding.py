from __future__ import annotations

import os

import pytest

from evalyn_sdk.scaffolding import (
    ADVANCED_TEMPLATE,
    BASIC_TEMPLATE,
    ProjectTemplate,
    ScaffoldConfig,
    ScaffoldResult,
    get_template,
    list_templates,
    plan_scaffold,
    render_template_file,
    validate_project_name,
)


# -- ProjectTemplate -----------------------------------------------------------


class TestProjectTemplate:
    def test_as_dict(self):
        t = ProjectTemplate(name="test", description="desc", files={"a.txt": "hello"})
        d = t.as_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"
        assert d["files"] == {"a.txt": "hello"}

    def test_from_dict(self):
        t = ProjectTemplate.from_dict({
            "name": "t1",
            "description": "d1",
            "files": {"f": "c"},
            "directories": ["dir"],
        })
        assert t.name == "t1"
        assert t.files == {"f": "c"}
        assert t.directories == ["dir"]

    def test_from_dict_defaults(self):
        t = ProjectTemplate.from_dict({})
        assert t.name == ""
        assert t.files == {}
        assert t.directories == []

    def test_roundtrip(self):
        original = ProjectTemplate(
            name="rt", description="roundtrip",
            files={"x": "y"}, directories=["a", "b"],
        )
        rebuilt = ProjectTemplate.from_dict(original.as_dict())
        assert rebuilt.as_dict() == original.as_dict()

    def test_empty_template(self):
        t = ProjectTemplate(name="empty")
        assert t.files == {}
        assert t.directories == []
        assert t.description == ""


# -- ScaffoldConfig ------------------------------------------------------------


class TestScaffoldConfig:
    def test_as_dict(self):
        c = ScaffoldConfig(project_name="proj", template="advanced", provider="openai")
        d = c.as_dict()
        assert d["project_name"] == "proj"
        assert d["template"] == "advanced"
        assert d["provider"] == "openai"

    def test_from_dict(self):
        c = ScaffoldConfig.from_dict({
            "project_name": "foo",
            "template": "basic",
            "output_dir": "/tmp",
            "provider": "anthropic",
            "include_tests": False,
        })
        assert c.project_name == "foo"
        assert c.output_dir == "/tmp"
        assert c.include_tests is False

    def test_from_dict_defaults(self):
        c = ScaffoldConfig.from_dict({})
        assert c.template == "basic"
        assert c.provider == "gemini"
        assert c.include_tests is True

    def test_roundtrip(self):
        original = ScaffoldConfig(
            project_name="rt", template="advanced",
            output_dir="/x", provider="gemini", include_tests=False,
        )
        rebuilt = ScaffoldConfig.from_dict(original.as_dict())
        assert rebuilt.as_dict() == original.as_dict()


# -- ScaffoldResult ------------------------------------------------------------


class TestScaffoldResult:
    def test_as_dict(self):
        r = ScaffoldResult(
            created_files=["a.txt"], created_dirs=["data"], project_path="/p",
        )
        d = r.as_dict()
        assert d["created_files"] == ["a.txt"]
        assert d["project_path"] == "/p"

    def test_format_text(self):
        r = ScaffoldResult(
            created_files=["f1", "f2"], created_dirs=["d1"], project_path="/proj",
        )
        text = r.format_text()
        assert "Project: /proj" in text
        assert "Directories: 1" in text
        assert "Files: 2" in text
        assert "d1/" in text
        assert "f1" in text

    def test_format_text_empty(self):
        r = ScaffoldResult(project_path="/empty")
        text = r.format_text()
        assert "Project: /empty" in text
        assert "Directories" not in text
        assert "Files" not in text


# -- get_template --------------------------------------------------------------


class TestGetTemplate:
    def test_basic(self):
        t = get_template("basic")
        assert t.name == "basic"
        assert "evalyn.yaml" in t.files

    def test_advanced(self):
        t = get_template("advanced")
        assert t.name == "advanced"
        assert "tests/test_eval.py" in t.files

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown template"):
            get_template("nonexistent")


# -- list_templates ------------------------------------------------------------


class TestListTemplates:
    def test_has_two(self):
        templates = list_templates()
        assert len(templates) == 2

    def test_names(self):
        names = [t.name for t in list_templates()]
        assert "basic" in names
        assert "advanced" in names


# -- render_template_file ------------------------------------------------------


class TestRenderTemplateFile:
    def test_replaces_vars(self):
        content = "provider: {{provider}}\nname: {{name}}"
        result = render_template_file(content, {"provider": "openai", "name": "proj"})
        assert result == "provider: openai\nname: proj"

    def test_no_variables(self):
        content = "plain text"
        result = render_template_file(content, {})
        assert result == "plain text"

    def test_missing_variable_left_as_is(self):
        content = "{{missing}}"
        result = render_template_file(content, {})
        assert result == "{{missing}}"

    def test_multiple_occurrences(self):
        content = "{{x}} and {{x}}"
        result = render_template_file(content, {"x": "val"})
        assert result == "val and val"


# -- validate_project_name -----------------------------------------------------


class TestValidateProjectName:
    def test_valid_simple(self):
        ok, msg = validate_project_name("my_project")
        assert ok is True
        assert msg == ""

    def test_valid_with_hyphens(self):
        ok, _ = validate_project_name("my-project-2")
        assert ok is True

    def test_valid_alphanumeric(self):
        ok, _ = validate_project_name("project123")
        assert ok is True

    def test_invalid_empty(self):
        ok, msg = validate_project_name("")
        assert ok is False
        assert "empty" in msg.lower()

    def test_invalid_spaces(self):
        ok, msg = validate_project_name("my project")
        assert ok is False

    def test_invalid_special_chars(self):
        ok, msg = validate_project_name("project@name")
        assert ok is False

    def test_invalid_starts_with_hyphen(self):
        ok, msg = validate_project_name("-bad")
        assert ok is False
        assert "hyphen" in msg.lower()


# -- plan_scaffold -------------------------------------------------------------


class TestPlanScaffold:
    def test_lists_files(self):
        config = ScaffoldConfig(project_name="proj", template="basic", output_dir="/tmp")
        result = plan_scaffold(config)
        assert result.project_path == os.path.join("/tmp", "proj")
        assert len(result.created_files) > 0
        assert len(result.created_dirs) > 0

    def test_basic_template_files(self):
        config = ScaffoldConfig(project_name="p", template="basic")
        result = plan_scaffold(config)
        file_names = [os.path.basename(f) for f in result.created_files]
        assert "evalyn.yaml" in file_names

    def test_advanced_template_includes_tests(self):
        config = ScaffoldConfig(project_name="p", template="advanced", include_tests=True)
        result = plan_scaffold(config)
        has_test = any("test_eval.py" in f for f in result.created_files)
        assert has_test

    def test_exclude_tests(self):
        config = ScaffoldConfig(project_name="p", template="advanced", include_tests=False)
        result = plan_scaffold(config)
        has_test = any("test_eval.py" in f for f in result.created_files)
        assert not has_test

    def test_unknown_template_raises(self):
        config = ScaffoldConfig(project_name="p", template="nonexistent")
        with pytest.raises(KeyError):
            plan_scaffold(config)
