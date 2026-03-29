"""Tests for simulation template library module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.simulation_templates import (
    PersonaMix,
    SimulationTemplate,
    TemplateLibrary,
    BUILTIN_TEMPLATES,
    get_template,
    list_templates,
    format_template_catalog,
    format_template_detail,
    customize_template,
)


# ---------------------------------------------------------------------------
# PersonaMix dataclass
# ---------------------------------------------------------------------------


class TestPersonaMix:
    def test_construction(self):
        p = PersonaMix(persona_id="novice", weight=0.4)
        assert p.persona_id == "novice"
        assert p.weight == 0.4

    def test_as_dict(self):
        p = PersonaMix(persona_id="expert", weight=0.6)
        d = p.as_dict()
        assert d["persona_id"] == "expert"
        assert d["weight"] == 0.6

    def test_from_dict(self):
        d = {"persona_id": "adversarial", "weight": 0.2}
        p = PersonaMix.from_dict(d)
        assert p.persona_id == "adversarial"
        assert p.weight == 0.2

    def test_roundtrip(self):
        original = PersonaMix(persona_id="power_user", weight=0.3)
        restored = PersonaMix.from_dict(original.as_dict())
        assert restored.persona_id == original.persona_id
        assert restored.weight == original.weight

    def test_zero_weight(self):
        p = PersonaMix(persona_id="rare", weight=0.0)
        assert p.weight == 0.0

    def test_full_weight(self):
        p = PersonaMix(persona_id="sole", weight=1.0)
        assert p.weight == 1.0


# ---------------------------------------------------------------------------
# SimulationTemplate dataclass
# ---------------------------------------------------------------------------


class TestSimulationTemplate:
    def test_construction(self):
        t = SimulationTemplate(
            template_id="test",
            name="Test",
            description="A test template",
        )
        assert t.template_id == "test"
        assert t.output_format == "text"
        assert t.min_turns == 1
        assert t.max_turns == 1
        assert t.persona_mix == []
        assert t.edge_case_types == []
        assert t.topics == []
        assert t.constraints == []

    def test_as_dict(self):
        t = SimulationTemplate(
            template_id="t1",
            name="T1",
            description="desc",
            persona_mix=[PersonaMix("a", 0.5)],
            edge_case_types=["empty"],
            topics=["billing"],
            min_turns=2,
            max_turns=4,
        )
        d = t.as_dict()
        assert d["template_id"] == "t1"
        assert d["min_turns"] == 2
        assert d["max_turns"] == 4
        assert len(d["persona_mix"]) == 1
        assert d["persona_mix"][0]["persona_id"] == "a"

    def test_from_dict(self):
        d = {
            "template_id": "t2",
            "name": "T2",
            "description": "desc2",
            "persona_mix": [{"persona_id": "x", "weight": 0.8}],
            "edge_case_types": ["unicode"],
            "output_format": "json",
            "topics": ["search"],
            "min_turns": 1,
            "max_turns": 3,
            "constraints": ["max_tokens_100"],
        }
        t = SimulationTemplate.from_dict(d)
        assert t.template_id == "t2"
        assert t.output_format == "json"
        assert t.max_turns == 3
        assert len(t.persona_mix) == 1
        assert t.constraints == ["max_tokens_100"]

    def test_from_dict_defaults(self):
        d = {"template_id": "t3", "name": "T3", "description": "d3"}
        t = SimulationTemplate.from_dict(d)
        assert t.output_format == "text"
        assert t.min_turns == 1
        assert t.max_turns == 1
        assert t.persona_mix == []

    def test_roundtrip(self):
        original = SimulationTemplate(
            template_id="rt",
            name="Roundtrip",
            description="roundtrip test",
            persona_mix=[PersonaMix("a", 0.3), PersonaMix("b", 0.7)],
            edge_case_types=["empty", "long"],
            output_format="json",
            topics=["t1", "t2"],
            min_turns=2,
            max_turns=5,
            constraints=["c1"],
        )
        restored = SimulationTemplate.from_dict(original.as_dict())
        assert restored.template_id == original.template_id
        assert restored.name == original.name
        assert restored.output_format == original.output_format
        assert restored.min_turns == original.min_turns
        assert restored.max_turns == original.max_turns
        assert len(restored.persona_mix) == 2
        assert restored.edge_case_types == original.edge_case_types
        assert restored.constraints == original.constraints


# ---------------------------------------------------------------------------
# TemplateLibrary dataclass
# ---------------------------------------------------------------------------


class TestTemplateLibrary:
    def test_construction(self):
        lib = TemplateLibrary()
        assert lib.templates == {}

    def test_with_templates(self):
        t = SimulationTemplate(template_id="x", name="X", description="x")
        lib = TemplateLibrary(templates={"x": t})
        assert "x" in lib.templates

    def test_as_dict(self):
        t = SimulationTemplate(template_id="a", name="A", description="a")
        lib = TemplateLibrary(templates={"a": t})
        d = lib.as_dict()
        assert "templates" in d
        assert "a" in d["templates"]
        assert d["templates"]["a"]["template_id"] == "a"

    def test_from_dict(self):
        d = {
            "templates": {
                "b": {
                    "template_id": "b",
                    "name": "B",
                    "description": "b",
                }
            }
        }
        lib = TemplateLibrary.from_dict(d)
        assert "b" in lib.templates
        assert lib.templates["b"].name == "B"

    def test_roundtrip(self):
        t1 = SimulationTemplate(template_id="t1", name="T1", description="d1")
        t2 = SimulationTemplate(template_id="t2", name="T2", description="d2")
        original = TemplateLibrary(templates={"t1": t1, "t2": t2})
        restored = TemplateLibrary.from_dict(original.as_dict())
        assert set(restored.templates.keys()) == {"t1", "t2"}

    def test_empty_from_dict(self):
        lib = TemplateLibrary.from_dict({})
        assert lib.templates == {}


# ---------------------------------------------------------------------------
# BUILTIN_TEMPLATES
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    def test_has_four_templates(self):
        assert len(BUILTIN_TEMPLATES.templates) == 4

    def test_customer_support(self):
        t = BUILTIN_TEMPLATES.templates["customer_support"]
        assert t.name == "Customer Support"
        assert len(t.persona_mix) == 4
        weights = [p.weight for p in t.persona_mix]
        assert abs(sum(weights) - 1.0) < 1e-9
        assert "empty" in t.edge_case_types
        assert "special_chars" in t.edge_case_types
        assert "refund" in t.topics

    def test_rag_qa(self):
        t = BUILTIN_TEMPLATES.templates["rag_qa"]
        assert t.name == "RAG QA"
        assert len(t.persona_mix) == 3
        weights = [p.weight for p in t.persona_mix]
        assert abs(sum(weights) - 1.0) < 1e-9
        assert "unicode" in t.edge_case_types
        assert "knowledge retrieval" in t.topics

    def test_code_review(self):
        t = BUILTIN_TEMPLATES.templates["code_review"]
        assert t.name == "Code Review"
        assert len(t.persona_mix) == 3
        expert = next(p for p in t.persona_mix if p.persona_id == "expert")
        assert expert.weight == 0.6
        assert "security" in t.topics

    def test_multi_step_agent(self):
        t = BUILTIN_TEMPLATES.templates["multi_step_agent"]
        assert t.name == "Multi-Step Agent"
        assert t.min_turns == 2
        assert t.max_turns == 6
        assert "boundary" in t.edge_case_types
        assert "contradiction" in t.edge_case_types

    def test_all_weights_sum_to_one(self):
        for tid, t in BUILTIN_TEMPLATES.templates.items():
            weights = [p.weight for p in t.persona_mix]
            assert abs(sum(weights) - 1.0) < 1e-9, f"{tid} weights do not sum to 1.0"

    def test_all_have_topics(self):
        for tid, t in BUILTIN_TEMPLATES.templates.items():
            assert len(t.topics) > 0, f"{tid} has no topics"

    def test_all_have_edge_cases(self):
        for tid, t in BUILTIN_TEMPLATES.templates.items():
            assert len(t.edge_case_types) > 0, f"{tid} has no edge cases"


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_existing(self):
        t = get_template("customer_support")
        assert t is not None
        assert t.template_id == "customer_support"

    def test_missing(self):
        assert get_template("nonexistent") is None

    def test_all_builtins_accessible(self):
        for tid in ["customer_support", "rag_qa", "code_review", "multi_step_agent"]:
            assert get_template(tid) is not None


# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_returns_sorted(self):
        ids = list_templates()
        assert ids == sorted(ids)

    def test_contains_all_builtins(self):
        ids = list_templates()
        assert "customer_support" in ids
        assert "rag_qa" in ids
        assert "code_review" in ids
        assert "multi_step_agent" in ids

    def test_count(self):
        assert len(list_templates()) == 4


# ---------------------------------------------------------------------------
# format_template_catalog
# ---------------------------------------------------------------------------


class TestFormatTemplateCatalog:
    def test_basic(self):
        catalog = format_template_catalog(BUILTIN_TEMPLATES)
        assert "SIMULATION TEMPLATE CATALOG" in catalog
        assert "Templates: 4" in catalog

    def test_contains_all_ids(self):
        catalog = format_template_catalog(BUILTIN_TEMPLATES)
        for tid in BUILTIN_TEMPLATES.templates:
            assert tid in catalog

    def test_shows_turns_for_multi_turn(self):
        catalog = format_template_catalog(BUILTIN_TEMPLATES)
        assert "2-6" in catalog  # multi_step_agent turns

    def test_empty_library(self):
        catalog = format_template_catalog(TemplateLibrary())
        assert "Templates: 0" in catalog

    def test_contains_personas(self):
        catalog = format_template_catalog(BUILTIN_TEMPLATES)
        assert "novice" in catalog
        assert "expert" in catalog


# ---------------------------------------------------------------------------
# format_template_detail
# ---------------------------------------------------------------------------


class TestFormatTemplateDetail:
    def test_basic(self):
        t = get_template("customer_support")
        detail = format_template_detail(t)
        assert "Customer Support" in detail
        assert "customer_support" in detail

    def test_shows_personas(self):
        t = get_template("rag_qa")
        detail = format_template_detail(t)
        assert "novice" in detail
        assert "expert" in detail
        assert "adversarial" in detail

    def test_shows_topics(self):
        t = get_template("code_review")
        detail = format_template_detail(t)
        assert "bug detection" in detail
        assert "security" in detail

    def test_shows_edge_cases(self):
        t = get_template("customer_support")
        detail = format_template_detail(t)
        assert "empty" in detail

    def test_shows_turns(self):
        t = get_template("multi_step_agent")
        detail = format_template_detail(t)
        assert "2-6" in detail

    def test_custom_template(self):
        t = SimulationTemplate(
            template_id="custom",
            name="Custom",
            description="Custom template",
            constraints=["max_tokens_50", "english_only"],
        )
        detail = format_template_detail(t)
        assert "max_tokens_50" in detail
        assert "english_only" in detail


# ---------------------------------------------------------------------------
# customize_template
# ---------------------------------------------------------------------------


class TestCustomizeTemplate:
    def test_override_name(self):
        base = get_template("customer_support")
        custom = customize_template(base, {"name": "My Support"})
        assert custom.name == "My Support"
        assert custom.template_id == "customer_support"

    def test_override_topics(self):
        base = get_template("rag_qa")
        custom = customize_template(base, {"topics": ["custom_topic"]})
        assert custom.topics == ["custom_topic"]

    def test_override_turns(self):
        base = get_template("multi_step_agent")
        custom = customize_template(base, {"min_turns": 3, "max_turns": 10})
        assert custom.min_turns == 3
        assert custom.max_turns == 10

    def test_override_persona_mix_with_dicts(self):
        base = get_template("code_review")
        new_mix = [{"persona_id": "student", "weight": 0.7}, {"persona_id": "expert", "weight": 0.3}]
        custom = customize_template(base, {"persona_mix": new_mix})
        assert len(custom.persona_mix) == 2
        assert custom.persona_mix[0].persona_id == "student"

    def test_override_persona_mix_with_objects(self):
        base = get_template("code_review")
        new_mix = [PersonaMix("student", 0.7), PersonaMix("expert", 0.3)]
        custom = customize_template(base, {"persona_mix": new_mix})
        assert len(custom.persona_mix) == 2
        assert custom.persona_mix[0].persona_id == "student"

    def test_does_not_modify_original(self):
        base = get_template("customer_support")
        original_name = base.name
        customize_template(base, {"name": "Modified"})
        assert base.name == original_name

    def test_override_output_format(self):
        base = get_template("rag_qa")
        custom = customize_template(base, {"output_format": "json"})
        assert custom.output_format == "json"
        assert base.output_format == "text"

    def test_override_constraints(self):
        base = get_template("customer_support")
        custom = customize_template(base, {"constraints": ["no_profanity"]})
        assert custom.constraints == ["no_profanity"]

    def test_empty_overrides(self):
        base = get_template("rag_qa")
        custom = customize_template(base, {})
        assert custom.template_id == base.template_id
        assert custom.name == base.name
        assert len(custom.persona_mix) == len(base.persona_mix)

    def test_override_template_id(self):
        base = get_template("customer_support")
        custom = customize_template(base, {"template_id": "custom_support"})
        assert custom.template_id == "custom_support"
